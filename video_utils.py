"""视频文件工具。"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urljoin, urlparse

import aiohttp

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

# MIME 类型与官方文档对齐：https://ai.google.dev/gemini-api/docs/video-understanding
_MIME_MAP: dict[str, str] = {
    "mp4": "video/mp4",
    "mov": "video/mov",
    "webm": "video/webm",
    "avi": "video/avi",
    "mpeg": "video/mpeg",
    "mpg": "video/mpeg",
    "flv": "video/x-flv",
    "wmv": "video/wmv",
    "3gpp": "video/3gpp",
}

_MB = 1024 * 1024
_TEMP_SUBDIR = "astrbot_video_sense"
_MAX_REDIRECTS = 3
_CHUNK_SIZE = 1024 * 1024
_TEMP_FILE_TTL_SECONDS = 24 * 3600
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
# 这些状态码通常意味着平台链接已过期或文件已删除
_EXPIRED_STATUSES = frozenset({401, 403, 404, 410})


@dataclass
class VideoFile:
    path: str
    mime_type: str
    filename: str
    size_bytes: int


class VideoError(Exception):
    """视频处理错误。"""


class VideoExpiredError(VideoError):
    """缓存引用已失效：链接过期、文件被删除或缓存已被回收。"""


def _get_ext(name: str) -> str:
    return Path(name).suffix.lstrip(".").lower()


def resolve_component_ref(comp) -> tuple[str, str]:
    """从 File/Video 组件提取 (local_path, remote_url)。local_path 为空时返回空字符串。"""
    if type(comp).__name__ == "Video":
        return _resolve_video_component_ref(comp)

    local = ""
    url = ""

    raw_file = getattr(comp, "file_", "") or ""
    raw_url = getattr(comp, "url", "") or ""

    if raw_file:
        if raw_file.startswith("file://") or raw_file.startswith("file:"):
            try:
                parsed = urlparse(raw_file)
                local = unquote(parsed.path)
                if local and local[0] == "/" and len(local) > 2 and local[2] == ":":
                    local = local[1:]  # Windows: /C:/... → C:/...
            except Exception:
                local = raw_file
        else:
            local = raw_file

    if raw_url:
        url = raw_url

    return local, url


def _resolve_video_component_ref(comp) -> tuple[str, str]:
    """Video 组件：字段为 file（路径/文件名/URL）、url、path。"""
    local = ""
    url = ""

    path = getattr(comp, "path", "") or ""
    file_ = getattr(comp, "file", "") or ""
    raw_url = getattr(comp, "url", "") or ""

    if path:
        local = path
    elif file_:
        if file_.startswith(("http://", "https://")):
            url = file_
        elif file_.startswith(("file://", "file:")):
            parsed = urlparse(file_)
            local = unquote(parsed.path)
            if local and local[0] == "/" and len(local) > 2 and local[2] == ":":
                local = local[1:]
        elif not file_.startswith(("base64://", "data:")):
            local = file_  # 本地路径或纯文件名（如 aiocqhttp 的 file 字段）

    if raw_url:
        url = raw_url

    return local, url


def video_component_name(comp) -> str:
    """从 File/Video 组件推断文件名；Video 组件无 name 字段，从路径/URL 提取。"""
    name = getattr(comp, "name", "") or ""
    if name:
        return name
    for cand in (
        getattr(comp, "path", "") or "",
        getattr(comp, "file", "") or "",
        getattr(comp, "url", "") or "",
    ):
        if not cand or cand.startswith(("base64://", "data:")):
            continue
        cand_name = Path(cand.split("?")[0]).name
        if cand_name and "." in cand_name:
            return cand_name
    return "video.mp4"


def extract_file_component(event: AstrMessageEvent):
    """从消息链（含引用消息）中提取第一个 File/Video 组件。"""
    messages = _get_messages(event)

    file_comp = _find_file_in_chain(messages)
    if file_comp is not None:
        return file_comp

    for comp in messages:
        if type(comp).__name__ == "Reply":
            chain = getattr(comp, "chain", None) or []
            file_comp = _find_file_in_chain(chain)
            if file_comp is not None:
                return file_comp

    return None


def _find_file_in_chain(chain) -> object | None:
    for comp in chain or []:
        if type(comp).__name__ in ("File", "Video"):
            return comp
    return None


def video_ref_status(item: dict) -> tuple[bool, str]:
    """检查缓存条目的文件引用当前是否可用（只查本地文件，不发网络请求）。

    Args:
        item: 缓存条目，至少包含 ref / is_local。

    Returns:
        (是否可用, 不可用原因)。远程链接不做探测，按可用处理，真正过期在下载时报错。
    """
    ref = (item.get("ref") or "").strip()
    if not ref:
        return False, "缓存文件已被回收"
    if item.get("is_local") and not Path(ref).is_file():
        return False, "本地文件已被清理"
    return True, ""


def _get_messages(event: AstrMessageEvent) -> list:
    if hasattr(event, "get_messages"):
        try:
            msgs = event.get_messages()
            if msgs is not None:
                return list(msgs)
        except Exception:
            pass
    if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
        return list(event.message_obj.message or [])
    return []


async def load_video(
    file_comp, supported_formats: list[str], max_size_mb: int
) -> VideoFile:
    """校验并获取 File/Video 组件的本地路径，返回 VideoFile（不读取内容）。

    Video 组件通过 convert_to_file_path() 统一解析（远程 URL 自动下载）。
    """
    name = video_component_name(file_comp)
    ext = _get_ext(name)
    if not ext and type(file_comp).__name__ == "Video":
        ext = "mp4"  # Video 组件无扩展名时按 mp4 处理（框架默认后缀）

    if ext not in supported_formats:
        supported_str = "、".join(supported_formats)
        raise VideoError(f"文件格式 .{ext} 不支持，当前支持的格式：{supported_str}。")

    try:
        if type(file_comp).__name__ == "Video":
            local_path: str = await file_comp.convert_to_file_path()
        else:
            local_path = await file_comp.get_file()
    except Exception as e:
        raise VideoError(f"获取文件失败：{e}") from e

    return await _validate(local_path, ext, max_size_mb, name)


async def load_video_from_path(file_path: str, max_size_mb: int) -> VideoFile:
    """从本地路径加载视频，返回 VideoFile（不读取内容）。"""
    p = Path(file_path)
    if not p.is_file():
        raise VideoError(f"文件不存在或无法访问：{file_path}")

    ext = _get_ext(p.name)
    return await _validate(str(p), ext, max_size_mb, p.name)


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def _assert_public_url(url: str) -> None:
    """校验下载地址指向公网地址，降低 SSRF 风险。

    Args:
        url: 待校验的下载地址。

    Raises:
        VideoError: 协议非 http(s)，或主机名/IP 指向本机、内网与保留地址。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise VideoError(f"不支持的下载协议：{parsed.scheme or '未知'}。")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise VideoError("下载地址缺少主机名。")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise VideoError("下载地址指向本机或内网主机，已拒绝。")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_private_ip(literal):
            raise VideoError("下载地址指向内网或保留地址，已拒绝。")
        return

    # 域名：解析结果也逐条校验，避免解析到内网地址
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as e:
        raise VideoError(f"无法解析下载地址：{e}") from e
    for info in infos:
        address = str(info[4][0]).split("%", 1)[0]
        try:
            resolved = ipaddress.ip_address(address)
        except ValueError:
            continue
        if _is_private_ip(resolved):
            raise VideoError("下载地址解析到内网或保留地址，已拒绝。")


def _sanitize_filename(save_name: str) -> str:
    """清洗消息里带出的文件名，仅保留 basename 并替换非法字符。"""
    safe_name = Path(save_name or "video").name
    safe_name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", safe_name).strip()
    return safe_name or "video"


def _prune_temp_files(
    directory: Path, max_age_seconds: int = _TEMP_FILE_TTL_SECONDS
) -> None:
    """删除临时目录里超过保留期的文件（下载的视频此前从不清理）。"""
    deadline = time.time() - max_age_seconds
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_file() and entry.stat().st_mtime < deadline:
                entry.unlink(missing_ok=True)
        except OSError:
            continue


async def _open_stream(session: aiohttp.ClientSession, url: str):
    """逐跳校验重定向后返回最终响应，重定向次数超限则拒绝。

    Args:
        session: 复用的 aiohttp 会话。
        url: 初始下载地址。

    Returns:
        最终响应对象（未经读取）。

    Raises:
        VideoError: 重定向缺少 Location、次数超限或目标地址不安全。
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        await _assert_public_url(current)
        resp = await session.get(current, allow_redirects=False)
        if resp.status in _REDIRECT_STATUSES:
            location = resp.headers.get("Location") or ""
            resp.close()
            if not location:
                raise VideoError("下载地址返回重定向但缺少 Location，已拒绝。")
            current = urljoin(current, location)
            continue
        return resp
    raise VideoError("下载地址重定向次数过多，已拒绝。")


async def download_video_file(
    url: str,
    save_name: str,
    timeout: int = 300,
    max_size_mb: int | None = None,
) -> Path:
    """下载远程视频到临时目录。

    下载前校验目标地址（仅公网 http/https，含重定向逐跳校验），
    下载中按块流式写入磁盘并按 max_size_mb 硬截断，避免整段读入内存。
    文件名加随机前缀，避免不同会话的同名视频互相覆盖。

    Args:
        url: 视频地址。
        save_name: 消息里的原始文件名（会被清洗）。
        timeout: 下载超时（秒）。
        max_size_mb: 大小上限，超出立即中断。

    Returns:
        下载后的本地路径。

    Raises:
        VideoError: 地址不安全、HTTP 非 200 或文件超过上限。
    """
    tmp_dir = Path(tempfile.gettempdir()) / _TEMP_SUBDIR
    tmp_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_prune_temp_files, tmp_dir)

    safe_name = _sanitize_filename(save_name)
    dest = tmp_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    part_path = dest.with_name(dest.name + ".part")
    max_bytes = max_size_mb * _MB if max_size_mb else None

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    try:
        async with aiohttp.ClientSession(
            timeout=timeout_obj, trust_env=False
        ) as session:
            resp = await _open_stream(session, url)
            async with resp:
                if resp.status in _EXPIRED_STATUSES:
                    raise VideoExpiredError(
                        f"下载失败（HTTP {resp.status}），平台链接可能已过期或文件已删除",
                    )
                if resp.status != 200:
                    raise VideoError(f"下载视频失败：HTTP {resp.status}")
                declared = resp.headers.get("Content-Length")
                if (
                    max_bytes is not None
                    and declared
                    and declared.isdigit()
                    and int(declared) > max_bytes
                ):
                    raise VideoError(f"文件大小超过限制 {max_size_mb} MB。")
                written = 0
                with part_path.open("wb") as fh:
                    async for chunk in resp.content.iter_chunked(_CHUNK_SIZE):
                        written += len(chunk)
                        if max_bytes is not None and written > max_bytes:
                            raise VideoError(f"文件大小超过限制 {max_size_mb} MB。")
                        await asyncio.to_thread(fh.write, chunk)
        part_path.replace(dest)
        return dest
    finally:
        if part_path.exists():
            await asyncio.to_thread(part_path.unlink, True)


async def _validate(
    local_path: str, ext: str, max_size_mb: int, filename: str
) -> VideoFile:
    p = Path(local_path)
    if not p.is_file():
        raise VideoError("文件不存在或无法访问，请确认文件已上传完成。")

    size_bytes = p.stat().st_size
    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        size_mb = size_bytes / 1024 / 1024
        raise VideoError(f"文件大小 {size_mb:.1f} MB 超过限制 {max_size_mb} MB。")

    return VideoFile(
        path=str(p),
        mime_type=_MIME_MAP.get(ext, f"video/{ext}"),
        filename=filename,
        size_bytes=size_bytes,
    )

"""视频文件工具。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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


@dataclass
class VideoFile:
    path: str
    mime_type: str
    filename: str
    size_bytes: int


class VideoError(Exception):
    """视频处理错误。"""


def _get_ext(name: str) -> str:
    return Path(name).suffix.lstrip(".").lower()


def resolve_component_ref(comp) -> tuple[str, str]:
    """从 File 组件提取 (local_path, remote_url)。local_path 为空时返回空字符串。"""
    local = ""
    url = ""

    raw_file = getattr(comp, "file_", "") or ""
    raw_url = getattr(comp, "url", "") or ""

    if raw_file:
        if raw_file.startswith("file://") or raw_file.startswith("file:"):
            try:
                from urllib.parse import unquote, urlparse

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


def extract_file_component(event: AstrMessageEvent):
    """从消息链（含引用消息）中提取第一个 File 组件。"""
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
        if type(comp).__name__ == "File":
            return comp
    return None


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
    """校验并获取 File 组件的本地路径，返回 VideoFile（不读取内容）。"""
    name: str = getattr(file_comp, "name", "") or ""
    ext = _get_ext(name)

    if ext not in supported_formats:
        supported_str = "、".join(supported_formats)
        raise VideoError(f"文件格式 .{ext} 不支持，当前支持的格式：{supported_str}。")

    try:
        local_path: str = await file_comp.get_file()
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


async def download_video_file(
    url: str,
    save_name: str,
    timeout: int = 300,
    max_size_mb: int | None = None,
) -> Path:
    """下载远程视频到临时目录。

    可选传入 max_size_mb：下载前通过 Content-Length 预检，超出直接拒绝。
    """
    import re
    import tempfile

    tmp_dir = Path(tempfile.gettempdir()) / "astrbot_video_sense"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # 消息文件名可能含路径分隔符或非法字符，仅取 basename 并清理
    safe_name = Path(save_name or "video").name
    safe_name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", safe_name).strip()
    dest = tmp_dir / (safe_name or "video")

    max_bytes = max_size_mb * 1024 * 1024 if max_size_mb else None

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_obj, trust_env=False) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise VideoError(f"下载视频失败：HTTP {resp.status}")
            if max_bytes is not None:
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise VideoError(f"文件大小超过限制 {max_size_mb} MB。")
            data = await resp.read()

    if max_bytes is not None and len(data) > max_bytes:
        size_mb = len(data) / 1024 / 1024
        raise VideoError(f"文件大小 {size_mb:.1f} MB 超过限制 {max_size_mb} MB。")

    await asyncio.to_thread(dest.write_bytes, data)
    return dest


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

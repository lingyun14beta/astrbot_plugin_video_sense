"""Kimi（Moonshot）视频上传模块。

官方文档（https://platform.kimi.com/docs/guide/use-kimi-vision-model）要点：
- 支持视频理解的模型仅有 kimi-k3 / kimi-k2.6 / kimi-k2.7-code /
  kimi-k2.7-code-highspeed；moonshot-v1-*-vision-preview 系列仅支持图片。
- 视频不支持 base64 或 URL 直接传入：必须先 POST /v1/files（purpose="video"）
  上传，再通过 video_url 的 ms://<file-id> 引用（ms = moonshot storage）。
- 单文件上限 100MB；视觉请求体上限 100M。

本模块只负责上传，请求体构造与协议选择见 gemini_client.py。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp

_MB = 1024 * 1024
# 官方限制：单文件不超过 100MB（https://platform.kimi.com/docs/api/files-upload）
_MAX_VIDEO_BYTES = 100 * _MB

_HTTP_OK = 200

# base_url 可能带这些尾部，推导 /v1/files 时先剥离
_TAIL_SUFFIXES = ("/chat/completions", "/v1beta/openai", "/v1beta")


class KimiUploadError(Exception):
    """Kimi 视频上传失败，message 可直接透传给用户。"""


async def upload_video_file(
    api_key: str,
    base_url: str,
    file_path: str,
    mime_type: str,
    filename: str = "",
    timeout: int = 300,
) -> str:
    """上传视频到 Moonshot，返回 ms://<file-id> 引用。

    Args:
        api_key: Moonshot 平台 API Key。
        base_url: OpenAI 兼容接口地址（如 https://api.moonshot.cn/v1）。
        file_path: 本地视频文件路径。
        mime_type: 视频 MIME 类型（如 video/mp4）。
        filename: 上传文件名，留空使用文件路径中的名称。
        timeout: 上传请求总超时（秒）。

    Returns:
        ms://<file-id>，可直接作为 video_url 的 url 使用。

    Raises:
        KimiUploadError: Key 缺失、文件不存在/为空、超出大小限制或上传失败。
    """
    if not api_key:
        raise KimiUploadError("未配置 API Key，请在插件设置中填写。")
    p = Path(file_path)
    if not p.is_file():
        raise KimiUploadError(f"文件不存在或无法访问：{file_path}")
    size = p.stat().st_size
    if size <= 0:
        raise KimiUploadError("视频文件为空，无法上传。")
    if size > _MAX_VIDEO_BYTES:
        raise KimiUploadError(
            f"文件 {size / _MB:.1f} MB 超过 Kimi 单文件上限 100 MB，"
            "请压缩或截取视频片段后重试。",
        )

    url = build_upload_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"}
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    try:
        # 用文件句柄流式上传，避免整文件读入内存（上限 100MB）
        with p.open("rb") as fh:
            form = aiohttp.FormData()
            form.add_field(
                "file", fh, filename=filename or p.name, content_type=mime_type
            )
            form.add_field("purpose", "video")
            async with aiohttp.ClientSession(
                timeout=client_timeout, trust_env=False
            ) as session:
                async with session.post(url, headers=headers, data=form) as resp:
                    raw = await resp.text()
                    if resp.status != _HTTP_OK:
                        msg = _extract_error_message(raw) or f"HTTP {resp.status}"
                        raise KimiUploadError(f"上传失败（{resp.status}）：{msg}")
                    try:
                        info = json.loads(raw)
                    except json.JSONDecodeError as e:
                        raise KimiUploadError(f"上传响应解析失败：{e}") from e
    except aiohttp.ClientError as e:
        raise KimiUploadError(f"网络请求异常：{e}") from e

    if not isinstance(info, dict):
        raise KimiUploadError("上传响应格式异常（非 JSON 对象）。")
    file_id = (info.get("id") or "").strip()
    if not file_id:
        raise KimiUploadError("上传响应缺少文件 ID（id）。")
    return f"ms://{file_id}"


def build_upload_url(base_url: str) -> str:
    """由 OpenAI 兼容 base_url 推导 /v1/files 上传地址。

    Args:
        base_url: 配置的 Base URL（如 https://api.moonshot.cn/v1）。

    Returns:
        上传接口地址，如 https://api.moonshot.cn/v1/files。
    """
    base = (base_url or "").strip().rstrip("/")
    for suffix in _TAIL_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.endswith("/files"):
        return base
    return f"{base}/files"


def _extract_error_message(raw: str) -> str:
    """从响应文本提取用户可读的错误信息。"""
    if not raw:
        return ""
    if raw.lstrip().startswith("<"):
        return "网关返回 HTML 错误页（可能为反向代理限制请求大小，或上游服务错误）。"
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            err = data.get("error") or {}
            if isinstance(err, dict):
                return err.get("message", "")
            return str(err)
    except Exception:
        pass
    return raw[:200]

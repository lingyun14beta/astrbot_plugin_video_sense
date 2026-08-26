"""视频分析客户端，支持 Gemini 协议与 OpenAI 兼容协议。

协议选择（按模型名自动判断，可用 protocol 参数强制）：
- 模型名包含 "gemini"（或官方接口）→ Gemini 协议（generateContent）
- 其他模型（qwen-vl、gpt 等）→ OpenAI 兼容协议（/v1/chat/completions）

传输方式：
- 内嵌传输（inline_data / video_url data URL）：文件 ≤ max_inline_size_mb。
- Files API（官方推荐，免费层 2GB）：仅 Gemini 官方接口 + 大文件时使用。
  参考：https://ai.google.dev/gemini-api/docs/files
- Kimi（模型名含 "kimi"）：视频必须经 /v1/files（purpose=video）上传，
  再用 ms://<file-id> 引用，实现见 kimi_uploader.py。
  参考：https://platform.kimi.com/docs/guide/use-kimi-vision-model
- 百炼 qwen（模型名含 "qwen" 且直连百炼）：小视频 base64 内嵌（官方上限约
  7.5MB），大视频走免费临时 URL 上传（oss://，48 小时有效，≤1GB），
  实现见 qwen_uploader.py。
  参考：https://help.aliyun.com/zh/model-studio/get-temporary-file-url
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

try:
    from .ffmpeg_utils import FfmpegError, compress_video, find_ffmpeg
except ImportError:  # 非包上下文（测试直接导入模块）
    from ffmpeg_utils import FfmpegError, compress_video, find_ffmpeg

try:
    from .kimi_uploader import KimiUploadError, upload_video_file
except ImportError:  # 非包上下文（测试直接导入模块）
    from kimi_uploader import KimiUploadError, upload_video_file

try:
    from .qwen_uploader import QwenUploadError, upload_video_to_temp_url
except ImportError:  # 非包上下文（测试直接导入模块）
    from qwen_uploader import QwenUploadError, upload_video_to_temp_url

_OFFICIAL_HOSTS: frozenset[str] = frozenset(
    {
        "generativelanguage.googleapis.com",
        "aiplatform.googleapis.com",
    },
)

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
_MAX_BACKOFF = 8.0
_MB = 1024 * 1024

_HTTP_OK = 200
_HTTP_4XX_MIN = 400
_HTTP_5XX_MIN = 500

_FILE_STATE_ACTIVE = "ACTIVE"
_FILE_STATE_FAILED = "FAILED"

# 百炼官方限制：Base64 编码后的视频字符串 < 10MB（严格小于）。
# 7.4MB 原始视频编码后约 9.87MB（含 data URL 前缀仍 < 10MB）；7.5MB 会恰好等于 10MB。
_QWEN_BASE64_RAW_BYTES = int(7.4 * _MB)

# 百炼官方要求：使用 oss:// 临时 URL 调用时必须添加此请求头，否则接口报错。
# 参考：https://help.aliyun.com/zh/model-studio/get-temporary-file-url
_OSS_RESOLVE_HEADER = "X-DashScope-OssResourceResolve"
_OSS_RESOLVE_VALUE = "enable"

_PROTOCOL_AUTO = "auto"
_PROTOCOL_GEMINI = "gemini"
_PROTOCOL_OPENAI = "openai"


class GeminiClientError(Exception):
    """API 调用失败，message 可直接透传给 LLM。"""


class GeminiClient:
    """向视频理解 API 发送分析请求，自动选择协议与传输方式。

    同时支持官方 API（x-goog-api-key 鉴权）和 OpenAI 兼容中转站（Bearer 鉴权）。
    接入模式根据 base_url 的 host 自动判断。
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        base_url: str = "",
        timeout: int = 300,
        retry_times: int = 2,
        max_inline_size_mb: int = 15,
        use_files_api: bool = True,
        protocol: str = _PROTOCOL_AUTO,
        fps: float = 0.0,
        compress: bool = False,
        compress_max_duration: int = 120,
        compress_resolution: int = 720,
        compress_crf: int = 28,
    ) -> None:
        self._api_key = api_key.strip()
        self._model = self._normalize_model(model)
        self._system_prompt = system_prompt.strip()
        self._base_url = (base_url or "").strip().rstrip("/")
        self._timeout = timeout
        self._retry_times = retry_times
        self._max_inline_size_mb = max(0, int(max_inline_size_mb))
        self._use_files_api = bool(use_files_api)
        self._protocol = (protocol or _PROTOCOL_AUTO).strip().lower()
        try:
            self._fps = float(fps) if fps else 0.0
        except (TypeError, ValueError):
            self._fps = 0.0
        self._compress = bool(compress)
        self._compress_max_duration = max(0, int(compress_max_duration))
        self._compress_resolution = max(0, int(compress_resolution))
        self._compress_crf = max(0, int(compress_crf))
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    async def analyze_video(self, video) -> str:
        """分析一个视频文件（VideoFile），自动选择协议与传输方式。

        Args:
            video: 具有 path / mime_type / filename / size_bytes 属性的对象。

        Raises:
            GeminiClientError: 调用失败或返回为空。
        """
        # 百炼 qwen：小视频 base64 内嵌（官方上限 10MB 字符串），大视频走临时 URL
        if not self._is_gemini_protocol() and self._is_qwen() and self._is_dashscope():
            return await self._analyze_qwen(video)
        # Kimi：官方接口 + 视频只能经文件上传 + ms:// 引用（见 kimi_uploader.py）。
        # 仅 Moonshot 官方域名触发；中转站模型名叫 kimi-* 不受影响（ms:// 为 Moonshot 内部协议）
        if not self._is_gemini_protocol() and self._is_kimi() and self._is_moonshot():
            return await self._analyze_kimi(video)
        inline_limit = self._max_inline_size_mb * _MB
        if video.size_bytes <= inline_limit:
            return await self.analyze(
                await self._read_base64(video.path),
                video.mime_type,
            )
        # 大文件：Files API（仅 Gemini 官方）→ 压缩（可选）→ 报错
        can_files_api = (
            self._is_gemini_protocol() and self._use_files_api and self._is_official()
        )
        if can_files_api:
            file_uri = await self.upload_file(
                video.path, video.mime_type, video.filename
            )
            return await self.analyze_file(file_uri, video.mime_type)
        if self._compress:
            return await self._analyze_compressed(video)
        size_mb = video.size_bytes / _MB
        reason = (
            "Files API 仅 Gemini 官方接口提供"
            if self._is_gemini_protocol()
            else "OpenAI 兼容协议不支持 Files API 大文件上传"
        )
        raise GeminiClientError(
            f"文件 {size_mb:.1f} MB 超过内嵌上限 {self._max_inline_size_mb} MB，"
            f"{reason}。请在配置中开启「自动压缩」（需 ffmpeg），"
            "或手动压缩/截取视频片段后重试。",
        )

    async def analyze(self, video_b64: str, mime_type: str) -> str:
        """内嵌传输：分析视频，返回文字描述（按协议自动选择请求格式）。"""
        self._require_api_key()
        if self._is_gemini_protocol():
            payload = self._build_payload(video_b64, mime_type)
        else:
            payload = self._build_openai_payload(video_b64, mime_type)
        return await self._request_text(
            self._build_url(),
            self._build_headers(),
            payload,
        )

    async def analyze_file(self, file_uri: str, mime_type: str) -> str:
        """Files API：通过已上传文件的 URI 分析视频（仅 Gemini 协议）。"""
        self._require_api_key()
        if not self._is_gemini_protocol():
            raise GeminiClientError(
                "Files API 仅 Gemini 协议支持，当前协议无法引用文件。"
            )
        return await self._request_text(
            self._build_url(),
            self._build_headers(),
            self._build_file_payload(file_uri, mime_type),
        )

    async def upload_file(
        self, file_path: str, mime_type: str, display_name: str = ""
    ) -> str:
        """通过 Files API resumable 协议上传文件，返回可引用的 file_uri。

        流程（官方文档）：
        1. POST /upload/v1beta/files 发起上传（X-Goog-Upload-Protocol: resumable，
           X-Goog-Upload-Command: start），从响应头 X-Goog-Upload-URL 获取上传地址。
        2. PUT 上传地址写入文件字节（X-Goog-Upload-Command: upload, finalize）。
        3. 轮询 GET /v1beta/files/{name} 直到状态 ACTIVE。

        Raises:
            GeminiClientError: 上传或处理失败。
        """
        self._require_api_key()
        if not self._is_official():
            raise GeminiClientError(
                "Files API 仅 Gemini 官方接口提供，当前接入方（中转站）不支持，"
                "请使用官方接口或在配置中关闭「启用 Files API」。",
            )

        p = Path(file_path)
        if not p.is_file():
            raise GeminiClientError(f"文件不存在或无法访问：{file_path}")

        size = p.stat().st_size
        session = await self._get_session()

        # 1) 初始化上传
        start_headers = {
            **self._build_headers(),
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
        }
        metadata = {
            "file": {
                "display_name": display_name or p.name,
                "mime_type": mime_type,
            },
        }
        async with session.post(
            self._build_upload_url(), headers=start_headers, json=metadata
        ) as resp:
            if resp.status != _HTTP_OK:
                raw = await resp.text()
                msg = _extract_error_message(raw) or f"HTTP {resp.status}"
                raise GeminiClientError(f"上传初始化失败（{resp.status}）：{msg}")
            upload_url = resp.headers.get("X-Goog-Upload-URL", "").strip()
        if not upload_url:
            raise GeminiClientError("上传初始化失败：响应缺少 X-Goog-Upload-URL。")

        # 2) 写入文件字节
        data = await asyncio.to_thread(p.read_bytes)
        put_headers = {
            **self._build_headers(),
            "Content-Length": str(len(data)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }
        async with session.put(upload_url, headers=put_headers, data=data) as resp:
            raw = await resp.text()
            if resp.status != _HTTP_OK:
                msg = _extract_error_message(raw) or f"HTTP {resp.status}"
                raise GeminiClientError(f"上传失败（{resp.status}）：{msg}")
            try:
                file_info = json.loads(raw)
            except json.JSONDecodeError as e:
                raise GeminiClientError(f"上传响应解析失败：{e}") from e

        f = file_info.get("file") or {}
        name = f.get("name", "")
        uri = f.get("uri", "")
        state = f.get("state", "")
        if not name or not uri:
            raise GeminiClientError("上传响应缺少文件信息（name/uri）。")
        if state == _FILE_STATE_FAILED:
            raise GeminiClientError(f"文件处理失败：{f.get('error')}")
        if state == _FILE_STATE_ACTIVE:
            return uri

        # 3) 轮询直到处理完成
        return await self._wait_file_active(name, uri)

    async def close(self) -> None:
        """关闭底层 aiohttp session。"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _require_api_key(self) -> None:
        if not self._api_key:
            raise GeminiClientError("未配置 API Key，请在插件设置中填写。")

    async def _read_base64(self, file_path: str) -> str:
        raw = await asyncio.to_thread(Path(file_path).read_bytes)
        return base64.b64encode(raw).decode("ascii")

    async def _analyze_compressed(self, video) -> str:
        """大视频走 ffmpeg 压缩后内嵌分析（中转站/OpenAI 协议场景）。"""
        ffmpeg_path = find_ffmpeg()
        if not ffmpeg_path:
            raise GeminiClientError(
                "未检测到 ffmpeg，无法自动压缩。请在「平台日志」页面点击"
                "「安装 pip 库」安装 imageio-ffmpeg（依赖较大，约 80MB），"
                "或在系统安装 ffmpeg，然后重启 AstrBot。",
            )
        import tempfile

        dest_dir = Path(tempfile.gettempdir()) / "astrbot_video_sense" / "compress"
        compressed = None
        try:
            compressed = await compress_video(
                ffmpeg_path,
                video.path,
                dest_dir,
                self._max_inline_size_mb,
                max_duration_s=self._compress_max_duration,
                resolution=self._compress_resolution,
                crf=self._compress_crf,
            )
            b64 = await self._read_base64(compressed.path)
            return await self.analyze(b64, video.mime_type)
        except FfmpegError as e:
            raise GeminiClientError(f"视频压缩失败：{e}") from e
        finally:
            if compressed is not None:
                await asyncio.to_thread(Path(compressed.path).unlink, missing_ok=True)

    async def _analyze_qwen(self, video) -> str:
        """百炼 qwen 视频分析：小视频 base64 内嵌，大视频走免费临时 URL 上传。

        Args:
            video: 具有 path / mime_type / filename / size_bytes 属性的对象。

        Returns:
            视频分析结果文本。

        Raises:
            GeminiClientError: 上传或分析失败。
        """
        inline_limit = min(self._max_inline_size_mb * _MB, _QWEN_BASE64_RAW_BYTES)
        headers = self._build_headers()
        if video.size_bytes <= inline_limit:
            b64 = await self._read_base64(video.path)
            payload = self._build_openai_payload(b64, video.mime_type, fps=self._fps)
        else:
            try:
                file_ref = await upload_video_to_temp_url(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    model=self._model,
                    file_path=video.path,
                    mime_type=video.mime_type,
                    filename=getattr(video, "filename", "") or Path(video.path).name,
                    timeout=self._timeout,
                )
            except QwenUploadError as e:
                raise GeminiClientError(str(e)) from e
            payload = self._build_openai_payload_url(file_ref, fps=self._fps)
            # 官方要求：使用 oss:// 临时 URL 调用时必须显式声明资源解析
            headers = {**headers, _OSS_RESOLVE_HEADER: _OSS_RESOLVE_VALUE}
        return await self._request_text(
            self._build_url(),
            headers,
            payload,
        )

    async def _analyze_kimi(self, video) -> str:
        """Kimi 视频分析：上传文件（purpose=video）后用 ms:// 引用。

        Args:
            video: 具有 path / mime_type / filename 属性的对象。

        Returns:
            视频分析结果文本。

        Raises:
            GeminiClientError: 上传或分析失败。
        """
        try:
            file_ref = await upload_video_file(
                api_key=self._api_key,
                base_url=self._base_url,
                file_path=video.path,
                mime_type=video.mime_type,
                filename=getattr(video, "filename", "") or Path(video.path).name,
                timeout=self._timeout,
            )
        except KimiUploadError as e:
            raise GeminiClientError(str(e)) from e
        return await self._request_text(
            self._build_url(),
            self._build_headers(),
            self._build_kimi_payload(file_ref),
        )

    def _build_kimi_payload(self, file_ref: str) -> dict:
        """Kimi 请求体：视频通过 ms:// 文件 ID 引用（官方唯一视频接入方式）。

        Args:
            file_ref: ms://<file-id> 形式的文件引用。

        Returns:
            OpenAI 兼容协议请求体（content 为多模态 part 数组）。
        """
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请分析这段视频。"},
                        {
                            "type": "video_url",
                            "video_url": {"url": file_ref},
                        },
                    ],
                },
            ],
        }

    async def _request_text(self, url: str, headers: dict, payload: dict) -> str:
        last_error: str = "未知错误"
        for attempt in range(self._retry_times + 1):
            try:
                return await self._post(url, headers, payload)
            except GeminiClientError as e:
                last_error = str(e)
                if not _is_retryable_error(last_error):
                    raise
                if attempt < self._retry_times:
                    wait = min(_MAX_BACKOFF, 2**attempt) + random.uniform(0, 0.3)
                    await asyncio.sleep(wait)
        raise GeminiClientError(last_error)

    async def _wait_file_active(self, name: str, uri: str, timeout: int = 180) -> str:
        get_url = f"{self._effective_base()}/v1beta/{name}"
        session = await self._get_session()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        delay = 1.0
        while True:
            async with session.get(get_url, headers=self._build_headers()) as resp:
                raw = await resp.text()
                if resp.status != _HTTP_OK:
                    msg = _extract_error_message(raw) or f"HTTP {resp.status}"
                    raise GeminiClientError(f"查询文件状态失败（{resp.status}）：{msg}")
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise GeminiClientError(f"查询文件状态响应解析失败：{e}") from e
            f = data.get("file") or {}
            state = f.get("state", "")
            if state == _FILE_STATE_ACTIVE:
                return uri
            if state == _FILE_STATE_FAILED:
                raise GeminiClientError(f"文件处理失败：{f.get('error')}")
            if loop.time() >= deadline:
                raise GeminiClientError("等待文件处理超时，请稍后重试。")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout, trust_env=False)
        return self._session

    def _effective_base(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def _is_official(self) -> bool:
        try:
            host = (urlparse(self._effective_base()).hostname or "").lower()
            return host in _OFFICIAL_HOSTS
        except Exception:
            return False

    def _is_kimi(self) -> bool:
        """按模型名判断是否为 Kimi（Moonshot）：视频走上传 + ms:// 引用。"""
        return "kimi" in self._model.lower()

    def _is_moonshot(self) -> bool:
        """是否为 Moonshot 官方接口（api.moonshot.cn / api.moonshot.ai）。"""
        try:
            host = (urlparse(self._effective_base()).hostname or "").lower()
        except Exception:
            return False
        return host.endswith("moonshot.cn") or host.endswith("moonshot.ai")

    def _is_qwen(self) -> bool:
        """按模型名判断是否为千问系列（qwen）：视频走百炼专用链路。"""
        return "qwen" in self._model.lower()

    def _is_dashscope(self) -> bool:
        """是否为阿里云百炼直连（dashscope 域名或业务空间专属域名）。"""
        try:
            host = (urlparse(self._effective_base()).hostname or "").lower()
        except Exception:
            return False
        return host.endswith("dashscope.aliyuncs.com") or host.endswith(
            "maas.aliyuncs.com"
        )

    def _is_gemini_protocol(self) -> bool:
        """协议判定：官方接口强制 Gemini；否则按 protocol 配置或模型名判断。"""
        if self._is_official():
            return True
        if self._protocol == _PROTOCOL_GEMINI:
            return True
        if self._protocol == _PROTOCOL_OPENAI:
            return False
        return "gemini" in self._model.lower()

    def _build_url(self) -> str:
        base = self._effective_base()
        if self._is_gemini_protocol():
            for suffix in ("/v1/chat/completions", "/v1beta/openai", "/v1beta", "/v1"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            return f"{base}/v1beta/models/{self._model}:generateContent"
        # OpenAI 兼容协议
        if base.endswith("/chat/completions"):
            return base  # base 已包含完整端点，原样使用
        if base.endswith("/v1beta/openai"):
            base = base[: -len("/v1beta/openai")]  # Gemini 中转端点 → 根路径
        return f"{base}/chat/completions"

    def _build_upload_url(self) -> str:
        base = self._effective_base()
        for suffix in ("/v1/chat/completions", "/v1beta/openai", "/v1beta", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return f"{base}/upload/v1beta/files"

    def _build_headers(self) -> dict[str, str]:
        if self._is_official():
            return {
                "x-goog-api-key": self._api_key,
                "Content-Type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, video_b64: str, mime_type: str) -> dict:
        return {
            "system_instruction": {
                "parts": [{"text": self._system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": video_b64,
                            },
                        },
                        {"text": "请分析这段视频。"},
                    ],
                },
            ],
        }

    def _build_file_payload(self, file_uri: str, mime_type: str) -> dict:
        return {
            "system_instruction": {
                "parts": [{"text": self._system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "file_data": {
                                "mime_type": mime_type,
                                "file_uri": file_uri,
                            },
                        },
                        {"text": "请分析这段视频。"},
                    ],
                },
            ],
        }

    def _build_openai_payload(
        self, video_b64: str, mime_type: str, fps: float | None = None
    ) -> dict:
        """OpenAI 兼容协议请求体：视频通过 video_url data URL 内嵌。"""
        data_url = f"data:{mime_type};base64,{video_b64}"
        return self._openai_multimodal_payload(data_url, fps)

    def _build_openai_payload_url(self, url: str, fps: float | None = None) -> dict:
        """OpenAI 兼容协议请求体：视频通过外部 URL（如 oss:// 临时 URL）引用。"""
        return self._openai_multimodal_payload(url, fps)

    def _openai_multimodal_payload(self, video_ref: str, fps: float | None) -> dict:
        """构造 OpenAI 兼容协议多模态请求体（video_url 引用，可选 fps）。

        Args:
            video_ref: video_url.url（公网 URL、oss:// 临时 URL 或 data URL）。
            fps: 抽帧频率（每秒帧数），仅在 >0 时随请求发送。

        Returns:
            OpenAI 兼容协议请求体（content 为多模态 part 数组）。
        """
        video_part: dict = {"type": "video_url", "video_url": {"url": video_ref}}
        if fps and fps > 0:
            video_part["fps"] = fps
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请分析这段视频。"},
                        video_part,
                    ],
                },
            ],
        }

    async def _post(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
    ) -> str:
        session = await self._get_session()
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                raw = await resp.text()

                if resp.status == _HTTP_OK:
                    if self._is_gemini_protocol():
                        return _parse_response(raw)
                    return _parse_openai_response(raw)

                msg = _extract_error_message(raw) or f"HTTP {resp.status}"
                if _HTTP_4XX_MIN <= resp.status < _HTTP_5XX_MIN:
                    raise GeminiClientError(
                        f"请求被拒绝（{resp.status}）：{msg}",
                    )
                raise GeminiClientError(
                    f"[retryable] 服务端错误（{resp.status}）：{msg}",
                )

        except aiohttp.ClientError as e:
            raise GeminiClientError(f"[retryable] 网络请求异常：{e}") from e

    @staticmethod
    def _normalize_model(model: str) -> str:
        model = (model or "").strip().removeprefix("models/")
        return model or "gemini-2.0-flash"


def _parse_response(raw: str) -> str:
    """解析 Gemini generateContent 响应，返回文本内容。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GeminiClientError(f"响应解析失败（非 JSON）：{e}") from e

    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise GeminiClientError(f"API 返回错误：{msg}")

    candidates = data.get("candidates") or []
    if not candidates:
        raise GeminiClientError(
            "API 返回空结果（candidates 为空），可能触发了内容过滤。",
        )

    parts = candidates[0].get("content", {}).get("parts") or []
    for part in parts:
        text = part.get("text", "")
        if text and text.strip():
            return text.strip()

    raise GeminiClientError("API 返回结果中没有文本内容。")


def _parse_openai_response(raw: str) -> str:
    """解析 OpenAI 兼容协议响应（chat/completions），返回文本内容。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GeminiClientError(f"响应解析失败（非 JSON）：{e}") from e

    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise GeminiClientError(f"API 返回错误：{msg}")

    choices = data.get("choices") or []
    if not choices:
        raise GeminiClientError(
            "API 返回空结果（choices 为空），可能触发了内容过滤。",
        )

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    # 部分模型以内容块列表返回（多模态输出）
    if isinstance(content, list):
        texts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("text")
        ]
        joined = "".join(texts).strip()
        if joined:
            return joined

    raise GeminiClientError("API 返回结果中没有文本内容。")


def _extract_error_message(raw: str) -> str:
    if not raw:
        return ""
    if raw.lstrip().startswith("<"):
        # 网关返回 HTML 错误页（nginx/反代），常见于请求体过大或上游错误
        return (
            "网关返回 HTML 错误页（可能为反向代理限制请求大小，"
            "或上游服务错误）。可尝试降低「内嵌传输上限」并开启「自动压缩」"
            "以减小请求体。"
        )
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


def _is_retryable_error(message: str) -> bool:
    return "[retryable]" in message

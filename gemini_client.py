"""视频分析客户端，支持 Gemini 协议与 OpenAI 兼容协议。

协议选择（按模型名自动判断，可用 protocol 参数强制）：
- 模型名包含 "gemini"（或官方接口）→ Gemini 协议（generateContent）
- 其他模型（qwen-vl、gpt 等）→ OpenAI 兼容协议（/v1/chat/completions）

传输方式：
- 内嵌传输（inline_data / video_url data URL）：文件 ≤ max_inline_size_mb。
- Files API（官方推荐，免费层 2GB）：仅 Gemini 官方接口 + 大文件时使用。
  参考：https://ai.google.dev/gemini-api/docs/files
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

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
        inline_limit = self._max_inline_size_mb * _MB
        if video.size_bytes <= inline_limit:
            return await self.analyze(
                await self._read_base64(video.path),
                video.mime_type,
            )
        if not self._is_gemini_protocol():
            size_mb = video.size_bytes / _MB
            raise GeminiClientError(
                f"文件 {size_mb:.1f} MB 超过内嵌上限 {self._max_inline_size_mb} MB，"
                "OpenAI 兼容协议不支持 Files API 大文件上传，"
                "请压缩或截取视频片段后重试。",
            )
        if self._use_files_api:
            if not self._is_official():
                size_mb = video.size_bytes / _MB
                raise GeminiClientError(
                    f"文件 {size_mb:.1f} MB 超过内嵌上限 {self._max_inline_size_mb} MB，"
                    "但当前接入方不支持 Files API（Files API 仅 Gemini 官方接口提供）。"
                    "请在配置中关闭「启用 Files API」，"
                    "或将视频压缩/截取到内嵌上限以内。",
                )
            file_uri = await self.upload_file(
                video.path, video.mime_type, video.filename
            )
            return await self.analyze_file(file_uri, video.mime_type)
        size_mb = video.size_bytes / _MB
        raise GeminiClientError(
            f"文件 {size_mb:.1f} MB 超过内嵌上限 {self._max_inline_size_mb} MB，"
            "且未启用 Files API，请压缩视频或在配置中开启 Files API。",
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

    def _build_openai_payload(self, video_b64: str, mime_type: str) -> dict:
        """OpenAI 兼容协议请求体：视频通过 video_url data URL 内嵌。"""
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
                            "video_url": {
                                "url": f"data:{mime_type};base64,{video_b64}",
                            },
                        },
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
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            err = data.get("error") or {}
            if isinstance(err, dict):
                return err.get("message", "")
            return str(err)
    except Exception:
        pass
    return raw[:200] if raw else ""


def _is_retryable_error(message: str) -> bool:
    return "[retryable]" in message

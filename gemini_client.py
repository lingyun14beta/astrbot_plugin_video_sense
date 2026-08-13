"""Gemini 视频分析客户端，同时支持官方 API 和 OpenAI 兼容中转站。"""

from __future__ import annotations

import asyncio
import json
import random
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

_HTTP_OK = 200
_HTTP_4XX_MIN = 400
_HTTP_5XX_MIN = 500


class GeminiClientError(Exception):
    """Gemini 调用失败，message 可直接透传给 LLM。"""


class GeminiClient:
    """向 Gemini generateContent 接口发送视频分析请求。

    同时支持官方 API（x-goog-api-key 鉴权）和 OpenAI 兼容中转站（Bearer 鉴权）。
    接入模式根据 base_url 的 host 自动判断。
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        base_url: str = "",
        timeout: int = 180,
        retry_times: int = 2,
    ) -> None:
        self._api_key = api_key.strip()
        self._model = self._normalize_model(model)
        self._system_prompt = system_prompt.strip()
        self._base_url = (base_url or "").strip().rstrip("/")
        self._timeout = timeout
        self._retry_times = retry_times
        self._session: aiohttp.ClientSession | None = None

    async def analyze(self, video_b64: str, mime_type: str) -> str:
        """分析视频，返回 Gemini 的文字描述。

        Raises:
            GeminiClientError: 调用失败或返回为空。
        """
        if not self._api_key:
            raise GeminiClientError("未配置 API Key，请在插件设置中填写。")

        url = self._build_url()
        headers = self._build_headers()
        payload = self._build_payload(video_b64, mime_type)

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

    async def close(self) -> None:
        """关闭底层 aiohttp session。"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

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

    def _build_url(self) -> str:
        base = self._effective_base()
        for suffix in ("/v1/chat/completions", "/v1beta/openai", "/v1beta", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return f"{base}/v1beta/models/{self._model}:generateContent"

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
                    return _parse_response(raw)

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

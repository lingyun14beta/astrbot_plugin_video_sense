"""Tests for gemini_client.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from gemini_client import (
    GeminiClient,
    GeminiClientError,
    _is_retryable_error,
    _parse_response,
)


class TestNormalizeModel:
    def test_normal_model(self):
        assert GeminiClient._normalize_model("gemini-2.0-flash") == "gemini-2.0-flash"

    def test_models_prefix(self):
        assert (
            GeminiClient._normalize_model("models/gemini-2.0-flash")
            == "gemini-2.0-flash"
        )

    def test_empty(self):
        assert GeminiClient._normalize_model("") == "gemini-2.0-flash"

    def test_whitespace(self):
        assert GeminiClient._normalize_model("  gemini-pro  ") == "gemini-pro"


class TestIsOfficial:
    def test_default_url(self):
        client = GeminiClient(
            api_key="test",
            model="gemini-2.0-flash",
            system_prompt="test",
        )
        assert client._is_official() is True

    def test_custom_url(self):
        client = GeminiClient(
            api_key="test",
            model="gemini-2.0-flash",
            system_prompt="test",
            base_url="https://my-proxy.example.com",
        )
        assert client._is_official() is False

    def test_aiplatform(self):
        client = GeminiClient(
            api_key="test",
            model="gemini-2.0-flash",
            system_prompt="test",
            base_url="https://aiplatform.googleapis.com",
        )
        assert client._is_official() is True


class TestBuildUrl:
    def test_default_url(self):
        client = GeminiClient(
            api_key="test",
            model="gemini-2.0-flash",
            system_prompt="test",
        )
        url = client._build_url()
        assert "generativelanguage.googleapis.com" in url
        assert "gemini-2.0-flash" in url
        assert url.endswith(":generateContent")

    def test_custom_base_url(self):
        client = GeminiClient(
            api_key="test",
            model="gemini-pro",
            system_prompt="test",
            base_url="https://my-proxy.example.com/v1",
        )
        url = client._build_url()
        assert "my-proxy.example.com" in url
        assert "gemini-pro" in url

    def test_base_with_trailing_chat_completions(self):
        client = GeminiClient(
            api_key="test",
            model="gemini-2.0-flash",
            system_prompt="test",
            base_url="https://my-proxy.example.com/v1/chat/completions",
        )
        url = client._build_url()
        assert "/v1/chat/completions" not in url


class TestBuildHeaders:
    def test_official_headers(self):
        client = GeminiClient(
            api_key="test-key",
            model="gemini-2.0-flash",
            system_prompt="test",
        )
        headers = client._build_headers()
        assert headers["x-goog-api-key"] == "test-key"
        assert "Authorization" not in headers

    def test_proxy_headers(self):
        client = GeminiClient(
            api_key="test-key",
            model="gemini-2.0-flash",
            system_prompt="test",
            base_url="https://my-proxy.example.com",
        )
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert "x-goog-api-key" not in headers


class TestBuildPayload:
    def test_payload_structure(self, sample_video_base64):
        client = GeminiClient(
            api_key="test",
            model="gemini-2.0-flash",
            system_prompt="Analyze this video.",
        )
        payload = client._build_payload(sample_video_base64, "video/mp4")

        assert (
            payload["system_instruction"]["parts"][0]["text"] == "Analyze this video."
        )
        assert len(payload["contents"]) == 1
        assert payload["contents"][0]["role"] == "user"
        assert len(payload["contents"][0]["parts"]) == 2
        inline = payload["contents"][0]["parts"][0]["inline_data"]
        assert inline["mime_type"] == "video/mp4"
        assert inline["data"] == sample_video_base64
        assert payload["contents"][0]["parts"][1]["text"] == "请分析这段视频。"


class TestParseResponse:
    def test_valid_response(self):
        raw = json.dumps(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "This is a beautiful video."}]}}
                ]
            }
        )
        assert _parse_response(raw) == "This is a beautiful video."

    def test_multiple_parts_takes_first_text(self):
        raw = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": ""},
                                {"text": "Second part"},
                            ]
                        }
                    }
                ]
            }
        )
        assert _parse_response(raw) == "Second part"

    def test_error_response(self):
        raw = json.dumps({"error": {"message": "Invalid API key"}})
        with pytest.raises(GeminiClientError, match="API 返回错误"):
            _parse_response(raw)

    def test_empty_candidates(self):
        raw = json.dumps({"candidates": []})
        with pytest.raises(GeminiClientError, match="空结果"):
            _parse_response(raw)

    def test_no_text_in_parts(self):
        raw = json.dumps(
            {"candidates": [{"content": {"parts": [{"inline_data": {}}]}}]}
        )
        with pytest.raises(GeminiClientError, match="没有文本内容"):
            _parse_response(raw)

    def test_invalid_json(self):
        with pytest.raises(GeminiClientError, match="非 JSON"):
            _parse_response("not json {{{")


class TestIsRetryableError:
    def test_retryable(self):
        assert _is_retryable_error("[retryable] Network error") is True

    def test_not_retryable(self):
        assert _is_retryable_error("Invalid API key") is False


class TestAnalyzeWithoutApiKey:
    async def test_empty_api_key(self, sample_video_base64):
        client = GeminiClient(
            api_key="",
            model="gemini-2.0-flash",
            system_prompt="test",
        )
        with pytest.raises(GeminiClientError, match="未配置 API Key"):
            await client.analyze(sample_video_base64, "video/mp4")

    async def test_analyze_file_empty_api_key(self):
        client = GeminiClient(
            api_key="",
            model="gemini-2.0-flash",
            system_prompt="test",
        )
        with pytest.raises(GeminiClientError, match="未配置 API Key"):
            await client.analyze_file("https://example.com/v1beta/files/1", "video/mp4")


class _FakeResp:
    """简易 aiohttp 响应替身。"""

    def __init__(self, status=200, json_data=None, headers=None, texts=None):
        self.status = status
        self.headers = headers or {}
        self._json = json_data
        self._text = json.dumps(json_data) if json_data is not None else ""
        self._texts = list(texts or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def text(self):
        if self._texts:
            return self._texts.pop(0)
        return self._text


class _FakeSession:
    """按调用顺序分发 post/put/get 响应的会话替身。

    模拟 aiohttp：方法为同步，返回上下文管理器。
    """

    def __init__(self, post=None, put=None, gets=None):
        self._post = post
        self._put = put
        self._gets = list(gets or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, url, headers=None, json=None):
        return self._post

    def put(self, url, headers=None, data=None):
        return self._put

    def get(self, url, headers=None):
        return self._gets.pop(0) if self._gets else None


class TestBuildFilePayload:
    def test_payload_structure(self):
        client = GeminiClient(
            api_key="test",
            model="gemini-2.0-flash",
            system_prompt="Analyze this video.",
        )
        payload = client._build_file_payload(
            "https://generativelanguage.googleapis.com/v1beta/files/abc-123",
            "video/mp4",
        )
        assert (
            payload["system_instruction"]["parts"][0]["text"] == "Analyze this video."
        )
        part = payload["contents"][0]["parts"][0]
        assert part["file_data"]["mime_type"] == "video/mp4"
        assert part["file_data"]["file_uri"] == (
            "https://generativelanguage.googleapis.com/v1beta/files/abc-123"
        )
        assert "inline_data" not in part


class TestBuildUploadUrl:
    def test_default_base(self):
        client = GeminiClient(api_key="k", model="m", system_prompt="s")
        assert client._build_upload_url().endswith("/upload/v1beta/files")

    def test_proxy_base_strips_suffix(self):
        client = GeminiClient(
            api_key="k",
            model="m",
            system_prompt="s",
            base_url="https://proxy.example.com/v1",
        )
        assert (
            client._build_upload_url()
            == "https://proxy.example.com/upload/v1beta/files"
        )


class TestAnalyzeVideo:
    def _make_video(self, size_bytes, tmp_path, name="clip.mp4"):
        p = tmp_path / name
        p.write_bytes(b"\x00" * size_bytes)
        return type(
            "Video",
            (),
            {
                "path": str(p),
                "mime_type": "video/mp4",
                "filename": name,
                "size_bytes": size_bytes,
            },
        )()

    async def test_inline_branch(self, tmp_path):
        client = GeminiClient(
            api_key="k", model="m", system_prompt="s", max_inline_size_mb=15
        )
        video = self._make_video(1024, tmp_path)
        client.analyze = AsyncMock(return_value="内嵌分析结果")
        client.upload_file = AsyncMock(return_value="uri")
        result = await client.analyze_video(video)
        assert result == "内嵌分析结果"
        client.analyze.assert_awaited_once()
        client.upload_file.assert_not_awaited()

    async def test_files_api_branch(self, tmp_path):
        client = GeminiClient(
            api_key="k", model="m", system_prompt="s", max_inline_size_mb=1
        )
        video = self._make_video(2 * 1024 * 1024, tmp_path)
        client.upload_file = AsyncMock(return_value="https://gen/v1beta/files/1")
        client.analyze_file = AsyncMock(return_value="Files API 分析结果")
        result = await client.analyze_video(video)
        assert result == "Files API 分析结果"
        client.upload_file.assert_awaited_once_with(video.path, "video/mp4", "clip.mp4")
        client.analyze_file.assert_awaited_once()

    async def test_files_api_disabled_and_too_large(self, tmp_path):
        client = GeminiClient(
            api_key="k",
            model="m",
            system_prompt="s",
            max_inline_size_mb=1,
            use_files_api=False,
        )
        video = self._make_video(2 * 1024 * 1024, tmp_path)
        with pytest.raises(GeminiClientError, match="超过内嵌上限"):
            await client.analyze_video(video)


class TestUploadFile:
    def _client(self):
        return GeminiClient(api_key="k", model="m", system_prompt="s")

    async def test_upload_returns_active_uri(self, tmp_path):
        client = self._client()
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 64)
        session = _FakeSession(
            post=_FakeResp(headers={"X-Goog-Upload-URL": "https://up.example.com/x"}),
            put=_FakeResp(
                json_data={
                    "file": {
                        "name": "files/abc-123",
                        "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc-123",
                        "state": "ACTIVE",
                    },
                }
            ),
        )
        with patch.object(
            GeminiClient, "_get_session", new=AsyncMock(return_value=session)
        ):
            uri = await client.upload_file(str(p), "video/mp4", "clip.mp4")
        assert uri == "https://generativelanguage.googleapis.com/v1beta/files/abc-123"

    async def test_upload_polls_until_active(self, tmp_path):
        client = self._client()
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 64)
        session = _FakeSession(
            post=_FakeResp(headers={"X-Goog-Upload-URL": "https://up.example.com/x"}),
            put=_FakeResp(
                json_data={
                    "file": {"name": "files/1", "uri": "uri1", "state": "PROCESSING"}
                }
            ),
            gets=[
                _FakeResp(
                    json_data={
                        "file": {
                            "name": "files/1",
                            "uri": "uri1",
                            "state": "PROCESSING",
                        }
                    }
                ),
                _FakeResp(
                    json_data={
                        "file": {"name": "files/1", "uri": "uri1", "state": "ACTIVE"}
                    }
                ),
            ],
        )
        with patch.object(
            GeminiClient, "_get_session", new=AsyncMock(return_value=session)
        ):
            uri = await client.upload_file(str(p), "video/mp4", "clip.mp4")
        assert uri == "uri1"

    async def test_upload_failed_state(self, tmp_path):
        client = self._client()
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 64)
        session = _FakeSession(
            post=_FakeResp(headers={"X-Goog-Upload-URL": "https://up.example.com/x"}),
            put=_FakeResp(
                json_data={
                    "file": {
                        "name": "files/1",
                        "uri": "uri1",
                        "state": "FAILED",
                        "error": {"message": "invalid video"},
                    },
                }
            ),
        )
        with patch.object(
            GeminiClient, "_get_session", new=AsyncMock(return_value=session)
        ):
            with pytest.raises(GeminiClientError, match="invalid video"):
                await client.upload_file(str(p), "video/mp4", "clip.mp4")

    async def test_upload_missing_upload_url(self, tmp_path):
        client = self._client()
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 64)
        session = _FakeSession(post=_FakeResp(headers={}))
        with patch.object(
            GeminiClient, "_get_session", new=AsyncMock(return_value=session)
        ):
            with pytest.raises(GeminiClientError, match="X-Goog-Upload-URL"):
                await client.upload_file(str(p), "video/mp4", "clip.mp4")

    async def test_upload_file_not_exists(self):
        client = self._client()
        with pytest.raises(GeminiClientError, match="不存在"):
            await client.upload_file("/nonexistent/clip.mp4", "video/mp4", "clip.mp4")

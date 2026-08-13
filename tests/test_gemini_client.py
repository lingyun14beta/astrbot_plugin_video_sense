"""Tests for gemini_client.py."""

from __future__ import annotations

import json

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
        assert (
            GeminiClient._normalize_model("  gemini-pro  ") == "gemini-pro"
        )


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

        assert payload["system_instruction"]["parts"][0]["text"] == "Analyze this video."
        assert len(payload["contents"]) == 1
        assert payload["contents"][0]["role"] == "user"
        assert len(payload["contents"][0]["parts"]) == 2
        inline = payload["contents"][0]["parts"][0]["inline_data"]
        assert inline["mime_type"] == "video/mp4"
        assert inline["data"] == sample_video_base64
        assert payload["contents"][0]["parts"][1]["text"] == "请分析这段视频。"


class TestParseResponse:
    def test_valid_response(self):
        raw = json.dumps({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "This is a beautiful video."}
                        ]
                    }
                }
            ]
        })
        assert _parse_response(raw) == "This is a beautiful video."

    def test_multiple_parts_takes_first_text(self):
        raw = json.dumps({
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
        })
        assert _parse_response(raw) == "Second part"

    def test_error_response(self):
        raw = json.dumps({
            "error": {"message": "Invalid API key"}
        })
        with pytest.raises(GeminiClientError, match="API 返回错误"):
            _parse_response(raw)

    def test_empty_candidates(self):
        raw = json.dumps({"candidates": []})
        with pytest.raises(GeminiClientError, match="空结果"):
            _parse_response(raw)

    def test_no_text_in_parts(self):
        raw = json.dumps({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inline_data": {}}
                        ]
                    }
                }
            ]
        })
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

"""Tests for kimi_uploader.py."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from kimi_uploader import (
    KimiUploadError,
    build_upload_url,
    upload_video_file,
)


class TestBuildUploadUrl:
    def test_moonshot_base(self):
        assert (
            build_upload_url("https://api.moonshot.cn/v1")
            == "https://api.moonshot.cn/v1/files"
        )

    def test_base_without_v1(self):
        assert build_upload_url("https://api.moonshot.cn") == "https://api.moonshot.cn/files"

    def test_base_with_chat_completions_suffix(self):
        assert (
            build_upload_url("https://proxy.example.com/v1/chat/completions")
            == "https://proxy.example.com/v1/files"
        )

    def test_base_with_v1beta_openai_suffix(self):
        assert (
            build_upload_url("https://proxy.example.com/v1beta/openai")
            == "https://proxy.example.com/files"
        )

    def test_base_already_ends_with_files(self):
        assert (
            build_upload_url("https://proxy.example.com/v1/files")
            == "https://proxy.example.com/v1/files"
        )

    def test_empty_base(self):
        assert build_upload_url("") == "/files"


class _FakeResp:
    """简化 aiohttp 响应替身。"""

    def __init__(self, status=200, json_data=None):
        self.status = status
        self._json = json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def text(self):
        return json.dumps(self._json) if self._json is not None else ""


class _FakeClientSession:
    """aiohttp.ClientSession 替身：post 返回预设响应。"""

    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, url, headers=None, data=None):
        return self._resp


class TestUploadVideoFile:
    def _patch_session(self, resp):
        return patch(
            "kimi_uploader.aiohttp.ClientSession",
            return_value=_FakeClientSession(resp),
        )

    async def test_upload_success_returns_ms_ref(self, sample_video_path):
        with self._patch_session(_FakeResp(json_data={"id": "file-abc"})):
            ref = await upload_video_file(
                "sk-test",
                "https://api.moonshot.cn/v1",
                str(sample_video_path),
                "video/mp4",
            )
        assert ref == "ms://file-abc"

    async def test_empty_api_key(self, sample_video_path):
        with pytest.raises(KimiUploadError, match="未配置 API Key"):
            await upload_video_file(
                "",
                "https://api.moonshot.cn/v1",
                str(sample_video_path),
                "video/mp4",
            )

    async def test_missing_file(self):
        with pytest.raises(KimiUploadError, match="不存在"):
            await upload_video_file(
                "sk-test",
                "https://api.moonshot.cn/v1",
                "/nonexistent/x.mp4",
                "video/mp4",
            )

    async def test_empty_file(self, temp_dir):
        p = temp_dir / "empty.mp4"
        p.write_bytes(b"")
        with pytest.raises(KimiUploadError, match="为空"):
            await upload_video_file(
                "sk-test",
                "https://api.moonshot.cn/v1",
                str(p),
                "video/mp4",
            )

    async def test_too_large(self, sample_video_path):
        with patch("kimi_uploader._MAX_VIDEO_BYTES", 10):
            with pytest.raises(KimiUploadError, match="100 MB"):
                await upload_video_file(
                    "sk-test",
                    "https://api.moonshot.cn/v1",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_upload_error_response(self, sample_video_path):
        resp = _FakeResp(status=400, json_data={"error": {"message": "invalid purpose"}})
        with self._patch_session(resp):
            with pytest.raises(KimiUploadError, match="上传失败（400）：invalid purpose"):
                await upload_video_file(
                    "sk-test",
                    "https://api.moonshot.cn/v1",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_missing_file_id(self, sample_video_path):
        with self._patch_session(_FakeResp(json_data={"object": "file"})):
            with pytest.raises(KimiUploadError, match="文件 ID"):
                await upload_video_file(
                    "sk-test",
                    "https://api.moonshot.cn/v1",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_invalid_response_shape(self, sample_video_path):
        """响应为 JSON 数组等非对象结构：给出明确错误而非 AttributeError。"""
        with self._patch_session(_FakeResp(json_data=["not", "an", "object"])):
            with pytest.raises(KimiUploadError, match="格式异常"):
                await upload_video_file(
                    "sk-test",
                    "https://api.moonshot.cn/v1",
                    str(sample_video_path),
                    "video/mp4",
                )

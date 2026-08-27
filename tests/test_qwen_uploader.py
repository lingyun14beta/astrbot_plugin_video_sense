"""Tests for qwen_uploader.py."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from qwen_uploader import (
    QwenUploadError,
    build_policy_url,
    upload_video_to_temp_url,
)

_POLICY = {
    "data": {
        "upload_dir": "upload/2024/12/01",
        "upload_host": "https://dashscope.oss-cn-beijing.aliyuncs.com",
        "oss_access_key_id": "LTAI-test",
        "policy": "eyJleHBpcmF0aW9uIjoi...",
        "signature": "abc123",
        "x_oss_object_acl": "public-read",
        "x_oss_forbid_overwrite": "true",
    },
    "request_id": "req-1",
}


class TestBuildPolicyUrl:
    def test_dashscope_base(self):
        assert (
            build_policy_url("https://dashscope.aliyuncs.com/compatible-mode/v1")
            == "https://dashscope.aliyuncs.com/api/v1/uploads"
        )

    def test_maas_workspace_base(self):
        assert (
            build_policy_url(
                "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            )
            == "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/uploads"
        )

    def test_root_base(self):
        assert (
            build_policy_url("https://dashscope.aliyuncs.com")
            == "https://dashscope.aliyuncs.com/api/v1/uploads"
        )

    def test_base_already_uploads_endpoint(self):
        assert (
            build_policy_url("https://dashscope.aliyuncs.com/api/v1/uploads")
            == "https://dashscope.aliyuncs.com/api/v1/uploads"
        )

    def test_empty_base(self):
        assert build_policy_url("") == "/api/v1/uploads"


class _FakeResp:
    """简化 aiohttp 响应替身。"""

    def __init__(self, status=200, json_data=None, text=""):
        self.status = status
        self._json = json_data
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def text(self):
        if self._text:
            return self._text
        return json.dumps(self._json) if self._json is not None else ""


class _FakeClientSession:
    """aiohttp.ClientSession 替身：get/post 按顺序返回预设响应。"""

    def __init__(self, get_resp=None, post_resp=None):
        self._get_resp = get_resp
        self._post_resp = post_resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get(self, url, headers=None, params=None):
        return self._get_resp

    def post(self, url, headers=None, data=None):
        return self._post_resp


class TestUploadVideoToTempUrl:
    def _patch_session(self, get_resp, post_resp=None):
        return patch(
            "qwen_uploader.aiohttp.ClientSession",
            return_value=_FakeClientSession(get_resp, post_resp),
        )

    async def test_upload_success_returns_oss_url(self, sample_video_path):
        with self._patch_session(
            _FakeResp(json_data=_POLICY), _FakeResp(status=200, text="")
        ):
            url = await upload_video_to_temp_url(
                "sk-test",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "qwen-vl-max",
                str(sample_video_path),
                "video/mp4",
            )
        assert url == "oss://upload/2024/12/01/test.mp4"

    async def test_empty_api_key(self, sample_video_path):
        with pytest.raises(QwenUploadError, match="未配置 API Key"):
            await upload_video_to_temp_url(
                "",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "qwen-vl-max",
                str(sample_video_path),
                "video/mp4",
            )

    async def test_empty_model(self, sample_video_path):
        with pytest.raises(QwenUploadError, match="未指定模型"):
            await upload_video_to_temp_url(
                "sk-test",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "",
                str(sample_video_path),
                "video/mp4",
            )

    async def test_missing_file(self):
        with pytest.raises(QwenUploadError, match="不存在"):
            await upload_video_to_temp_url(
                "sk-test",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "qwen-vl-max",
                "/nonexistent/x.mp4",
                "video/mp4",
            )

    async def test_empty_file(self, temp_dir):
        p = temp_dir / "empty.mp4"
        p.write_bytes(b"")
        with pytest.raises(QwenUploadError, match="为空"):
            await upload_video_to_temp_url(
                "sk-test",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "qwen-vl-max",
                str(p),
                "video/mp4",
            )

    async def test_too_large(self, sample_video_path):
        with patch("qwen_uploader._MAX_UPLOAD_BYTES", 10):
            with pytest.raises(QwenUploadError, match="1GB"):
                await upload_video_to_temp_url(
                    "sk-test",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "qwen-vl-max",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_policy_http_error(self, sample_video_path):
        resp = _FakeResp(status=400, json_data={"error": {"message": "model not found"}})
        with self._patch_session(resp):
            with pytest.raises(QwenUploadError, match="获取上传凭证失败（400）"):
                await upload_video_to_temp_url(
                    "sk-test",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "qwen-vl-max",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_policy_error_field(self, sample_video_path):
        resp = _FakeResp(json_data={"data": {"error": "no permission"}})
        with self._patch_session(resp):
            with pytest.raises(QwenUploadError, match="no permission"):
                await upload_video_to_temp_url(
                    "sk-test",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "qwen-vl-max",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_policy_missing_upload_host(self, sample_video_path):
        policy = json.loads(json.dumps(_POLICY))
        del policy["data"]["upload_host"]
        with self._patch_session(_FakeResp(json_data=policy)):
            with pytest.raises(QwenUploadError, match="upload_host"):
                await upload_video_to_temp_url(
                    "sk-test",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "qwen-vl-max",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_policy_missing_signature(self, sample_video_path):
        policy = json.loads(json.dumps(_POLICY))
        del policy["data"]["signature"]
        with self._patch_session(_FakeResp(json_data=policy)):
            with pytest.raises(QwenUploadError, match="signature"):
                await upload_video_to_temp_url(
                    "sk-test",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "qwen-vl-max",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_policy_invalid_json(self, sample_video_path):
        with self._patch_session(_FakeResp(status=200, text="not json {{{")):
            with pytest.raises(QwenUploadError, match="解析失败"):
                await upload_video_to_temp_url(
                    "sk-test",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "qwen-vl-max",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_policy_response_not_object(self, sample_video_path):
        """凭证响应为 JSON 数组等非对象结构：给出明确错误而非 AttributeError。"""
        with self._patch_session(_FakeResp(json_data=["not", "an", "object"])):
            with pytest.raises(QwenUploadError, match="格式异常"):
                await upload_video_to_temp_url(
                    "sk-test",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "qwen-vl-max",
                    str(sample_video_path),
                    "video/mp4",
                )

    async def test_upload_http_error(self, sample_video_path):
        post_resp = _FakeResp(
            status=500, json_data={"error": {"message": "oss internal error"}}
        )
        with self._patch_session(_FakeResp(json_data=_POLICY), post_resp):
            with pytest.raises(QwenUploadError, match="上传文件失败（500）"):
                await upload_video_to_temp_url(
                    "sk-test",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "qwen-vl-max",
                    str(sample_video_path),
                    "video/mp4",
                )

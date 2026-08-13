"""Tests for llm_tools.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemini_client import GeminiClient, GeminiClientError
from llm_tools import resolve_video_ref, run_video_analysis_from_path


@pytest.fixture
def mock_client():
    client = MagicMock(spec=GeminiClient)
    client.analyze = AsyncMock(return_value="这是一段精彩的视频。")
    client.close = AsyncMock()
    return client


class TestRunVideoAnalysisFromPath:
    async def test_success(self, sample_video_path, mock_client):
        result = await run_video_analysis_from_path(
            str(sample_video_path), 20, mock_client
        )
        assert "精彩的视频" in result
        mock_client.analyze.assert_called_once()

    async def test_file_not_found(self, mock_client):
        result = await run_video_analysis_from_path(
            "/nonexistent.mp4", 20, mock_client
        )
        assert "视频分析失败" in result
        mock_client.analyze.assert_not_called()

    async def test_api_error(self, sample_video_path, mock_client):
        mock_client.analyze.side_effect = GeminiClientError("API 超时")
        result = await run_video_analysis_from_path(
            str(sample_video_path), 20, mock_client
        )
        assert "API 超时" in result


class TestResolveVideoRef:
    async def test_local_file_exists(self, sample_video_path):
        item = {
            "name": "test.mp4",
            "ref": str(sample_video_path),
            "is_local": True,
            "result": None,
        }
        result = await resolve_video_ref(item, 20)
        assert result == str(sample_video_path)

    async def test_local_file_gone(self):
        item = {
            "name": "gone.mp4",
            "ref": "/nonexistent/gone.mp4",
            "is_local": True,
            "result": None,
        }
        from video_utils import VideoError
        with pytest.raises(VideoError, match="过期"):
            await resolve_video_ref(item, 20)

    async def test_remote_downloads(self, sample_video_path):
        item = {
            "name": "clip.mp4",
            "ref": "https://example.com/clip.mp4",
            "is_local": False,
            "result": None,
        }

        class MockResp:
            status = 200
            async def read(self):
                return sample_video_path.read_bytes()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        class MockSession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def get(self, url):
                return MockResp()

        with patch("video_utils.aiohttp.ClientSession", return_value=MockSession()):
            result = await resolve_video_ref(item, 20)
            assert Path(result).is_file()
            assert item["is_local"] is True
            assert item["ref"] == result

    async def test_empty_ref(self):
        item = {
            "name": "empty.mp4",
            "ref": "",
            "is_local": False,
            "result": None,
        }
        from video_utils import VideoError
        with pytest.raises(VideoError, match="为空"):
            await resolve_video_ref(item, 20)

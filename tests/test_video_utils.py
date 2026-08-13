"""Tests for video_utils.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from video_utils import (
    VideoError,
    _get_ext,
    _read_and_validate,
    download_video_file,
    load_video_from_path,
    resolve_component_ref,
)


class TestGetExt:
    def test_simple_extension(self):
        assert _get_ext("clip.mp4") == "mp4"

    def test_no_extension(self):
        assert _get_ext("clip") == ""

    def test_multiple_dots(self):
        assert _get_ext("clip.2024.mp4") == "mp4"

    def test_uppercase(self):
        assert _get_ext("CLIP.MP4") == "mp4"

    def test_path_with_dirs(self):
        assert _get_ext("/home/user/videos/clip.mov") == "mov"


class TestResolveComponentRef:
    def test_local_file_path(self):
        class MockComp:
            file_ = r"C:\Users\test\clip.mp4"
            url = ""

        local, url = resolve_component_ref(MockComp())
        assert local == r"C:\Users\test\clip.mp4"
        assert url == ""

    def test_file_uri(self):
        class MockComp:
            file_ = "file:///C:/Users/test/clip.mp4"
            url = ""

        local, url = resolve_component_ref(MockComp())
        assert local == "C:/Users/test/clip.mp4"

    def test_remote_url_only(self):
        class MockComp:
            file_ = ""
            url = "https://example.com/clip.mp4"

        local, url = resolve_component_ref(MockComp())
        assert local == ""
        assert url == "https://example.com/clip.mp4"

    def test_both_local_and_url(self):
        class MockComp:
            file_ = r"/tmp/clip.mp4"
            url = "https://example.com/backup.mp4"

        local, url = resolve_component_ref(MockComp())
        assert local == "/tmp/clip.mp4"
        assert url == "https://example.com/backup.mp4"

    def test_empty(self):
        class MockComp:
            file_ = ""
            url = ""

        local, url = resolve_component_ref(MockComp())
        assert local == ""
        assert url == ""

    def test_none_values(self):
        class MockComp:
            file_ = None
            url = None

        local, url = resolve_component_ref(MockComp())
        assert local == ""
        assert url == ""


class TestReadAndValidate:
    async def test_valid_file(self, sample_video_path):
        result = await _read_and_validate(
            str(sample_video_path), "mp4", 20, "test.mp4"
        )
        assert result.filename == "test.mp4"
        assert result.mime_type == "video/mp4"
        assert result.b64

    async def test_mov_mime(self, sample_mov_path):
        result = await _read_and_validate(
            str(sample_mov_path), "mov", 20, "clip.mov"
        )
        assert result.mime_type == "video/mov"

    async def test_unsupported_extension_uses_generic_mime(self, temp_dir):
        p = temp_dir / "test.xyz"
        p.write_bytes(b"\x00" * 100)
        result = await _read_and_validate(str(p), "xyz", 20, "test.xyz")
        assert result.mime_type == "video/xyz"

    async def test_file_too_large(self, sample_large_video_path):
        with pytest.raises(VideoError, match="超过限制"):
            await _read_and_validate(
                str(sample_large_video_path), "mp4", 1, "large.mp4"
            )

    async def test_file_not_exists(self):
        with pytest.raises(VideoError, match="不存在"):
            await _read_and_validate("/nonexistent/file.mp4", "mp4", 20, "file.mp4")


class TestLoadVideoFromPath:
    async def test_valid(self, sample_video_path):
        result = await load_video_from_path(str(sample_video_path), 20)
        assert result.b64

    async def test_file_not_found(self):
        with pytest.raises(VideoError, match="不存在"):
            await load_video_from_path("/nonexistent.mp4", 20)


class TestDownloadVideoFile:
    def _mock_session(self, payload: bytes, content_length: str | None = None):
        class MockResp:
            status = 200
            headers = {"Content-Length": content_length} if content_length else {}

            async def read(self):
                return payload

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

        return MockSession()

    async def test_download_success(self, temp_dir, sample_video_path):
        payload = sample_video_path.read_bytes()
        with patch("video_utils.aiohttp.ClientSession", return_value=self._mock_session(payload)):
            dest = await download_video_file("https://example.com/clip.mp4", "clip.mp4")
        assert dest.is_file()
        assert dest.read_bytes() == payload
        assert dest.name == "clip.mp4"

    async def test_sanitizes_unsafe_filename(self, sample_video_path):
        payload = sample_video_path.read_bytes()
        with patch("video_utils.aiohttp.ClientSession", return_value=self._mock_session(payload)):
            dest = await download_video_file(
                "https://example.com/clip.mp4", r"..\..\evil:name?.mp4"
            )
        assert dest.name == "evil_name_.mp4"

    async def test_content_length_too_large(self, sample_video_path):
        payload = sample_video_path.read_bytes()
        mock = self._mock_session(payload, content_length=str(50 * 1024 * 1024))
        with patch("video_utils.aiohttp.ClientSession", return_value=mock):
            with pytest.raises(VideoError, match="超过限制"):
                await download_video_file(
                    "https://example.com/big.mp4", "big.mp4", max_size_mb=20
                )

    async def test_actual_size_too_large(self, sample_large_video_path):
        payload = sample_large_video_path.read_bytes()
        with patch("video_utils.aiohttp.ClientSession", return_value=self._mock_session(payload)):
            with pytest.raises(VideoError, match="超过限制"):
                await download_video_file(
                    "https://example.com/big.mp4", "big.mp4", max_size_mb=1
                )

    async def test_http_error(self):
        class MockResp:
            status = 404
            headers = {}

            async def read(self):
                return b""

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
            with pytest.raises(VideoError, match="HTTP 404"):
                await download_video_file("https://example.com/missing.mp4", "missing.mp4")

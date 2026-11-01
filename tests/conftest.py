"""Shared test fixtures."""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_video_path(temp_dir):
    """Create a small fake MP4-like file for testing (extension is what matters)."""
    p = temp_dir / "test.mp4"
    # MP4 box header stub: ftyp box (not a real video, but has the right extension)
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
    return p


@pytest.fixture
def sample_large_video_path(temp_dir):
    """Create a video file larger than 1MB for size tests."""
    p = temp_dir / "large.mp4"
    p.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2MB
    return p


@pytest.fixture
def sample_mov_path(temp_dir):
    """Create a minimal MOV-like file (just a header stub)."""
    p = temp_dir / "clip.mov"
    p.write_bytes(b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 100)
    return p


@pytest.fixture
def sample_video_base64(sample_video_path):
    raw = sample_video_path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


@pytest.fixture
def response_factory():
    """构造支持 iter_chunked 的假 aiohttp 响应，供下载器测试使用。"""

    class FakeContent:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def iter_chunked(self, size: int):
            for start in range(0, len(self._payload), max(1, size)):
                yield self._payload[start : start + size]

    class FakeResponse:
        def __init__(self, payload: bytes, status: int, headers: dict) -> None:
            self.status = status
            self.headers = headers
            self.content = FakeContent(payload)
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

        def close(self) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    def _factory(payload: bytes = b"", status: int = 200, headers: dict | None = None):
        return FakeResponse(payload, status, headers or {})

    return _factory


@pytest.fixture
def session_factory(response_factory):
    """按 URL 返回不同假响应的 ClientSession 替身。"""

    class FakeSession:
        def __init__(self, responses: dict[str, object]) -> None:
            self._responses = responses

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url: str, **kwargs):
            response = self._responses.get(url)
            if response is None:
                return response_factory(b"", status=404)
            return response

    def _factory(responses: dict[str, object]) -> FakeSession:
        return FakeSession(responses)

    return _factory


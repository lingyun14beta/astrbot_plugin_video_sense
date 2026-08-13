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

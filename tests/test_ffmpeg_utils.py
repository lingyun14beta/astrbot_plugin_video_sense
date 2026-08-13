"""Tests for ffmpeg_utils.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ffmpeg_utils import (
    CompressResult,
    FfmpegError,
    _build_args,
    compress_video,
    find_ffmpeg,
    is_available,
    probe_duration,
)


class TestFindFfmpeg:
    def test_system_ffmpeg_preferred(self):
        with patch("ffmpeg_utils.shutil.which", return_value="/usr/bin/ffmpeg"):
            assert find_ffmpeg() == "/usr/bin/ffmpeg"

    def test_imageio_fallback(self):
        import sys
        import types

        stub = types.ModuleType("imageio_ffmpeg")
        stub.get_ffmpeg_exe = lambda: "/path/to/imageio/ffmpeg.exe"
        with (
            patch("ffmpeg_utils.shutil.which", return_value=None),
            patch("ffmpeg_utils._HAS_IMAGEIO_FFMPEG", True),
            patch.dict("sys.modules", {"imageio_ffmpeg": stub}),
        ):
            assert find_ffmpeg() == "/path/to/imageio/ffmpeg.exe"

    def test_imageio_not_installed(self):
        with (
            patch("ffmpeg_utils.shutil.which", return_value=None),
            patch("ffmpeg_utils._HAS_IMAGEIO_FFMPEG", False),
        ):
            assert find_ffmpeg() is None

    def test_imageio_raises(self):
        import sys
        import types

        stub = types.ModuleType("imageio_ffmpeg")
        stub.get_ffmpeg_exe = lambda: (_ for _ in ()).throw(RuntimeError("broken"))
        with (
            patch("ffmpeg_utils.shutil.which", return_value=None),
            patch("ffmpeg_utils._HAS_IMAGEIO_FFMPEG", True),
            patch.dict("sys.modules", {"imageio_ffmpeg": stub}),
        ):
            assert find_ffmpeg() is None

    def test_is_available(self):
        with patch("ffmpeg_utils.shutil.which", return_value="ffmpeg"):
            assert is_available() is True
        with patch("ffmpeg_utils.shutil.which", return_value=None):
            assert is_available() is False


class TestProbeDuration:
    def _mock_run(self, stderr: str):
        proc = type("Proc", (), {"stderr": stderr, "returncode": 1})()
        return patch("ffmpeg_utils.subprocess.run", return_value=proc)

    async def test_parses_duration(self):
        stderr = "  Duration: 00:01:23.45, start: 0.000000, bitrate: 1234 kb/s\n"
        with self._mock_run(stderr):
            assert await probe_duration("/usr/bin/ffmpeg", "a.mp4") == 83.45

    async def test_no_duration(self):
        with self._mock_run("no duration here"):
            assert await probe_duration("/usr/bin/ffmpeg", "a.mp4") is None

    async def test_timeout(self):
        import subprocess

        with patch(
            "ffmpeg_utils.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 30),
        ):
            assert await probe_duration("/usr/bin/ffmpeg", "a.mp4") is None

    async def test_oserror(self):
        with patch("ffmpeg_utils.subprocess.run", side_effect=OSError("no binary")):
            assert await probe_duration("/usr/bin/ffmpeg", "a.mp4") is None


class TestBuildArgs:
    def test_target_bitrate_mode(self):
        args = _build_args("in.mp4", "out.mp4", 720, 28, duration=60, max_size_mb=15)
        # 预算 = 15MB × 8 × 1024 × 0.9 / 60s = 1843 kbps
        assert "-b:v" in args
        assert "1843k" in args
        assert "scale=-2:720,fps=15" in args
        assert "-crf" not in args

    def test_crf_fallback_when_duration_unknown(self):
        args = _build_args("in.mp4", "out.mp4", 720, 28, duration=None, max_size_mb=15)
        assert "-crf" in args
        assert "28" in args
        assert "-b:v" not in args

    def test_bitrate_clamped(self):
        args = _build_args("in.mp4", "out.mp4", 720, 28, duration=0.1, max_size_mb=15)
        assert "8000k" in args  # 上限 8000


class TestCompressVideo:
    async def _run_ffmpeg_writer(self, src, dest, **kwargs):
        """模拟 ffmpeg：往 dest 写文件（大小可控）。"""

        def make_writer(size_bytes):
            async def _write(ffmpeg_path, args, timeout=300):
                assert "-i" in args
                out = args[-1]
                Path(out).write_bytes(b"\x00" * size_bytes)

            return _write

        return make_writer

    async def test_success_first_attempt(self, temp_dir):
        src = temp_dir / "src.mp4"
        src.write_bytes(b"\x00" * 1024)
        with (
            patch("ffmpeg_utils.probe_duration", new=AsyncMock(return_value=60.0)),
            patch(
                "ffmpeg_utils._run_ffmpeg",
                new=AsyncMock(
                    side_effect=lambda ffmpeg, args, timeout=300: Path(
                        args[-1]
                    ).write_bytes(b"\x00" * 1024)
                ),
            ),
        ):
            result = await compress_video(
                "/usr/bin/ffmpeg", str(src), temp_dir, max_size_mb=15, max_duration_s=None
            )
        assert isinstance(result, CompressResult)
        assert result.size_bytes == 1024
        assert result.attempts == 1

    async def test_degrades_to_480p(self, temp_dir):
        """第一次压缩仍超限 → 降级 480p 第二次压缩达标。"""
        src = temp_dir / "src.mp4"
        src.write_bytes(b"\x00" * 1024)
        calls = {"n": 0}

        async def fake_run(ffmpeg_path, args, timeout=300):
            calls["n"] += 1
            if calls["n"] == 1:
                Path(args[-1]).write_bytes(b"\x00" * (16 * 1024 * 1024))
            else:
                Path(args[-1]).write_bytes(b"\x00" * 1024)

        with (
            patch("ffmpeg_utils.probe_duration", new=AsyncMock(return_value=60.0)),
            patch("ffmpeg_utils._run_ffmpeg", new=AsyncMock(side_effect=fake_run)),
        ):
            result = await compress_video(
                "/usr/bin/ffmpeg", str(src), temp_dir, max_size_mb=15
            )
        assert result.attempts == 2

    async def test_trim_long_video_first(self, temp_dir):
        """超过 max_duration_s → 先截取前 N 秒（-c copy）。"""
        src = temp_dir / "src.mp4"
        src.write_bytes(b"\x00" * 1024)
        calls = []

        async def fake_run(ffmpeg_path, args, timeout=300):
            calls.append(args)
            Path(args[-1]).write_bytes(b"\x00" * 1024)

        with (
            patch("ffmpeg_utils.probe_duration", new=AsyncMock(return_value=300.0)),
            patch("ffmpeg_utils._run_ffmpeg", new=AsyncMock(side_effect=fake_run)),
        ):
            await compress_video(
                "/usr/bin/ffmpeg", str(src), temp_dir, max_size_mb=15, max_duration_s=120
            )
        # 第一次调用是截取（-c copy -t 120）
        trim_args = calls[0]
        assert "-c" in trim_args and "copy" in trim_args
        assert "-t" in trim_args and "120" in trim_args

    async def test_still_oversize_raises(self, temp_dir):
        src = temp_dir / "src.mp4"
        src.write_bytes(b"\x00" * 1024)
        with (
            patch("ffmpeg_utils.probe_duration", new=AsyncMock(return_value=60.0)),
            patch(
                "ffmpeg_utils._run_ffmpeg",
                new=AsyncMock(
                    side_effect=lambda ffmpeg, args, timeout=300: Path(
                        args[-1]
                    ).write_bytes(b"\x00" * (20 * 1024 * 1024))
                ),
            ),
        ):
            with pytest.raises(FfmpegError, match="压缩后仍"):
                await compress_video(
                    "/usr/bin/ffmpeg", str(src), temp_dir, max_size_mb=15
                )

    async def test_missing_ffmpeg(self, temp_dir):
        src = temp_dir / "src.mp4"
        src.write_bytes(b"\x00" * 1024)
        with pytest.raises(FfmpegError, match="未检测到 ffmpeg"):
            await compress_video(None, str(src), temp_dir, max_size_mb=15)

    async def test_missing_source(self, temp_dir):
        with pytest.raises(FfmpegError, match="不存在"):
            await compress_video("/usr/bin/ffmpeg", "/nonexistent.mp4", temp_dir, 15)

    async def test_ffmpeg_failure_propagates(self, temp_dir):
        src = temp_dir / "src.mp4"
        src.write_bytes(b"\x00" * 1024)

        async def fake_run(ffmpeg_path, args, timeout=300):
            raise FfmpegError("ffmpeg 压缩失败：invalid data")

        with (
            patch("ffmpeg_utils.probe_duration", new=AsyncMock(return_value=60.0)),
            patch("ffmpeg_utils._run_ffmpeg", new=AsyncMock(side_effect=fake_run)),
        ):
            with pytest.raises(FfmpegError, match="invalid data"):
                await compress_video(
                    "/usr/bin/ffmpeg", str(src), temp_dir, max_size_mb=15
                )

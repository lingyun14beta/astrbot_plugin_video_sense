"""Tests for video_utils.py."""

from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock, patch

import pytest

from video_utils import (
    VideoError,
    VideoExpiredError,
    _assert_public_url,
    _get_ext,
    _open_stream,
    _prune_temp_files,
    _validate,
    download_video_file,
    load_video,
    load_video_from_path,
    resolve_component_ref,
    video_component_name,
    video_ref_status,
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


class TestValidate:
    async def test_valid_file(self, sample_video_path):
        result = await _validate(str(sample_video_path), "mp4", 100, "test.mp4")
        assert result.path == str(sample_video_path)
        assert result.filename == "test.mp4"
        assert result.mime_type == "video/mp4"
        assert result.size_bytes == sample_video_path.stat().st_size

    async def test_mov_mime(self, sample_mov_path):
        result = await _validate(str(sample_mov_path), "mov", 100, "clip.mov")
        assert result.mime_type == "video/mov"

    async def test_unsupported_extension_uses_generic_mime(self, temp_dir):
        p = temp_dir / "test.xyz"
        p.write_bytes(b"\x00" * 100)
        result = await _validate(str(p), "xyz", 100, "test.xyz")
        assert result.mime_type == "video/xyz"

    async def test_file_too_large(self, sample_large_video_path):
        with pytest.raises(VideoError, match="超过限制"):
            await _validate(str(sample_large_video_path), "mp4", 1, "large.mp4")

    async def test_file_not_exists(self):
        with pytest.raises(VideoError, match="不存在"):
            await _validate("/nonexistent/file.mp4", "mp4", 100, "file.mp4")


class TestLoadVideoFromPath:
    async def test_valid(self, sample_video_path):
        result = await load_video_from_path(str(sample_video_path), 100)
        assert result.path == str(sample_video_path)
        assert result.mime_type == "video/mp4"

    async def test_file_not_found(self):
        with pytest.raises(VideoError, match="不存在"):
            await load_video_from_path("/nonexistent.mp4", 100)


class TestVideoComponent:
    """Video 组件（QQ/Telegram 等平台视频消息段）支持。"""

    @staticmethod
    def _make_video(**fields):
        """动态构造 __name__ == "Video" 的组件替身（代码按组件名判断类型）。"""
        defaults = {"file": "", "url": "", "path": "", "name": ""}
        defaults.update(fields)
        return type("Video", (), defaults)()

    def test_resolve_video_with_url_only(self):
        comp = self._make_video(
            file="562322f7f448ca042fa7e6144f04d584.mp4",
            url="https://multimedia.nt.qq.com.cn/download?rkey=abc",
        )
        local, url = resolve_component_ref(comp)
        assert url == "https://multimedia.nt.qq.com.cn/download?rkey=abc"
        assert "562322f7f448ca042fa7e6144f04d584.mp4" in local  # 文件名兜底为本地引用

    def test_resolve_video_with_path(self, temp_dir):
        p = temp_dir / "clip.mp4"
        p.write_bytes(b"\x00" * 10)
        comp = self._make_video(file=str(p), path=str(p))
        local, url = resolve_component_ref(comp)
        assert local == str(p)
        assert url == ""

    def test_resolve_video_file_uri(self):
        comp = self._make_video(file="file:///C:/Users/test/clip.mp4")
        local, url = resolve_component_ref(comp)
        assert local == "C:/Users/test/clip.mp4"

    def test_video_name_from_url(self):
        comp = self._make_video(
            file="562322f7.mp4", url="https://example.com/clip.mp4?sign=1"
        )
        assert video_component_name(comp) == "562322f7.mp4"

    def test_video_name_query_string_stripped(self):
        comp = self._make_video(url="https://example.com/movie.mp4?rkey=abc123")
        assert video_component_name(comp) == "movie.mp4"

    def test_video_name_fallback(self):
        assert video_component_name(self._make_video()) == "video.mp4"

    async def test_load_video_video_component(self, temp_dir):
        """Video 组件走 convert_to_file_path。"""
        p = temp_dir / "clip.mp4"
        p.write_bytes(b"\x00" * 1024)
        comp = self._make_video()
        comp.convert_to_file_path = AsyncMock(return_value=str(p))

        result = await load_video(comp, ["mp4"], 100)
        assert result.path == str(p)
        assert result.filename == "video.mp4"  # 无名称时兜底

    async def test_load_video_video_component_no_ext(self, temp_dir):
        """Video 组件无扩展名时按 mp4 处理。"""
        p = temp_dir / "clip"
        p.write_bytes(b"\x00" * 1024)
        comp = self._make_video()
        comp.convert_to_file_path = AsyncMock(return_value=str(p))

        result = await load_video(comp, ["mp4"], 100)
        assert result.mime_type == "video/mp4"


class _PatchDownloadEnv:
    """下载器测试前提：跳过真实 DNS 解析（地址校验另有独立用例）。"""

    @pytest.fixture(autouse=True)
    def _skip_url_check(self):
        with patch("video_utils._assert_public_url", new=AsyncMock(return_value=None)):
            yield


class TestDownloadVideoFile(_PatchDownloadEnv):
    async def test_download_success(
        self, sample_video_path, response_factory, session_factory
    ):
        payload = sample_video_path.read_bytes()
        session = session_factory(
            {"https://example.com/clip.mp4": response_factory(payload)}
        )
        with patch("video_utils.aiohttp.ClientSession", return_value=session):
            dest = await download_video_file("https://example.com/clip.mp4", "clip.mp4")
        assert dest.is_file()
        assert dest.read_bytes() == payload
        assert dest.name.endswith("clip.mp4")

    async def test_filename_has_unique_prefix(
        self, sample_video_path, response_factory, session_factory
    ):
        """同名视频重复下载不再互相覆盖（此前会写到同一个固定路径）。"""
        payload = sample_video_path.read_bytes()
        session = session_factory(
            {"https://example.com/clip.mp4": response_factory(payload)}
        )
        with patch("video_utils.aiohttp.ClientSession", return_value=session):
            first = await download_video_file(
                "https://example.com/clip.mp4", "clip.mp4"
            )
            second = await download_video_file(
                "https://example.com/clip.mp4", "clip.mp4"
            )
        assert first != second
        assert first.is_file() and second.is_file()

    async def test_sanitizes_unsafe_filename(
        self, sample_video_path, response_factory, session_factory
    ):
        """文件名净化：路径分隔符与非法字符必须被剥离/替换（平台无关）。"""
        payload = sample_video_path.read_bytes()
        session = session_factory(
            {"https://example.com/clip.mp4": response_factory(payload)}
        )

        # 正斜杠路径：Path.name 在任意平台都剥离目录部分，仅净化非法字符
        with patch("video_utils.aiohttp.ClientSession", return_value=session):
            dest = await download_video_file(
                "https://example.com/clip.mp4", "../evil/name?.mp4"
            )
        assert dest.name.endswith("name_.mp4")

        # 反斜杠路径（Windows 风格）：在 Linux 上反斜杠不是分隔符，
        # 但净化逻辑必须保证不含路径分隔符（无路径穿越风险）
        with patch("video_utils.aiohttp.ClientSession", return_value=session):
            dest2 = await download_video_file(
                "https://example.com/clip.mp4", r"..\..\evil:name?.mp4"
            )
        assert "/" not in dest2.name
        assert "\\" not in dest2.name
        assert ":" not in dest2.name
        assert dest2.name.endswith(".mp4")

    async def test_content_length_too_large(
        self, sample_video_path, response_factory, session_factory
    ):
        payload = sample_video_path.read_bytes()
        session = session_factory(
            {
                "https://example.com/big.mp4": response_factory(
                    payload, headers={"Content-Length": str(50 * 1024 * 1024)}
                )
            }
        )
        with patch("video_utils.aiohttp.ClientSession", return_value=session):
            with pytest.raises(VideoError, match="超过限制"):
                await download_video_file(
                    "https://example.com/big.mp4", "big.mp4", max_size_mb=20
                )

    async def test_streaming_cap_without_content_length(
        self, sample_large_video_path, response_factory, session_factory
    ):
        """没有 Content-Length 时也要在流式读取阶段按大小中断。"""
        payload = sample_large_video_path.read_bytes()
        session = session_factory(
            {"https://example.com/big.mp4": response_factory(payload)}
        )
        with patch("video_utils.aiohttp.ClientSession", return_value=session):
            with pytest.raises(VideoError, match="超过限制"):
                await download_video_file(
                    "https://example.com/big.mp4", "big.mp4", max_size_mb=1
                )

    async def test_http_error(self, response_factory, session_factory):
        session = session_factory(
            {"https://example.com/missing.mp4": response_factory(b"", status=404)}
        )
        with patch("video_utils.aiohttp.ClientSession", return_value=session):
            with pytest.raises(VideoError, match="HTTP 404"):
                await download_video_file(
                    "https://example.com/missing.mp4", "missing.mp4"
                )

    async def test_follows_redirect(
        self, sample_video_path, response_factory, session_factory
    ):
        payload = sample_video_path.read_bytes()
        session = session_factory(
            {
                "https://example.com/clip.mp4": response_factory(
                    b"", status=302, headers={"Location": "/real.mp4"}
                ),
                "https://example.com/real.mp4": response_factory(payload),
            }
        )
        with patch("video_utils.aiohttp.ClientSession", return_value=session):
            dest = await download_video_file(
                "https://example.com/clip.mp4", "clip.mp4"
            )
        assert dest.read_bytes() == payload

    async def test_too_many_redirects(self, response_factory, session_factory):
        session = session_factory(
            {
                "https://example.com/a.mp4": response_factory(
                    b"", status=302, headers={"Location": "/b.mp4"}
                ),
                "https://example.com/b.mp4": response_factory(
                    b"", status=302, headers={"Location": "/c.mp4"}
                ),
                "https://example.com/c.mp4": response_factory(
                    b"", status=302, headers={"Location": "/d.mp4"}
                ),
                "https://example.com/d.mp4": response_factory(
                    b"", status=302, headers={"Location": "/e.mp4"}
                ),
            }
        )
        with patch("video_utils.aiohttp.ClientSession", return_value=session):
            with pytest.raises(VideoError, match="重定向次数过多"):
                await download_video_file("https://example.com/a.mp4", "a.mp4")


class TestAssertPublicUrl:
    async def test_allows_public_ip(self):
        await _assert_public_url("https://93.184.216.34/clip.mp4")

    async def test_rejects_non_http_scheme(self):
        with pytest.raises(VideoError, match="不支持的下载协议"):
            await _assert_public_url("file:///etc/passwd")

    async def test_rejects_localhost(self):
        with pytest.raises(VideoError, match="本机或内网"):
            await _assert_public_url("http://localhost:8000/clip.mp4")

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/clip.mp4",
            "http://10.1.2.3/clip.mp4",
            "http://192.168.1.10/clip.mp4",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/clip.mp4",
        ],
    )
    async def test_rejects_private_ip_literal(self, url):
        with pytest.raises(VideoError, match="内网或保留地址"):
            await _assert_public_url(url)

    async def test_rejects_hostname_resolving_to_private_ip(self):
        with patch(
            "socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("192.168.1.10", 0))],
        ):
            with pytest.raises(VideoError, match="解析到内网"):
                await _assert_public_url("https://internal.example.com/clip.mp4")

    async def test_rejects_missing_host(self):
        with pytest.raises(VideoError, match="缺少主机名"):
            await _assert_public_url("https:///clip.mp4")


class TestOpenStreamRejectsPrivateRedirect:
    async def test_redirect_to_private_ip_is_rejected(
        self, response_factory, session_factory
    ):
        session = session_factory(
            {
                "https://93.184.216.34/a.mp4": response_factory(
                    b"", status=302, headers={"Location": "http://127.0.0.1/b.mp4"}
                )
            }
        )
        with pytest.raises(VideoError, match="内网或保留地址"):
            await _open_stream(session, "https://93.184.216.34/a.mp4")


class TestPruneTempFiles:
    def test_removes_old_files_keeps_new(self, temp_dir):
        old = temp_dir / "old.mp4"
        new = temp_dir / "new.mp4"
        old.write_bytes(b"x")
        new.write_bytes(b"x")
        past = time.time() - 48 * 3600
        os.utime(old, (past, past))

        _prune_temp_files(temp_dir)

        assert not old.exists()
        assert new.exists()


class TestVideoRefStatus:
    def test_empty_ref_is_expired(self):
        usable, reason = video_ref_status({"ref": "", "is_local": False})
        assert usable is False
        assert "回收" in reason

    def test_missing_local_file_is_expired(self, temp_dir):
        usable, reason = video_ref_status(
            {"ref": str(temp_dir / "gone.mp4"), "is_local": True}
        )
        assert usable is False
        assert "清理" in reason

    def test_existing_local_file_is_usable(self, sample_video_path):
        usable, reason = video_ref_status(
            {"ref": str(sample_video_path), "is_local": True}
        )
        assert usable is True
        assert reason == ""

    def test_remote_ref_is_not_probed(self):
        """远程链接不做网络探测，避免 list 时发请求；真正过期在下载时报。"""
        usable, reason = video_ref_status(
            {"ref": "https://example.com/a.mp4", "is_local": False}
        )
        assert usable is True
        assert reason == ""


class TestDownloadExpiredLink:
    async def test_403_raises_expired_error(self, response_factory, session_factory):
        session = session_factory(
            {"https://example.com/x.mp4": response_factory(b"", status=403)}
        )
        with (
            patch("video_utils.aiohttp.ClientSession", return_value=session),
            patch("video_utils._assert_public_url", new=AsyncMock(return_value=None)),
        ):
            with pytest.raises(VideoExpiredError, match="已过期"):
                await download_video_file("https://example.com/x.mp4", "x.mp4")

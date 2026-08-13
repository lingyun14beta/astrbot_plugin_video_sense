"""ffmpeg 视频压缩工具（可选依赖）。

探测顺序：系统 PATH 中的 ffmpeg → imageio-ffmpeg 内置二进制。
未检测到 ffmpeg 时，压缩功能自动禁用（is_available() 返回 False），
不会影响插件其他功能。
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

# 仅探测 imageio-ffmpeg 是否可用（运行时再动态导入，便于测试注入）
_HAS_IMAGEIO_FFMPEG = importlib.util.find_spec("imageio_ffmpeg") is not None

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")


class FfmpegError(Exception):
    """ffmpeg 调用失败，message 可直接透传给用户。"""


@dataclass
class CompressResult:
    path: str
    size_bytes: int
    attempts: int


def find_ffmpeg() -> str | None:
    """探测可用的 ffmpeg 可执行文件路径，找不到返回 None。"""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    if not _HAS_IMAGEIO_FFMPEG:
        return None
    try:
        import imageio_ffmpeg  # 运行时解析，便于测试注入

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def is_available() -> bool:
    """ffmpeg 是否可用。"""
    return find_ffmpeg() is not None


async def probe_duration(ffmpeg_path: str, file_path: str) -> float | None:
    """通过 ffmpeg -i 输出探测视频时长（秒），失败返回 None。"""

    def _run() -> float | None:
        try:
            proc = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-i", file_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        match = _DURATION_RE.search(proc.stderr or "")
        if not match:
            return None
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    return await asyncio.to_thread(_run)


async def compress_video(
    ffmpeg_path: str,
    src: str,
    dest_dir: Path,
    max_size_mb: int,
    max_duration_s: int | None = None,
    resolution: int = 720,
    crf: int = 28,
    timeout: int = 300,
) -> CompressResult:
    """将视频压缩到 max_size_mb 以内（目标码率 + 阶梯降级）。

    流程：
    1. 若 max_duration_s 且视频更长，先无损截取前 max_duration_s 秒。
    2. 按目标码率（预算 × 0.9 / 时长）以 resolution 分辨率压缩。
    3. 未达标则降级 480p + CRF 32 再压一次。
    4. 仍超限则抛 FfmpegError。

    Args:
        ffmpeg_path: ffmpeg 可执行文件路径（find_ffmpeg 返回值）。
        src: 源视频路径。
        dest_dir: 压缩产物临时目录。
        max_size_mb: 压缩目标大小上限（MB）。
        max_duration_s: 超过此时长先截取前 N 秒；None 不截取。
        resolution: 首选压缩分辨率（高度）。
        crf: 首选 CRF 质量参数（18-32，越大越省空间）。
        timeout: 单次 ffmpeg 调用超时（秒）。

    Returns:
        压缩结果（路径与最终大小）。

    Raises:
        FfmpegError: ffmpeg 不可用、调用失败或压缩后仍超限。
    """
    if not ffmpeg_path:
        raise FfmpegError("未检测到 ffmpeg，无法压缩视频。")

    src_path = Path(src)
    if not src_path.is_file():
        raise FfmpegError(f"文件不存在或无法访问：{src}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    work_path = dest_dir / f"compress_{tag}.mp4"

    # 1) 可选截取前 N 秒（长视频兜底）
    if max_duration_s and max_duration_s > 0:
        duration = await probe_duration(ffmpeg_path, str(src_path))
        if duration is not None and duration > max_duration_s:
            trimmed = dest_dir / f"trim_{tag}.mp4"
            await _run_ffmpeg(
                ffmpeg_path,
                [
                    "-i",
                    str(src_path),
                    "-t",
                    str(max_duration_s),
                    "-c",
                    "copy",
                    str(trimmed),
                ],
                timeout=timeout,
            )
            src_path = trimmed

    # 2) 目标码率阶梯压缩
    max_bytes = max_size_mb * 1024 * 1024
    duration = await probe_duration(ffmpeg_path, str(src_path))
    attempts = 0
    for res, quality in ((resolution, crf), (480, 32)):
        attempts += 1
        await _run_ffmpeg(
            ffmpeg_path,
            _build_args(
                str(src_path), str(work_path), res, quality, duration, max_size_mb
            ),
            timeout=timeout,
        )
        size = work_path.stat().st_size
        if size <= max_bytes:
            return CompressResult(
                path=str(work_path), size_bytes=size, attempts=attempts
            )

    size_mb = work_path.stat().st_size / 1024 / 1024
    raise FfmpegError(
        f"压缩后仍为 {size_mb:.1f} MB，超过限制 {max_size_mb} MB。"
        "视频可能过长，请手动截取片段后重试。",
    )


def _build_args(
    src: str,
    dest: str,
    resolution: int,
    crf: int,
    duration: float | None,
    max_size_mb: int,
) -> list[str]:
    """构造压缩参数：目标码率模式，时长未知时回退 CRF 模式。"""
    args = [
        "-i",
        src,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-vf",
        f"scale=-2:{resolution},fps=15",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
    ]
    if duration and duration > 0:
        # 预算 = 上限 × 0.9（留出容器/音频余量），换算为 kbps
        target_kbps = int(max_size_mb * 1024 * 8 * 0.9 / duration)
        target_kbps = max(100, min(target_kbps, 8000))
        args += ["-b:v", f"{target_kbps}k", "-maxrate", f"{int(target_kbps * 1.1)}k"]
    else:
        args += ["-crf", str(crf)]
    args += ["-movflags", "+faststart", dest]
    return args


async def _run_ffmpeg(ffmpeg_path: str, args: list[str], timeout: int = 300) -> None:
    """同步执行 ffmpeg（在线程池中），失败抛 FfmpegError。"""

    def _run() -> None:
        try:
            proc = subprocess.run(
                [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error", *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise FfmpegError(f"ffmpeg 执行超时（>{timeout} 秒）。") from e
        except OSError as e:
            raise FfmpegError(f"ffmpeg 执行失败：{e}") from e
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip()[:500]
            raise FfmpegError(
                f"ffmpeg 压缩失败：{detail or f'退出码 {proc.returncode}'}"
            )

    await asyncio.to_thread(_run)

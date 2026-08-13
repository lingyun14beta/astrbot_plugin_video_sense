"""LLM 工具业务逻辑。"""

from __future__ import annotations

try:
    from .gemini_client import GeminiClient, GeminiClientError
    from .video_utils import (
        VideoError,
        download_video_file,
        extract_file_component,
        load_video,
        load_video_from_path,
    )
except ImportError:
    from gemini_client import GeminiClient, GeminiClientError
    from video_utils import (
        VideoError,
        download_video_file,
        extract_file_component,
        load_video,
        load_video_from_path,
    )


async def run_video_analysis(
    event,
    supported_formats: list[str],
    max_size_mb: int,
    client: GeminiClient,
) -> str:
    """从当前 event 的 File 组件执行视频分析（自动选择内嵌/Files API 传输）。"""
    file_comp = extract_file_component(event)
    if file_comp is None:
        return "当前消息中没有找到视频文件，无法进行分析。"

    try:
        video = await load_video(file_comp, supported_formats, max_size_mb)
        result = await client.analyze_video(video)
    except (VideoError, GeminiClientError) as e:
        return f"视频分析失败：{e}"
    else:
        return result


async def run_video_analysis_from_path(
    file_path: str,
    max_size_mb: int,
    client: GeminiClient,
) -> str:
    """从本地文件路径执行视频分析（自动选择内嵌/Files API 传输）。"""
    try:
        video = await load_video_from_path(file_path, max_size_mb)
        result = await client.analyze_video(video)
    except (VideoError, GeminiClientError) as e:
        return f"视频分析失败：{e}"
    else:
        return result


async def resolve_video_ref(
    item: dict,
    max_size_mb: int,
) -> str:
    """将缓存的视频引用解析为可用路径。local 直接用，remote 按需下载。"""
    ref = item["ref"]
    if not ref:
        raise VideoError("文件引用为空。")

    if item["is_local"]:
        from pathlib import Path

        if Path(ref).is_file():
            return ref
        raise VideoError(f"文件已过期或不可访问：{ref}")

    # 远程 URL，按需下载（下载前按 max_size_mb 预检大小）
    name = item.get("name", "video")
    dest = await download_video_file(ref, name, max_size_mb=max_size_mb)
    item["ref"] = str(dest)
    item["is_local"] = True
    return str(dest)

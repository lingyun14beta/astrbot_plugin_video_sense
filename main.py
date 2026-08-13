"""astrbot_plugin_video_sense — 让 Bot 看懂视频。

通过 Gemini 原生多模态视频理解，分析群里分享的视频文件，
将结果注入对话历史，让 Bot 能就视频展开自然对话。

  触发方式：
  - /分析视频 [序号] [追问] + 视频文件 或 引用视频消息
  - LLM 工具 list_video_files / analyze_video_by_number
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from astrbot.api import AstrBotConfig, llm_tool, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import filter as astr_filter
from astrbot.api.event.filter import CustomFilter
from astrbot.api.star import Context, Star

from .gemini_client import GeminiClient, GeminiClientError
from .llm_tools import (
    resolve_video_ref,
    run_video_analysis,
    run_video_analysis_from_path,
)
from .video_utils import (
    VideoError,
    extract_file_component,
    load_video,
    resolve_component_ref,
)

_DEFAULT_SYSTEM_PROMPT = (
    "像一个朋友随便聊聊这段视频，突出你最想说的。不要列表。60字以内。"
)

_ERROR_PREFIXES = ("文件处理失败", "视频分析失败")


def _is_error(text: str) -> bool:
    return text.startswith(_ERROR_PREFIXES)


class _FileComponentFilter(CustomFilter):
    """匹配含有 File 组件的消息（含引用消息），用于缓存视频元数据。"""

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        for comp in getattr(event.message_obj, "message", []):
            if type(comp).__name__ == "File":
                return True
            if type(comp).__name__ == "Reply":
                for rc in getattr(comp, "chain", []) or []:
                    if type(rc).__name__ == "File":
                        return True
        return False


class VideoSensePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None) -> None:
        super().__init__(context)
        self.config: AstrBotConfig = config or {}
        self._registry: dict[str, list[dict]] = {}
        self._pending_injections: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()
        self._auto_sem = asyncio.Semaphore(2)  # 最多 2 个并发自动分析（视频请求更重）
        logger.info("[VideoSense] 插件已加载，支持格式：%s", self._supported_formats)

    # ------------------------------------------------------------------
    # 配置快捷属性
    # ------------------------------------------------------------------

    @property
    def _analysis_cfg(self) -> dict:
        return self.config.get("analysis", {})

    @property
    def _supported_formats(self) -> list[str]:
        return self._analysis_cfg.get(
            "supported_formats",
            ["mp4", "mov", "webm", "avi", "mpeg", "mpg", "flv", "wmv", "3gpp"],
        )

    @property
    def _max_size_mb(self) -> int:
        return int(self._analysis_cfg.get("max_file_size_mb", 100))

    @property
    def _max_inline_mb(self) -> int:
        return int(self._analysis_cfg.get("max_inline_size_mb", 15))

    @property
    def _use_files_api(self) -> bool:
        return bool(self._analysis_cfg.get("use_files_api", True))

    @property
    def _max_cached_files(self) -> int:
        return max(1, int(self._analysis_cfg.get("max_cached_files", 50)))

    @property
    def _system_prompt(self) -> str:
        return self._analysis_cfg.get("system_prompt", _DEFAULT_SYSTEM_PROMPT).strip()

    @property
    def _auto_analyze(self) -> bool:
        return bool(self._analysis_cfg.get("auto_analyze", False))

    @property
    def _inject_context(self) -> bool:
        return bool(self._analysis_cfg.get("inject_context", False))

    @property
    def _separate_prompts(self) -> bool:
        return bool(self._analysis_cfg.get("separate_prompts", False))

    @property
    def _debug(self) -> bool:
        return bool(self._analysis_cfg.get("debug", False))

    @property
    def _command_system_prompt(self) -> str:
        return self._analysis_cfg.get(
            "command_system_prompt", _DEFAULT_SYSTEM_PROMPT
        ).strip()

    def _resolve_provider_and_model(self) -> tuple[dict, str]:
        providers: list = self.config.get("api_provider", [])
        if not isinstance(providers, list):
            providers = []

        fallback_provider = providers[0] if providers else {}
        fallback_model = fallback_provider.get("model", "gemini-2.0-flash")

        model_cfg: str = self._analysis_cfg.get("model", "").strip()
        if not model_cfg or "/" not in model_cfg:
            return fallback_provider, fallback_model

        provider_name, model_name = model_cfg.split("/", 1)
        provider_name = provider_name.strip()
        model_name = model_name.strip()

        if not provider_name or not model_name:
            return fallback_provider, fallback_model

        for p in providers:
            if isinstance(p, dict) and p.get("name", "").strip() == provider_name:
                return p, model_name or p.get("model", "gemini-2.0-flash")

        return fallback_provider, fallback_model

    def _make_client(
        self, extra_prompt: str = "", use_command_prompt: bool = False
    ) -> GeminiClient:
        provider, model = self._resolve_provider_and_model()
        if self._separate_prompts and use_command_prompt:
            sp = self._command_system_prompt
        else:
            sp = self._system_prompt
        if extra_prompt:
            sp = f"{sp}\n\n用户的追加问题：{extra_prompt}"
        return GeminiClient(
            api_key=provider.get("api_key", ""),
            model=model,
            system_prompt=sp,
            base_url=provider.get("base_url", ""),
            timeout=int(provider.get("timeout", 300)),
            max_inline_size_mb=self._max_inline_mb,
            use_files_api=self._use_files_api,
        )

    # ------------------------------------------------------------------
    # 视频文件缓存钩子
    # ------------------------------------------------------------------

    @astr_filter.custom_filter(_FileComponentFilter)
    async def _on_file_message(self, event: AstrMessageEvent):
        """消息中包含 File 组件时，缓存其元数据。不下载，不下发消息。"""
        umo = event.unified_msg_origin

        def _cache(comp, items):
            name = getattr(comp, "name", "") or ""
            ext = Path(name).suffix.lstrip(".").lower()
            if ext not in self._supported_formats:
                if self._debug:
                    logger.info(
                        "[VideoSense] 跳过非视频文件：%s (扩展名 .%s)",
                        name or "(无名称)",
                        ext,
                    )
                return
            local, url = resolve_component_ref(comp)
            if local and Path(local).is_file():
                items.append(
                    {"name": name, "ref": local, "is_local": True, "result": None}
                )
            else:
                if self._debug and url:
                    logger.info("[VideoSense] 远程文件暂不自动分析：%s", name)
                items.append(
                    {
                        "name": name,
                        "ref": url or local,
                        "is_local": False,
                        "result": None,
                    }
                )

        new_items = []
        async with self._lock:
            items = self._registry.setdefault(umo, [])
            for comp in getattr(event.message_obj, "message", []):
                if type(comp).__name__ == "File":
                    before = len(items)
                    _cache(comp, items)
                    if len(items) > before:
                        new_items.append(items[-1])
                elif type(comp).__name__ == "Reply":
                    for rc in getattr(comp, "chain", []) or []:
                        if type(rc).__name__ == "File":
                            before = len(items)
                            _cache(rc, items)
                            if len(items) > before:
                                new_items.append(items[-1])

        if self._auto_analyze:
            triggered = 0
            for item in new_items:
                if item["is_local"]:
                    if self._debug:
                        logger.info("[VideoSense] 触发自动分析：%s", item.get("name"))
                    asyncio.create_task(self._auto_analyze_task(umo, item))
                    triggered += 1
            if self._debug and new_items and not triggered:
                logger.info("[VideoSense] 所有新文件均为远程，跳过自动分析")

        if self._debug and new_items:
            logger.info("[VideoSense] 缓存完成：%d 个视频文件", len(new_items))

        # 缓存上限裁剪：保留最近的 N 个，避免长会话无限膨胀
        max_cached = self._max_cached_files
        if len(items) > max_cached:
            overflow = len(items) - max_cached
            async with self._lock:
                del items[:overflow]
            if self._debug:
                logger.info("[VideoSense] 缓存超限，裁剪 %d 个最旧条目", overflow)

        yield

    async def _auto_analyze_task(self, umo: str, item: dict) -> None:
        """后台自动分析视频，可选注入对话上下文。"""
        async with self._auto_sem:
            async with self._lock:
                if item["result"]:
                    return
            try:
                resolved = await resolve_video_ref(item, self._max_size_mb)
            except VideoError:
                logger.warning("[VideoSense] 自动分析：文件不可用 %s", item.get("name"))
                return

            client = self._make_client()
            try:
                result = await run_video_analysis_from_path(
                    resolved, self._max_size_mb, client
                )
            except Exception:
                logger.warning(
                    "[VideoSense] 自动分析失败：%s", item.get("name"), exc_info=True
                )
                return
            finally:
                await client.close()

            if not _is_error(result):
                async with self._lock:
                    item["result"] = result
                if self._debug:
                    logger.info("[VideoSense] 自动分析完成：%s", item.get("name"))
                if self._inject_context:
                    async with self._lock:
                        self._pending_injections.setdefault(umo, []).append(
                            {
                                "role": "user",
                                "content": (
                                    f"[视频感知] 刚刚收到的视频「{item['name']}」"
                                    f"分析：{result}"
                                ),
                            }
                        )

    @astr_filter.on_llm_request()
    async def _on_llm_request(self, event: AstrMessageEvent, req):
        """每次 LLM 请求前注入待发送的分析结果。"""
        if not self._inject_context:
            return
        async with self._lock:
            pending = self._pending_injections.pop(event.unified_msg_origin, [])
        if pending:
            if self._debug:
                logger.info("[VideoSense] 注入 %d 条分析到上下文", len(pending))
            req.contexts.extend(pending)

    # ------------------------------------------------------------------
    # 指令处理
    # ------------------------------------------------------------------

    @astr_filter.command("分析视频")
    async def handle_video_command(self, event: AstrMessageEvent):
        """分析视频文件。用法：/分析视频 [序号|追问] [视频文件]"""
        file_comp = extract_file_component(event)
        extra = _extract_extra(event.message_str, "分析视频")

        if file_comp is not None:
            yield event.plain_result("正在分析视频，请稍候...")
            result = await self._analyze_file_comp(
                file_comp, extra, use_command_prompt=True
            )
            yield event.plain_result(result)
            # 同步缓存结果（排除错误）
            if not _is_error(result):
                name = getattr(file_comp, "name", "") or ""
                async with self._lock:
                    for item in self._registry.get(event.unified_msg_origin, []):
                        if item["name"] == name and item["result"] is None:
                            item["result"] = result
                            break
            return

        # 当前消息和引用消息中都没有 File 组件，尝试从缓存取
        umo = event.unified_msg_origin
        async with self._lock:
            items = list(self._registry.get(umo, []))

        if not items:
            yield event.plain_result(
                "没有找到视频文件，请在发送命令时同时附带视频文件，"
                "或引用一条含视频文件的消息。",
            )
            return

        # 解析序号和追问："分析视频 1 这个视频在干嘛"
        # → idx=0, question="这个视频在干嘛"
        idx = -1
        question = ""
        if extra:
            parts = extra.split(None, 1)
            try:
                idx = int(parts[0]) - 1
                question = parts[1] if len(parts) > 1 else ""
            except (ValueError, TypeError):
                question = extra  # 不是数字，全部当追问，但没指定序号

        if idx < 0:
            # 没有指定序号，列出缓存让用户选
            lines = [f"{i + 1}. {it['name']}" for i, it in enumerate(items)]
            yield event.plain_result(
                "对话中有以下视频文件，请指定序号，如：/分析视频 1\n" + "\n".join(lines)
            )
            return

        if idx >= len(items):
            yield event.plain_result(f"序号无效，可选范围 1-{len(items)}。")
            return

        item = items[idx]
        if item["result"]:
            yield event.plain_result(f"「{item['name']}」(已缓存) {item['result']}")
            return

        try:
            resolved = await resolve_video_ref(item, self._max_size_mb)
        except VideoError as e:
            yield event.plain_result(f"「{item['name']}」文件不可用：{e}")
            return

        yield event.plain_result(f"正在分析「{item['name']}」，请稍候...")
        client = self._make_client(question, use_command_prompt=True)
        try:
            result = await run_video_analysis_from_path(
                resolved, self._max_size_mb, client
            )
            if not _is_error(result):
                async with self._lock:
                    item["result"] = result
            yield event.plain_result(f"「{item['name']}」{result}")
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # LLM 工具
    # ------------------------------------------------------------------

    @llm_tool("analyze_current_video")
    async def analyze_current_video(self, event: AstrMessageEvent, question: str = ""):
        """分析当前消息或引用消息中的视频文件。
        当用户直接发送了视频并希望 bot 理解、评价时调用。

        Args:
            question(str, optional): 用户对视频的具体追问，如"这个视频在干嘛？"
        """
        if not self._analysis_cfg.get("enable_llm_tool", True):
            return "视频分析功能未启用。"

        client = self._make_client(question)
        try:
            result = await run_video_analysis(
                event,
                self._supported_formats,
                self._max_size_mb,
                client,
            )
        finally:
            await client.close()

        file_comp = extract_file_component(event)
        if file_comp and not _is_error(result):
            name = getattr(file_comp, "name", "") or ""
            async with self._lock:
                for item in self._registry.get(event.unified_msg_origin, []):
                    if item["name"] == name and item["result"] is None:
                        item["result"] = result
                        break
        return result

    @llm_tool("list_video_files")
    async def list_video_files(self, event: AstrMessageEvent):
        """列出当前对话中出现过的所有视频文件及其序号。
        当用户提到之前的视频但未指定具体哪个时调用。
        """
        umo = event.unified_msg_origin
        async with self._lock:
            items = list(self._registry.get(umo, []))

        if not items:
            return "对话中未收到过视频文件。"

        lines = []
        for i, item in enumerate(items):
            tag = " [已分析]" if item["result"] else ""
            lines.append(f"{i + 1}. {item['name']}{tag}")
        return "对话中的视频文件：\n" + "\n".join(lines)

    @llm_tool("analyze_video_by_number")
    async def analyze_video_by_number(self, event: AstrMessageEvent, number: int):
        """分析对话中指定序号的视频文件。需先调用 list_video_files 获取序号。

        Args:
            number(int): 视频文件序号，从 list_video_files 返回的列表中选择
        """
        if not self._analysis_cfg.get("enable_llm_tool", True):
            return "视频分析功能未启用。"

        umo = event.unified_msg_origin
        async with self._lock:
            items = list(self._registry.get(umo, []))

        if not items:
            return "未找到视频文件，请先发送视频文件。"
        if number < 1 or number > len(items):
            return f"序号无效，可选范围 1-{len(items)}。"

        item = items[number - 1]

        if item["result"]:
            return f"「{item['name']}」(已缓存结果) {item['result']}"

        try:
            resolved_path = await resolve_video_ref(item, self._max_size_mb)
        except VideoError as e:
            return f"「{item['name']}」文件不可用：{e}"

        client = self._make_client()
        try:
            result = await run_video_analysis_from_path(
                resolved_path, self._max_size_mb, client
            )
            if not _is_error(result):
                async with self._lock:
                    item["result"] = result
            return f"「{item['name']}」{result}"
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # 核心分析逻辑
    # ------------------------------------------------------------------

    async def _analyze_file_comp(
        self, file_comp, extra_prompt: str = "", use_command_prompt: bool = False
    ) -> str:
        client = self._make_client(extra_prompt, use_command_prompt)
        try:
            video = await load_video(
                file_comp,
                self._supported_formats,
                self._max_size_mb,
            )
            result = await client.analyze_video(video)
        except VideoError as e:
            return f"文件处理失败：{e}"
        except GeminiClientError as e:
            return f"视频分析失败：{e}"
        else:
            return result
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def terminate(self) -> None:
        self._registry.clear()
        self._pending_injections.clear()
        logger.info("[VideoSense] 插件已卸载")


def _extract_extra(message_str: str, command: str) -> str:
    if not message_str:
        return ""
    text = message_str.strip()
    for prefix in (f"/{command}", command):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return ""

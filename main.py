"""astrbot_plugin_video_sense — 让 Bot 看懂视频。

分析群里分享的视频文件，让 Bot 能就视频展开自然对话。

  触发方式：
  - /分析视频 [序号] [追问] + 视频文件 或 引用视频消息
  - LLM 工具 list_video_files / analyze_video_by_number
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from astrbot.api import AstrBotConfig, llm_tool, logger
from astrbot.api.event import AstrMessageEvent, MessageChain
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
    VideoExpiredError,
    extract_file_component,
    load_video,
    resolve_component_ref,
    video_component_name,
    video_ref_status,
)

_DEFAULT_SYSTEM_PROMPT = (
    "你收到了一段视频。像朋友聊天一样描述这段视频：画面里发生了什么、"
    "有什么值得注意的亮点、整体氛围如何。直接说重点，不要列表，100字以内。"
)

_DEFAULT_COMMAND_SYSTEM_PROMPT = (
    "你收到了一段视频。请像朋友聊天一样详细描述这段视频："
    "画面里发生了什么、人物/场景/动作有哪些值得注意的细节、整体氛围如何。"
    "如果用户带有追问，优先回答追问；没有追问就自然聊聊视频内容。"
    "不要列表，200字以内。"
)

_ERROR_PREFIXES = ("文件处理失败", "视频分析失败")

# 全局并发上限：视频分析单次耗时数十秒，避免多会话同时打满上游配额
_MAX_CONCURRENT_ANALYSES = 2
# 待注入的"视频感知"提示最多保留多少条，防止 LLM 长期不响应时无限累积
_MAX_PENDING_HINTS = 10
# 唤醒主 Agent 的最多步数：正常一次工具调用即可，留些余量给模型纠错
_WAKE_MAX_STEPS = 10
# 单会话条目上限的默认值（可在配置里改，见 analysis.max_entries_per_session）
_DEFAULT_MAX_ENTRIES_PER_SESSION = 50
# 缓存落盘：文件名与合并写入的窗口
_STORE_FILENAME = "video_cache.json"
_SAVE_DEBOUNCE_SECONDS = 0.5
# 落盘保留会话数的默认值（可在配置里改，见 analysis.max_sessions_in_store）
_DEFAULT_MAX_SESSIONS_IN_STORE = 100

# 过期提示：日志、LLM 工具返回、指令回复三处统一口径
_EXPIRED_TOOL_HINT = (
    "缓存只保存文件引用，平台链接和临时文件都会过期。"
    "请让用户重新发送一次这个视频（引用原消息也可以），再调用 analyze_current_video；"
    "不要重复调用本工具，也不要改用其他序号。"
)


def _make_delivery_tool():
    """构造带投递记账的 ``send_message_to_user``（核心工具的子类）。

    发送行为完全复用核心实现（文本/图片/文件、工作区路径映射等），这里只额外
    记录"这一轮是否真的发出去"，并拒绝同一轮内重复发送相同内容。

    Returns:
        工具实例；调用成功后其 ``delivered`` 为 True。
    """
    from astrbot.core.tools.message_tools import SendMessageToUserTool
    from pydantic import Field
    from pydantic.dataclasses import dataclass as pydantic_dataclass

    @pydantic_dataclass
    class _DeliverySendMessageToUserTool(SendMessageToUserTool):
        delivered: bool = False
        sent_signatures: list[str] = Field(default_factory=list)

        async def call(self, context, **kwargs):
            messages = kwargs.get("messages")
            signature = json.dumps(
                messages, ensure_ascii=False, sort_keys=True, default=str
            )
            if signature in self.sent_signatures:
                return (
                    "error: this content was already sent to the user in this turn; "
                    "do not send it again."
                )
            result = await super().call(context, **kwargs)
            if isinstance(result, str) and result.lstrip().startswith("error"):
                return result
            self.sent_signatures.append(signature)
            self.delivered = True
            return result

    return _DeliverySendMessageToUserTool()


def _is_error(text: str) -> bool:
    return text.startswith(_ERROR_PREFIXES)


def _reclaim_item(item: dict) -> str | None:
    """回收插件自己下载的临时文件，返回待删除路径（适配器临时文件不归我们删）。"""
    if not item.get("downloaded"):
        return None
    ref = (item.get("ref") or "").strip()
    item["ref"] = ""
    item["is_local"] = False
    item["downloaded"] = False
    return ref or None


def _delete_files(paths: list[str]) -> None:
    """删除回收的临时文件（在线程池里调用，失败只记日志）。"""
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            logger.debug("[VideoSense] 删除临时文件失败：%s", path)


def _format_relative_time(ts: float) -> str:
    """将时间戳格式化为相对时间（刚刚/N分钟前/N小时前/N天前）。"""
    diff = time.time() - ts
    if diff < 60:
        return "刚刚"
    if diff < 3600:
        return f"{int(diff / 60)}分钟前"
    if diff < 86400:
        return f"{int(diff / 3600)}小时前"
    return f"{int(diff / 86400)}天前"


def _format_video_list(items: list[dict]) -> str:
    """格式化缓存视频列表（含相对接收时间与可用性），供 LLM 分辨新旧与能否分析。"""
    lines = []
    for i, item in enumerate(items):
        when = _format_relative_time(item.get("received_at", 0))
        if item["result"]:
            # 已分析：结果已缓存，文件是否还在都不影响查看
            tag = " [已分析]"
        else:
            usable, _reason = video_ref_status(item)
            tag = "" if usable else " [已过期]"
        lines.append(f"{i + 1}. {item['name']}（{when}）{tag}")
    return "\n".join(lines)


class _FileComponentFilter(CustomFilter):
    """匹配含有 File/Video 组件的消息（含引用消息），用于缓存视频元数据。"""

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        for comp in getattr(event.message_obj, "message", []):
            if type(comp).__name__ in ("File", "Video"):
                return True
            if type(comp).__name__ == "Reply":
                for rc in getattr(comp, "chain", []) or []:
                    if type(rc).__name__ in ("File", "Video"):
                        return True
        return False


class VideoSensePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None) -> None:
        super().__init__(context)
        self.config: AstrBotConfig = config or {}
        self._registry: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()
        self._bg_tasks: set[asyncio.Task] = set()
        self._inflight: dict[tuple[str, str], asyncio.Task] = {}  # 在跑的分析任务
        self._analysis_sem = asyncio.Semaphore(_MAX_CONCURRENT_ANALYSES)
        self._video_hints: dict[str, set[str]] = {}  # umo -> 已提示过 LLM 的视频名
        self._pending_hints: dict[str, list[str]] = {}  # umo -> 待注入的视频名
        self._data_dir: Path | None = None
        self._save_task: asyncio.Task | None = None
        self._init_store()
        self._log_ffmpeg_status()
        logger.info("[VideoSense] 插件已加载，支持格式：%s", self._supported_formats)

    def _log_ffmpeg_status(self) -> None:
        """启动时探测 ffmpeg 并在平台日志输出状态与安装建议。"""
        from .ffmpeg_utils import find_ffmpeg

        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path:
            logger.info("[VideoSense] 已检测到 ffmpeg：%s，压缩功能可用", ffmpeg_path)
            return
        logger.warning(
            "[VideoSense] 未检测到 ffmpeg，中转站/大视频自动压缩功能不可用。"
            "如需启用，两种方式任选：\n"
            "  ① 在 WebUI「平台日志」页面点击「安装 pip 库」，输入 imageio-ffmpeg 并安装"
            "（⚠ 依赖较大，约 80MB，内含 ffmpeg 二进制）；\n"
            "  ② 系统安装 ffmpeg（如 winget install ffmpeg / apt install ffmpeg）。\n"
            "安装完成后重启 AstrBot 生效。"
        )

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
    def _max_entries_per_session(self) -> int:
        """每会话保留的条目上限（内存与落盘一致）。"""
        return max(
            1,
            int(
                self._analysis_cfg.get(
                    "max_entries_per_session", _DEFAULT_MAX_ENTRIES_PER_SESSION
                )
            ),
        )

    @property
    def _max_sessions_in_store(self) -> int:
        """落盘时最多保留多少个最近活跃的会话。"""
        return max(
            1,
            int(
                self._analysis_cfg.get(
                    "max_sessions_in_store", _DEFAULT_MAX_SESSIONS_IN_STORE
                )
            ),
        )

    @property
    def _auto_compress(self) -> bool:
        return bool(self._analysis_cfg.get("auto_compress", False))

    @property
    def _compress_max_duration(self) -> int:
        return max(0, int(self._analysis_cfg.get("compress_max_duration", 120)))

    @property
    def _compress_resolution(self) -> int:
        return max(0, int(self._analysis_cfg.get("compress_resolution", 720)))

    @property
    def _compress_crf(self) -> int:
        return max(0, int(self._analysis_cfg.get("compress_crf", 28)))

    @property
    def _system_prompt(self) -> str:
        return self._analysis_cfg.get("system_prompt", _DEFAULT_SYSTEM_PROMPT).strip()

    @property
    def _separate_prompts(self) -> bool:
        return bool(self._analysis_cfg.get("separate_prompts", False))

    @property
    def _debug(self) -> bool:
        return bool(self._analysis_cfg.get("debug", False))

    @property
    def _command_system_prompt(self) -> str:
        return self._analysis_cfg.get(
            "command_system_prompt", _DEFAULT_COMMAND_SYSTEM_PROMPT
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
        try:
            fps = float(provider.get("fps") or 0)
        except (TypeError, ValueError):
            fps = 0.0
        return GeminiClient(
            api_key=provider.get("api_key", ""),
            model=model,
            system_prompt=sp,
            base_url=provider.get("base_url", ""),
            timeout=int(provider.get("timeout", 300)),
            max_inline_size_mb=self._max_inline_mb,
            use_files_api=self._use_files_api,
            protocol=str(provider.get("protocol", "auto") or "auto"),
            fps=fps,
            compress=self._auto_compress,
            compress_max_duration=self._compress_max_duration,
            compress_resolution=self._compress_resolution,
            compress_crf=self._compress_crf,
        )

    # ------------------------------------------------------------------
    # 视频文件缓存钩子
    # ------------------------------------------------------------------

    @astr_filter.custom_filter(_FileComponentFilter)
    async def _on_file_message(self, event: AstrMessageEvent):
        """消息中包含 File 组件时，缓存其元数据。不下载，不下发消息。"""
        umo = event.unified_msg_origin

        def _cache(comp, items):
            name = video_component_name(comp)
            ext = Path(name).suffix.lstrip(".").lower()
            if not ext and type(comp).__name__ == "Video":
                ext = "mp4"
            if ext not in self._supported_formats:
                if self._debug:
                    logger.info(
                        "[VideoSense] 跳过非视频文件：%s (扩展名 .%s)",
                        name or "(无名称)",
                        ext,
                    )
                return
            local, url = resolve_component_ref(comp)
            ref = local if (local and Path(local).is_file()) else (url or local)
            is_local = bool(local and Path(local).is_file())
            # 去重：同名同引用的视频（如反复引用同一消息）只缓存一次，但刷新接收时间，
            # 并用本次组件重新绑定引用（旧 URL 过期/本地文件被清理时能顺手修正）
            for it in items:
                if it["name"] == name and it["ref"] == ref:
                    it["received_at"] = time.time()
                    if local and Path(local).is_file():
                        it["ref"] = local
                        it["is_local"] = True
                        it["downloaded"] = False
                    elif it["is_local"] and not Path(it["ref"]).is_file():
                        it["ref"] = url or ""
                        it["is_local"] = False
                    return
            items.append(
                {
                    "name": name,
                    "ref": ref,
                    "is_local": is_local,
                    "result": None,
                    "received_at": time.time(),
                }
            )

        new_hints = []
        async with self._lock:
            items = self._registry.setdefault(umo, [])
            hinted = self._video_hints.setdefault(umo, set())
            for comp in getattr(event.message_obj, "message", []):
                if type(comp).__name__ in ("File", "Video"):
                    _cache(comp, items)
                elif type(comp).__name__ == "Reply":
                    for rc in getattr(comp, "chain", []) or []:
                        if type(rc).__name__ in ("File", "Video"):
                            _cache(rc, items)

            # 超出文件预算：只回收最旧条目的文件引用，条目本身保留（序号不漂移），
            # 列表里会显示为「已过期」；已分析结果不受影响。
            to_delete = self._apply_limits(items)

            # 已提示集合跟着缓存一起收敛，被回收的条目再次发送时会重新提示
            hinted &= {item["name"] for item in items}

            # 记录需要提示 LLM 的新视频（去重后新增的、未提示过的）
            for item in items:
                if item["name"] not in hinted:
                    new_hints.append(item["name"])
                    hinted.add(item["name"])
            if new_hints:
                pending = self._pending_hints.setdefault(umo, [])
                pending.extend(new_hints)
                if len(pending) > _MAX_PENDING_HINTS:
                    del pending[:-_MAX_PENDING_HINTS]

        if to_delete:
            await asyncio.to_thread(_delete_files, to_delete)
        self._schedule_save()
        yield

    # ------------------------------------------------------------------
    # 视频感知提示（引导 LLM 调用分析工具）
    # ------------------------------------------------------------------

    @astr_filter.on_llm_request()
    async def _on_llm_request(self, event: AstrMessageEvent, req):
        """每次 LLM 请求前注入"对话中有视频"提示（每视频仅一次，不消耗 API）。

        LLM 上下文中视频消息只是占位符，它不知道视频可分析，
        此提示引导它调用 list_video_files / analyze_current_video。
        """
        umo = event.unified_msg_origin
        if not self._analysis_cfg.get("enable_llm_tool", True):
            return
        async with self._lock:
            pending = self._pending_hints.pop(umo, [])
        if not pending:
            return
        names = "、".join(pending[:5])
        req.contexts.append(
            {
                "role": "user",
                "content": (
                    f"[视频感知] 本对话中有视频文件：{names}。"
                    "如果用户提到视频、想看视频内容或询问视频里有什么，"
                    "请调用 list_video_files 或 analyze_current_video 工具分析视频。"
                ),
            }
        )
        if self._debug:
            logger.info("[VideoSense] 已注入视频感知提示：%s", names)

    # ------------------------------------------------------------------
    # 后台分析（LLM 工具场景：立即返回，结果唤醒 AI 发送）
    # ------------------------------------------------------------------

    def _run_background_analysis(
        self,
        umo: str,
        coro,
        task_key: tuple[str, str] | None = None,
    ) -> bool:
        """后台执行分析协程，完成后唤醒主 Agent 以角色口吻发送结果。

        AstrBot 对 LLM 工具调用有 120 秒硬超时，视频分析可能超时，
        故工具只提交任务立即返回，耗时分析在此后台执行（无超时限制）。

        Args:
            umo: 会话标识，用于回复与唤醒。
            coro: 实际执行分析的协程。
            task_key: 去重键 (umo, 资源标识)；同一资源已有任务在跑时不重复提交。

        Returns:
            是否真的提交了新任务；重复提交时返回 False（传入的协程会被关闭）。
        """
        if task_key is not None:
            existing = self._inflight.get(task_key)
            if existing is not None and not existing.done():
                # 丢弃重复提交的协程，避免 "coroutine was never awaited" 警告
                coro.close()
                return False

        async def _wrapper() -> None:
            try:
                async with self._analysis_sem:
                    result = await coro
            except Exception as e:
                logger.error("[VideoSense] 后台分析异常", exc_info=True)
                await self._deliver_result(umo, f"视频分析失败：{e}")
                return
            await self._deliver_result(umo, result)

        task = asyncio.create_task(_wrapper())
        self._bg_tasks.add(task)
        if task_key is not None:
            self._inflight[task_key] = task

        def _cleanup(done: asyncio.Task) -> None:
            self._bg_tasks.discard(done)
            if task_key is not None and self._inflight.get(task_key) is done:
                self._inflight.pop(task_key, None)

        task.add_done_callback(_cleanup)
        return True

    async def _deliver_result(self, umo: str, result_text: str) -> None:
        """交付分析结果：唤醒 AI 以角色口吻发送；仅在唤醒轮异常时兜底直发。

        与 image_generation 的语义对齐：AI 正常结束但没有调用发送工具，视为它的
        判断（例如它认为这一步不需要打扰用户），不再代发原始分析文本。
        """
        try:
            sent = await self._wake_ai_for_result(umo, result_text)
        except Exception as e:
            logger.warning("[VideoSense] 唤醒 AI 发送失败，改为直接发送：%s", e)
            await self._safe_send(umo, result_text)
            return
        if not sent:
            logger.warning(
                "[VideoSense] 唤醒轮未投递分析结果（AI 未调用 send_message_to_user），"
                "按 AI 判断不代发原文",
            )

    async def _wake_ai_for_result(self, umo: str, result_text: str) -> bool:
        """唤醒主 Agent，用角色口吻把分析结果投递给用户。

        与 image_generation 的唤醒实现对齐：
        - 会话历史直接放进 ``req.contexts``，不再压进 system prompt（保留 provider
          的前缀缓存），分析结果作为额外的用户内容块传入；
        - 发送工具用核心 ``send_message_to_user`` 的子类，成功与否以核心工具返回值
          （``error:`` 前缀）为准，并记录是否真的投递；
        - 保留 build_main_agent 合并出来的完整工具集，只把同名发送工具换成带记账的
          变体（必须在 reset_coro 之前替换，reset 持有同一个 request 对象）。

        Args:
            umo: 会话标识。
            result_text: 待交付的分析结果（或失败原因）。

        Returns:
            AI 是否真的把消息发给了用户。

        Raises:
            RuntimeError: 无法构建主 Agent（例如没有可用的对话模型）。
        """
        from astrbot.core.agent.message import TextPart
        from astrbot.core.agent.tool import ToolSet
        from astrbot.core.astr_main_agent import (
            MainAgentBuildConfig,
            _get_session_conv,
            build_main_agent,
        )
        from astrbot.core.cron.events import CronMessageEvent
        from astrbot.core.platform.message_session import MessageSession
        from astrbot.core.provider.entities import ProviderRequest

        system_prompt = (
            "你是一个自主 Agent。你被唤醒是因为之前提交的视频分析任务已完成。\n"
            "# 重要规则\n"
            "1. 这不是普通对话回合，不要寒暄，不要提问。\n"
            "2. send_message_to_user 是这里唯一能把内容送到用户面前的通道："
            "你的最终回复文本不会被发送。\n"
            "3. 用你平时的角色口吻组织语言（保持人设），但不要改变事实内容。\n"
            "4. 如果任务失败，用角色口吻简要说明失败原因。\n"
            "5. 不要再调用视频分析工具（视频已经分析完），也不要把同一段内容发送两次。\n"
        )

        session = MessageSession.from_str(umo)
        cron_event = CronMessageEvent(
            context=self.context,
            session=session,
            message=f"视频分析任务完成：{result_text[:100]}",
            sender_id="astrbot",
            sender_name="VideoSense",
            message_type=session.message_type,
        )

        cfg = self.context.get_config(umo=umo)
        provider_settings = cfg.get("provider_settings", {})
        tool_call_timeout = provider_settings.get("tool_call_timeout", 120)
        provider = self.context.get_using_provider(umo)
        delivery_tool = _make_delivery_tool()

        req = ProviderRequest()
        req.conversation = await _get_session_conv(
            event=cron_event,
            plugin_context=self.context,
        )
        history_context = json.loads(req.conversation.history or "[]")
        if history_context:
            req.contexts = history_context
        req.system_prompt += system_prompt
        req.prompt = (
            "请按系统指令把视频分析结果发送给用户；如果本轮不需要发送，直接结束即可。"
        )
        req.extra_user_content_parts.append(
            TextPart(
                text=(
                    f"<video_analysis_result>\n{result_text}\n</video_analysis_result>"
                ),
            ),
        )
        req.func_tool = ToolSet()
        req.func_tool.add_tool(delivery_tool)

        config = MainAgentBuildConfig(
            tool_call_timeout=tool_call_timeout,
            llm_safety_mode=False,
            streaming_response=False,
            provider_settings=provider_settings,
            computer_use_runtime="none",
            add_cron_tools=False,
        )
        result = await build_main_agent(
            event=cron_event,
            plugin_context=self.context,
            config=config,
            provider=provider,
            req=req,
            apply_reset=False,
        )
        if not result:
            raise RuntimeError("无法构建主 Agent（可能没有可用的对话模型）")

        # build 会把 persona/插件工具集合并进来，其中核心的同名 send_message_to_user
        # 会覆盖变体；这里换回带记账的版本，必须在 reset 之前完成。
        result.provider_request.func_tool.add_tool(delivery_tool)
        if result.reset_coro:
            await result.reset_coro

        async for _resp in result.agent_runner.step_until_done(_WAKE_MAX_STEPS):
            # 是否发送、发送什么由 AI 决定，本轮不代它发送任何内容。
            pass

        await self._record_wake_note(
            umo=umo,
            conversation=req.conversation,
            runner=result.agent_runner,
            delivered=delivery_tool.delivered,
        )
        return delivery_tool.delivered

    async def _record_wake_note(
        self,
        *,
        umo: str,
        conversation,
        runner,
        delivered: bool,
    ) -> None:
        """把唤醒轮的投递状态与 AI 文本写回会话历史，供后续轮次参考。"""
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None or conversation is None:
            return
        try:
            latest = await manager.get_conversation(umo, conversation.cid)
            raw = getattr(latest, "history", None) or "[]"
            try:
                history = json.loads(raw)
            except (TypeError, ValueError):
                history = []
            if not isinstance(history, list):
                history = []
            llm_resp = runner.get_final_llm_resp()
            note = f"[VideoSenseTask] delivered={str(delivered).lower()}"
            if llm_resp is not None and llm_resp.completion_text:
                note += f" AI response: {llm_resp.completion_text}"
            history.append({"role": "assistant", "content": note})
            token_usage = (
                llm_resp.usage.total
                if llm_resp is not None and llm_resp.usage
                else None
            )
            await manager.update_conversation(
                umo,
                conversation.cid,
                history=history,
                token_usage=token_usage,
            )
        except Exception:
            logger.warning("[VideoSense] 写入唤醒轮历史记录失败", exc_info=True)

    async def _safe_send(self, umo: str, text: str) -> None:
        """向会话主动发送文本，失败仅记日志（不中断后台任务）。"""
        try:
            await self.context.send_message(umo, MessageChain().message(text))
        except Exception:
            logger.warning("[VideoSense] 主动发送分析结果失败", exc_info=True)

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
            # 已分析：结果还在，文件是否已被清理都不影响
            yield event.plain_result(f"「{item['name']}」(已缓存) {item['result']}")
            return

        usable, reason = video_ref_status(item)
        if not usable:
            logger.warning(
                "[VideoSense] 缓存视频已过期：umo=%s name=%s ref=%s reason=%s",
                umo,
                item["name"],
                item.get("ref", ""),
                reason,
            )
            yield event.plain_result(
                f"「{item['name']}」已过期：{reason}。"
                "请重新发送一次这个视频（引用原消息也可以）再试。"
            )
            return

        try:
            resolved = await resolve_video_ref(item, self._max_size_mb)
        except VideoExpiredError as e:
            logger.warning(
                "[VideoSense] 缓存视频已过期：umo=%s name=%s ref=%s reason=%s",
                umo,
                item["name"],
                item.get("ref", ""),
                e,
            )
            yield event.plain_result(
                f"「{item['name']}」已过期：{e}。"
                "请重新发送一次这个视频（引用原消息也可以）再试。"
            )
            return
        except VideoError as e:
            logger.warning("[VideoSense] 缓存视频不可用：%s", e)
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
                self._schedule_save()
            yield event.plain_result(f"「{item['name']}」{result}")
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # LLM 工具
    # ------------------------------------------------------------------

    @llm_tool("analyze_current_video")
    async def analyze_current_video(self, event: AstrMessageEvent, question: str = ""):
        """分析本条消息中直接携带的视频文件（用户刚刚随消息发送的，或引用消息中的视频）。
        仅当本条消息（或引用的消息）确实带有视频时调用本工具。
        如果用户提到的视频不是本条消息附带的（是之前发过的），不要调用本工具，
        请先调用 list_video_files 查看对话中的视频列表，再用 analyze_video_by_number 按序号分析。

        Args:
            question(str, optional): 用户对视频的具体追问，如"这个视频在干嘛？"
        """
        if not self._analysis_cfg.get("enable_llm_tool", True):
            return "视频分析功能未启用。"

        umo = event.unified_msg_origin
        file_comp = extract_file_component(event)
        if file_comp is None:
            # 当前消息没有携带视频：列出缓存（含时间），引导 LLM 按语境选序号
            async with self._lock:
                items = list(self._registry.get(umo, []))
            if not items:
                return (
                    "当前消息中没有视频文件，对话中也没有缓存视频，请让用户先发送视频。"
                )
            return (
                "当前消息没有附带视频。对话中的视频文件（按接收时间）：\n"
                + _format_video_list(items)
                + "\n请根据用户语境判断指的是哪个，"
                "调用 analyze_video_by_number 指定序号分析。"
            )

        submitted = self._run_background_analysis(
            umo,
            self._analyze_current_async(event, question),
            task_key=(umo, f"current:{video_component_name(file_comp)}"),
        )
        if not submitted:
            return (
                "⏳ 这个视频已经在后台分析了，完成后插件会直接把结果发送给用户，"
                "无需重复调用。"
            )
        return (
            "⏳ 视频分析任务已提交，正在后台执行（视频分析可能需要数十秒）。"
            "分析完成后插件会直接把结果发送给用户，"
            "无需重复调用分析工具，等待结果即可。"
        )

    async def _analyze_current_async(
        self, event: AstrMessageEvent, question: str
    ) -> str:
        """后台执行当前消息视频分析，返回要发送给用户的结果文本。"""
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

        if not _is_error(result):
            file_comp = extract_file_component(event)
            if file_comp:
                name = video_component_name(file_comp)
                async with self._lock:
                    for item in self._registry.get(event.unified_msg_origin, []):
                        if item["name"] == name and item["result"] is None:
                            item["result"] = result
                            break
                self._schedule_save()
        return result

    @llm_tool("list_video_files")
    async def list_video_files(self, event: AstrMessageEvent):
        """列出当前对话中出现过的所有视频文件及其序号。
        当用户提到之前发过的视频、询问对话中有哪些视频，
        或需要按序号分析历史视频时调用。调用后再用 analyze_video_by_number 分析指定视频。
        """
        umo = event.unified_msg_origin
        async with self._lock:
            items = list(self._registry.get(umo, []))

        if not items:
            return "对话中未收到过视频文件。"

        return "对话中的视频文件（按接收时间）：\n" + _format_video_list(items)

    @llm_tool("analyze_video_by_number")
    async def analyze_video_by_number(self, event: AstrMessageEvent, number: int):
        """分析对话中指定序号的视频文件（适用于之前发过、本条消息未附带的视频）。
        先调用 list_video_files 获取序号，再调用本工具。

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
        try:
            number = int(number)
        except (TypeError, ValueError):
            return "序号无效，必须是整数。"
        if number < 1 or number > len(items):
            return f"序号无效，可选范围 1-{len(items)}。"

        item = items[number - 1]

        if item["result"]:
            # 已分析：直接返回缓存结果，不依赖文件是否还在
            return f"「{item['name']}」(已缓存结果) {item['result']}"

        usable, reason = video_ref_status(item)
        if not usable:
            logger.warning(
                "[VideoSense] 缓存视频已过期：umo=%s name=%s ref=%s reason=%s",
                umo,
                item["name"],
                item.get("ref", ""),
                reason,
            )
            return f"「{item['name']}」已过期：{reason}。{_EXPIRED_TOOL_HINT}"

        submitted = self._run_background_analysis(
            umo,
            self._analyze_number_async(umo, item),
            task_key=(umo, f"number:{item['ref'] or item['name']}"),
        )
        if not submitted:
            return f"⏳ 「{item['name']}」正在分析中，完成后会自动把结果发给用户。"
        return (
            f"⏳ 已提交后台分析「{item['name']}」（视频分析可能需要数十秒）。"
            "分析完成后插件会直接把结果发送给用户，"
            "无需重复调用分析工具，等待结果即可。"
        )

    async def _analyze_number_async(self, umo: str, item: dict) -> str:
        """后台执行按序号分析，返回要发送给用户的结果文本。"""
        try:
            resolved_path = await resolve_video_ref(item, self._max_size_mb)
        except VideoExpiredError as e:
            logger.warning(
                "[VideoSense] 缓存视频已过期：umo=%s name=%s ref=%s reason=%s",
                umo,
                item["name"],
                item.get("ref", ""),
                e,
            )
            return (
                f"「{item['name']}」已过期：{e}。"
                "请让用户重新发送一次这个视频（引用原消息也可以）后再分析。"
            )
        except VideoError as e:
            logger.warning("[VideoSense] 缓存视频不可用：%s", e)
            return f"「{item['name']}」文件不可用：{e}"
        self._schedule_save()  # ref 已更新为本地文件

        client = self._make_client()
        try:
            result = await run_video_analysis_from_path(
                resolved_path, self._max_size_mb, client
            )
        finally:
            await client.close()

        if not _is_error(result):
            async with self._lock:
                item["result"] = result
            self._schedule_save()
        return f"「{item['name']}」{result}"

    # ------------------------------------------------------------------
    # 缓存限额与持久化
    # ------------------------------------------------------------------

    def _apply_limits(self, items: list[dict]) -> list[str]:
        """按预算回收临时文件，条目本身保留（避免序号漂移）。

        已分析的条目优先回收：结果已经缓存，文件只剩"再分析一次"的用途；
        未分析的文件尽量留着，保证还能分析。

        Args:
            items: 当前会话的缓存条目（就地修改）。

        Returns:
            需要删除的本地文件路径。
        """
        to_delete: list[str] = []
        max_cached = self._max_cached_files
        keep_files = [
            item for item in items if item.get("downloaded") and item.get("ref")
        ]
        if len(keep_files) > max_cached:
            overflow = len(keep_files) - max_cached
            ordered = sorted(
                keep_files,
                key=lambda it: (
                    0 if it.get("result") else 1,
                    it.get("received_at") or 0,
                ),
            )
            for item in ordered[:overflow]:
                path = _reclaim_item(item)
                if path:
                    to_delete.append(path)
            if self._debug:
                logger.info("[VideoSense] 超出文件预算，回收 %d 个临时文件", overflow)

        # 条目硬上限：优先删除最旧的、没有分析结果的条目
        max_entries = self._max_entries_per_session
        while len(items) > max_entries:
            index = next(
                (i for i, item in enumerate(items) if not item.get("result")), 0
            )
            removed = items.pop(index)
            path = _reclaim_item(removed)
            if path:
                to_delete.append(path)
        return to_delete

    def _init_store(self) -> None:
        """初始化缓存存储目录并恢复上次的缓存（失败则退化为纯内存）。"""
        try:
            from astrbot.api.star import StarTools

            self._data_dir = Path(StarTools.get_data_dir("astrbot_plugin_video_sense"))
        except Exception:
            self._data_dir = None
            logger.info("[VideoSense] 无法获取插件数据目录，缓存仅保存在内存中")
            return
        self._load_registry()

    def _store_path(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir / _STORE_FILENAME

    def _load_registry(self) -> None:
        """从磁盘恢复缓存条目。失效引用会在使用时报「已过期」，不影响已分析结果。"""
        if self._data_dir is None:
            return
        path = self._store_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[VideoSense] 读取缓存失败，忽略旧缓存：%s", e)
            return
        sessions = data.get("sessions") if isinstance(data, dict) else None
        if not isinstance(sessions, dict):
            return

        restored = 0
        for umo, items in sessions.items():
            if not isinstance(umo, str) or not isinstance(items, list):
                continue
            entries: list[dict] = []
            for item in items[-self._max_entries_per_session :]:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                ref = item.get("ref")
                if not isinstance(name, str) or not name or not isinstance(ref, str):
                    continue
                result = item.get("result")
                try:
                    received_at = float(item.get("received_at") or 0)
                except (TypeError, ValueError):
                    received_at = 0.0
                entries.append(
                    {
                        "name": name,
                        "ref": ref,
                        "is_local": bool(item.get("is_local")),
                        "result": result if isinstance(result, str) else None,
                        "received_at": received_at,
                        "downloaded": bool(item.get("downloaded")),
                    }
                )
            if entries:
                self._registry[umo] = entries
                restored += len(entries)
        max_sessions = self._max_sessions_in_store
        if len(self._registry) > max_sessions:
            keep = sorted(
                self._registry,
                key=lambda umo: max(
                    (item.get("received_at") or 0 for item in self._registry[umo]),
                    default=0,
                ),
                reverse=True,
            )[:max_sessions]
            self._registry = {umo: self._registry[umo] for umo in keep}
        if restored:
            logger.info("[VideoSense] 已从磁盘恢复 %d 个视频缓存条目", restored)

    async def _persist(self) -> None:
        """把当前缓存原子写回磁盘（失败只记日志）。"""
        if self._data_dir is None:
            return
        async with self._lock:
            active = [(umo, items) for umo, items in self._registry.items() if items]
        active.sort(
            key=lambda pair: max(
                (item.get("received_at") or 0 for item in pair[1]), default=0
            ),
            reverse=True,
        )
        snapshot = {
            umo: [dict(item) for item in items]
            for umo, items in active[: self._max_sessions_in_store]
        }
        path = self._store_path()

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(
                json.dumps({"version": 1, "sessions": snapshot}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(path)

        try:
            await asyncio.to_thread(_write)
        except OSError as e:
            logger.warning("[VideoSense] 写入缓存失败：%s", e)

    def _schedule_save(self) -> None:
        """合并短时间内的多次改动，避免频繁写盘。"""
        if self._data_dir is None:
            return
        if self._save_task is not None and not self._save_task.done():
            return
        self._save_task = asyncio.create_task(self._save_later())

    async def _save_later(self) -> None:
        try:
            await asyncio.sleep(_SAVE_DEBOUNCE_SECONDS)
            await self._persist()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[VideoSense] 缓存落盘失败", exc_info=True)

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
        for task in list(self._bg_tasks):
            task.cancel()
        self._bg_tasks.clear()
        self._inflight.clear()
        if self._save_task is not None:
            self._save_task.cancel()
            self._save_task = None
        await self._persist()
        self._registry.clear()
        self._video_hints.clear()
        self._pending_hints.clear()
        logger.info("[VideoSense] 插件已卸载")


def _extract_extra(message_str: str, command: str) -> str:
    if not message_str:
        return ""
    text = message_str.strip()
    for prefix in (f"/{command}", command):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return ""

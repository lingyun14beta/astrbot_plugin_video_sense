"""唤醒交付链路测试（打桩 astrbot.core.*）。

对齐 image_generation 的语义：
- 发送走核心 send_message_to_user 的子类，只额外记账；
- AI 正常结束但没投递时不再代发原文；只有唤醒轮异常才兜底直发。
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tests.test_main_helpers import _import_main

_main = _import_main()

# 记录核心发送工具真正发出的内容（用于断言实际投递）
SENT_MESSAGES: list[list] = []


@pydantic_dataclass
class _StubSendMessageToUserTool:
    """核心 send_message_to_user 的最小替身：成功返回文案，失败返回 error: 前缀。"""

    name: str = "send_message_to_user"
    description: str = "send message to the user"
    parameters: dict = Field(default_factory=dict)

    async def call(self, context, **kwargs):
        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not messages:
            return "error: messages parameter is empty or invalid."
        SENT_MESSAGES.append(messages)
        return "Message sent to session current"


class _StubToolSet:
    def __init__(self):
        self.tools = []

    def add_tool(self, tool):
        self.tools = [item for item in self.tools if item.name != tool.name]
        self.tools.append(tool)


def _install_core_stubs():
    modules: dict[str, ModuleType] = {}

    def mod(name: str) -> ModuleType:
        module = ModuleType(name)
        modules[name] = module
        sys.modules[name] = module
        return module

    core = mod("astrbot.core")
    agent = mod("astrbot.core.agent")
    agent_tool = mod("astrbot.core.agent.tool")
    agent_tool.ToolSet = _StubToolSet
    agent_message = mod("astrbot.core.agent.message")

    class _TextPart:
        def __init__(self, text: str = "") -> None:
            self.text = text

    agent_message.TextPart = _TextPart

    main_agent = mod("astrbot.core.astr_main_agent")
    main_agent.MainAgentBuildConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    main_agent._get_session_conv = AsyncMock(
        return_value=SimpleNamespace(cid="cid-1", history="[]")
    )

    async def _default_build_main_agent(**_kwargs):
        return None

    main_agent.build_main_agent = _default_build_main_agent

    cron = mod("astrbot.core.cron")
    cron_events = mod("astrbot.core.cron.events")
    cron_events.CronMessageEvent = lambda **kwargs: SimpleNamespace(**kwargs)

    platform = mod("astrbot.core.platform")
    message_session = mod("astrbot.core.platform.message_session")
    message_session.MessageSession = SimpleNamespace(
        from_str=lambda umo: SimpleNamespace(
            message_type="FriendMessage", session_id="1"
        )
    )

    provider = mod("astrbot.core.provider")
    entities = mod("astrbot.core.provider.entities")

    class _Request:
        def __init__(self):
            self.contexts = []
            self.system_prompt = ""
            self.prompt = None
            self.conversation = None
            self.func_tool = None
            self.extra_user_content_parts = []

    entities.ProviderRequest = _Request

    tools = mod("astrbot.core.tools")
    tools_message = mod("astrbot.core.tools.message_tools")
    tools_message.SendMessageToUserTool = _StubSendMessageToUserTool

    core.agent = agent
    core.astr_main_agent = main_agent
    core.cron = cron
    core.platform = platform
    core.provider = provider
    core.tools = tools
    agent.tool = agent_tool
    agent.message = agent_message
    cron.events = cron_events
    platform.message_session = message_session
    provider.entities = entities
    tools.message_tools = tools_message
    return modules


class _Runner:
    def __init__(self, req, behavior: dict) -> None:
        self.req = req
        self.behavior = behavior
        self.text = behavior.get("text", "模型文本")

    async def step_until_done(self, max_step):
        mode = self.behavior.get("mode", "send")
        if mode in ("send", "send_twice"):
            tool = self.req.func_tool.tools[0]
            await tool.call(
                None, messages=[{"type": "plain", "text": "人格化后的结果"}]
            )
            if mode == "send_twice":
                await tool.call(
                    None, messages=[{"type": "plain", "text": "人格化后的结果"}]
                )
        yield None

    def get_final_llm_resp(self):
        return SimpleNamespace(completion_text=self.text, usage=None)


class _ConvManager:
    def __init__(self) -> None:
        self.updated: list[dict] = []

    async def get_conversation(self, umo, cid):
        return SimpleNamespace(cid=cid, history="[]")

    async def update_conversation(self, umo, cid, history=None, token_usage=None):
        self.updated.append(
            {"umo": umo, "cid": cid, "history": history, "token_usage": token_usage}
        )


def _make_plugin(behavior: dict | None = None):
    behavior = behavior or {"mode": "send"}
    conv = _ConvManager()
    send = AsyncMock(return_value=True)
    context = SimpleNamespace(
        send_message=send,
        conversation_manager=conv,
        get_config=lambda umo=None: {"provider_settings": {}},
        get_using_provider=lambda umo=None: None,
    )
    plugin = _main.VideoSensePlugin(context=context, config={})
    return plugin, send, conv, behavior


@pytest.fixture
def wake_env():
    modules = _install_core_stubs()
    SENT_MESSAGES.clear()
    yield modules
    for name in modules:
        sys.modules.pop(name, None)
    SENT_MESSAGES.clear()


def _patch_build(monkeypatch, behavior: dict, built: dict):
    async def build_main_agent(
        *, event, plugin_context, config, provider=None, req=None, apply_reset=True
    ):
        built["req"] = req
        return SimpleNamespace(
            agent_runner=_Runner(req, behavior),
            provider_request=req,
            reset_coro=None,
        )

    monkeypatch.setattr(
        sys.modules["astrbot.core.astr_main_agent"], "build_main_agent", build_main_agent
    )


class TestWakeDelivery:
    async def test_returns_true_when_tool_sent(self, wake_env, monkeypatch):
        behavior = {"mode": "send"}
        built: dict = {}
        _patch_build(monkeypatch, behavior, built)
        plugin, send, conv, _ = _make_plugin(behavior)

        assert await plugin._wake_ai_for_result("umo:1", "分析结果") is True

        assert len(SENT_MESSAGES) == 1
        assert SENT_MESSAGES[0][0]["text"] == "人格化后的结果"
        send.assert_not_awaited()  # 没有走插件自己的兜底发送
        assert conv.updated
        note = conv.updated[-1]["history"][-1]["content"]
        assert "delivered=true" in note
        assert "人格化后的结果" not in note  # note 里是 AI 文本，不是投递内容
        assert "模型文本" in note

    async def test_returns_false_when_model_only_replies_text(
        self, wake_env, monkeypatch
    ):
        behavior = {"mode": "text_only"}
        built: dict = {}
        _patch_build(monkeypatch, behavior, built)
        plugin, _, conv, _ = _make_plugin(behavior)

        assert await plugin._wake_ai_for_result("umo:1", "分析结果") is False
        assert SENT_MESSAGES == []
        assert "delivered=false" in conv.updated[-1]["history"][-1]["content"]

    async def test_duplicate_send_is_rejected(self, wake_env, monkeypatch):
        behavior = {"mode": "send_twice"}
        built: dict = {}
        _patch_build(monkeypatch, behavior, built)
        plugin, _, _, _ = _make_plugin(behavior)

        assert await plugin._wake_ai_for_result("umo:1", "分析结果") is True
        assert len(SENT_MESSAGES) == 1  # 第二次被工具自己拒绝

    async def test_build_failure_raises_for_fallback(self, wake_env, monkeypatch):
        async def build_main_agent(**_kwargs):
            return None

        monkeypatch.setattr(
            sys.modules["astrbot.core.astr_main_agent"],
            "build_main_agent",
            build_main_agent,
        )
        plugin, _, _, _ = _make_plugin()

        with pytest.raises(RuntimeError, match="无法构建主 Agent"):
            await plugin._wake_ai_for_result("umo:1", "分析结果")


class TestDeliverResult:
    async def test_exception_falls_back_to_raw_send(self):
        plugin, send, _, _ = _make_plugin()
        plugin._wake_ai_for_result = AsyncMock(side_effect=RuntimeError("boom"))

        await plugin._deliver_result("umo:1", "原始分析")

        send.assert_awaited_once()
        assert "原始分析" in str(send.await_args.args[1])

    async def test_ai_skipped_send_does_not_fall_back(self):
        """AI 正常结束但没投递：不代发原文（与 image_generation 一致）。"""
        plugin, send, _, _ = _make_plugin()
        plugin._wake_ai_for_result = AsyncMock(return_value=False)

        await plugin._deliver_result("umo:1", "原始分析")

        send.assert_not_awaited()

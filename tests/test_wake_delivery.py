"""唤醒交付链路测试（打桩 astrbot.core.*）：

发送是否成功由插件自己的工具确认，不依赖上游返回的界面文案。
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.test_main_helpers import _import_main

_main = _import_main()


class _FakeFunctionTool:
    def __init__(self, name, description, parameters, handler=None, **kwargs):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler


class _FakeToolSet:
    def __init__(self):
        self.tools = []

    def add_tool(self, tool):
        self.tools = [item for item in self.tools if item.name != tool.name]
        self.tools.append(tool)


def _install_core_stubs():
    """补齐 astrbot.core.* 子模块，供 main._wake_ai_for_result 内部导入。"""
    modules: dict[str, ModuleType] = {}

    def mod(name: str) -> ModuleType:
        module = ModuleType(name)
        modules[name] = module
        sys.modules[name] = module
        return module

    core = mod("astrbot.core")
    agent = mod("astrbot.core.agent")
    tool = mod("astrbot.core.agent.tool")
    tool.FunctionTool = _FakeFunctionTool
    tool.ToolSet = _FakeToolSet

    main_agent = mod("astrbot.core.astr_main_agent")
    main_agent.MainAgentBuildConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    main_agent._get_session_conv = AsyncMock(
        return_value=SimpleNamespace(history="[]")
    )

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

    entities.ProviderRequest = _Request

    core.agent = agent
    core.astr_main_agent = main_agent
    core.cron = cron
    core.platform = platform
    core.provider = provider
    agent.tool = tool
    cron.events = cron_events
    platform.message_session = message_session
    provider.entities = entities
    return modules


@pytest.fixture
def wake_env(request):
    """安装 core 桩，并按测试需要构造 build_main_agent 的行为。"""
    modules = _install_core_stubs()
    behavior = {"mode": "send"}

    async def build_main_agent(
        *, event, plugin_context, config, provider=None, req=None, apply_reset=True
    ):
        async def _step():
            if behavior["mode"] == "no_tool":
                return None
            await req.func_tool.tools[0].handler(
                event, messages=[{"type": "plain", "text": "人格化后的结果"}]
            )
            return SimpleNamespace(type="tool_call_result", data={})

        class _Runner:
            async def step_until_done(self, max_step):
                yield await _step()

        return SimpleNamespace(
            agent_runner=_Runner(),
            provider_request=req,
            reset_coro=None,
        )

    modules["astrbot.core.astr_main_agent"].build_main_agent = build_main_agent
    yield behavior
    for name in modules:
        sys.modules.pop(name, None)


def _make_plugin(send_message: AsyncMock):
    context = SimpleNamespace(
        send_message=send_message,
        get_config=lambda umo=None: {"provider_settings": {}},
        get_using_provider=lambda umo=None: None,
    )
    return _main.VideoSensePlugin(context=context, config={})


class TestWakeDelivery:
    async def test_returns_true_when_tool_sent(self, wake_env):
        send = AsyncMock(return_value=True)
        plugin = _make_plugin(send)

        assert await plugin._wake_ai_for_result("umo:1", "分析结果") is True
        send.assert_awaited_once()
        sent_chain = send.await_args.args[1]
        assert sent_chain.chain == ["人格化后的结果"]

    async def test_returns_false_when_model_skips_tool(self, wake_env):
        """模型没调工具（只回了文本）时判定失败，交给兜底直发。"""
        wake_env["mode"] = "no_tool"
        send = AsyncMock(return_value=True)
        plugin = _make_plugin(send)

        assert await plugin._wake_ai_for_result("umo:1", "分析结果") is False
        send.assert_not_awaited()

    async def test_returns_false_when_platform_send_fails(self, wake_env):
        """平台发送失败时也不误判为成功（不依赖任何界面文案）。"""
        send = AsyncMock(return_value=False)
        plugin = _make_plugin(send)

        assert await plugin._wake_ai_for_result("umo:1", "分析结果") is False
        send.assert_awaited_once()

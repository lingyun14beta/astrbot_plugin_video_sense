"""main.py 纯函数测试（打桩 astrbot 依赖，不启动插件）。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _stub_astrbot():
    """在 sys.modules 中打桩 astrbot 包，使 main.py 可独立导入。"""
    stub_modules: dict[str, ModuleType] = {}
    for name in ("astrbot", "astrbot.api", "astrbot.api.event", "astrbot.api.star"):
        m = ModuleType(name)
        stub_modules[name] = m
        sys.modules[name] = m

    stub_modules["astrbot.api"].AstrBotConfig = dict
    stub_modules["astrbot.api"].llm_tool = lambda name=None: lambda f: f
    stub_modules["astrbot.api"].logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    stub_modules["astrbot.api.event"].AstrMessageEvent = type(
        "AstrMessageEvent", (), {}
    )

    class _MessageChainStub:
        def __init__(self):
            self._text = ""

        def message(self, text):
            self._text = str(text)
            return self

        def __str__(self):
            return self._text

    stub_modules["astrbot.api.event"].MessageChain = _MessageChainStub

    filter_mod = ModuleType("astrbot.api.event.filter")
    filter_mod.CustomFilter = type("CustomFilter", (), {})
    filter_mod.custom_filter = lambda cls: lambda f: f
    filter_mod.command = lambda name=None: lambda f: f
    filter_mod.on_llm_request = lambda: lambda f: f
    sys.modules["astrbot.api.event.filter"] = filter_mod

    stub_modules["astrbot.api.star"].Context = type("Context", (), {})

    class _StarStub:
        def __init__(self, context, config=None):
            self.context = context

    stub_modules["astrbot.api.star"].Star = _StarStub


def _import_main():
    _stub_astrbot()
    pkg = ModuleType("astrbot_plugin_video_sense")
    pkg.__path__ = [str(PLUGIN_ROOT)]
    sys.modules["astrbot_plugin_video_sense"] = pkg
    return importlib.import_module("astrbot_plugin_video_sense.main")


_main = _import_main()


class TestIsError:
    def test_file_error(self):
        assert _main._is_error("文件处理失败：格式不支持") is True

    def test_analysis_error(self):
        assert _main._is_error("视频分析失败：API 超时") is True

    def test_normal_result(self):
        assert _main._is_error("这是一个猫的视频") is False

    def test_empty(self):
        assert _main._is_error("") is False


class TestExtractExtra:
    def test_slash_command(self):
        assert _main._extract_extra("/分析视频 1 追问内容", "分析视频") == "1 追问内容"

    def test_slash_command_no_extra(self):
        assert _main._extract_extra("/分析视频", "分析视频") == ""

    def test_plain_command(self):
        assert _main._extract_extra("分析视频 1", "分析视频") == "1"

    def test_empty_message(self):
        assert _main._extract_extra("", "分析视频") == ""

    def test_unrelated_message(self):
        assert _main._extract_extra("hello", "分析视频") == ""

    def test_multi_space(self):
        assert (
            _main._extract_extra("/分析视频   1   这个视频", "分析视频")
            == "1   这个视频"
        )


class TestBackgroundAnalysis:
    """LLM 工具后台分析：立即返回 + 结果主动发送 + 异常兜底。"""

    def _make_plugin(self, send_side_effect=None):
        send = AsyncMock(side_effect=send_side_effect)
        context = SimpleNamespace(send_message=send)
        plugin = _main.VideoSensePlugin(context=context, config={})
        return plugin, send

    async def test_result_sent_to_session(self):
        plugin, send = self._make_plugin()

        async def fake_coro():
            return "分析结果"

        plugin._run_background_analysis("umo1", fake_coro())
        task = next(iter(plugin._bg_tasks))
        await task

        send.assert_awaited_once()
        args = send.await_args
        assert args.args[0] == "umo1"
        assert "分析结果" in str(args.args[1])

    async def test_exception_sends_failure_notice(self):
        plugin, send = self._make_plugin()

        async def broken_coro():
            raise RuntimeError("API 爆炸")

        plugin._run_background_analysis("umo1", broken_coro())
        task = next(iter(plugin._bg_tasks))
        await task

        send.assert_awaited_once()
        assert "视频分析失败" in str(send.await_args.args[1])

    async def test_send_failure_does_not_crash(self):
        plugin, send = self._make_plugin(send_side_effect=RuntimeError("平台不可达"))

        async def fake_coro():
            return "分析结果"

        plugin._run_background_analysis("umo1", fake_coro())
        task = next(iter(plugin._bg_tasks))
        await task  # 不应抛出

        send.assert_awaited_once()


class TestVideoHintInjection:
    """视频感知提示注入：引导 LLM 调用分析工具，每视频仅提示一次。"""

    def _make_plugin(self):
        plugin, _ = TestBackgroundAnalysis()._make_plugin()
        return plugin

    async def test_pending_hint_injected_once(self):
        plugin = self._make_plugin()
        # 模拟缓存钩子记录的新视频提示
        plugin._pending_hints["umo1"] = ["clip.mp4"]
        plugin._video_hints["umo1"] = {"clip.mp4"}

        req = SimpleNamespace(contexts=[])
        event = SimpleNamespace(unified_msg_origin="umo1")

        await plugin._on_llm_request(event, req)
        assert len(req.contexts) == 1
        assert "视频感知" in req.contexts[0]["content"]
        assert "clip.mp4" in req.contexts[0]["content"]

        # 注入后 pending 清空，第二次不再注入
        await plugin._on_llm_request(event, req)
        assert len(req.contexts) == 1

    async def test_no_pending_no_injection(self):
        plugin = self._make_plugin()
        req = SimpleNamespace(contexts=[])
        event = SimpleNamespace(unified_msg_origin="umo1")
        await plugin._on_llm_request(event, req)
        assert req.contexts == []

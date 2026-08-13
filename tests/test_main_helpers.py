"""main.py 纯函数测试（打桩 astrbot 依赖，不启动插件）。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    )
    stub_modules["astrbot.api.event"].AstrMessageEvent = type(
        "AstrMessageEvent", (), {}
    )

    filter_mod = ModuleType("astrbot.api.event.filter")
    filter_mod.CustomFilter = type("CustomFilter", (), {})
    filter_mod.custom_filter = lambda cls: lambda f: f
    filter_mod.command = lambda name=None: lambda f: f
    filter_mod.on_llm_request = lambda: lambda f: f
    sys.modules["astrbot.api.event.filter"] = filter_mod

    stub_modules["astrbot.api.star"].Context = type("Context", (), {})
    stub_modules["astrbot.api.star"].Star = type("Star", (), {})


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

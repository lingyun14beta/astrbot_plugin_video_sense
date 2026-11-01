"""缓存生命周期测试：过期提示、限额保留、持久化与引用重绑定。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.test_main_helpers import _import_main

_main = _import_main()


def _make_plugin(config: dict | None = None):
    return _main.VideoSensePlugin(
        context=SimpleNamespace(send_message=AsyncMock()), config=config or {}
    )


def _entry(
    name: str = "a.mp4",
    ref: str = "",
    is_local: bool = False,
    result: str | None = None,
    received_at: float = 0.0,
    downloaded: bool = False,
) -> dict:
    return {
        "name": name,
        "ref": ref,
        "is_local": is_local,
        "result": result,
        "received_at": received_at,
        "downloaded": downloaded,
    }


def _command_event(umo: str, text: str):
    return SimpleNamespace(
        unified_msg_origin=umo,
        message_str=text,
        message_obj=SimpleNamespace(message=[]),
        plain_result=lambda content: content,
    )


def _file_comp(name: str, path: str):
    return type("File", (), {"name": name, "file_": path, "url": ""})()


class TestListTags:
    def test_missing_local_file_marked_expired(self, temp_dir):
        items = [_entry(ref=str(temp_dir / "gone.mp4"), is_local=True)]
        assert "[已过期]" in _main._format_video_list(items)

    def test_analyzed_entry_not_marked_expired(self, temp_dir):
        """已分析的条目文件没了也照常显示已分析（结果已缓存）。"""
        items = [
            _entry(ref=str(temp_dir / "gone.mp4"), is_local=True, result="结果文本")
        ]
        text = _main._format_video_list(items)
        assert "[已分析]" in text
        assert "[已过期]" not in text

    def test_remote_entry_has_no_tag(self):
        items = [_entry(ref="https://example.com/a.mp4", is_local=False)]
        assert "[已过期]" not in _main._format_video_list(items)


class TestToolExpiry:
    async def test_expired_entry_returns_clear_error(self):
        plugin = _make_plugin()
        plugin._registry["umo"] = [_entry()]
        event = SimpleNamespace(unified_msg_origin="umo")

        text = await plugin.analyze_video_by_number(event, 1)

        assert "已过期" in text
        assert "重新发送" in text
        assert "analyze_current_video" in text

    async def test_missing_local_file_returns_clear_error(self, temp_dir):
        plugin = _make_plugin()
        plugin._registry["umo"] = [
            _entry(ref=str(temp_dir / "gone.mp4"), is_local=True)
        ]

        text = await plugin.analyze_video_by_number(
            SimpleNamespace(unified_msg_origin="umo"), 1
        )

        assert "已过期" in text
        assert "本地文件已被清理" in text

    async def test_analyzed_entry_returns_cached_result(self, temp_dir):
        """已分析：文件被清理也直接返回结果，不报错。"""
        plugin = _make_plugin()
        plugin._registry["umo"] = [
            _entry(
                ref=str(temp_dir / "gone.mp4"),
                is_local=True,
                result="这是之前的结果",
            )
        ]

        text = await plugin.analyze_video_by_number(
            SimpleNamespace(unified_msg_origin="umo"), 1
        )

        assert "已缓存结果" in text
        assert "这是之前的结果" in text


class TestCommandExpiry:
    async def test_expired_entry_replies_with_error(self):
        plugin = _make_plugin()
        plugin._registry["umo"] = [_entry()]
        event = _command_event("umo", "分析视频 1")

        replies = [reply async for reply in plugin.handle_video_command(event)]

        assert replies
        assert "已过期" in replies[-1]
        assert "重新发送" in replies[-1]

    async def test_analyzed_entry_replies_cached_result(self, temp_dir):
        plugin = _make_plugin()
        plugin._registry["umo"] = [
            _entry(
                ref=str(temp_dir / "gone.mp4"),
                is_local=True,
                result="缓存的结果",
            )
        ]
        event = _command_event("umo", "分析视频 1")

        replies = [reply async for reply in plugin.handle_video_command(event)]

        assert "缓存的结果" in replies[-1]


class TestLimitsKeepEntries:
    def test_overflow_reclaims_files_but_keeps_entries(self, temp_dir):
        path = temp_dir / "a.mp4"
        path.write_bytes(b"x")
        items = [
            _entry(
                name=f"v{i}.mp4",
                ref=str(path),
                is_local=True,
                received_at=i,
                downloaded=True,
            )
            for i in range(3)
        ]
        plugin = _make_plugin({"analysis": {"max_cached_files": 2}})

        to_delete = plugin._apply_limits(items)

        assert len(items) == 3  # 条目保留，序号不漂移
        assert items[0]["ref"] == ""  # 最旧的被回收
        assert items[0]["is_local"] is False
        assert str(path) in to_delete
        assert items[2]["ref"] == str(path)

    def test_non_downloaded_local_ref_untouched(self, temp_dir):
        """适配器自己的临时文件既不由插件删除，也不清引用。"""
        path = str(temp_dir / "v0.mp4")
        items = [_entry(name="v0.mp4", ref=path, is_local=True)]
        plugin = _make_plugin({"analysis": {"max_cached_files": 0}})

        assert plugin._apply_limits(items) == []
        assert items[0]["ref"] == path
        assert items[0]["is_local"] is True

    def test_analyzed_entries_reclaimed_first(self, temp_dir):
        """已分析的条目优先回收：结果已缓存，文件只剩再分析用途。"""
        items = [
            _entry(
                name="old-unanalyzed.mp4",
                ref=str(temp_dir / "a.mp4"),
                is_local=True,
                received_at=0,
                downloaded=True,
            ),
            _entry(
                name="analyzed.mp4",
                ref=str(temp_dir / "b.mp4"),
                is_local=True,
                result="已分析结果",
                received_at=1,
                downloaded=True,
            ),
            _entry(
                name="new-unanalyzed.mp4",
                ref=str(temp_dir / "c.mp4"),
                is_local=True,
                received_at=2,
                downloaded=True,
            ),
        ]
        plugin = _make_plugin({"analysis": {"max_cached_files": 2}})

        to_delete = plugin._apply_limits(items)

        assert str(temp_dir / "b.mp4") in to_delete
        assert items[0]["ref"] != ""
        assert items[1]["ref"] == ""
        assert items[2]["ref"] != ""

    def test_entries_cap_from_config(self):
        items = [_entry(name=f"v{i}.mp4", result="结果") for i in range(5)]
        plugin = _make_plugin({"analysis": {"max_entries_per_session": 3}})

        plugin._apply_limits(items)

        assert len(items) == 3

    def test_hard_cap_drops_oldest_without_result(self):
        items = [
            _entry(name=f"v{i}.mp4", result="结果" if i >= 2 else None)
            for i in range(_main._DEFAULT_MAX_ENTRIES_PER_SESSION + 3)
        ]
        plugin = _make_plugin()

        plugin._apply_limits(items)

        assert len(items) == _main._DEFAULT_MAX_ENTRIES_PER_SESSION
        # 没有结果的旧条目先被删掉
        assert all(item["result"] for item in items)


class TestPersistence:
    async def test_roundtrip_restores_entries_and_results(self, temp_dir):
        plugin = _make_plugin()
        plugin._data_dir = temp_dir
        plugin._registry["umo"] = [
            _entry(name="a.mp4", ref="https://example.com/a.mp4", result="旧结果")
        ]

        await plugin._persist()

        restored = _make_plugin()
        restored._data_dir = temp_dir
        restored._load_registry()

        assert restored._registry["umo"][0]["name"] == "a.mp4"
        assert restored._registry["umo"][0]["result"] == "旧结果"
        assert restored._registry["umo"][0]["is_local"] is False

    async def test_corrupt_store_is_ignored(self, temp_dir):
        (temp_dir / _main._STORE_FILENAME).write_text("{not json", encoding="utf-8")
        plugin = _make_plugin()
        plugin._data_dir = temp_dir

        plugin._load_registry()

        assert plugin._registry == {}

    async def test_store_keeps_recent_sessions_only(self, temp_dir):
        plugin = _make_plugin()
        plugin._data_dir = temp_dir
        total = _main._DEFAULT_MAX_SESSIONS_IN_STORE + 5
        for i in range(total):
            plugin._registry[f"umo{i}"] = [
                _entry(name=f"v{i}.mp4", received_at=float(i))
            ]

        await plugin._persist()

        data = json.loads(
            (temp_dir / _main._STORE_FILENAME).read_text(encoding="utf-8")
        )
        assert len(data["sessions"]) == _main._DEFAULT_MAX_SESSIONS_IN_STORE
        assert f"umo{total - 1}" in data["sessions"]
        assert "umo0" not in data["sessions"]

    async def test_sessions_cap_from_config(self, temp_dir):
        plugin = _make_plugin({"analysis": {"max_sessions_in_store": 2}})
        plugin._data_dir = temp_dir
        for i in range(4):
            plugin._registry[f"umo{i}"] = [_entry(received_at=float(i))]

        await plugin._persist()

        data = json.loads(
            (temp_dir / _main._STORE_FILENAME).read_text(encoding="utf-8")
        )
        assert set(data["sessions"]) == {"umo2", "umo3"}

    async def test_load_honors_entries_cap(self, temp_dir):
        payload = {
            "version": 1,
            "sessions": {"umo": [_entry(name=f"v{i}.mp4") for i in range(6)]},
        }
        (temp_dir / _main._STORE_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        plugin = _make_plugin({"analysis": {"max_entries_per_session": 2}})
        plugin._data_dir = temp_dir

        plugin._load_registry()

        assert [item["name"] for item in plugin._registry["umo"]] == [
            "v4.mp4",
            "v5.mp4",
        ]

    async def test_store_file_written(self, temp_dir):
        plugin = _make_plugin()
        plugin._data_dir = temp_dir
        plugin._registry["umo"] = [_entry()]

        await plugin._persist()

        data = json.loads(
            (temp_dir / _main._STORE_FILENAME).read_text(encoding="utf-8")
        )
        assert data["sessions"]["umo"][0]["name"] == "a.mp4"


class TestDedupeRebinding:
    async def test_reference_upgraded_to_existing_local_file(self, temp_dir):
        """同一 (name, ref) 再次出现时，用最新组件重新绑定本地文件。"""
        path = temp_dir / "clip.mp4"
        plugin = _make_plugin()

        first = SimpleNamespace(
            unified_msg_origin="umo",
            message_obj=SimpleNamespace(message=[_file_comp("clip.mp4", str(path))]),
        )
        async for _ in plugin._on_file_message(first):
            pass
        item = plugin._registry["umo"][0]
        assert item["is_local"] is False  # 此时文件还不存在

        path.write_bytes(b"data")
        second = SimpleNamespace(
            unified_msg_origin="umo",
            message_obj=SimpleNamespace(message=[_file_comp("clip.mp4", str(path))]),
        )
        async for _ in plugin._on_file_message(second):
            pass

        assert len(plugin._registry["umo"]) == 1
        assert plugin._registry["umo"][0]["is_local"] is True

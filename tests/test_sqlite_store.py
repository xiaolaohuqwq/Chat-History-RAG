from datetime import UTC, datetime
from pathlib import Path

from chat_rag.domain import Message
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.windowing import build_windows


def test_schema_persists_messages_and_windows_idempotently(tmp_path: Path) -> None:
    message = Message(
        message_id="m_1",
        source_id="source",
        source_line=1,
        time_raw="2026-01-01 10:00:00",
        time_utc=datetime(2026, 1, 1, 10, tzinfo=UTC),
        uid="u1",
        name="甲",
        text="项目编号 ABC-123",
        content_hash="hash",
    )
    window = build_windows([message])[0]

    with SQLiteStore(tmp_path / "app.db") as store:
        store.upsert_messages([message])
        store.upsert_messages([message])
        store.upsert_windows([window])
        store.upsert_windows([window])
        assert store.count("messages") == 1
        assert store.count("windows") == 1
        assert store.get_message("m_1") == message
        assert store.get_window(window.window_id) is not None

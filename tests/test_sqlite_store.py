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


def test_store_enables_wal_for_concurrent_tui_reads(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path / "app.db") as store:
        mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = store.connection.execute("PRAGMA busy_timeout").fetchone()[0]
    assert mode == "wal"
    assert timeout >= 5_000


def test_starting_run_marks_abandoned_running_run_interrupted(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path / "app.db") as store:
        first = store.start_ingestion_run("source", "a" * 64, 1, "model", 3, datetime.now(UTC))
        store.start_ingestion_run("source", "b" * 64, 2, "model", 3, datetime.now(UTC))
        status, error_summary, completed_at = store.connection.execute(
            """SELECT status, error_summary, completed_at
            FROM ingestion_runs WHERE run_id = ?""",
            (first,),
        ).fetchone()
    assert status == "interrupted"
    assert error_summary == "Superseded by a new ingestion run"
    assert completed_at is not None

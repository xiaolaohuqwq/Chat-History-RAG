from datetime import UTC, datetime, timedelta
from pathlib import Path

from chat_rag.context_builder import build_evidence_blocks, pack_evidence
from chat_rag.domain import Message
from chat_rag.retrieval import SearchResult
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.token_estimator import estimate_tokens
from chat_rag.windowing import build_windows


def message(line: int, text: str = "消息") -> Message:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=line)
    return Message(
        f"m{line}",
        "source",
        line,
        timestamp.isoformat(),
        timestamp,
        f"u{line % 2}",
        f"用户{line % 2}",
        text,
        f"h{line}",
    )


def test_adjacent_expansion_merges_overlapping_ranges(tmp_path: Path) -> None:
    messages = [message(line) for line in range(1, 6)]
    windows = [build_windows([messages[1]])[0], build_windows([messages[2]])[0]]
    with SQLiteStore(tmp_path / "app.db") as store:
        store.upsert_messages(messages)
        store.upsert_windows(windows)
        store.get_source_messages = lambda source_id: (_ for _ in ()).throw(
            AssertionError("context expansion must not load the entire source")
        )
        results = [
            SearchResult(windows[0], (messages[1],), 1.0),
            SearchResult(windows[1], (messages[2],), 0.9),
        ]

        blocks = build_evidence_blocks(store, results, adjacent_messages=1)

    assert len(blocks) == 1
    assert [item.message_id for item in blocks[0].messages] == ["m1", "m2", "m3", "m4"]
    assert blocks[0].evidence_id == "e1"


def test_evidence_packing_never_exceeds_budget_or_splits_messages(tmp_path: Path) -> None:
    messages = [message(line, "中" * 20) for line in range(1, 5)]
    windows = [build_windows([item])[0] for item in messages]
    with SQLiteStore(tmp_path / "app.db") as store:
        store.upsert_messages(messages)
        store.upsert_windows(windows)
        results = [
            SearchResult(window, (item,), 1 / item.source_line)
            for window, item in zip(windows, messages, strict=True)
        ]
        blocks = build_evidence_blocks(store, results, adjacent_messages=0)

    packed = pack_evidence(blocks, max_tokens=60)

    assert estimate_tokens(packed.text) <= 60
    assert set(packed.message_ids).issubset({item.message_id for item in messages})
    assert "m1" in packed.message_ids

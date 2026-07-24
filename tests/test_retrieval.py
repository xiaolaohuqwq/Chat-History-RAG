from datetime import UTC, datetime
from pathlib import Path

import pytest

from chat_rag.domain import Message
from chat_rag.retrieval import HybridRetriever, lexical_terms, reciprocal_rank_fusion
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.vector_store import VectorResult
from chat_rag.windowing import build_windows


class QueryEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class QueryVectors:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def upsert(self, rows):
        pass

    def search(self, vector: list[float], limit: int) -> list[VectorResult]:
        return [
            VectorResult(window_id, index / 10) for index, window_id in enumerate(self.ids[:limit])
        ]

    def count(self) -> int:
        return len(self.ids)

    def clear(self) -> None:
        pass


def stored_windows(store: SQLiteStore) -> list[str]:
    messages = [
        Message(
            "m1",
            "s",
            1,
            "2026-01-01 10:00:00",
            datetime(2026, 1, 1, tzinfo=UTC),
            "u1",
            "甲",
            "部署编号 ABC-123",
            "h1",
        ),
        Message(
            "m2",
            "s",
            2,
            "2026-01-01 10:01:00",
            datetime(2026, 1, 1, tzinfo=UTC),
            "u2",
            "乙",
            "大家讨论是否推迟上线",
            "h2",
        ),
        Message(
            "m3",
            "s",
            3,
            "2026-01-01 11:00:00",
            datetime(2026, 1, 1, 1, tzinfo=UTC),
            "u3",
            "丙",
            "测试仍然没有完成",
            "h3",
        ),
    ]
    windows = [build_windows([message])[0] for message in messages]
    store.upsert_messages(messages)
    store.upsert_windows(windows)
    return [window.window_id for window in windows]


def test_lexical_query_rejects_empty_and_punctuation_only() -> None:
    with pytest.raises(ValueError, match="meaningful"):
        lexical_terms("？！...")
    assert lexical_terms(" ABC-123 上线 ")


def test_rrf_has_deterministic_tie_breaking() -> None:
    scores = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], constant=60)
    assert list(scores) == ["a", "b"]
    assert scores["a"] == scores["b"]


def test_sqlite_lexical_search_finds_identifier_and_chinese_fragment(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path / "app.db") as store:
        ids = stored_windows(store)
        assert store.lexical_search("ABC-123", 10)[0] == ids[0]
        assert store.lexical_search("推迟上线", 10)[0] == ids[1]


def test_hybrid_results_resolve_original_message_metadata(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path / "app.db") as store:
        ids = stored_windows(store)
        retriever = HybridRetriever(store, QueryVectors([ids[2], ids[0]]), QueryEmbedder())

        results = retriever.search("ABC-123", vector_limit=10, lexical_limit=10, limit=3)

        assert results[0].window.window_id == ids[0]
        assert results[0].messages[0].message_id == "m1"
        assert results[0].messages[0].name == "甲"
        assert any(result.window.window_id == ids[2] for result in results)

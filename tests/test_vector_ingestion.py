import json
from pathlib import Path

import pytest

from chat_rag.embedding_client import EmbeddingProvider
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.vector_ingestion import IndexIdentityError, ingest_vectors
from chat_rag.vector_store import VectorStore


class FakeEmbedder(EmbeddingProvider):
    def __init__(self, dimension: int, fail_call: int | None = None) -> None:
        self.dimension = dimension
        self.fail_call = fail_call
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self.fail_call == len(self.calls):
            raise RuntimeError("interrupted")
        return [[float(index + 1)] * self.dimension for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [1.0] * self.dimension


class FakeVectors(VectorStore):
    def __init__(self) -> None:
        self.rows: dict[str, list[float]] = {}

    def upsert(self, rows: list[tuple[str, list[float]]]) -> None:
        self.rows.update(rows)

    def search(self, vector: list[float], limit: int):
        return []

    def count(self) -> int:
        return len(self.rows)

    def clear(self) -> None:
        self.rows.clear()


def source_file(path: Path, count: int = 5) -> Path:
    rows = [
        {"time": f"2026-01-01 10:0{i}:00", "uid": str(i), "name": "用户", "text": "消息" * 20}
        for i in range(count)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_unchanged_reingestion_makes_zero_embedding_calls(tmp_path: Path) -> None:
    source = source_file(tmp_path / "messages.jsonl")
    vectors = FakeVectors()
    with SQLiteStore(tmp_path / "app.db") as store:
        first = FakeEmbedder(3)
        report = ingest_vectors(
            source,
            store,
            vectors,
            first,
            model="embedding-v1",
            dimension=3,
            batch_size=2,
            target_tokens=50,
            max_tokens=80,
        )
        assert report.embedded_windows > 0

        second = FakeEmbedder(3)
        repeated = ingest_vectors(
            source,
            store,
            vectors,
            second,
            model="embedding-v1",
            dimension=3,
            batch_size=2,
            target_tokens=50,
            max_tokens=80,
        )
        assert repeated.embedded_windows == 0
        assert second.calls == []


def test_interrupted_batches_resume_and_identity_change_requires_rebuild(tmp_path: Path) -> None:
    source = source_file(tmp_path / "messages.jsonl", count=7)
    vectors = FakeVectors()
    with SQLiteStore(tmp_path / "app.db") as store:
        failing = FakeEmbedder(3, fail_call=2)
        with pytest.raises(RuntimeError, match="interrupted"):
            ingest_vectors(
                source,
                store,
                vectors,
                failing,
                model="embedding-v1",
                dimension=3,
                batch_size=1,
                target_tokens=50,
                max_tokens=80,
            )
        completed = vectors.count()
        assert completed == 1

        resumed = FakeEmbedder(3)
        result = ingest_vectors(
            source,
            store,
            vectors,
            resumed,
            model="embedding-v1",
            dimension=3,
            batch_size=1,
            target_tokens=50,
            max_tokens=80,
        )
        assert result.embedded_windows > 0
        assert vectors.count() > completed

        with pytest.raises(IndexIdentityError, match="rebuild"):
            ingest_vectors(
                source,
                store,
                vectors,
                FakeEmbedder(4),
                model="embedding-v2",
                dimension=4,
                rebuild=False,
            )

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from chat_rag.embedding_client import EmbeddingProvider
from chat_rag.ingest import iter_messages
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.vector_store import VectorStore
from chat_rag.windowing import build_windows


class IndexIdentityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IngestionReport:
    message_count: int
    window_count: int
    embedded_windows: int
    malformed_rows: int


def ingest_vectors(
    path: Path,
    store: SQLiteStore,
    vectors: VectorStore,
    embedder: EmbeddingProvider,
    *,
    model: str,
    dimension: int,
    batch_size: int = 64,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_messages: int = 2,
    session_gap_minutes: int = 20,
    rebuild: bool = False,
) -> IngestionReport:
    identities = store.embedding_identities()
    requested_identity = (model, dimension)
    if identities and identities != {requested_identity}:
        if not rebuild:
            raise IndexIdentityError(
                "embedding model or dimension changed; rerun with --rebuild-vectors"
            )
        vectors.clear()
        store.clear_embeddings()

    message_iterator, stats = iter_messages(path)
    messages = list(message_iterator)
    windows = build_windows(
        messages,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_messages=overlap_messages,
        session_gap_minutes=session_gap_minutes,
    )
    store.upsert_messages(messages)
    store.upsert_windows(windows)

    pending = store.windows_needing_embedding(model, dimension)
    embedded = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        result = embedder.embed_documents([window.text for window in batch])
        if len(result) != len(batch) or any(len(vector) != dimension for vector in result):
            raise ValueError(
                f"embedding provider returned vectors with dimension other than {dimension}"
            )
        rows = [(window.window_id, vector) for window, vector in zip(batch, result, strict=True)]
        vectors.upsert(rows)
        store.mark_embedded(
            [window.window_id for window in batch], model, dimension, datetime.now(UTC)
        )
        embedded += len(batch)

    return IngestionReport(
        message_count=len(messages),
        window_count=len(windows),
        embedded_windows=embedded,
        malformed_rows=stats.malformed_rows,
    )

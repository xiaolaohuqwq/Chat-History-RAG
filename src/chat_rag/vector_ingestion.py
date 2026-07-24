from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from chat_rag.embedding_client import EmbeddingProvider
from chat_rag.ingest import iter_messages, source_fingerprint
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.vector_store import VectorStore
from chat_rag.windowing import iter_windows


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
    persistence_batch_size: int = 500,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_messages: int = 2,
    session_gap_minutes: int = 20,
    rebuild: bool = False,
) -> IngestionReport:
    if batch_size <= 0 or persistence_batch_size <= 0:
        raise ValueError("ingestion batch sizes must be positive")
    source_id = f"file:{path.resolve()}"
    run_id = store.start_ingestion_run(
        source_id,
        source_fingerprint(path),
        path.stat().st_size,
        model,
        dimension,
        datetime.now(UTC),
    )
    message_count = 0
    window_count = 0
    estimated_tokens = 0
    embedded = 0
    message_iterator, stats = iter_messages(path)

    def checkpoint(
        status: str = "running",
        *,
        error_summary: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        store.update_ingestion_run(
            run_id,
            last_completed_line=stats.total_rows,
            message_count=message_count,
            window_count=window_count,
            malformed_count=stats.malformed_rows,
            estimated_tokens=estimated_tokens,
            status=status,
            error_summary=error_summary,
            completed_at=completed_at,
        )

    def persisted_messages():
        nonlocal message_count
        batch = []
        for message in message_iterator:
            batch.append(message)
            if len(batch) >= persistence_batch_size:
                store.upsert_messages(batch)
                message_count += len(batch)
                checkpoint()
                yield from batch
                batch = []
        if batch:
            store.upsert_messages(batch)
            message_count += len(batch)
            checkpoint()
            yield from batch

    def embed_pending_batch() -> int:
        batch = store.windows_needing_embedding(model, dimension, limit=batch_size)
        if not batch:
            return 0
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
        return len(batch)

    try:
        identities = store.embedding_identities()
        requested_identity = (model, dimension)
        if rebuild:
            vectors.clear()
            store.clear_embeddings()
        elif identities and identities != {requested_identity}:
            raise IndexIdentityError(
                "embedding model or dimension changed; rerun with --rebuild-vectors"
            )

        store.begin_window_refresh()

        windows = iter_windows(
            persisted_messages(),
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_messages=overlap_messages,
            session_gap_minutes=session_gap_minutes,
        )
        window_batch = []
        for window in windows:
            window_batch.append(window)
            window_count += 1
            estimated_tokens += window.estimated_tokens
            if len(window_batch) >= persistence_batch_size:
                store.upsert_windows(window_batch)
                store.record_current_windows([window.window_id for window in window_batch])
                window_batch = []
                embedded += embed_pending_batch()
                checkpoint()
        if window_batch:
            store.upsert_windows(window_batch)
            store.record_current_windows([window.window_id for window in window_batch])
            embedded += embed_pending_batch()
            checkpoint()
        vectors.delete(store.finish_window_refresh(source_id))
        while completed := embed_pending_batch():
            embedded += completed
            checkpoint()
    except Exception as error:
        checkpoint("failed", error_summary=type(error).__name__, completed_at=datetime.now(UTC))
        raise

    checkpoint("completed", completed_at=datetime.now(UTC))
    return IngestionReport(message_count, window_count, embedded, stats.malformed_rows)

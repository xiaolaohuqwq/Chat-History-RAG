from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from chat_rag.config import Settings
from chat_rag.embedding_client import DashScopeEmbeddingClient
from chat_rag.ingest import analyze_jsonl
from chat_rag.retrieval import HybridRetriever
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.vector_ingestion import ingest_vectors
from chat_rag.vector_store import LanceVectorStore

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.callback()
def main() -> None:
    """Search and analyze a local chat-history index."""


@app.command()
def ingest(
    path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    rebuild_vectors: Annotated[bool, typer.Option("--rebuild-vectors")] = False,
) -> None:
    """Analyze or ingest a preprocessed chat-history JSONL file."""
    settings = Settings()
    if not dry_run:
        try:
            api_key = settings.require_embedding()
            data_dir = Path(settings.data_dir)
            with SQLiteStore(data_dir / "app.db") as store:
                report = ingest_vectors(
                    path,
                    store,
                    LanceVectorStore(data_dir / "vectors", settings.embedding_dimension),
                    DashScopeEmbeddingClient(
                        api_key, settings.embedding_model, settings.embedding_dimension
                    ),
                    model=settings.embedding_model,
                    dimension=settings.embedding_dimension,
                    batch_size=settings.embed_batch_size,
                    target_tokens=settings.window_target_tokens,
                    max_tokens=settings.window_max_tokens,
                    overlap_messages=settings.window_overlap_messages,
                    session_gap_minutes=settings.session_gap_minutes,
                    rebuild=rebuild_vectors,
                )
        except (ValueError, RuntimeError) as error:
            raise typer.ClickException(str(error)) from None
        typer.echo(f"messages: {report.message_count:,}")
        typer.echo(f"windows: {report.window_count:,}")
        typer.echo(f"embedded windows: {report.embedded_windows:,}")
        return
    report = analyze_jsonl(
        path,
        target_tokens=settings.window_target_tokens,
        max_tokens=settings.window_max_tokens,
        overlap_messages=settings.window_overlap_messages,
        session_gap_minutes=settings.session_gap_minutes,
        embedding_dimension=settings.embedding_dimension,
    )
    typer.echo(f"valid rows: {report.rows.valid_rows:,}")
    typer.echo(f"malformed rows: {report.rows.malformed_rows:,}")
    typer.echo(f"empty rows: {report.rows.empty_rows:,}")
    typer.echo(f"duplicate rows: {report.rows.duplicate_rows:,}")
    typer.echo(f"windows: {report.window_count:,}")
    typer.echo(f"estimated tokens: {report.estimated_tokens:,}")
    typer.echo(f"estimated cost: CNY {report.estimated_cost_cny:.6f}")
    typer.echo(f"estimated vector storage: {report.estimated_vector_bytes:,} bytes")


@app.command()
def search(
    query: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 20,
    no_rerank: Annotated[bool, typer.Option("--no-rerank")] = False,
) -> None:
    """Run hybrid retrieval and print traceable source messages."""
    del no_rerank
    settings = Settings()
    try:
        api_key = settings.require_embedding()
        data_dir = Path(settings.data_dir)
        with SQLiteStore(data_dir / "app.db") as store:
            retriever = HybridRetriever(
                store,
                LanceVectorStore(data_dir / "vectors", settings.embedding_dimension),
                DashScopeEmbeddingClient(
                    api_key, settings.embedding_model, settings.embedding_dimension
                ),
            )
            results = retriever.search(
                query,
                vector_limit=settings.vector_top_k_per_query,
                lexical_limit=settings.lexical_top_k_per_query,
                limit=limit,
            )
    except (ValueError, RuntimeError) as error:
        raise typer.ClickException(str(error)) from None
    for result in results:
        typer.echo(f"{result.window.window_id} score={result.score:.6f}")
        for message in result.messages:
            typer.echo(
                f"  [{message.message_id} | {message.time_raw} | {message.name}({message.uid})] "
                f"{message.text}"
            )


@app.command("inspect")
def inspect_item(item_id: Annotated[str, typer.Argument()]) -> None:
    """Inspect one original message or retrieval window by ID."""
    settings = Settings()
    with SQLiteStore(Path(settings.data_dir) / "app.db") as store:
        message = store.get_message(item_id)
        if message is not None:
            typer.echo(
                f"[{message.message_id} | {message.time_raw} | {message.name}({message.uid})]"
            )
            typer.echo(message.text)
            return
        window = store.get_window(item_id)
        if window is not None:
            typer.echo(f"[{window.window_id} | lines {window.start_line}-{window.end_line}]")
            typer.echo(window.text)
            return
    raise typer.ClickException(f"no message or window found for {item_id}")


@app.command()
def stats() -> None:
    """Show index metadata and counts without exposing chat text."""
    settings = Settings()
    data_dir = Path(settings.data_dir)
    with SQLiteStore(data_dir / "app.db") as store:
        typer.echo(f"messages: {store.count('messages'):,}")
        typer.echo(f"windows: {store.count('windows'):,}")
        identities = sorted(store.embedding_identities())
        identity = ", ".join(f"{model}/{dimension}" for model, dimension in identities) or "none"
        typer.echo(f"index identity: {identity}")
    vectors = LanceVectorStore(data_dir / "vectors", settings.embedding_dimension)
    typer.echo(f"vectors: {vectors.count():,}")


if __name__ == "__main__":
    app()

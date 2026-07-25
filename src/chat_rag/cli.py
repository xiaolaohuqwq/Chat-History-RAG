from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from chat_rag.config import Settings
from chat_rag.embedding_client import DashScopeEmbeddingClient
from chat_rag.evaluation import evaluate, iter_eval_cases
from chat_rag.ingest import analyze_jsonl
from chat_rag.llm_client import OpenAICompatibleClient
from chat_rag.rerank_client import DashScopeReranker
from chat_rag.retrieval import HybridRetriever
from chat_rag.rpc import StdioServer
from chat_rag.rpc_app import RpcApplication
from chat_rag.service import ChatRAGService
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.vector_ingestion import ingest_vectors
from chat_rag.vector_store import LanceVectorStore

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


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
                        api_key,
                        settings.embedding_model,
                        settings.embedding_dimension,
                        concurrency=settings.embedding_concurrency,
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
            _fail(str(error))
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
    settings = Settings()
    try:
        api_key = settings.require_embedding()
        data_dir = Path(settings.data_dir)
        with SQLiteStore(data_dir / "app.db") as store:
            retriever = HybridRetriever(
                store,
                LanceVectorStore(data_dir / "vectors", settings.embedding_dimension),
                DashScopeEmbeddingClient(
                    api_key,
                    settings.embedding_model,
                    settings.embedding_dimension,
                    concurrency=settings.embedding_concurrency,
                ),
                reranker=None if no_rerank else DashScopeReranker(api_key, settings.rerank_model),
                rerank_candidates=settings.rerank_candidates,
            )
            results = retriever.search(
                query,
                vector_limit=settings.vector_top_k_per_query,
                lexical_limit=settings.lexical_top_k_per_query,
                limit=limit,
            )
            degraded_reason = retriever.degraded_reason
    except (ValueError, RuntimeError) as error:
        _fail(str(error))
    if degraded_reason:
        typer.echo(f"warning: {degraded_reason}; using fused retrieval scores", err=True)
    for result in results:
        typer.echo(f"{result.window.window_id} score={result.score:.6f}")
        for message in result.messages:
            typer.echo(
                f"  [{message.message_id} | {message.time_raw} | {message.name}({message.uid})] "
                f"{message.text}"
            )


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    no_rerank: Annotated[bool, typer.Option("--no-rerank")] = False,
) -> None:
    """Answer a question using retrieved and citation-checked evidence."""
    settings = Settings()
    try:
        embedding_key = settings.require_embedding()
        llm_key, base_url, model = settings.require_llm()
        data_dir = Path(settings.data_dir)
        with SQLiteStore(data_dir / "app.db") as store:
            retriever = HybridRetriever(
                store,
                LanceVectorStore(data_dir / "vectors", settings.embedding_dimension),
                DashScopeEmbeddingClient(
                    embedding_key,
                    settings.embedding_model,
                    settings.embedding_dimension,
                    concurrency=settings.embedding_concurrency,
                ),
                reranker=None
                if no_rerank
                else DashScopeReranker(embedding_key, settings.rerank_model),
                rerank_candidates=settings.rerank_candidates,
            )
            llm = OpenAICompatibleClient(
                llm_key,
                base_url,
                model,
                max_output_tokens=settings.llm_max_output_tokens,
            )
            result = ChatRAGService(
                store,
                retriever,
                llm,
                max_input_tokens=settings.llm_max_input_tokens,
                final_evidence_blocks=settings.final_evidence_blocks,
            ).ask(question)
    except (ValueError, RuntimeError) as error:
        _fail(str(error))
    typer.echo(result.answer)
    if result.degraded_reason:
        typer.echo(f"warning: {result.degraded_reason}", err=True)
    if result.citation_warning:
        typer.echo(f"warning: {result.citation_warning}", err=True)


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
    _fail(f"no message or window found for {item_id}")


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
        full_identity = store.index_identity()
        if full_identity is not None:
            typer.echo(f"normalization version: {full_identity[2]}")
            typer.echo(f"windowing version: {full_identity[3]}")
        run = store.latest_ingestion_run()
        if run is not None:
            typer.echo(f"source fingerprint: {run['source_fingerprint']}")
            typer.echo(f"last ingestion status: {run['status']}")
            typer.echo(f"malformed rows: {run['malformed_count']:,}")
            typer.echo(f"estimated API spend: CNY {run['estimated_cost_cny']:.6f}")
    vectors = LanceVectorStore(data_dir / "vectors", settings.embedding_dimension)
    typer.echo(f"vectors: {vectors.count():,}")


@app.command()
def reset(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Delete without interactive confirmation."),
    ] = False,
) -> None:
    """Permanently delete the current local message and vector index."""
    data_dir = Path(Settings().data_dir).expanduser().resolve()
    if not yes and not typer.confirm(
        f"Permanently delete indexed data from {data_dir}?", default=False
    ):
        typer.echo("Cancelled.")
        return

    database_paths = [
        data_dir / "app.db",
        data_dir / "app.db-wal",
        data_dir / "app.db-shm",
        data_dir / "app.db-journal",
    ]
    vectors_path = data_dir / "vectors"
    existed = any(path.exists() or path.is_symlink() for path in [*database_paths, vectors_path])
    try:
        for path in database_paths:
            path.unlink(missing_ok=True)
        if vectors_path.is_symlink() or vectors_path.is_file():
            vectors_path.unlink()
        elif vectors_path.is_dir():
            shutil.rmtree(vectors_path)
    except OSError as error:
        _fail(f"failed to reset indexed data: {error}")

    if existed:
        typer.echo(f"Deleted indexed data from {data_dir}")
    else:
        typer.echo(f"No indexed data found in {data_dir}")


@app.command("eval")
def evaluate_file(
    eval_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
) -> None:
    """Measure retrieval quality against a local evaluation JSONL file."""
    settings = Settings()
    try:
        embedding_key = settings.require_embedding()
        data_dir = Path(settings.data_dir)
        with SQLiteStore(data_dir / "app.db") as store:
            retriever = HybridRetriever(
                store,
                LanceVectorStore(data_dir / "vectors", settings.embedding_dimension),
                DashScopeEmbeddingClient(
                    embedding_key,
                    settings.embedding_model,
                    settings.embedding_dimension,
                    concurrency=settings.embedding_concurrency,
                ),
                reranker=DashScopeReranker(embedding_key, settings.rerank_model),
                rerank_candidates=settings.rerank_candidates,
            )
            report = evaluate(list(iter_eval_cases(eval_file)), retriever.search)
    except (ValueError, RuntimeError) as error:
        _fail(str(error))
    typer.echo(f"queries: {report.query_count}")
    typer.echo(f"Recall@20: {report.recall_at_20:.4f}")
    typer.echo(f"Recall@50: {report.recall_at_50:.4f}")
    typer.echo(f"MRR: {report.mean_reciprocal_rank:.4f}")
    typer.echo(f"multi-sender coverage: {report.multi_sender_rate:.4f}")
    typer.echo(f"multi-date coverage: {report.multi_date_rate:.4f}")
    typer.echo(f"local latency excluding cloud APIs: {report.local_latency_seconds:.3f}s")
    typer.echo(f"cloud API latency: {report.api_latency_seconds:.3f}s")
    typer.echo(f"latency including cloud APIs: {report.total_latency_seconds:.3f}s")


@app.command()
def serve(stdio: Annotated[bool, typer.Option("--stdio")] = False) -> None:
    """Run the versioned newline-delimited JSON protocol server."""
    if not stdio:
        _fail("serve currently requires --stdio")
    StdioServer(RpcApplication(Settings())).serve(sys.stdin, sys.stdout)


@app.command("smoke-embedding")
def smoke_embedding() -> None:
    """Embed three synthetic messages to verify cloud model configuration."""
    settings = Settings()
    try:
        client = DashScopeEmbeddingClient(
            settings.require_embedding(),
            settings.embedding_model,
            settings.embedding_dimension,
            concurrency=settings.embedding_concurrency,
        )
        vectors = client.embed_documents(
            [
                "合成消息：项目计划在下周评审。",
                "Synthetic message: deployment identifier DEMO-123.",
                "合成消息：测试尚未完成，建议延期。",
            ]
        )
    except (ValueError, RuntimeError) as error:
        _fail(str(error))
    typer.echo(f"model: {settings.embedding_model}")
    typer.echo(f"vectors: {len(vectors)}")
    typer.echo(f"dimension: {len(vectors[0]) if vectors else 0}")


if __name__ == "__main__":
    app()

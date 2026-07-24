from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from chat_rag.config import Settings
from chat_rag.ingest import analyze_jsonl

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
        del rebuild_vectors
        raise typer.BadParameter("real ingestion is not available yet; use --dry-run")
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


if __name__ == "__main__":
    app()

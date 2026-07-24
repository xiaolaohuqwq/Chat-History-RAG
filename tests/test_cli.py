import json
from pathlib import Path

from typer.testing import CliRunner

from chat_rag.cli import app


def test_ingest_dry_run_needs_no_credentials(tmp_path: Path) -> None:
    source = tmp_path / "messages.jsonl"
    source.write_text(
        json.dumps(
            {"time": "2026-01-01 10:00:00", "uid": "1", "name": "甲", "text": "测试"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["ingest", str(source), "--dry-run"])

    assert result.exit_code == 0
    assert "API" not in result.stdout
    assert "valid rows" in result.stdout
    assert "estimated cost" in result.stdout


def test_ingest_reports_missing_file_without_traceback(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["ingest", str(tmp_path / "missing.jsonl"), "--dry-run"])
    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert "Traceback" not in result.output

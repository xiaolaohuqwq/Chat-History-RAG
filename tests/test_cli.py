import json
from pathlib import Path

from typer.testing import CliRunner

from chat_rag.cli import app
from chat_rag.domain import Message
from chat_rag.sqlite_store import SQLiteStore


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


def test_stats_does_not_print_private_text(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    private_text = "private conversation content"
    with SQLiteStore(tmp_path / "app.db") as store:
        store.upsert_messages(
            [Message("m1", "s", 1, "bad-time", None, "u1", "甲", private_text, "hash")]
        )

    result = CliRunner().invoke(app, ["stats"])

    assert result.exit_code == 0
    assert "messages: 1" in result.stdout
    assert private_text not in result.stdout


def test_inspect_resolves_message_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with SQLiteStore(tmp_path / "app.db") as store:
        store.upsert_messages(
            [Message("m1", "s", 1, "bad-time", None, "u1", "甲", "原始内容", "hash")]
        )

    result = CliRunner().invoke(app, ["inspect", "m1"])

    assert result.exit_code == 0
    assert "甲" in result.stdout
    assert "原始内容" in result.stdout


def test_ask_reports_missing_configuration_without_traceback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    result = CliRunner().invoke(app, ["ask", "发生了什么？"])

    assert result.exit_code != 0
    assert "required" in result.output
    assert "Traceback" not in result.output

import json
from pathlib import Path

from chat_rag.ingest import analyze_jsonl, iter_messages


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_stream_parser_counts_malformed_empty_and_exact_duplicates(tmp_path: Path) -> None:
    valid = {"time": "2026-01-01 10:00:00", "uid": "1", "name": "甲", "text": "  内容 "}
    path = tmp_path / "sample.jsonl"
    write_lines(
        path,
        [
            json.dumps(valid, ensure_ascii=False),
            "{bad",
            json.dumps({**valid, "text": "  "}),
            json.dumps(valid, ensure_ascii=False),
        ],
    )

    messages, stats = iter_messages(path)
    rows = list(messages)

    assert [message.text for message in rows] == ["内容"]
    assert stats.valid_rows == 1
    assert stats.malformed_rows == 1
    assert stats.empty_rows == 1
    assert stats.duplicate_rows == 1


def test_dry_run_reports_windows_cost_and_storage_without_an_embedder(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    records = [
        {
            "time": f"2026-01-01 10:0{i}:00",
            "uid": str(i),
            "name": "用户",
            "text": "一段测试内容" * 8,
        }
        for i in range(4)
    ]
    write_lines(path, [json.dumps(record, ensure_ascii=False) for record in records])

    report = analyze_jsonl(path, target_tokens=30, max_tokens=50, embedding_dimension=1024)

    assert report.rows.valid_rows == 4
    assert report.window_count >= 1
    assert report.estimated_tokens > 0
    assert report.estimated_cost_cny == report.estimated_tokens * 0.5 / 1_000_000
    assert report.estimated_vector_bytes == report.window_count * 1024 * 4

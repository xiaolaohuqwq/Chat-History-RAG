import json
from datetime import UTC, datetime
from pathlib import Path

from chat_rag.domain import Message
from chat_rag.evaluation import evaluate, iter_eval_cases
from chat_rag.retrieval import RetrievalTiming, SearchResult
from chat_rag.windowing import build_windows


def result(message_id: str, sender: str, day: int) -> SearchResult:
    message = Message(
        message_id,
        "s",
        day,
        f"2026-01-{day:02d}",
        datetime(2026, 1, day, tzinfo=UTC),
        sender,
        sender,
        "synthetic",
        message_id,
    )
    return SearchResult(build_windows([message])[0], (message,), 1.0)


def test_eval_jsonl_and_retrieval_metrics(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"query": "原因", "relevant_message_ids": ["m2", "m9"], "notes": "synthetic"})
        + "\n",
        encoding="utf-8",
    )
    cases = list(iter_eval_cases(path))

    report = evaluate(cases, lambda query: [result("m1", "u1", 1), result("m2", "u2", 2)])

    assert report.query_count == 1
    assert report.recall_at_20 == 0.5
    assert report.recall_at_50 == 0.5
    assert report.mean_reciprocal_rank == 0.5
    assert report.multi_sender_rate == 1.0
    assert report.multi_date_rate == 1.0
    assert report.total_latency_seconds >= 0


def test_evaluation_reports_local_and_api_latency_separately(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"query": "原因", "relevant_message_ids": ["m1"]}) + "\n",
        encoding="utf-8",
    )

    class TimedRetriever:
        last_timing = RetrievalTiming(local_seconds=0.1, api_seconds=0.2, total_seconds=0.3)

        def search(self, query: str) -> list[SearchResult]:
            return [result("m1", "u1", 1)]

    report = evaluate(list(iter_eval_cases(path)), TimedRetriever().search)
    assert report.local_latency_seconds == 0.1
    assert report.api_latency_seconds == 0.2
    assert report.total_latency_seconds == 0.3

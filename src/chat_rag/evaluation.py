from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from chat_rag.retrieval import SearchResult


@dataclass(frozen=True, slots=True)
class EvalCase:
    query: str
    relevant_message_ids: frozenset[str]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    query_count: int
    recall_at_20: float
    recall_at_50: float
    mean_reciprocal_rank: float
    multi_sender_rate: float
    multi_date_rate: float
    local_latency_seconds: float
    api_latency_seconds: float
    total_latency_seconds: float


def iter_eval_cases(path: Path) -> Iterator[EvalCase]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                payload = json.loads(line)
                query = payload["query"]
                relevant = payload["relevant_message_ids"]
                if not isinstance(query, str) or not query.strip():
                    raise ValueError
                if (
                    not isinstance(relevant, list)
                    or not relevant
                    or any(not isinstance(item, str) for item in relevant)
                ):
                    raise ValueError
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid evaluation record at line {line_number}") from error
            yield EvalCase(query.strip(), frozenset(relevant), str(payload.get("notes", "")))


def _message_ids(results: list[SearchResult], limit: int) -> list[str]:
    ids: list[str] = []
    for result in results[:limit]:
        for message in result.messages:
            if message.message_id not in ids:
                ids.append(message.message_id)
    return ids


def evaluate(
    cases: list[EvalCase],
    retrieve: Callable[[str], list[SearchResult]],
) -> EvaluationReport:
    if not cases:
        raise ValueError("evaluation file contains no cases")
    recall_20 = 0.0
    recall_50 = 0.0
    reciprocal_rank = 0.0
    multi_sender = 0
    multi_date = 0
    local_latency = 0.0
    api_latency = 0.0
    total_latency = 0.0
    for case in cases:
        started = time.perf_counter()
        results = retrieve(case.query)
        elapsed = time.perf_counter() - started
        owner = getattr(retrieve, "__self__", None)
        timing = getattr(owner, "last_timing", None)
        if timing is None:
            local_latency += elapsed
            total_latency += elapsed
        else:
            local_latency += timing.local_seconds
            api_latency += timing.api_seconds
            total_latency += timing.total_seconds
        ids_20 = _message_ids(results, 20)
        ids_50 = _message_ids(results, 50)
        recall_20 += len(case.relevant_message_ids & set(ids_20)) / len(case.relevant_message_ids)
        recall_50 += len(case.relevant_message_ids & set(ids_50)) / len(case.relevant_message_ids)
        for rank, message_id in enumerate(ids_50, start=1):
            if message_id in case.relevant_message_ids:
                reciprocal_rank += 1 / rank
                break
        senders = {message.uid for result in results[:50] for message in result.messages}
        dates = {
            message.time_utc.date()
            for result in results[:50]
            for message in result.messages
            if message.time_utc is not None
        }
        multi_sender += len(senders) > 1
        multi_date += len(dates) > 1
    count = len(cases)
    return EvaluationReport(
        query_count=count,
        recall_at_20=recall_20 / count,
        recall_at_50=recall_50 / count,
        mean_reciprocal_rank=reciprocal_rank / count,
        multi_sender_rate=multi_sender / count,
        multi_date_rate=multi_date / count,
        local_latency_seconds=local_latency,
        api_latency_seconds=api_latency,
        total_latency_seconds=total_latency,
    )

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from chat_rag.domain import DryRunReport, Message, ParseStats
from chat_rag.normalize import content_hash, make_message_id, normalize_text, parse_time
from chat_rag.windowing import build_windows

REQUIRED_FIELDS = ("time", "uid", "name", "text")


def _source_id(path: Path) -> str:
    return f"file:{path.resolve()}"


def iter_messages(path: Path) -> tuple[Iterator[Message], ParseStats]:
    stats = ParseStats()
    source_id = _source_id(path)

    def generate() -> Iterator[Message]:
        seen: set[tuple[str, str, str]] = set()
        with path.open(encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                stats.total_rows += 1
                try:
                    record: Any = json.loads(raw_line)
                    if not isinstance(record, dict) or any(
                        field not in record or not isinstance(record[field], str)
                        for field in REQUIRED_FIELDS
                    ):
                        raise ValueError("invalid message schema")
                except (json.JSONDecodeError, ValueError):
                    stats.malformed_rows += 1
                    continue
                text = normalize_text(record["text"])
                if not text:
                    stats.empty_rows += 1
                    continue
                duplicate_key = (record["uid"], record["time"], text)
                if duplicate_key in seen:
                    stats.duplicate_rows += 1
                    continue
                seen.add(duplicate_key)
                parsed_time = parse_time(record["time"])
                if parsed_time is None:
                    stats.invalid_time_rows += 1
                stats.valid_rows += 1
                yield Message(
                    message_id=make_message_id(
                        source_id, line_number, record["uid"], record["time"], text
                    ),
                    source_id=source_id,
                    source_line=line_number,
                    time_raw=record["time"],
                    time_utc=parsed_time,
                    uid=record["uid"],
                    name=record["name"],
                    text=text,
                    content_hash=content_hash(text),
                )

    return generate(), stats


def analyze_jsonl(
    path: Path,
    *,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_messages: int = 2,
    session_gap_minutes: int = 20,
    embedding_dimension: int = 1024,
) -> DryRunReport:
    messages, stats = iter_messages(path)
    windows = build_windows(
        messages,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_messages=overlap_messages,
        session_gap_minutes=session_gap_minutes,
    )
    estimated_tokens = sum(window.estimated_tokens for window in windows)
    return DryRunReport(
        rows=stats,
        window_count=len(windows),
        estimated_tokens=estimated_tokens,
        estimated_cost_cny=estimated_tokens * 0.5 / 1_000_000,
        estimated_vector_bytes=len(windows) * embedding_dimension * 4,
    )

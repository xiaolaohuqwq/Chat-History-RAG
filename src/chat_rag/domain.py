from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Message:
    message_id: str
    source_id: str
    source_line: int
    time_raw: str
    time_utc: datetime | None
    uid: str
    name: str
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class Window:
    window_id: str
    source_id: str
    start_line: int
    end_line: int
    start_time: datetime | None
    end_time: datetime | None
    text: str
    estimated_tokens: int
    content_hash: str
    windowing_version: str
    message_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class ParseStats:
    total_rows: int = 0
    valid_rows: int = 0
    malformed_rows: int = 0
    empty_rows: int = 0
    duplicate_rows: int = 0
    invalid_time_rows: int = 0


@dataclass(frozen=True, slots=True)
class DryRunReport:
    rows: ParseStats
    window_count: int
    estimated_tokens: int
    estimated_cost_cny: float
    estimated_vector_bytes: int

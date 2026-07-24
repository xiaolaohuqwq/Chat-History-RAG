from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import timedelta

from chat_rag.domain import Message, Window
from chat_rag.token_estimator import estimate_tokens

WINDOWING_VERSION = "v1"


def format_message(message: Message) -> str:
    timestamp = message.time_raw or "unknown-time"
    return f"[{message.message_id} | {timestamp} | {message.name}({message.uid})] {message.text}"


def _make_window(messages: list[Message]) -> Window:
    text = "\n".join(format_message(message) for message in messages)
    digest = hashlib.sha256(f"{WINDOWING_VERSION}\0{text}".encode()).hexdigest()
    return Window(
        window_id=f"w_{digest[:24]}",
        source_id=messages[0].source_id,
        start_line=messages[0].source_line,
        end_line=messages[-1].source_line,
        start_time=messages[0].time_utc,
        end_time=messages[-1].time_utc,
        text=text,
        estimated_tokens=estimate_tokens(text),
        content_hash=digest,
        windowing_version=WINDOWING_VERSION,
        message_ids=tuple(message.message_id for message in messages),
    )


def _session_break(previous: Message, current: Message, gap_minutes: int) -> bool:
    if previous.time_utc is None or current.time_utc is None:
        return False
    return current.time_utc - previous.time_utc > timedelta(minutes=gap_minutes)


def build_windows(
    messages: Iterable[Message],
    *,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_messages: int = 2,
    session_gap_minutes: int = 20,
) -> list[Window]:
    if target_tokens <= 0 or max_tokens < target_tokens:
        raise ValueError("window token limits are inconsistent")
    windows: list[Window] = []
    current: list[Message] = []

    def flush(*, carry_overlap: bool) -> None:
        nonlocal current
        if not current:
            return
        windows.append(_make_window(current))
        current = current[-overlap_messages:] if carry_overlap and overlap_messages else []

    for message in messages:
        if current and _session_break(current[-1], message, session_gap_minutes):
            flush(carry_overlap=False)
        candidate = [*current, message]
        candidate_tokens = estimate_tokens("\n".join(format_message(item) for item in candidate))
        if current and candidate_tokens > max_tokens:
            flush(carry_overlap=True)
            while (
                current
                and estimate_tokens("\n".join(format_message(item) for item in [*current, message]))
                > max_tokens
            ):
                current.pop(0)
        current.append(message)
        if estimate_tokens("\n".join(format_message(item) for item in current)) >= target_tokens:
            flush(carry_overlap=True)
    flush(carry_overlap=False)
    return windows

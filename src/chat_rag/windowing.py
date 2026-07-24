from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import replace
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
        message_ids=tuple(dict.fromkeys(message.message_id for message in messages)),
    )


def _session_break(previous: Message, current: Message, gap_minutes: int) -> bool:
    if previous.time_utc is None or current.time_utc is None:
        return False
    return current.time_utc - previous.time_utc > timedelta(minutes=gap_minutes)


def _preferred_cut(text: str, maximum: int) -> int:
    lower_bound = max(1, maximum // 2)
    prefix = text[:maximum]
    candidates = [
        prefix.rfind("\n\n"),
        prefix.rfind("\n"),
        max((match.end() for match in re.finditer(r"[。！？.!?]", prefix)), default=-1),
        max((match.end() for match in re.finditer(r"\s", prefix)), default=-1),
    ]
    return next((cut for cut in candidates if cut >= lower_bound), maximum)


def _split_long_message(message: Message, max_tokens: int) -> list[Message]:
    if estimate_tokens(format_message(message)) <= max_tokens:
        return [message]
    pieces: list[Message] = []
    remaining = message.text
    while remaining:
        low, high = 1, len(remaining)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            candidate = replace(message, text=remaining[:middle])
            if estimate_tokens(format_message(candidate)) <= max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise ValueError("message metadata alone exceeds the configured window maximum")
        cut = _preferred_cut(remaining, best)
        pieces.append(replace(message, text=remaining[:cut]))
        remaining = remaining[cut:]
    return pieces


def iter_windows(
    messages: Iterable[Message],
    *,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_messages: int = 2,
    session_gap_minutes: int = 20,
) -> Iterable[Window]:
    if target_tokens <= 0 or max_tokens < target_tokens:
        raise ValueError("window token limits are inconsistent")
    current: list[Message] = []
    new_since_flush = 0

    for message in messages:
        if current and _session_break(current[-1], message, session_gap_minutes):
            if new_since_flush:
                yield _make_window(current)
            current = []
            new_since_flush = 0
        for chunk in _split_long_message(message, max_tokens):
            candidate = [*current, chunk]
            candidate_tokens = estimate_tokens(
                "\n".join(format_message(item) for item in candidate)
            )
            if current and candidate_tokens > max_tokens:
                if new_since_flush:
                    yield _make_window(current)
                current = current[-overlap_messages:] if overlap_messages else []
                new_since_flush = 0
                while (
                    current
                    and estimate_tokens(
                        "\n".join(format_message(item) for item in [*current, chunk])
                    )
                    > max_tokens
                ):
                    current.pop(0)
            current.append(chunk)
            new_since_flush += 1
            if (
                estimate_tokens("\n".join(format_message(item) for item in current))
                >= target_tokens
            ):
                yield _make_window(current)
                current = current[-overlap_messages:] if overlap_messages else []
                new_since_flush = 0
    if current and new_since_flush:
        yield _make_window(current)


def build_windows(
    messages: Iterable[Message],
    *,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_messages: int = 2,
    session_gap_minutes: int = 20,
) -> list[Window]:
    return list(
        iter_windows(
            messages,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_messages=overlap_messages,
            session_gap_minutes=session_gap_minutes,
        )
    )

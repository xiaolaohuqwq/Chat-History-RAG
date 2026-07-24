from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from chat_rag.domain import Message
from chat_rag.retrieval import SearchResult
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.token_estimator import estimate_tokens


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    evidence_id: str
    source_id: str
    start_line: int
    end_line: int
    messages: tuple[Message, ...]
    relevance: float


@dataclass(frozen=True, slots=True)
class PackedEvidence:
    text: str
    message_ids: tuple[str, ...]
    estimated_tokens: int


def _same_session(left: Message, right: Message, gap_minutes: int) -> bool:
    if left.time_utc is None or right.time_utc is None:
        return True
    return abs(right.time_utc - left.time_utc) <= timedelta(minutes=gap_minutes)


def build_evidence_blocks(
    store: SQLiteStore,
    results: list[SearchResult],
    *,
    adjacent_messages: int = 2,
    session_gap_minutes: int = 20,
) -> list[EvidenceBlock]:
    ranges: list[tuple[str, int, int, float]] = []
    by_source: dict[str, list[Message]] = {}
    for result in results:
        source_id = result.window.source_id
        source_messages = by_source.setdefault(source_id, store.get_source_messages(source_id))
        positions = {message.message_id: index for index, message in enumerate(source_messages)}
        seed_positions = [
            positions[message_id]
            for message_id in result.window.message_ids
            if message_id in positions
        ]
        if not seed_positions:
            continue
        start = min(seed_positions)
        end = max(seed_positions)
        for _ in range(adjacent_messages):
            if start > 0 and _same_session(
                source_messages[start - 1], source_messages[start], session_gap_minutes
            ):
                start -= 1
            if end + 1 < len(source_messages) and _same_session(
                source_messages[end], source_messages[end + 1], session_gap_minutes
            ):
                end += 1
        ranges.append(
            (
                source_id,
                source_messages[start].source_line,
                source_messages[end].source_line,
                result.score,
            )
        )

    merged: list[tuple[str, int, int, float]] = []
    for source_id, start, end, relevance in sorted(ranges, key=lambda item: (item[0], item[1])):
        if merged and merged[-1][0] == source_id and start <= merged[-1][2] + 1:
            previous = merged[-1]
            merged[-1] = (
                source_id,
                previous[1],
                max(previous[2], end),
                max(previous[3], relevance),
            )
        else:
            merged.append((source_id, start, end, relevance))

    blocks: list[EvidenceBlock] = []
    for index, (source_id, start, end, relevance) in enumerate(merged, start=1):
        messages = tuple(
            message for message in by_source[source_id] if start <= message.source_line <= end
        )
        blocks.append(EvidenceBlock(f"e{index}", source_id, start, end, messages, relevance))
    return sorted(blocks, key=lambda block: (-block.relevance, block.start_line))


def _format_block(block: EvidenceBlock, messages: tuple[Message, ...] | None = None) -> str:
    selected = messages if messages is not None else block.messages
    if not selected:
        return ""
    start = selected[0].time_raw
    end = selected[-1].time_raw
    lines = [f'<evidence id="{block.evidence_id}" start="{start}" end="{end}">']
    lines.extend(f"[{message.message_id} | {message.name}] {message.text}" for message in selected)
    lines.append("</evidence>")
    return "\n".join(lines)


def pack_evidence(blocks: list[EvidenceBlock], *, max_tokens: int) -> PackedEvidence:
    rendered: list[str] = []
    message_ids: list[str] = []
    for block in blocks:
        accepted: list[Message] = []
        for message in block.messages:
            candidate_block = _format_block(block, tuple([*accepted, message]))
            candidate_text = "\n\n".join([*rendered, candidate_block])
            if estimate_tokens(candidate_text) > max_tokens:
                break
            accepted.append(message)
        if accepted:
            rendered.append(_format_block(block, tuple(accepted)))
            message_ids.extend(message.message_id for message in accepted)
    text = "\n\n".join(rendered)
    return PackedEvidence(text, tuple(message_ids), estimate_tokens(text) if text else 0)

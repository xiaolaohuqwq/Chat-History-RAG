from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidenceCard:
    topic: str
    source_ids: tuple[str, ...]
    period: str = ""
    claims: tuple[str, ...] = ()
    proposals: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    disagreements: tuple[str, ...] = ()
    uncertainty: str = ""


def validate_card_sources(card: EvidenceCard, allowed_ids: set[str]) -> None:
    invalid = set(card.source_ids) - allowed_ids
    if invalid:
        raise ValueError("evidence card cites message IDs outside its input batch")


_CITATION_GROUP_PATTERN = re.compile(
    r"\[([A-Za-z][A-Za-z0-9_-]*(?:\s*[,|]\s*[A-Za-z][A-Za-z0-9_-]*)*)\]"
)
_CITATION_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_EVIDENCE_ID_PATTERN = re.compile(r"e\d+")


def cited_ids(answer: str) -> set[str]:
    return {
        item
        for group in _CITATION_GROUP_PATTERN.findall(answer)
        for item in _CITATION_ID_PATTERN.findall(group)
    }


def invalid_citations(answer: str, allowed_ids: set[str]) -> set[str]:
    return {
        item
        for item in cited_ids(answer) - allowed_ids
        if _EVIDENCE_ID_PATTERN.fullmatch(item) is None
    }


def strip_citation_labels(answer: str) -> str:
    stripped = _CITATION_GROUP_PATTERN.sub("", answer)
    stripped = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", stripped)
    return stripped.strip()

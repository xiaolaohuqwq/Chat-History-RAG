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


_CITATION_PATTERN = re.compile(r"\[(m[A-Za-z0-9_-]*)\]")


def cited_ids(answer: str) -> set[str]:
    return set(_CITATION_PATTERN.findall(answer))


def invalid_citations(answer: str, allowed_ids: set[str]) -> set[str]:
    return cited_ids(answer) - allowed_ids

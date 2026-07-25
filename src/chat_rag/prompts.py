from __future__ import annotations

import json
from dataclasses import dataclass

PLANNER_SYSTEM_PROMPT = """Plan retrieval over chat history. Resolve follow-up references from
the supplied conversation, but do not treat earlier assistant claims as evidence. Return JSON only:
{"standalone_question":"...","intent":"...","subqueries":[{"purpose":"...","query":"..."}]}
Use three to five short, distinct subqueries. Choose an intent such as fact_lookup,
timeline, opinion_summary, cause_analysis, decision_history, comparison, or topic_summary.
Purposes should describe needed evidence such as fact, proposal, support, objection, cause,
decision, outcome, or later_change. Do not invent sender or date filters."""

FINAL_SYSTEM_PROMPT = """Answer using only the supplied evidence. Synthesize an answer to the
question instead of listing search matches. Distinguish proposals, personal opinions, temporary
decisions, final decisions, and observed outcomes when relevant. Cite every important claim using
exact [message_id] identifiers from the evidence. State material missing evidence and uncertainty
explicitly. Never infer consensus from repetition or silence, and never claim a proposal was
implemented without an outcome message. Use concise Chinese and choose a structure appropriate
for the question; omit irrelevant sections."""

_INTENT_GUIDANCE = {
    "fact_lookup": "Answer the requested fact directly and briefly before any qualification.",
    "timeline": "Organize the answer chronologically and separate events from later recollections.",
    "opinion_summary": "Group views by position and do not infer consensus from message volume.",
    "cause_analysis": (
        "Separate stated causes, inferred contributing factors, and unsupported guesses."
    ),
    "decision_history": "Separate proposals, temporary decisions, final decisions, and outcomes.",
    "comparison": "Compare the requested subjects on the same evidence-backed dimensions.",
    "topic_summary": "Synthesize themes and important changes instead of listing search matches.",
}

MAP_SYSTEM_PROMPT = """Summarize only the supplied evidence as one JSON EvidenceCard with
topic, period, claims, proposals, decisions, outcomes, disagreements, source_ids, and uncertainty.
Every source_ids value must be an exact message ID present in the input."""

CITATION_REPAIR_SYSTEM_PROMPT = """Repair invalid citations in the answer using only the
provided allowed message IDs and evidence. Do not add new factual claims. Return the full repaired
answer only."""


@dataclass(frozen=True, slots=True)
class RetrievalSubquery:
    purpose: str
    query: str


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    standalone_question: str
    intent: str
    subqueries: tuple[RetrievalSubquery, ...]


def answer_system_prompt(intent: str) -> str:
    guidance = _INTENT_GUIDANCE.get(intent, _INTENT_GUIDANCE["topic_summary"])
    return f"{FINAL_SYSTEM_PROMPT}\n\nQuestion-specific guidance: {guidance}"


def parse_retrieval_plan(raw: str) -> RetrievalPlan:
    try:
        payload = json.loads(raw)
        standalone_question = payload["standalone_question"]
        intent = payload["intent"]
        subqueries = payload["subqueries"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("retrieval planner must return the structured JSON contract") from error
    if not isinstance(standalone_question, str) or not standalone_question.strip():
        raise ValueError("standalone question must be a non-empty string")
    if not isinstance(intent, str) or not intent.strip() or len(intent) > 50:
        raise ValueError("retrieval intent must be a short string")
    if not isinstance(subqueries, list) or not 3 <= len(subqueries) <= 5:
        raise ValueError("retrieval plan must contain three to five subqueries")
    normalized: list[RetrievalSubquery] = []
    for item in subqueries:
        if not isinstance(item, dict):
            raise ValueError("retrieval subqueries must be objects")
        purpose = item.get("purpose")
        query = item.get("query")
        if (
            not isinstance(purpose, str)
            or not purpose.strip()
            or len(purpose) > 50
            or not isinstance(query, str)
            or not query.strip()
            or len(query) > 200
        ):
            raise ValueError("retrieval purposes and queries must be non-empty short strings")
        normalized.append(RetrievalSubquery(purpose.strip(), query.strip()))
    if len({item.query for item in normalized}) != len(normalized):
        raise ValueError("retrieval queries must be unique")
    return RetrievalPlan(standalone_question.strip(), intent.strip(), tuple(normalized))

from __future__ import annotations

import json

PLANNER_SYSTEM_PROMPT = """You plan retrieval over chat history. Return JSON only with a
single `queries` array containing four to eight short queries. Cover supporting and opposing
views, causes or risks, proposed actions, final decisions or outcomes, and later changes when
relevant. Do not invent sender, date, or other metadata filters not stated by the user."""

FINAL_SYSTEM_PROMPT = """Answer using only the supplied evidence. Distinguish proposals,
personal opinions, temporary decisions, final decisions, and observed outcomes. Present
supporting and opposing views and explain chronological changes when relevant. Cite every
important claim using exact [message_id] identifiers from the evidence. State missing evidence
and uncertainty explicitly. Never infer consensus from repetition or silence, and never claim a
proposal was implemented without an outcome message.

Use this structure:
结论
主要依据
不同意见
时间变化
不确定或缺失的信息
引用消息"""

MAP_SYSTEM_PROMPT = """Summarize only the supplied evidence as one JSON EvidenceCard with
topic, period, claims, proposals, decisions, outcomes, disagreements, source_ids, and uncertainty.
Every source_ids value must be an exact message ID present in the input."""

CITATION_REPAIR_SYSTEM_PROMPT = """Repair invalid citations in the answer using only the
provided allowed message IDs and evidence. Do not add new factual claims. Return the full repaired
answer only."""


def parse_retrieval_plan(raw: str) -> list[str]:
    try:
        payload = json.loads(raw)
        queries = payload["queries"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("retrieval planner must return a JSON queries array") from error
    if not isinstance(queries, list) or not 4 <= len(queries) <= 8:
        raise ValueError("retrieval plan must contain four to eight queries")
    if any(
        not isinstance(query, str) or not query.strip() or len(query) > 200 for query in queries
    ):
        raise ValueError("retrieval queries must be non-empty short strings")
    normalized = [query.strip() for query in queries]
    if len(set(normalized)) != len(normalized):
        raise ValueError("retrieval queries must be unique")
    return normalized

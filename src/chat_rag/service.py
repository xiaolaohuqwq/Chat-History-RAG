from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from chat_rag.context_builder import EvidenceBlock, build_evidence_blocks, pack_evidence
from chat_rag.evidence import (
    EvidenceCard,
    cited_ids,
    invalid_citations,
    strip_citation_labels,
    validate_card_sources,
)
from chat_rag.llm_client import LLMProvider
from chat_rag.prompts import (
    CITATION_REPAIR_SYSTEM_PROMPT,
    COMPACT_FINAL_SYSTEM_PROMPT,
    MAP_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    RetrievalPlan,
    answer_system_prompt,
    parse_retrieval_plan,
)
from chat_rag.retrieval import SearchResult
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.token_estimator import estimate_tokens

ProgressCallback = Callable[[str], None]


class Retriever(Protocol):
    degraded_reason: str | None

    def search(self, query: str, **kwargs: object) -> list[SearchResult]: ...


@dataclass(frozen=True, slots=True)
class AskResult:
    answer: str
    citations: tuple[str, ...]
    empty: bool = False
    used_map_reduce: bool = False
    degraded_reason: str | None = None
    citation_warning: str | None = None


_BROAD_MARKERS = (
    "为什么",
    "原因",
    "意见",
    "争议",
    "分歧",
    "决定",
    "结论",
    "最后",
    "结果",
    "变化",
    "后来",
    "why",
    "decision",
    "outcome",
    "change over time",
)

_LATEST_MARKERS = ("最新", "当前", "现在", "目前", "latest", "current", "recent")


class ChatRAGService:
    def __init__(
        self,
        store: SQLiteStore,
        retriever: Retriever,
        llm: LLMProvider,
        *,
        max_input_tokens: int = 140_000,
        hierarchical_threshold: int = 80_000,
        map_batch_tokens: int = 30_000,
        final_evidence_blocks: int = 30,
        progress: ProgressCallback | None = None,
        answer_delta: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.llm = llm
        self.max_input_tokens = max_input_tokens
        self.hierarchical_threshold = hierarchical_threshold
        self.map_batch_tokens = map_batch_tokens
        self.final_evidence_blocks = final_evidence_blocks
        self.progress = progress or (lambda _stage: None)
        self.answer_delta = answer_delta

    def ask(
        self, question: str, *, history: Sequence[Mapping[str, str]] = ()
    ) -> AskResult:
        standalone_question = question
        intent = "fact_lookup"
        subqueries: list[tuple[str, str]] = []
        rankings: list[tuple[str, list[SearchResult]]] = []
        if history:
            self.progress("planning")
            try:
                plan = self._plan_question(question, history)
            except (RuntimeError, ValueError):
                plan = None
            if plan is not None:
                standalone_question = plan.standalone_question
                intent = plan.intent
                subqueries = [(item.purpose, item.query) for item in plan.subqueries]
        else:
            self.progress("retrieval")
            initial = self.retriever.search(question)
            if not initial:
                return AskResult(
                    answer="未检索到可用于回答该问题的聊天记录。",
                    citations=(),
                    empty=True,
                    degraded_reason=self.retriever.degraded_reason,
                )
            rankings.append(("direct", initial))
            if self._is_broad(question):
                self.progress("planning")
                try:
                    plan = self._plan_question(question, history)
                except (RuntimeError, ValueError):
                    plan = None
                if plan is not None:
                    standalone_question = plan.standalone_question
                    intent = plan.intent
                    subqueries = [(item.purpose, item.query) for item in plan.subqueries]

        self.progress("retrieval")
        if not rankings or standalone_question != question:
            rankings.append(("direct", self.retriever.search(standalone_question)))
        for purpose, query in subqueries:
            if query != standalone_question:
                rankings.append((purpose, self.retriever.search(query)))
        all_results, result_roles = self._fuse_rankings(rankings)
        if not all_results:
            return AskResult(
                answer="未检索到可用于回答该问题的聊天记录。",
                citations=(),
                empty=True,
                degraded_reason=self.retriever.degraded_reason,
            )

        selected = self._select_diverse(
            all_results,
            self.final_evidence_blocks,
            prefer_recent=any(
                marker in standalone_question.lower() for marker in _LATEST_MARKERS
            ),
            result_roles=result_roles,
        )
        evidence_roles = self._evidence_roles(selected, result_roles)
        blocks = build_evidence_blocks(self.store, selected)
        full = pack_evidence(blocks, max_tokens=max(self.max_input_tokens * 10, 1_000_000))

        cards: list[EvidenceCard] = []
        used_map_reduce = full.estimated_tokens > self.hierarchical_threshold
        if used_map_reduce:
            self.progress("context_reduction")
            cards = self._map_evidence(blocks)

        self.progress("generation")
        answer, allowed_ids = self._final_answer(
            standalone_question, blocks, cards, intent, evidence_roles
        )
        invalid = invalid_citations(answer, allowed_ids)
        warning = None
        if invalid:
            repair_user = (
                f"Allowed IDs: {sorted(allowed_ids)}\n"
                f"Invalid IDs: {sorted(invalid)}\nOriginal answer:\n{answer}"
            )
            try:
                repaired = self._complete(CITATION_REPAIR_SYSTEM_PROMPT, repair_user)
            except RuntimeError:
                warning = "citation repair unavailable"
            else:
                remaining = invalid_citations(repaired, allowed_ids)
                if remaining:
                    warning = f"citation validation failed for: {', '.join(sorted(remaining))}"
                else:
                    answer = repaired

        citations = tuple(sorted(cited_ids(answer) & allowed_ids))
        return AskResult(
            answer=strip_citation_labels(answer),
            citations=citations,
            used_map_reduce=used_map_reduce,
            degraded_reason=self.retriever.degraded_reason,
            citation_warning=warning,
        )

    def _is_broad(self, question: str) -> bool:
        lowered = question.lower()
        return any(marker in lowered for marker in _BROAD_MARKERS)

    def _planner_input(
        self, question: str, history: Sequence[Mapping[str, str]]
    ) -> str:
        turns = [
            {"role": turn.get("role", ""), "content": turn.get("content", "")}
            for turn in history[-6:]
        ]
        return (
            f"Conversation:\n{json.dumps(turns, ensure_ascii=False)}\n\n"
            f"Current question:\n{question}"
        )

    def _plan_question(
        self, question: str, history: Sequence[Mapping[str, str]]
    ) -> RetrievalPlan:
        raw = self._complete(PLANNER_SYSTEM_PROMPT, self._planner_input(question, history))
        return parse_retrieval_plan(raw)

    def _fuse_rankings(
        self, rankings: list[tuple[str, list[SearchResult]]]
    ) -> tuple[list[SearchResult], dict[str, set[str]]]:
        results_by_id: dict[str, SearchResult] = {}
        scores: dict[str, float] = {}
        roles: dict[str, set[str]] = {}
        for purpose, results in rankings:
            weight = 2.0 if purpose == "direct" else 1.0
            for rank, result in enumerate(results, start=1):
                window_id = result.window.window_id
                results_by_id.setdefault(window_id, result)
                scores[window_id] = scores.get(window_id, 0.0) + weight / (60 + rank)
                roles.setdefault(window_id, set()).add(purpose)
        fused = [
            SearchResult(result.window, result.messages, scores[window_id])
            for window_id, result in results_by_id.items()
        ]
        return sorted(fused, key=lambda item: (-item.score, item.window.window_id)), roles

    def _evidence_roles(
        self, results: list[SearchResult], result_roles: Mapping[str, set[str]]
    ) -> dict[str, list[str]]:
        roles: dict[str, list[str]] = {}
        for result in results:
            message_ids = [message.message_id for message in result.messages]
            for role in sorted(result_roles.get(result.window.window_id, {"direct"})):
                bucket = roles.setdefault(role, [])
                bucket.extend(message_id for message_id in message_ids if message_id not in bucket)
        return roles

    def _complete(
        self, system: str, user: str, on_delta: Callable[[str], None] | None = None
    ) -> str:
        if estimate_tokens(f"{system}\n{user}") > self.max_input_tokens:
            raise ValueError("LLM request would exceed LLM_MAX_INPUT_TOKENS")
        return self.llm.complete(system, user, on_delta)

    def _select_diverse(
        self,
        results: list[SearchResult],
        limit: int,
        *,
        prefer_recent: bool = False,
        result_roles: Mapping[str, set[str]] | None = None,
    ) -> list[SearchResult]:
        unique: dict[str, SearchResult] = {}
        for result in results:
            existing = unique.get(result.window.window_id)
            if existing is None or result.score > existing.score:
                unique[result.window.window_id] = result
        remaining = list(unique.values())
        selected: list[SearchResult] = []
        seen_senders: set[str] = set()
        seen_days: set[str] = set()
        seen_roles: set[str] = set()
        newest_time = max(
            (
                message.time_utc
                for result in remaining
                for message in result.messages
                if message.time_utc is not None
            ),
            default=None,
        )
        while remaining and len(selected) < limit:

            def diversity_score(result: SearchResult) -> tuple[float, str]:
                senders = {message.uid for message in result.messages}
                days = {
                    message.time_utc.date().isoformat()
                    for message in result.messages
                    if message.time_utc is not None
                }
                overlap = max(
                    (
                        len(set(result.window.message_ids) & set(item.window.message_ids))
                        / max(1, len(result.window.message_ids))
                        for item in selected
                    ),
                    default=0.0,
                )
                bonus = 0.01 * bool(senders - seen_senders) + 0.01 * bool(days - seen_days)
                roles = (result_roles or {}).get(result.window.window_id, set())
                bonus += 0.03 * bool(roles - seen_roles)
                if prefer_recent and newest_time is not None:
                    candidate_time = max(
                        (
                            message.time_utc
                            for message in result.messages
                            if message.time_utc is not None
                        ),
                        default=None,
                    )
                    bonus += 0.05 * bool(candidate_time == newest_time)
                return result.score - 0.2 * overlap + bonus, result.window.window_id

            best = max(remaining, key=diversity_score)
            remaining.remove(best)
            selected.append(best)
            seen_senders.update(message.uid for message in best.messages)
            seen_roles.update((result_roles or {}).get(best.window.window_id, set()))
            seen_days.update(
                message.time_utc.date().isoformat()
                for message in best.messages
                if message.time_utc is not None
            )
        return selected

    def _map_evidence(self, blocks: list[EvidenceBlock]) -> list[EvidenceCard]:
        cards: list[EvidenceCard] = []
        groups: list[list[EvidenceBlock]] = []
        current: list[EvidenceBlock] = []
        for block in blocks:
            candidate = [*current, block]
            packed = pack_evidence(candidate, max_tokens=max(self.map_batch_tokens, 1))
            if current and len(packed.message_ids) < sum(len(item.messages) for item in candidate):
                groups.append(current)
                current = [block]
            else:
                current = candidate
        if current:
            groups.append(current)

        available = self.max_input_tokens - estimate_tokens(MAP_SYSTEM_PROMPT) - 10
        for group in groups:
            packed = pack_evidence(group, max_tokens=max(available, 1))
            if not packed.message_ids:
                continue
            raw = self._complete(MAP_SYSTEM_PROMPT, packed.text)
            try:
                card = self._parse_card(raw)
                validate_card_sources(card, set(packed.message_ids))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            cards.append(card)
        return cards

    def _parse_card(self, raw: str) -> EvidenceCard:
        payload = json.loads(raw)
        tuple_fields = (
            "source_ids",
            "claims",
            "proposals",
            "decisions",
            "outcomes",
            "disagreements",
        )
        values = {}
        for field in tuple_fields:
            value = payload.get(field, [])
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"invalid EvidenceCard field: {field}")
            values[field] = tuple(value)
        return EvidenceCard(
            topic=str(payload["topic"]),
            period=str(payload.get("period", "")),
            uncertainty=str(payload.get("uncertainty", "")),
            **values,
        )

    def _final_answer(
        self,
        question: str,
        blocks: list[EvidenceBlock],
        cards: list[EvidenceCard],
        intent: str,
        evidence_roles: Mapping[str, list[str]],
    ) -> tuple[str, set[str]]:
        system_prompt = answer_system_prompt(intent)
        if estimate_tokens(system_prompt + question) + 40 >= self.max_input_tokens:
            system_prompt = COMPACT_FINAL_SYSTEM_PROMPT
        card_payloads: list[str] = []
        card_ids: set[str] = set()
        for card in cards:
            candidate = json.dumps(asdict(card), ensure_ascii=False)
            prefix = "Evidence cards:\n" + "\n".join([*card_payloads, candidate])
            if estimate_tokens(system_prompt + question + prefix) >= self.max_input_tokens:
                break
            card_payloads.append(candidate)
            card_ids.update(card.source_ids)
        cards_text = "\n".join(card_payloads)
        role_text = json.dumps(evidence_roles, ensure_ascii=False, sort_keys=True)
        role_section = (
            "Evidence roles:\n"
            f"{role_text}\nRoles are retrieval goals, not established facts.\n\n"
        )
        base_fixed = (
            f"Question:\n{question}\n\nIntent:\n{intent}\n\n"
            f"Evidence cards:\n{cards_text}\n\nRaw evidence:\n"
        )
        fixed = base_fixed
        if estimate_tokens(system_prompt + role_section + base_fixed) + 20 < self.max_input_tokens:
            fixed = base_fixed.replace("Evidence cards:\n", role_section + "Evidence cards:\n")
        remaining = self.max_input_tokens - estimate_tokens(system_prompt + fixed) - 5
        packed = pack_evidence(blocks, max_tokens=max(remaining, 1))
        user = fixed + packed.text
        answer = self._complete(system_prompt, user, self.answer_delta)
        return answer, card_ids | set(packed.message_ids)

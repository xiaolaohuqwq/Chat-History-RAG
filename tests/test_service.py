import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from chat_rag.domain import Message
from chat_rag.prompts import CITATION_REPAIR_SYSTEM_PROMPT, MAP_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from chat_rag.retrieval import SearchResult
from chat_rag.service import ChatRAGService
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.token_estimator import estimate_tokens
from chat_rag.windowing import build_windows


class FakeRetriever:
    degraded_reason = None

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, **kwargs) -> list[SearchResult]:
        self.queries.append(query)
        return self.results


class ScriptedLLM:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answers.pop(0)


def make_results(store: SQLiteStore, count: int, text_size: int = 5) -> list[SearchResult]:
    messages = []
    for line in range(1, count + 1):
        timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=line)
        messages.append(
            Message(
                f"m{line}",
                "s",
                line,
                timestamp.isoformat(),
                timestamp,
                f"u{line}",
                f"用户{line}",
                "中" * text_size,
                f"h{line}",
            )
        )
    windows = [build_windows([message])[0] for message in messages]
    store.upsert_messages(messages)
    store.upsert_windows(windows)
    return [
        SearchResult(window, (message,), 1 / message.source_line)
        for window, message in zip(windows, messages, strict=True)
    ]


def test_empty_retrieval_skips_every_llm_call(tmp_path: Path) -> None:
    llm = ScriptedLLM([])
    with SQLiteStore(tmp_path / "app.db") as store:
        result = ChatRAGService(store, FakeRetriever([]), llm).ask("为什么没有结果？")
    assert result.empty
    assert llm.calls == []


def test_narrow_question_uses_one_answer_call_and_validates_citation(tmp_path: Path) -> None:
    llm = ScriptedLLM(["结论 [m1]"])
    with SQLiteStore(tmp_path / "app.db") as store:
        results = make_results(store, 1)
        result = ChatRAGService(store, FakeRetriever(results), llm, max_input_tokens=500).ask(
            "ABC-123是什么？"
        )
    assert result.answer == "结论 [m1]"
    assert result.citations == ("m1",)
    assert len(llm.calls) == 1
    assert estimate_tokens("\n".join(llm.calls[0])) <= 500


def test_broad_question_plans_multiple_retrieval_queries(tmp_path: Path) -> None:
    plan = '{"queries":["支持意见","反对意见","原因风险","最终结果"]}'
    llm = ScriptedLLM([plan, "综合结论 [m1]"])
    with SQLiteStore(tmp_path / "app.db") as store:
        results = make_results(store, 1)
        retriever = FakeRetriever(results)
        result = ChatRAGService(store, retriever, llm, max_input_tokens=1000).ask(
            "为什么推迟，最后结果是什么？"
        )
    assert result.answer == "综合结论 [m1]"
    assert len(retriever.queries) == 5
    assert llm.calls[0][0] == PLANNER_SYSTEM_PROMPT


def test_large_evidence_uses_map_reduce_with_bounded_calls(tmp_path: Path) -> None:
    plan = '{"queries":["支持意见","反对意见","原因风险","最终结果"]}'
    card = json.dumps(
        {
            "topic": "上线",
            "period": "2026",
            "claims": ["有风险"],
            "proposals": [],
            "decisions": [],
            "outcomes": [],
            "disagreements": [],
            "source_ids": ["m1"],
            "uncertainty": "未知",
        },
        ensure_ascii=False,
    )
    llm = ScriptedLLM([plan, card, card, card, "综合结论 [m1]"])
    with SQLiteStore(tmp_path / "app.db") as store:
        results = make_results(store, 3, text_size=25)
        result = ChatRAGService(
            store,
            FakeRetriever(results),
            llm,
            max_input_tokens=250,
            hierarchical_threshold=40,
            map_batch_tokens=45,
        ).ask("为什么推迟，最后结果是什么？")
    assert result.used_map_reduce
    assert any(system == MAP_SYSTEM_PROMPT for system, _ in llm.calls)
    assert all(estimate_tokens(system + "\n" + user) <= 250 for system, user in llm.calls)


def test_invalid_citation_gets_one_bounded_repair_call(tmp_path: Path) -> None:
    llm = ScriptedLLM(["错误引用 [m_fake]", "修复引用 [m1]"])
    with SQLiteStore(tmp_path / "app.db") as store:
        results = make_results(store, 1)
        result = ChatRAGService(store, FakeRetriever(results), llm).ask("编号是什么？")
    assert result.answer == "修复引用 [m1]"
    assert llm.calls[-1][0] == CITATION_REPAIR_SYSTEM_PROMPT
    assert result.citation_warning is None

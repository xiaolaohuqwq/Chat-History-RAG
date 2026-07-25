from chat_rag.prompts import FINAL_SYSTEM_PROMPT, parse_retrieval_plan


def test_retrieval_plan_preserves_intent_and_evidence_purposes() -> None:
    raw = """{
      "standalone_question": "项目为什么延期，最后上线了吗？",
      "intent": "decision_history",
      "subqueries": [
        {"purpose": "cause", "query": "项目 延期 原因"},
        {"purpose": "objection", "query": "项目 反对 风险"},
        {"purpose": "outcome", "query": "项目 最终 上线 结果"}
      ]
    }"""
    plan = parse_retrieval_plan(raw)

    assert plan.standalone_question == "项目为什么延期，最后上线了吗？"
    assert plan.intent == "decision_history"
    assert [(item.purpose, item.query) for item in plan.subqueries] == [
        ("cause", "项目 延期 原因"),
        ("objection", "项目 反对 风险"),
        ("outcome", "项目 最终 上线 结果"),
    ]
def test_final_prompt_contains_evidence_and_decision_safety_contract() -> None:
    assert "only" in FINAL_SYSTEM_PROMPT.lower()
    assert "proposal" in FINAL_SYSTEM_PROMPT.lower()
    assert "[message_id]" in FINAL_SYSTEM_PROMPT
    assert "consensus" in FINAL_SYSTEM_PROMPT.lower()
    assert "有人认为" in FINAL_SYSTEM_PROMPT

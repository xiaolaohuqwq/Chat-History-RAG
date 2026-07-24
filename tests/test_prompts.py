import pytest

from chat_rag.prompts import FINAL_SYSTEM_PROMPT, parse_retrieval_plan


def test_retrieval_plan_requires_four_to_eight_unique_short_queries() -> None:
    raw = '{"queries":["支持上线", "反对上线", "最终决定", "后续结果"]}'
    assert parse_retrieval_plan(raw) == ["支持上线", "反对上线", "最终决定", "后续结果"]
    with pytest.raises(ValueError, match="four to eight"):
        parse_retrieval_plan('{"queries":["only one"]}')
    with pytest.raises(ValueError, match="unique"):
        parse_retrieval_plan('{"queries":["a", "a", "b", "c"]}')


def test_final_prompt_contains_evidence_and_decision_safety_contract() -> None:
    assert "only" in FINAL_SYSTEM_PROMPT.lower()
    assert "proposal" in FINAL_SYSTEM_PROMPT.lower()
    assert "[message_id]" in FINAL_SYSTEM_PROMPT
    assert "consensus" in FINAL_SYSTEM_PROMPT.lower()

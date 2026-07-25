import pytest

from chat_rag.evidence import (
    EvidenceCard,
    cited_ids,
    invalid_citations,
    validate_card_sources,
)


def test_evidence_card_rejects_ids_outside_its_input_batch() -> None:
    card = EvidenceCard(topic="上线", source_ids=("m1", "m9"), claims=("测试未完成",))
    with pytest.raises(ValueError, match="outside"):
        validate_card_sources(card, {"m1", "m2"})


def test_citation_validation_only_accepts_provided_message_ids() -> None:
    answer = "结论来自 [m1]，另一个说法来自 [m_fake]，窗口引用 [w_fake] 也不合法。"
    assert invalid_citations(answer, {"m1", "m2"}) == {"m_fake", "w_fake"}
    assert invalid_citations("没有引用", {"m1"}) == set()


def test_grouped_citations_are_extracted_and_hidden_from_display_text() -> None:
    answer = "结论 [e6, m_5c0dbc4b7af05f113ed9f1bd]，依据 [m2]。"
    assert cited_ids(answer) == {"e6", "m_5c0dbc4b7af05f113ed9f1bd", "m2"}
    assert invalid_citations(answer, {"m_5c0dbc4b7af05f113ed9f1bd", "m2"}) == set()


def test_pipe_separated_citations_are_extracted() -> None:
    answer = "结论 [m_148d687d606705adeb17e3cc | e4]"
    assert cited_ids(answer) == {"m_148d687d606705adeb17e3cc", "e4"}
    assert invalid_citations(answer, {"m_148d687d606705adeb17e3cc"}) == set()

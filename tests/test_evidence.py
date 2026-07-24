import pytest

from chat_rag.evidence import EvidenceCard, invalid_citations, validate_card_sources


def test_evidence_card_rejects_ids_outside_its_input_batch() -> None:
    card = EvidenceCard(topic="上线", source_ids=("m1", "m9"), claims=("测试未完成",))
    with pytest.raises(ValueError, match="outside"):
        validate_card_sources(card, {"m1", "m2"})


def test_citation_validation_only_accepts_provided_message_ids() -> None:
    answer = "结论来自 [m1]，另一个说法来自 [m_fake]。"
    assert invalid_citations(answer, {"m1", "m2"}) == {"m_fake"}
    assert invalid_citations("没有引用", {"m1"}) == set()

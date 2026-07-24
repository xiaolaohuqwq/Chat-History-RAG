from chat_rag.normalize import NORMALIZATION_VERSION, make_message_id, normalize_text, parse_time


def test_normalize_text_preserves_content_but_normalizes_whitespace() -> None:
    assert normalize_text("  Hello\r\n\u3000世界  ") == "Hello\n 世界"
    assert NORMALIZATION_VERSION


def test_message_id_is_stable_and_source_sensitive() -> None:
    first = make_message_id("source-a", 7, "u1", "2026-01-01", "hello")
    assert first == make_message_id("source-a", 7, "u1", "2026-01-01", "hello")
    assert first != make_message_id("source-a", 8, "u1", "2026-01-01", "hello")
    assert first.startswith("m_")


def test_parse_time_supports_export_formats_without_dropping_invalid_values() -> None:
    assert parse_time("2026-07-01 10:15:00") is not None
    assert parse_time("2026-07-01T10:15:00+08:00") is not None
    assert parse_time("not-a-time") is None

from datetime import UTC, datetime

from chat_rag.domain import Message
from chat_rag.windowing import WINDOWING_VERSION, build_windows, iter_windows


def message(line: int, minute: int, text: str = "测试消息") -> Message:
    parsed = datetime(2026, 1, 1, 10, minute, tzinfo=UTC)
    return Message(
        message_id=f"m_{line}",
        source_id="source",
        source_line=line,
        time_raw=parsed.isoformat(),
        time_utc=parsed,
        uid=str(line),
        name="用户",
        text=text,
        content_hash=str(line),
    )


def test_windowing_never_crosses_session_gap() -> None:
    windows = build_windows(
        [message(1, 0), message(2, 1), message(3, 40)], target_tokens=100, max_tokens=120
    )
    assert len(windows) == 2
    assert windows[0].end_line == 2
    assert windows[1].start_line == 3
    assert all(window.windowing_version == WINDOWING_VERSION for window in windows)


def test_windowing_uses_overlap_and_respects_hard_maximum() -> None:
    windows = build_windows(
        [message(i, i, "中" * 22) for i in range(1, 6)],
        target_tokens=55,
        max_tokens=75,
        overlap_messages=1,
    )
    assert len(windows) > 1
    assert all(window.estimated_tokens <= 75 for window in windows)
    assert windows[0].message_ids[-1] == windows[1].message_ids[0]


def test_window_generator_emits_before_consuming_the_whole_source() -> None:
    consumed = 0

    def source():
        nonlocal consumed
        for line in range(1, 101):
            consumed += 1
            yield message(line, line % 60, "中" * 20)

    windows = iter_windows(source(), target_tokens=50, max_tokens=70, overlap_messages=0)
    assert next(windows).estimated_tokens <= 70
    assert consumed < 100


def test_exceptionally_long_message_is_split_below_hard_limit_without_data_loss() -> None:
    original = "中" * 500
    windows = build_windows(
        [message(1, 0, original)], target_tokens=50, max_tokens=80, overlap_messages=0
    )
    assert len(windows) > 1
    assert all(window.estimated_tokens <= 80 for window in windows)
    assert sum(window.text.count("中") for window in windows) == len(original)
    assert all(window.message_ids == ("m_1",) for window in windows)


def test_overlap_does_not_create_a_trailing_window_with_no_new_message() -> None:
    windows = build_windows(
        [message(1, 0, "中" * 40), message(2, 1, "中" * 40)],
        target_tokens=45,
        max_tokens=80,
        overlap_messages=1,
    )
    assert len({window.window_id for window in windows}) == len(windows)

from datetime import UTC, datetime

from chat_rag.domain import Message
from chat_rag.windowing import WINDOWING_VERSION, build_windows


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

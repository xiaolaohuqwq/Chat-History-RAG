from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime

NORMALIZATION_VERSION = "v1"

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_KNOWN_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HORIZONTAL_WHITESPACE.sub(" ", line).rstrip() for line in value.split("\n")]
    return "\n".join(lines).strip()


def content_hash(text: str) -> str:
    payload = f"{NORMALIZATION_VERSION}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def make_message_id(source_id: str, line: int, uid: str, time_raw: str, text: str) -> str:
    payload = "\0".join((source_id, str(line), uid, time_raw, text)).encode()
    return f"m_{hashlib.sha256(payload).hexdigest()[:24]}"


def parse_time(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is None:
        for format_string in _KNOWN_TIME_FORMATS:
            try:
                parsed = datetime.strptime(value, format_string)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)

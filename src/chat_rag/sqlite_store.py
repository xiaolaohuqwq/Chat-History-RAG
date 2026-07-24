from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from chat_rag.domain import Message, Window

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_line INTEGER NOT NULL,
    time_raw TEXT NOT NULL,
    time_utc TEXT,
    uid TEXT NOT NULL,
    name TEXT NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    UNIQUE(source_id, source_line)
);
CREATE TABLE IF NOT EXISTS windows (
    window_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    start_time TEXT,
    end_time TEXT,
    text TEXT NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    windowing_version TEXT NOT NULL,
    embedding_model TEXT,
    embedding_dimension INTEGER,
    embedded_at TEXT
);
CREATE TABLE IF NOT EXISTS window_messages (
    window_id TEXT NOT NULL REFERENCES windows(window_id) ON DELETE CASCADE,
    message_id TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    PRIMARY KEY(window_id, message_id)
);
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    last_completed_line INTEGER NOT NULL DEFAULT 0,
    message_count INTEGER NOT NULL DEFAULT 0,
    window_count INTEGER NOT NULL DEFAULT 0,
    malformed_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    embedding_model TEXT,
    embedding_dimension INTEGER,
    error_summary TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS windows_fts USING fts5(window_id UNINDEXED, text);
"""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def count(self, table: str) -> int:
        if table not in {"messages", "windows", "window_messages", "ingestion_runs"}:
            raise ValueError("unsupported table")
        row = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0])

    def upsert_messages(self, messages: Iterable[Message]) -> None:
        self.connection.executemany(
            """INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
              time_raw=excluded.time_raw, time_utc=excluded.time_utc,
              uid=excluded.uid, name=excluded.name, text=excluded.text,
              content_hash=excluded.content_hash""",
            [
                (
                    item.message_id,
                    item.source_id,
                    item.source_line,
                    item.time_raw,
                    _iso(item.time_utc),
                    item.uid,
                    item.name,
                    item.text,
                    item.content_hash,
                )
                for item in messages
            ],
        )
        self.connection.commit()

    def upsert_windows(self, windows: Iterable[Window]) -> None:
        for window in windows:
            self.connection.execute(
                """INSERT INTO windows
                (window_id, source_id, start_line, end_line, start_time, end_time, text,
                 estimated_tokens, content_hash, windowing_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(window_id) DO UPDATE SET text=excluded.text,
                  estimated_tokens=excluded.estimated_tokens, content_hash=excluded.content_hash""",
                (
                    window.window_id,
                    window.source_id,
                    window.start_line,
                    window.end_line,
                    _iso(window.start_time),
                    _iso(window.end_time),
                    window.text,
                    window.estimated_tokens,
                    window.content_hash,
                    window.windowing_version,
                ),
            )
            self.connection.execute(
                "DELETE FROM window_messages WHERE window_id = ?", (window.window_id,)
            )
            self.connection.executemany(
                "INSERT INTO window_messages VALUES (?, ?, ?)",
                [
                    (window.window_id, message_id, position)
                    for position, message_id in enumerate(window.message_ids)
                ],
            )
            self.connection.execute(
                "DELETE FROM windows_fts WHERE window_id = ?", (window.window_id,)
            )
            self.connection.execute(
                "INSERT INTO windows_fts VALUES (?, ?)", (window.window_id, window.text)
            )
        self.connection.commit()

    def get_message(self, message_id: str) -> Message | None:
        row = self.connection.execute(
            """SELECT message_id, source_id, source_line, time_raw, time_utc,
            uid, name, text, content_hash FROM messages WHERE message_id = ?""",
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        return Message(*row[:4], _datetime(row[4]), *row[5:])

    def get_window(self, window_id: str) -> Window | None:
        row = self.connection.execute(
            """SELECT window_id, source_id, start_line, end_line, start_time, end_time,
            text, estimated_tokens, content_hash, windowing_version
            FROM windows WHERE window_id = ?""",
            (window_id,),
        ).fetchone()
        if row is None:
            return None
        ids = tuple(
            item[0]
            for item in self.connection.execute(
                "SELECT message_id FROM window_messages WHERE window_id = ? ORDER BY position",
                (window_id,),
            )
        )
        return Window(*row[:4], _datetime(row[4]), _datetime(row[5]), *row[6:], message_ids=ids)

    def embedding_identities(self) -> set[tuple[str, int]]:
        return {
            (str(row[0]), int(row[1]))
            for row in self.connection.execute(
                """SELECT DISTINCT embedding_model, embedding_dimension FROM windows
                WHERE embedding_model IS NOT NULL AND embedding_dimension IS NOT NULL"""
            )
        }

    def clear_embeddings(self) -> None:
        self.connection.execute(
            """UPDATE windows SET embedding_model = NULL,
            embedding_dimension = NULL, embedded_at = NULL"""
        )
        self.connection.commit()

    def windows_needing_embedding(self, model: str, dimension: int) -> list[Window]:
        ids = [
            str(row[0])
            for row in self.connection.execute(
                """SELECT window_id FROM windows
                WHERE embedding_model IS NULL OR embedding_model != ?
                   OR embedding_dimension IS NULL OR embedding_dimension != ?
                ORDER BY source_id, start_line""",
                (model, dimension),
            )
        ]
        return [window for window_id in ids if (window := self.get_window(window_id)) is not None]

    def mark_embedded(
        self,
        window_ids: list[str],
        model: str,
        dimension: int,
        embedded_at: datetime,
    ) -> None:
        self.connection.executemany(
            """UPDATE windows SET embedding_model = ?, embedding_dimension = ?, embedded_at = ?
            WHERE window_id = ?""",
            [(model, dimension, embedded_at.isoformat(), window_id) for window_id in window_ids],
        )
        self.connection.commit()

    def get_window_messages(self, window_id: str) -> list[Message]:
        ids = [
            str(row[0])
            for row in self.connection.execute(
                """SELECT message_id FROM window_messages
                WHERE window_id = ? ORDER BY position""",
                (window_id,),
            )
        ]
        return [message for message_id in ids if (message := self.get_message(message_id))]

    def lexical_search(self, query: str, limit: int) -> list[str]:
        from chat_rag.retrieval import lexical_terms

        terms = lexical_terms(query)
        fts_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        fts_ids = [
            str(row[0])
            for row in self.connection.execute(
                """SELECT window_id FROM windows_fts WHERE windows_fts MATCH ?
                ORDER BY bm25(windows_fts), window_id LIMIT ?""",
                (fts_query, limit),
            )
        ]
        fragment = "".join(terms)
        escaped = fragment.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        substring_ids = [
            str(row[0])
            for row in self.connection.execute(
                """SELECT window_id FROM windows WHERE text LIKE ? ESCAPE '\\'
                ORDER BY start_line, window_id LIMIT ?""",
                (f"%{escaped}%", limit),
            )
        ]
        combined: list[str] = []
        for window_id in [*fts_ids, *substring_ids]:
            if window_id not in combined:
                combined.append(window_id)
        return combined[:limit]

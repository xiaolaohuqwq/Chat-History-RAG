from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import lancedb


@dataclass(frozen=True, slots=True)
class VectorResult:
    window_id: str
    distance: float


class VectorStore(Protocol):
    def upsert(self, rows: list[tuple[str, list[float]]]) -> None: ...

    def search(self, vector: list[float], limit: int) -> list[VectorResult]: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...

    def delete(self, window_ids: list[str]) -> None: ...


class LanceVectorStore:
    table_name = "windows"

    def __init__(self, path: Path, dimension: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.database = lancedb.connect(path)
        self.dimension = dimension

    def _table(self):
        if self.table_name not in self.database.list_tables().tables:
            return None
        return self.database.open_table(self.table_name)

    def upsert(self, rows: list[tuple[str, list[float]]]) -> None:
        if not rows:
            return
        if any(len(vector) != self.dimension for _, vector in rows):
            raise ValueError(f"all vectors must have dimension {self.dimension}")
        records = [{"window_id": window_id, "vector": vector} for window_id, vector in rows]
        table = self._table()
        if table is None:
            self.database.create_table(self.table_name, data=records)
            return
        (
            table.merge_insert("window_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(records)
        )

    def search(self, vector: list[float], limit: int) -> list[VectorResult]:
        if len(vector) != self.dimension:
            raise ValueError(f"query vector must have dimension {self.dimension}")
        table = self._table()
        if table is None:
            return []
        rows = table.search(vector).distance_type("cosine").limit(limit).to_list()
        return [VectorResult(str(row["window_id"]), float(row["_distance"])) for row in rows]

    def count(self) -> int:
        table = self._table()
        return table.count_rows() if table is not None else 0

    def clear(self) -> None:
        if self._table() is not None:
            self.database.drop_table(self.table_name)

    def delete(self, window_ids: list[str]) -> None:
        if not window_ids:
            return
        table = self._table()
        if table is None:
            return
        for offset in range(0, len(window_ids), 500):
            batch = window_ids[offset : offset + 500]
            quoted = ",".join(f"'{window_id}'" for window_id in batch)
            table.delete(f"window_id IN ({quoted})")

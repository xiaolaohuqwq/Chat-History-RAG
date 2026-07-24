from pathlib import Path

from chat_rag.vector_store import LanceVectorStore


def test_lance_store_upserts_and_searches_vectors(tmp_path: Path) -> None:
    store = LanceVectorStore(tmp_path / "vectors", dimension=2)
    store.upsert([("w1", [1.0, 0.0]), ("w2", [0.0, 1.0])])
    store.upsert([("w1", [0.9, 0.1])])

    results = store.search([1.0, 0.0], limit=2)

    assert [result.window_id for result in results] == ["w1", "w2"]
    assert store.count() == 2

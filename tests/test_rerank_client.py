import httpx
import pytest

from chat_rag.rerank_client import DashScopeReranker, RerankError


def test_reranker_maps_provider_indices_to_candidate_ids() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.2},
                    ]
                }
            },
        )
    )
    reranker = DashScopeReranker("secret", "model", transport=transport, max_attempts=1)
    assert reranker.rerank("query", [("w1", "one"), ("w2", "two")]) == [
        ("w2", 0.9),
        ("w1", 0.2),
    ]


def test_reranker_retries_transient_errors_and_rejects_auth_immediately() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500 if attempts == 1 else 200, json={"output": {"results": []}})

    reranker = DashScopeReranker(
        "secret", "model", transport=httpx.MockTransport(handler), max_attempts=2, backoff_seconds=0
    )
    assert reranker.rerank("query", []) == []
    assert attempts == 0
    assert reranker.rerank("query", [("w", "text")]) == []
    assert attempts == 2

    auth = DashScopeReranker(
        "secret",
        "model",
        transport=httpx.MockTransport(lambda request: httpx.Response(403)),
        max_attempts=3,
    )
    with pytest.raises(RerankError, match="authentication"):
        auth.rerank("query", [("w", "text")])

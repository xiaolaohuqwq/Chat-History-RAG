import httpx
import pytest

from chat_rag.embedding_client import DashScopeEmbeddingClient, EmbeddingError


def response(status: int, payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(status, json=payload)


def test_embedding_response_is_mapped_by_index_and_dimension_checked() -> None:
    transport = httpx.MockTransport(
        lambda request: response(
            200,
            {
                "data": [
                    {"index": 1, "embedding": [3.0, 4.0]},
                    {"index": 0, "embedding": [1.0, 2.0]},
                ]
            },
        )
    )
    client = DashScopeEmbeddingClient("secret", "model", 2, transport=transport, max_attempts=1)
    assert client.embed_documents(["first", "second"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_embedding_retries_429_but_not_authentication_failure() -> None:
    attempts = 0

    def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return response(429, {"message": "slow down"})
        return response(200, {"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    client = DashScopeEmbeddingClient(
        "secret",
        "model",
        2,
        transport=httpx.MockTransport(retry_handler),
        max_attempts=2,
        backoff_seconds=0,
    )
    assert client.embed_documents(["text"]) == [[1.0, 2.0]]
    assert attempts == 2

    auth_attempts = 0

    def auth_handler(request: httpx.Request) -> httpx.Response:
        nonlocal auth_attempts
        auth_attempts += 1
        return response(401, {"message": "denied"})

    auth_client = DashScopeEmbeddingClient(
        "secret", "model", 2, transport=httpx.MockTransport(auth_handler), max_attempts=3
    )
    with pytest.raises(EmbeddingError, match="authentication"):
        auth_client.embed_documents(["text"])
    assert auth_attempts == 1


def test_embedding_rejects_wrong_vector_dimension() -> None:
    transport = httpx.MockTransport(
        lambda request: response(200, {"data": [{"index": 0, "embedding": [1.0]}]})
    )
    client = DashScopeEmbeddingClient("secret", "model", 2, transport=transport, max_attempts=1)
    with pytest.raises(EmbeddingError, match="dimension"):
        client.embed_documents(["text"])

import httpx
import pytest

from chat_rag.llm_client import LLMError, OpenAICompatibleClient


def completion_payload(content: str) -> dict[str, object]:
    return {
        "id": "completion",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
        ("https://relay.example/custom/v1", "https://relay.example/custom/v1/chat/completions"),
    ],
)
def test_client_preserves_configured_base_path(base_url: str, expected_url: str) -> None:
    observed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(str(request.url))
        return httpx.Response(200, json=completion_payload("answer"))

    client = OpenAICompatibleClient(
        "secret", base_url, "test-model", transport=httpx.MockTransport(handler), max_attempts=1
    )
    assert client.complete("system", "user") == "answer"
    assert observed == [expected_url]


def test_client_retries_429_once_and_redacts_key_from_errors() -> None:
    secret = "key-that-must-not-leak"
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": f"bad {secret}"}})
        return httpx.Response(200, json=completion_payload("ok"))

    client = OpenAICompatibleClient(
        secret,
        "https://relay.example/v1",
        "model",
        transport=httpx.MockTransport(handler),
        max_attempts=2,
        backoff_seconds=0,
    )
    assert client.complete("system", "user") == "ok"
    assert attempts == 2

    denied = OpenAICompatibleClient(
        secret,
        "https://relay.example/v1",
        "model",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"error": {"message": secret}})
        ),
        max_attempts=1,
    )
    with pytest.raises(LLMError) as error:
        denied.complete("system", "user")
    assert secret not in str(error.value)

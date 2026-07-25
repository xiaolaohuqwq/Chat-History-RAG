import pytest

from chat_rag.config import Settings


def test_default_context_budget_is_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("API_KEY", "BASE_URL", "MODEL"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.embedding_dimension == 1024
    assert settings.llm_max_output_tokens == 15_000
    assert settings.final_evidence_blocks == 30
    assert settings.rerank_candidates == 100
    assert (
        settings.llm_max_input_tokens + settings.llm_max_output_tokens
        <= settings.llm_context_window
    )


def test_invalid_context_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="context budget"):
        Settings(
            _env_file=None,
            llm_context_window=100,
            llm_max_input_tokens=90,
            llm_max_output_tokens=20,
        )


@pytest.mark.parametrize("workers", [0, 5])
def test_embedding_concurrency_is_bounded(workers: int) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        Settings(_env_file=None, embedding_concurrency=workers)


def test_cloud_validation_never_echoes_secret() -> None:
    secret = "super-secret-key"
    settings = Settings(_env_file=None, api_key=secret, base_url="", model="")
    with pytest.raises(ValueError) as error:
        settings.require_llm()
    assert secret not in str(error.value)

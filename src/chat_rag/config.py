from __future__ import annotations

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    dashscope_api_key: SecretStr | None = None
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimension: int = 1024
    rerank_model: str = "qwen3-rerank"

    api_key: SecretStr | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    llm_context_window: int = 200_000
    llm_max_input_tokens: int = 140_000
    llm_max_output_tokens: int = 15_000

    window_target_tokens: int = 500
    window_max_tokens: int = 800
    window_overlap_messages: int = 2
    session_gap_minutes: int = 20
    embed_batch_size: int = 64
    vector_top_k_per_query: int = 40
    lexical_top_k_per_query: int = 40
    rerank_candidates: int = 100
    final_evidence_blocks: int = 30
    data_dir: str = "data"

    @model_validator(mode="after")
    def validate_budgets(self) -> Settings:
        if self.llm_max_input_tokens + self.llm_max_output_tokens > self.llm_context_window:
            raise ValueError("LLM context budget is inconsistent")
        if self.window_target_tokens <= 0 or self.window_max_tokens < self.window_target_tokens:
            raise ValueError("window context budget is inconsistent")
        if not 256 <= self.embedding_dimension <= 2560:
            raise ValueError("embedding dimension must be between 256 and 2560")
        return self

    def require_embedding(self) -> str:
        if self.dashscope_api_key is None or not self.dashscope_api_key.get_secret_value():
            raise ValueError("DASHSCOPE_API_KEY is required for embedding")
        return self.dashscope_api_key.get_secret_value()

    def require_llm(self) -> tuple[str, str, str]:
        if self.api_key is None or not self.api_key.get_secret_value():
            raise ValueError("API_KEY is required for answer generation")
        if not self.base_url.strip():
            raise ValueError("BASE_URL is required for answer generation")
        if not self.model.strip():
            raise ValueError("MODEL is required for answer generation")
        return self.api_key.get_secret_value(), self.base_url, self.model

from __future__ import annotations

import random
import time
from typing import Protocol

import httpx


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class DashScopeEmbeddingClient:
    endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int,
        *,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 4,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
            transport=transport,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._post(texts)
        try:
            items = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = [[float(value) for value in item["embedding"]] for item in items]
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingError("embedding provider returned an invalid response") from error
        if len(vectors) != len(texts):
            raise EmbeddingError("embedding provider returned a partial response")
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingError(f"embedding dimension mismatch; expected {self.dimension} values")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _post(self, texts: list[str]) -> httpx.Response:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.post(
                    self.endpoint,
                    json={"model": self.model, "input": texts, "dimensions": self.dimension},
                )
            except httpx.TransportError as error:
                if attempt == self.max_attempts:
                    raise EmbeddingError("embedding provider connection failed") from error
                self._backoff(attempt)
                continue
            if response.status_code in {401, 403}:
                raise EmbeddingError("embedding provider authentication or permission failed")
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                    continue
                raise EmbeddingError(
                    f"embedding provider remained unavailable (HTTP {response.status_code})"
                )
            if response.is_error:
                raise EmbeddingError(
                    f"embedding request was rejected (HTTP {response.status_code})"
                )
            return response
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int) -> None:
        delay = self.backoff_seconds * (2 ** (attempt - 1))
        time.sleep(delay + random.uniform(0, delay / 4 if delay else 0))

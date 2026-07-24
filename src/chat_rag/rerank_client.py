from __future__ import annotations

import random
import time
from typing import Protocol

import httpx

from chat_rag.token_estimator import estimate_tokens


class RerankError(RuntimeError):
    pass


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]: ...


class DashScopeReranker:
    endpoint = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 4,
        backoff_seconds: float = 0.5,
        max_input_tokens: int = 30_000,
    ) -> None:
        self.model = model
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.max_input_tokens = max_input_tokens
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
            transport=transport,
        )

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        if not candidates:
            return []
        selected: list[tuple[str, str]] = []
        used_tokens = estimate_tokens(query) + 100
        for candidate in candidates:
            candidate_tokens = estimate_tokens(candidate[1])
            if used_tokens + candidate_tokens > self.max_input_tokens:
                continue
            selected.append(candidate)
            used_tokens += candidate_tokens
        if not selected:
            raise RerankError("no complete reranking candidate fits the provider context limit")
        response = self._post(query, [text for _, text in selected])
        try:
            results = response.json()["output"]["results"]
            ranked = [
                (selected[int(item["index"])][0], float(item["relevance_score"]))
                for item in results
            ]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise RerankError("reranking provider returned an invalid response") from error
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    def _post(self, query: str, documents: list[str]) -> httpx.Response:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.post(
                    self.endpoint,
                    json={
                        "model": self.model,
                        "input": {"query": query, "documents": documents},
                        "parameters": {"return_documents": False, "top_n": len(documents)},
                    },
                )
            except httpx.TransportError as error:
                if attempt == self.max_attempts:
                    raise RerankError("reranking provider connection failed") from error
                self._backoff(attempt)
                continue
            if response.status_code in {401, 403}:
                raise RerankError("reranking provider authentication or permission failed")
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                    continue
                raise RerankError(
                    f"reranking provider remained unavailable (HTTP {response.status_code})"
                )
            if response.is_error:
                raise RerankError(f"reranking request was rejected (HTTP {response.status_code})")
            return response
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int) -> None:
        delay = self.backoff_seconds * (2 ** (attempt - 1))
        time.sleep(delay + random.uniform(0, delay / 4 if delay else 0))

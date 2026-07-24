from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from chat_rag.domain import Message, Window
from chat_rag.embedding_client import EmbeddingProvider
from chat_rag.rerank_client import Reranker
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.vector_store import VectorStore

_TERM_PATTERN = re.compile(r"[A-Za-z0-9_\u3400-\u9fff]+(?:[-./][A-Za-z0-9_\u3400-\u9fff]+)*")


@dataclass(frozen=True, slots=True)
class SearchResult:
    window: Window
    messages: tuple[Message, ...]
    score: float


def lexical_terms(query: str) -> list[str]:
    terms = _TERM_PATTERN.findall(query)
    if not terms:
        raise ValueError("lexical query must contain meaningful letters, numbers, or CJK text")
    return terms


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[str]], *, constant: int = 60
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (constant + rank)
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


class HybridRetriever:
    def __init__(
        self,
        store: SQLiteStore,
        vectors: VectorStore,
        embedder: EmbeddingProvider,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker
        self.degraded_reason: str | None = None

    def search(
        self,
        query: str,
        *,
        vector_limit: int = 40,
        lexical_limit: int = 40,
        limit: int = 30,
    ) -> list[SearchResult]:
        vector = self.embedder.embed_query(query)
        vector_ids = [item.window_id for item in self.vectors.search(vector, vector_limit)]
        lexical_ids = self.store.lexical_search(query, lexical_limit)
        scores = reciprocal_rank_fusion([vector_ids, lexical_ids])
        ordered_ids = list(scores)
        self.degraded_reason = None
        if self.reranker is not None:
            candidates = [
                (window_id, window.text)
                for window_id in ordered_ids
                if (window := self.store.get_window(window_id)) is not None
            ]
            try:
                reranked = self.reranker.rerank(query, candidates)
                rerank_scores = dict(reranked)
                reranked_ids = [window_id for window_id, _score in reranked]
                ordered_ids = [
                    *reranked_ids,
                    *(item for item in ordered_ids if item not in rerank_scores),
                ]
                scores.update(rerank_scores)
            except RuntimeError:
                self.degraded_reason = "reranking unavailable"
        results: list[SearchResult] = []
        for window_id in ordered_ids[:limit]:
            window = self.store.get_window(window_id)
            if window is None:
                continue
            results.append(
                SearchResult(
                    window=window,
                    messages=tuple(self.store.get_window_messages(window_id)),
                    score=scores[window_id],
                )
            )
        return results

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

from chat_rag.config import Settings
from chat_rag.embedding_client import DashScopeEmbeddingClient
from chat_rag.llm_client import OpenAICompatibleClient
from chat_rag.rerank_client import DashScopeReranker
from chat_rag.retrieval import HybridRetriever, SearchResult
from chat_rag.rpc import EventEmitter, RpcRequest
from chat_rag.service import ChatRAGService
from chat_rag.sqlite_store import SQLiteStore
from chat_rag.vector_store import LanceVectorStore


class RpcApplication:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def __call__(self, request: RpcRequest, cancelled: Event, emit: EventEmitter) -> dict[str, Any]:
        if cancelled.is_set():
            raise InterruptedError
        if request.method == "stats":
            return self._stats()
        if request.method == "inspect":
            return self._inspect(self._string_param(request, "id"))
        if request.method == "search":
            query = self._string_param(request, "query")
            limit = self._int_param(request, "limit", default=20, minimum=1, maximum=100)
            results, degraded = self._search(
                query, limit, bool(request.params.get("no_rerank", False))
            )
            payload = {
                "results": [self._search_result(result) for result in results],
                "degraded_reason": degraded,
            }
            emit("retrieval", payload)
            return payload
        if request.method == "ask":
            return self._ask(
                self._string_param(request, "question"),
                bool(request.params.get("no_rerank", False)),
                cancelled,
                emit,
            )
        raise ValueError(f"unsupported method: {request.method}")

    def _providers(self, no_rerank: bool):
        key = self.settings.require_embedding()
        embedder = DashScopeEmbeddingClient(
            key, self.settings.embedding_model, self.settings.embedding_dimension
        )
        reranker = None if no_rerank else DashScopeReranker(key, self.settings.rerank_model)
        return embedder, reranker

    def _search(
        self, query: str, limit: int, no_rerank: bool
    ) -> tuple[list[SearchResult], str | None]:
        embedder, reranker = self._providers(no_rerank)
        data_dir = Path(self.settings.data_dir)
        with SQLiteStore(data_dir / "app.db") as store:
            retriever = HybridRetriever(
                store,
                LanceVectorStore(data_dir / "vectors", self.settings.embedding_dimension),
                embedder,
                reranker,
            )
            results = retriever.search(
                query,
                vector_limit=self.settings.vector_top_k_per_query,
                lexical_limit=self.settings.lexical_top_k_per_query,
                limit=limit,
            )
            return results, retriever.degraded_reason

    def _ask(
        self,
        question: str,
        no_rerank: bool,
        cancelled: Event,
        emit: EventEmitter,
    ) -> dict[str, Any]:
        embedding_key = self.settings.require_embedding()
        llm_key, base_url, model = self.settings.require_llm()
        data_dir = Path(self.settings.data_dir)

        def progress(stage: str) -> None:
            if cancelled.is_set():
                raise InterruptedError
            emit("progress", {"stage": stage})

        with SQLiteStore(data_dir / "app.db") as store:
            retriever = HybridRetriever(
                store,
                LanceVectorStore(data_dir / "vectors", self.settings.embedding_dimension),
                DashScopeEmbeddingClient(
                    embedding_key,
                    self.settings.embedding_model,
                    self.settings.embedding_dimension,
                ),
                None if no_rerank else DashScopeReranker(embedding_key, self.settings.rerank_model),
            )
            result = ChatRAGService(
                store,
                retriever,
                OpenAICompatibleClient(
                    llm_key,
                    base_url,
                    model,
                    max_output_tokens=self.settings.llm_max_output_tokens,
                ),
                max_input_tokens=self.settings.llm_max_input_tokens,
                final_evidence_blocks=self.settings.final_evidence_blocks,
                progress=progress,
            ).ask(question)
        if cancelled.is_set():
            raise InterruptedError
        emit("answer_delta", {"text": result.answer})
        return {
            "answer": result.answer,
            "citations": list(result.citations),
            "empty": result.empty,
            "used_map_reduce": result.used_map_reduce,
            "degraded_reason": result.degraded_reason,
            "citation_warning": result.citation_warning,
        }

    def _inspect(self, item_id: str) -> dict[str, Any]:
        with SQLiteStore(Path(self.settings.data_dir) / "app.db") as store:
            message = store.get_message(item_id)
            if message is not None:
                return {
                    "kind": "message",
                    "id": message.message_id,
                    "sender": message.name,
                    "uid": message.uid,
                    "timestamp": message.time_raw,
                    "text": message.text,
                }
            window = store.get_window(item_id)
            if window is not None:
                return {
                    "kind": "window",
                    "id": window.window_id,
                    "start_line": window.start_line,
                    "end_line": window.end_line,
                    "text": window.text,
                    "message_ids": list(window.message_ids),
                }
        raise ValueError(f"no message or window found for {item_id}")

    def _stats(self) -> dict[str, Any]:
        data_dir = Path(self.settings.data_dir)
        with SQLiteStore(data_dir / "app.db") as store:
            identities = sorted(store.embedding_identities())
            counts = {
                "messages": store.count("messages"),
                "windows": store.count("windows"),
                "index_identities": [
                    {"model": model, "dimension": dimension} for model, dimension in identities
                ],
            }
        counts["vectors"] = LanceVectorStore(
            data_dir / "vectors", self.settings.embedding_dimension
        ).count()
        return counts

    def _search_result(self, result: SearchResult) -> dict[str, Any]:
        return {
            "window_id": result.window.window_id,
            "score": result.score,
            "start_time": result.window.start_time.isoformat()
            if result.window.start_time
            else None,
            "end_time": result.window.end_time.isoformat() if result.window.end_time else None,
            "messages": [
                {
                    "id": message.message_id,
                    "sender": message.name,
                    "uid": message.uid,
                    "timestamp": message.time_raw,
                    "text": message.text,
                }
                for message in result.messages
            ],
        }

    def _string_param(self, request: RpcRequest, name: str) -> str:
        value = request.params.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"parameter {name} must be a non-empty string")
        return value.strip()

    def _int_param(
        self,
        request: RpcRequest,
        name: str,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        value = request.params.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            raise ValueError(f"parameter {name} must be between {minimum} and {maximum}")
        return value

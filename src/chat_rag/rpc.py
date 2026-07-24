from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROTOCOL_VERSION = 1


class RpcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    id: str = Field(min_length=1, max_length=128)
    method: Literal["ask", "search", "inspect", "stats", "cancel"]
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def current_protocol_version(cls, value: int) -> int:
        if value != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {value}")
        return value


def parse_request(line: str) -> RpcRequest:
    try:
        return RpcRequest.model_validate_json(line)
    except ValidationError as error:
        raise ValueError(f"invalid RPC request: {error.errors(include_url=False)}") from None


EventEmitter = Callable[[str, dict[str, Any]], None]
RequestHandler = Callable[[RpcRequest, Event, EventEmitter], dict[str, Any]]


class StdioServer:
    def __init__(self, handler: RequestHandler, *, max_workers: int = 4) -> None:
        self.handler = handler
        self.max_workers = max_workers
        self._output_lock = Lock()
        self._active_lock = Lock()
        self._active: dict[str, Event] = {}

    def serve(self, source: TextIO, output: TextIO) -> None:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for line in source:
                if not line.strip():
                    continue
                try:
                    request = parse_request(line)
                except ValueError as error:
                    self._write(
                        output,
                        {
                            "version": PROTOCOL_VERSION,
                            "id": None,
                            "type": "error",
                            "code": "invalid_request",
                            "message": str(error),
                        },
                    )
                    continue
                if request.method == "cancel":
                    target = request.params.get("request_id")
                    with self._active_lock:
                        event = self._active.get(target) if isinstance(target, str) else None
                    if event is not None:
                        event.set()
                    self._write(
                        output,
                        {
                            "version": PROTOCOL_VERSION,
                            "id": request.id,
                            "type": "result",
                            "payload": {"cancelled": event is not None},
                        },
                    )
                    continue
                cancelled = Event()
                with self._active_lock:
                    if request.id in self._active:
                        self._write(
                            output,
                            {
                                "version": PROTOCOL_VERSION,
                                "id": request.id,
                                "type": "error",
                                "code": "duplicate_id",
                                "message": "request ID is already active",
                            },
                        )
                        continue
                    self._active[request.id] = cancelled
                executor.submit(self._run, request, cancelled, output)

    def _run(self, request: RpcRequest, cancelled: Event, output: TextIO) -> None:
        def emit(event_type: str, payload: dict[str, Any]) -> None:
            self._write(
                output,
                {
                    "version": PROTOCOL_VERSION,
                    "id": request.id,
                    "type": event_type,
                    "payload": payload,
                },
            )

        try:
            result = self.handler(request, cancelled, emit)
            if cancelled.is_set():
                raise InterruptedError("request cancelled")
            emit("result", result)
        except InterruptedError:
            self._write(
                output,
                {
                    "version": PROTOCOL_VERSION,
                    "id": request.id,
                    "type": "error",
                    "code": "cancelled",
                    "message": "request cancelled",
                },
            )
        except Exception as error:
            self._write(
                output,
                {
                    "version": PROTOCOL_VERSION,
                    "id": request.id,
                    "type": "error",
                    "code": "request_failed",
                    "message": str(error),
                },
            )
        finally:
            with self._active_lock:
                self._active.pop(request.id, None)

    def _write(self, output: TextIO, event: dict[str, Any]) -> None:
        serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._output_lock:
            output.write(serialized + "\n")
            output.flush()

import io
import json
import time
from threading import Event

import pytest

from chat_rag.rpc import PROTOCOL_VERSION, RpcRequest, StdioServer, parse_request


def request(request_id: str, method: str, params: dict | None = None) -> str:
    return json.dumps(
        {"version": PROTOCOL_VERSION, "id": request_id, "method": method, "params": params or {}}
    )


def test_request_schema_rejects_protocol_version_mismatch() -> None:
    with pytest.raises(ValueError, match="version"):
        parse_request('{"version":99,"id":"x","method":"stats","params":{}}')


def test_malformed_line_does_not_crash_later_request() -> None:
    source = io.StringIO("not-json\n" + request("ok", "stats") + "\n")
    output = io.StringIO()

    def handler(req: RpcRequest, cancelled: Event, emit) -> dict:
        return {"messages": 2}

    StdioServer(handler, max_workers=1).serve(source, output)
    events = [json.loads(line) for line in output.getvalue().splitlines()]

    assert any(event["type"] == "error" for event in events)
    assert any(event["id"] == "ok" and event["type"] == "result" for event in events)


def test_cancelled_request_and_recoverable_error_do_not_end_session() -> None:
    source = io.StringIO(
        "\n".join(
            [
                request("slow", "ask", {"question": "wait"}),
                request("cancel-1", "cancel", {"request_id": "slow"}),
                request("bad", "search", {"query": "fail"}),
                request("later", "stats"),
            ]
        )
        + "\n"
    )
    output = io.StringIO()

    def handler(req: RpcRequest, cancelled: Event, emit) -> dict:
        if req.id == "slow":
            while not cancelled.wait(0.001):
                time.sleep(0.001)
            raise InterruptedError("cancelled")
        if req.id == "bad":
            raise ValueError("recoverable")
        emit("progress", {"stage": "done"})
        return {"ok": True}

    StdioServer(handler, max_workers=2).serve(source, output)
    events = [json.loads(line) for line in output.getvalue().splitlines()]

    assert any(
        event["id"] == "slow" and event["type"] == "error" and event["code"] == "cancelled"
        for event in events
    )
    assert any(event["id"] == "bad" and event["type"] == "error" for event in events)
    assert any(event["id"] == "later" and event["type"] == "result" for event in events)
    assert any(event["id"] == "cancel-1" and event["type"] == "result" for event in events)

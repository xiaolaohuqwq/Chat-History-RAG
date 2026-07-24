from threading import Event

import pytest

from chat_rag.config import Settings
from chat_rag.rpc import PROTOCOL_VERSION, RpcRequest
from chat_rag.rpc_app import RpcApplication


def test_application_rejects_invalid_method_parameters_before_provider_calls() -> None:
    app = RpcApplication(Settings(_env_file=None))
    request = RpcRequest(
        version=PROTOCOL_VERSION,
        id="x",
        method="search",
        params={"query": "valid query", "no_rerank": "false"},
    )
    with pytest.raises(ValueError, match="no_rerank"):
        app(request, Event(), lambda event_type, payload: None)

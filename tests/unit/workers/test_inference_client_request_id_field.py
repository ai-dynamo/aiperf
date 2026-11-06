# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPERF_ENDPOINT_REQUEST_ID_FIELD: the X-Request-ID value is also written into the body.

Covers every payload source in ``InferenceClient._send_request_to_transport``
(format_payload, turns[-1].raw_payload, pre-encoded payload_bytes) and that
shared payloads are never mutated.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.environment import Environment
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.workers.inference_client import InferenceClient, with_request_id

REQUEST_ID = "3f1c9a52-0d7e-4b8e-9a51-6c2f4e8d1b70"


@pytest.fixture
def model_endpoint() -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_url="http://localhost:8000/v1/test",
        ),
    )


def _client(
    model_endpoint: ModelEndpointInfo,
    monkeypatch: pytest.MonkeyPatch,
    *,
    field: str | None,
    formatted: dict | None = None,
) -> InferenceClient:
    monkeypatch.setattr(Environment.ENDPOINT, "REQUEST_ID_FIELD", field)
    transport_entry = MagicMock()
    transport_entry.name = TransportType.HTTP.value
    transport_entry.metadata = {"url_schemes": ["http", "https"]}
    endpoint = MagicMock()
    endpoint.get_endpoint_headers.return_value = {}
    endpoint.get_endpoint_params.return_value = {}
    endpoint.format_payload.return_value = formatted or {}

    def get_class(protocol, name):
        if protocol == "endpoint":
            return lambda **kwargs: endpoint
        if protocol == "transport":
            return lambda **kwargs: MagicMock()
        raise ValueError(f"Unknown protocol: {protocol}")

    with (
        patch(
            "aiperf.workers.inference_client.plugins.get_class",
            side_effect=get_class,
        ),
        patch(
            "aiperf.workers.inference_client.plugins.list_entries",
            return_value=[transport_entry],
        ),
    ):
        return InferenceClient(
            model_endpoint=model_endpoint, service_id="test-service-id"
        )


def _request_info(
    model_endpoint: ModelEndpointInfo,
    *,
    turns: list[Turn] | None = None,
    payload_bytes: bytes | None = None,
) -> RequestInfo:
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=turns if turns is not None else [],
        turn_index=0,
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        x_request_id=REQUEST_ID,
        x_correlation_id="cid",
        conversation_id="conv",
        payload_bytes=payload_bytes,
    )


async def _sent_body(client: InferenceClient, info: RequestInfo) -> dict:
    client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )
    await client.send_request(info)
    wire = client.transport.send_request.call_args.kwargs["payload"]
    # The raw export reads payload_bytes, so it must hold what went on the wire.
    assert wire == info.payload_bytes
    return orjson.loads(wire)


@pytest.mark.asyncio
async def test_format_payload_gets_request_id_without_mutating_the_formatted_dict(
    model_endpoint, monkeypatch
):
    formatted = {"model": "test-model", "messages": []}
    client = _client(model_endpoint, monkeypatch, field="rid", formatted=formatted)
    turn = Turn(texts=[Text(contents=["hi"])])

    body = await _sent_body(client, _request_info(model_endpoint, turns=[turn]))

    assert body == {"model": "test-model", "messages": [], "rid": REQUEST_ID}
    assert "rid" not in formatted


@pytest.mark.asyncio
async def test_raw_payload_turn_is_not_mutated(model_endpoint, monkeypatch):
    raw = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    turn = Turn(
        role="user", raw_payload=raw, texts=[Text(contents=["hi"])], model="test-model"
    )
    client = _client(model_endpoint, monkeypatch, field="rid")

    body = await _sent_body(client, _request_info(model_endpoint, turns=[turn]))

    assert body["rid"] == REQUEST_ID
    assert "rid" not in turn.raw_payload


@pytest.mark.asyncio
async def test_preencoded_payload_bytes_get_request_id(model_endpoint, monkeypatch):
    shared = orjson.dumps({"model": "test-model", "prompt": "hi"})
    client = _client(model_endpoint, monkeypatch, field="rid")

    body = await _sent_body(client, _request_info(model_endpoint, payload_bytes=shared))

    assert body == {"model": "test-model", "prompt": "hi", "rid": REQUEST_ID}


@pytest.mark.asyncio
async def test_unset_field_sends_payload_bytes_verbatim(model_endpoint, monkeypatch):
    shared = orjson.dumps({"model": "test-model", "prompt": "hi"})
    client = _client(model_endpoint, monkeypatch, field=None)

    await _sent_body(client, _request_info(model_endpoint, payload_bytes=shared))

    assert client.transport.send_request.call_args.kwargs["payload"] is shared


def test_dataset_supplied_field_is_overwritten_so_server_and_export_ids_agree():
    assert with_request_id({"rid": "from-dataset"}, "rid", REQUEST_ID) == {
        "rid": REQUEST_ID
    }


@pytest.mark.parametrize("payload", [b"not json", b"[1, 2]"])
def test_non_object_bytes_are_forwarded_verbatim(payload: bytes):
    assert with_request_id(payload, "rid", REQUEST_ID) is payload

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial coverage for InferenceClient payload_bytes fast path.

Pins behaviour of the priority chain in
``InferenceClient._send_request_to_transport``:

    request_info.payload_bytes
        -> turns[-1].raw_payload
        -> endpoint.format_payload(request_info)

and the empty-turns guard in ``send_request`` (relaxed to accept
turn-less requests when ``payload_bytes`` is present).

Note: per-request orjson round-trip validation of pre-serialised
``payload_bytes`` was removed — invalid-JSON detection now happens at
dataset-load time, not on every send. ``payload_bytes`` is forwarded to
the transport verbatim.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import orjson
import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy, RequestContentType
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.workers.inference_client import InferenceClient


@pytest.fixture
def mock_http_transport_entry():
    entry = MagicMock()
    entry.name = TransportType.HTTP.value
    entry.metadata = {"url_schemes": ["http", "https"]}
    return entry


@pytest.fixture
def model_endpoint():
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


@pytest.fixture
def inference_client(model_endpoint, mock_http_transport_entry):
    mock_transport = MagicMock()
    mock_endpoint = MagicMock()
    mock_endpoint.get_endpoint_headers.return_value = {}
    mock_endpoint.get_endpoint_params.return_value = {}
    mock_endpoint.format_payload.return_value = {"from": "format_payload"}

    def mock_get_class(protocol, name):
        if protocol == "endpoint":
            return lambda **kwargs: mock_endpoint
        if protocol == "transport":
            return lambda **kwargs: mock_transport
        raise ValueError(f"Unknown protocol: {protocol}")

    with (
        patch(
            "aiperf.workers.inference_client.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch(
            "aiperf.workers.inference_client.plugins.list_entries",
            return_value=[mock_http_transport_entry],
        ),
    ):
        return InferenceClient(
            model_endpoint=model_endpoint, service_id="test-service-id"
        )


def _make_request_info(
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
        x_request_id="rid",
        x_correlation_id="cid",
        conversation_id="conv",
        payload_bytes=payload_bytes,
    )


@pytest.mark.asyncio
async def test_send_request_allows_empty_turns_with_payload_bytes(
    inference_client, model_endpoint
):
    """Empty turns are accepted when payload_bytes is set."""
    info = _make_request_info(model_endpoint, turns=[], payload_bytes=b'{"a":1}')
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    record = await inference_client.send_request(info)

    assert record is not None
    call_args = inference_client.transport.send_request.call_args
    assert call_args.kwargs["payload"] == b'{"a":1}'
    inference_client.endpoint.format_payload.assert_not_called()


@pytest.mark.asyncio
async def test_send_request_rejects_empty_turns_and_none_payload_bytes(
    inference_client, model_endpoint
):
    """Both empty must raise (guard still holds when neither source is set)."""
    info = _make_request_info(model_endpoint, turns=[], payload_bytes=None)

    with pytest.raises(ValueError, match="no turns"):
        await inference_client.send_request(info)


@pytest.mark.asyncio
async def test_send_request_empty_bytes_payload_bytes_with_empty_turns_behavior(
    inference_client, model_endpoint
):
    """Pin current behaviour for ``payload_bytes=b""`` + empty turns.

    Empty bytes are falsy, so the ``not request_info.payload_bytes``
    guard in ``send_request`` currently treats this identically to
    ``payload_bytes=None`` and raises.
    """
    info = _make_request_info(model_endpoint, turns=[], payload_bytes=b"")

    with pytest.raises(ValueError, match="no turns"):
        await inference_client.send_request(info)


@pytest.mark.asyncio
async def test_send_request_dict_raw_payload_serialized_and_cached_on_payload_bytes(
    inference_client, model_endpoint
):
    """dict raw_payload on last turn is serialised and cached back on request_info."""
    raw = {"messages": [{"role": "user", "content": "hi"}], "model": "m"}
    turn = Turn(
        role="user",
        raw_payload=raw,
        texts=[Text(contents=["hi"])],
        model="test-model",
    )
    info = _make_request_info(model_endpoint, turns=[turn], payload_bytes=None)
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    await inference_client.send_request(info)

    expected = orjson.dumps(raw)
    call_args = inference_client.transport.send_request.call_args
    assert call_args.kwargs["payload"] == expected
    assert info.payload_bytes == expected
    inference_client.endpoint.format_payload.assert_not_called()


@pytest.mark.asyncio
async def test_send_request_payload_bytes_takes_priority_over_raw_payload(
    inference_client, model_endpoint
):
    """payload_bytes wins over turn.raw_payload when both are set."""
    raw = {"from": "raw_payload"}
    turn = Turn(
        role="user",
        raw_payload=raw,
        texts=[Text(contents=["x"])],
        model="test-model",
    )
    pre_bytes = b'{"from":"payload_bytes"}'
    info = _make_request_info(model_endpoint, turns=[turn], payload_bytes=pre_bytes)
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    await inference_client.send_request(info)

    call_args = inference_client.transport.send_request.call_args
    assert call_args.kwargs["payload"] == pre_bytes
    assert info.payload_bytes == pre_bytes
    inference_client.endpoint.format_payload.assert_not_called()


@pytest.mark.asyncio
async def test_send_request_raw_payload_fallback_when_payload_bytes_none(
    inference_client, model_endpoint
):
    """With payload_bytes=None, turn.raw_payload is used (not format_payload)."""
    raw = {"from": "raw_payload"}
    turn = Turn(
        role="user",
        raw_payload=raw,
        texts=[Text(contents=["x"])],
        model="test-model",
    )
    info = _make_request_info(model_endpoint, turns=[turn], payload_bytes=None)
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    await inference_client.send_request(info)

    call_args = inference_client.transport.send_request.call_args
    assert call_args.kwargs["payload"] == orjson.dumps(raw)
    inference_client.endpoint.format_payload.assert_not_called()


@pytest.mark.asyncio
async def test_send_request_format_payload_fallback_when_no_raw_payload_no_bytes(
    inference_client, model_endpoint
):
    """Without payload_bytes and without raw_payload, format_payload is called."""
    turn = Turn(texts=[Text(contents=["x"])], role="user", model="test-model")
    info = _make_request_info(model_endpoint, turns=[turn], payload_bytes=None)

    formatted = {"from": "format_payload"}
    inference_client.endpoint.format_payload.return_value = formatted
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    await inference_client.send_request(info)

    inference_client.endpoint.format_payload.assert_called_once_with(info)
    call_args = inference_client.transport.send_request.call_args
    # dict is canonicalised into orjson bytes before transport.
    assert call_args.kwargs["payload"] == orjson.dumps(formatted)
    assert info.payload_bytes == orjson.dumps(formatted)


@pytest.mark.asyncio
async def test_send_request_format_payload_raises_not_implemented_propagates(
    inference_client, model_endpoint
):
    """NotImplementedError from format_payload flows out through the error record path.

    ``send_request`` wraps transport errors into an error ``RequestRecord``
    rather than re-raising. ``_send_request_to_transport`` is called
    from inside ``_send_request_internal`` which catches ``Exception``
    — ``NotImplementedError`` is an ``Exception`` subclass, so it gets
    converted into an error record with the exception preserved on
    ``record.error``.
    """
    turn = Turn(texts=[Text(contents=["x"])], role="user", model="test-model")
    info = _make_request_info(model_endpoint, turns=[turn], payload_bytes=None)

    inference_client.endpoint.format_payload.side_effect = NotImplementedError(
        "RawEndpoint does not construct payloads"
    )
    inference_client.transport.send_request = AsyncMock()

    record = await inference_client.send_request(info)

    inference_client.transport.send_request.assert_not_called()
    assert record.error is not None
    assert "RawEndpoint" in record.error.message or "NotImplementedError" in str(
        record.error
    )


@pytest.mark.asyncio
async def test_send_request_payload_bytes_unicode_bytes_sent_verbatim(
    inference_client, model_endpoint
):
    """Non-ASCII UTF-8 bytes in payload_bytes flow through byte-for-byte."""
    body = '{"msg":"héllo"}'.encode()
    info = _make_request_info(model_endpoint, turns=[], payload_bytes=body)
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    await inference_client.send_request(info)

    call_args = inference_client.transport.send_request.call_args
    sent = call_args.kwargs["payload"]
    assert sent == body
    # Preserve the bytes exactly: the UTF-8 encoding of 'é' is 0xc3 0xa9.
    assert b"\xc3\xa9" in sent
    inference_client.endpoint.format_payload.assert_not_called()


@pytest.mark.asyncio
async def test_inference_client_forwards_invalid_json_payload_bytes_verbatim(
    inference_client, model_endpoint
):
    """Per-request ``orjson.loads`` validation of pre-serialised
    ``payload_bytes`` was removed — invalid-JSON detection happens at
    dataset-load time, not on every send. Unparsable bytes are forwarded to
    the transport verbatim rather than being turned into an error record."""
    info = _make_request_info(model_endpoint, turns=[], payload_bytes=b"}")
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    await inference_client.send_request(info)

    inference_client.transport.send_request.assert_called_once()
    call_args = inference_client.transport.send_request.call_args
    assert call_args.kwargs["payload"] == b"}"


@pytest.mark.asyncio
async def test_valid_payload_bytes_sent_verbatim(inference_client, model_endpoint):
    """Well-formed JSON ``payload_bytes`` is sent verbatim."""
    body = b'{"messages":[{"role":"user","content":"hi"}]}'
    info = _make_request_info(model_endpoint, turns=[], payload_bytes=body)
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    record = await inference_client._send_request_to_transport(info)

    assert record is not None
    call_args = inference_client.transport.send_request.call_args
    assert call_args.kwargs["payload"] == body


@pytest.mark.asyncio
async def test_multipart_endpoint_keeps_dict_payload_for_transport(
    mock_http_transport_entry,
):
    """Multipart endpoints must receive the structured dict so the transport can
    build FormData; payload_bytes still carries the JSON for records/raw-export.

    Guards the multipart carve-out: without it the dict would be pre-dumped to
    JSON bytes and sent verbatim to a multipart endpoint, which the server's
    form parser rejects (HTTP 422, prompt field required / input null).
    """
    model_endpoint = ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.IMAGE_EDIT,
            base_url="http://localhost:8000/v1/images/edits",
            request_content_type=RequestContentType.MULTIPART_FORM_DATA,
        ),
    )
    payload = {"prompt": "a cat wearing a hat", "model": "test-model"}
    mock_transport = MagicMock()
    mock_transport.send_request = AsyncMock(return_value=MagicMock(spec=RequestRecord))
    mock_endpoint = MagicMock()
    mock_endpoint.get_endpoint_headers.return_value = {}
    mock_endpoint.get_endpoint_params.return_value = {}
    mock_endpoint.format_payload.return_value = payload

    def mock_get_class(protocol, name):
        if protocol == "endpoint":
            return lambda **kwargs: mock_endpoint
        if protocol == "transport":
            return lambda **kwargs: mock_transport
        raise ValueError(f"Unknown protocol: {protocol}")

    with (
        patch(
            "aiperf.workers.inference_client.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch(
            "aiperf.workers.inference_client.plugins.list_entries",
            return_value=[mock_http_transport_entry],
        ),
    ):
        client = InferenceClient(
            model_endpoint=model_endpoint, service_id="test-service-id"
        )

    turn = Turn(texts=[Text(contents=["a cat wearing a hat"])])
    info = _make_request_info(model_endpoint, turns=[turn], payload_bytes=None)
    await client._send_request_to_transport(info)

    wire_payload = mock_transport.send_request.call_args.kwargs["payload"]
    assert isinstance(wire_payload, dict), (
        "multipart endpoint must receive the structured dict, not pre-dumped bytes"
    )
    assert wire_payload == payload
    # payload_bytes (records / raw-export) still carries the JSON-encoded form.
    assert info.payload_bytes == orjson.dumps(payload)


@pytest.mark.asyncio
async def test_send_request_trace_logging_skips_turn_dump_with_empty_turns(
    inference_client, model_endpoint
):
    """With TRACE enabled and no turns (payload-bytes path), the per-turn trace
    dump must be skipped instead of crashing on ``turns[-1]``."""
    info = _make_request_info(model_endpoint, turns=[], payload_bytes=b'{"a":1}')
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=info)
    )

    with patch.object(
        type(inference_client), "is_trace_enabled", PropertyMock(return_value=True)
    ):
        record = await inference_client.send_request(info)

    assert record is not None
    call_args = inference_client.transport.send_request.call_args
    assert call_args.kwargs["payload"] == b'{"a":1}'

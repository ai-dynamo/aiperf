# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 3a.5: ``InferenceResultsMessage`` is a ``msgspec.Struct``.

Pins the post-migration contract:

* Construction is kwargs-only.
* ``msgspec.msgpack`` encode/decode round-trips natively -- no Pydantic bridge.
* ``message_type`` is on the wire as a top-level discriminator (via msgspec
  ``tag_field``) so the wire codec routes inbound bytes back to this class.
* ``MetricInputs.payload_bytes`` (``bytes | None``) rides the wire as a
  length-prefixed msgpack ``bin`` span -- no base64, byte-identical round-trip.
* The wire codec dispatches encode/decode by encoder family.
"""

from __future__ import annotations

import msgspec
import orjson

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages import InferenceResultsMessage, Message
from aiperf.common.messages.wire_codec import decode_message, encode_message
from aiperf.common.models import (
    BinaryResponse,
    ErrorDetails,
    MetricInputs,
    RequestRecord,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.record_models import SSEField


def _base_metric_inputs(payload: bytes | None = None, **overrides) -> MetricInputs:
    base = dict(
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        conversation_id="conv",
        turn_index=0,
        x_request_id="req",
        x_correlation_id="corr",
    )
    base.update(overrides)
    if payload is not None:
        base["payload_bytes"] = payload
    return MetricInputs(**base)


def _full_record() -> RequestRecord:
    """RequestRecord with every embedded type populated for round-trip coverage."""
    return RequestRecord(
        metric_inputs=_base_metric_inputs(
            payload=b'{"messages":[{"role":"user","content":"hi"}]}'
        ),
        model_name="gpt-4",
        status=200,
        responses=[
            SSEMessage(
                perf_ns=100,
                packets=[SSEField(name="data", value='{"choices":[]}')],
            ),
            TextResponse(perf_ns=200, text="hello"),
            BinaryResponse(perf_ns=300, raw_bytes=b"\x00\x01"),
        ],
        error=ErrorDetails(code=500, message="boom"),
    )


class TestConstruction:
    def test_kwargs_only(self) -> None:
        m = InferenceResultsMessage(service_id="w1", record=RequestRecord())
        assert m.service_id == "w1"
        assert isinstance(m.record, RequestRecord)
        assert m.request_id is None
        assert m.request_ns is None
        # message_type is exposed via property reading the tag.
        assert m.message_type == MessageType.INFERENCE_RESULTS
        assert m.message_type == "inference_results"

    def test_is_msgspec_struct(self) -> None:
        m = InferenceResultsMessage(service_id="w1", record=RequestRecord())
        assert isinstance(m, msgspec.Struct)


class TestWireEncode:
    def test_message_type_is_first_wire_field(self) -> None:
        """tag_field='message_type' guarantees the discriminator on the wire."""
        m = InferenceResultsMessage(service_id="w1", record=RequestRecord())
        wire = encode_message(m)
        # msgpack wire -- decode via msgspec generic decoder.
        data = msgspec.msgpack.decode(wire)
        assert data["message_type"] == "inference_results"
        assert data["service_id"] == "w1"

    def test_none_request_fields_omitted(self) -> None:
        """omit_defaults strips None-defaulted request_id / request_ns."""
        m = InferenceResultsMessage(service_id="w1", record=RequestRecord())
        wire = encode_message(m)
        data = msgspec.msgpack.decode(wire)
        assert "request_id" not in data
        assert "request_ns" not in data

    def test_payload_bytes_rides_as_msgpack_bin(self) -> None:
        """payload_bytes rides as a length-prefixed msgpack bin span; no base64."""
        payload = b'{"messages":[{"role":"user","content":"\xc3\xa9"}]}'
        m = InferenceResultsMessage(
            service_id="w1",
            record=RequestRecord(
                metric_inputs=_base_metric_inputs(payload=payload),
            ),
        )
        wire = encode_message(m)
        # The raw payload bytes appear verbatim inside the msgpack bin span --
        # not base64-encoded, not re-encoded.
        assert payload in wire
        # Round-trip preserves bytes exactly.
        m2 = decode_message(wire)
        assert isinstance(m2, InferenceResultsMessage)
        assert m2.record.metric_inputs.payload_bytes == payload


class TestWireRoundTrip:
    def test_minimal_round_trip(self) -> None:
        m = InferenceResultsMessage(service_id="w1", record=RequestRecord())
        wire = encode_message(m)
        m2 = decode_message(wire)
        assert isinstance(m2, InferenceResultsMessage)
        assert m2.service_id == "w1"
        assert m2.message_type == "inference_results"

    def test_full_record_round_trip_byte_for_byte(self) -> None:
        """Encoding twice yields identical bytes."""
        m = InferenceResultsMessage(service_id="w1", record=_full_record())
        wire1 = encode_message(m)
        m2 = decode_message(wire1)
        assert isinstance(m2, InferenceResultsMessage)
        wire2 = encode_message(m2)
        assert wire1 == wire2

    def test_polymorphic_responses_round_trip(self) -> None:
        m = InferenceResultsMessage(service_id="w1", record=_full_record())
        wire = encode_message(m)
        m2 = decode_message(wire)
        assert isinstance(m2, InferenceResultsMessage)
        responses = m2.record.responses
        assert len(responses) == 3
        assert isinstance(responses[0], SSEMessage)
        assert isinstance(responses[1], TextResponse)
        assert isinstance(responses[2], BinaryResponse)
        assert responses[2].raw_bytes == b"\x00\x01"


class TestWireCodecDispatch:
    def test_decode_routes_to_msgspec_for_registered_type(self) -> None:
        m = InferenceResultsMessage(service_id="w1", record=RequestRecord())
        wire = encode_message(m)
        decoded = decode_message(wire)
        assert type(decoded).__name__ == "InferenceResultsMessage"

    def test_decode_falls_back_to_pydantic_for_unregistered_type(self) -> None:
        """Pydantic ErrorMessage path still works via AutoRoutedModel."""
        from aiperf.common.messages import ErrorMessage
        from aiperf.common.models import ErrorDetails as ED

        m = ErrorMessage(error=ED(code=1, message="boom"))
        wire = encode_message(m)
        decoded = decode_message(wire)
        assert isinstance(decoded, ErrorMessage)
        assert isinstance(decoded, Message)
        assert decoded.error.message == "boom"

    def test_encode_dispatches_by_family(self) -> None:
        """Pydantic encodes to JSON; msgspec encodes to msgpack."""
        msg_msgspec = InferenceResultsMessage(service_id="w1", record=RequestRecord())
        from aiperf.common.messages import ErrorMessage
        from aiperf.common.models import ErrorDetails as ED

        msg_pydantic = ErrorMessage(error=ED(code=1, message="x"))

        wire_msgspec = encode_message(msg_msgspec)
        wire_pydantic = encode_message(msg_pydantic)

        # Pydantic -> JSON (starts with '{', decodable by orjson).
        assert wire_pydantic[0:1] == b"{"
        assert orjson.loads(wire_pydantic)["message_type"] == "error"

        # msgspec -> msgpack (starts with a fixmap byte 0x80-0x8f, not '{').
        assert wire_msgspec[0:1] != b"{"
        assert (
            msgspec.msgpack.decode(wire_msgspec)["message_type"] == "inference_results"
        )

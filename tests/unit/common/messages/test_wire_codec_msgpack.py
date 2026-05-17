# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3c: records-pipeline wire is msgpack (binary), not JSON.

Pins the post-Phase-3c contract:

* Registered msgspec.Struct messages encode via msgspec.msgpack.Encoder --
  wire bytes are msgpack, NOT JSON. First byte is a fixmap (0x80-0x8f),
  map16 (0xde), or map32 (0xdf) prefix; never ``{`` (0x7b).
* The registered tagged-union decoder reconstructs the concrete message type
  directly from the ``message_type`` tag.
* Round-trip through the wire codec is byte-identical and preserves every
  embedded Struct (MetricInputs, tagged union responses, tagged union
  trace_data, ErrorDetails).
* Pydantic messages still ride the wire as JSON via the same codec dispatch
  -- the codec sniffs the first byte to route inbound traffic to the right
  family.
"""

from __future__ import annotations

import msgspec
import orjson

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages import (
    ErrorMessage,
    InferenceResultsMessage,
    MetricRecordsMessage,
    RealtimeMetricsMessage,
)
from aiperf.common.messages.wire_codec import decode_message, encode_message
from aiperf.common.models import (
    ErrorDetails,
    MetricInputs,
    MetricRecordMetadata,
    MetricResult,
    RequestRecord,
)


def _metric_inputs() -> MetricInputs:
    return MetricInputs(
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        conversation_id="c",
        turn_index=0,
        x_request_id="r",
        x_correlation_id="x",
        payload_bytes=b'{"messages":[{"role":"user","content":"hi"}]}',
    )


def _metadata() -> MetricRecordMetadata:
    return MetricRecordMetadata(
        session_num=1,
        request_start_ns=100,
        request_end_ns=200,
        worker_id="w1",
        record_processor_id="rp1",
        benchmark_phase=CreditPhase.PROFILING,
    )


class TestMsgpackEncoding:
    """Wire bytes for registered msgspec messages are msgpack, not JSON."""

    def test_inference_results_wire_is_msgpack(self) -> None:
        msg = InferenceResultsMessage(
            service_id="w1",
            record=RequestRecord(metric_inputs=_metric_inputs()),
        )
        wire = encode_message(msg)
        # JSON would lead with ``{`` (0x7b); msgpack maps lead with 0x80-0x8f
        # (fixmap) or 0xde/0xdf (map16/map32).
        assert wire[0:1] != b"{", (
            f"wire still appears to be JSON, not msgpack: {wire[:20]!r}"
        )
        first = wire[0]
        assert 0x80 <= first <= 0x8F or first in (0xDE, 0xDF), (
            f"first byte {first:#x} is not a msgpack map prefix"
        )

    def test_metric_records_wire_is_msgpack(self) -> None:
        msg = MetricRecordsMessage(metadata=_metadata(), metrics={"a": 1.0})
        wire = encode_message(msg)
        assert wire[0:1] != b"{"
        # Round-trip via generic decode recovers the dict mirror.
        data = msgspec.msgpack.decode(wire)
        assert data["message_type"] == "metric_records"

    def test_realtime_metrics_wire_is_msgpack(self) -> None:
        msg = RealtimeMetricsMessage(
            metrics=[MetricResult(tag="lat", header="Lat", unit="ms", avg=1.5)]
        )
        wire = encode_message(msg)
        assert wire[0:1] != b"{"
        data = msgspec.msgpack.decode(wire)
        assert data["message_type"] == "realtime_metrics"

    def test_pydantic_message_still_rides_as_json(self) -> None:
        """The codec sniffs the first byte; Pydantic messages remain JSON."""
        msg = ErrorMessage(error=ErrorDetails(code=1, message="boom"))
        wire = encode_message(msg)
        # JSON leads with ``{``.
        assert wire[0:1] == b"{"
        # Round-trip via orjson confirms it parses as JSON.
        data = orjson.loads(wire)
        assert data["message_type"] == "error"


class TestMsgpackRoundTrip:
    """Decode_message routes msgpack bytes back to the registered Struct."""

    def test_inference_results_round_trip(self) -> None:
        msg = InferenceResultsMessage(
            service_id="w1",
            record=RequestRecord(metric_inputs=_metric_inputs()),
        )
        wire = encode_message(msg)
        rt = decode_message(wire)
        assert isinstance(rt, InferenceResultsMessage)
        assert rt.service_id == "w1"
        assert rt.message_type == MessageType.INFERENCE_RESULTS
        assert rt.record.metric_inputs is not None
        assert (
            rt.record.metric_inputs.payload_bytes_or_none
            == b'{"messages":[{"role":"user","content":"hi"}]}'
        )

    def test_payload_bytes_rides_as_bin_span(self) -> None:
        """Payload bytes appear verbatim in the msgpack envelope (no base64)."""
        payload = b'{"messages":[{"role":"user","content":"x"}]}'
        mi = MetricInputs(
            credit_num=0,
            credit_phase=CreditPhase.PROFILING,
            conversation_id="c",
            turn_index=0,
            x_request_id="r",
            x_correlation_id="x",
            payload_bytes=payload,
        )
        msg = InferenceResultsMessage(
            service_id="w1", record=RequestRecord(metric_inputs=mi)
        )
        wire = encode_message(msg)
        # The payload bytes appear verbatim inside the msgpack envelope, not
        # base64-encoded and not JSON-spliced.
        assert payload in wire

    def test_metric_records_round_trip(self) -> None:
        msg = MetricRecordsMessage(
            service_id="svc",
            metadata=_metadata(),
            metrics={"a": 1.5, "b": 2.0},
        )
        wire = encode_message(msg)
        rt = decode_message(wire)
        assert isinstance(rt, MetricRecordsMessage)
        assert rt.service_id == "svc"
        assert rt.metrics == {"a": 1.5, "b": 2.0}

    def test_realtime_metrics_round_trip(self) -> None:
        mr = MetricResult(tag="lat", header="Lat", unit="ms", avg=1.5, p99=99.0)
        msg = RealtimeMetricsMessage(service_id="rm", metrics=[mr])
        wire = encode_message(msg)
        rt = decode_message(wire)
        assert isinstance(rt, RealtimeMetricsMessage)
        assert rt.service_id == "rm"
        assert rt.metrics[0].tag == "lat"
        assert rt.metrics[0].p99 == 99.0

    def test_byte_for_byte_round_trip(self) -> None:
        msg = MetricRecordsMessage(
            service_id="svc",
            metadata=_metadata(),
            metrics={"a": 1.5},
            error=ErrorDetails(message="x", code=1),
        )
        wire1 = encode_message(msg)
        rt = decode_message(wire1)
        assert isinstance(rt, MetricRecordsMessage)
        wire2 = encode_message(rt)
        assert wire1 == wire2


class TestMixedFamilyDispatch:
    """The codec routes JSON and msgpack to the right family on the same channel."""

    def test_codec_routes_msgpack_to_msgspec(self) -> None:
        msg = InferenceResultsMessage(service_id="w1", record=RequestRecord())
        rt = decode_message(encode_message(msg))
        assert isinstance(rt, InferenceResultsMessage)

    def test_msgpack_decode_uses_single_typed_decoder(self, monkeypatch) -> None:
        msg = InferenceResultsMessage(service_id="w1", record=RequestRecord())
        wire = encode_message(msg)

        def fail_generic_decode(*args: object, **kwargs: object) -> None:
            raise AssertionError("generic msgpack decode should not run")

        monkeypatch.setattr(msgspec.msgpack, "decode", fail_generic_decode)

        rt = decode_message(wire)
        assert isinstance(rt, InferenceResultsMessage)
        assert rt.service_id == "w1"

    def test_codec_routes_json_to_pydantic(self) -> None:
        msg = ErrorMessage(error=ErrorDetails(code=1, message="boom"))
        rt = decode_message(encode_message(msg))
        assert isinstance(rt, ErrorMessage)
        assert rt.error.message == "boom"

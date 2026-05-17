# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 3a.4: ``RequestRecord`` is a ``msgspec.Struct``.

Pins the post-migration contract:

* Construction is kwargs-only, all fields default-able.
* Every embedded msgspec.Struct type (``MetricInputs``, the
  ``InferenceServerResponseUnion`` tagged union, ``ErrorDetails``,
  ``BaseTraceDataUnion`` tagged union) round-trips natively through
  ``msgspec.msgpack`` (the records-pipeline wire encoding) -- no Pydantic
  bridge.
* ``omit_defaults=True`` strips None-defaulted fields from the wire so the
  shape matches the prior Pydantic ``exclude_none=True`` form.
* ``has_error`` / ``valid`` / ``was_cancelled`` properties still work.
"""

from __future__ import annotations

import msgspec
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    BinaryResponse,
    ErrorDetails,
    MetricInputs,
    RequestRecord,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.record_models import SSEField
from aiperf.common.models.trace_models import AioHttpTraceData


def _base_metric_inputs(**overrides) -> MetricInputs:
    """Minimal MetricInputs honoring the required-field contract."""
    base = dict(
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        conversation_id="conv",
        turn_index=0,
        x_request_id="req",
        x_correlation_id="corr",
    )
    base.update(overrides)
    return MetricInputs(**base)


class TestConstruction:
    """Default construction + kwargs-only contract."""

    def test_default_constructor_no_args(self) -> None:
        """All fields default; bare ``RequestRecord()`` is valid."""
        rec = RequestRecord()
        assert rec.metric_inputs is None
        assert rec.responses == []
        assert rec.error is None
        assert rec.trace_data is None
        assert rec.status is None
        # Timestamp factories fire.
        assert rec.timestamp_ns > 0
        assert rec.start_perf_ns > 0

    def test_full_kwargs_construction(self) -> None:
        mi = _base_metric_inputs()
        err = ErrorDetails(message="x", type="X")
        trace = AioHttpTraceData(reference_time_ns=1, reference_perf_ns=2)
        resp = TextResponse(perf_ns=10, text="ok")
        rec = RequestRecord(
            metric_inputs=mi,
            request_headers={"a": "b"},
            model_name="m",
            timestamp_ns=100,
            start_perf_ns=200,
            end_perf_ns=300,
            recv_start_perf_ns=250,
            status=200,
            responses=[resp],
            error=err,
            credit_drop_latency=42,
            cancellation_perf_ns=290,
            trace_data=trace,
        )
        assert rec.metric_inputs is mi
        assert rec.responses == [resp]
        assert rec.error is err
        assert rec.trace_data is trace
        assert rec.status == 200
        assert rec.cancellation_perf_ns == 290


class TestRoundTrip:
    """msgspec.msgpack round-trips preserve every embedded Struct type."""

    @pytest.mark.parametrize(
        "payload",
        [
            param(b'{"a":1}', id="payload_bytes"),
            param(None, id="none_payload"),
        ],
    )
    def test_metric_inputs_roundtrip(self, payload: bytes | None) -> None:
        mi = _base_metric_inputs(payload_bytes=payload)
        rec = RequestRecord(metric_inputs=mi)
        wire = msgspec.msgpack.encode(rec)
        rt = msgspec.msgpack.decode(wire, type=RequestRecord)
        assert rt.metric_inputs is not None
        assert rt.metric_inputs.credit_num == mi.credit_num
        assert rt.metric_inputs.payload_bytes_or_none == payload
        if payload is not None:
            assert payload in wire

    def test_responses_tag_dispatch(self) -> None:
        responses = [
            SSEMessage(perf_ns=10, packets=[SSEField(name="data", value="x")]),
            TextResponse(perf_ns=20, text="body"),
            BinaryResponse(perf_ns=30, raw_bytes=b"\x00\xff"),
        ]
        rec = RequestRecord(responses=responses)
        rt = msgspec.msgpack.decode(msgspec.msgpack.encode(rec), type=RequestRecord)
        assert [type(r).__name__ for r in rt.responses] == [
            "SSEMessage",
            "TextResponse",
            "BinaryResponse",
        ]
        assert isinstance(rt.responses[0], SSEMessage)
        assert rt.responses[0].packets[0].value == "x"
        assert isinstance(rt.responses[2], BinaryResponse)
        assert rt.responses[2].raw_bytes == b"\x00\xff"

    def test_trace_data_tag_dispatch(self) -> None:
        trace = AioHttpTraceData(
            reference_time_ns=1,
            reference_perf_ns=2,
            request_send_start_perf_ns=3,
        )
        rec = RequestRecord(trace_data=trace)
        rt = msgspec.msgpack.decode(msgspec.msgpack.encode(rec), type=RequestRecord)
        assert isinstance(rt.trace_data, AioHttpTraceData)
        assert rt.trace_data.request_send_start_perf_ns == 3

    def test_full_record_lossless_roundtrip(self) -> None:
        mi = _base_metric_inputs(payload_bytes=b'{"k":1}')
        err = ErrorDetails(message="x", type="X")
        trace = AioHttpTraceData(reference_time_ns=1, reference_perf_ns=2)
        responses = [TextResponse(perf_ns=5, text="t")]
        rec = RequestRecord(
            metric_inputs=mi,
            request_headers={"a": "b"},
            model_name="m",
            timestamp_ns=10,
            start_perf_ns=20,
            end_perf_ns=30,
            recv_start_perf_ns=25,
            status=200,
            responses=responses,
            error=err,
            credit_drop_latency=7,
            cancellation_perf_ns=29,
            trace_data=trace,
        )
        rt = msgspec.msgpack.decode(msgspec.msgpack.encode(rec), type=RequestRecord)
        assert rt.timestamp_ns == 10
        assert rt.model_name == "m"
        assert rt.request_headers == {"a": "b"}
        assert rt.recv_start_perf_ns == 25
        assert rt.credit_drop_latency == 7
        assert rt.cancellation_perf_ns == 29
        assert isinstance(rt.error, ErrorDetails)
        assert isinstance(rt.trace_data, AioHttpTraceData)
        assert isinstance(rt.responses[0], TextResponse)
        assert rt.responses[0].text == "t"


class TestOmitDefaults:
    """``omit_defaults=True`` keeps the wire form lean.

    None-valued / empty-list defaulted fields drop out of the msgpack wire,
    mirroring Pydantic's prior ``exclude_none=True`` shape. Required-default
    factory fields (``timestamp_ns``, ``start_perf_ns``) always serialize
    because their values are not the static default.
    """

    def test_empty_record_omits_none_defaults(self) -> None:
        rec = RequestRecord()
        wire = msgspec.msgpack.encode(rec)
        decoded = msgspec.msgpack.decode(wire)
        # All None defaults dropped.
        assert "metric_inputs" not in decoded
        assert "request_headers" not in decoded
        assert "model_name" not in decoded
        assert "end_perf_ns" not in decoded
        assert "recv_start_perf_ns" not in decoded
        assert "status" not in decoded
        assert "error" not in decoded
        assert "credit_drop_latency" not in decoded
        assert "cancellation_perf_ns" not in decoded
        assert "trace_data" not in decoded
        # Empty list default also dropped.
        assert "responses" not in decoded
        # Factory-defaulted ints always serialize (live values != static default).
        assert "timestamp_ns" in decoded
        assert "start_perf_ns" in decoded

    def test_populated_fields_present(self) -> None:
        rec = RequestRecord(status=200, model_name="m")
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(rec))
        assert decoded["status"] == 200
        assert decoded["model_name"] == "m"


class TestProperties:
    """``has_error`` / ``was_cancelled`` / ``valid`` still work on the Struct."""

    def test_has_error_false_when_none(self) -> None:
        assert RequestRecord().has_error is False

    def test_has_error_true_when_set(self) -> None:
        rec = RequestRecord(error=ErrorDetails(message="x", type="X"))
        assert rec.has_error is True

    def test_was_cancelled_tracks_cancellation_perf_ns(self) -> None:
        assert RequestRecord().was_cancelled is False
        assert RequestRecord(cancellation_perf_ns=10).was_cancelled is True

    def test_valid_requires_responses_and_no_error(self) -> None:
        rec = RequestRecord(
            start_perf_ns=10,
            responses=[TextResponse(perf_ns=20, text="x")],
        )
        assert rec.valid is True

    def test_valid_false_with_error(self) -> None:
        rec = RequestRecord(
            start_perf_ns=10,
            responses=[TextResponse(perf_ns=20, text="x")],
            error=ErrorDetails(message="x", type="X"),
        )
        assert rec.valid is False


class TestMutation:
    """msgspec.Struct fields are settable; the records pipeline mutates records
    in place (e.g., to free response data after parsing)."""

    def test_set_error_after_construction(self) -> None:
        rec = RequestRecord()
        rec.error = ErrorDetails(message="x", type="X")
        assert rec.has_error is True

    def test_set_responses_to_none_for_memory_free(self) -> None:
        """``record_processor_service`` sets ``record.responses = None`` to
        free raw SSE chunks after parsing. msgspec.Struct attribute set is
        unchecked at runtime; the type annotation is only enforced at decode."""
        rec = RequestRecord(responses=[TextResponse(perf_ns=1, text="t")])
        rec.responses = None  # type: ignore[assignment]
        assert rec.responses is None

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msgspec-tagged-union round-trip coverage for ``BaseTraceDataUnion`` + ``ErrorDetails``.

These tests confirm:

- ``ErrorDetails`` encodes/decodes via msgspec.json; identity equality
  (``code``/``type``/``message``) survives the wire trip.
- Each ``BaseTraceData`` variant encodes/decodes via msgspec.json with the
  correct tag, and the tagged-union decoder dispatches to the right subtype.
- ``RequestRecord``, ``MetricRecordsData``, and ``MetricRecordsMessage``
  round-trip both ``error`` and ``trace_data`` preserving their concrete types.
"""

from __future__ import annotations

import msgspec
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages.inference_messages import (
    MetricRecordsData,
    MetricRecordsMessage,
)
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.record_models import (
    MetricRecordMetadata,
    RequestRecord,
)
from aiperf.common.models.trace_models import (
    AioHttpTraceData,
    BaseTraceData,
    BaseTraceDataUnion,
)


def _metadata() -> MetricRecordMetadata:
    return MetricRecordMetadata(
        x_request_id="req-1",
        x_correlation_id="cor-1",
        conversation_id="conv-1",
        turn_index=0,
        session_num=0,
        request_start_ns=0,
        request_end_ns=1,
        worker_id="worker-1",
        record_processor_id="rp-1",
        benchmark_phase=CreditPhase.PROFILING,
    )


class TestErrorDetailsMsgspec:
    """ErrorDetails round-trips through msgspec and preserves identity equality."""

    def test_roundtrip_preserves_fields(self) -> None:
        e = ErrorDetails(message="boom", code=500, type="RuntimeError", cause="prev")
        back = msgspec.json.decode(msgspec.json.encode(e), type=ErrorDetails)
        assert back == e
        assert back.message == "boom"
        assert back.code == 500
        assert back.cause == "prev"

    def test_eq_compares_only_code_type_message(self) -> None:
        a = ErrorDetails(message="boom", code=500, type="X", cause="alpha")
        b = ErrorDetails(message="boom", code=500, type="X", cause="beta")
        # ``cause`` differs but equality contract only looks at the identity triple.
        assert a == b
        assert hash(a) == hash(b)

    def test_from_exception_carries_chain(self) -> None:
        try:
            try:
                raise ValueError("root")
            except ValueError as inner:
                raise RuntimeError("wrapper") from inner
        except RuntimeError as e:
            details = ErrorDetails.from_exception(e)
        assert details.type == "RuntimeError"
        assert details.cause_chain == ["RuntimeError", "ValueError"]


class TestBaseTraceDataUnionDispatch:
    """The tagged-union decoder dispatches by ``trace_type`` field."""

    @pytest.mark.parametrize(
        "factory,expected_type,expected_tag",
        [
            pytest.param(
                lambda: BaseTraceData(reference_time_ns=1, reference_perf_ns=2),
                BaseTraceData,
                "base",
                id="base",
            ),
            pytest.param(
                lambda: AioHttpTraceData(reference_time_ns=1, reference_perf_ns=2),
                AioHttpTraceData,
                "aiohttp",
                id="aiohttp",
            ),
        ],
    )  # fmt: skip
    def test_union_dispatch(self, factory, expected_type, expected_tag) -> None:
        msg = factory()
        blob = msgspec.json.encode(msg)
        # Tag rides the wire.
        raw = msgspec.json.decode(blob, type=dict)
        assert raw["trace_type"] == expected_tag
        # Decoder picks the right subclass.
        back = msgspec.json.decode(blob, type=BaseTraceDataUnion)
        assert isinstance(back, expected_type)


class TestRequestRecordBridge:
    """``RequestRecord`` (Task 3a.4: now msgspec.Struct) ferries ErrorDetails + AioHttpTraceData natively."""

    def test_error_and_trace_data_roundtrip(self) -> None:
        err = ErrorDetails(message="boom", code=500, type="RuntimeError")
        trace = AioHttpTraceData(
            reference_time_ns=100,
            reference_perf_ns=10,
            request_send_start_perf_ns=20,
            tcp_connect_start_perf_ns=15,
        )
        rec = RequestRecord(error=err, trace_data=trace, status=500)
        rt = msgspec.json.decode(msgspec.json.encode(rec), type=RequestRecord)
        assert isinstance(rt.error, ErrorDetails)
        assert rt.error == err
        assert isinstance(rt.trace_data, AioHttpTraceData)
        assert rt.trace_data.request_send_start_perf_ns == 20
        assert rt.trace_data.tcp_connect_start_perf_ns == 15

    def test_none_error_and_trace_data_roundtrip(self) -> None:
        rec = RequestRecord(status=200)
        rt = msgspec.json.decode(msgspec.json.encode(rec), type=RequestRecord)
        assert rt.error is None
        assert rt.trace_data is None


class TestMetricRecordsDataBridge:
    """MetricRecordsData / MetricRecordsMessage round-trip error + trace_data fields."""

    def test_metric_records_data_roundtrip(self) -> None:
        err = ErrorDetails(message="boom", type="RuntimeError")
        trace = AioHttpTraceData(reference_time_ns=1, reference_perf_ns=2)
        data = MetricRecordsData(
            metadata=_metadata(), metrics={}, trace_data=trace, error=err
        )
        rt = msgspec.json.decode(msgspec.json.encode(data), type=MetricRecordsData)
        assert isinstance(rt.error, ErrorDetails)
        assert rt.error == err
        assert isinstance(rt.trace_data, AioHttpTraceData)

    def test_metric_records_message_roundtrip(self) -> None:
        err = ErrorDetails(message="boom", type="RuntimeError")
        trace = AioHttpTraceData(reference_time_ns=1, reference_perf_ns=2)
        msg = MetricRecordsMessage(
            service_id="svc-1",
            metadata=_metadata(),
            metrics={},
            trace_data=trace,
            error=err,
        )
        rt = msgspec.json.decode(msgspec.json.encode(msg), type=MetricRecordsMessage)
        assert isinstance(rt.error, ErrorDetails)
        assert rt.error == err
        assert isinstance(rt.trace_data, AioHttpTraceData)

    def test_to_data_preserves_subtype(self) -> None:
        trace = AioHttpTraceData(reference_time_ns=1, reference_perf_ns=2)
        msg = MetricRecordsMessage(
            service_id="svc-1",
            metadata=_metadata(),
            metrics={},
            trace_data=trace,
            error=None,
        )
        data = msg.to_data()
        assert isinstance(data.trace_data, AioHttpTraceData)

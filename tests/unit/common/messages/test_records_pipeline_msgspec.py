# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 3b: ``MetricRecordsMessage`` / ``RealtimeMetricsMessage`` /
``MetricRecordsData`` are ``msgspec.Struct``.

Pins the post-migration contract for the records-pipeline wire messages:

* Construction is kwargs-only.
* ``msgspec.msgpack`` encode/decode round-trips natively (no Pydantic bridge).
* ``message_type`` rides the wire as a top-level discriminator (via msgspec
  ``tag_field``) so the wire codec routes inbound bytes back to the right class.
* ``omit_defaults`` strips None-valued fields from the msgpack wire shape.
* The wire codec dispatches encode/decode by encoder family.
"""

from __future__ import annotations

import msgspec

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages import (
    MetricRecordsData,
    MetricRecordsMessage,
    RealtimeMetricsMessage,
)
from aiperf.common.messages.wire_codec import decode_message, encode_message
from aiperf.common.models import ErrorDetails, MetricRecordMetadata, MetricResult


def _metadata() -> MetricRecordMetadata:
    return MetricRecordMetadata(
        session_num=1,
        request_start_ns=100,
        request_end_ns=200,
        worker_id="w1",
        record_processor_id="rp1",
        benchmark_phase=CreditPhase.PROFILING,
    )


class TestMetricRecordsMessageConstruction:
    def test_kwargs_only(self) -> None:
        m = MetricRecordsMessage(metadata=_metadata(), metrics={"a": 1.0})
        assert m.metadata.session_num == 1
        assert m.metrics == {"a": 1.0}
        assert m.message_type == MessageType.METRIC_RECORDS
        assert m.message_type == "metric_records"
        assert m.error is None
        assert m.trace_data is None
        assert m.valid is True

    def test_is_msgspec_struct(self) -> None:
        m = MetricRecordsMessage(metadata=_metadata(), metrics={})
        assert isinstance(m, msgspec.Struct)

    def test_error_makes_valid_false(self) -> None:
        m = MetricRecordsMessage(
            metadata=_metadata(),
            metrics={},
            error=ErrorDetails(message="boom", code=500),
        )
        assert m.valid is False


class TestMetricRecordsMessageWire:
    def test_message_type_is_on_the_wire(self) -> None:
        m = MetricRecordsMessage(metadata=_metadata(), metrics={})
        wire = encode_message(m)
        data = msgspec.msgpack.decode(wire)
        assert data["message_type"] == "metric_records"

    def test_omit_defaults_strips_nones(self) -> None:
        m = MetricRecordsMessage(metadata=_metadata(), metrics={})
        wire = encode_message(m)
        data = msgspec.msgpack.decode(wire)
        assert "error" not in data
        assert "trace_data" not in data
        assert "request_id" not in data
        assert "service_id" not in data

    def test_round_trip(self) -> None:
        m = MetricRecordsMessage(
            service_id="svc",
            metadata=_metadata(),
            metrics={"a": 1.5, "b": 2},
        )
        wire = encode_message(m)
        decoded = decode_message(wire)
        assert isinstance(decoded, MetricRecordsMessage)
        assert decoded.service_id == "svc"
        assert decoded.metrics == {"a": 1.5, "b": 2}

    def test_to_data_preserves_merged_metrics(self) -> None:
        m = MetricRecordsMessage(
            metadata=_metadata(),
            metrics={"a": 1.0, "b": 2.0},
        )
        data = m.to_data()
        assert isinstance(data, MetricRecordsData)
        assert data.metrics == {"a": 1.0, "b": 2.0}
        assert data.valid is True

    def test_byte_for_byte_round_trip(self) -> None:
        m = MetricRecordsMessage(
            service_id="svc",
            metadata=_metadata(),
            metrics={"a": 1.5},
            error=ErrorDetails(message="x", code=1),
        )
        wire1 = encode_message(m)
        m2 = decode_message(wire1)
        assert isinstance(m2, MetricRecordsMessage)
        wire2 = encode_message(m2)
        assert wire1 == wire2


class TestRealtimeMetricsMessageWire:
    def test_kwargs_only(self) -> None:
        mr = MetricResult(tag="lat", header="Lat", unit="ms", avg=1.5)
        m = RealtimeMetricsMessage(metrics=[mr])
        assert m.message_type == MessageType.REALTIME_METRICS
        assert m.metrics[0].avg == 1.5

    def test_is_msgspec_struct(self) -> None:
        m = RealtimeMetricsMessage(metrics=[])
        assert isinstance(m, msgspec.Struct)

    def test_round_trip(self) -> None:
        mr = MetricResult(tag="lat", header="Lat", unit="ms", avg=1.5, p99=99.0)
        m = RealtimeMetricsMessage(service_id="rm", metrics=[mr])
        wire = encode_message(m)
        decoded = decode_message(wire)
        assert isinstance(decoded, RealtimeMetricsMessage)
        assert decoded.service_id == "rm"
        assert decoded.metrics[0].tag == "lat"
        assert decoded.metrics[0].avg == 1.5
        assert decoded.metrics[0].p99 == 99.0

    def test_message_type_is_on_the_wire(self) -> None:
        m = RealtimeMetricsMessage(metrics=[])
        wire = encode_message(m)
        data = msgspec.msgpack.decode(wire)
        assert data["message_type"] == "realtime_metrics"


class TestMetricRecordsDataConstruction:
    def test_kwargs_only_and_valid(self) -> None:
        d = MetricRecordsData(metadata=_metadata(), metrics={"a": 1.0})
        assert d.metrics == {"a": 1.0}
        assert d.valid is True

    def test_error_makes_valid_false(self) -> None:
        d = MetricRecordsData(
            metadata=_metadata(),
            metrics={},
            error=ErrorDetails(message="boom"),
        )
        assert d.valid is False


class TestWireCodecDispatch:
    def test_decode_routes_metric_records_to_msgspec(self) -> None:
        m = MetricRecordsMessage(metadata=_metadata(), metrics={})
        decoded = decode_message(encode_message(m))
        assert type(decoded).__name__ == "MetricRecordsMessage"

    def test_decode_routes_realtime_metrics_to_msgspec(self) -> None:
        m = RealtimeMetricsMessage(metrics=[])
        decoded = decode_message(encode_message(m))
        assert type(decoded).__name__ == "RealtimeMetricsMessage"

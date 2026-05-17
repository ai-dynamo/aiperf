# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 3b: ``RawRecordInfo`` is a ``msgspec.Struct``.

Pins the post-migration contract for the exporter-side raw record carrier:

* Construction is kwargs-only.
* ``msgspec.json`` encode/decode round-trips natively (no Pydantic bridges).
* ``omit_defaults`` strips None-valued fields from the JSONL on-disk shape.
* Embedded ``InferenceServerResponseUnion`` tag rides each response entry.
"""

from __future__ import annotations

import msgspec
import orjson

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    BinaryResponse,
    ErrorDetails,
    MetricRecordMetadata,
    RawRecordInfo,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.record_models import SSEField


def _metadata() -> MetricRecordMetadata:
    return MetricRecordMetadata(
        session_num=1,
        request_start_ns=100,
        request_end_ns=200,
        worker_id="w1",
        record_processor_id="rp1",
        benchmark_phase=CreditPhase.PROFILING,
    )


class TestRawRecordInfoConstruction:
    def test_kwargs_only(self) -> None:
        r = RawRecordInfo(metadata=_metadata(), payload={}, responses=[])
        assert r.metadata.session_num == 1
        assert r.payload == {}
        assert r.responses == []
        assert r.error is None

    def test_is_msgspec_struct(self) -> None:
        r = RawRecordInfo(metadata=_metadata(), payload={}, responses=[])
        assert isinstance(r, msgspec.Struct)


class TestRawRecordInfoWire:
    def test_omit_defaults_strips_nones(self) -> None:
        r = RawRecordInfo(metadata=_metadata(), payload={}, responses=[])
        wire = msgspec.json.encode(r)
        data = orjson.loads(wire)
        assert "error" not in data
        assert "request_headers" not in data
        assert "response_headers" not in data
        assert "status" not in data

    def test_full_round_trip(self) -> None:
        r = RawRecordInfo(
            metadata=_metadata(),
            payload={"messages": [{"role": "user"}]},
            request_headers={"Content-Type": "application/json"},
            status=200,
            response_headers={"Content-Type": "text/event-stream"},
            responses=[
                SSEMessage(perf_ns=100, packets=[SSEField(name="data", value="{}")]),
                TextResponse(perf_ns=200, text="hello"),
                BinaryResponse(perf_ns=300, raw_bytes=b"\x00"),
            ],
            error=ErrorDetails(code=200, message="ok"),
        )
        wire = msgspec.json.encode(r)
        rt = msgspec.json.decode(wire, type=RawRecordInfo)
        assert rt.metadata.session_num == 1
        assert rt.status == 200
        assert len(rt.responses) == 3
        assert isinstance(rt.responses[0], SSEMessage)
        assert isinstance(rt.responses[1], TextResponse)
        assert isinstance(rt.responses[2], BinaryResponse)
        assert rt.responses[2].raw_bytes == b"\x00"
        assert rt.error == ErrorDetails(code=200, message="ok")

    def test_byte_for_byte_round_trip(self) -> None:
        r = RawRecordInfo(
            metadata=_metadata(),
            payload={"a": 1},
            responses=[TextResponse(perf_ns=1, text="x")],
        )
        wire1 = msgspec.json.encode(r)
        rt = msgspec.json.decode(wire1, type=RawRecordInfo)
        wire2 = msgspec.json.encode(rt)
        assert wire1 == wire2

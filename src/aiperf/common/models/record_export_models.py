# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
import orjson

from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.metric_result_models import MetricValue
from aiperf.common.models.trace_models import TraceDataExport

# NOTE: MetricRecordMetadata and metric_record_metadata_from_model live in
# aiperf.common.metric_records_wire, which imports aiperf.common.models at
# module load time. A top-level import here forms a circular that fails when
# metric_records_wire is the first-entry module in a load chain. Type
# annotations below reference MetricRecordMetadata as a string thanks to
# `from __future__ import annotations`; the only runtime use is inside
# decode_metric_record_info_json, which does a local import.
if TYPE_CHECKING:
    from aiperf.common.metric_records_wire import MetricRecordMetadata


class MetricRecordInfo(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """The full info of a metric record including the metadata, metrics, and error for export."""

    metadata: MetricRecordMetadata
    """Record metadata (timestamps, credit info, phase)."""

    metrics: dict[str, MetricValue]
    """Computed metric values keyed by metric tag."""

    trace_data: TraceDataExport | None = None
    """Optional trace data captured via a trace config."""

    error: ErrorDetails | None = None
    """Error details if the underlying request failed."""

    def to_json_bytes(self) -> bytes:
        return _METRIC_RECORD_INFO_ENCODER.encode(self)


class RawRecordInfo(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """The full info of a raw record including the request record for export."""

    metadata: MetricRecordMetadata
    """Record metadata (timestamps, credit info, phase)."""

    start_perf_ns: int
    """Request start timestamp in nanoseconds (perf_counter_ns)."""

    payload: dict[str, Any]
    """The serialized request payload sent to the inference server."""

    request_headers: dict[str, str] | None = None
    """HTTP request headers, if captured."""

    status: int | None = None
    """HTTP response status code."""

    response_headers: dict[str, str] | None = None
    """HTTP response headers, if captured."""

    responses: list[Any]
    """Raw response objects from the inference server."""

    error: ErrorDetails | None = None
    """Error details if the request failed."""

    def to_json_bytes(self) -> bytes:
        return _RAW_RECORD_INFO_ENCODER.encode(self)


def _record_info_enc_hook(obj: Any) -> Any:
    # MetricValue is a dataclass, which msgspec encodes natively — no hook
    # needed. Only the Pydantic fallback below is load-bearing: TraceDataExport
    # (and its AioHttpTraceDataExport subtype) plus ErrorDetails are Pydantic
    # final-export models, and msgspec can't serialize them directly.
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True, mode="json")
    raise TypeError(f"Unsupported record artifact type: {type(obj)}")


_METRIC_RECORD_INFO_ENCODER = msgspec.json.Encoder(enc_hook=_record_info_enc_hook)
_RAW_RECORD_INFO_ENCODER = msgspec.json.Encoder(enc_hook=_record_info_enc_hook)


def decode_metric_record_info_json(data: str | bytes) -> MetricRecordInfo:
    """Decode a JSON-encoded ``MetricRecordInfo`` (as written by the JSONL exporter)."""
    from aiperf.common.metric_records_wire import metric_record_metadata_from_model

    payload = orjson.loads(data)
    trace_data = payload.get("trace_data")
    return MetricRecordInfo(
        metadata=metric_record_metadata_from_model(payload["metadata"]),
        metrics={
            key: MetricValue(**value) for key, value in payload["metrics"].items()
        },
        trace_data=TraceDataExport.model_validate(trace_data) if trace_data else None,
        error=msgspec.convert(payload["error"], ErrorDetails)
        if payload.get("error")
        else None,
    )


def decode_raw_record_info_json(data: str | bytes) -> RawRecordInfo:
    """Decode a JSON-encoded ``RawRecordInfo`` (as written by the raw-record JSONL exporter)."""
    from aiperf.common.metric_records_wire import metric_record_metadata_from_model

    payload = orjson.loads(data)
    return RawRecordInfo(
        metadata=metric_record_metadata_from_model(payload["metadata"]),
        start_perf_ns=payload["start_perf_ns"],
        payload=payload["payload"],
        request_headers=payload.get("request_headers"),
        status=payload.get("status"),
        response_headers=payload.get("response_headers"),
        responses=payload["responses"],
        error=msgspec.convert(payload["error"], ErrorDetails)
        if payload.get("error")
        else None,
    )

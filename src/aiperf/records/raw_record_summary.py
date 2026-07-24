# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact raw response metadata extraction."""

from __future__ import annotations

from typing import Any

from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.models import MetricRecordMetadata, ParsedResponseRecord
from aiperf.common.models.record_models import (
    RawRecordSummary,
    RawRecordSummaryInfo,
    RawRecordSummaryNvext,
)


def _chunk_ms(start_perf_ns: int, chunk_perf_ns: int) -> float | None:
    delta_ns = chunk_perf_ns - start_perf_ns
    return delta_ns / NANOS_PER_MILLIS if delta_ns >= 0 else None


def _extract_finish_reason(packet: dict[str, Any]) -> str | None:
    finish_reason = packet.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason

    choices = packet.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    finish_reason = first_choice.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) and finish_reason else None


def _extract_request_id(packet: dict[str, Any]) -> str | None:
    request_id = packet.get("request_id") or packet.get("id")
    return request_id if isinstance(request_id, str) else None


def _extract_nvext(packet: dict[str, Any]) -> RawRecordSummaryNvext | None:
    nvext = packet.get("nvext")
    if not isinstance(nvext, dict):
        return None

    timing = nvext.get("timing")
    worker_id = nvext.get("worker_id")
    timing = timing if isinstance(timing, dict) else None
    worker_id = str(worker_id) if worker_id is not None else None
    if timing is None and worker_id is None:
        return None
    return RawRecordSummaryNvext(timing=timing, worker_id=worker_id)


def _chunk_offsets_ms(
    start_perf_ns: int, chunk_perf_ns: list[int]
) -> tuple[float | None, float | None, float | None]:
    if not chunk_perf_ns:
        return None, None, None

    first_chunk_ms = _chunk_ms(start_perf_ns, chunk_perf_ns[0])
    last_chunk_ms = _chunk_ms(start_perf_ns, chunk_perf_ns[-1])
    if (
        first_chunk_ms is None
        or last_chunk_ms is None
        or last_chunk_ms < first_chunk_ms
    ):
        return first_chunk_ms, last_chunk_ms, None
    return first_chunk_ms, last_chunk_ms, last_chunk_ms - first_chunk_ms


def build_raw_record_summary(record: ParsedResponseRecord) -> RawRecordSummary:
    """Extract compact timing and nvext metadata from raw response packets."""
    chunk_perf_ns: list[int] = []
    request_id = None
    finish_reason = None
    nvext_timing = None
    nvext_worker_id = None

    for response in record.request.responses or []:
        text = response.get_text()
        if text not in (None, "", "[DONE]"):
            chunk_perf_ns.append(response.perf_ns)

        packet = response.get_json()
        if not isinstance(packet, dict):
            continue

        packet_request_id = _extract_request_id(packet)
        if request_id is None and packet_request_id is not None:
            request_id = packet_request_id

        packet_finish_reason = _extract_finish_reason(packet)
        if packet_finish_reason is not None:
            finish_reason = packet_finish_reason

        packet_nvext = _extract_nvext(packet)
        if packet_nvext is not None:
            if packet_nvext.timing is not None:
                nvext_timing = packet_nvext.timing
            if packet_nvext.worker_id is not None:
                nvext_worker_id = packet_nvext.worker_id

    first_chunk_ms, last_chunk_ms, stream_decode_ms = _chunk_offsets_ms(
        record.request.start_perf_ns,
        chunk_perf_ns,
    )
    nvext = None
    if nvext_timing is not None or nvext_worker_id is not None:
        nvext = RawRecordSummaryNvext(
            timing=nvext_timing,
            worker_id=nvext_worker_id,
        )

    return RawRecordSummary(
        request_id=request_id,
        status=record.request.status,
        data_chunk_count=len(chunk_perf_ns),
        finish_reason=finish_reason,
        first_chunk_ms=first_chunk_ms,
        last_chunk_ms=last_chunk_ms,
        stream_decode_ms=stream_decode_ms,
        nvext=nvext,
    )


def build_raw_record_summary_info(
    record: ParsedResponseRecord,
    metadata: MetricRecordMetadata,
) -> RawRecordSummaryInfo:
    """Build a standalone compact raw response summary row."""
    summary = build_raw_record_summary(record)
    return RawRecordSummaryInfo(
        metadata=metadata,
        **summary.model_dump(),
    )

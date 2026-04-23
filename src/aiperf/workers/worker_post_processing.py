# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers the Worker uses to project RequestRecord/ProcessHealth into wire payloads.

These functions are free of Worker state that changes across credits — they read only
configuration (`run.cfg.artifacts`) and per-call inputs. Keeping them out of `worker.py`
keeps the service file focused on credit/lifecycle orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.inference_wire import (
    InferenceResultsWireMessage,
    build_inference_results_wire_message,
    encode_inference_results_wire_message,
)
from aiperf.common.messages import WorkerHealthMessage
from aiperf.common.models import (
    ReasoningResponseData,
    RequestRecord,
    Text,
    Turn,
    WorkerTaskStats,
)
from aiperf.common.pod_lifecycle_structs import GroupWorkerHealth

if TYPE_CHECKING:
    from aiperf.common.models import ProcessHealth
    from aiperf.workers.inference_client import InferenceClient


def process_response_sync(
    inference_client: InferenceClient, record: RequestRecord
) -> Turn | None:
    """Synchronous response processing — runs in a thread pool.

    Parses responses into structured data, extracts text from all responses,
    and returns a Turn with role="assistant" if text is present, else None.
    """
    resp = inference_client.endpoint.extract_response_data(record)
    output_texts = []
    for response in resp:
        if not response.data:
            continue
        if isinstance(response.data, ReasoningResponseData):
            if response.data.content:
                output_texts.append(response.data.content)
        else:
            output_texts.append(response.data.get_text())
    resp_text = "".join(output_texts)

    return (
        Turn(role="assistant", texts=[Text(contents=[resp_text])])
        if resp_text
        else None
    )


def build_inference_wire_message(
    *,
    service_id: str,
    inference_client: InferenceClient,
    record: RequestRecord,
    include_raw_export_fields: bool,
    include_trace_data: bool,
) -> InferenceResultsWireMessage:
    """Build the msgspec worker->record-processor wire payload."""
    raw_payload = None
    if include_raw_export_fields and record.request_info is not None:
        raw_payload = inference_client.endpoint.format_payload(record.request_info)
    return build_inference_results_wire_message(
        service_id=service_id,
        record=record,
        raw_payload=raw_payload,
        include_request_headers=include_raw_export_fields,
        include_status=include_raw_export_fields,
        include_trace_data=include_trace_data,
    )


def serialize_inference_wire(
    *,
    service_id: str,
    inference_client: InferenceClient,
    record: RequestRecord,
    include_raw_export_fields: bool,
    include_trace_data: bool,
) -> bytes:
    """Serialize the msgspec worker->record-processor wire payload."""
    return encode_inference_results_wire_message(
        build_inference_wire_message(
            service_id=service_id,
            inference_client=inference_client,
            record=record,
            include_raw_export_fields=include_raw_export_fields,
            include_trace_data=include_trace_data,
        )
    )


def create_health_message(
    *,
    service_id: str,
    health: ProcessHealth,
    task_stats: WorkerTaskStats,
) -> WorkerHealthMessage:
    """Build the pub/sub worker health message."""
    return WorkerHealthMessage(
        service_id=service_id,
        health=health,
        task_stats=task_stats,
    )


def create_pod_worker_health(
    *,
    service_id: str,
    health: ProcessHealth,
    task_stats: WorkerTaskStats,
) -> GroupWorkerHealth:
    """Build the group-local msgspec health snapshot."""
    io_counters = tuple(health.io_counters) if health.io_counters is not None else None
    cpu_times = tuple(health.cpu_times) if health.cpu_times is not None else None
    num_ctx_switches = (
        tuple(health.num_ctx_switches) if health.num_ctx_switches is not None else None
    )
    return GroupWorkerHealth(
        service_id=service_id,
        pid=health.pid,
        create_time=health.create_time,
        uptime=health.uptime,
        cpu_usage=health.cpu_usage,
        memory_usage=health.memory_usage,
        pss_memory=health.pss_memory,
        io_counters=io_counters,
        cpu_times=cpu_times,
        num_ctx_switches=num_ctx_switches,
        num_threads=health.num_threads,
        task_total=task_stats.total,
        task_failed=task_stats.failed,
        task_completed=task_stats.completed,
    )

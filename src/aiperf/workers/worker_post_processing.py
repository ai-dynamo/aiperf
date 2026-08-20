# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers the Worker uses to project RequestRecord/ProcessHealth into wire payloads.

These functions are free of Worker state that changes across credits — they read only
configuration (`run.cfg.artifacts`) and per-call inputs. Keeping them out of `worker.py`
keeps the service file focused on credit/lifecycle orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.messages import WorkerHealthMessage
from aiperf.common.models import (
    RequestRecord,
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

    Delegates to the endpoint's ``build_assistant_turn`` so the captured
    assistant Turn carries the endpoint's full replay semantics: the base
    text + reasoning-only fallback (Qwen3-style / mock-server responses that
    put everything in ``reasoning`` with empty ``content``) and the chat
    endpoint's ``tool_calls`` / ``function_call`` preservation. A hand-rolled
    text-only copy here silently dropped the reasoning fallback, so FORK-mode
    DAG children inherited a parent context with no captured assistant turn.
    Returns ``None`` when the record has no replayable assistant content.
    """
    return inference_client.endpoint.build_assistant_turn(record)


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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for the service-lifecycle envelopes that now carry
msgspec-typed payload structs (ProcessHealth, WorkerTaskStats).

Exists to confirm the PydanticStructMixin shim handles the mix of mutable
and frozen structs introduced by S1/S2 of the msgspec-service-lifecycle
migration.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import LifecycleState
from aiperf.common.messages.progress_messages import BenchmarkCompleteMessage
from aiperf.common.messages.service_messages import (
    HeartbeatMessage,
    RegistrationMessage,
    StatusMessage,
)
from aiperf.common.messages.worker_messages import WorkerHealthMessage
from aiperf.common.models import (
    CPUTimes,
    CtxSwitches,
    IOCounters,
    ProcessHealth,
    WorkerTaskStats,
)
from aiperf.plugin.enums import ServiceType


def _process_health() -> ProcessHealth:
    return ProcessHealth(
        pid=1234,
        create_time=1.5,
        uptime=2.0,
        cpu_usage=35.5,
        memory_usage=1024 * 1024,
        io_counters=IOCounters(1, 2, 3, 4, 5, 6),
        cpu_times=CPUTimes(user=1.0, system=0.5, iowait=0.1),
        num_ctx_switches=CtxSwitches(100, 10),
        num_threads=4,
    )


def _task_stats() -> WorkerTaskStats:
    return WorkerTaskStats(total=100, failed=3, completed=80)


@pytest.mark.parametrize(
    "message_factory",
    [
        param(
            lambda: WorkerHealthMessage(
                service_id="w1",
                health=_process_health(),
                task_stats=_task_stats(),
            ),
            id="WorkerHealthMessage",
        ),
        param(
            lambda: StatusMessage(
                service_id="w1",
                service_type=ServiceType.WORKER,
                state=LifecycleState.RUNNING,
                request_ns=1,
                request_id="r",
            ),
            id="StatusMessage",
        ),
        param(
            lambda: HeartbeatMessage(
                service_id="w1",
                service_type=ServiceType.WORKER,
                state=LifecycleState.RUNNING,
                request_ns=1,
                request_id="r",
            ),
            id="HeartbeatMessage",
        ),
        param(
            lambda: RegistrationMessage(
                service_id="w1",
                service_type=ServiceType.WORKER,
                state=LifecycleState.RUNNING,
                request_ns=1,
                request_id="r",
            ),
            id="RegistrationMessage",
        ),
        param(
            lambda: BenchmarkCompleteMessage(service_id="sc", was_cancelled=False),
            id="BenchmarkCompleteMessage",
        ),
    ],
)
def test_lifecycle_envelope_roundtrips(message_factory) -> None:
    """Envelope with msgspec payload must round-trip through Pydantic JSON."""
    message = message_factory()

    payload = message.model_dump_json()
    decoded = type(message).model_validate_json(payload)

    assert decoded == message


def test_process_health_decodes_from_dict_payload() -> None:
    """WorkerHealthMessage decoded from a dict matches a re-queued record."""
    payload = {
        "service_id": "w1",
        "request_ns": 1,
        "message_type": "worker_health",
        "health": {
            "pid": 1234,
            "create_time": 1.5,
            "uptime": 2.0,
            "cpu_usage": 35.5,
            "memory_usage": 1048576,
            "io_counters": {
                "read_count": 1,
                "write_count": 2,
                "read_bytes": 3,
                "write_bytes": 4,
                "read_chars": 5,
                "write_chars": 6,
            },
            "cpu_times": {"user": 1.0, "system": 0.5, "iowait": 0.1},
            "num_ctx_switches": {"voluntary": 100, "involuntary": 10},
            "num_threads": 4,
        },
        "task_stats": {"total": 100, "failed": 3, "completed": 80},
    }

    msg = WorkerHealthMessage.model_validate(payload)

    assert msg.health.io_counters.read_bytes == 3
    assert msg.health.cpu_times.iowait == 0.1
    assert msg.health.num_ctx_switches.voluntary == 100
    assert msg.task_stats.total == 100

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_worker_group_stats aggregation helper."""

from __future__ import annotations

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.workers.worker_group_state import WorkerStatusInfo
from aiperf.workers.worker_pod_helpers import build_worker_group_stats


def _info(
    *,
    worker_id: str,
    status: WorkerStatus,
    cpu: float,
    mem: int,
    total: int,
    failed: int = 0,
    startup: WorkerStartupState | None = WorkerStartupState.READY,
) -> WorkerStatusInfo:
    info = WorkerStatusInfo(worker_id=worker_id)
    info.status = status
    info.startup_state = startup
    info.health = ProcessHealth(
        pid=1, create_time=0.0, uptime=1.0, cpu_usage=cpu, memory_usage=mem
    )
    info.task_stats = WorkerTaskStats(total=total, failed=failed)
    return info


def test_aggregates_task_stats_as_sum() -> None:
    workers = {
        "w-0": _info(
            worker_id="w-0",
            status=WorkerStatus.HEALTHY,
            cpu=10.0,
            mem=100,
            total=5,
            failed=0,
        ),
        "w-1": _info(
            worker_id="w-1",
            status=WorkerStatus.HEALTHY,
            cpu=20.0,
            mem=200,
            total=7,
            failed=2,
        ),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0",
        declared_workers=2,
        worker_infos=workers,
    )
    assert msg.task_stats.total == 12
    assert msg.task_stats.failed == 2


def test_aggregates_cpu_as_average_memory_as_sum() -> None:
    workers = {
        "w-0": _info(
            worker_id="w-0", status=WorkerStatus.HEALTHY, cpu=10.0, mem=100, total=0
        ),
        "w-1": _info(
            worker_id="w-1", status=WorkerStatus.HEALTHY, cpu=30.0, mem=200, total=0
        ),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=2, worker_infos=workers
    )
    assert msg.health is not None
    assert msg.health.cpu_usage == 20.0
    assert msg.health.memory_usage == 300


def test_group_status_uses_worst_child() -> None:
    workers = {
        "w-0": _info(
            worker_id="w-0", status=WorkerStatus.HEALTHY, cpu=0.0, mem=0, total=0
        ),
        "w-1": _info(
            worker_id="w-1", status=WorkerStatus.ERROR, cpu=0.0, mem=0, total=0
        ),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=2, worker_infos=workers
    )
    assert msg.status == WorkerStatus.ERROR


def test_per_worker_maps_populated() -> None:
    workers = {
        "w-0": _info(
            worker_id="w-0", status=WorkerStatus.HEALTHY, cpu=5.0, mem=10, total=3
        ),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=1, worker_infos=workers
    )
    assert msg.worker_statuses == {"w-0": WorkerStatus.HEALTHY}
    assert msg.worker_task_stats["w-0"].total == 3
    assert msg.worker_health["w-0"].cpu_usage == 5.0
    assert msg.worker_startup_states["w-0"] == WorkerStartupState.READY


def test_ready_workers_counts_ready_startup_state_only() -> None:
    workers = {
        "w-0": _info(
            worker_id="w-0",
            status=WorkerStatus.HEALTHY,
            cpu=0.0,
            mem=0,
            total=0,
            startup=WorkerStartupState.READY,
        ),
        "w-1": _info(
            worker_id="w-1",
            status=WorkerStatus.IDLE,
            cpu=0.0,
            mem=0,
            total=0,
            startup=WorkerStartupState.WAITING_FOR_DATASET,
        ),
    }
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=2, worker_infos=workers
    )
    assert msg.ready_workers == 1


def test_empty_group_yields_idle_zeroed_message() -> None:
    msg = build_worker_group_stats(
        service_id="wgm-0", declared_workers=0, worker_infos={}
    )
    assert msg.status == WorkerStatus.IDLE
    assert msg.task_stats.total == 0
    assert msg.worker_statuses == {}
    assert msg.health is None

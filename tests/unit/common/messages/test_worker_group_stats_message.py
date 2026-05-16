# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkerGroupStatsMessage round-trip and defaults."""

from __future__ import annotations

import msgspec

from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.messages import WorkerGroupStatsMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats


def _health() -> ProcessHealth:
    return ProcessHealth(
        pid=1, create_time=0.0, uptime=1.0, cpu_usage=12.5, memory_usage=2048
    )


def test_message_tag_matches_enum() -> None:
    msg = WorkerGroupStatsMessage(
        service_id="wgm-0",
        group_id="wgm-0",
        status=WorkerStatus.HEALTHY,
        task_stats=WorkerTaskStats(),
    )
    assert msg.message_type == MessageType.WORKER_GROUP_STATS


def test_round_trip_preserves_per_worker_maps() -> None:
    msg = WorkerGroupStatsMessage(
        service_id="wgm-0",
        group_id="wgm-0",
        status=WorkerStatus.HIGH_LOAD,
        startup_state=WorkerStartupState.READY,
        declared_workers=2,
        ready_workers=2,
        health=_health(),
        task_stats=WorkerTaskStats(total=10, failed=1),
        worker_statuses={"w-0": WorkerStatus.HEALTHY, "w-1": WorkerStatus.HIGH_LOAD},
        worker_startup_states={"w-0": WorkerStartupState.READY},
        worker_task_stats={"w-0": WorkerTaskStats(total=5)},
        worker_health={"w-0": _health()},
    )
    # Branch carries WorkerTaskStats and ProcessHealth as Pydantic models;
    # msgspec.json.encode needs an enc_hook to dump them.
    def _enc(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        raise TypeError(f"Cannot encode object of type {type(obj).__name__}")

    def _dec(typ, obj):
        if hasattr(typ, "model_validate") and isinstance(obj, dict):
            return typ.model_validate(obj)
        raise NotImplementedError

    encoded = msgspec.json.encode(msg, enc_hook=_enc)
    decoded = msgspec.json.decode(
        encoded, type=WorkerGroupStatsMessage, dec_hook=_dec
    )
    assert decoded.group_id == "wgm-0"
    assert decoded.worker_statuses == msg.worker_statuses
    assert decoded.worker_task_stats["w-0"].total == 5
    assert decoded.health.cpu_usage == 12.5


def test_defaults_are_empty_maps() -> None:
    msg = WorkerGroupStatsMessage(
        service_id="wgm-0",
        group_id="wgm-0",
        status=WorkerStatus.IDLE,
        task_stats=WorkerTaskStats(),
    )
    assert msg.worker_statuses == {}
    assert msg.worker_task_stats == {}
    assert msg.worker_health == {}

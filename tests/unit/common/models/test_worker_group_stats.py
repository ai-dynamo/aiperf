# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the aggregate worker-group and process-health models."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.models import (
    NumericAggregate,
    ProcessHealthAggregates,
    WorkerGroupStats,
    WorkerStats,
)


class TestWorkerGroupStats:
    """Defaults, mutation, and Pydantic interop for WorkerGroupStats."""

    def test_worker_group_stats_defaults(self) -> None:
        stats = WorkerGroupStats(group_id="group-0")
        assert stats.status == WorkerStatus.IDLE
        assert stats.startup_state is None
        assert stats.declared_workers == 0
        assert stats.ready_workers == 0
        assert stats.health is None
        assert stats.workers == {}
        assert stats.last_update_ns is None

    def test_worker_group_stats_tracks_startup_state(self) -> None:
        stats = WorkerGroupStats(
            group_id="group-0", startup_state=WorkerStartupState.READY
        )
        assert stats.startup_state == WorkerStartupState.READY

    def test_worker_group_stats_is_mutable(self) -> None:
        stats = WorkerGroupStats(group_id="group-0")
        stats.workers = {"w-0": WorkerStats(worker_id="w-0")}
        stats.ready_workers = 1
        assert stats.ready_workers == 1
        assert set(stats.workers) == {"w-0"}

    def test_worker_group_stats_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            WorkerGroupStats("group-0")  # type: ignore[misc]

    def test_worker_group_stats_forbids_extra_via_pydantic(self) -> None:
        adapter = TypeAdapter(WorkerGroupStats)
        assert adapter.validate_python({"group_id": "g"}).group_id == "g"
        with pytest.raises(ValidationError):
            adapter.validate_python({"group_id": "g", "bogus": 1})

    def test_worker_group_stats_round_trips_children_through_pydantic(self) -> None:
        adapter = TypeAdapter(WorkerGroupStats)
        loaded = adapter.validate_python(
            {
                "group_id": "g",
                "workers": {
                    "w-0": {
                        "worker_id": "w-0",
                        "startup_state": WorkerStartupState.READY,
                    }
                },
            }
        )
        assert loaded.workers["w-0"].startup_state == WorkerStartupState.READY


class TestProcessHealthAggregates:
    """Defaults and in-place update behaviour for the health aggregates."""

    def test_numeric_aggregate_updates_in_place(self) -> None:
        agg = NumericAggregate()
        assert agg.avg is None
        agg.update(2.0)
        agg.update(None)
        agg.update(4)
        assert (agg.min, agg.max, agg.count) == (2.0, 4.0, 2)
        assert agg.avg == 3.0

    def test_process_health_aggregates_defaults_are_independent(self) -> None:
        first = ProcessHealthAggregates()
        second = ProcessHealthAggregates()
        first.cpu_usage.update(1.0)
        assert second.cpu_usage.count == 0
        assert first.memory_usage.count == 0

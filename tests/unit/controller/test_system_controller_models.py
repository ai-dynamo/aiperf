# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for system_controller_models.

Focuses on:
- AggregateWorkerStatus default-zero construction and serialization round-trip
- K8sServiceTopology dataclass shape and immutability
- build_aggregate_worker_status summing semantics across pod snapshots,
  including the ready/degraded "useful pod" predicates
"""

from __future__ import annotations

import dataclasses

import orjson
import pytest
from pytest import param

from aiperf.common.messages import WorkerPodStateMessage
from aiperf.controller.system_controller_models import (
    AggregateWorkerStatus,
    K8sServiceTopology,
    build_aggregate_worker_status,
)

# ============================================================
# Helpers
# ============================================================


def make_pod_state(
    pod_index: str = "0",
    *,
    declared_workers: int = 0,
    declared_record_processors: int = 0,
    pod_state: str = "ready",
    admission_state: str = "admitted",
    router_connected_workers: int = 0,
    dispatchable_workers: int = 0,
    ready_workers: int = 0,
    ready_record_processors: int = 0,
    degraded_workers: int = 0,
    degraded_record_processors: int = 0,
) -> WorkerPodStateMessage:
    """Build a WorkerPodStateMessage with sensible defaults for testing."""
    return WorkerPodStateMessage(
        service_id=f"wgm_{pod_index}",
        pod_index=pod_index,
        declared_workers=declared_workers,
        declared_record_processors=declared_record_processors,
        pod_state=pod_state,
        admission_state=admission_state,
        router_connected_workers=router_connected_workers,
        dispatchable_workers=dispatchable_workers,
        ready_workers=ready_workers,
        ready_record_processors=ready_record_processors,
        degraded_workers=degraded_workers,
        degraded_record_processors=degraded_record_processors,
    )


# ============================================================
# AggregateWorkerStatus
# ============================================================


class TestAggregateWorkerStatusDefaults:
    """All numeric fields default to zero so an unfilled snapshot is meaningful."""

    def test_default_construction_zeros_all_counters(self) -> None:
        status = AggregateWorkerStatus()
        assert status.ready == 0
        assert status.total == 0
        assert status.dispatchable == 0
        assert status.router_connected == 0
        assert status.ready_record_processors == 0
        assert status.declared_record_processors == 0
        assert status.ready_pods == 0
        assert status.total_pods == 0
        assert status.degraded_pods == 0

    def test_explicit_construction_preserves_values(self) -> None:
        status = AggregateWorkerStatus(
            ready=4,
            total=8,
            dispatchable=4,
            router_connected=4,
            ready_record_processors=2,
            declared_record_processors=2,
            ready_pods=2,
            total_pods=2,
            degraded_pods=1,
        )
        assert status.ready == 4
        assert status.total == 8
        assert status.degraded_pods == 1


class TestAggregateWorkerStatusSerialization:
    """Pydantic round-trip via dump and JSON should be lossless."""

    def test_model_dump_round_trip(self) -> None:
        original = AggregateWorkerStatus(
            ready=3,
            total=5,
            dispatchable=3,
            router_connected=3,
            ready_record_processors=1,
            declared_record_processors=2,
            ready_pods=1,
            total_pods=2,
            degraded_pods=0,
        )
        dumped = original.model_dump()
        restored = AggregateWorkerStatus(**dumped)
        assert restored == original

    def test_model_dump_json_round_trip(self) -> None:
        original = AggregateWorkerStatus(ready=7, total=10, total_pods=2)
        json_bytes = original.model_dump_json().encode()
        restored = AggregateWorkerStatus(**orjson.loads(json_bytes))
        assert restored == original


# ============================================================
# K8sServiceTopology dataclass
# ============================================================


class TestK8sServiceTopology:
    """Frozen-slotted dataclass: holds five derived ints, immutable after construction."""

    def test_construction_stores_all_fields(self) -> None:
        topo = K8sServiceTopology(
            num_worker_pods=4,
            workers_per_pod=8,
            record_processors_per_pod=2,
            total_workers=32,
            total_record_processors=8,
        )
        assert topo.num_worker_pods == 4
        assert topo.workers_per_pod == 8
        assert topo.record_processors_per_pod == 2
        assert topo.total_workers == 32
        assert topo.total_record_processors == 8

    def test_is_frozen_assignment_raises(self) -> None:
        topo = K8sServiceTopology(
            num_worker_pods=1,
            workers_per_pod=1,
            record_processors_per_pod=1,
            total_workers=1,
            total_record_processors=1,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            topo.num_worker_pods = 99  # type: ignore[misc]

    def test_uses_slots(self) -> None:
        topo = K8sServiceTopology(
            num_worker_pods=1,
            workers_per_pod=1,
            record_processors_per_pod=1,
            total_workers=1,
            total_record_processors=1,
        )
        # slots=True means no __dict__
        assert not hasattr(topo, "__dict__")

    def test_equality_by_value(self) -> None:
        a = K8sServiceTopology(
            num_worker_pods=2,
            workers_per_pod=4,
            record_processors_per_pod=1,
            total_workers=8,
            total_record_processors=2,
        )
        b = K8sServiceTopology(
            num_worker_pods=2,
            workers_per_pod=4,
            record_processors_per_pod=1,
            total_workers=8,
            total_record_processors=2,
        )
        assert a == b


# ============================================================
# build_aggregate_worker_status
# ============================================================


class TestBuildAggregateWorkerStatusEmpty:
    """No pods yields the all-zero baseline."""

    def test_empty_dict_returns_zero_status(self) -> None:
        result = build_aggregate_worker_status({})
        assert result == AggregateWorkerStatus()
        assert result.total_pods == 0


class TestBuildAggregateWorkerStatusSums:
    """Counters sum across all pod messages regardless of pod readiness."""

    def test_sums_all_counters_across_pods(self) -> None:
        states = {
            "0": make_pod_state(
                "0",
                declared_workers=4,
                declared_record_processors=2,
                router_connected_workers=4,
                dispatchable_workers=3,
                ready_workers=3,
                ready_record_processors=2,
            ),
            "1": make_pod_state(
                "1",
                declared_workers=4,
                declared_record_processors=2,
                router_connected_workers=2,
                dispatchable_workers=2,
                ready_workers=2,
                ready_record_processors=1,
            ),
        }
        result = build_aggregate_worker_status(states)
        assert result.total == 8
        assert result.declared_record_processors == 4
        assert result.router_connected == 6
        assert result.dispatchable == 5
        assert result.ready == 5
        assert result.ready_record_processors == 3
        assert result.total_pods == 2

    def test_single_pod_passthrough(self) -> None:
        state = make_pod_state(
            "0",
            declared_workers=2,
            declared_record_processors=1,
            ready_workers=2,
            dispatchable_workers=2,
            ready_record_processors=1,
        )
        result = build_aggregate_worker_status({"0": state})
        assert result.total == 2
        assert result.ready == 2
        assert result.dispatchable == 2
        assert result.ready_record_processors == 1
        assert result.total_pods == 1


class TestReadyAndDegradedPodPredicates:
    """ready_pods and degraded_pods apply the dual-readiness predicate."""

    @pytest.mark.parametrize(
        "dispatchable,ready_rp,expected_ready",
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            param(1, 1, 1, id="both-met-counts"),
            param(10, 5, 1, id="ready-pod-counted-once-not-by-capacity"),
        ],
    )  # fmt: skip
    def test_ready_pods_requires_dispatchable_and_ready_rp(
        self, dispatchable: int, ready_rp: int, expected_ready: int
    ) -> None:
        state = make_pod_state(
            "0",
            dispatchable_workers=dispatchable,
            ready_record_processors=ready_rp,
        )
        result = build_aggregate_worker_status({"0": state})
        assert result.ready_pods == expected_ready

    def test_degraded_pod_requires_ready_predicate_plus_any_degraded(self) -> None:
        # ready predicate met AND has degraded workers → degraded pod
        state = make_pod_state(
            "0",
            dispatchable_workers=2,
            ready_record_processors=1,
            degraded_workers=1,
        )
        result = build_aggregate_worker_status({"0": state})
        assert result.ready_pods == 1
        assert result.degraded_pods == 1

    def test_degraded_pod_requires_ready_predicate_plus_degraded_rp(self) -> None:
        state = make_pod_state(
            "0",
            dispatchable_workers=2,
            ready_record_processors=1,
            degraded_record_processors=1,
        )
        result = build_aggregate_worker_status({"0": state})
        assert result.degraded_pods == 1

    def test_degraded_workers_alone_not_counted_when_pod_not_ready(self) -> None:
        # not dispatch-ready → not counted as degraded even with degraded counters
        state = make_pod_state(
            "0",
            dispatchable_workers=0,
            ready_record_processors=0,
            degraded_workers=5,
            degraded_record_processors=2,
        )
        result = build_aggregate_worker_status({"0": state})
        assert result.ready_pods == 0
        assert result.degraded_pods == 0

    def test_ready_pod_without_degradation_not_degraded(self) -> None:
        state = make_pod_state(
            "0",
            dispatchable_workers=2,
            ready_record_processors=1,
            degraded_workers=0,
            degraded_record_processors=0,
        )
        result = build_aggregate_worker_status({"0": state})
        assert result.ready_pods == 1
        assert result.degraded_pods == 0

    def test_mixed_ready_and_degraded_pods(self) -> None:
        states = {
            "ready_clean": make_pod_state(
                "0",
                dispatchable_workers=2,
                ready_record_processors=1,
            ),
            "ready_degraded": make_pod_state(
                "1",
                dispatchable_workers=2,
                ready_record_processors=1,
                degraded_workers=1,
            ),
            "not_ready": make_pod_state(
                "2",
                dispatchable_workers=0,
                ready_record_processors=0,
            ),
        }
        result = build_aggregate_worker_status(states)
        assert result.total_pods == 3
        assert result.ready_pods == 2
        assert result.degraded_pods == 1

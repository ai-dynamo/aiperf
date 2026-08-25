# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pure helpers in aiperf.kubernetes.jobset_resources.

Covers weighted-total splitting, CPU/memory formatting, worker-pod resource
allocation (including the pinned record-processor CPU override), and health
port allocation for worker pods.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset_resources import (
    _compute_cpu_shares,
    allocate_worker_health_ports,
    format_mcpu,
    format_mib,
    split_weighted_total,
    split_worker_pod_resources,
)
from aiperf.kubernetes.utils import parse_cpu, parse_memory_mib


class TestSplitWeightedTotal:
    """Largest-remainder allocation must preserve the sum exactly."""

    def test_empty_weights_returns_empty_list(self) -> None:
        assert split_weighted_total(100, []) == []

    def test_zero_total_returns_zero_per_bucket(self) -> None:
        assert split_weighted_total(0, [1, 2, 3]) == [0, 0, 0]

    def test_negative_total_treated_as_zero(self) -> None:
        """Negative totals degrade gracefully to an all-zero split."""
        assert split_weighted_total(-5, [1, 2]) == [0, 0]

    def test_equal_weights_divide_evenly(self) -> None:
        assert split_weighted_total(9, [1, 1, 1]) == [3, 3, 3]

    def test_sum_is_preserved_under_remainders(self) -> None:
        """Largest-remainder: total must equal sum of shares even when weights don't divide evenly."""
        shares = split_weighted_total(10, [1, 1, 1])
        assert sum(shares) == 10

    def test_heavier_bucket_gets_more(self) -> None:
        shares = split_weighted_total(100, [1, 4])
        assert shares[1] > shares[0]
        assert sum(shares) == 100

    @pytest.mark.parametrize(
        "total,weights",
        [
            param(1000, [1, 1, 1, 1, 1, 1, 1], id="seven-equal"),
            param(2500, [100, 131, 131, 389, 389], id="manager-worker-rp-shape"),
            param(37, [2, 3, 5], id="small-numbers"),
        ],
    )  # fmt: skip
    def test_arbitrary_shapes_preserve_total(
        self, total: int, weights: list[int]
    ) -> None:
        """For any valid shape, shares must sum exactly to the total."""
        assert sum(split_weighted_total(total, weights)) == total


class TestFormatMcpu:
    """Millicore formatting matches Kubernetes quantity conventions."""

    @pytest.mark.parametrize(
        "mcpu,expected",
        [
            param(1000, "1", id="exact-core"),
            param(2000, "2", id="exact-two-cores"),
            param(500, "500m", id="half-core"),
            param(1500, "1500m", id="one-and-a-half-core"),
            param(0, "0", id="zero-formats-as-cores"),
        ],
    )  # fmt: skip
    def test_format_mcpu(self, mcpu: int, expected: str) -> None:
        """Whole cores drop the ``m`` suffix; fractional values keep it."""
        assert format_mcpu(mcpu) == expected


class TestFormatMib:
    """MiB formatting always uses the ``Mi`` suffix."""

    @pytest.mark.parametrize(
        "mib,expected",
        [
            param(256, "256Mi", id="small"),
            param(1024, "1024Mi", id="one-gib-as-mib"),
            param(0, "0Mi", id="zero"),
        ],
    )  # fmt: skip
    def test_format_mib(self, mib: int, expected: str) -> None:
        assert format_mib(mib) == expected


class TestComputeCpuShares:
    """CPU share distribution across manager/workers/record-processors."""

    def test_uses_weighted_split_when_no_fixed_rp_request(self) -> None:
        """Without a pinned RP CPU request, shares come from the weighted split."""
        shares = _compute_cpu_shares(
            total_mcpu=2500,
            worker_count=2,
            record_processor_count=1,
            record_processor_cpu_request=None,
        )
        # Manager + 2 workers + 1 RP = 4 containers
        assert len(shares) == 4
        assert sum(shares) == 2500

    def test_zero_record_processors_falls_back_to_weighted(self) -> None:
        """With rp_count=0 the fixed-request branch is skipped even if a request is set."""
        shares = _compute_cpu_shares(
            total_mcpu=1000,
            worker_count=2,
            record_processor_count=0,
            record_processor_cpu_request="500m",
        )
        assert len(shares) == 3
        assert sum(shares) == 1000

    def test_fixed_rp_request_is_pinned(self) -> None:
        """Pinned record-processor CPU request must be allocated exactly to each RP."""
        shares = _compute_cpu_shares(
            total_mcpu=4000,
            worker_count=2,
            record_processor_count=2,
            record_processor_cpu_request="500m",
        )
        # Last 2 entries are record processors
        assert shares[-1] == 500
        assert shares[-2] == 500
        # Manager + workers split the remainder 4000 - 1000 = 3000
        assert shares[0] + shares[1] + shares[2] == 3000

    def test_fixed_rp_request_larger_than_budget_clamps_remainder_to_zero(self) -> None:
        """If RP requests exceed total, manager/worker shares go to zero without negatives."""
        shares = _compute_cpu_shares(
            total_mcpu=500,
            worker_count=1,
            record_processor_count=2,
            record_processor_cpu_request="1000m",
        )
        assert shares[-1] == 1000
        assert shares[-2] == 1000
        assert shares[0] == 0  # manager
        assert shares[1] == 0  # worker


class TestSplitWorkerPodResources:
    """End-to-end split preserves the worker-pod CPU and memory budget."""

    def test_returns_none_list_when_budget_is_none(self) -> None:
        """A None budget (resource_mode='none') yields one None per container."""
        result = split_worker_pod_resources(
            None,
            worker_count=2,
            record_processor_count=1,
            record_processor_cpu_request=None,
            burstable=False,
        )
        assert result == [None, None, None, None]

    def test_preserves_total_budget(self) -> None:
        """Sum of per-container requests must equal the worker-pod budget."""
        budget = {"requests": {"cpu": "4000m", "memory": "4096Mi"}}
        result = split_worker_pod_resources(
            budget,
            worker_count=2,
            record_processor_count=2,
            record_processor_cpu_request=None,
            burstable=False,
        )
        total_cpu = sum(parse_cpu(r["requests"]["cpu"]) for r in result if r)
        total_mem = sum(parse_memory_mib(r["requests"]["memory"]) for r in result if r)
        # allow fractional drift from millicore rounding; total is in cores
        assert abs(total_cpu - 4.0) < 0.001
        assert total_mem == 4096

    def test_guaranteed_mode_emits_matching_limits(self) -> None:
        """Non-burstable mode must mirror each request value to a limit."""
        budget = {"requests": {"cpu": "2000m", "memory": "2048Mi"}}
        result = split_worker_pod_resources(
            budget,
            worker_count=1,
            record_processor_count=1,
            record_processor_cpu_request=None,
            burstable=False,
        )
        for entry in result:
            assert entry is not None
            assert entry["limits"] == entry["requests"]

    def test_burstable_mode_omits_limits(self) -> None:
        """Burstable QoS emits requests only; limits must be absent."""
        budget = {"requests": {"cpu": "2000m", "memory": "2048Mi"}}
        result = split_worker_pod_resources(
            budget,
            worker_count=1,
            record_processor_count=1,
            record_processor_cpu_request=None,
            burstable=True,
        )
        for entry in result:
            assert entry is not None
            assert "limits" not in entry

    def test_container_count_equals_sum_of_components(self) -> None:
        """Always manager + workers + record_processors entries."""
        budget = {"requests": {"cpu": "8000m", "memory": "16Gi"}}
        result = split_worker_pod_resources(
            budget,
            worker_count=3,
            record_processor_count=2,
            record_processor_cpu_request=None,
            burstable=False,
        )
        assert len(result) == 1 + 3 + 2


class TestAllocateWorkerHealthPorts:
    """Each container in a worker pod gets a unique health port."""

    def test_returns_manager_workers_and_record_processor_ports(self) -> None:
        manager, workers, rps = allocate_worker_health_ports(2, 1)
        assert manager == K8sEnvironment.PORTS.WORKER_HEALTH
        assert len(workers) == 2
        assert len(rps) == 1

    def test_all_ports_are_unique(self) -> None:
        """A pod shares a network namespace so each container needs its own port."""
        manager, workers, rps = allocate_worker_health_ports(4, 3)
        allocated = [manager, *workers, *rps]
        assert len(set(allocated)) == len(allocated)

    def test_worker_ports_start_after_manager(self) -> None:
        """Workers occupy the contiguous range above the manager port."""
        manager, workers, _rps = allocate_worker_health_ports(3, 0)
        assert workers == [manager + 1, manager + 2, manager + 3]

    def test_record_processor_ports_avoid_worker_range(self) -> None:
        """RP ports start at max(RECORD_PROCESSOR_HEALTH, end-of-worker-range)."""
        # Many workers force RP ports past their configured default.
        manager, workers, rps = allocate_worker_health_ports(100, 2)
        assert min(rps) > max(workers)

    def test_raises_when_range_exceeds_65535(self) -> None:
        """A port request larger than the IP port space raises ValueError."""
        with pytest.raises(ValueError, match="65535"):
            allocate_worker_health_ports(70000, 0)

    def test_zero_counts_returns_empty_worker_and_rp_lists(self) -> None:
        manager, workers, rps = allocate_worker_health_ports(0, 0)
        assert manager == K8sEnvironment.PORTS.WORKER_HEALTH
        assert workers == []
        assert rps == []

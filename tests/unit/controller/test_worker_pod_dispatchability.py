# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-side worker-pod dispatchability accounting and start gate.

A worker pod that is alive but has no dataset must not be counted as able to
serve credits. The router keeps such a worker out of the routing pool, so the
controller has to notice and wait rather than starting profiling against a
fraction of the requested load.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.messages import GetPodStatesCommand, WorkerPodStateMessage
from aiperf.common.mixins.pod_state_tracker_mixin import PodStateTracker
from aiperf.controller.system_controller import SystemController
from aiperf.controller.system_controller_models import build_aggregate_worker_status


def _pod(
    pod_index: str,
    *,
    dispatchable: int = 2,
    ready_workers: int = 2,
    declared_workers: int = 2,
    router_connected: int = 2,
    ready_rps: int = 1,
    declared_rps: int = 1,
    degraded_workers: int = 0,
    degraded_rps: int = 0,
) -> WorkerPodStateMessage:
    return WorkerPodStateMessage(
        service_id=f"wgm-{pod_index}",
        pod_index=pod_index,
        benchmark_generation=None,
        dataset_generation=None,
        pod_state="ready" if ready_workers >= 1 and ready_rps >= 1 else "starting",
        admission_state="dispatchable" if dispatchable >= 1 else "admitting",
        declared_workers=declared_workers,
        declared_record_processors=declared_rps,
        router_connected_workers=router_connected,
        dispatchable_workers=dispatchable,
        ready_workers=ready_workers,
        ready_record_processors=ready_rps,
        degraded_workers=degraded_workers,
        degraded_record_processors=degraded_rps,
    )


class TestBuildAggregateWorkerStatus:
    def test_empty_pod_states_is_all_zero(self) -> None:
        status = build_aggregate_worker_status({})
        assert status.total_pods == 0
        assert status.dispatchable == 0
        assert status.ready_pods == 0

    def test_sums_across_pods(self) -> None:
        status = build_aggregate_worker_status({"0": _pod("0"), "1": _pod("1")})
        assert status.total_pods == 2
        assert status.ready_pods == 2
        assert status.dispatchable == 4
        assert status.total == 4
        assert status.router_connected == 4
        assert status.ready_record_processors == 2

    @pytest.mark.parametrize(
        "dispatchable, ready_rps, expected_ready_pods",
        [
            param(2, 1, 1, id="dispatchable-with-record-processor"),
            param(0, 1, 0, id="no-dataset-not-ready"),
            param(2, 0, 0, id="no-record-processor-not-ready"),
        ],
    )  # fmt: skip
    def test_pod_needs_both_a_worker_and_a_record_processor(
        self, dispatchable: int, ready_rps: int, expected_ready_pods: int
    ) -> None:
        """A worker with no record processor emits records nobody aggregates."""
        pods = {"0": _pod("0", dispatchable=dispatchable, ready_rps=ready_rps)}
        assert build_aggregate_worker_status(pods).ready_pods == expected_ready_pods

    def test_connected_but_undispatchable_pod_is_visible_but_not_ready(self) -> None:
        """The exact shape of the pod that hung a live run.

        Its containers are up and its workers are connected to the credit
        router, but no dataset ever arrived, so nothing may be routed to it.
        """
        pods = {"3": _pod("3", dispatchable=0, ready_workers=0, router_connected=2)}
        status = build_aggregate_worker_status(pods)
        assert status.total_pods == 1
        assert status.router_connected == 2
        assert status.dispatchable == 0
        assert status.ready_pods == 0


@pytest.mark.asyncio
async def test_get_pod_states_command_reads_controller_tracker() -> None:
    """The RPC handler snapshots the controller cache, not an API-side mirror."""
    controller = object.__new__(SystemController)
    controller._pod_state_tracker = PodStateTracker()
    controller._pod_state_tracker.update_pod_state(_pod("7", ready_workers=1))

    data = await controller._handle_get_pod_states_command(
        GetPodStatesCommand(service_id="api-service")
    )

    pod_states = data["pod_states"]
    assert isinstance(pod_states, dict)
    assert pod_states["7"]["ready_workers"] == 1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone pod-state cache behind PodStateTrackerMixin."""

import pytest

from aiperf.common.enums import WorkerStartupState
from aiperf.common.messages import (
    WorkerPodStateMessage,
    WorkerStartupStateMessage,
    WorkerStatusSummaryMessage,
)
from aiperf.common.mixins import PodStateTrackerMixin
from aiperf.common.mixins.pod_state_tracker_mixin import PodStateTracker


def _pod_state(pod_index: str, ready: int) -> WorkerPodStateMessage:
    return WorkerPodStateMessage(
        service_id=f"wgm-{pod_index}",
        pod_index=pod_index,
        declared_workers=2,
        declared_record_processors=1,
        pod_state="ready" if ready else "starting",
        admission_state="dispatchable" if ready else "admitting",
        ready_workers=ready,
    )


def test_update_pod_state_replaces_the_entry_for_that_pod_index() -> None:
    tracker = PodStateTracker()
    tracker.update_pod_state(_pod_state("0", ready=1))
    tracker.update_pod_state(_pod_state("0", ready=2))
    tracker.update_pod_state(_pod_state("1", ready=0))
    assert set(tracker.pod_states) == {"0", "1"}
    assert tracker.pod_states["0"].ready_workers == 2


def test_update_worker_startup_state_records_the_latest_state() -> None:
    tracker = PodStateTracker()
    tracker.update_worker_startup_state(
        WorkerStartupStateMessage(
            service_id="worker-0",
            startup_state=WorkerStartupState.READY,
        )
    )
    assert tracker.worker_startup_states == {"worker-0": str(WorkerStartupState.READY)}


def test_summary_folds_the_wgm_aggregate_into_the_per_worker_cache() -> None:
    tracker = PodStateTracker()
    tracker.update_worker_startup_states_from_summary(
        WorkerStatusSummaryMessage(
            service_id="wgm-0",
            worker_statuses={},
            worker_startup_states={
                "worker-0": WorkerStartupState.READY,
                "worker-1": WorkerStartupState.WAITING_FOR_DATASET,
            },
        )
    )
    assert tracker.worker_startup_states == {
        "worker-0": str(WorkerStartupState.READY),
        "worker-1": str(WorkerStartupState.WAITING_FOR_DATASET),
    }


@pytest.mark.asyncio
async def test_mixin_handlers_populate_the_tracker(mock_zmq, benchmark_run) -> None:
    class _Tracked(PodStateTrackerMixin):
        pass

    tracked = _Tracked(run=benchmark_run)
    await tracked._on_worker_pod_state(_pod_state("0", ready=1))
    await tracked._on_worker_startup_state(
        WorkerStartupStateMessage(
            service_id="worker-0", startup_state=WorkerStartupState.READY
        )
    )
    await tracked._on_worker_status_summary(
        WorkerStatusSummaryMessage(
            service_id="wgm-0",
            worker_statuses={},
            worker_startup_states={"worker-1": WorkerStartupState.READY},
        )
    )
    assert set(tracked._pod_state_tracker.pod_states) == {"0"}
    assert set(tracked._pod_state_tracker.worker_startup_states) == {
        "worker-0",
        "worker-1",
    }

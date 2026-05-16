# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PodStateTracker and PodStateTrackerMixin.

These exercise the bus-fed cache that the API service relies on in K8s mode
(where it lives in a separate container from the SystemController). The
``test_pod_state_tracker_mixin_subscribes_via_bus`` case exists to lock in
the fact that the mixin actually subscribes to the right message types —
without that, the K8s ``status.workers.ready=0`` regression resurfaces.
"""

from __future__ import annotations

import pytest

from aiperf.common.enums import MessageType, WorkerStartupState
from aiperf.common.messages import (
    WorkerPodStateMessage,
    WorkerStartupStateMessage,
    WorkerStatusSummaryMessage,
)
from aiperf.common.mixins.pod_state_tracker_mixin import (
    PodStateTracker,
    PodStateTrackerMixin,
)


def _pod(pod_index: str, *, declared: int, ready: int) -> WorkerPodStateMessage:
    return WorkerPodStateMessage(
        service_id=f"wpm-{pod_index}",
        pod_index=pod_index,
        benchmark_generation="gen-1",
        dataset_generation="data-1",
        declared_workers=declared,
        declared_record_processors=1,
        router_connected_workers=ready,
        dispatchable_workers=ready,
        ready_workers=ready,
        ready_record_processors=1,
        degraded_workers=max(0, declared - ready),
        degraded_record_processors=0,
        pod_state="ready" if ready >= 1 else "starting",
        admission_state="dispatchable" if ready >= 1 else "admitting",
    )


@pytest.fixture
def tracker() -> PodStateTracker:
    return PodStateTracker()


class TestPodStateTrackerPodStates:
    """Test PodStateTracker.update_pod_state / pod_states."""

    def test_empty_initially(self, tracker: PodStateTracker) -> None:
        assert tracker.pod_states == {}

    def test_records_message_keyed_by_pod_index(self, tracker: PodStateTracker) -> None:
        tracker.update_pod_state(_pod("0", declared=4, ready=4))
        assert set(tracker.pod_states.keys()) == {"0"}
        assert tracker.pod_states["0"].ready_workers == 4

    def test_subsequent_message_overwrites_pod_entry(
        self, tracker: PodStateTracker
    ) -> None:
        tracker.update_pod_state(_pod("0", declared=4, ready=1))
        tracker.update_pod_state(_pod("0", declared=4, ready=4))
        assert len(tracker.pod_states) == 1
        assert tracker.pod_states["0"].ready_workers == 4
        assert tracker.pod_states["0"].degraded_workers == 0

    def test_distinct_pods_kept_separate(self, tracker: PodStateTracker) -> None:
        tracker.update_pod_state(_pod("0", declared=4, ready=4))
        tracker.update_pod_state(_pod("1", declared=4, ready=2))
        assert set(tracker.pod_states.keys()) == {"0", "1"}
        assert tracker.pod_states["0"].ready_workers == 4
        assert tracker.pod_states["1"].ready_workers == 2

    def test_empty_pod_index_collides_silently(self, tracker: PodStateTracker) -> None:
        """Documenting the failure mode flagged in the diagnosis: pods with
        an empty AIPERF_POD_INDEX would all collide on the empty-string key
        and overwrite each other. The tracker honors the message; the env-var
        wiring is what guarantees uniqueness."""
        tracker.update_pod_state(_pod("", declared=4, ready=4))
        tracker.update_pod_state(_pod("", declared=4, ready=2))
        assert len(tracker.pod_states) == 1
        assert tracker.pod_states[""].ready_workers == 2


class TestPodStateTrackerWorkerStartupStates:
    """Test PodStateTracker.update_worker_startup_state / worker_startup_states."""

    def test_empty_initially(self, tracker: PodStateTracker) -> None:
        assert tracker.worker_startup_states == {}

    def test_records_state_string_per_service_id(
        self, tracker: PodStateTracker
    ) -> None:
        tracker.update_worker_startup_state(
            WorkerStartupStateMessage(
                service_id="w-0", startup_state=WorkerStartupState.READY
            )
        )
        assert tracker.worker_startup_states == {"w-0": "ready"}

    def test_overwrites_previous_state_for_same_worker(
        self, tracker: PodStateTracker
    ) -> None:
        tracker.update_worker_startup_state(
            WorkerStartupStateMessage(
                service_id="w-0",
                startup_state=WorkerStartupState.WAITING_FOR_DATASET,
            )
        )
        tracker.update_worker_startup_state(
            WorkerStartupStateMessage(
                service_id="w-0", startup_state=WorkerStartupState.READY
            )
        )
        assert tracker.worker_startup_states == {"w-0": "ready"}

    def test_tracks_multiple_workers(self, tracker: PodStateTracker) -> None:
        for service_id, state in [
            ("w-0", WorkerStartupState.READY),
            ("w-1", WorkerStartupState.READY),
            ("w-2", WorkerStartupState.WAITING_FOR_DATASET),
        ]:
            tracker.update_worker_startup_state(
                WorkerStartupStateMessage(service_id=service_id, startup_state=state)
            )
        assert tracker.worker_startup_states == {
            "w-0": "ready",
            "w-1": "ready",
            "w-2": "waiting_for_dataset",
        }


class TestPodStateTrackerMixinSubscriptions:
    """Lock in that the mixin subscribes to the right message types.

    A regression here (renamed enum, missing decorator) would silently put
    the API service back in the all-zeros K8s state.
    """

    def test_mixin_handler_handles_pod_state(self) -> None:
        handler = PodStateTrackerMixin._on_worker_pod_state
        params = getattr(handler, "__aiperf_hook_params__", ())
        assert MessageType.WORKER_POD_STATE in params

    def test_mixin_handler_handles_startup_state(self) -> None:
        handler = PodStateTrackerMixin._on_worker_startup_state
        params = getattr(handler, "__aiperf_hook_params__", ())
        assert MessageType.WORKER_STARTUP_STATE in params

    def test_mixin_handler_handles_status_summary(self) -> None:
        """Required for the K8s wire path — workers send their startup state
        to the WGM over DEALER, and the WGM republishes the per-worker map
        as ``WorkerStatusSummaryMessage.worker_startup_states``. Without
        this subscription, ``worker_startup_states`` is always empty in K8s
        (verified live on the DGX cluster on 2026-04-25)."""
        handler = PodStateTrackerMixin._on_worker_status_summary
        params = getattr(handler, "__aiperf_hook_params__", ())
        assert MessageType.WORKER_STATUS_SUMMARY in params


class TestPodStateTrackerStatusSummaryFold:
    """Test that WGM-aggregated summaries fold into the per-worker map."""

    def test_summary_populates_worker_startup_states(
        self, tracker: PodStateTracker
    ) -> None:
        msg = WorkerStatusSummaryMessage(
            service_id="wgm-0",
            worker_statuses={},
            worker_startup_states={
                "w-0": WorkerStartupState.READY,
                "w-1": WorkerStartupState.WAITING_FOR_DATASET,
            },
        )
        tracker.update_worker_startup_states_from_summary(msg)
        assert tracker.worker_startup_states == {
            "w-0": "ready",
            "w-1": "waiting_for_dataset",
        }

    def test_summary_overlays_subsequent_state_transitions(
        self, tracker: PodStateTracker
    ) -> None:
        first = WorkerStatusSummaryMessage(
            service_id="wgm-0",
            worker_statuses={},
            worker_startup_states={"w-0": WorkerStartupState.WAITING_FOR_DATASET},
        )
        second = WorkerStatusSummaryMessage(
            service_id="wgm-0",
            worker_statuses={},
            worker_startup_states={"w-0": WorkerStartupState.READY},
        )
        tracker.update_worker_startup_states_from_summary(first)
        tracker.update_worker_startup_states_from_summary(second)
        assert tracker.worker_startup_states == {"w-0": "ready"}

    def test_empty_summary_is_a_no_op(self, tracker: PodStateTracker) -> None:
        tracker.update_worker_startup_state(
            WorkerStartupStateMessage(
                service_id="w-0", startup_state=WorkerStartupState.READY
            )
        )
        empty = WorkerStatusSummaryMessage(
            service_id="wgm-0", worker_statuses={}, worker_startup_states={}
        )
        tracker.update_worker_startup_states_from_summary(empty)
        assert tracker.worker_startup_states == {"w-0": "ready"}

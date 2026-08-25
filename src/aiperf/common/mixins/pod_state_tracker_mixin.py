# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracks WorkerPodStateMessage and WorkerStartupStateMessage on the message
bus, so any service (notably the FastAPI sidecar that doesn't share a process
with the SystemController) can answer questions about K8s pod readiness
without an in-process handle to the controller.

The SystemController also tracks these messages — both observers are
independent eventually-consistent caches of the same pub/sub topic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import MessageType
from aiperf.common.hooks import on_message
from aiperf.common.messages import (
    WorkerPodStateMessage,
    WorkerStartupStateMessage,
    WorkerStatusSummaryMessage,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class PodStateTracker:
    """Standalone aggregate-pod-state cache.

    Mirrors the per-pod ``WorkerPodStateMessage`` snapshots (keyed by
    ``pod_index``) and the per-worker ``WorkerStartupState`` strings
    (keyed by worker service_id) that the SystemController also tracks
    for K8s startup gating.
    """

    def __init__(self) -> None:
        self._pod_states: dict[str, WorkerPodStateMessage] = {}
        self._worker_startup_states: dict[str, str] = {}

    def update_pod_state(self, message: WorkerPodStateMessage) -> None:
        """Replace the entry for ``message.pod_index`` with the new snapshot."""
        self._pod_states[message.pod_index] = message

    def update_worker_startup_state(self, message: WorkerStartupStateMessage) -> None:
        """Record a worker's most recently reported startup state."""
        self._worker_startup_states[message.service_id] = str(message.startup_state)

    def update_worker_startup_states_from_summary(
        self, message: WorkerStatusSummaryMessage
    ) -> None:
        """Fold a WGM-published per-pod summary into the per-worker cache.

        Workers publish ``WorkerStartupStateMessage`` on the global pub/sub
        (``Worker._publish_startup_state``), NOT over a per-pod DEALER. Because
        that topic is cluster-wide, every such message carries ``pod_index`` and
        each WorkerGroupManager filters on it; assuming a per-pod transport is
        what previously let one WGM adopt every worker in the cluster.

        The WGM republishes its filtered aggregate as
        ``WorkerStatusSummaryMessage.worker_startup_states`` — that republished
        summary is the wire path this cache listens on, since it is already
        scoped to a single pod.
        """
        for service_id, state in message.worker_startup_states.items():
            self._worker_startup_states[service_id] = str(state)

    @property
    def pod_states(self) -> dict[str, WorkerPodStateMessage]:
        """All tracked pods keyed by ``pod_index``."""
        return self._pod_states

    @property
    def worker_startup_states(self) -> dict[str, str]:
        """All tracked worker startup states keyed by service_id."""
        return self._worker_startup_states


class PodStateTrackerMixin(MessageBusClientMixin):
    """Subscribes to the K8s worker-pod state topics and keeps a local cache.

    Use on FastAPI router components that need to answer questions like
    "how many worker pods are ready?" without holding a handle to the
    SystemController. The cache lives on ``self._pod_state_tracker``.
    """

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(run=run, **kwargs)
        self._pod_state_tracker = PodStateTracker()

    @on_message(MessageType.WORKER_POD_STATE)
    async def _on_worker_pod_state(self, message: WorkerPodStateMessage) -> None:
        """Cache the most recent per-pod aggregate from each WorkerGroupManager."""
        self._pod_state_tracker.update_pod_state(message)

    @on_message(MessageType.WORKER_STARTUP_STATE)
    async def _on_worker_startup_state(
        self, message: WorkerStartupStateMessage
    ) -> None:
        """Cache the most recent startup-state transition from each worker.

        Fires only in non-group-managed modes (component-integration tests).
        In K8s the per-worker message goes to the WGM over DEALER instead;
        :meth:`_on_worker_status_summary` handles that path.
        """
        self._pod_state_tracker.update_worker_startup_state(message)

    @on_message(MessageType.WORKER_STATUS_SUMMARY)
    async def _on_worker_status_summary(
        self, message: WorkerStatusSummaryMessage
    ) -> None:
        """Fold the WGM-aggregated per-worker startup-state map into the cache.

        This is the K8s-mode wire path for ``worker_startup_states``.
        """
        self._pod_state_tracker.update_worker_startup_states_from_summary(message)

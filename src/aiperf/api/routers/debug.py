# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug router exposing per-pod / per-worker state used to diagnose the CR
``status.workers`` reporting chain.

In Kubernetes mode the API service runs as its own container in the
controller pod, so it cannot read the SystemController's in-memory caches
directly. The endpoints query that authoritative state over the command bus,
then fall back to the independent bus-fed mirror maintained by
:class:`PodStateTrackerMixin` when the controller is unavailable.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter
from pydantic import Field
from starlette.requests import HTTPConnection

from aiperf.api.pod_state_rpc import query_controller_pod_states
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.mixins import PodStateTrackerMixin
from aiperf.common.models import AIPerfBaseModel

DebugDep = Annotated["DebugRouter", component_dependency("debug")]

debug_router = APIRouter()


class PodStatesResponse(AIPerfBaseModel):
    """Snapshot of the per-pod ``WorkerPodStateMessage`` cache.

    ``pods`` is keyed by ``WorkerPodStateMessage.pod_index`` (the
    ``AIPERF_POD_INDEX`` env var on the worker pod) and contains the full
    last-known message payload from each WorkerGroupManager.
    """

    pod_count: int = Field(ge=0, description="Number of pod entries currently tracked.")
    pods: dict[str, dict[str, Any]] = Field(
        description="Per-pod last-known WorkerPodStateMessage, keyed by pod_index."
    )
    snapshot_time_ns: int = Field(
        ge=0, description="time.time_ns() when this snapshot was taken."
    )
    source: str = Field(
        description=(
            "Where the snapshot came from: 'controller' for the authoritative "
            "query or 'cache' for the bus-fed availability fallback."
        )
    )


class WorkerStartupStatesResponse(AIPerfBaseModel):
    """Snapshot of the per-worker startup-state cache.

    Each entry is a worker's most recently reported ``WorkerStartupState``
    (e.g. ``WAITING_FOR_DATASET``, ``ROUTER_PROBING``, ``READY``). If this
    map is empty during a benchmark, no worker has reported its startup
    state on the message bus.
    """

    worker_count: int = Field(
        ge=0, description="Number of distinct workers seen so far."
    )
    workers: dict[str, str] = Field(
        description="Per-worker startup state, keyed by worker service_id."
    )
    ready_count: int = Field(
        ge=0, description="Number of workers in WorkerStartupState.READY."
    )
    snapshot_time_ns: int = Field(
        ge=0, description="time.time_ns() when this snapshot was taken."
    )
    source: str = Field(
        description=(
            "Where the snapshot came from: 'controller' for the authoritative "
            "query or 'cache' for the bus-fed availability fallback."
        )
    )


class DebugRouter(PodStateTrackerMixin, BaseRouter):
    """Owns ``/api/debug/*`` diagnostic endpoints.

    The controller command is authoritative; :class:`PodStateTrackerMixin`
    supplies a best-effort fallback when that query is unavailable.
    """

    def get_router(self) -> APIRouter:
        return debug_router


@debug_router.get(
    "/api/debug/pod-states",
    response_model=PodStatesResponse,
    tags=["Debug"],
)
async def get_pod_states(
    conn: HTTPConnection, component: DebugDep
) -> PodStatesResponse:
    """Return the controller's per-pod ``WorkerPodStateMessage`` cache.

    A controller snapshot handles late API subscribers and dropped pub/sub
    updates. If that query fails, the endpoint serves its bus-fed mirror.
    """
    snapshot = await query_controller_pod_states(conn)
    if snapshot is not None:
        pods = {
            pod_index: message.model_dump(mode="json")
            for pod_index, message in snapshot.pod_states.items()
        }
        return PodStatesResponse(
            pod_count=len(pods),
            pods=pods,
            snapshot_time_ns=time.time_ns(),
            source="controller",
        )
    pod_states = component._pod_state_tracker.pod_states
    pods = {
        pod_index: message.model_dump(mode="json")
        for pod_index, message in pod_states.items()
    }
    return PodStatesResponse(
        pod_count=len(pods),
        pods=pods,
        snapshot_time_ns=time.time_ns(),
        source="cache",
    )


@debug_router.get(
    "/api/debug/worker-startup-states",
    response_model=WorkerStartupStatesResponse,
    tags=["Debug"],
)
async def get_worker_startup_states(
    conn: HTTPConnection, component: DebugDep
) -> WorkerStartupStatesResponse:
    """Return the controller's per-worker startup-state cache.

    Uses the same authoritative-query and cache-fallback policy as
    :func:`get_pod_states`.
    """
    snapshot = await query_controller_pod_states(conn)
    if snapshot is not None:
        states = snapshot.worker_startup_states
        ready_count = sum(1 for state in states.values() if state == "ready")
        return WorkerStartupStatesResponse(
            worker_count=len(states),
            workers=dict(states),
            ready_count=ready_count,
            snapshot_time_ns=time.time_ns(),
            source="controller",
        )
    states = component._pod_state_tracker.worker_startup_states
    ready_count = sum(1 for state in states.values() if state == "ready")
    return WorkerStartupStatesResponse(
        worker_count=len(states),
        workers=dict(states),
        ready_count=ready_count,
        snapshot_time_ns=time.time_ns(),
        source="cache",
    )

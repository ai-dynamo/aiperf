# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug router exposing per-pod / per-worker state used to diagnose the CR
``status.workers`` reporting chain.

The API service runs as its own container (``api`` sidecar in the controller
pod), so it cannot read the SystemController's in-memory caches directly.
Instead, this router subscribes to ``WORKER_POD_STATE`` and
``WORKER_STARTUP_STATE`` on the message bus via :class:`PodStateTrackerMixin`
and serves an eventually-consistent mirror — the same topology the controller
also tracks for K8s startup gating.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter
from pydantic import Field
from starlette.requests import HTTPConnection

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

    pod_count: int = Field(description="Number of pod entries currently tracked.")
    pods: dict[str, dict[str, Any]] = Field(
        description="Per-pod last-known WorkerPodStateMessage, keyed by pod_index."
    )
    snapshot_time_ns: int = Field(
        description="time.time_ns() when this snapshot was taken."
    )


class WorkerStartupStatesResponse(AIPerfBaseModel):
    """Snapshot of the per-worker startup-state cache.

    Each entry is a worker's most recently reported ``WorkerStartupState``
    (e.g. ``WAITING_FOR_DATASET``, ``ROUTER_PROBING``, ``READY``). If this
    map is empty during a benchmark, no worker has reported its startup
    state on the message bus.
    """

    worker_count: int = Field(description="Number of distinct workers seen so far.")
    workers: dict[str, str] = Field(
        description="Per-worker startup state, keyed by worker service_id."
    )
    ready_count: int = Field(
        description="Number of workers in WorkerStartupState.READY."
    )
    snapshot_time_ns: int = Field(
        description="time.time_ns() when this snapshot was taken."
    )


class DebugRouter(PodStateTrackerMixin, BaseRouter):
    """Owns ``/api/debug/*`` diagnostic endpoints fed by the message bus."""

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
    """Return the per-pod ``WorkerPodStateMessage`` cache.

    If ``pod_count == 0`` during a benchmark, no WorkerGroupManager has
    successfully published ``WORKER_POD_STATE`` on the bus reachable from
    this API service — this is the most common cause of
    ``status.workers.ready=0`` on the CR.
    """
    pod_states = component._pod_state_tracker.pod_states
    pods = {pod_index: msg.model_dump() for pod_index, msg in pod_states.items()}
    return PodStatesResponse(
        pod_count=len(pods),
        pods=pods,
        snapshot_time_ns=time.time_ns(),
    )


@debug_router.get(
    "/api/debug/worker-startup-states",
    response_model=WorkerStartupStatesResponse,
    tags=["Debug"],
)
async def get_worker_startup_states(
    conn: HTTPConnection, component: DebugDep
) -> WorkerStartupStatesResponse:
    """Return the per-worker startup-state cache.

    A non-empty map with no ``READY`` entries means workers connected but
    never finished startup (e.g. stuck in ``WAITING_FOR_DATASET``); an empty
    map means no worker reported at all.
    """
    states = component._pod_state_tracker.worker_startup_states
    ready_count = sum(1 for s in states.values() if s == "ready")
    return WorkerStartupStatesResponse(
        worker_count=len(states),
        workers=dict(states),
        ready_count=ready_count,
        snapshot_time_ns=time.time_ns(),
    )

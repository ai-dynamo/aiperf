# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug router exposing controller-internal state used to diagnose CR
``status.workers`` reporting issues.

Each endpoint returns a JSON snapshot of one piece of controller state so an
operator-side debugger can ``curl`` the controller pod and tell *why* the CR
shows ``ready=0``: did ``WorkerPodStateMessage`` ever arrive? Did any worker
report ``startup_state == READY``? What does the resulting aggregate look
like before it gets serialized into the CR patch?

These endpoints are intended for live diagnosis only; they do not stream and
do not retain history.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from pydantic import Field
from starlette.requests import HTTPConnection

from aiperf.api.routers.base_router import BaseRouter
from aiperf.common.models import AIPerfBaseModel

debug_router = APIRouter()


class PodStatesResponse(AIPerfBaseModel):
    """Snapshot of ``SystemController._pod_states``.

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
    """Snapshot of ``SystemController._worker_startup_states``.

    Each entry is a worker's most recently reported ``WorkerStartupState``
    (e.g. ``WAITING_FOR_DATASET``, ``ROUTER_PROBING``, ``READY``). If this
    map is empty during a benchmark, no worker has reported its startup
    state to the controller.
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


def _get_controller(conn: HTTPConnection) -> Any | None:
    """Return the SystemController bound to this app, or None if absent."""
    controller = getattr(conn.app.state, "controller", None)
    if controller is None:
        service = getattr(conn.app.state, "service", None)
        controller = getattr(service, "controller", None)
    return controller


class DebugRouter(BaseRouter):
    """Owns controller-internal diagnostic endpoints under ``/api/debug``."""

    def get_router(self) -> APIRouter:
        return debug_router


@debug_router.get(
    "/api/debug/pod-states",
    response_model=PodStatesResponse,
    tags=["Debug"],
)
async def get_pod_states(conn: HTTPConnection) -> PodStatesResponse:
    """Return the controller's per-pod ``WorkerPodStateMessage`` cache.

    If ``pod_count == 0`` during a benchmark, no WorkerGroupManager has
    successfully published ``WORKER_POD_STATE`` to the controller — this is
    the most common cause of ``status.workers.ready=0`` on the CR.
    """
    controller = _get_controller(conn)
    if controller is None:
        return PodStatesResponse(
            pod_count=0,
            pods={},
            snapshot_time_ns=time.time_ns(),
        )
    pod_states: dict[str, Any] = getattr(controller, "_pod_states", {}) or {}
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
    conn: HTTPConnection,
) -> WorkerStartupStatesResponse:
    """Return the controller's per-worker startup-state cache.

    A non-empty map with no ``READY`` entries means workers connected but
    never finished startup (e.g. stuck in ``WAITING_FOR_DATASET``); an empty
    map means no worker reported at all.
    """
    controller = _get_controller(conn)
    if controller is None:
        return WorkerStartupStatesResponse(
            worker_count=0,
            workers={},
            ready_count=0,
            snapshot_time_ns=time.time_ns(),
        )
    states: dict[str, str] = getattr(controller, "_worker_startup_states", {}) or {}
    ready_count = sum(1 for s in states.values() if s == "ready")
    return WorkerStartupStatesResponse(
        worker_count=len(states),
        workers=dict(states),
        ready_count=ready_count,
        snapshot_time_ns=time.time_ns(),
    )

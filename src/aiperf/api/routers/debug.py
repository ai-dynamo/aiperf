# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug router exposing per-pod / per-worker state used to diagnose the CR
``status.workers`` reporting chain.

In Kubernetes mode the API service runs as its own container in the
controller pod, so it cannot read the SystemController's in-memory caches
directly. The endpoints below first try a ``GET_POD_STATES`` RPC to the
controller over the existing DEALER↔ROUTER control channel — that returns
the controller's authoritative view, which is what the operator ultimately
writes to the CR. If the RPC is unavailable (controller starting, shutting
down, etc.), they fall back to the bus-fed mirror maintained by
:class:`PodStateTrackerMixin`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any

import orjson
from fastapi import APIRouter
from pydantic import Field
from starlette.requests import HTTPConnection

from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.control_structs import CommandOk
from aiperf.common.enums import CommandType
from aiperf.common.mixins import PodStateTrackerMixin
from aiperf.common.models import AIPerfBaseModel

DebugDep = Annotated["DebugRouter", component_dependency("debug")]

debug_router = APIRouter()

# Timeout for the GET_POD_STATES RPC. Kept short so a slow / unresponsive
# controller falls back to the bus-fed cache rather than hanging the endpoint.
_GET_POD_STATES_TIMEOUT = 2.0


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
    source: str = Field(
        description=(
            "Where the snapshot came from: 'controller' (authoritative RPC), "
            "'cache' (bus-fed mirror fallback)."
        )
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
    source: str = Field(
        description=(
            "Where the snapshot came from: 'controller' (authoritative RPC), "
            "'cache' (bus-fed mirror fallback)."
        )
    )


class DebugRouter(PodStateTrackerMixin, BaseRouter):
    """Owns ``/api/debug/*`` diagnostic endpoints.

    Primary path is the ``GET_POD_STATES`` RPC to the SystemController;
    :class:`PodStateTrackerMixin` provides the bus-fed fallback cache.
    """

    def get_router(self) -> APIRouter:
        return debug_router


async def _query_controller_snapshot(conn: HTTPConnection) -> dict[str, Any] | None:
    """Issue a GET_POD_STATES RPC to the SystemController.

    Returns the decoded snapshot dict (``{pod_states, worker_startup_states}``)
    on success, or ``None`` if the controller is unreachable / unavailable —
    callers fall back to the local cache in that case.
    """
    service = getattr(conn.app.state, "service", None)
    if service is None or not hasattr(service, "send_command_to_controller"):
        return None
    try:
        response = await service.send_command_to_controller(
            CommandType.GET_POD_STATES,
            timeout=_GET_POD_STATES_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - any RPC failure → cache fallback
        return None
    if not isinstance(response, CommandOk):
        return None
    return orjson.loads(response.payload)


@debug_router.get(
    "/api/debug/pod-states",
    response_model=PodStatesResponse,
    tags=["Debug"],
)
async def get_pod_states(
    conn: HTTPConnection, component: DebugDep
) -> PodStatesResponse:
    """Return the controller's per-pod ``WorkerPodStateMessage`` cache.

    Asks the SystemController authoritatively over the control channel.
    Falls back to this router's own bus-fed mirror if the RPC fails — in
    which case ``pod_count == 0`` during a benchmark indicates the bus
    isn't delivering ``WORKER_POD_STATE`` messages either, which is the
    common cause of ``status.workers.ready=0`` on the CR.
    """
    snapshot = await _query_controller_snapshot(conn)
    if snapshot is not None:
        pods = snapshot.get("pod_states", {}) or {}
        return PodStatesResponse(
            pod_count=len(pods),
            pods=pods,
            snapshot_time_ns=time.time_ns(),
            source="controller",
        )
    pod_states = component._pod_state_tracker.pod_states
    pods = {pod_index: msg.model_dump() for pod_index, msg in pod_states.items()}
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

    Same RPC-then-fallback pattern as :func:`get_pod_states`. A non-empty
    map with ``ready_count == 0`` means workers connected but never
    finished startup (typically stuck in ``WAITING_FOR_DATASET``).
    """
    snapshot = await _query_controller_snapshot(conn)
    if snapshot is not None:
        states = snapshot.get("worker_startup_states", {}) or {}
        ready_count = sum(1 for s in states.values() if s == "ready")
        return WorkerStartupStatesResponse(
            worker_count=len(states),
            workers=dict(states),
            ready_count=ready_count,
            snapshot_time_ns=time.time_ns(),
            source="controller",
        )
    states = component._pod_state_tracker.worker_startup_states
    ready_count = sum(1 for s in states.values() if s == "ready")
    return WorkerStartupStatesResponse(
        worker_count=len(states),
        workers=dict(states),
        ready_count=ready_count,
        snapshot_time_ns=time.time_ns(),
        source="cache",
    )

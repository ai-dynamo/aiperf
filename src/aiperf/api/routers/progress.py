# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress router component -- owns benchmark progress state and /api/progress endpoint.

When running in Kubernetes mode, periodically patches JobSet annotations with
current progress so external tools can observe status without connecting to
the controller pod's API.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter
from starlette.requests import HTTPConnection

from aiperf.api.models.responses import ProgressResponse
from aiperf.api.pod_state_rpc import query_controller_pod_states
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_message
from aiperf.common.messages import ResultsExportedMessage, WorkerPodStateMessage
from aiperf.common.mixins import PodStateTrackerMixin
from aiperf.common.mixins.progress_tracker_mixin import (
    CombinedPhaseStats,
    ProgressTrackerMixin,
)
from aiperf.controller.system_controller import AggregateWorkerStatus
from aiperf.controller.system_controller_models import build_aggregate_worker_status

ProgressDep = Annotated["ProgressRouter", component_dependency("progress")]

progress_router = APIRouter()

# Interval between JobSet annotation patches (seconds)
_JOBSET_PATCH_INTERVAL = 10.0


def _build_progress_annotations(
    phases: dict[CreditPhase, CombinedPhaseStats],
) -> dict[str, str]:
    """Build annotation values from current progress state.

    Returns a dict of annotation key -> value for patching onto the JobSet.
    """
    from aiperf.kubernetes.constants import ProgressAnnotations

    if not phases:
        return {
            ProgressAnnotations.STATUS: "initializing",
        }

    # Use profiling phase if present, otherwise warmup
    if "profiling" in phases:
        active = phases["profiling"]
        phase_name = "profiling"
    elif "warmup" in phases:
        active = phases["warmup"]
        phase_name = "warmup"
    else:
        active = next(iter(phases.values()))
        phase_name = str(active.phase)

    completed = active.requests_completed
    total = active.total_expected_requests
    pct = active.requests_progress_percent

    # Determine status
    if pct is not None and pct >= 100.0:
        status = "completing"
    elif completed > 0:
        status = "running"
    else:
        status = "starting"

    annotations: dict[str, str] = {
        ProgressAnnotations.PHASE: phase_name,
        ProgressAnnotations.STATUS: status,
    }

    if pct is not None:
        annotations[ProgressAnnotations.PERCENT] = f"{pct:.1f}"

    if total is not None and total > 0:
        annotations[ProgressAnnotations.REQUESTS] = f"{completed}/{total}"

    return annotations


class ProgressRouter(PodStateTrackerMixin, ProgressTrackerMixin, BaseRouter):
    """Owns benchmark progress state and exposes /api/progress.

    Subscribes to ``WORKER_POD_STATE`` directly so the K8s API sidecar
    (which runs in a different container from the SystemController) can
    answer ``progress.workers`` without an in-process controller handle.

    In Kubernetes mode, a background task periodically patches the JobSet
    annotations with current progress so that ``kubectl get jobset`` or
    external controllers can inspect benchmark status.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._k8s_job_id: str | None = os.environ.get("AIPERF_JOB_ID")
        self._k8s_namespace: str | None = os.environ.get("AIPERF_NAMESPACE")
        self._k8s_patching_enabled = bool(self._k8s_job_id and self._k8s_namespace)
        self._last_patched_annotations: dict[str, str] = {}
        # Flips True only after the SystemController publishes
        # ResultsExportedMessage — i.e. after ExporterManager.export_data()
        # AND (in K8s mode) write_ready_marker() have completed. The operator
        # gates JobProgress.is_complete on this so sub-second benchmarks don't
        # let the kopf-timer monitor claim completion mid-export.
        self._results_exported: bool = False

    def get_router(self) -> APIRouter:
        return progress_router

    @on_message(MessageType.RESULTS_EXPORTED)
    async def _on_results_exported(self, _message: ResultsExportedMessage) -> None:
        """Record that the controller has finished writing artifacts to disk."""
        self._results_exported = True

    @background_task(interval=_JOBSET_PATCH_INTERVAL, immediate=False)
    async def _patch_jobset_progress(self) -> None:
        """Periodically patch JobSet annotations with current progress."""
        if not self._k8s_patching_enabled:
            return

        annotations = _build_progress_annotations(self._progress_tracker._phases)

        # Skip patch if annotations haven't changed
        if annotations == self._last_patched_annotations:
            return

        try:
            await _patch_jobset_annotations(
                job_id=self._k8s_job_id,  # type: ignore[arg-type]
                namespace=self._k8s_namespace,  # type: ignore[arg-type]
                annotations=annotations,
            )
            self._last_patched_annotations = annotations
        except Exception:  # noqa: BLE001 - periodic JobSet annotation patch is best-effort; k8s API flakes must not crash the background task
            self.debug("Failed to patch JobSet progress annotations")


async def _patch_jobset_annotations(
    job_id: str,
    namespace: str,
    annotations: dict[str, str],
) -> None:
    """Patch annotations on the JobSet for the given job."""
    from kubernetes_asyncio import client

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION

    jobset_name = f"aiperf-{job_id}"

    async with k8s_client() as api:
        await client.CustomObjectsApi(api).patch_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=namespace,
            name=jobset_name,
            body={"metadata": {"annotations": annotations}},
            _content_type="application/merge-patch+json",
        )


async def _get_controller_workers(conn: HTTPConnection) -> AggregateWorkerStatus:
    """Resolve aggregate worker status, preferring authoritative paths.

    Resolution order:

    1. Single-process / local-dev — SystemController is on
       ``app.state.controller`` and we ask it directly.
    2. K8s sidecar mode — send a ``GET_POD_STATES`` command to the
       SystemController over the existing DEALER↔ROUTER control channel
       and rebuild the aggregate from its authoritative cache.
    3. Fallback — rebuild from the ProgressRouter's own bus-fed
       ``_pod_state_tracker.pod_states`` mirror. Only reached when the
       controller RPC is unavailable (controller starting / shutting down).
    """
    controller = getattr(conn.app.state, "controller", None)
    if controller is None:
        service = getattr(conn.app.state, "service", None)
        controller = getattr(service, "controller", None)
    if controller is not None:
        return controller.get_aggregate_worker_status()

    snapshot = await query_controller_pod_states(
        conn, timeout=Environment.API_SERVER.GET_POD_STATES_TIMEOUT
    )
    if snapshot is not None:
        return _aggregate_from_snapshot(snapshot)

    progress: ProgressRouter | None = getattr(conn.app.state, "progress", None)
    if progress is None:
        return AggregateWorkerStatus()
    return build_aggregate_worker_status(progress._pod_state_tracker.pod_states)


def _aggregate_from_snapshot(snapshot: dict) -> AggregateWorkerStatus:
    """Rebuild the aggregate from a ``GET_POD_STATES`` snapshot dict.

    The controller encodes ``{"pod_states": {pod_index: msg.model_dump()},
    "worker_startup_states": {...}}``; we re-validate each ``pod_states``
    entry through :class:`WorkerPodStateMessage` so the existing
    :func:`build_aggregate_worker_status` aggregator does not have to learn
    the dict shape.

    The msgspec ``model_dump()`` shim emits the tag field ``message_type``
    which the constructor does not accept as a kwarg — strip it before
    re-instantiating.
    """
    raw_pods: dict[str, dict] = snapshot.get("pod_states", {}) or {}
    pod_states = {
        pod_index: WorkerPodStateMessage(
            **{k: v for k, v in raw.items() if k != "message_type"}
        )
        for pod_index, raw in raw_pods.items()
    }
    return build_aggregate_worker_status(pod_states)


@progress_router.get("/api/progress", response_model=ProgressResponse, tags=["API"])
async def get_progress(
    conn: HTTPConnection, component: ProgressDep
) -> ProgressResponse:
    """Get benchmark progress with full phase stats."""
    return ProgressResponse(
        phases=component._progress_tracker._phases,
        workers=await _get_controller_workers(conn),
        results_exported=component._results_exported,
    )

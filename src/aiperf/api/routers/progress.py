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
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.enums import CreditPhase
from aiperf.common.hooks import background_task
from aiperf.common.mixins.progress_tracker_mixin import (
    CombinedPhaseStats,
    ProgressTrackerMixin,
)
from aiperf.controller.system_controller import AggregateWorkerStatus

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


class ProgressRouter(ProgressTrackerMixin, BaseRouter):
    """Owns benchmark progress state and exposes /api/progress.

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

    def get_router(self) -> APIRouter:
        return progress_router

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
        )


def _get_controller_workers(conn: HTTPConnection) -> AggregateWorkerStatus:
    """Read aggregate worker status from controller-owned state when available."""
    controller = getattr(conn.app.state, "controller", None)
    if controller is None:
        service = getattr(conn.app.state, "service", None)
        controller = getattr(service, "controller", None)
    if controller is None:
        return AggregateWorkerStatus()
    return controller.get_aggregate_worker_status()


@progress_router.get("/api/progress", response_model=ProgressResponse, tags=["API"])
async def get_progress(
    conn: HTTPConnection, component: ProgressDep
) -> ProgressResponse:
    """Get benchmark progress with full phase stats."""
    return ProgressResponse(
        phases=component._progress_tracker._phases,
        workers=_get_controller_workers(conn),
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Progress router component -- owns benchmark progress state and /api/progress endpoint.

When running in Kubernetes mode, periodically pushes controller status and a
heartbeat to the AIPerfJob, then mirrors progress onto JobSet annotations.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Annotated, Any

from fastapi import APIRouter
from starlette.requests import HTTPConnection

from aiperf.api.models.responses import ProgressResponse
from aiperf.api.pod_state_rpc import query_controller_pod_states
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.enums import MessageType, SystemState
from aiperf.common.hooks import on_message, on_start
from aiperf.common.messages import (
    BaseServiceErrorMessage,
    RealtimeMetricsMessage,
    RealtimeServerMetricsMessage,
    ResultsExportedMessage,
    SystemStateChangedMessage,
)
from aiperf.common.mixins import PodStateTrackerMixin
from aiperf.common.mixins.progress_tracker_mixin import (
    CombinedPhaseStats,
    ProgressTrackerMixin,
)
from aiperf.common.mixins.realtime_metrics_mixin import RealtimeMetricsMixin
from aiperf.common.models import MetricResult
from aiperf.controller.system_controller_models import (
    AggregateWorkerStatus,
    build_aggregate_worker_status,
)

ProgressDep = Annotated["ProgressRouter", component_dependency("progress")]

logger = logging.getLogger(__name__)

progress_router = APIRouter()

# Interval between JobSet annotation patches (seconds)
_JOBSET_PATCH_INTERVAL = 10.0


def _build_progress_annotations(
    phases: dict[str, CombinedPhaseStats],
    system_state: SystemState,
) -> dict[str, str]:
    """Build annotation values from current progress state.

    Returns a dict of annotation key -> value for patching onto the JobSet
    and AIPerfJob CR. Always includes ``SYSTEM_STATE`` so observers can poll
    controller-side outer-lifecycle state without parsing status objects.
    """
    from aiperf.kubernetes.constants import ProgressAnnotations

    if not phases:
        return {
            ProgressAnnotations.STATUS: "initializing",
            ProgressAnnotations.SYSTEM_STATE: str(system_state),
        }

    named_phases = {
        phase_name: stats
        for phase_name, stats in phases.items()
        if stats.phase_name is not None
    }
    phase_name, active = max(
        (named_phases or phases).items(),
        key=lambda item: item[1].start_ns or 0,
    )

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
        ProgressAnnotations.SYSTEM_STATE: str(system_state),
    }

    if pct is not None:
        annotations[ProgressAnnotations.PERCENT] = f"{pct:.1f}"

    if total is not None and total > 0:
        annotations[ProgressAnnotations.REQUESTS] = f"{completed}/{total}"

    return annotations


def _controller_failure_status_patch(controller_failure: str | None) -> dict[str, str]:
    """Return the controller's fatal failure field when one was reported."""
    return {"controllerFailure": controller_failure} if controller_failure else {}


def _current_phase_name(phases: dict[str, CombinedPhaseStats]) -> str | None:
    """Name the most recently started phase, mirroring JobProgress.current_phase.

    Phases carrying an explicit identity win over legacy aggregate entries, the
    same preference ``JobProgress._concrete_phases`` applies.
    """
    if not phases:
        return None
    concrete = {
        name: stats
        for name, stats in phases.items()
        if stats.phase_name is not None or name != str(stats.phase)
    }
    return max((concrete or phases).items(), key=lambda item: item[1].start_ns or 0)[0]


def _is_json_patch_test_failure(exc: Any) -> bool:
    """Tell a rejected ``test`` op apart from a rejected payload.

    Both arrive as 422. Swallowing the status code wholesale would hide a CRD
    structural-schema violation -- a new ``status`` key of the wrong type, say
    -- as a lost race, and status updates would stop silently on every CR that
    has a phase. The apiserver's wording for a failed test op has changed
    across evanphx/json-patch versions, hence several markers.
    """
    if exc.status not in (409, 422):
        return False
    body = exc.body
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    haystack = f"{body} {exc.reason}".lower()
    return any(
        marker in haystack
        for marker in (
            "test operation does not apply",
            "testing value",
            "jsonpatch test",
            "test failed",
        )
    )


# Status keys whose JSON-patch value is a whole-object SNAPSHOT, not a merge.
# _merge_patch_value recursively unions dicts, which is right for every key that
# accumulates (phases, summary) and wrong for a key that must mirror one tick:
# status.serverMetrics is keyed by metric name with dict-valued stats, so a
# metric that stops being projected -- exactly what the projection's caps do --
# would linger in the CR with stale values indistinguishable from live ones.
# A JSON-patch "add" on an existing member replaces it outright, which is the
# snapshot semantic. Keep this list minimal; merging is correct by default.
_SNAPSHOT_STATUS_KEYS: frozenset[str] = frozenset({"serverMetrics"})


def _merge_patch_value(existing: Any, patch: Any) -> Any:
    """Resolve one status key against its live value using RFC 7386 semantics.

    The terminal fence sends the status update as a JSON patch, whose ``add``
    op replaces an object member outright instead of merging into it.
    Pre-resolving each value against the CR read moments earlier keeps the
    fenced write equivalent to the merge-patch it stands in for.
    """
    if not isinstance(patch, dict):
        return patch
    merged = dict(existing) if isinstance(existing, dict) else {}
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = _merge_patch_value(merged.get(key), value)
    return merged


class ProgressRouter(
    PodStateTrackerMixin, RealtimeMetricsMixin, ProgressTrackerMixin, BaseRouter
):
    """Owns benchmark progress state and exposes /api/progress.

    ``progress.workers`` prefers the SystemController's authoritative state,
    queried over the command bus when the API is a Kubernetes sidecar. This
    router's independently bus-fed ``PodStateTrackerMixin`` cache remains the
    availability fallback while the controller is starting or stopping.

    In Kubernetes mode, a background task periodically patches the JobSet
    annotations with current progress so that ``kubectl get jobset`` or
    external controllers can inspect benchmark status.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._k8s_job_id: str | None = os.environ.get("AIPERF_JOB_ID")
        self._k8s_job_uid: str | None = os.environ.get("AIPERF_JOB_UID")
        self._k8s_namespace: str | None = os.environ.get("AIPERF_NAMESPACE")
        self._k8s_patching_enabled = bool(
            self._k8s_job_id and self._k8s_job_uid and self._k8s_namespace
        )
        self._last_patched_jobset_annotations: dict[str, str] = {}
        self._last_patched_aiperfjob_annotations: dict[str, str] = {}
        # Flips True only after the SystemController publishes
        # ResultsExportedMessage — i.e. after ExporterManager.export_data()
        # AND (in K8s mode) write_ready_marker() have completed. The operator
        # gates JobProgress.is_complete on this so sub-second benchmarks don't
        # let the kopf-timer monitor claim completion mid-export.
        self._results_exported: bool = False
        self._controller_failure: str | None = None
        # Mirrors SystemController's outer-lifecycle SystemState. Updated via
        # SYSTEM_STATE_CHANGED bus messages so the API sidecar (which lives in
        # a separate container) can surface controller-side state on
        # /api/progress without an in-process controller handle.
        self._system_state: SystemState = SystemState.INITIALIZING
        # Latest REALTIME_SERVER_METRICS payload, kept for the curated
        # status.serverMetrics projection. Held raw and projected at push time
        # so the caps stay reconfigurable without a restart-ordering hazard.
        self._server_metrics: dict[str, Any] | None = None

    def get_router(self) -> APIRouter:
        return progress_router

    def _schedule_status_push(self) -> None:
        """Fire a best-effort AIPerfJob status push, but only in Kubernetes mode.

        Every caller is a bus handler on a message the controller publishes in
        EVERY run mode, several of them at per-tick rates (REALTIME_METRICS,
        CREDIT_PHASE_PROGRESS). Outside Kubernetes there is no CR to patch, so
        the guard lives here rather than inside the coroutine: checking a bool
        is free, whereas spawning a task per bus message just to have it return
        on its first line is not.
        """
        if not self._k8s_patching_enabled:
            return
        asyncio.create_task(self._patch_aiperfjob_status())  # noqa: RUF006

    @on_message(MessageType.RESULTS_EXPORTED)
    async def _on_results_exported(self, _message: ResultsExportedMessage) -> None:
        """Record that the controller has finished writing artifacts to disk."""
        self._results_exported = True
        self._schedule_status_push()

    @on_message(MessageType.REALTIME_SERVER_METRICS)
    async def _on_realtime_server_metrics(
        self, message: RealtimeServerMetricsMessage
    ) -> None:
        """Cache the latest server-metrics scrape for the CR fallback projection.

        No push is scheduled here: server metrics scrape at their own cadence
        and the value rides along on the next progress push, which already
        fires at per-tick rates.

        Guarded on Kubernetes mode for the same reason ``_schedule_status_push``
        is: this fires at ~3Hz in every run mode, and the dump is pure waste
        outside Kubernetes where there is no CR to carry it.
        """
        if not self._k8s_patching_enabled:
            return
        self._server_metrics = message.model_dump(
            mode="json", exclude={"message_type", "service_id"}, exclude_none=True
        )

    @on_message(MessageType.SERVICE_ERROR)
    async def _on_service_error(self, message: BaseServiceErrorMessage) -> None:
        """Push a controller-plane failure before its pod exits."""
        self._controller_failure = f"{message.service_id}: {message.error.message}"
        self._schedule_status_push()

    @on_message(MessageType.SYSTEM_STATE_CHANGED)
    async def _on_system_state_changed(
        self, message: SystemStateChangedMessage
    ) -> None:
        """Record the controller's most-recent outer-lifecycle SystemState."""
        self._system_state = message.state
        self._schedule_status_push()

    @on_message(MessageType.CREDIT_PHASE_PROGRESS)
    async def _on_credit_phase_progress_status_push(self, _message: Any) -> None:
        """Fire a status push on every phase-progress tick."""
        self._schedule_status_push()

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete_status_push(self, _message: Any) -> None:
        """Fire a status push when a credit phase finishes."""
        self._schedule_status_push()

    @on_message(MessageType.REALTIME_METRICS)
    async def _on_realtime_metrics_status_push(
        self, _message: RealtimeMetricsMessage
    ) -> None:
        """Fire a status push on every realtime-metrics update.

        RealtimeMetricsMixin._on_realtime_metrics already updates self._metrics
        before this handler runs (message dispatch is ordered by subscription
        order; mixin subscribes first). The push writes liveMetrics and summary
        so kubectl columns update without waiting for the kopf monitor timer.
        """
        self._schedule_status_push()

    @on_start
    async def _start_k8s_patch_loops(self) -> None:
        """Start the CR-patching loops, in Kubernetes mode only.

        These are registered here rather than with ``@background_task`` because
        the decorator starts them unconditionally: a local ``aiperf profile``
        would then hold two forever-sleeping tasks whose only job is to wake up
        and return because there is no AIPerfJob to patch.
        """
        if not self._k8s_patching_enabled:
            return
        # Imported here, not at module scope: this is the only use of anything
        # from aiperf.kubernetes in this module's import graph, and every other
        # k8s import below is already function-local for the same reason.
        from aiperf.kubernetes.constants import CONTROLLER_HEARTBEAT_INTERVAL_SECONDS

        self.start_background_task(
            self._patch_aiperfjob_status,
            interval=CONTROLLER_HEARTBEAT_INTERVAL_SECONDS,
            immediate=False,
            stop_event=self._stop_requested_event,
        )
        self.start_background_task(
            self._patch_jobset_progress,
            interval=_JOBSET_PATCH_INTERVAL,
            immediate=False,
            stop_event=self._stop_requested_event,
        )

    async def _patch_aiperfjob_status(self) -> None:
        """Push current progress state directly onto the AIPerfJob CR status subresource.

        Called immediately on state/phase changes and at the bounded heartbeat
        cadence. Best-effort: any k8s API failure is logged at debug and dropped;
        the operator's reconciliation catches gaps.
        """
        if not self._k8s_patching_enabled:
            return
        try:
            await _push_aiperfjob_status(
                job_id=self._k8s_job_id,  # type: ignore[arg-type]
                job_uid=self._k8s_job_uid,  # type: ignore[arg-type]
                namespace=self._k8s_namespace,  # type: ignore[arg-type]
                phases=dict(self._progress_tracker._phases),
                system_state=self._system_state,
                results_exported=self._results_exported,
                controller_failure=self._controller_failure,
                metrics=list(self._metrics),
                server_metrics=self._server_metrics,
            )
        except Exception:  # noqa: BLE001
            self.debug("Failed to push AIPerfJob status update")

    async def _patch_jobset_progress(self) -> None:
        """Periodically patch JobSet annotations with current progress."""
        if not self._k8s_patching_enabled:
            return

        annotations = _build_progress_annotations(
            self._progress_tracker._phases,
            self._system_state,
        )

        if annotations != self._last_patched_jobset_annotations:
            try:
                await _patch_jobset_annotations(
                    job_id=self._k8s_job_id,  # type: ignore[arg-type]
                    job_uid=self._k8s_job_uid,  # type: ignore[arg-type]
                    namespace=self._k8s_namespace,  # type: ignore[arg-type]
                    annotations=annotations,
                )
                self._last_patched_jobset_annotations = annotations
            except Exception:  # noqa: BLE001 - periodic JobSet annotation patch is best-effort; k8s API flakes must not crash the background task
                self.debug("Failed to patch JobSet progress annotations")

        # Mirror the same annotations onto the AIPerfJob CR so kubectl
        # watchers can poll a single object instead of chasing JobSets.
        # Best-effort: AIPerfJob patch failures must not crash the loop
        # nor invalidate the JobSet patch above.
        if annotations != self._last_patched_aiperfjob_annotations:
            try:
                await _patch_aiperfjob_annotations(
                    job_id=self._k8s_job_id,  # type: ignore[arg-type]
                    job_uid=self._k8s_job_uid,  # type: ignore[arg-type]
                    namespace=self._k8s_namespace,  # type: ignore[arg-type]
                    annotations=annotations,
                )
                self._last_patched_aiperfjob_annotations = annotations
            except Exception:  # noqa: BLE001 - best-effort; AIPerfJob patch must not crash the background task
                self.debug("Failed to patch AIPerfJob progress annotations")


async def _patch_jobset_annotations(
    job_id: str,
    job_uid: str,
    namespace: str,
    annotations: dict[str, str],
) -> None:
    """Patch annotations on the JobSet for the given job."""
    from kubernetes_asyncio import client

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import (
        AIPERF_API_VERSION,
        JOBSET_GROUP,
        JOBSET_PLURAL,
        JOBSET_VERSION,
    )

    jobset_name = f"aiperf-{job_id}"

    async with k8s_client() as api:
        custom_api = client.CustomObjectsApi(api)
        resource = await custom_api.get_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=namespace,
            name=jobset_name,
        )
        metadata = resource.get("metadata") if isinstance(resource, dict) else None
        if not isinstance(metadata, dict):
            raise ValueError(f"JobSet {namespace}/{jobset_name} has no metadata")
        owner_references = metadata.get("ownerReferences") or []
        owned_by_job = any(
            isinstance(owner, dict)
            and owner.get("apiVersion") == AIPERF_API_VERSION
            and owner.get("kind") == "AIPerfJob"
            and owner.get("name") == job_id
            and owner.get("uid") == job_uid
            and owner.get("controller") is True
            for owner in owner_references
        )
        if not owned_by_job:
            raise ValueError(
                f"JobSet {namespace}/{jobset_name} is not owned by the expected "
                "AIPerfJob incarnation"
            )
        patch_body = _uid_fenced_annotation_patch(
            metadata=metadata,
            expected_uid=metadata.get("uid"),
            annotations=annotations,
        )
        await custom_api.patch_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=namespace,
            name=jobset_name,
            body=patch_body,
            _content_type="application/json-patch+json",
        )


async def _patch_aiperfjob_annotations(
    job_id: str,
    job_uid: str,
    namespace: str,
    annotations: dict[str, str],
) -> None:
    """Patch annotations on the AIPerfJob CR for the given job.

    The exact AIPerfJob UID is tested atomically with the annotation update so
    a controller pod from an old incarnation cannot mutate a replacement CR.
    """
    from kubernetes_asyncio import client

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import AIPERF_GROUP, AIPERF_PLURAL, AIPERF_VERSION

    async with k8s_client() as api:
        custom_api = client.CustomObjectsApi(api)
        resource = await custom_api.get_namespaced_custom_object(
            group=AIPERF_GROUP,
            version=AIPERF_VERSION,
            plural=AIPERF_PLURAL,
            namespace=namespace,
            name=job_id,
        )
        metadata = resource.get("metadata") if isinstance(resource, dict) else None
        if not isinstance(metadata, dict) or metadata.get("uid") != job_uid:
            raise ValueError(
                f"AIPerfJob {namespace}/{job_id} is not the expected incarnation"
            )
        patch_body = _uid_fenced_annotation_patch(
            metadata=metadata,
            expected_uid=job_uid,
            annotations=annotations,
        )
        await custom_api.patch_namespaced_custom_object(
            group=AIPERF_GROUP,
            version=AIPERF_VERSION,
            plural=AIPERF_PLURAL,
            namespace=namespace,
            name=job_id,
            body=patch_body,
            _content_type="application/json-patch+json",
        )


async def _push_aiperfjob_status(
    *,
    job_id: str,
    job_uid: str,
    namespace: str,
    phases: dict[str, CombinedPhaseStats],
    system_state: SystemState,
    results_exported: bool,
    controller_failure: str | None = None,
    metrics: list[MetricResult] | None = None,
    server_metrics: dict[str, Any] | None = None,
) -> None:
    """Refresh the controller heartbeat and merge-patch current progress.

    The heartbeat uses a UID-fenced metadata JSON patch. The progress update
    keeps the existing status-subresource merge patch so operator-owned fields
    (phase, conditions, etc.) remain untouched, except when the CR already
    carries a phase — then it is sent as a phase-fenced JSON patch instead.
    """
    from kubernetes_asyncio import client

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.constants import Annotations
    from aiperf.kubernetes.cr_refs import AIPERF_GROUP, AIPERF_PLURAL, AIPERF_VERSION
    from aiperf.kubernetes.phase import format_timestamp
    from aiperf.kubernetes.server_metrics_projection import (
        project_server_metrics_for_cr,
    )

    async with k8s_client() as api:
        custom_api = client.CustomObjectsApi(api)

        # UID fence: abort if this CR is a different incarnation
        resource = await custom_api.get_namespaced_custom_object(
            group=AIPERF_GROUP,
            version=AIPERF_VERSION,
            plural=AIPERF_PLURAL,
            namespace=namespace,
            name=job_id,
        )
        metadata = resource.get("metadata") if isinstance(resource, dict) else None
        if not isinstance(metadata, dict) or metadata.get("uid") != job_uid:
            raise ValueError(
                f"AIPerfJob {namespace}/{job_id} UID mismatch — skipping push"
            )

        heartbeat_patch = _uid_fenced_annotation_patch(
            metadata=metadata,
            expected_uid=job_uid,
            annotations={Annotations.CONTROLLER_HEARTBEAT: format_timestamp()},
        )
        await custom_api.patch_namespaced_custom_object(
            group=AIPERF_GROUP,
            version=AIPERF_VERSION,
            plural=AIPERF_PLURAL,
            namespace=namespace,
            name=job_id,
            body=heartbeat_patch,
            _content_type="application/json-patch+json",
        )

        phases_data, current_phase = _build_phases_payload(phases)

        # Find the primary (profiling-kind) phase for top-level request counters,
        # mirroring JobProgress.primary_phase_stats: take the profiling-kind phase
        # with the latest start_ns.
        profiling = {
            n: s
            for n, s in phases.items()
            if (s.phase_kind or "profiling") == "profiling"
        }
        primary_stats = (
            max(profiling.values(), key=lambda s: s.start_ns or 0)
            if profiling
            else None
        )

        # Build a metrics dict in the shape MetricsSummary.from_metrics() accepts.
        live_metrics_dict: dict[str, Any] | None = None
        if metrics:
            live_metrics_dict = {
                "metrics": {
                    m.tag: m.model_dump(mode="json", exclude_none=True, exclude={"tag"})
                    for m in metrics
                }
            }

        # If results are already flowing but the controller hasn't published a
        # state-change past INITIALIZING/CONFIGURING/READY yet, advance subPhase
        # to "profiling" so kubectl columns don't show a stale lifecycle state.
        effective_system_state = system_state
        if live_metrics_dict and system_state in (
            SystemState.INITIALIZING,
            SystemState.CONFIGURING,
            SystemState.READY,
        ):
            effective_system_state = SystemState.PROFILING

        status_patch: dict[str, Any] = {
            "subPhase": str(effective_system_state),
            "phases": phases_data,
        }
        if current_phase is not None:
            status_patch["currentPhase"] = current_phase
        if primary_stats is not None:
            status_patch["requestsCompleted"] = primary_stats.requests_completed
            status_patch["requestsTotal"] = primary_stats.total_expected_requests or 0
            status_patch["requestsPerSecond"] = round(
                primary_stats.requests_per_second or 0.0, 2
            )

        if live_metrics_dict:
            from aiperf.kubernetes.crd_models import MetricsSummary

            summary = MetricsSummary.from_metrics(live_metrics_dict)
            summary_dict = summary.to_status_dict()
            if summary_dict:
                status_patch["liveMetrics"] = live_metrics_dict
                status_patch["liveSummary"] = summary_dict
                # Also write to summary so kubectl printer columns
                # (which reference status.summary.*) show live values during
                # the run. The completion handler overwrites summary with final
                # authoritative values when the benchmark finishes.
                status_patch["summary"] = summary_dict

        if curated_server_metrics := project_server_metrics_for_cr(server_metrics):
            status_patch["serverMetrics"] = curated_server_metrics

        if results_exported:
            status_patch["resultsExported"] = True
        status_patch.update(_controller_failure_status_patch(controller_failure))

        existing_status = resource.get("status")
        await _write_status_patch(
            custom_api,
            job_id=job_id,
            namespace=namespace,
            existing_status=existing_status
            if isinstance(existing_status, dict)
            else {},
            status_patch=status_patch,
        )


def _build_phases_payload(
    phases: dict[str, CombinedPhaseStats],
) -> tuple[dict[str, Any], str | None]:
    """Serialize per-phase progress and name the phase the job is currently in.

    Phase data goes through ``PhaseProgress`` (camelCase, the same schema as
    the completion handler's final snapshot) so merge-patch never accumulates
    both snake_case and camelCase versions of the same fields.

    The returned name is always a key of the returned map, or ``None``.
    ``_requests_progress_percent`` falls back to alphabetized iteration when
    ``status.currentPhase`` misses ``status.phases`` -- which resolves to
    warmup's 100% -- so a dangling pointer is strictly worse than no pointer.
    A phase that has started but not yet sent a request is the current-phase
    winner while being dropped from the map, so a zeroed entry is emitted for
    it here rather than loosening the shared builder.
    """
    from aiperf.operator.handlers.monitor import _build_phase_progress

    phases_data: dict[str, Any] = {}
    for name, stats in phases.items():
        if phase_progress := _build_phase_progress(stats):
            phases_data[name] = phase_progress.to_k8s_dict()

    current_phase = _current_phase_name(phases)
    if current_phase is not None and current_phase not in phases_data:
        zeroed = _build_phase_progress(phases[current_phase], allow_empty=True)
        if zeroed is None:
            return phases_data, None
        phases_data[current_phase] = zeroed.to_k8s_dict()
    return phases_data, current_phase


async def _write_status_patch(
    custom_api: Any,
    *,
    job_id: str,
    namespace: str,
    existing_status: dict[str, Any],
    status_patch: dict[str, Any],
) -> None:
    """Write the controller's status update, fenced against terminal transitions.

    kopf clears ``currentPhase``/``subPhase`` when it stamps a terminal phase
    (``StatusBuilder.set_phase``); an in-flight push that passed the UID fence
    just before that patch would resurrect both keys. A merge patch cannot
    carry a precondition, so once the CR has a phase at all the write goes out
    as a JSON patch whose leading ``test`` op binds it to the phase just read,
    and the apiserver -- not wall-clock order -- settles the race. A CR with no
    phase yet has not reached Pending: there is nothing to race against, and a ``test``
    op on an absent path would fail with 422.

    The terminal skip drops the whole payload, not just the racy keys -- most
    of it (``summary`` above all) would overwrite the completion handler's
    authoritative final values with live ones. Two dropped keys are worth
    naming because they are read elsewhere: ``resultsExported`` and
    ``controllerFailure``. Neither is lost in practice -- a terminal CR has
    already had its completion or failure adjudicated by the operator, which
    is what set the terminal phase in the first place -- but a caller that
    starts relying on a *post*-terminal push to deliver either one will not
    get it.
    """
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.cr_refs import AIPERF_GROUP, AIPERF_PLURAL, AIPERF_VERSION
    from aiperf.operator.handlers.monitor import _as_phase

    ref = {
        "group": AIPERF_GROUP,
        "version": AIPERF_VERSION,
        "plural": AIPERF_PLURAL,
        "namespace": namespace,
        "name": job_id,
    }
    observed_phase = existing_status.get("phase")

    if not isinstance(observed_phase, str) or not observed_phase:
        await custom_api.patch_namespaced_custom_object_status(
            **ref,
            body={"status": status_patch},
            _content_type="application/merge-patch+json",
        )
        return

    if _as_phase(observed_phase).is_terminal:
        logger.debug(
            "Skipping status push for %s/%s: CR is already terminal (%s)",
            namespace,
            job_id,
            observed_phase,
        )
        return

    patch_ops: list[dict[str, Any]] = [
        {"op": "test", "path": "/status/phase", "value": observed_phase}
    ]
    patch_ops.extend(
        {
            "op": "add",
            "path": f"/status/{key}",
            "value": value
            if key in _SNAPSHOT_STATUS_KEYS
            else _merge_patch_value(existing_status.get(key), value),
        }
        for key, value in status_patch.items()
    )
    try:
        await custom_api.patch_namespaced_custom_object_status(
            **ref,
            body=patch_ops,
            _content_type="application/json-patch+json",
        )
    except ApiException as exc:
        if _is_json_patch_test_failure(exc):
            logger.debug(
                "Status push for %s/%s lost the race with a phase transition (%s)",
                namespace,
                job_id,
                exc.status,
            )
            return
        logger.warning(
            "Status push for %s/%s was rejected (%s: %s): %s",
            namespace,
            job_id,
            exc.status,
            exc.reason,
            exc.body,
        )
        raise


def _uid_fenced_annotation_patch(
    *,
    metadata: dict[str, Any],
    expected_uid: str | None,
    annotations: dict[str, str],
) -> list[dict[str, Any]]:
    """Build an annotation patch bound to one resource incarnation."""
    if not isinstance(expected_uid, str) or not expected_uid:
        raise ValueError("A Kubernetes resource UID is required for annotation writes")
    if metadata.get("uid") != expected_uid:
        raise ValueError("Kubernetes resource UID changed before patch construction")

    patch: list[dict[str, Any]] = [
        {"op": "test", "path": "/metadata/uid", "value": expected_uid}
    ]
    current_annotations = metadata.get("annotations")
    if isinstance(current_annotations, dict):
        patch.extend(
            {
                "op": "add",
                "path": "/metadata/annotations/"
                + key.replace("~", "~0").replace("/", "~1"),
                "value": value,
            }
            for key, value in annotations.items()
        )
        return patch

    resource_version = metadata.get("resourceVersion")
    if not isinstance(resource_version, str) or not resource_version:
        raise ValueError(
            "Kubernetes resourceVersion is required when annotations are absent"
        )
    patch.extend(
        [
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": resource_version,
            },
            {"op": "add", "path": "/metadata/annotations", "value": annotations},
        ]
    )
    return patch


async def _get_controller_workers(
    conn: HTTPConnection, component: ProgressRouter | None = None
) -> AggregateWorkerStatus:
    """Resolve aggregate status from the controller, then the bus cache.

    An empty successful controller snapshot is authoritative. This distinction
    prevents a stale sidecar cache from resurrecting workers after the
    controller has intentionally cleared its state.
    """
    snapshot = await query_controller_pod_states(conn)
    if snapshot is not None:
        return build_aggregate_worker_status(snapshot.pod_states)
    if component is not None:
        return build_aggregate_worker_status(component._pod_state_tracker.pod_states)
    return AggregateWorkerStatus()


@progress_router.get("/api/progress", response_model=ProgressResponse, tags=["API"])
async def get_progress(
    conn: HTTPConnection, component: ProgressDep
) -> ProgressResponse:
    """Get benchmark progress with full phase stats."""
    return ProgressResponse(
        phases=component._progress_tracker._phases,
        workers=await _get_controller_workers(conn, component),
        results_exported=component._results_exported,
        system_state=component._system_state,
    )

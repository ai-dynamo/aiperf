# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""monitor_progress timer handler logic for AIPerfJob CRD.

This module contains the business logic only — no kopf decorators.
Decorators live in ``aiperf.operator.main``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import aiohttp
import kopf
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient, CustomObjectsApi
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.constants import Containers, JobSetLabels
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.operator import events
from aiperf.operator.client_cache import (
    _shutdown_sent,
    _warned_pod_restarts,
    close_progress_client,
    get_or_create_progress_client,
    is_cancellation_requested,
    is_completion_claimed,
    job_key,
    try_claim_completion,
)
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers.completion import (
    _parse_metrics_from_files,
    fetch_results_with_retry,
    handle_completion,
)
from aiperf.operator.models import MetricsSummary, PhaseProgress
from aiperf.operator.status import (
    ConditionType,
    Phase,
    StatusBuilder,
    parse_timestamp,
)

if TYPE_CHECKING:
    from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats

logger = logging.getLogger(__name__)


def _get_elapsed_seconds(status: dict[str, Any]) -> float | None:
    """Calculate elapsed seconds since startTime, or None if unavailable."""
    start_time = status.get("startTime")
    if not start_time:
        return None
    try:
        start_dt = parse_timestamp(start_time)
        return (datetime.now(timezone.utc) - start_dt).total_seconds()
    except (ValueError, TypeError):
        return None


def _get_job_timeout(spec: dict[str, Any]) -> float:
    """Get job timeout from spec or global default. 0 means no timeout."""
    return float(spec.get("timeoutSeconds", OperatorEnvironment.JOB_TIMEOUT_SECONDS))


def _classify_jobset_failure(jobset_status: dict[str, Any]) -> tuple[bool, str | None]:
    """Classify whether a JobSet failure should fail the benchmark."""
    replicated = {
        rj.get("name"): rj for rj in jobset_status.get("replicatedJobsStatus", [])
    }
    controller_failed = replicated.get("controller", {}).get("failed", 0) > 0
    workers_failed = replicated.get("workers", {}).get("failed", 0) > 0

    if controller_failed:
        return True, "controller"
    if workers_failed:
        return False, "workers"
    return True, None


def _apply_controller_progress_status(
    patch: kopf.Patch,
    sb: StatusBuilder,
    progress: Any,
    current_phase: Phase,
) -> None:
    """Apply controller-authored progress to CR status."""
    sb.set_worker_aggregate_status(progress.workers.model_dump())

    if not progress.current_phase:
        return

    patch.status["currentPhase"] = progress.current_phase

    if progress.current_phase == "profiling":
        sb.set_phase(Phase.RUNNING)
        sb.conditions.set_true(
            ConditionType.BENCHMARK_RUNNING,
            "BenchmarkStarted",
            "Benchmark is running",
        )
    elif current_phase in (Phase.PENDING, Phase.QUEUED, Phase.INITIALIZING):
        sb.set_phase(Phase.INITIALIZING)


async def _delete_jobset_silently(
    custom: CustomObjectsApi,
    namespace: str,
    jobset_name: str,
    *,
    context: str,
) -> None:
    """Delete a JobSet, swallowing 404 and logging any other ApiException."""
    try:
        await custom.delete_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=namespace,
            name=jobset_name,
        )
        logger.info(f"Deleted JobSet {jobset_name} after {context}")
    except ApiException as e:
        if e.status != 404:
            logger.warning(
                f"Failed to delete JobSet {jobset_name} after {context}: {e}"
            )


async def _reconcile_missing_jobset(
    custom: CustomObjectsApi,
    *,
    namespace: str,
    name: str,
    jobset_name: str,
    current_phase: Phase,
    sb: StatusBuilder,
) -> bool:
    """Reconcile the "JobSet not found" case with a fresh CR re-read.

    Returns True if the caller should short-circuit (terminal phase already
    reached by the completion handler); False if the caller should mark FAILED.
    """
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        logger.debug(
            f"JobSet {jobset_name} not found but phase is already "
            f"{current_phase} - skipping"
        )
        return True

    # Re-read the CR to catch completion handler's update
    # (it may have set phase=Completed and deleted the JobSet
    # between our phase read and the JobSet lookup).
    await asyncio.sleep(2)

    try:
        fresh = await custom.get_namespaced_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            namespace=namespace,
            name=name,
        )
        fresh_phase = fresh.get("status", {}).get("phase", "")
        if fresh_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
            logger.debug(
                f"JobSet {jobset_name} not found but fresh phase is "
                f"{fresh_phase} - skipping"
            )
            return True
    except Exception:
        logger.exception(
            f"Stale-read recovery failed while reconciling "
            f"{namespace}/{name} after JobSet {jobset_name} not found"
        )
    sb.set_phase(Phase.FAILED).set_error("JobSet not found")
    sb.finalize()
    return False


async def _check_job_timeout(
    custom: CustomObjectsApi,
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
    namespace: str,
    jobset_name: str | None,
    job_id: str,
    key: str,
    sb: StatusBuilder,
) -> bool:
    """Fail the CR if elapsed time exceeds the configured timeout.

    Returns True if the job timed out and the caller should return early.
    """
    timeout_sec = _get_job_timeout(spec)
    if timeout_sec <= 0:
        return False

    elapsed = _get_elapsed_seconds(status)
    if elapsed is None or elapsed <= timeout_sec:
        return False

    sb.set_phase(Phase.FAILED).set_error(
        f"Job timed out after {elapsed:.0f}s (limit: {timeout_sec:.0f}s)"
    )
    sb.set_completion_time()
    sb.finalize()
    events.job_timeout(body, job_id, elapsed)
    if jobset_name:
        await _delete_jobset_silently(custom, namespace, jobset_name, context="timeout")
    await close_progress_client(key)
    return True


async def _handle_jobset_terminal_condition(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    jobset_status: dict[str, Any],
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    key: str,
    sb: StatusBuilder,
) -> bool:
    """Inspect JobSet terminal conditions and handle completion/failure.

    Returns True if the caller should return early (terminal state handled).
    """
    for condition in jobset_status.get("conditions", []):
        if condition.get("status") != "True":
            continue
        cond_type = condition.get("type")
        if cond_type == "Completed":
            # Only enter handle_completion if we successfully claim the
            # completion branch via a durable CR annotation. A claim
            # set by a previous operator run (or by the annotation
            # handler) causes this to return False and we skip.
            # try_claim_completion also latches the in-process
            # _shutdown_sent set, so a second concurrent handler in
            # the same process short-circuits without hitting the API.
            if await try_claim_completion(namespace, name, body):
                await handle_completion(
                    body, namespace, jobset_name, job_id, status=status, sb=sb
                )
            await close_progress_client(key)
            return True
        if cond_type == "Failed" and await _handle_jobset_failed_condition(
            body=body,
            condition=condition,
            jobset_status=jobset_status,
            job_id=job_id,
            key=key,
            sb=sb,
        ):
            return True
    return False


async def _handle_jobset_failed_condition(
    *,
    body: dict[str, Any],
    condition: dict[str, Any],
    jobset_status: dict[str, Any],
    job_id: str,
    key: str,
    sb: StatusBuilder,
) -> bool:
    """Handle a single JobSet 'Failed' condition.

    Returns True if the failure was fatal and the caller should return early.
    """
    is_fatal, failed_scope = _classify_jobset_failure(jobset_status)
    if is_fatal:
        sb.set_phase(Phase.FAILED)
        sb.set_error(condition.get("message", "JobSet failed"))
        sb.finalize()
        events.failed(body, job_id, condition.get("message", "JobSet failed"))
        await close_progress_client(key)
        return True

    logger.warning(
        "Ignoring non-fatal JobSet failure for %s: failed_scope=%s message=%s",
        job_id,
        failed_scope,
        condition.get("message", "JobSet failed"),
    )

    # The JobSet default cascade kills the controller pod even when
    # only workers failed. If the controller pod is gone, the
    # benchmark is unrecoverable regardless of the failure scope.
    ctrl_replicated = {
        rj.get("name"): rj for rj in jobset_status.get("replicatedJobsStatus", [])
    }
    ctrl_active = ctrl_replicated.get("controller", {}).get("active", 0)
    ctrl_succeeded = ctrl_replicated.get("controller", {}).get("succeeded", 0)
    if ctrl_active == 0 and ctrl_succeeded == 0:
        error_msg = (
            f"Controller terminated after worker failure "
            f"(JobSet cascade): {condition.get('message', '')}"
        )
        logger.error(
            "Escalating non-fatal failure to fatal for %s: "
            "controller pod is gone (active=%s, succeeded=%s)",
            job_id,
            ctrl_active,
            ctrl_succeeded,
        )
        sb.set_phase(Phase.FAILED)
        sb.set_error(error_msg)
        sb.finalize()
        events.failed(body, job_id, error_msg)
        await close_progress_client(key)
        return True
    return False


def _update_worker_counts(
    *,
    status: dict[str, Any],
    jobset_status: dict[str, Any],
    sb: StatusBuilder,
) -> tuple[int, int, int]:
    """Update worker ready/total on StatusBuilder.

    Returns (workers_ready, workers_succeeded, total_workers).
    """
    total_workers = status.get("workers", {}).get("total", 0)
    workers_ready = 0
    workers_succeeded = 0

    for rj in jobset_status.get("replicatedJobsStatus", []):
        if rj.get("name") == "workers":
            workers_ready = rj.get("ready", 0)
            workers_succeeded = rj.get("succeeded", 0)
            # Derive total from JobSet if CRD status doesn't have it yet
            if total_workers == 0:
                total_workers = (
                    rj.get("ready", 0)
                    + rj.get("active", 0)
                    + rj.get("succeeded", 0)
                    + rj.get("failed", 0)
                    + rj.get("suspended", 0)
                ) or 1  # Fallback to 1 if all zero
            sb.set_workers(workers_ready, total_workers)

    return workers_ready, workers_succeeded, total_workers


def _should_poll_progress(
    effective_phase: Phase,
    workers_succeeded: int,
    total_workers: int,
) -> bool:
    """Decide whether this tick should poll controller progress.

    Including PENDING is deliberate: for fast-completing benchmarks the
    worker JobSet can transition from active → succeeded inside a single
    monitor-poll interval, so the phase-transition guard
    (PENDING → INITIALIZING on workers_ready > 0) may never fire. Without
    this, `_fetch_progress()` is skipped, the controller's completion
    annotation is observed too late, and the CR stays Pending forever.
    `_fetch_progress` handles controller-not-yet-ready gracefully, so
    polling during early startup is safe — it just no-ops.
    """
    if effective_phase in (Phase.PENDING, Phase.INITIALIZING, Phase.RUNNING):
        return True
    return workers_succeeded > 0 and workers_succeeded >= total_workers


async def _poll_controller_progress(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    patch: kopf.Patch,
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    key: str,
    effective_phase: Phase,
    sb: StatusBuilder,
) -> bool:
    """Poll controller progress and handle completion.

    Returns True if the caller should return early (benchmark completed).
    """
    progress_client = await get_or_create_progress_client(key)
    benchmark_complete = await _fetch_progress(
        namespace,
        jobset_name,
        patch,
        sb,
        progress_client,
        key,
        effective_phase,
        body=body,
    )

    if not benchmark_complete:
        return False

    # If benchmark is done, fetch results then shutdown controller
    if not await try_claim_completion(namespace, name, body):
        await close_progress_client(key)
        return True
    logger.info(
        f"Benchmark complete for {jobset_name}, "
        f"fetching results and shutting down controller"
    )
    host = controller_dns_name(jobset_name, namespace)
    await handle_completion(body, namespace, jobset_name, job_id, status=status, sb=sb)
    # Shutdown controller after results are fetched
    await progress_client.send_shutdown(host)
    await close_progress_client(key)
    return True


async def _fetch_jobset_or_reconcile(
    custom: CustomObjectsApi,
    *,
    namespace: str,
    name: str,
    jobset_name: str,
    current_phase: Phase,
    key: str,
    sb: StatusBuilder,
) -> dict[str, Any] | None:
    """Fetch the JobSet, reconciling the 404 (deleted) case.

    Returns the JobSet dict on success, or None if the caller should return
    early (404 path already handled by `_reconcile_missing_jobset`).
    """
    try:
        return await custom.get_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=namespace,
            name=jobset_name,
        )
    except ApiException as e:
        if e.status != 404:
            raise
        # JobSet may have been deleted by the completion handler after
        # successful results fetch. Don't overwrite a terminal phase.
        await _reconcile_missing_jobset(
            custom,
            namespace=namespace,
            name=name,
            jobset_name=jobset_name,
            current_phase=current_phase,
            sb=sb,
        )
        await close_progress_client(key)
        return None


def _handle_kueue_suspension(
    *,
    jobset: dict[str, Any],
    current_phase: Phase,
    sb: StatusBuilder,
) -> bool:
    """Detect Kueue-managed gang-scheduling suspension.

    Returns True if the JobSet is suspended and the caller should return early.
    """
    jobset_labels = jobset.get("metadata", {}).get("labels", {})
    is_kueue_managed = "kueue.x-k8s.io/queue-name" in jobset_labels
    jobset_suspended = jobset.get("spec", {}).get("suspend", False)

    if (
        is_kueue_managed
        and jobset_suspended
        and current_phase in (Phase.PENDING, Phase.QUEUED)
    ):
        sb.set_phase(Phase.QUEUED)
        sb.finalize()
        return True
    return False


async def _run_worker_and_progress_phase(
    api: ApiClient,
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    patch: kopf.Patch,
    jobset_status: dict[str, Any],
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    current_phase: Phase,
    key: str,
    sb: StatusBuilder,
) -> None:
    """Worker aggregation, pod-restart scan, salvage, and progress polling."""
    workers_ready, workers_succeeded, total_workers = _update_worker_counts(
        status=status, jobset_status=jobset_status, sb=sb
    )

    # Phase transitions based on worker readiness
    if current_phase in (Phase.PENDING, Phase.QUEUED) and (
        workers_ready > 0 or workers_succeeded > 0
    ):
        sb.set_phase(Phase.INITIALIZING)

    # Check for pod restarts (CrashLoopBackOff detection)
    await _check_pod_restarts(api, body, namespace, jobset_name, key=key)

    if await _maybe_recover_terminated_controller(
        api, body, namespace, jobset_name, job_id, status=status, sb=sb, key=key
    ):
        await close_progress_client(key)
        return

    effective_phase = sb.get_phase() or current_phase
    if _should_poll_progress(
        effective_phase, workers_succeeded, total_workers
    ) and await _poll_controller_progress(
        body=body,
        status=status,
        patch=patch,
        namespace=namespace,
        name=name,
        jobset_name=jobset_name,
        job_id=job_id,
        key=key,
        effective_phase=effective_phase,
        sb=sb,
    ):
        return

    sb.finalize()


async def _monitor_tick(
    api: ApiClient,
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
    patch: kopf.Patch,
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    current_phase: Phase,
    key: str,
    sb: StatusBuilder,
) -> None:
    """Execute a single monitor tick against the shared ApiClient."""
    custom = client.CustomObjectsApi(api)

    if await _check_job_timeout(
        custom,
        body=body,
        status=status,
        spec=spec,
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        key=key,
        sb=sb,
    ):
        return

    jobset = await _fetch_jobset_or_reconcile(
        custom,
        namespace=namespace,
        name=name,
        jobset_name=jobset_name,
        current_phase=current_phase,
        key=key,
        sb=sb,
    )
    if jobset is None:
        return

    jobset_status = jobset.get("status", {})

    if _handle_kueue_suspension(jobset=jobset, current_phase=current_phase, sb=sb):
        return

    if await _handle_jobset_terminal_condition(
        body=body,
        status=status,
        jobset_status=jobset_status,
        namespace=namespace,
        name=name,
        jobset_name=jobset_name,
        job_id=job_id,
        key=key,
        sb=sb,
    ):
        return

    await _run_worker_and_progress_phase(
        api,
        body=body,
        status=status,
        patch=patch,
        jobset_status=jobset_status,
        namespace=namespace,
        name=name,
        jobset_name=jobset_name,
        job_id=job_id,
        current_phase=current_phase,
        key=key,
        sb=sb,
    )


async def monitor_progress(
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Monitor job progress and update status."""
    current_phase: Phase = status.get("phase", Phase.PENDING)

    # Stop monitoring terminal jobs
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        return

    jobset_name = status.get("jobSetName")
    job_id = status.get("jobId")
    if not jobset_name or not job_id:
        return

    sb = StatusBuilder(patch, status)
    key = job_key(namespace, job_id)

    # Short-circuit if on_delete has signaled cancellation for this job.
    # Without this, a delete has to wait for the entire monitor tick
    # (including handle_completion's fetch backoff) to complete before
    # kopf's per-object serialization lets the delete handler run.
    if is_cancellation_requested(key):
        logger.debug(
            f"Cancellation requested for {namespace}/{name}, skipping monitor tick"
        )
        return

    try:
        async with k8s_client() as api:
            await _monitor_tick(
                api,
                body=body,
                status=status,
                spec=spec,
                patch=patch,
                namespace=namespace,
                name=name,
                jobset_name=jobset_name,
                job_id=job_id,
                current_phase=current_phase,
                key=key,
                sb=sb,
            )
    except (
        ApiException,
        aiohttp.ClientError,
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    ) as e:
        logger.warning(f"Transient error monitoring {namespace}/{name}: {e}")
        sb.finalize()
    except Exception:
        logger.exception(f"Unexpected error monitoring {namespace}/{name}")
        sb.finalize()
        raise


async def _check_pod_restarts(
    api: ApiClient,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    *,
    key: str,
) -> None:
    """Check for excessive pod restarts and emit warning events (deduplicated)."""
    try:
        pod_list = await client.CoreV1Api(api).list_namespaced_pod(
            namespace=namespace,
            label_selector=f"jobset.sigs.k8s.io/jobset-name={jobset_name}",
        )
        pods = pod_list.items
        warned = _warned_pod_restarts.setdefault(key, set())
        for pod in pods:
            pod_name = (pod.metadata.name if pod.metadata else "") or ""
            container_statuses = (
                (pod.status.container_statuses or []) if pod.status else []
            )
            init_statuses = (
                (pod.status.init_container_statuses or []) if pod.status else []
            )
            all_statuses = list(container_statuses) + list(init_statuses)
            for cs in all_statuses:
                restart_count = cs.restart_count or 0
                if restart_count < OperatorEnvironment.POD_RESTART_THRESHOLD:
                    continue
                dedup_key = (pod_name, restart_count)
                if dedup_key in warned:
                    continue
                warned.add(dedup_key)
                reason = "Unknown"
                if cs.last_state and cs.last_state.terminated:
                    reason = cs.last_state.terminated.reason or "Unknown"
                if cs.state and cs.state.waiting:
                    reason = cs.state.waiting.reason or reason
                events.pod_restarts(body, pod_name, restart_count, reason)
    except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.warning(f"Failed to check pod restarts: {e}")
    except Exception as e:  # noqa: BLE001 - best-effort event emission; restart-scan failure must not abort the monitor tick
        logger.warning(f"Failed to check pod restarts: {e}")


def _container_status_by_name(statuses: list[Any], name: str) -> Any | None:
    """Return the first container status matching the given name."""
    for status in statuses:
        if getattr(status, "name", None) == name:
            return status
    return None


async def _get_controller_pod(
    api: ApiClient, namespace: str, jobset_name: str
) -> Any | None:
    """List and return the first controller pod, or None on failure/absence."""
    try:
        pod_list = await client.CoreV1Api(api).list_namespaced_pod(
            namespace=namespace,
            label_selector=(
                f"{JobSetLabels.JOBSET_NAME}={jobset_name},"
                f"{JobSetLabels.REPLICATED_JOB_NAME}=controller"
            ),
        )
        pods = pod_list.items
    except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.warning(f"Failed to inspect controller pod for salvage: {e}")
        return None
    except Exception as e:  # noqa: BLE001 - salvage path must not raise; skipping recovery is preferred over aborting the monitor tick
        logger.warning(f"Failed to inspect controller pod for salvage: {e}")
        return None

    return pods[0] if pods else None


def _get_terminated_controller_info(pod: Any) -> tuple[int, str] | None:
    """Inspect controller pod and return (exit_code, reason) if terminated non-zero.

    Returns None if the controller container is not terminated, terminated
    successfully, or the sidecar is missing.
    """
    statuses = (pod.status.container_statuses or []) if pod.status else []
    controller_status = _container_status_by_name(statuses, Containers.CONTROL_PLANE)
    sidecar_status = _container_status_by_name(statuses, Containers.RESULTS_SIDECAR)
    if controller_status is None or sidecar_status is None:
        return None

    terminated = (
        controller_status.state.terminated
        if controller_status.state and controller_status.state.terminated
        else None
    )
    if not terminated:
        return None

    exit_code = int(terminated.exit_code or 0)
    reason = terminated.reason or "Error"
    if exit_code == 0:
        return None
    return exit_code, reason


async def _recover_from_partial_checkpoints(
    *,
    body: dict[str, Any],
    result: Any,
    namespace: str,
    jobset_name: str,
    job_id: str,
    sb: StatusBuilder,
    custom: CustomObjectsApi,
) -> None:
    """Salvage partial checkpoint files and mark the CR FAILED."""
    dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id
    checkpoint_metrics = _parse_metrics_from_files(
        result.checkpoints, namespace, job_id
    )
    if checkpoint_metrics:
        sb.set_results(checkpoint_metrics)

        summary = MetricsSummary.from_metrics(checkpoint_metrics)
        summary_dict = summary.to_status_dict()
        if summary_dict:
            sb.set_summary(summary_dict)

    error = (
        f"Controller container terminated before final export; "
        f"recovered {len(result.checkpoints)} partial checkpoint file(s)"
    )
    sb.set_phase(Phase.FAILED).set_error(error).set_completion_time()
    sb.set_results_path(str(dest_dir))
    sb.conditions.set_true(
        ConditionType.RESULTS_AVAILABLE,
        "PartialCheckpointRecovered",
        f"Recovered {len(result.checkpoints)} partial checkpoint file(s)",
    )
    sb.finalize()
    events.results_stored(body, str(dest_dir), len(result.checkpoints))
    events.failed(body, job_id, error)

    await _delete_jobset_silently(
        custom, namespace, jobset_name, context="partial checkpoint recovery"
    )


async def _fail_unrecoverable_controller(
    *,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    reason: str,
    sb: StatusBuilder,
    custom: CustomObjectsApi,
) -> None:
    """Mark CR FAILED and delete the JobSet when no results can be recovered."""
    error = f"Controller container terminated before results were recoverable: {reason}"
    sb.set_phase(Phase.FAILED).set_error(error).set_completion_time()
    sb.conditions.set_false(
        ConditionType.RESULTS_AVAILABLE,
        "ControllerTerminated",
        "Controller terminated before exporting recoverable result files",
    )
    sb.finalize()
    events.failed(body, job_id, error)

    await _delete_jobset_silently(
        custom, namespace, jobset_name, context="unrecoverable controller termination"
    )


async def _maybe_recover_terminated_controller(
    api: ApiClient,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    *,
    status: dict[str, Any],
    sb: StatusBuilder,
    key: str,
) -> bool:
    """Recover results from the sidecar if the controller container terminated.

    A regular sidecar keeps the pod alive long enough for salvage, but that also
    means we cannot rely solely on JobSet terminal conditions. If the main
    controller container exits unexpectedly, attempt to recover exported files
    from the sidecar immediately.
    """
    if key in _shutdown_sent:
        return False

    pod = await _get_controller_pod(api, namespace, jobset_name)
    if pod is None:
        return False

    info = _get_terminated_controller_info(pod)
    if info is None:
        return False
    exit_code, reason = info
    pod_name = (pod.metadata.name if pod.metadata else "") or ""

    logger.warning(
        "Controller container terminated in pod %s (reason=%s, exitCode=%s), "
        "attempting results recovery from sidecar",
        pod_name,
        reason,
        exit_code,
    )

    result = await fetch_results_with_retry(
        controller_dns_name(jobset_name, namespace),
        namespace,
        job_id,
    )
    if result.downloaded:
        _shutdown_sent.add(key)
        await handle_completion(
            body,
            namespace,
            jobset_name,
            job_id,
            status=status,
            sb=sb,
            result=result,
        )
        return True

    custom = client.CustomObjectsApi(api)
    if result.checkpoints:
        await _recover_from_partial_checkpoints(
            body=body,
            result=result,
            namespace=namespace,
            jobset_name=jobset_name,
            job_id=job_id,
            sb=sb,
            custom=custom,
        )
        return True

    await _fail_unrecoverable_controller(
        body=body,
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        reason=reason,
        sb=sb,
        custom=custom,
    )
    return True


async def _fetch_live_metrics(
    progress_client: Any,
    host: str,
    jobset_name: str,
    patch: kopf.Patch,
) -> None:
    """Fetch live metrics from controller and stamp them onto the CR patch."""
    try:
        metrics = await progress_client.get_metrics(host)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.warning(f"Live metrics fetch failed for {jobset_name}: {e}")
        return
    except Exception as e:  # noqa: BLE001 - live metrics are optional; any parse/transport failure downgrades to 'no live metrics this tick'
        logger.warning(f"Live metrics fetch failed for {jobset_name}: {e}")
        return

    if isinstance(metrics, dict) and metrics.get("metrics"):
        patch.status["liveMetrics"] = metrics

        summary = MetricsSummary.from_metrics(metrics)
        summary_dict = summary.to_status_dict()
        if summary_dict:
            patch.status["liveSummary"] = summary_dict


async def _fetch_server_metrics(
    progress_client: Any,
    host: str,
    jobset_name: str,
    patch: kopf.Patch,
) -> None:
    """Fetch server metrics from controller and stamp them onto the CR patch."""
    try:
        server_metrics = await progress_client.get_server_metrics(host)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.debug(
            f"Server metrics unavailable for {jobset_name} "
            f"(endpoint may not be ready yet): {e}"
        )
        return
    except Exception as e:  # noqa: BLE001 - server-metrics endpoint is optional and may return any shape during startup; debug-log and continue
        logger.debug(
            f"Server metrics unavailable for {jobset_name} "
            f"(endpoint may not be ready yet): {e}"
        )
        return

    if isinstance(server_metrics, dict) and server_metrics.get("endpoint_summaries"):
        patch.status["serverMetrics"] = server_metrics


async def _fetch_progress(
    namespace: str,
    jobset_name: str,
    patch: kopf.Patch,
    sb: StatusBuilder,
    progress_client: Any,
    key: str,
    current_phase: Phase = Phase.RUNNING,
    *,
    body: dict[str, Any] | None = None,
) -> bool:
    """Fetch progress and live metrics from controller pod.

    Returns True if the benchmark is complete (all profiling requests done).
    """

    host = controller_dns_name(jobset_name, namespace)

    try:
        progress = await progress_client.get_progress(host)

        if progress.connection_error:
            logger.debug(
                f"Progress API unreachable for {jobset_name}: connection error"
            )
            return False

        phases_data: dict[str, Any] = {}
        for phase, stats in progress.phases.items():
            if phase_progress := _build_phase_progress(stats):
                phases_data[phase] = phase_progress.to_k8s_dict()

        if phases_data:
            patch.status["phases"] = phases_data

        _apply_controller_progress_status(patch, sb, progress, current_phase)

        if progress.error:
            patch.status["error"] = progress.error

        await _fetch_live_metrics(progress_client, host, jobset_name, patch)
        await _fetch_server_metrics(progress_client, host, jobset_name, patch)

        # Return completion status for caller to handle. Skip signaling
        # completion if another path (this or a previous operator run)
        # has already claimed it: in-process set OR durable annotation.
        if progress.is_complete and key not in _shutdown_sent:
            if body is not None and is_completion_claimed(body):
                _shutdown_sent.add(key)
                return False
            return True

    except (
        ApiException,
        aiohttp.ClientError,
        asyncio.TimeoutError,
        OSError,
    ) as e:
        logger.warning(f"Failed to fetch progress for {jobset_name}: {e}")
    except Exception as e:  # noqa: BLE001 - best-effort progress polling; no error here must abort the monitor tick
        logger.warning(f"Failed to fetch progress for {jobset_name}: {e}")

    return False


def _build_phase_progress(stats: CombinedPhaseStats) -> PhaseProgress | None:
    """Build PhaseProgress from CombinedPhaseStats."""
    total = stats.total_expected_requests or 0
    if total == 0 and stats.requests_sent == 0:
        return None

    elapsed = None
    if stats.start_ns is not None and stats.last_update_ns is not None:
        elapsed = round((stats.last_update_ns - stats.start_ns) / 1_000_000_000, 1)

    return PhaseProgress(
        requests_completed=stats.requests_completed,
        requests_sent=stats.requests_sent,
        requests_total=total,
        requests_cancelled=stats.requests_cancelled,
        requests_errors=stats.request_errors,
        requests_in_flight=stats.in_flight_requests,
        requests_per_second=round(stats.requests_per_second or 0, 2),
        requests_progress_percent=round(stats.requests_progress_percent or 0, 1),
        sessions_sent=stats.sent_sessions,
        sessions_completed=stats.completed_sessions,
        sessions_cancelled=stats.cancelled_sessions,
        sessions_in_flight=stats.in_flight_sessions,
        records_success=stats.success_records,
        records_error=stats.error_records,
        records_per_second=round(stats.records_per_second or 0, 2),
        records_progress_percent=round(stats.records_progress_percent or 0, 1),
        sending_complete=stats.is_requests_complete,
        timeout_triggered=stats.timeout_triggered,
        was_cancelled=stats.was_cancelled,
        requests_eta_seconds=round(stats.requests_eta_sec)
        if stats.requests_eta_sec is not None
        else None,
        records_eta_seconds=round(stats.records_eta_sec)
        if stats.records_eta_sec is not None
        else None,
        expected_duration_seconds=round(stats.expected_duration_sec, 1)
        if stats.expected_duration_sec is not None
        else None,
        elapsed_time_seconds=elapsed,
    )

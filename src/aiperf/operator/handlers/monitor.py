# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""monitor_progress timer handler logic for AIPerfJob CRD.

This module contains the business logic only — no kopf decorators.
Decorators live in ``aiperf.operator.main``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiohttp
import kopf
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient, CustomObjectsApi
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.constants import Annotations, Containers, JobSetLabels
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.results_sidecar import write_ready_marker
from aiperf.operator import events
from aiperf.operator.client_cache import (
    _shutdown_sent,
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
    _recover_result_from_disk,
    fetch_results_with_retry,
    handle_completion,
)
from aiperf.operator.models import ControllerFetchResult, MetricsSummary
from aiperf.operator.progress_client import ProgressClient
from aiperf.operator.results_layout import epoch_key_from_body, run_dir
from aiperf.operator.status import (
    ConditionType,
    Phase,
    StatusBuilder,
    parse_timestamp,
)

logger = logging.getLogger(__name__)

FATAL_POD_WAITING_REASONS = frozenset(
    {
        "CreateContainerConfigError",
        "CreateContainerError",
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
    }
)
KEY_RESULT_FILES = frozenset(
    {"profile_export_aiperf.json", "profile_export_aiperf.csv"}
)
CHECKPOINTS_PREFIX = "checkpoints/"


@dataclass(frozen=True, slots=True)
class FatalPodWaitingReason:
    """Fatal container waiting state observed on a JobSet pod."""

    pod_name: str
    container_name: str
    reason: str
    message: str


def _get_elapsed_seconds(status: dict[str, Any]) -> float | None:
    """Calculate elapsed seconds since startTime, or None if unavailable."""
    start_time = status.get("startTime")
    if not start_time:
        return None
    try:
        start_dt = parse_timestamp(start_time)
        return (datetime.now(UTC) - start_dt).total_seconds()
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


def _get_fatal_pod_waiting_reason(pods: list[Any]) -> FatalPodWaitingReason | None:
    """Return the first fatal container waiting reason from a pod list."""
    for pod in pods:
        pod_name = getattr(getattr(pod, "metadata", None), "name", "") or "unknown"
        statuses = (
            getattr(getattr(pod, "status", None), "container_statuses", None) or []
        )
        for container_status in statuses:
            state = getattr(container_status, "state", None)
            waiting = getattr(state, "waiting", None) if state is not None else None
            reason = getattr(waiting, "reason", None) if waiting is not None else None
            if reason not in FATAL_POD_WAITING_REASONS:
                continue
            message = getattr(waiting, "message", "") or ""
            if not message:
                message = getattr(container_status, "image", "") or ""
            return FatalPodWaitingReason(
                pod_name=pod_name,
                container_name=getattr(container_status, "name", "") or "unknown",
                reason=reason,
                message=message,
            )
    return None


def _fatal_pod_waiting_message(
    name: str, jobset_name: str, waiting: FatalPodWaitingReason
) -> str:
    """Format an actionable error for fatal pod startup waiting states."""
    detail = f": {waiting.message}" if waiting.message else ""
    return (
        f"AIPerfJob {name} failed because JobSet {jobset_name} pod "
        f"{waiting.pod_name} container {waiting.container_name} is waiting with "
        f"fatal reason {waiting.reason}{detail}"
    )


async def _delete_jobset_or_retry(
    custom: CustomObjectsApi,
    namespace: str,
    jobset_name: str,
    *,
    context: str,
) -> None:
    """Delete a JobSet; non-404 failures raise for a retryable monitor tick."""
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
        if e.status == 404:
            return
        msg = f"Failed to delete JobSet {namespace}/{jobset_name} after {context}: {e}"
        logger.warning(msg)
        raise kopf.TemporaryError(msg, delay=15) from e


async def _reconcile_missing_jobset(
    custom: CustomObjectsApi,
    *,
    body: dict[str, Any],
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

    # Completion-claim annotation is the authoritative cross-tick signal that
    # the success branch owns the CR. ``try_claim_completion`` stamps it via
    # JSON-patch BEFORE ``handle_completion`` runs, and only the success path
    # in ``_maybe_delete_jobset_after_success`` deletes the JobSet — so a
    # claimed body with a 404'd JobSet is positive evidence of completion,
    # not failure. Without this gate, a kopf timer firing on a stale body
    # snapshot (phase still pre-terminal because the watch event for our
    # own patch hasn't propagated to kopf's local cache yet) would stamp
    # ``Phase.FAILED`` over a CR that already wrote ``Phase.COMPLETED``.
    if is_completion_claimed(body):
        logger.debug(
            f"JobSet {jobset_name} not found but completion-claim annotation "
            f"is set on {namespace}/{name} - success handler owns this CR, "
            f"skipping FAILED stamp"
        )
        return True

    # Belt-and-suspenders: if the claim isn't on our cached body either
    # (e.g. claim never set because monitor took a different branch),
    # re-read the CR after a short delay to give the success handler's
    # phase patch a chance to land.
    await asyncio.sleep(2)

    try:
        fresh = await custom.get_namespaced_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            namespace=namespace,
            name=name,
        )
    except Exception:
        # Fresh-read failure is NOT evidence the benchmark failed; keep the
        # CR in its current phase and let the next monitor tick retry.
        # Falling through to ``set_phase(FAILED)`` here is the original
        # JobSet-not-found phase-stomp bug — an apiserver hiccup must not
        # overwrite a (possibly already-Completed) CR.
        logger.exception(
            f"Stale-read recovery failed while reconciling "
            f"{namespace}/{name} after JobSet {jobset_name} not found; "
            f"deferring to next monitor tick"
        )
        return True

    fresh_phase = fresh.get("status", {}).get("phase", "")
    if fresh_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        logger.debug(
            f"JobSet {jobset_name} not found but fresh phase is "
            f"{fresh_phase} - skipping"
        )
        return True

    # Re-check the claim annotation on the fresh body too: between the
    # caller's body snapshot and now, ``try_claim_completion`` may have
    # stamped the claim from a peer operator pod (HA) or from a concurrent
    # monitor tick that observed ``progress.is_complete`` first.
    fresh_annotations = fresh.get("metadata", {}).get("annotations") or {}
    if fresh_annotations.get(Annotations.COMPLETION_CLAIMED):
        logger.debug(
            f"JobSet {jobset_name} not found and fresh CR carries "
            f"completion-claim annotation - skipping FAILED stamp"
        )
        return True

    sb.set_phase(Phase.FAILED).set_error("JobSet not found").set_completion_time()
    sb.finalize()
    return False


async def _fail_on_fatal_pod_waiting_reason(
    api: ApiClient,
    *,
    body: dict[str, Any],
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    key: str,
    sb: StatusBuilder,
) -> bool:
    """Fail active jobs when any JobSet pod is stuck in a fatal waiting reason."""
    try:
        pod_list = await client.CoreV1Api(api).list_namespaced_pod(
            namespace=namespace,
            label_selector=f"{JobSetLabels.JOBSET_NAME}={jobset_name}",
        )
    except (TimeoutError, ApiException, aiohttp.ClientError, OSError, TypeError) as e:
        logger.warning(
            "Failed to inspect pods for fatal startup states on %s/%s: %s",
            namespace,
            name,
            e,
        )
        return False

    waiting = _get_fatal_pod_waiting_reason(pod_list.items)
    if waiting is None:
        return False

    error = _fatal_pod_waiting_message(name, jobset_name, waiting)
    await _delete_jobset_or_retry(
        client.CustomObjectsApi(api),
        namespace,
        jobset_name,
        context=f"fatal pod startup reason {waiting.reason}",
    )
    sb.set_phase(Phase.FAILED).set_error(error).set_completion_time()
    sb.conditions.set_false(
        ConditionType.WORKERS_READY,
        "PodStartupFailed",
        error,
    )
    sb.finalize()
    events.failed(body, job_id, error)
    await close_progress_client(key)
    return True


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

    # Do not fail a run that has already succeeded but is still draining.
    # The completion-claim annotation is the authoritative cross-tick signal
    # that the success branch owns the CR (mirrors ``_reconcile_missing_jobset``);
    # ``currentPhase == "processing"`` means the controller reported
    # ``is_complete`` and the operator is fetching/aggregating results.
    # Either signal means a subsequent ``_reconcile_and_handle_jobset`` tick
    # will claim completion and harvest results — stamping FAILED here would
    # discard a succeeded run and delete its JobSet mid-drain.
    if is_completion_claimed(body) or status.get("currentPhase") == "processing":
        logger.debug(
            "Job timeout reached for %s but run is draining/claimed "
            "(currentPhase=%s, claimed=%s); deferring to completion handler",
            jobset_name,
            status.get("currentPhase"),
            is_completion_claimed(body),
        )
        return False

    if jobset_name:
        await _delete_jobset_or_retry(custom, namespace, jobset_name, context="timeout")
    sb.set_phase(Phase.FAILED).set_error(
        f"Job timed out after {elapsed:.0f}s (limit: {timeout_sec:.0f}s)"
    )
    sb.set_completion_time()
    sb.finalize()
    events.job_timeout(body, job_id, elapsed)
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
        sb.set_completion_time()
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
        sb.set_completion_time()
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


def _split_downloaded_results(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split downloaded result paths into final exports and checkpoint files."""
    final_files: list[str] = []
    checkpoint_files: list[str] = []
    for path in paths:
        if path.startswith(CHECKPOINTS_PREFIX):
            checkpoint_files.append(path)
        else:
            final_files.append(path)
    return final_files, checkpoint_files


async def _maybe_recover_exported_results_from_sidecar(
    *,
    body: dict[str, Any],
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    status: dict[str, Any],
    sb: StatusBuilder,
    key: str,
) -> bool:
    """Complete a job from final exports served by the results sidecar.

    This is the success-path counterpart to terminated-controller salvage. If
    controller API traffic is blackholed, the control-plane container keeps
    running and ``_maybe_recover_terminated_controller`` never fires. The
    sidecar is independent of that API port and only exposes top-level result
    files after the ready marker exists, so key exports there are sufficient
    evidence that the benchmark completed and can be finalized.
    """
    if key in _shutdown_sent:
        return False

    host = controller_dns_name(jobset_name, namespace)
    epoch = epoch_key_from_body(body)
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    try:
        async with ProgressClient(port=K8sEnvironment.PORTS.RESULTS_SIDECAR) as sidecar:
            downloaded = await sidecar.download_all_results(host, dest_dir)
    except (TimeoutError, aiohttp.ClientError, OSError) as e:
        logger.debug(
            "sidecar export recovery for %s/%s unavailable: %s", namespace, name, e
        )
        return False
    except Exception as e:  # noqa: BLE001 - sidecar recovery is best-effort; normal monitor retry continues
        logger.debug(
            "sidecar export recovery for %s/%s unavailable: %s", namespace, name, e
        )
        return False

    final_files, checkpoint_files = _split_downloaded_results(downloaded)
    if not (KEY_RESULT_FILES & set(final_files)):
        return False

    if is_cancellation_requested(key):
        logger.debug(
            "Cancellation requested for %s/%s during sidecar export recovery; "
            "skipping completion side effects",
            namespace,
            name,
        )
        return True

    if not await try_claim_completion(namespace, name, body):
        return False

    await handle_completion(
        body,
        namespace,
        jobset_name,
        job_id,
        status=status,
        sb=sb,
        result=ControllerFetchResult(
            metrics=None,
            downloaded=final_files,
            checkpoints=checkpoint_files,
            error="",
        ),
    )
    return True


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
    # Shutdown controller after results are fetched. handle_completion has
    # already deleted the JobSet and staged phase=Completed into the patch;
    # the shutdown signal is best-effort and must not re-raise, or the
    # terminal status patch would be discarded for this tick.
    try:
        await progress_client.send_shutdown(host)
    except (TimeoutError, aiohttp.ClientError, OSError) as e:
        logger.debug(
            "send_shutdown after completion for %s/%s failed "
            "(expected if controller pod already gone): %s",
            namespace,
            name,
            e,
        )
    except Exception as e:  # noqa: BLE001 - completion already finalized; shutdown signal is best-effort and must not re-raise
        logger.debug(
            "send_shutdown after completion for %s/%s failed: %s",
            namespace,
            name,
            e,
        )
    await close_progress_client(key)
    return True


async def _fetch_jobset_or_reconcile(
    custom: CustomObjectsApi,
    *,
    body: dict[str, Any],
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
            body=body,
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


def _set_initializing_when_workers_start(
    current_phase: Phase,
    workers_ready: int,
    workers_succeeded: int,
    sb: StatusBuilder,
) -> None:
    if current_phase in (Phase.PENDING, Phase.QUEUED) and (
        workers_ready > 0 or workers_succeeded > 0
    ):
        sb.set_phase(Phase.INITIALIZING)


async def _poll_progress_or_recover_sidecar(
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
    if await _poll_controller_progress(
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
        return True
    return (
        effective_phase == Phase.RUNNING
        and await _maybe_recover_exported_results_from_sidecar(
            body=body,
            namespace=namespace,
            name=name,
            jobset_name=jobset_name,
            job_id=job_id,
            status=status,
            sb=sb,
            key=key,
        )
    )


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

    _set_initializing_when_workers_start(
        current_phase, workers_ready, workers_succeeded, sb
    )

    if await _fail_on_fatal_pod_waiting_reason(
        api,
        body=body,
        namespace=namespace,
        name=name,
        jobset_name=jobset_name,
        job_id=job_id,
        key=key,
        sb=sb,
    ):
        return

    if await _maybe_recover_terminated_controller(
        api,
        body,
        namespace,
        jobset_name,
        job_id,
        status=status,
        sb=sb,
        key=key,
        name=name,
    ):
        await close_progress_client(key)
        return

    effective_phase = sb.get_phase() or current_phase
    if _should_poll_progress(
        effective_phase, workers_succeeded, total_workers
    ) and await _poll_progress_or_recover_sidecar(
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


async def _jobset_has_terminal_condition(
    api: ApiClient,
    namespace: str,
    jobset_name: str,
) -> bool:
    """Return True if the JobSet is in a terminal state or has been deleted.

    A 404 on JobSet lookup means the prior completion handler reached
    ``_maybe_delete_jobset_after_success`` (which only fires on a successful
    fetch+store). Either way — Completed condition, Failed condition, or
    deleted entirely — the benchmark is done and orphan-claim recovery
    is safe to run.
    """
    try:
        custom = CustomObjectsApi(api)
        jobset = await custom.get_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=namespace,
            name=jobset_name,
        )
    except ApiException as e:
        return e.status == 404
    except Exception:  # noqa: BLE001 - gate is best-effort; transient errors fall through to "no evidence yet"
        return False
    for cond in (jobset.get("status") or {}).get("conditions", []) or []:
        if cond.get("status") == "True" and cond.get("type") in (
            "Completed",
            "Failed",
        ):
            return True
    return False


async def _benchmark_appears_complete(
    *,
    api: ApiClient,
    namespace: str,
    jobset_name: str,
    key: str,
) -> bool:
    """Return True only when there is evidence the benchmark is actually done.

    Signal: the control-plane container in the controller pod is terminated. (The
    former first signal -- polling the controller's ``/api/progress`` for
    ``is_complete`` -- is retired along with the rest of the mesh poll; completion
    now arrives on the ``benchmark-complete`` annotation the run pushes, so this
    orphan-claim gate only needs the pod-terminated fallback for the case where the
    annotation handler crashed after side effects but before flushing status.)

    The signal is quick, read-only, and side-effect-free; if it does not fire we
    return False so callers skip eager completion work (e.g.
    ``_recover_orphaned_completion_claim``) while the benchmark is still in flight.
    A return value of False therefore means "no evidence yet, try again next tick" --
    never "definitely still running". ``key`` is retained for signature stability.
    """
    pod = await _get_controller_pod(api, namespace, jobset_name)
    if pod is None:
        # No controller pod. Two scenarios put us here:
        #
        # 1. The benchmark finished, _maybe_delete_jobset_after_success
        #    deleted the JobSet (success-only path), and pods went with it.
        #    Recovery should fire — the previous handler must have crashed
        #    after side effects but before sb.finalize() flushed.
        # 2. The JobSet still exists but its pods reached terminal state
        #    and were reaped (TTL or kubelet GC). The JobSet's own
        #    Completed/Failed condition is then authoritative.
        #
        # Both are detectable by looking at the JobSet itself.
        return await _jobset_has_terminal_condition(api, namespace, jobset_name)
    statuses = (pod.status.container_statuses or []) if pod.status else []
    # Native cellular runs name the aggregate container ``controller``
    # (Containers.CELL_CONTROLLER); the legacy mesh pod named it
    # ``control-plane``. The two never coexist, so accept whichever is present —
    # otherwise the orphan-recovery gate silently never fires for cellular runs.
    controller_status = _container_status_by_name(
        statuses, Containers.CELL_CONTROLLER
    ) or _container_status_by_name(statuses, Containers.CONTROL_PLANE)
    if controller_status is None:
        return False
    terminated = (
        controller_status.state.terminated
        if controller_status.state and controller_status.state.terminated
        else None
    )
    return terminated is not None


async def _recover_orphaned_completion_claim(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    key: str,
    sb: StatusBuilder,
) -> None:
    """Re-invoke ``handle_completion`` for a CR with a stale completion claim.

    Side effects:
        - Runs ``handle_completion`` (results fetch, status patch, JobSet delete).
        - Best-effort shutdown signal to the controller pod (may already be gone).
        - Closes the cached ProgressClient on exit.

    Why this exists:
        ``try_claim_completion`` sets the ``aiperf.nvidia.com/completion-claimed``
        annotation *before* ``handle_completion`` runs. If the operator pod
        crashes in that window, the new process starts with an empty
        ``_shutdown_sent`` set, but the annotation on the CR persists — so every
        subsequent claim attempt short-circuits. Without this recovery, the CR
        stays ``phase=Running`` with the claim annotation forever.

    Callers MUST gate this behind ``_benchmark_appears_complete`` — firing
    it while the benchmark is still running drives ``handle_completion``
    into a retry-stagnation loop (no key export files yet) that ends in
    ``phase=Failed`` even though the benchmark would have finished
    successfully. See ``tests/kubernetes/chaos/test_chaos_operator_
    resilience.py::test_c5_orphaned_claim_recovers``.
    """
    logger.warning(
        "Recovering orphaned completion-claim for %s/%s (phase=%s): "
        "previous handler did not reach a terminal phase; re-running "
        "handle_completion to converge",
        namespace,
        name,
        status.get("phase"),
    )
    try:
        await handle_completion(
            body, namespace, jobset_name, job_id, status=status, sb=sb
        )
        host = controller_dns_name(jobset_name, namespace)
        progress_client = await get_or_create_progress_client(key)
        try:
            await progress_client.send_shutdown(host)
        except (TimeoutError, aiohttp.ClientError, OSError) as e:
            logger.debug(
                "send_shutdown during orphaned-claim recovery for %s/%s failed "
                "(expected if controller pod already gone): %s",
                namespace,
                name,
                e,
            )
        except Exception as e:  # noqa: BLE001 - recovery path must not raise; shutdown signal is best-effort
            logger.debug(
                "send_shutdown during orphaned-claim recovery for %s/%s failed: %s",
                namespace,
                name,
                e,
            )
    finally:
        await close_progress_client(key)


async def _maybe_recover_orphan_claim(
    api: ApiClient,
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    current_phase: Phase,
    key: str,
    sb: StatusBuilder,
) -> bool:
    """Run orphan-claim recovery when claim+non-terminal+benchmark-done all hold.

    Returns True if recovery ran (caller should return early), False otherwise.

    The ``_benchmark_appears_complete`` gate is load-bearing: without it a
    claim stamped while the benchmark is still running drives
    ``handle_completion`` into a retry-stagnation loop that marks the CR
    Failed even though the benchmark itself is still in flight. Only run
    recovery once we have positive evidence that the benchmark is done.
    See tests/kubernetes/chaos/test_chaos_operator_resilience.py::
    test_c5_orphaned_claim_recovers.
    """
    if not is_completion_claimed(body) or current_phase in (
        Phase.COMPLETED,
        Phase.FAILED,
        Phase.CANCELLED,
    ):
        return False

    if not await _benchmark_appears_complete(
        api=api,
        namespace=namespace,
        jobset_name=jobset_name,
        key=key,
    ):
        logger.debug(
            "Orphan-claim recovery deferred for %s/%s: benchmark not yet "
            "complete; continuing normal monitor tick",
            namespace,
            name,
        )
        return False

    await _recover_orphaned_completion_claim(
        body=body,
        status=status,
        namespace=namespace,
        name=name,
        jobset_name=jobset_name,
        job_id=job_id,
        key=key,
        sb=sb,
    )
    return True


async def _reconcile_and_handle_jobset(
    api: ApiClient,
    custom: CustomObjectsApi,
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    patch: kopf.Patch,
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    current_phase: Phase,
    key: str,
    sb: StatusBuilder,
) -> None:
    """Fetch the JobSet and drive the per-phase reconciliation branches.

    Split out of ``_monitor_tick`` to keep the top-level tick small. Handles
    the "JobSet not found / kueue-suspended / terminal / running" quartet and
    delegates worker + progress aggregation to ``_run_worker_and_progress_phase``.
    """
    jobset = await _fetch_jobset_or_reconcile(
        custom,
        body=body,
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

    if await _maybe_recover_orphan_claim(
        api,
        body=body,
        status=status,
        namespace=namespace,
        name=name,
        jobset_name=jobset_name,
        job_id=job_id,
        current_phase=current_phase,
        key=key,
        sb=sb,
    ):
        return

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

    await _reconcile_and_handle_jobset(
        api,
        custom,
        body=body,
        status=status,
        patch=patch,
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
        # observedGeneration is a success-path-only stamp: a tick that
        # terminally FAILED/CANCELLED the job must not signal spec acceptance.
        # sb.get_phase() returns the phase the failure helpers just wrote (None
        # on a non-terminal tick, which legitimately acknowledges the spec).
        # A mid-completion cancellation short-circuit ALSO leaves get_phase()
        # None (handle_completion returns before copying its staged phase into
        # sb), and is indistinguishable from a non-terminal tick by phase
        # alone -- re-check the sticky cancellation flag to exclude it.
        if sb.get_phase() not in (
            str(Phase.FAILED),
            str(Phase.CANCELLED),
        ) and not is_cancellation_requested(key):
            generation = body.get("metadata", {}).get("generation")
            if generation is not None:
                sb.set_observed_generation(int(generation))
    except (ApiException, aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        logger.warning(f"Transient error monitoring {namespace}/{name}: {e}")
        sb.finalize()
    except Exception:
        logger.exception(f"Unexpected error monitoring {namespace}/{name}")
        sb.finalize()
        raise


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
    except (TimeoutError, ApiException, aiohttp.ClientError, OSError) as e:
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
    if not terminated and int(getattr(controller_status, "restart_count", 0) or 0) > 0:
        last_state = getattr(controller_status, "last_state", None)
        terminated = (
            last_state.terminated if last_state and last_state.terminated else None
        )
    if not terminated:
        return None

    exit_code = int(terminated.exit_code or 0)
    reason = terminated.reason or "Error"
    if exit_code == 0:
        return None
    return exit_code, reason


def _apply_live_status_partial_results(
    status: dict[str, Any],
    sb: StatusBuilder,
) -> bool:
    """Copy CR live metrics into terminal partial result fields."""
    recovered = False
    live_metrics = status.get("liveMetrics")
    if (
        isinstance(live_metrics, dict)
        and isinstance(live_metrics.get("metrics"), dict)
        and live_metrics["metrics"]
    ):
        sb.set_results(live_metrics)
        recovered = True

    live_summary = status.get("liveSummary")
    if isinstance(live_summary, dict) and live_summary:
        sb.set_summary(live_summary)
        recovered = True
    elif recovered and isinstance(live_metrics, dict):
        summary = MetricsSummary.from_metrics(live_metrics)
        summary_dict = summary.to_status_dict()
        if summary_dict:
            sb.set_summary(summary_dict)
    return recovered


async def _recover_from_live_status(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    reason: str,
    sb: StatusBuilder,
    custom: CustomObjectsApi,
) -> bool:
    """Salvage CR live metrics as partial results and mark the CR FAILED."""
    if not _apply_live_status_partial_results(status, sb):
        return False

    await _delete_jobset_or_retry(
        custom, namespace, jobset_name, context="partial live metrics recovery"
    )
    error = (
        "Controller container terminated before final export; "
        f"recovered partial live metrics from CR status: {reason}"
    )
    sb.set_phase(Phase.FAILED).set_error(error).set_completion_time()
    sb.conditions.set_true(
        ConditionType.RESULTS_AVAILABLE,
        "PartialLiveMetricsRecovered",
        "Recovered partial live metrics from CR status",
    )
    sb.finalize()
    events.failed(body, job_id, error)
    return True


async def _recover_from_partial_checkpoints(
    *,
    body: dict[str, Any],
    result: Any,
    namespace: str,
    jobset_name: str,
    job_id: str,
    sb: StatusBuilder,
    custom: CustomObjectsApi,
    status: dict[str, Any] | None = None,
) -> None:
    """Salvage partial checkpoint files and mark the CR FAILED."""
    epoch = epoch_key_from_body(body)
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    checkpoint_metrics = _parse_metrics_from_files(
        result.checkpoints, namespace, job_id, epoch=epoch
    )
    if checkpoint_metrics:
        sb.set_results(checkpoint_metrics)

        summary = MetricsSummary.from_metrics(checkpoint_metrics)
        summary_dict = summary.to_status_dict()
        if summary_dict:
            sb.set_summary(summary_dict)
    elif status is not None:
        _apply_live_status_partial_results(status, sb)

    error = (
        f"Controller container terminated before final export; "
        f"recovered {len(result.checkpoints)} partial checkpoint file(s)"
    )
    await _delete_jobset_or_retry(
        custom, namespace, jobset_name, context="partial checkpoint recovery"
    )

    sb.set_phase(Phase.FAILED).set_error(error).set_completion_time()
    # Write the readiness marker so the operator results-server actually serves
    # the salvaged checkpoint artifacts; without it ``_require_run_ready`` 404s
    # the bundle / profile-export routes forever even though resultsPath points
    # at on-disk files. ``was_cancelled=False`` — this is a salvaged failure,
    # not a user cancellation.
    write_ready_marker(dest_dir, was_cancelled=False)
    sb.set_results_path(str(dest_dir))
    # Stamp runEpoch so the operator-API metrics fallback in
    # ``K8sChildJobExecutor._fetch_summary_from_operator`` can resolve the
    # canonical ``/api/v1/results/<ns>/<job>/runs/<epoch>/...`` URL.
    # Without this, sweep children that hit partial-checkpoint recovery
    # silently drop out of the parent aggregate even though the artifacts
    # are on disk and the operator's results-server would serve them.
    if epoch.isdigit():
        sb.set_run_epoch(int(epoch))
    sb.conditions.set_true(
        ConditionType.RESULTS_AVAILABLE,
        "PartialCheckpointRecovered",
        f"Recovered {len(result.checkpoints)} partial checkpoint file(s)",
    )
    sb.finalize()
    events.results_stored(body, str(dest_dir), len(result.checkpoints))
    events.failed(body, job_id, error)


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
    await _delete_jobset_or_retry(
        custom, namespace, jobset_name, context="unrecoverable controller termination"
    )

    error = f"Controller container terminated before results were recoverable: {reason}"
    sb.set_phase(Phase.FAILED).set_error(error).set_completion_time()
    sb.conditions.set_false(
        ConditionType.RESULTS_AVAILABLE,
        "ControllerTerminated",
        "Controller terminated before exporting recoverable result files",
    )
    sb.finalize()
    events.failed(body, job_id, error)


async def _salvage_terminated_controller_results(
    api: ApiClient,
    *,
    body: dict[str, Any],
    result: ControllerFetchResult,
    status: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    reason: str,
    sb: StatusBuilder,
) -> None:
    """Dispatch the claimed salvage branches for a terminated controller.

    Tries, in order: partial checkpoint files, live CR status metrics, and
    finally the unrecoverable-failure path. Exactly one branch runs; every
    branch stamps a terminal FAILED phase and deletes the JobSet. Callers
    MUST hold the durable completion claim before invoking.
    """
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
            status=status,
        )
        return

    if await _recover_from_live_status(
        body=body,
        status=status,
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        reason=reason,
        sb=sb,
        custom=custom,
    ):
        return

    await _fail_unrecoverable_controller(
        body=body,
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        reason=reason,
        sb=sb,
        custom=custom,
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
    name: str,
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
        body=body,
    )
    if (body.get("metadata") or {}).get("creationTimestamp"):
        result = _recover_result_from_disk(
            body=body,
            namespace=namespace,
            job_id=job_id,
            result=result,
        )
    if is_cancellation_requested(key):
        logger.debug(
            "Cancellation requested for %s/%s during terminated-controller salvage; "
            "skipping recovery side effects",
            namespace,
            name,
        )
        return True
    if result.downloaded:
        # Go through the durable claim — the in-process _shutdown_sent set
        # above is a fast path, but a peer operator pod (HA) has its own
        # set, so only the CR annotation patch is authoritative.
        if not await try_claim_completion(namespace, name, body):
            await close_progress_client(key)
            return False
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

    # Gate the partial-checkpoint / live-status / unrecoverable salvage branches
    # behind the same atomic claim the downloaded branch uses, so a peer operator
    # replica (or a racing benchmark-complete/sidecar-export path) cannot
    # double-delete the JobSet, double-emit events, or stomp a COMPLETED terminal
    # phase with FAILED.
    if not await try_claim_completion(namespace, name, body):
        return False

    await _salvage_terminated_controller_results(
        api,
        body=body,
        result=result,
        status=status,
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        reason=reason,
        sb=sb,
    )
    return True


async def _fetch_progress(
    namespace: str,
    jobset_name: str,
    patch: kopf.Patch,
    sb: StatusBuilder,
    progress_client: ProgressClient,
    key: str,
    current_phase: Phase = Phase.RUNNING,
    *,
    body: dict[str, Any] | None = None,
) -> bool:
    """Retired: the run pushes progress; the operator does not poll for it.

    Under the native execution model there is no per-run ``/api/progress`` service
    to poll. The run pod (``aiperf controller``) pushes its live progress straight
    into its own AIPerfJob ``.status.phases.<phase>`` via
    ``completion_signal.report_benchmark_progress`` (the same in-cluster push it
    already used for the completion annotation), and completion arrives on the
    ``benchmark-complete`` annotation, which ``lifecycle.on_benchmark_complete``
    turns into the full completion (results fetch + JobSet delete + phase=Completed)
    via a kopf field watcher. So this function no longer performs the mesh HTTP poll
    (``progress_client.get_progress`` + the ``.status.phases`` overwrite + the
    live/server-metrics fetches): those all targeted the retired ZMQ ``api`` service.

    It keeps its signature and always returns ``False`` (never "poll-observed
    complete") so the surrounding tick orchestration is unchanged and completion
    flows exclusively through the push/annotation path — identical to how this poll
    already behaved on the native path, where ``get_progress`` hit a connection error
    and returned ``False`` every tick. ``progress_client`` stays a parameter because
    the caller still uses it for results-sidecar recovery and controller shutdown.
    """
    return False

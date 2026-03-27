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
import kr8s
from kr8s.asyncio.objects import Pod

from aiperf.common.enums import WorkerStartupState
from aiperf.kubernetes.constants import Containers, JobSetLabels
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.kr8s_resources import AsyncJobSet
from aiperf.operator import events
from aiperf.operator.client_cache import (
    _shutdown_sent,
    _warned_pod_restarts,
    close_progress_client,
    get_or_create_progress_client,
    job_key,
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
    from aiperf.kubernetes.client import get_api
else:
    from aiperf.kubernetes.client import get_api

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


async def _get_app_ready_worker_count(
    client: Any,
    controller_host: str,
) -> int | None:
    """Get the count of workers that reported app-level READY startup state."""
    states = await client.get_worker_startup_states(controller_host)
    if states is None:
        return None
    return sum(1 for state in states.values() if state == WorkerStartupState.READY)


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
    current_phase = status.get("phase", Phase.PENDING)

    # Stop monitoring terminal jobs
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        return

    jobset_name = status.get("jobSetName")
    job_id = status.get("jobId")
    if not jobset_name or not job_id:
        return

    sb = StatusBuilder(patch, status)

    try:
        key = job_key(namespace, job_id)

        # Check job timeout
        timeout_sec = _get_job_timeout(spec)
        if timeout_sec > 0:
            elapsed = _get_elapsed_seconds(status)
            if elapsed is not None and elapsed > timeout_sec:
                sb.set_phase(Phase.FAILED).set_error(
                    f"Job timed out after {elapsed:.0f}s (limit: {timeout_sec:.0f}s)"
                )
                sb.set_completion_time()
                sb.finalize()
                events.job_timeout(body, job_id, elapsed)
                # Delete JobSet to free resources (like cancel does)
                if jobset_name:
                    try:
                        api = await get_api()
                        js = await AsyncJobSet.get(
                            jobset_name, namespace=namespace, api=api
                        )
                        await js.delete()
                        logger.info(f"Deleted JobSet {jobset_name} after timeout")
                    except kr8s.NotFoundError:
                        pass
                    except kr8s.ServerError as e:
                        logger.warning(f"Failed to delete JobSet on timeout: {e}")
                await close_progress_client(key)
                return

        api = await get_api()

        # Get JobSet status
        try:
            jobset_obj = await AsyncJobSet.get(
                jobset_name, namespace=namespace, api=api
            )
            jobset = jobset_obj.raw
        except kr8s.NotFoundError:
            # JobSet may have been deleted by the completion handler after
            # successful results fetch. Don't overwrite a terminal phase.
            if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
                logger.debug(
                    f"JobSet {jobset_name} not found but phase is already "
                    f"{current_phase} - skipping"
                )
            else:
                # Re-read the CR to catch completion handler's update
                # (it may have set phase=Completed and deleted the JobSet
                # between our phase read and the JobSet lookup).
                await asyncio.sleep(2)
                from aiperf.kubernetes.kr8s_resources import AsyncAIPerfJob

                try:
                    api = await get_api()
                    fresh = await AsyncAIPerfJob.get(name, namespace=namespace, api=api)
                    fresh_phase = fresh.raw.get("status", {}).get("phase", "")
                    if fresh_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
                        logger.debug(
                            f"JobSet {jobset_name} not found but fresh phase is "
                            f"{fresh_phase} - skipping"
                        )
                        await close_progress_client(key)
                        return
                except Exception:
                    pass
                sb.set_phase(Phase.FAILED).set_error("JobSet not found")
                sb.finalize()
            await close_progress_client(key)
            return

        jobset_status = jobset.get("status", {})

        # Detect Kueue-managed suspension (gang-scheduling)
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
            return

        # Check for terminal state
        for condition in jobset_status.get("conditions", []):
            if condition.get("status") != "True":
                continue
            if condition.get("type") == "Completed":
                # Skip if already handled by the annotation handler
                if key not in _shutdown_sent:
                    await handle_completion(
                        body, namespace, jobset_name, job_id, status, sb
                    )
                await close_progress_client(key)
                return
            if condition.get("type") == "Failed":
                is_fatal, failed_scope = _classify_jobset_failure(jobset_status)
                if is_fatal:
                    sb.set_phase(Phase.FAILED)
                    sb.set_error(condition.get("message", "JobSet failed"))
                    sb.finalize()
                    events.failed(
                        body, job_id, condition.get("message", "JobSet failed")
                    )
                    await close_progress_client(key)
                    return

                logger.warning(
                    "Ignoring non-fatal JobSet failure for %s: failed_scope=%s message=%s",
                    job_id,
                    failed_scope,
                    condition.get("message", "JobSet failed"),
                )

        # Update worker count and phase
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

        # Phase transitions based on worker readiness
        if current_phase in (Phase.PENDING, Phase.QUEUED) and (
            workers_ready > 0 or workers_succeeded > 0
        ):
            sb.set_phase(Phase.INITIALIZING)

        controller_host = controller_dns_name(jobset_name, namespace)
        client = None
        app_workers_ready: int | None = None

        effective_phase = sb.get_phase() or current_phase
        if (
            effective_phase == Phase.INITIALIZING
            and total_workers > 0
            and (workers_ready == total_workers or workers_succeeded == total_workers)
        ):
            client = await get_or_create_progress_client(key)
            app_workers_ready = await _get_app_ready_worker_count(
                client, controller_host
            )
            if app_workers_ready is not None:
                sb.set_workers(app_workers_ready, total_workers)

            if app_workers_ready == total_workers:
                sb.set_phase(Phase.RUNNING)
                sb.conditions.set_true(
                    ConditionType.WORKERS_READY,
                    "AllWorkersReady",
                    f"All {total_workers} workers are ready",
                )
                sb.conditions.set_true(
                    ConditionType.BENCHMARK_RUNNING,
                    "BenchmarkStarted",
                    "Benchmark is running",
                )
                events.workers_ready(body, app_workers_ready, total_workers)
                events.started(body, job_id)

        # Check for pod restarts (CrashLoopBackOff detection)
        await _check_pod_restarts(api, body, namespace, jobset_name, key)

        if await _maybe_recover_terminated_controller(
            api, body, namespace, jobset_name, job_id, status, sb, key
        ):
            await close_progress_client(key)
            return

        # Fetch progress from controller if running or workers already completed
        effective_phase = sb.get_phase() or current_phase
        if effective_phase == Phase.RUNNING or (
            workers_succeeded > 0 and workers_succeeded >= total_workers
        ):
            if client is None:
                client = await get_or_create_progress_client(key)
            benchmark_complete = await _fetch_progress(
                namespace, jobset_name, patch, sb, client, key
            )

            # If benchmark is done, fetch results then shutdown controller
            if benchmark_complete:
                _shutdown_sent.add(key)
                logger.info(
                    f"Benchmark complete for {jobset_name}, "
                    f"fetching results and shutting down controller"
                )
                host = controller_dns_name(jobset_name, namespace)
                await handle_completion(
                    body, namespace, jobset_name, job_id, status, sb
                )
                # Shutdown controller after results are fetched
                await client.send_shutdown(host)
                await close_progress_client(key)
                return

        sb.finalize()

    except (
        kr8s.ServerError,
        kr8s.NotFoundError,
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
    api: kr8s.Api,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    key: str,
) -> None:
    """Check for excessive pod restarts and emit warning events (deduplicated)."""
    try:
        pods = [
            pod
            async for pod in Pod.list(
                namespace=namespace,
                label_selector=f"jobset.sigs.k8s.io/jobset-name={jobset_name}",
                api=api,
            )
        ]
        warned = _warned_pod_restarts.setdefault(key, set())
        for pod in pods:
            pod_status = pod.raw.get("status", {})
            all_statuses = pod_status.get("containerStatuses", []) + pod_status.get(
                "initContainerStatuses", []
            )
            for cs in all_statuses:
                restart_count = cs.get("restartCount", 0)
                if restart_count < OperatorEnvironment.POD_RESTART_THRESHOLD:
                    continue
                dedup_key = (pod.name, restart_count)
                if dedup_key in warned:
                    continue
                warned.add(dedup_key)
                reason = "Unknown"
                last_state = cs.get("lastState", {})
                terminated = last_state.get("terminated", {})
                if terminated:
                    reason = terminated.get("reason", "Unknown")
                waiting = cs.get("state", {}).get("waiting", {})
                if waiting:
                    reason = waiting.get("reason", reason)
                events.pod_restarts(body, pod.name, restart_count, reason)
    except Exception as e:
        logger.warning(f"Failed to check pod restarts: {e}")


def _container_status_by_name(
    statuses: list[dict[str, Any]], name: str
) -> dict[str, Any] | None:
    """Return the first container status matching the given name."""
    for status in statuses:
        if status.get("name") == name:
            return status
    return None


async def _maybe_recover_terminated_controller(
    api: kr8s.Api,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
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

    try:
        pods = [
            pod
            async for pod in Pod.list(
                namespace=namespace,
                label_selector=(
                    f"{JobSetLabels.JOBSET_NAME}={jobset_name},"
                    f"{JobSetLabels.REPLICATED_JOB_NAME}=controller"
                ),
                api=api,
            )
        ]
    except Exception as e:
        logger.warning(f"Failed to inspect controller pod for salvage: {e}")
        return False

    if not pods:
        return False

    pod = pods[0]
    statuses = pod.raw.get("status", {}).get("containerStatuses", [])
    controller_status = _container_status_by_name(statuses, Containers.CONTROL_PLANE)
    sidecar_status = _container_status_by_name(statuses, Containers.RESULTS_SIDECAR)
    if controller_status is None or sidecar_status is None:
        return False

    terminated = controller_status.get("state", {}).get("terminated")
    if not terminated:
        return False

    exit_code = int(terminated.get("exitCode", 0) or 0)
    reason = terminated.get("reason", "Error")
    if exit_code == 0:
        return False

    logger.warning(
        "Controller container terminated in pod %s (reason=%s, exitCode=%s), "
        "attempting results recovery from sidecar",
        pod.name,
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
            status,
            sb,
            result=result,
        )
        return True

    if result.checkpoints:
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

        try:
            js = await AsyncJobSet.get(jobset_name, namespace=namespace, api=api)
            await js.delete()
            logger.info(
                "Deleted JobSet %s after partial checkpoint recovery", jobset_name
            )
        except kr8s.NotFoundError:
            pass
        except kr8s.ServerError as e:
            logger.warning(
                f"Failed to delete JobSet {jobset_name} after checkpoint recovery: {e}"
            )
        return True

    error = f"Controller container terminated before results were recoverable: {reason}"
    sb.set_phase(Phase.FAILED).set_error(error).set_completion_time()
    sb.conditions.set_false(
        ConditionType.RESULTS_AVAILABLE,
        "ControllerTerminated",
        "Controller terminated before exporting recoverable result files",
    )
    sb.finalize()
    events.failed(body, job_id, error)

    try:
        js = await AsyncJobSet.get(jobset_name, namespace=namespace, api=api)
        await js.delete()
        logger.info(
            "Deleted JobSet %s after unrecoverable controller termination", jobset_name
        )
    except kr8s.NotFoundError:
        pass
    except kr8s.ServerError as e:
        logger.warning(
            f"Failed to delete JobSet {jobset_name} after controller termination: {e}"
        )
    return True


async def _fetch_progress(
    namespace: str,
    jobset_name: str,
    patch: kopf.Patch,
    sb: StatusBuilder,
    client: Any,
    key: str,
) -> bool:
    """Fetch progress and live metrics from controller pod.

    Returns True if the benchmark is complete (all profiling requests done).
    """

    host = controller_dns_name(jobset_name, namespace)

    try:
        progress = await client.get_progress(host)

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

        if progress.current_phase:
            patch.status["currentPhase"] = progress.current_phase

        if progress.error:
            patch.status["error"] = progress.error

        # Fetch live metrics
        try:
            metrics = await client.get_metrics(host)
        except Exception as e:
            logger.warning(f"Live metrics fetch failed for {jobset_name}: {e}")
            metrics = None

        if isinstance(metrics, dict) and metrics.get("metrics"):
            patch.status["liveMetrics"] = metrics

            summary = MetricsSummary.from_metrics(metrics)
            summary_dict = summary.to_status_dict()
            if summary_dict:
                patch.status["liveSummary"] = summary_dict

        # Server metrics: fetch if available (endpoint may not exist yet)
        try:
            server_metrics = await client.get_server_metrics(host)
            if isinstance(server_metrics, dict) and server_metrics.get(
                "endpoint_summaries"
            ):
                patch.status["serverMetrics"] = server_metrics
        except Exception:
            pass

        # Return completion status for caller to handle
        if progress.is_complete and key not in _shutdown_sent:
            return True

    except Exception as e:
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

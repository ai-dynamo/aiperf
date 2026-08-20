# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-driven recovery and heartbeat-watchdog logic for AIPerfJob CRD.

This module contains the business logic only — no kopf decorators.
Decorators live in ``aiperf.operator.main``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import kopf
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient, CustomObjectsApi
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.enums import SystemState
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.common.results_markers import READY_MARKER_NAME, write_ready_marker
from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.constants import (
    CONTROLLER_HEARTBEAT_INTERVAL_SECONDS,
    Annotations,
    Containers,
    JobSetLabels,
)
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.kubernetes.crd_models import (
    ControllerFetchResult,
    MetricsSummary,
    PhaseProgress,
)
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.phase import Phase, format_timestamp, parse_timestamp
from aiperf.kubernetes.spec_converter import (
    DEFAULT_KEY_EXPORT_NAMES,
    key_export_names_from_body,
)
from aiperf.operator import events, runs_index
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
from aiperf.operator.handlers._job_identity import (
    StaleAIPerfJobCallback,
    aiperfjob_jobset_uid,
    body_name,
    body_uid,
    current_aiperfjob_body,
    current_aiperfjob_resource_version,
    delete_owned_aiperfjob_jobset,
    fence_status_patch,
    owned_aiperfjob_jobset_uid,
)
from aiperf.operator.handlers.completion import (
    _parse_metrics_from_files,
    _recover_result_from_disk,
    fetch_results_with_retry,
    handle_completion,
)
from aiperf.operator.progress_client import ProgressClient
from aiperf.operator.results_layout import epoch_key_from_body, run_dir
from aiperf.operator.status import (
    ConditionType,
    StatusBuilder,
)

logger = logging.getLogger(__name__)

IMAGE_PULL_WAITING_REASONS = frozenset(
    {"ErrImageNeverPull", "ErrImagePull", "ImagePullBackOff", "InvalidImageName"}
)
CONFIG_WAITING_REASONS = frozenset(
    {"CreateContainerConfigError", "CreateContainerError", "RunContainerError"}
)
CRASH_LOOP_WAITING_REASONS = frozenset({"CrashLoopBackOff"})

# Scheduler messages are aggregates across nodes. Any capacity signal keeps the
# issue recoverable even when another node also has a structural mismatch: the
# matching node may become available without a spec change.
_TRANSIENT_SCHEDULING_MARKERS = (
    "insufficient ",
    "too many pods",
    "preemption is not helpful",
    "unbound immediate persistentvolumeclaims",
    "didn't find available persistent volumes",
)
_STRUCTURAL_SCHEDULING_MARKERS = (
    "didn't match pod's node affinity/selector",
    "did not match pod's node affinity/selector",
    "had untolerated taint",
    "volume node affinity conflict",
)
KEY_RESULT_FILES = DEFAULT_KEY_EXPORT_NAMES.names
CHECKPOINTS_PREFIX = "checkpoints/"
STARTUP_ISSUE_STATUS_KEY = "startupIssue"
CONTROLLER_HEARTBEAT_EXPIRY_SECONDS = 3 * CONTROLLER_HEARTBEAT_INTERVAL_SECONDS


@dataclass(slots=True)
class _EventStatusPatch:
    """Minimal patch target for status changes triggered by watched resources."""

    status: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PodStartupIssue:
    """A pod startup blocker observed by the operator."""

    pod_name: str
    """Pod reporting the blocker."""

    container_name: str
    """Container reporting the blocker, or ``pod`` for scheduling failures."""

    reason: str
    """Kubernetes waiting or scheduling reason."""

    message: str
    """Kubernetes diagnostic message, if present."""

    category: str
    """Stable category used to track the same blocker across reason changes."""

    terminal_after_threshold: bool
    """Whether the operator may fail the job once the blocker is stable."""

    @property
    def fingerprint(self) -> str:
        """Return the durable identity used by ``status.startupIssue``."""
        return f"{self.category}:{self.pod_name}:{self.container_name}"


@dataclass(frozen=True, slots=True)
class _StartupIssueDecision:
    """Status mutation and side effects due for one cached startup issue."""

    state: dict[str, Any]
    condition_message: str
    warning_due: bool
    is_critical: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class _StartupDeadlineContext:
    """Immutable identity and blocker fence for one deadline callback."""

    parent_name: str
    jobset_name: str
    job_id: str
    uid: str
    key: str
    fingerprint: str


def _resource_field(
    resource: Any,
    camel_name: str,
    snake_name: str | None = None,
    default: Any = None,
) -> Any:
    """Read one Kubernetes field from a watch dict or generated model."""
    if resource is None:
        return default
    if isinstance(resource, Mapping):
        return resource.get(camel_name, default)
    return getattr(resource, snake_name or camel_name, default)


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
    """Get job timeout from spec or global default. 0 means no timeout.

    The CRD declares ``spec.timeoutSeconds`` with ``default: 0``, so the
    apiserver materializes the key into every CR. A plain
    ``spec.get(key, ENV)`` therefore always finds 0 and the operator-wide
    ``AIPERF_JOB_TIMEOUT_SECONDS`` (and its helm ``jobTimeoutSeconds``) could
    never take effect -- the knob was dead configuration.

    Treat a falsy value as "unset" so the operator default applies. Because the
    apiserver defaults the field, an explicit user 0 is already
    indistinguishable from an omitted one, so nothing is lost here.

    The default stays 0 (no timeout), matching upstream batch/v1 Job, whose
    SetDefaults_Job never sets ActiveDeadlineSeconds. Upstream likewise never
    fails a Job whose pods cannot be scheduled, so an opt-in deadline is the
    sanctioned backstop for a job that can never progress.
    """
    return float(spec.get("timeoutSeconds") or OperatorEnvironment.JOB_TIMEOUT_SECONDS)


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


def _scheduling_issue_is_structural(message: str) -> bool:
    """Return whether an Unschedulable message requires a spec/cluster change."""
    normalized = message.casefold()
    if any(marker in normalized for marker in _TRANSIENT_SCHEDULING_MARKERS):
        return False
    return any(marker in normalized for marker in _STRUCTURAL_SCHEDULING_MARKERS)


def _container_startup_issues(pod: Any) -> list[PodStartupIssue]:
    """Extract actionable waiting states from regular and init containers."""
    pod_status = _resource_field(pod, "status")
    metadata = _resource_field(pod, "metadata")
    pod_name = _resource_field(metadata, "name", default="") or "unknown"
    statuses = [
        *(
            _resource_field(
                pod_status, "initContainerStatuses", "init_container_statuses"
            )
            or []
        ),
        *(_resource_field(pod_status, "containerStatuses", "container_statuses") or []),
    ]
    issues: list[PodStartupIssue] = []
    for container_status in statuses:
        state = _resource_field(container_status, "state")
        waiting = _resource_field(state, "waiting")
        reason = _resource_field(waiting, "reason")
        if reason in IMAGE_PULL_WAITING_REASONS:
            category = "ImagePull"
            terminal = True
        elif reason in CONFIG_WAITING_REASONS:
            category = "ContainerConfig"
            terminal = True
        elif reason in CRASH_LOOP_WAITING_REASONS:
            category = "CrashLoop"
            restarts = int(
                _resource_field(container_status, "restartCount", "restart_count", 0)
                or 0
            )
            terminal = restarts >= K8sEnvironment.WATCHDOG.CRASHLOOP_RESTART_THRESHOLD
        else:
            continue
        message = _resource_field(waiting, "message", default="") or ""
        if not message:
            message = _resource_field(container_status, "image", default="") or ""
        issues.append(
            PodStartupIssue(
                pod_name=pod_name,
                container_name=_resource_field(container_status, "name", default="")
                or "unknown",
                reason=reason,
                message=message,
                category=category,
                terminal_after_threshold=terminal,
            )
        )
    return issues


def _pod_scheduling_issue(pod: Any) -> PodStartupIssue | None:
    """Extract a PodScheduled=False/Unschedulable condition, if present."""
    pod_status = _resource_field(pod, "status")
    metadata = _resource_field(pod, "metadata")
    pod_name = _resource_field(metadata, "name", default="") or "unknown"
    for condition in _resource_field(pod_status, "conditions") or []:
        if _resource_field(condition, "type") != "PodScheduled":
            continue
        if str(_resource_field(condition, "status", default="")).casefold() != "false":
            continue
        if _resource_field(condition, "reason") != "Unschedulable":
            continue
        message = _resource_field(condition, "message", default="") or ""
        structural = _scheduling_issue_is_structural(message)
        return PodStartupIssue(
            pod_name=pod_name,
            container_name="pod",
            reason="Unschedulable",
            message=message,
            category="SchedulingConstraint" if structural else "SchedulingDelay",
            terminal_after_threshold=structural,
        )
    return None


def _get_pod_startup_issue(pods: list[Any]) -> PodStartupIssue | None:
    """Return the highest-priority deterministic startup issue from pods."""
    issues: list[PodStartupIssue] = []
    for pod in pods:
        issues.extend(_container_startup_issues(pod))
        scheduling_issue = _pod_scheduling_issue(pod)
        if scheduling_issue is not None:
            issues.append(scheduling_issue)
    if not issues:
        return None
    return min(
        issues,
        key=lambda issue: (
            not issue.terminal_after_threshold,
            issue.category,
            issue.pod_name,
            issue.container_name,
        ),
    )


def _pod_startup_message(name: str, jobset_name: str, issue: PodStartupIssue) -> str:
    """Format an actionable message for a pod startup blocker."""
    detail = f": {issue.message}" if issue.message else ""
    return (
        f"AIPerfJob {name} JobSet {jobset_name} pod {issue.pod_name} "
        f"container {issue.container_name} is blocked by {issue.reason}{detail}"
    )


async def _delete_jobset_or_retry(
    custom: CustomObjectsApi,
    namespace: str,
    jobset_name: str,
    *,
    body: dict[str, Any],
    context: str,
) -> bool:
    """Delete the exact owned JobSet and report whether status may finalize."""
    del custom  # Shared identity helper owns a short-lived, closed API client.
    deleted = await delete_owned_aiperfjob_jobset(
        namespace,
        jobset_name,
        parent_name=body_name(body, jobset_name.removeprefix("aiperf-")),
        parent_uid=body_uid(body),
        context=context,
    )
    if deleted:
        logger.info(f"Deleted JobSet {jobset_name} after {context}")
    return deleted


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
    expected_uid = body_uid(body)
    fresh_uid = (fresh.get("metadata") or {}).get("uid")
    if expected_uid is not None and fresh_uid != expected_uid:
        logger.info(
            "Skipping stale missing-JobSet callback for %s/%s uid=%s; live uid=%s",
            namespace,
            name,
            expected_uid,
            fresh_uid,
        )
        return True
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


def _startup_issue_state(
    status: dict[str, Any], issue: PodStartupIssue
) -> tuple[dict[str, Any], float]:
    """Build durable status for an issue and return its stable duration."""
    previous = status.get(STARTUP_ISSUE_STATUS_KEY) or {}
    same_issue = previous.get("fingerprint") == issue.fingerprint
    first_observed = (
        previous.get("firstObservedTime") if same_issue else None
    ) or format_timestamp()
    try:
        elapsed = max(
            0.0, (datetime.now(UTC) - parse_timestamp(first_observed)).total_seconds()
        )
    except (TypeError, ValueError):
        first_observed = format_timestamp()
        elapsed = 0.0
    state = {
        "fingerprint": issue.fingerprint,
        "podName": issue.pod_name,
        "containerName": issue.container_name,
        "reason": issue.reason,
        "message": issue.message,
        "category": issue.category,
        "terminalAfterThreshold": issue.terminal_after_threshold,
        "firstObservedTime": first_observed,
        "warningEmitted": bool(previous.get("warningEmitted")) if same_issue else False,
    }
    return state, elapsed


def _startup_condition_message(
    name: str,
    jobset_name: str,
    issue: PodStartupIssue,
    elapsed: float,
) -> str:
    """Describe a startup blocker and its recovery/failure policy."""
    base = _pod_startup_message(name, jobset_name, issue)
    if issue.terminal_after_threshold:
        critical = K8sEnvironment.WATCHDOG.PENDING_CRITICAL_THRESHOLD_SECONDS
        return (
            f"{base}. Observed unchanged for {elapsed:.0f}s; the operator fails "
            f"the job after {critical:.0f}s. Inspect with `aiperf kube debug "
            f"--job-id {name}`."
        )
    return (
        f"{base}. Observed for {elapsed:.0f}s. This may recover when cluster "
        f"capacity or dependent resources become available; the operator will "
        f"not fail the job for this condition alone. Inspect with `aiperf kube "
        f"debug --job-id {name}`."
    )


def _startup_issue_decision(
    *,
    status: dict[str, Any],
    issue: PodStartupIssue,
    name: str,
    jobset_name: str,
) -> _StartupIssueDecision:
    """Build the current warning/failure decision for one stable blocker."""
    issue_state, elapsed = _startup_issue_state(status, issue)
    condition_message = _startup_condition_message(name, jobset_name, issue, elapsed)
    warning_due = (
        elapsed >= K8sEnvironment.WATCHDOG.PENDING_THRESHOLD_SECONDS
        and not issue_state["warningEmitted"]
    )
    is_critical = (
        issue.terminal_after_threshold
        and elapsed >= K8sEnvironment.WATCHDOG.PENDING_CRITICAL_THRESHOLD_SECONDS
    )
    error = None
    if is_critical:
        critical_threshold = K8sEnvironment.WATCHDOG.PENDING_CRITICAL_THRESHOLD_SECONDS
        error = (
            f"{_pod_startup_message(name, jobset_name, issue)}; blocker remained "
            f"stable for {elapsed:.0f}s (critical threshold: "
            f"{critical_threshold:.0f}s)"
        )
    return _StartupIssueDecision(
        state=issue_state,
        condition_message=condition_message,
        warning_due=warning_due,
        is_critical=is_critical,
        error=error,
    )


def _apply_startup_issue_decision(
    *,
    decision: _StartupIssueDecision,
    issue: PodStartupIssue,
    patch: kopf.Patch,
    sb: StatusBuilder,
) -> None:
    """Stage one previously-derived startup issue decision."""
    if decision.warning_due:
        decision.state["warningEmitted"] = True
    patch.status[STARTUP_ISSUE_STATUS_KEY] = decision.state
    sb.conditions.set_false(
        ConditionType.WORKERS_READY,
        "PodStartupBlocked" if issue.terminal_after_threshold else "SchedulingDelayed",
        decision.condition_message,
    )
    if not decision.is_critical or decision.error is None:
        return
    sb.set_phase(Phase.FAILED).set_error(decision.error).set_completion_time()
    sb.conditions.set_false(
        ConditionType.WORKERS_READY,
        "PodStartupFailed",
        decision.error,
    )
    sb.finalize()


def _reconcile_known_startup_issue(
    *,
    issue: PodStartupIssue,
    body: dict[str, Any],
    status: dict[str, Any],
    patch: kopf.Patch,
    name: str,
    jobset_name: str,
    sb: StatusBuilder,
) -> bool:
    """Persist one blocker; the fenced deadline owns critical cleanup."""
    decision = _startup_issue_decision(
        status=status,
        issue=issue,
        name=name,
        jobset_name=jobset_name,
    )
    if decision.is_critical:
        patch.status[STARTUP_ISSUE_STATUS_KEY] = decision.state
        sb.conditions.set_false(
            ConditionType.WORKERS_READY,
            "PodStartupBlocked",
            decision.condition_message,
        )
        sb.finalize()
        return True
    _apply_startup_issue_decision(
        decision=decision,
        issue=issue,
        patch=patch,
        sb=sb,
    )
    if decision.warning_due:
        events.pod_startup_blocked(body, decision.condition_message)
    return False


async def _reconcile_pod_startup_issue(
    api: ApiClient,
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    patch: kopf.Patch,
    namespace: str,
    name: str,
    jobset_name: str,
    job_id: str,
    key: str,
    sb: StatusBuilder,
    pods: list[Any] | None = None,
) -> bool:
    """Surface pod startup blockers and fail stable non-recoverable states."""
    if pods is None:
        try:
            pod_list = await client.CoreV1Api(api).list_namespaced_pod(
                namespace=namespace,
                label_selector=f"{JobSetLabels.JOBSET_NAME}={jobset_name}",
            )
            pods = pod_list.items
        except (
            TimeoutError,
            ApiException,
            aiohttp.ClientError,
            OSError,
            TypeError,
        ) as e:
            logger.warning(
                "Failed to inspect pods for startup states on %s/%s: %s",
                namespace,
                name,
                e,
            )
            return False

    issue = _get_pod_startup_issue(pods)
    if issue is None:
        if status.get(STARTUP_ISSUE_STATUS_KEY) is not None:
            patch.status[STARTUP_ISSUE_STATUS_KEY] = None
            existing = sb.conditions.get_condition(ConditionType.WORKERS_READY)
            if existing is not None and existing.get("reason") in {
                "PodStartupBlocked",
                "SchedulingDelayed",
            }:
                sb.conditions.set_false(
                    ConditionType.WORKERS_READY,
                    "WorkersStarting",
                    "Startup blocker cleared; waiting for workers to become ready",
                )
        return False

    return _reconcile_known_startup_issue(
        issue=issue,
        body=body,
        status=status,
        patch=patch,
        name=name,
        jobset_name=jobset_name,
        sb=sb,
    )


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

    if jobset_name and not await _delete_jobset_or_retry(
        custom,
        namespace,
        jobset_name,
        body=body,
        context="timeout",
    ):
        return True
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
            # Job process exit is not proof that the controller completed its
            # durable results-ready handshake. Completion remains driven only
            # by the controller annotation or independently verified exports.
            continue
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
    spec: dict[str, Any],
    sb: StatusBuilder,
) -> tuple[int, int, int]:
    """Update worker ready/total on StatusBuilder.

    Returns (workers_ready, workers_succeeded, total_workers) all in
    *process* units (not pod units).

    The JobSet ``replicatedJobsStatus[name="workers"].ready`` field counts
    *pods*, not worker processes.  Multiplying by ``workers_per_pod`` converts
    to the same unit used by ``status.workers.total`` (set at job creation from
    ``RuntimeConfig.workers``), so ``aiperf kube list`` shows ``4/4`` instead
    of ``2/4`` for a job with ``workers: 4, workersPerPod: 2``.
    """
    total_workers = status.get("workers", {}).get("total", 0)
    workers_per_pod: int = (
        spec.get("benchmark", {}).get("runtime", {}).get("workersPerPod", 1) or 1
    )
    workers_ready = 0
    workers_succeeded = 0

    for rj in jobset_status.get("replicatedJobsStatus", []):
        if rj.get("name") == "workers":
            # rj counts are in *pod* units; scale to process units.
            workers_ready = rj.get("ready", 0) * workers_per_pod
            workers_succeeded = rj.get("succeeded", 0) * workers_per_pod
            # Derive total from JobSet if CRD status doesn't have it yet.
            if total_workers == 0:
                total_workers = (
                    (
                        rj.get("ready", 0)
                        + rj.get("active", 0)
                        + rj.get("succeeded", 0)
                        + rj.get("failed", 0)
                        + rj.get("suspended", 0)
                    )
                    * workers_per_pod
                ) or 1  # Fallback to 1 if all zero
            sb.set_workers(workers_ready, total_workers)
            if workers_ready > 0:
                # Nothing else ever sets WorkersReady true. Without this the
                # completion backfill always fires, so a completely healthy run
                # ends up asserting "Job completed before workers (N) were
                # observed ready" -- with N > 0, contradicting itself. Record
                # the observation when it actually happens.
                sb.conditions.set_true(
                    ConditionType.WORKERS_READY,
                    "WorkersReady",
                    f"{workers_ready} of {total_workers} worker job(s) ready",
                )

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


# Phases from which the results-sidecar completion path may still fire. A run
# that completes before the operator's first successful progress poll never
# leaves INITIALIZING, so it must be recoverable too.
_NON_TERMINAL_RECOVERABLE_PHASES = frozenset({Phase.INITIALIZING, Phase.RUNNING})


def _as_phase(value: Phase | str | None) -> Phase:
    """Coerce a raw ``status.phase`` string into a real ``Phase`` member.

    ``status.get("phase")`` and ``StatusBuilder.get_phase()`` both yield plain
    ``str``. ``Enum.__hash__`` hashes by member *name*, not value, so
    ``"Running" in frozenset({Phase.RUNNING})`` is ``False`` even though
    ``"Running" == Phase.RUNNING`` is ``True``. Tuple membership (``in (...)``)
    uses ``==`` and works; set membership silently does not. Any phase headed
    for a set-membership test must therefore be a genuine member first.

    Unrecognized values fall back to ``Phase.PENDING`` so a future phase string
    written by a newer operator cannot crash an older monitor tick.
    """
    try:
        return Phase(value)
    except ValueError:
        return Phase.PENDING


def _build_phase_progress(stats: CombinedPhaseStats) -> PhaseProgress | None:
    """Build PhaseProgress from CombinedPhaseStats."""
    total = stats.total_expected_requests or 0
    if total == 0 and stats.requests_sent == 0:
        return None

    elapsed = None
    if stats.start_ns is not None and stats.last_update_ns is not None:
        elapsed = round((stats.last_update_ns - stats.start_ns) / 1_000_000_000, 1)

    phase_kind = stats.phase_kind
    if phase_kind is None:
        phase_kind = "warmup" if str(stats.phase) == "warmup" else "profiling"

    return PhaseProgress(
        phase_name=stats.phase_name or str(stats.phase),
        phase_kind=phase_kind,
        phase_index=stats.phase_index,
        profiling_index=stats.profiling_index,
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
        sending_complete=stats.is_sending_complete,
        is_requests_complete=stats.is_requests_complete,
        is_records_complete=stats.is_records_complete,
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
    if not (key_export_names_from_body(body).names & set(final_files)):
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
        jobset = await custom.get_namespaced_custom_object(
            group=JOBSET_GROUP,
            version=JOBSET_VERSION,
            plural=JOBSET_PLURAL,
            namespace=namespace,
            name=jobset_name,
        )
        parent_uid = body_uid(body)
        if parent_uid is not None:
            aiperfjob_jobset_uid(
                jobset,
                jobset_name=jobset_name,
                parent_name=body_name(body, name),
                parent_uid=parent_uid,
            )
        return jobset
    except kopf.TemporaryError:
        raise
    except StaleAIPerfJobCallback as e:
        logger.info("Skipping stale monitor JobSet read: %s", e)
        await close_progress_client(key)
        return None
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


async def _run_worker_and_progress_phase(
    api: ApiClient,
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
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
        status=status, jobset_status=jobset_status, spec=spec, sb=sb
    )

    _set_initializing_when_workers_start(
        current_phase, workers_ready, workers_succeeded, sb
    )

    if await _reconcile_pod_startup_issue(
        api,
        body=body,
        status=status,
        patch=patch,
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

    effective_phase = _as_phase(sb.get_phase() or current_phase)
    # Any non-terminal phase: a benchmark can finish between two monitor ticks
    # (a 2000-request mock run takes under a second), and the controller API
    # dies with it. The CR is then stuck in Initializing with no live API to
    # promote it to Running, so gating recovery on RUNNING makes completion
    # unreachable for exactly the runs that finish fastest.
    if (
        effective_phase in _NON_TERMINAL_RECOVERABLE_PHASES
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
    ):
        return

    sb.finalize()


async def _jobset_has_terminal_condition(
    api: ApiClient,
    namespace: str,
    jobset_name: str,
    *,
    body: dict[str, Any] | None = None,
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
    parent_uid = body_uid(body) if body is not None else None
    if parent_uid is not None:
        try:
            aiperfjob_jobset_uid(
                jobset,
                jobset_name=jobset_name,
                parent_name=body_name(body, jobset_name.removeprefix("aiperf-")),
                parent_uid=parent_uid,
            )
        except (StaleAIPerfJobCallback, kopf.TemporaryError):
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
    body: dict[str, Any] | None = None,
) -> bool:
    """Return True only when there is evidence the benchmark is actually done.

    Checked signals (in order, short-circuiting on first hit):
        1. Controller ``/api/progress`` reports ``is_complete=True``.
        2. The control-plane container in the controller pod is terminated.

    Both signals are quick, read-only, and side-effect-free; if neither fires
    we return False so callers can skip eager completion work (e.g.
    ``_recover_orphaned_completion_claim``) while the benchmark is still in
    flight. A return value of False therefore means "no evidence yet, try
    again next tick" — never "definitely still running".
    """
    parent_uid = body_uid(body) if body is not None else None
    if parent_uid is not None:
        try:
            await owned_aiperfjob_jobset_uid(
                namespace,
                jobset_name,
                parent_name=body_name(body, jobset_name.removeprefix("aiperf-")),
                parent_uid=parent_uid,
            )
        except StaleAIPerfJobCallback:
            return False

    host = controller_dns_name(jobset_name, namespace)
    progress_client = await get_or_create_progress_client(key)
    try:
        progress = await progress_client.get_progress(host)
        if not progress.connection_error and progress.is_complete:
            return True
    except (TimeoutError, aiohttp.ClientError, OSError) as e:
        logger.debug(
            "progress probe for %s during orphan-claim gate failed: %s",
            jobset_name,
            e,
        )
    except Exception as e:  # noqa: BLE001 - gate is best-effort; fall through to the pod-status check on any parse/transport error
        logger.debug(
            "progress probe for %s during orphan-claim gate failed: %s",
            jobset_name,
            e,
        )

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
        return await _jobset_has_terminal_condition(
            api, namespace, jobset_name, body=body
        )
    statuses = (pod.status.container_statuses or []) if pod.status else []
    controller_status = _container_status_by_name(statuses, Containers.CONTROL_PLANE)
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
        if is_cancellation_requested(key):
            logger.info(
                "Cancellation requested for %s/%s during orphaned-claim "
                "recovery; skipping controller shutdown",
                namespace,
                name,
            )
            return
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
        body=body,
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
        spec=spec,
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


def _controller_heartbeat_is_fresh(
    body: dict[str, Any], status: dict[str, Any]
) -> bool:
    """Return whether controller liveness is still inside the recovery grace."""
    metadata = body.get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    heartbeat = annotations.get(Annotations.CONTROLLER_HEARTBEAT)
    reference = (
        heartbeat or metadata.get("creationTimestamp") or status.get("startTime")
    )
    if not isinstance(reference, str):
        return False
    try:
        age = (datetime.now(UTC) - parse_timestamp(reference)).total_seconds()
    except (TypeError, ValueError):
        return False
    return age <= CONTROLLER_HEARTBEAT_EXPIRY_SECONDS


def _explicit_timeout_is_due(status: dict[str, Any], spec: dict[str, Any]) -> bool:
    """Return whether the configured job deadline needs the recovery engine."""
    timeout = _get_job_timeout(spec)
    elapsed = _get_elapsed_seconds(status)
    return timeout > 0 and elapsed is not None and elapsed > timeout


async def heartbeat_watchdog(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Run broad recovery only after heartbeat expiry or an explicit timeout."""
    current_phase = status.get("phase", Phase.PENDING)
    if current_phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        return
    if _controller_heartbeat_is_fresh(body, status) and not _explicit_timeout_is_due(
        status, spec
    ):
        return
    await monitor_progress(
        body=body,
        status=status,
        spec=spec,
        name=name,
        namespace=namespace,
        patch=patch,
    )


def _cached_startup_issue(status: dict[str, Any]) -> PodStartupIssue | None:
    """Rehydrate a validated blocker from durable AIPerfJob status."""
    state = status.get(STARTUP_ISSUE_STATUS_KEY)
    if not isinstance(state, dict):
        return None
    fields = (
        "podName",
        "containerName",
        "reason",
        "message",
        "category",
    )
    if not all(isinstance(state.get(field), str) for field in fields):
        return None
    if not isinstance(state.get("terminalAfterThreshold"), bool):
        return None
    issue = PodStartupIssue(
        pod_name=state["podName"],
        container_name=state["containerName"],
        reason=state["reason"],
        message=state["message"],
        category=state["category"],
        terminal_after_threshold=state["terminalAfterThreshold"],
    )
    if state.get("fingerprint") != issue.fingerprint:
        return None
    return issue


async def _live_startup_deadline_parent(
    *,
    namespace: str,
    name: str,
    expected_uid: str,
    key: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    """Revalidate every parent precondition for a cached blocker action."""
    try:
        live_body = await current_aiperfjob_body(namespace, name, expected_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale startup deadline: %s", exc)
        return None
    if live_body is None:
        return None
    metadata = live_body.get("metadata") or {}
    if metadata.get("uid") != expected_uid or metadata.get("resourceVersion") is None:
        return None
    live_status = live_body.get("status") or {}
    if _as_phase(live_status.get("phase")).is_terminal:
        return None
    if (live_body.get("spec") or {}).get("cancel") is True:
        return None
    if is_cancellation_requested(key):
        return None
    annotations = metadata.get("annotations") or {}
    if annotations.get(Annotations.COMPLETION_CLAIMED):
        return None
    failure_claim = annotations.get(Annotations.STARTUP_FAILURE_CLAIMED)
    if failure_claim is not None and failure_claim != fingerprint:
        return None
    live_issue_state = live_status.get(STARTUP_ISSUE_STATUS_KEY)
    if (
        not isinstance(live_issue_state, dict)
        or live_issue_state.get("fingerprint") != fingerprint
        or _cached_startup_issue(live_status) is None
    ):
        return None
    return live_body


def _startup_deadline_context(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    namespace: str,
    name: str,
) -> _StartupDeadlineContext | None:
    """Validate the cached callback and capture its immutable identity."""
    issue = _cached_startup_issue(status)
    jobset_name = status.get("jobSetName")
    job_id = status.get("jobId")
    uid = body_uid(body)
    if (
        _as_phase(status.get("phase")).is_terminal
        or issue is None
        or not isinstance(jobset_name, str)
        or not isinstance(job_id, str)
        or uid is None
    ):
        return None
    decision = _startup_issue_decision(
        status=status,
        issue=issue,
        name=name,
        jobset_name=jobset_name,
    )
    key = job_key(namespace, job_id, uid)
    if (
        not decision.warning_due
        and not decision.is_critical
        or is_cancellation_requested(key)
        or (body.get("spec") or {}).get("cancel") is True
    ):
        return None
    return _StartupDeadlineContext(
        parent_name=body_name(body, name),
        jobset_name=jobset_name,
        job_id=job_id,
        uid=uid,
        key=key,
        fingerprint=issue.fingerprint,
    )


async def _live_startup_deadline_decision(
    *,
    context: _StartupDeadlineContext,
    namespace: str,
    name: str,
) -> tuple[dict[str, Any], PodStartupIssue, _StartupIssueDecision] | None:
    """Rebuild a due decision from the exact live parent."""
    live_body = await _live_startup_deadline_parent(
        namespace=namespace,
        name=context.parent_name,
        expected_uid=context.uid,
        key=context.key,
        fingerprint=context.fingerprint,
    )
    if live_body is None:
        return None
    live_status = live_body.get("status") or {}
    issue = _cached_startup_issue(live_status)
    if issue is None:
        return None
    decision = _startup_issue_decision(
        status=live_status,
        issue=issue,
        name=name,
        jobset_name=context.jobset_name,
    )
    if not decision.warning_due and not decision.is_critical:
        return None
    return live_body, issue, decision


async def _delete_startup_blocker_if_due(
    *,
    context: _StartupDeadlineContext,
    namespace: str,
    decision: _StartupIssueDecision,
    claimed_body: dict[str, Any],
) -> bool:
    """Delete only after the exact parent durably owns failure cleanup."""
    if not decision.is_critical:
        return True
    annotations = (claimed_body.get("metadata") or {}).get("annotations") or {}
    if annotations.get(Annotations.STARTUP_FAILURE_CLAIMED) != context.fingerprint:
        return False
    async with k8s_client() as api:
        return await _delete_jobset_or_retry(
            client.CustomObjectsApi(api),
            namespace,
            context.jobset_name,
            body=claimed_body,
            context="stable pod startup failure claim",
        )


def _startup_failure_claim_ops(
    body: dict[str, Any],
    fingerprint: str,
) -> list[dict[str, Any]]:
    """Build the atomic parent preconditions for stable-blocker cleanup."""
    metadata = body.get("metadata") or {}
    status = body.get("status") or {}
    annotations = metadata.get("annotations")
    operations: list[dict[str, Any]] = [
        {"op": "test", "path": "/metadata/uid", "value": metadata.get("uid")},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": metadata.get("resourceVersion"),
        },
        {"op": "test", "path": "/spec", "value": deepcopy(body.get("spec") or {})},
        {"op": "test", "path": "/status/phase", "value": status.get("phase")},
        {
            "op": "test",
            "path": f"/status/{STARTUP_ISSUE_STATUS_KEY}",
            "value": deepcopy(status.get(STARTUP_ISSUE_STATUS_KEY)),
        },
    ]
    if annotations is None:
        operations.append({"op": "add", "path": "/metadata/annotations", "value": {}})
    else:
        operations.append(
            {
                "op": "test",
                "path": "/metadata/annotations",
                "value": deepcopy(annotations),
            }
        )
    operations.append(
        {
            "op": "add",
            "path": (
                "/metadata/annotations/"
                f"{_json_pointer_token(Annotations.STARTUP_FAILURE_CLAIMED)}"
            ),
            "value": fingerprint,
        }
    )
    return operations


async def _claim_startup_failure(
    *,
    context: _StartupDeadlineContext,
    namespace: str,
    live_body: dict[str, Any],
) -> dict[str, Any] | None:
    """Atomically serialize stable-blocker cleanup against all parent winners."""
    annotations = (live_body.get("metadata") or {}).get("annotations") or {}
    existing = annotations.get(Annotations.STARTUP_FAILURE_CLAIMED)
    if existing == context.fingerprint:
        return live_body
    if existing is not None or annotations.get(Annotations.COMPLETION_CLAIMED):
        return None
    try:
        async with k8s_client() as api:
            claimed = await client.CustomObjectsApi(api).patch_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=namespace,
                name=context.parent_name,
                body=_startup_failure_claim_ops(live_body, context.fingerprint),
                _content_type="application/json-patch+json",
            )
    except ApiException as exc:
        if exc.status == 404:
            return None
        if exc.status not in (409, 422):
            raise kopf.TemporaryError(
                f"AIPerfJob {namespace}/{context.parent_name}: startup failure "
                f"claim failed ({exc.status}: {exc.reason})",
                delay=5,
            ) from exc
        current = await _live_startup_deadline_parent(
            namespace=namespace,
            name=context.parent_name,
            expected_uid=context.uid,
            key=context.key,
            fingerprint=context.fingerprint,
        )
        if current is None:
            return None
        current_annotations = (current.get("metadata") or {}).get("annotations") or {}
        if (
            current_annotations.get(Annotations.STARTUP_FAILURE_CLAIMED)
            == context.fingerprint
        ):
            return current
        raise kopf.TemporaryError(
            f"AIPerfJob {namespace}/{context.parent_name}: startup failure "
            "claim raced an unrelated writer; retrying",
            delay=1,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"AIPerfJob {namespace}/{context.parent_name}: startup failure "
            f"claim failed: {exc}",
            delay=5,
        ) from exc
    if not isinstance(claimed, dict):
        return None
    return claimed


async def _commit_startup_deadline_status(
    *,
    context: _StartupDeadlineContext,
    namespace: str,
    name: str,
) -> None:
    """Revalidate and atomically commit the cached blocker decision."""
    live = await _live_startup_deadline_decision(
        context=context,
        namespace=namespace,
        name=name,
    )
    if live is None:
        return
    live_body, issue, decision = live
    live_status = live_body.get("status") or {}
    event_patch = _EventStatusPatch(status={})
    sb = StatusBuilder(event_patch, live_status)
    _apply_startup_issue_decision(
        decision=decision,
        issue=issue,
        patch=event_patch,  # type: ignore[arg-type]
        sb=sb,
    )
    if not decision.is_critical:
        sb.finalize()
    committed = await _patch_event_status(
        body=live_body,
        namespace=namespace,
        name=name,
        status_patch_builder=lambda current: _rebase_event_status_patch(
            base_status=live_status,
            staged_status=event_patch.status,
            live_body=current,
        ),
        key=context.key,
        status_tests={
            STARTUP_ISSUE_STATUS_KEY: live_status.get(STARTUP_ISSUE_STATUS_KEY)
        },
        annotation_tests={Annotations.STARTUP_FAILURE_CLAIMED: context.fingerprint}
        if decision.is_critical
        else None,
    )
    if not committed:
        return
    if decision.warning_due:
        events.pod_startup_blocked(live_body, decision.condition_message)
    if decision.is_critical and decision.error is not None:
        events.failed(live_body, context.job_id, decision.error)
        await close_progress_client(context.key)


async def startup_issue_deadline(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """Age one cached startup blocker without polling Kubernetes resources."""
    context = _startup_deadline_context(
        body=body,
        status=status,
        namespace=namespace,
        name=name,
    )
    if context is None:
        return
    live = await _live_startup_deadline_decision(
        context=context,
        namespace=namespace,
        name=name,
    )
    if live is None:
        return
    live_body, _, decision = live
    claimed_body = live_body
    if decision.is_critical:
        claimed_body = await _claim_startup_failure(
            context=context,
            namespace=namespace,
            live_body=live_body,
        )
        if claimed_body is None:
            return
    if not await _delete_startup_blocker_if_due(
        context=context,
        namespace=namespace,
        decision=decision,
        claimed_body=claimed_body,
    ):
        return
    await _commit_startup_deadline_status(
        context=context,
        namespace=namespace,
        name=name,
    )


def _controller_subphase_target(
    *,
    status: dict[str, Any],
    new: str | None,
) -> Phase | None:
    """Return the monotonic coarse phase implied by a controller state."""
    try:
        system_state = SystemState(new)
    except (TypeError, ValueError):
        return None
    current_phase = _as_phase(status.get("phase"))
    if current_phase.is_terminal:
        return None

    if system_state in {
        SystemState.PROFILING,
        SystemState.PROCESSING,
        SystemState.STOPPING,
        SystemState.SHUTDOWN,
    } and current_phase in (Phase.PENDING, Phase.QUEUED, Phase.INITIALIZING):
        return Phase.RUNNING
    if current_phase in (Phase.PENDING, Phase.QUEUED):
        return Phase.INITIALIZING
    return None


async def handle_controller_subphase_event(
    *,
    body: dict[str, Any],
    new: str | None,
    namespace: str,
    name: str,
) -> None:
    """Project a controller lifecycle event through the live parent fence."""
    expected_uid = body_uid(body)
    if expected_uid is None:
        return
    stale_status = body.get("status") or {}
    job_id = stale_status.get("jobId")
    key = job_key(namespace, job_id, expected_uid) if isinstance(job_id, str) else None
    fence = await _live_event_status_fence(
        namespace=namespace,
        name=body_name(body, name),
        expected_uid=expected_uid,
        key=key,
    )
    if fence is None:
        return
    live_body, _, _ = fence
    live_status = live_body.get("status") or {}
    target = _controller_subphase_target(status=live_status, new=new)
    if target is None:
        return

    event_patch = _EventStatusPatch(status={})
    sb = StatusBuilder(event_patch, live_status)
    sb.set_phase(target).finalize()
    await _patch_event_status(
        body=body,
        namespace=namespace,
        name=name,
        status_patch_builder=lambda current: _rebase_event_status_patch(
            base_status=live_status,
            staged_status=event_patch.status,
            live_body=current,
        ),
        key=key,
    )


async def handle_controller_failure_event(
    *,
    body: dict[str, Any],
    new: str | None,
    namespace: str,
    name: str,
) -> None:
    """Commit a controller-reported fatal failure through the live status fence."""
    if not isinstance(new, str) or not (failure := new.strip()):
        return
    expected_uid = body_uid(body)
    if expected_uid is None:
        return
    stale_status = body.get("status") or {}
    job_id = stale_status.get("jobId")
    key = job_key(namespace, job_id, expected_uid) if isinstance(job_id, str) else None
    error = f"Controller reported fatal failure: {failure}"

    def _failure_status_patch(current: dict[str, Any]) -> dict[str, Any]:
        current_status = current.get("status") or {}
        event_patch = _EventStatusPatch(status={})
        (
            StatusBuilder(event_patch, current_status)
            .set_phase(Phase.FAILED)
            .set_error(error)
            .set_completion_time()
            .finalize()
        )
        return _rebase_event_status_patch(
            base_status=current_status,
            staged_status=event_patch.status,
            live_body=current,
        )

    committed = await _patch_event_status(
        body=body,
        namespace=namespace,
        name=name,
        status_patch_builder=_failure_status_patch,
        key=key,
        allow_completed=True,
    )
    if committed:
        events.failed(body, str(job_id or name), error)


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
    uid = body_uid(body)
    key = job_key(namespace, job_id, uid)

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
        resource_version = await current_aiperfjob_resource_version(
            namespace,
            body_name(body, name),
            uid,
        )
        fence_status_patch(sb, resource_version)
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
    except StaleAIPerfJobCallback as e:
        logger.info("Skipping stale monitor callback: %s", e)
        return
    except kopf.TemporaryError:
        raise
    except (ApiException, aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        logger.warning(f"Transient error monitoring {namespace}/{name}: {e}")
        # Clear any partial patch writes (e.g. resourceVersion from fence_status_patch,
        # or partial status from _monitor_tick) so that kopf's _timer does not call
        # patch_and_check with a non-empty patch.  patch_and_check has no try/except,
        # so an API failure there would escape _timer and add this handler to
        # memory.forever_stopped, permanently killing the timer for this job.
        patch.clear()
        raise kopf.TemporaryError(str(e), delay=10) from e
    except Exception:
        logger.exception(f"Unexpected error monitoring {namespace}/{name}")
        # Same rationale as the transient-error branch above: clear partial patch
        # writes before re-raising so patch_and_check is a no-op and cannot kill the timer.
        patch.clear()
        raise


def _json_pointer_token(value: str) -> str:
    """Escape one JSON Pointer token for a status patch path."""
    return value.replace("~", "~0").replace("/", "~1")


async def _live_event_status_fence(
    *,
    namespace: str,
    name: str,
    expected_uid: str,
    key: str | None,
    status_tests: dict[str, Any] | None = None,
    annotation_tests: dict[str, Any] | None = None,
    allow_completed: bool = False,
) -> tuple[dict[str, Any], str, str] | None:
    """Return a live non-terminal body and its atomic status fence values."""
    try:
        live_body = await current_aiperfjob_body(namespace, name, expected_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale event status patch: %s", exc)
        return None
    if live_body is None:
        return None
    live_metadata = live_body.get("metadata") or {}
    live_status = live_body.get("status") or {}
    live_phase = live_status.get("phase", Phase.PENDING)
    if live_phase in (Phase.FAILED, Phase.CANCELLED) or (
        live_phase == Phase.COMPLETED and not allow_completed
    ):
        return None
    if (live_body.get("spec") or {}).get("cancel") is True:
        return None
    if key is not None and is_cancellation_requested(key):
        return None
    live_annotations = live_metadata.get("annotations") or {}
    failure_claim = live_annotations.get(Annotations.STARTUP_FAILURE_CLAIMED)
    required_failure_claim = (annotation_tests or {}).get(
        Annotations.STARTUP_FAILURE_CLAIMED
    )
    if failure_claim is not None and failure_claim != required_failure_claim:
        return None
    if status_tests is not None and any(
        live_status.get(field) != value for field, value in status_tests.items()
    ):
        return None
    if annotation_tests is not None and any(
        live_annotations.get(field) != value
        for field, value in annotation_tests.items()
    ):
        return None
    resource_version = live_metadata.get("resourceVersion")
    if resource_version is None:
        raise kopf.TemporaryError(
            f"AIPerfJob {namespace}/{name}: event identity read returned no "
            "metadata.resourceVersion; retrying",
            delay=5,
        )
    return live_body, str(resource_version), str(live_phase)


async def _patch_event_status(
    *,
    body: dict[str, Any],
    namespace: str,
    name: str,
    status_patch_builder: Callable[[dict[str, Any]], dict[str, Any]],
    key: str | None = None,
    status_tests: dict[str, Any] | None = None,
    annotation_tests: dict[str, Any] | None = None,
    allow_completed: bool = False,
) -> bool:
    """Apply watched-resource status fields to the exact live AIPerfJob."""
    expected_uid = body_uid(body)
    if expected_uid is None:
        logger.warning(
            "Skipping unfenced event status patch for AIPerfJob %s/%s",
            namespace,
            name,
        )
        return False
    parent_name = body_name(body, name)
    fence = await _live_event_status_fence(
        namespace=namespace,
        name=parent_name,
        expected_uid=expected_uid,
        key=key,
        status_tests=status_tests,
        annotation_tests=annotation_tests,
        allow_completed=allow_completed,
    )
    if fence is None:
        return False
    live_body, resource_version, live_phase = fence
    status_patch = status_patch_builder(live_body)
    if not status_patch:
        return False

    operations: list[dict[str, Any]] = [
        {"op": "test", "path": "/metadata/uid", "value": expected_uid},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": str(resource_version),
        },
        {"op": "test", "path": "/status/phase", "value": str(live_phase)},
        *(
            {
                "op": "test",
                "path": f"/status/{_json_pointer_token(field)}",
                "value": value,
            }
            for field, value in (status_tests or {}).items()
        ),
        *(
            {
                "op": "test",
                "path": (f"/metadata/annotations/{_json_pointer_token(field)}"),
                "value": value,
            }
            for field, value in (annotation_tests or {}).items()
        ),
        *(
            {
                "op": "add",
                "path": f"/status/{_json_pointer_token(key)}",
                "value": value,
            }
            for key, value in status_patch.items()
        ),
    ]
    try:
        async with k8s_client() as api:
            await client.CustomObjectsApi(api).patch_namespaced_custom_object_status(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=namespace,
                name=parent_name,
                body=operations,
                _content_type="application/json-patch+json",
            )
        return True
    except ApiException as exc:
        if exc.status == 404:
            return False
        if exc.status in (409, 422):
            raise kopf.TemporaryError(
                f"AIPerfJob {namespace}/{name}: event status changed while "
                f"committing ({exc.status}: {exc.reason}); rebasing",
                delay=1,
            ) from exc
        raise kopf.TemporaryError(
            f"AIPerfJob {namespace}/{name}: event status patch failed "
            f"({exc.status}): {exc.reason}",
            delay=5,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"AIPerfJob {namespace}/{name}: event status patch failed: {exc}",
            delay=5,
        ) from exc


def _rebase_event_status_patch(
    *,
    base_status: dict[str, Any],
    staged_status: dict[str, Any],
    live_body: dict[str, Any],
) -> dict[str, Any]:
    """Rebase one event's changed condition types onto the live status."""
    rebased = deepcopy(staged_status)
    staged_conditions = staged_status.get("conditions")
    if not isinstance(staged_conditions, list):
        return rebased

    base_conditions = base_status.get("conditions") or []
    base_by_type = {
        condition.get("type"): condition
        for condition in base_conditions
        if isinstance(condition, dict) and isinstance(condition.get("type"), str)
    }
    staged_by_type = {
        condition.get("type"): condition
        for condition in staged_conditions
        if isinstance(condition, dict) and isinstance(condition.get("type"), str)
    }
    changed_types = {
        condition_type
        for condition_type, condition in staged_by_type.items()
        if base_by_type.get(condition_type) != condition
    }
    remaining = {
        condition_type: deepcopy(staged_by_type[condition_type])
        for condition_type in changed_types
    }
    conditions: list[dict[str, Any]] = []
    for condition in (live_body.get("status") or {}).get("conditions") or []:
        if not isinstance(condition, dict):
            continue
        condition_type = condition.get("type")
        replacement = remaining.pop(condition_type, None)
        conditions.append(replacement or deepcopy(condition))
    for condition in staged_conditions:
        if not isinstance(condition, dict):
            continue
        condition_type = condition.get("type")
        replacement = remaining.pop(condition_type, None)
        if replacement is not None:
            conditions.append(replacement)
    rebased["conditions"] = conditions
    return rebased


async def handle_jobset_progress_event(
    *,
    body: dict[str, Any],
    jobset_body: dict[str, Any],
    namespace: str,
    name: str,
) -> None:
    """Project one exact JobSet readiness update onto its AIPerfJob status."""
    status = body.get("status") or {}
    current_phase = _as_phase(status.get("phase"))
    if current_phase.is_terminal:
        return
    jobset_name = status.get("jobSetName")
    job_id = status.get("jobId")
    expected_uid = body_uid(body)
    if (
        not isinstance(jobset_name, str)
        or not isinstance(job_id, str)
        or expected_uid is None
        or (jobset_body.get("metadata") or {}).get("name") != jobset_name
    ):
        return
    try:
        aiperfjob_jobset_uid(
            jobset_body,
            jobset_name=jobset_name,
            parent_name=body_name(body, name),
            parent_uid=expected_uid,
        )
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale JobSet progress event: %s", exc)
        return

    key = job_key(namespace, job_id, expected_uid)
    if is_cancellation_requested(key):
        return
    event_patch = _EventStatusPatch(status={})
    sb = StatusBuilder(event_patch, status)
    workers_ready, workers_succeeded, _ = _update_worker_counts(
        status=status,
        jobset_status=jobset_body.get("status") or {},
        spec=body.get("spec") or {},
        sb=sb,
    )
    _set_initializing_when_workers_start(
        current_phase, workers_ready, workers_succeeded, sb
    )
    sb.finalize()
    await _patch_event_status(
        body=body,
        namespace=namespace,
        name=name,
        status_patch_builder=lambda current: _rebase_event_status_patch(
            base_status=status,
            staged_status=event_patch.status,
            live_body=current,
        ),
        key=key,
    )


async def handle_jobset_failure_event(
    *,
    body: dict[str, Any],
    jobset_body: dict[str, Any],
    namespace: str,
    name: str,
) -> None:
    """Handle one exact AIPerfJob-owned JobSet Failed watch transition."""
    status = body.get("status") or {}
    if status.get("phase", Phase.PENDING) in (
        Phase.COMPLETED,
        Phase.FAILED,
        Phase.CANCELLED,
    ):
        return
    jobset_name = status.get("jobSetName")
    job_id = status.get("jobId")
    if not isinstance(jobset_name, str) or not isinstance(job_id, str):
        return
    if (jobset_body.get("metadata") or {}).get("name") != jobset_name:
        return

    expected_uid = body_uid(body)
    if expected_uid is None:
        return
    try:
        aiperfjob_jobset_uid(
            jobset_body,
            jobset_name=jobset_name,
            parent_name=body_name(body, name),
            parent_uid=expected_uid,
        )
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale JobSet failure event: %s", exc)
        return

    key = job_key(namespace, job_id, expected_uid)
    if is_cancellation_requested(key):
        return
    failed = next(
        (
            condition
            for condition in (jobset_body.get("status") or {}).get("conditions", [])
            if isinstance(condition, dict)
            and condition.get("type") == "Failed"
            and condition.get("status") == "True"
        ),
        None,
    )
    if failed is None:
        return

    event_patch = _EventStatusPatch(status={})
    sb = StatusBuilder(event_patch, status)
    handled = await _handle_jobset_failed_condition(
        body=body,
        condition=failed,
        jobset_status=jobset_body.get("status") or {},
        job_id=job_id,
        key=key,
        sb=sb,
    )
    if handled:
        await _patch_event_status(
            body=body,
            namespace=namespace,
            name=name,
            status_patch_builder=lambda current: _rebase_event_status_patch(
                base_status=status,
                staged_status=event_patch.status,
                live_body=current,
            ),
            key=key,
        )


def _container_status_by_name(statuses: list[Any], name: str) -> Any | None:
    """Return the first container status matching the given name."""
    for status in statuses:
        if _resource_field(status, "name") == name:
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


def _terminated_state(status: Any) -> Any | None:
    """Return the terminated state for a container status, if any."""
    state = _resource_field(status, "state")
    terminated = _resource_field(state, "terminated")
    if (
        not terminated
        and int(_resource_field(status, "restartCount", "restart_count", 0) or 0) > 0
    ):
        last_state = _resource_field(status, "lastState", "last_state")
        terminated = _resource_field(last_state, "terminated")
    return terminated


def _get_terminated_controller_info(pod: Any) -> tuple[int, str] | None:
    """Return (exit_code, reason) if any control-plane container died non-zero.

    Every control-plane service runs as its own container in this pod. Only
    ``control-plane`` was inspected, so a records-manager or dataset-manager
    that OOMKilled left salvage un-triggered and the job hung until its
    timeout with a generic message. Any non-zero exit among them is grounds to
    salvage; ``control-plane`` is still preferred when several have died so
    the reported reason names the primary failure.

    Returns None if nothing terminated non-zero, or if the sidecar is missing
    (salvage reads results through it).
    """
    pod_status = _resource_field(pod, "status")
    statuses = (
        _resource_field(pod_status, "containerStatuses", "container_statuses") or []
    )
    sidecar_status = _container_status_by_name(statuses, Containers.RESULTS_SIDECAR)
    if sidecar_status is None:
        return None

    failures: list[tuple[str, int, str]] = []
    for status in statuses:
        name = _resource_field(status, "name", default="") or ""
        if name == Containers.RESULTS_SIDECAR:
            continue
        terminated = _terminated_state(status)
        if not terminated:
            continue
        exit_code = int(_resource_field(terminated, "exitCode", "exit_code", 0) or 0)
        if exit_code == 0:
            continue
        failures.append(
            (name, exit_code, _resource_field(terminated, "reason") or "Error")
        )

    if not failures:
        return None
    failures.sort(key=lambda f: f[0] != Containers.CONTROL_PLANE)
    name, exit_code, reason = failures[0]
    if name != Containers.CONTROL_PLANE:
        reason = f"{reason} (container {name})"
    return exit_code, reason


async def handle_pod_recovery_event(
    *,
    body: dict[str, Any],
    meta: dict[str, Any],
    namespace: str,
    name: str,
) -> None:
    """Handle startup blockers or controller termination from one Pod event."""
    issue = _get_pod_startup_issue([body])
    terminated = _get_terminated_controller_info(body)
    jobset_name = (meta.get("labels") or {}).get(JobSetLabels.JOBSET_NAME)
    if not isinstance(jobset_name, str):
        return

    from aiperf.operator.handlers import pod_restarts

    parent = await pod_restarts._lookup_aiperfjob_body(  # noqa: SLF001 - shared exact Pod-to-parent identity resolver
        namespace, jobset_name, body
    )
    if parent is None:
        return
    parent_metadata = parent.get("metadata") or {}
    parent_name = parent_metadata.get("name")
    parent_uid = body_uid(parent)
    status = parent.get("status") or {}
    job_id = status.get("jobId")
    startup_issue = status.get(STARTUP_ISSUE_STATUS_KEY) or {}
    clears_startup_issue = (
        issue is None and terminated is None and startup_issue.get("podName") == name
    )
    if (
        not isinstance(parent_name, str)
        or not isinstance(parent_uid, str)
        or not isinstance(job_id, str)
        or status.get("jobSetName") != jobset_name
        or status.get("phase", Phase.PENDING)
        in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED)
    ):
        return
    if issue is None and terminated is None and not clears_startup_issue:
        return

    key = job_key(namespace, job_id, parent_uid)
    if is_cancellation_requested(key):
        return

    event_patch = _EventStatusPatch(status={})
    sb = StatusBuilder(event_patch, status)
    async with k8s_client() as api:
        if terminated is not None:
            await _maybe_recover_terminated_controller(
                api,
                parent,
                namespace,
                jobset_name,
                job_id,
                status=status,
                sb=sb,
                key=key,
                name=parent_name,
                pod=body,
                jobset_verified=True,
            )
        else:
            handled = await _reconcile_pod_startup_issue(
                api,
                body=parent,
                status=status,
                patch=event_patch,  # type: ignore[arg-type]
                namespace=namespace,
                name=parent_name,
                jobset_name=jobset_name,
                job_id=job_id,
                key=key,
                sb=sb,
                pods=[body],
            )
            if not handled:
                sb.finalize()

    if not event_patch.status:
        return
    await _patch_event_status(
        body=parent,
        namespace=namespace,
        name=parent_name,
        status_patch_builder=lambda current: _rebase_event_status_patch(
            base_status=status,
            staged_status=event_patch.status,
            live_body=current,
        ),
        key=key,
    )


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
    if not await _delete_jobset_or_retry(
        custom,
        namespace,
        jobset_name,
        body=body,
        context="partial live metrics recovery",
    ):
        return True
    if not _apply_live_status_partial_results(status, sb):
        return False
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


def _run_artifact_inventory(run_path: Path) -> tuple[list[str], int, int]:
    """Return direct-file names, directory mtime, and bytes like ``list_runs``."""
    files = sorted(
        child
        for child in run_path.iterdir()
        if child.is_file() and child.name != READY_MARKER_NAME
    )
    return (
        [child.name for child in files],
        int(run_path.stat().st_mtime),
        sum(child.stat().st_size for child in files),
    )


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
    if not await _delete_jobset_or_retry(
        custom,
        namespace,
        jobset_name,
        body=body,
        context="partial checkpoint recovery",
    ):
        return
    epoch = epoch_key_from_body(body)
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    checkpoint_metrics = _parse_metrics_from_files(
        result.checkpoints,
        namespace,
        job_id,
        epoch=epoch,
        json_name=key_export_names_from_body(body).json_name,
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
    sb.set_phase(Phase.FAILED).set_error(error).set_completion_time()
    # Write the readiness marker so the operator results-server actually serves
    # the salvaged checkpoint artifacts; without it ``_require_run_ready`` 404s
    # the bundle / profile-export routes forever even though resultsPath points
    # at on-disk files. ``was_cancelled=False`` — this is a salvaged failure,
    # not a user cancellation.
    catalog_marker = runs_index.begin_catalog_update(
        OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch
    )
    write_ready_marker(dest_dir, was_cancelled=False)
    try:
        files, mtime_epoch, total_size_bytes = await asyncio.to_thread(
            _run_artifact_inventory, dest_dir
        )
        metrics = checkpoint_metrics or {}
        await runs_index.upsert_run_completed(
            namespace,
            job_id,
            epoch,
            summary_blob=runs_index._zstd_compress(metrics) if metrics else b"",
            metrics=metrics,
            files=files,
            mtime_epoch=mtime_epoch,
            start_time=(
                metrics.get("start_time")
                if isinstance(metrics.get("start_time"), str)
                else None
            ),
            total_size_bytes=total_size_bytes,
            phase="Failed",
        )
        await runs_index.upsert_run_failed(
            namespace,
            job_id,
            epoch,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001 - disk remains authoritative on index failure
        logger.exception("Failed to index partial checkpoints for %s", job_id)
        sb.conditions.set_false(
            ConditionType.INDEX_UPDATED,
            "IndexUpdateFailed",
            f"Index write failed: {exc}",
        )
        events.index_update_failed(body, str(exc))
    else:
        runs_index.finish_catalog_update(catalog_marker)
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
    if not await _delete_jobset_or_retry(
        custom,
        namespace,
        jobset_name,
        body=body,
        context="unrecoverable controller termination",
    ):
        return

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


async def _controller_pod_for_recovery(
    api: ApiClient,
    namespace: str,
    jobset_name: str,
    event_pod: Any | None,
) -> Any | None:
    """Use the watched Pod when present, otherwise find it for broad recovery."""
    if event_pod is not None:
        return event_pod
    return await _get_controller_pod(api, namespace, jobset_name)


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
    pod: Any | None = None,
    jobset_verified: bool = False,
) -> bool:
    """Recover results from the sidecar if the controller container terminated.

    A regular sidecar keeps the pod alive long enough for salvage, but that also
    means we cannot rely solely on JobSet terminal conditions. If the main
    controller container exits unexpectedly, attempt to recover exported files
    from the sidecar immediately.
    """
    if _skip_terminated_controller_recovery(key, status):
        # The running benchmark already pushed a fatal reason through its
        # sidecar. Its field watch owns the terminal status; never let a later
        # sidecar artifact probe reinterpret that intentional failure as a
        # successful completion.
        return True

    pod = await _controller_pod_for_recovery(api, namespace, jobset_name, pod)
    if pod is None:
        return False

    info = _get_terminated_controller_info(pod)
    if info is None:
        return False
    exit_code, reason = info
    pod_name = (
        _resource_field(_resource_field(pod, "metadata"), "name", default="") or ""
    )

    logger.warning(
        "Controller container terminated in pod %s (reason=%s, exitCode=%s), "
        "attempting results recovery from sidecar",
        pod_name,
        reason,
        exit_code,
    )

    if not jobset_verified and not await _terminated_controller_jobset_is_current(
        body=body,
        namespace=namespace,
        name=name,
        jobset_name=jobset_name,
    ):
        return True

    if is_cancellation_requested(key):
        logger.debug(
            "Cancellation requested for %s/%s before terminated-controller salvage; "
            "skipping recovery side effects",
            namespace,
            name,
        )
        return True

    # The Pod-event recovery path shares the epoch-keyed destination directory
    # and cached ProgressClient with normal controller completion. Claim before
    # the first fetch so a racing winner is the only path that can stream into
    # that directory or later close its client. A durable claim that survives a
    # crash is resumed by _recover_orphaned_completion_claim.
    if not await try_claim_completion(namespace, name, body):
        return False

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


def _skip_terminated_controller_recovery(key: str, status: dict[str, Any]) -> bool:
    """Return whether an existing lifecycle owner already decided this run."""
    return key in _shutdown_sent or bool(status.get("controllerFailure"))


async def _terminated_controller_jobset_is_current(
    *,
    body: dict[str, Any],
    namespace: str,
    name: str,
    jobset_name: str,
) -> bool:
    """Return whether salvage still targets this AIPerfJob's exact JobSet."""
    parent_uid = body_uid(body)
    if parent_uid is None:
        return True
    try:
        jobset_uid = await owned_aiperfjob_jobset_uid(
            namespace,
            jobset_name,
            parent_name=body_name(body, name),
            parent_uid=parent_uid,
        )
    except StaleAIPerfJobCallback:
        return False
    return jobset_uid is not None

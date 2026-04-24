# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes events posted via the Events API for the AIPerfJob operator.

NOT related to the AIPerf internal message-bus enums (see ``aiperf.common.enums``).
The ``EventType`` and ``EventReason`` names in this module refer to
Kubernetes Event resource fields (``type``, ``reason``) that surface in
``kubectl describe`` and ``kubectl get events``.
"""

from __future__ import annotations

import logging

import kopf

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum

logger = logging.getLogger(__name__)


class EventType(CaseInsensitiveStrEnum):
    """Kubernetes event types."""

    NORMAL = "Normal"
    WARNING = "Warning"


class EventReason(CaseInsensitiveStrEnum):
    """Standard event reasons for AIPerfJob.

    Follows Kubernetes naming conventions (CamelCase).
    """

    # Lifecycle events
    CREATED = "Created"
    STARTED = "Started"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

    # Validation events
    SPEC_VALID = "SpecValid"
    SPEC_INVALID = "SpecInvalid"
    ENDPOINT_REACHABLE = "EndpointReachable"
    ENDPOINT_UNREACHABLE = "EndpointUnreachable"
    PREFLIGHT_PASSED = "PreflightPassed"
    PREFLIGHT_FAILED = "PreflightFailed"
    PREFLIGHT_WARNING = "PreflightWarning"

    # Resource events
    RESOURCES_CREATED = "ResourcesCreated"
    RESOURCES_FAILED = "ResourcesFailed"
    WORKERS_READY = "WorkersReady"

    # Results events
    RESULTS_STORED = "ResultsStored"
    RESULTS_FAILED = "ResultsFailed"
    RESULTS_CLEANED = "ResultsCleaned"

    # Reliability events
    JOB_TIMEOUT = "JobTimeout"
    POD_RESTARTS = "PodRestarts"


def post_event(
    body: dict,
    reason: EventReason,
    message: str,
    event_type: EventType = EventType.NORMAL,
) -> None:
    """Post a Kubernetes event for the current resource.

    Args:
        body: The resource body from kopf handler.
        reason: The event reason (CamelCase identifier).
        message: Human-readable message describing the event.
        event_type: Normal or Warning.
    """
    try:
        if event_type == EventType.WARNING:
            kopf.warn(body, reason=str(reason), message=message)
        else:
            kopf.info(body, reason=str(reason), message=message)
    except LookupError:
        logger.warning("Could not post event: kopf context unavailable")
    except Exception as e:  # noqa: BLE001 - kopf event posting is best-effort; must not raise into handler bodies
        logger.warning(f"Failed to post event {reason}: {e}")


def created(body: dict, job_id: str, workers: int) -> None:
    """Emit event when AIPerfJob is created."""
    post_event(
        body,
        EventReason.CREATED,
        f"Created benchmark job {job_id} with {workers} workers",
    )


def started(body: dict, job_id: str) -> None:
    """Emit event when benchmark starts running."""
    post_event(body, EventReason.STARTED, f"Benchmark {job_id} started")


def completed(body: dict, job_id: str, duration_sec: float | None = None) -> None:
    """Emit event when benchmark completes successfully."""
    msg = f"Benchmark {job_id} completed successfully"
    if duration_sec is not None:
        msg += f" in {duration_sec:.1f}s"
    post_event(body, EventReason.COMPLETED, msg)


def failed(body: dict, job_id: str, error: str) -> None:
    """Emit event when benchmark fails."""
    post_event(
        body,
        EventReason.FAILED,
        f"Benchmark {job_id} failed: {error}",
        EventType.WARNING,
    )


def cancelled(body: dict, job_id: str) -> None:
    """Emit event when benchmark is cancelled."""
    post_event(
        body,
        EventReason.CANCELLED,
        f"Benchmark {job_id} was cancelled",
        EventType.WARNING,
    )


def spec_valid(body: dict) -> None:
    """Emit event when spec validation passes."""
    post_event(body, EventReason.SPEC_VALID, "Spec validation passed")


def spec_invalid(body: dict, error: str) -> None:
    """Emit event when spec validation fails."""
    post_event(
        body,
        EventReason.SPEC_INVALID,
        f"Spec validation failed: {error}",
        EventType.WARNING,
    )


def endpoint_reachable(body: dict, url: str) -> None:
    """Emit event when endpoint health check passes."""
    post_event(body, EventReason.ENDPOINT_REACHABLE, f"Endpoint {url} is reachable")


def endpoint_unreachable(body: dict, url: str, error: str) -> None:
    """Emit event when endpoint health check fails."""
    post_event(
        body,
        EventReason.ENDPOINT_UNREACHABLE,
        f"Endpoint {url} unreachable: {error}",
        EventType.WARNING,
    )


def resources_created(body: dict, configmap: str, jobset: str) -> None:
    """Emit event when Kubernetes resources are created."""
    post_event(
        body,
        EventReason.RESOURCES_CREATED,
        f"Created ConfigMap/{configmap} and JobSet/{jobset}",
    )


def workers_ready(body: dict, ready: int, total: int) -> None:
    """Emit event when all workers are ready."""
    post_event(body, EventReason.WORKERS_READY, f"All workers ready ({ready}/{total})")


def results_stored(body: dict, path: str, files: int) -> None:
    """Emit event when results are stored."""
    post_event(
        body,
        EventReason.RESULTS_STORED,
        f"Stored {files} result files to {path}",
    )


def results_failed(body: dict, error: str) -> None:
    """Emit event when results storage fails."""
    post_event(
        body,
        EventReason.RESULTS_FAILED,
        f"Failed to store results: {error}",
        EventType.WARNING,
    )


def results_cleaned(body: dict, job_id: str, age_days: int) -> None:
    """Emit event when old results are cleaned up."""
    post_event(
        body,
        EventReason.RESULTS_CLEANED,
        f"Cleaned up results for {job_id} (age: {age_days} days)",
    )


def job_timeout(body: dict, job_id: str, elapsed_sec: float) -> None:
    """Emit event when a job exceeds its timeout."""
    post_event(
        body,
        EventReason.JOB_TIMEOUT,
        f"Benchmark {job_id} timed out after {elapsed_sec:.0f}s",
        EventType.WARNING,
    )


def pod_restarts(body: dict, pod_name: str, restart_count: int, reason: str) -> None:
    """Emit event when a pod has excessive restarts."""
    post_event(
        body,
        EventReason.POD_RESTARTS,
        f"Pod {pod_name} has restarted {restart_count} times: {reason}",
        EventType.WARNING,
    )


def preflight_passed(body: dict, num_checks: int) -> None:
    """Emit event when all pre-flight checks pass."""
    post_event(
        body,
        EventReason.PREFLIGHT_PASSED,
        f"All {num_checks} pre-flight checks passed",
    )


def preflight_failed(body: dict, error: str) -> None:
    """Emit event when pre-flight checks fail."""
    post_event(
        body,
        EventReason.PREFLIGHT_FAILED,
        f"Pre-flight checks failed: {error}",
        EventType.WARNING,
    )


def preflight_warning(body: dict, check_name: str, message: str) -> None:
    """Emit event for a non-blocking pre-flight warning."""
    post_event(
        body,
        EventReason.PREFLIGHT_WARNING,
        f"Pre-flight warning [{check_name}]: {message}",
        EventType.WARNING,
    )

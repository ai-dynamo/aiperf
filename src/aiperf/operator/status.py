# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Status management for AIPerfJob custom resources.

This module provides utilities for managing AIPerfJob CR status including
phase transitions, conditions, and progress tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum

if TYPE_CHECKING:
    import kopf


def format_timestamp() -> str:
    """Generate a Kubernetes-compatible timestamp.

    Returns:
        ISO 8601 timestamp string with Z suffix.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(ts: str) -> datetime:
    """Parse a Kubernetes timestamp string.

    Args:
        ts: ISO 8601 timestamp string.

    Returns:
        datetime object in UTC.
    """
    # Handle both 'Z' suffix and '+00:00'
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


class Phase(CaseInsensitiveStrEnum):
    """AIPerfJob lifecycle phases.

    Phases represent the high-level state of a job:
    - PENDING: Resources created, waiting for pods to start
    - QUEUED: JobSet suspended by Kueue, waiting for quota admission
    - INITIALIZING: Pods starting, services initializing
    - RUNNING: Job actively executing
    - COMPLETED: Job finished successfully
    - FAILED: Job failed due to error
    - CANCELLED: Job was cancelled by user
    """

    PENDING = "Pending"
    QUEUED = "Queued"
    INITIALIZING = "Initializing"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class ConditionType(CaseInsensitiveStrEnum):
    """Standard condition types for AIPerfJob status.

    Conditions provide detailed status information about specific aspects
    of the job lifecycle.
    """

    CONFIG_VALID = "ConfigValid"
    ENDPOINT_REACHABLE = "EndpointReachable"
    PREFLIGHT_PASSED = "PreflightPassed"
    RESOURCES_CREATED = "ResourcesCreated"
    WORKERS_READY = "WorkersReady"
    BENCHMARK_RUNNING = "BenchmarkRunning"
    RESULTS_AVAILABLE = "ResultsAvailable"
    INDEX_UPDATED = "IndexUpdated"
    """Set to False when the job index write fails; results remain on disk."""

    PREFLIGHT_HAS_WARNINGS = "PreflightHasWarnings"
    """Set to True when preflight succeeded but one or more checks produced
    warnings. Observable via `kubectl get aiperfjob -o jsonpath` so admins
    can alert on degraded-but-passing cluster conditions without scraping
    kubectl events."""


class ConditionManager:
    """Manages the conditions list for AIPerfJob status.

    Conditions follow the Kubernetes convention of tracking specific aspects
    of resource state with timestamps for state transitions.

    Invariants:
        - ``lastTransitionTime`` is preserved across calls where the condition's
          ``status`` field does NOT change — only updated when the transition
          actually occurs. This matches upstream k8s condition semantics.
        - Condition ``status`` is stored as the string ``"True"``/``"False"``
          (not Python bool) to match the kubectl-visible serialized form.
    """

    __slots__ = ("_conditions",)

    def __init__(self) -> None:
        """Initialize an empty condition manager."""
        self._conditions: dict[ConditionType, dict[str, Any]] = {}

    def set_condition(
        self,
        condition_type: ConditionType,
        status: bool,
        reason: str = "",
        message: str = "",
    ) -> None:
        """Set or update a condition.

        Args:
            condition_type: The type of condition to set.
            status: True if the condition is met, False otherwise.
            reason: Short, CamelCase reason for the condition state.
            message: Human-readable message with details.
        """
        status_str = "True" if status else "False"

        # Only update timestamp if status changed
        existing = self._conditions.get(condition_type)
        if existing is None or existing["status"] != status_str:
            timestamp = format_timestamp()
        else:
            timestamp = existing["lastTransitionTime"]

        self._conditions[condition_type] = {
            "type": str(condition_type),  # Store string for Kubernetes
            "status": status_str,
            "reason": reason,
            "message": message,
            "lastTransitionTime": timestamp,
        }

    def set_true(
        self, condition_type: ConditionType, reason: str, message: str = ""
    ) -> None:
        """Convenience method to set a condition to True."""
        self.set_condition(condition_type, True, reason, message)

    def set_false(
        self, condition_type: ConditionType, reason: str, message: str = ""
    ) -> None:
        """Convenience method to set a condition to False."""
        self.set_condition(condition_type, False, reason, message)

    def get_condition(self, condition_type: ConditionType) -> dict[str, Any] | None:
        """Get a specific condition.

        Args:
            condition_type: The type of condition to retrieve.

        Returns:
            The condition dict or None if not set.
        """
        return self._conditions.get(condition_type)

    def is_condition_true(self, condition_type: ConditionType) -> bool:
        """Check if a condition is True.

        Args:
            condition_type: The type of condition to check.

        Returns:
            True if the condition exists and is True.
        """
        condition = self._conditions.get(condition_type)
        return condition is not None and condition["status"] == "True"

    def to_list(self) -> list[dict[str, Any]]:
        """Return conditions as a list for status patch.

        Returns:
            List of condition dicts suitable for Kubernetes status.
        """
        return list(self._conditions.values())

    def apply_to_patch(self, patch: kopf.Patch) -> None:
        """Apply conditions to a kopf status patch.

        Args:
            patch: The kopf.Patch object to update.
        """
        patch.status["conditions"] = self.to_list()

    @classmethod
    def from_status(cls, status: dict[str, Any] | None) -> ConditionManager:
        """Reconstruct ConditionManager from existing status.

        Args:
            status: Full status dict from Kubernetes CR, or None.

        Returns:
            ConditionManager populated with existing conditions.
        """
        manager = cls()
        if status is None:
            return manager

        for cond in status.get("conditions") or []:
            try:
                cond_type = ConditionType(cond["type"])
                manager._conditions[cond_type] = cond
            except (KeyError, ValueError):
                continue
        return manager


WORKER_AGGREGATE_STATUS_KEY_ALIASES = {
    "routerConnected": "router_connected",
    "readyRecordProcessors": "ready_record_processors",
    "declaredRecordProcessors": "declared_record_processors",
    "readyPods": "ready_pods",
    "totalPods": "total_pods",
    "degradedPods": "degraded_pods",
}
WORKER_AGGREGATE_STATUS_CRD_KEYS = {
    internal: external
    for external, internal in WORKER_AGGREGATE_STATUS_KEY_ALIASES.items()
}


class StatusBuilder:
    """Builder for AIPerfJob ``.status`` patches via the fluent
    ``sb.set_phase(...).set_error(...).finalize()`` pattern.

    Invariants:
        - ``finalize()`` MUST be called exactly once before the kopf handler
          returns; otherwise condition updates are never applied to the patch
          (condition writes queue in the ConditionManager until finalized).
        - Safe to construct from an existing ``status`` dict to preserve
          prior conditions across a handler invocation.

    Example:
        >>> sb = StatusBuilder(patch, existing_status=body.get("status"))
        >>> sb.set_phase(Phase.RUNNING).set_workers(ready=32, total=32)
        >>> sb.conditions.set_true(ConditionType.WORKERS_READY, "AllUp")
        >>> sb.finalize()
    """

    __slots__ = ("_patch", "_conditions")

    def __init__(
        self, patch: kopf.Patch, existing_status: dict[str, Any] | None = None
    ) -> None:
        """Initialize the status builder.

        Args:
            patch: The kopf.Patch object to update.
            existing_status: Existing status dict to preserve conditions.
        """
        self._patch = patch
        self._conditions = ConditionManager.from_status(existing_status)

    @property
    def conditions(self) -> ConditionManager:
        """Access the condition manager."""
        return self._conditions

    def set_phase(self, phase: Phase) -> StatusBuilder:
        """Set the job phase."""
        self._patch.status["phase"] = str(phase)
        return self

    def set_error(self, error: str) -> StatusBuilder:
        """Set an error message."""
        self._patch.status["error"] = error
        return self

    def set_completion_time(self) -> StatusBuilder:
        """Set completion timestamp to now."""
        self._patch.status["completionTime"] = format_timestamp()
        return self

    def set_worker_aggregate_status(self, workers: dict[str, int]) -> StatusBuilder:
        """Set the richer worker status snapshot using CR-facing camelCase keys."""
        normalized = {
            WORKER_AGGREGATE_STATUS_KEY_ALIASES.get(key, key): value
            for key, value in workers.items()
        }
        self._patch.status["workers"] = {
            WORKER_AGGREGATE_STATUS_CRD_KEYS.get(key, key): value
            for key, value in normalized.items()
        }
        return self

    def set_workers(self, ready: int, total: int) -> StatusBuilder:
        """Set worker counts."""
        return self.set_worker_aggregate_status({"ready": ready, "total": total})

    def set_results(self, results: dict[str, Any]) -> StatusBuilder:
        """Set the full results dict."""
        self._patch.status["results"] = results
        return self

    def set_results_path(self, path: str) -> StatusBuilder:
        """Set the results storage path."""
        self._patch.status["resultsPath"] = path
        return self

    def set_summary(self, summary: dict[str, Any]) -> StatusBuilder:
        """Set the metrics summary."""
        self._patch.status["summary"] = summary
        return self

    def get_phase(self) -> str | None:
        """Return the phase currently set in the patch, or None."""
        return self._patch.status.get("phase")

    def finalize(self) -> None:
        """Apply conditions to the patch. Call this last."""
        conditions_list = self._conditions.to_list()
        if conditions_list:
            self._patch.status["conditions"] = conditions_list

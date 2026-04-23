# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for Kubernetes pod status parsing and formatting.

Split out of ``kubernetes_service_manager.py`` to keep that module focused on
the ServiceManager itself. These helpers convert kubernetes_asyncio API objects
into the legacy dict shape used internally, extract actionable container
issues, and build human-readable failure reasons for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aiperf.kubernetes.constants import JobSetLabels
from aiperf.kubernetes.enums import PodPhase


@dataclass
class PodInfo:
    """Tracked state for a single Kubernetes worker pod."""

    pod_index: str
    """JobSet pod index (from JobSetLabels.POD_INDEX label)."""

    pod_name: str
    """Kubernetes pod name."""

    phase: PodPhase = PodPhase.PENDING
    """Current pod phase."""

    restart_count: int = 0
    """Total container restart count across all containers."""

    container_issues: list[str] = field(default_factory=list)
    """Active container-level issues (e.g. 'OOMKilled', 'CrashLoopBackOff')."""

    last_checked_ns: int = 0
    """Timestamp of last health check (nanoseconds)."""

    failed: bool = False
    """Whether this pod has been marked as failed in the registry."""

    @property
    def is_terminal(self) -> bool:
        """Whether the pod is in a terminal failure state."""
        return self.phase in (PodPhase.FAILED, PodPhase.UNKNOWN)


# Tuple shape used to pass the aggregated per-pod status summary between
# helpers. (pod_name, phase, container_statuses, status_dict)
PodSnapshot = tuple[str, PodPhase, list[dict], dict]


def container_statuses_as_dicts(container_statuses: list[Any]) -> list[dict]:
    """Convert a list of V1ContainerStatus objects to the legacy dict shape.

    Preserves the container-state field names (``state``, ``waiting``,
    ``terminated``, ``lastState``) used by ``extract_container_issues`` and
    ``format_pod_failure_reason`` so those helpers keep working unchanged.
    """
    results: list[dict] = []
    for cs in container_statuses:
        state_dict: dict[str, Any] = {}
        state = cs.state
        if state is not None:
            if state.waiting is not None:
                state_dict["waiting"] = {
                    "reason": state.waiting.reason or "",
                    "message": state.waiting.message or "",
                }
            if state.terminated is not None:
                state_dict["terminated"] = {
                    "reason": state.terminated.reason or "",
                    "message": state.terminated.message or "",
                    "exitCode": state.terminated.exit_code,
                }

        last_state_dict: dict[str, Any] = {}
        last_state = cs.last_state
        if last_state is not None and last_state.terminated is not None:
            last_state_dict["terminated"] = {
                "reason": last_state.terminated.reason or "",
            }

        results.append(
            {
                "name": cs.name or "unknown",
                "restartCount": cs.restart_count or 0,
                "state": state_dict,
                "lastState": last_state_dict,
            }
        )
    return results


def conditions_as_dicts(conditions: list[Any]) -> list[dict]:
    """Convert V1PodCondition objects to the legacy dict shape."""
    return [
        {
            "type": c.type or "",
            "status": c.status or "",
            "message": c.message or "",
        }
        for c in conditions
    ]


def extract_container_issues(container_statuses: list[dict]) -> list[str]:
    """Extract actionable issue labels from container statuses.

    Inspects waiting and terminated container states for known failure
    patterns like OOMKilled, CrashLoopBackOff, and ImagePullBackOff.
    """
    issues: list[str] = []
    seen: set[str] = set()
    for cs in container_statuses:
        state = cs.get("state", {})
        for state_key in ("waiting", "terminated"):
            entry = state.get(state_key, {})
            reason = entry.get("reason", "") if entry else ""
            if reason and reason not in seen:
                seen.add(reason)
                issues.append(reason)

        last_terminated = cs.get("lastState", {}).get("terminated", {})
        reason = last_terminated.get("reason", "") if last_terminated else ""
        if reason and reason not in seen:
            seen.add(reason)
            issues.append(reason)

    return issues


def _format_container_state(container_name: str, state: dict) -> list[str]:
    """Format the state-entries for a single container."""
    out: list[str] = []
    terminated = state.get("terminated", {})
    if terminated:
        reason = terminated.get("reason", "")
        exit_code = terminated.get("exitCode")
        detail = f"container '{container_name}': terminated"
        if reason:
            detail += f" ({reason})"
        if exit_code is not None:
            detail += f" exit_code={exit_code}"
        message = terminated.get("message", "")
        if message:
            detail += f" - {message[:200]}"
        out.append(detail)

    waiting = state.get("waiting", {})
    if waiting:
        reason = waiting.get("reason", "")
        if reason:
            detail = f"container '{container_name}': waiting ({reason})"
            message = waiting.get("message", "")
            if message:
                detail += f" - {message[:200]}"
            out.append(detail)
    return out


def format_pod_failure_reason(
    pod_name: str,
    phase: PodPhase,
    container_statuses: list[dict],
    status: dict,
) -> str:
    """Build a detailed failure reason string for a failed pod.

    Includes the pod phase, container exit codes, termination reasons,
    and any waiting state reasons to help operators diagnose the failure.
    """
    parts = [f"K8s pod '{pod_name}' is {phase}"]

    for cs in container_statuses:
        container_name = cs.get("name", "unknown")
        state = cs.get("state", {})
        parts.extend(_format_container_state(container_name, state))

    for cond in status.get("conditions", []):
        if cond.get("status") == "False" and cond.get("message"):
            parts.append(f"condition {cond['type']}: {cond['message'][:200]}")

    return " | ".join(parts)


def extract_pod_snapshot(pod: Any) -> tuple[str, PodSnapshot] | None:
    """Distill a kubernetes pod object into (pod_index, snapshot).

    Returns None for pods that don't belong to the ``workers`` replicated-job
    or lack a pod-index label.
    """
    pod_name = (pod.metadata.name if pod.metadata else "") or "unknown"
    labels = (pod.metadata.labels or {}) if pod.metadata else {}
    replicated_job = labels.get(JobSetLabels.REPLICATED_JOB_NAME)
    if replicated_job and replicated_job != "workers":
        return None
    pod_index = labels.get(JobSetLabels.POD_INDEX)
    if pod_index is None:
        return None

    phase_str = (pod.status.phase if pod.status else None) or str(PodPhase.UNKNOWN)
    phase = PodPhase(phase_str)
    cs_dicts = (
        container_statuses_as_dicts(pod.status.container_statuses or [])
        if pod.status
        else []
    )
    cond_dicts = conditions_as_dicts(pod.status.conditions or [] if pod.status else [])
    status_dict = {"conditions": cond_dicts, "containerStatuses": cs_dicts}
    return pod_index, (pod_name, phase, cs_dicts, status_dict)


def aggregate_pods_by_index(pods: list[Any]) -> dict[str, PodSnapshot]:
    """Pick the best (non-terminal > terminal) snapshot per pod index."""
    pods_by_index: dict[str, PodSnapshot] = {}
    for pod in pods:
        extracted = extract_pod_snapshot(pod)
        if extracted is None:
            continue
        pod_index, snapshot = extracted
        existing = pods_by_index.get(pod_index)
        if existing is None or (
            existing[1] in (PodPhase.FAILED, PodPhase.UNKNOWN)
            and snapshot[1] not in (PodPhase.FAILED, PodPhase.UNKNOWN)
        ):
            pods_by_index[pod_index] = snapshot
    return pods_by_index

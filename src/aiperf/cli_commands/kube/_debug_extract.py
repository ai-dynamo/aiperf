# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pod-info extraction helpers for ``aiperf kube debug``.

Separated from ``debug.py`` to keep the CLI module small and to split the
container-status decoding logic into focused helpers.
"""

from __future__ import annotations

from typing import Any

# Lazy module-scoped ApiClient used only for sanitize_for_serialization.
# Reused across calls to avoid leaking an aiohttp session per invocation;
# sanitize_for_serialization does not require the client to be started.
_serializer: Any = None


def _get_serializer() -> Any:
    """Return a cached ``ApiClient`` for ``sanitize_for_serialization``.

    The client is allocated on first use and never explicitly closed --
    it is only used for synchronous object-to-dict serialization, so no
    aiohttp session is opened beyond the default one aiohttp lazily
    creates (which is cleaned up at process exit).
    """
    global _serializer
    if _serializer is None:
        from kubernetes_asyncio.client import ApiClient

        _serializer = ApiClient()
    return _serializer


# Container states that indicate problems
_PROBLEM_STATES: dict[str, tuple[str, str]] = {
    "CrashLoopBackOff": (
        "CRITICAL",
        "Container is crash-looping. Check logs for the root cause.",
    ),
    "ImagePullBackOff": (
        "CRITICAL",
        "Image cannot be pulled. Verify the image name and registry access.",
    ),
    "ErrImagePull": (
        "CRITICAL",
        "Image pull failed. Check image name, tag, and pull secrets.",
    ),
    "OOMKilled": (
        "CRITICAL",
        "Container was killed due to out-of-memory. Increase memory limits.",
    ),
    "CreateContainerConfigError": (
        "ERROR",
        "Container config error. Check ConfigMaps, Secrets, and volume mounts.",
    ),
    "RunContainerError": (
        "ERROR",
        "Failed to run container. Check security context and resource limits.",
    ),
}


def _pod_to_raw(pod: Any) -> tuple[str, dict[str, Any]]:
    """Normalize V1Pod or legacy ``.raw``-style mock to (name, raw_dict)."""
    raw = getattr(pod, "raw", None)
    if raw is not None:
        name = getattr(pod, "name", None) or raw.get("metadata", {}).get("name", "")
        return (name, raw)
    # Real V1Pod from kubernetes_asyncio -- serialize to dict shape.
    # Use a module-scoped serializer to avoid leaking an aiohttp session per
    # pod; sanitize_for_serialization only reads class-level attrs so the
    # singleton is safe to reuse and never needs closing.
    raw = _get_serializer().sanitize_for_serialization(pod) or {}
    name = raw.get("metadata", {}).get("name", "")
    return (name, raw)


def _waiting_problem(
    container_name: str, waiting: dict[str, Any], phase: str
) -> dict[str, str] | None:
    """Build a problem dict for a waiting container, or None if not a problem."""
    reason = waiting.get("reason", "")
    if reason in _PROBLEM_STATES:
        severity, suggestion = _PROBLEM_STATES[reason]
        return {
            "container": container_name,
            "state": reason,
            "severity": severity,
            "suggestion": suggestion,
            "message": waiting.get("message", ""),
        }
    if reason and phase == "Pending":
        return {
            "container": container_name,
            "state": reason,
            "severity": "WARNING",
            "suggestion": f"Container is waiting: {reason}",
            "message": waiting.get("message", ""),
        }
    return None


def _oom_problem(
    container_name: str, terminated: dict[str, Any], *, previous: bool
) -> dict[str, str] | None:
    """Build an OOMKilled problem dict for a terminated state, or None."""
    reason = terminated.get("reason", "")
    if reason != "OOMKilled":
        return None
    severity, suggestion = _PROBLEM_STATES["OOMKilled"]
    state_label = f"{reason} (previous)" if previous else reason
    return {
        "container": container_name,
        "state": state_label,
        "severity": severity,
        "suggestion": suggestion,
        "message": terminated.get("message", ""),
    }


def _container_problems(cs: dict[str, Any], phase: str) -> list[dict[str, str]]:
    """Return all problem dicts derived from a single container status."""
    container_name = cs.get("name", "unknown")
    problems: list[dict[str, str]] = []

    waiting = cs.get("state", {}).get("waiting", {})
    if waiting:
        problem = _waiting_problem(container_name, waiting, phase)
        if problem is not None:
            problems.append(problem)

    terminated = cs.get("state", {}).get("terminated", {})
    if terminated:
        problem = _oom_problem(container_name, terminated, previous=False)
        if problem is not None:
            problems.append(problem)

    last_terminated = cs.get("lastState", {}).get("terminated", {})
    if last_terminated:
        problem = _oom_problem(container_name, last_terminated, previous=True)
        if problem is not None:
            problems.append(problem)

    return problems


def _unschedulable_problem(conditions: list[dict[str, Any]]) -> dict[str, str] | None:
    """Return an Unschedulable problem dict if the pod is unschedulable."""
    for cond in conditions:
        if (
            cond.get("type") == "PodScheduled"
            and cond.get("status") == "False"
            and cond.get("reason") == "Unschedulable"
        ):
            return {
                "container": "-",
                "state": "Unschedulable",
                "severity": "CRITICAL",
                "suggestion": (
                    "Pod cannot be scheduled. "
                    "Check node resources, taints/tolerations, and node selectors."
                ),
                "message": cond.get("message", ""),
            }
    return None


def _extract_pod_info(pod: Any) -> dict[str, Any]:
    """Extract diagnostic info from a Pod object.

    Args:
        pod: V1Pod (kubernetes_asyncio) or any object exposing ``.name`` and
            a ``.raw`` dict (legacy test mocks).

    Returns:
        Dict with pod name, phase, conditions, container statuses, and problems.
    """
    pod_name, raw = _pod_to_raw(pod)
    status = raw.get("status", {})
    phase = status.get("phase", "Unknown")
    container_statuses = status.get("containerStatuses", [])
    init_container_statuses = status.get("initContainerStatuses", [])
    conditions = status.get("conditions", [])

    all_statuses = init_container_statuses + container_statuses
    problems: list[dict[str, str]] = []
    restarts = 0
    for cs in all_statuses:
        restarts += cs.get("restartCount", 0)
        problems.extend(_container_problems(cs, phase))

    unschedulable = _unschedulable_problem(conditions)
    if unschedulable is not None:
        problems.append(unschedulable)

    metadata = raw.get("metadata", {})
    return {
        "name": pod_name,
        "namespace": metadata.get("namespace", ""),
        "phase": phase,
        "restarts": restarts,
        "problems": problems,
        "container_statuses": all_statuses,
        "node": raw.get("spec", {}).get("nodeName", ""),
    }

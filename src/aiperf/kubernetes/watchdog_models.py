# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain models and parsers for the benchmark watchdog.

Pure data: no network, no I/O. Imported by both the watchdog loop
(``aiperf.kubernetes.watchdog``) and the kubernetes_asyncio data source
(``aiperf.kubernetes.watchdog_source``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kubernetes_asyncio.client.models import V1ContainerStatus


# ---------------------------------------------------------------------------
# Data models (hot-path, slots=True)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ContainerInfo:
    """Container status within a pod."""

    name: str
    """Container name from the pod spec."""

    ready: bool
    """Whether the container's readiness probe is passing."""

    state: str
    """Current state: 'running', 'waiting', or 'terminated'."""

    reason: str | None = None
    """Kubernetes reason string (e.g. 'CrashLoopBackOff', 'OOMKilled')."""

    message: str | None = None
    """Human-readable state message from the container runtime."""

    exit_code: int | None = None
    """Process exit code if the container has terminated."""


@dataclass(slots=True)
class WatchdogPodSnapshot:
    """Snapshot of a Kubernetes pod's status.

    Distinct from ``aiperf.controller.kubernetes_service_manager.PodInfo``
    (service-manager bookkeeping) and
    ``aiperf.kubernetes.watch_models.PodSnapshot`` (CLI TUI); this model
    is the watchdog's internal pod view.
    """

    name: str
    """Pod name from Kubernetes metadata."""

    namespace: str
    """Namespace the pod belongs to."""

    phase: str
    """Pod phase: Pending, Running, Succeeded, Failed, or Unknown."""

    ready: bool
    """Whether all containers are ready and the pod is Running."""

    restarts: int
    """Total restart count across all containers."""

    container_statuses: list[ContainerInfo]
    """Per-container status details."""

    creation_timestamp: datetime | None = None
    """When the pod was created in the cluster."""


@dataclass(slots=True)
class EventInfo:
    """Kubernetes event summary."""

    type: str
    """Event type: 'Normal' or 'Warning'."""

    reason: str
    """Short machine-readable reason string."""

    message: str
    """Human-readable event message."""

    involved_object: str
    """Name of the Kubernetes object this event relates to."""

    last_timestamp: datetime | None = None
    """Most recent timestamp for this event."""


@dataclass(slots=True)
class NodeResources:
    """Node allocatable resource summary."""

    name: str
    """Kubernetes node name."""

    allocatable_cpu: str
    """Allocatable CPU as a Kubernetes quantity string."""

    allocatable_memory: str
    """Allocatable memory as a Kubernetes quantity string."""

    allocatable_gpu: int
    """Number of allocatable NVIDIA GPUs."""


# ---------------------------------------------------------------------------
# Watchdog domain models
# ---------------------------------------------------------------------------


class ProblemSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class WatchdogProblem:
    """A detected problem in the benchmark deployment."""

    severity: ProblemSeverity
    """Problem severity level (INFO, WARNING, CRITICAL)."""

    category: str
    """Machine-readable problem category for grouping."""

    message: str
    """Human-readable description of the problem."""

    timestamp: float = field(default_factory=time.time)
    """Epoch timestamp when the problem was detected."""

    pod_name: str | None = None
    """Name of the affected pod, if applicable."""

    namespace: str | None = None
    """Namespace where the problem was observed."""

    suggestion: str | None = None
    """Recommended kubectl command or action to investigate."""


@dataclass
class PodTimeline:
    """Tracks a pod's phase transitions and durations."""

    name: str
    """Pod name from Kubernetes metadata."""

    role: str = ""
    """Pod role: 'controller', 'worker', or 'unknown'."""

    first_seen: float = field(default_factory=time.time)
    """Epoch timestamp when the pod was first observed."""

    last_phase: str = "Unknown"
    """Most recently observed pod phase."""

    phase_history: list[tuple[float, str]] = field(default_factory=list)
    """Ordered list of (timestamp, phase) transitions."""

    restart_count: int = 0
    """Current total restart count."""

    last_restart_count: int = 0
    """Previous restart count for detecting new restarts."""

    pending_warned: bool = False
    """Whether a Pending warning has been emitted."""

    pending_critical_warned: bool = False
    """Whether a critical Pending warning has been emitted."""

    crashloop_warned: bool = False
    """Whether a crash-loop warning has been emitted."""

    ready: bool = False
    """Whether the pod is currently ready."""


@dataclass
class WatchdogReport:
    """Structured output from a watchdog run."""

    namespace: str
    """Kubernetes namespace that was monitored."""

    duration: float
    """Total monitoring duration in seconds."""

    timeout: float | None
    """Configured timeout, or None if no timeout was set."""

    problems: list[WatchdogProblem]
    """All problems detected during the monitoring period."""

    pod_timelines: dict[str, PodTimeline]
    """Per-pod phase transition history keyed by pod name."""

    completed_pods: set[str]
    """Pod names that reached a terminal state."""

    node_cpu_pct: int | None = None
    """Cluster-wide CPU allocation percentage, if measured."""

    node_mem_pct: int | None = None
    """Cluster-wide memory allocation percentage, if measured."""

    stale_ns_count: int = 0
    """Number of stale aiperf-* namespaces detected."""

    @property
    def has_critical(self) -> bool:
        """Whether any CRITICAL problems were detected."""
        return any(p.severity == ProblemSeverity.CRITICAL for p in self.problems)

    @property
    def succeeded_count(self) -> int:
        """Number of pods that completed successfully."""
        return sum(
            1
            for tl in self.pod_timelines.values()
            if tl.last_phase in ("Succeeded", "Completed")
        )

    @property
    def failed_count(self) -> int:
        """Number of pods that failed."""
        return sum(1 for tl in self.pod_timelines.values() if tl.last_phase == "Failed")

    @property
    def total_restarts(self) -> int:
        """Total restart count across all pods."""
        return sum(tl.restart_count for tl in self.pod_timelines.values())


# ---------------------------------------------------------------------------
# Pod metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PodMetrics:
    """Pod resource usage from metrics-server."""

    name: str
    """Pod name from Kubernetes metadata."""

    cpu_millicores: int
    """Current CPU usage in millicores."""

    memory_mib: int
    """Current memory usage in MiB."""


# ---------------------------------------------------------------------------
# Data source protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WatchdogDataSource(Protocol):
    """Protocol for fetching Kubernetes data.

    Implementations can use kubernetes_asyncio, kubectl subprocess, or mocks.
    """

    async def get_pods(self, namespace: str) -> list[WatchdogPodSnapshot]: ...

    async def get_events(self, namespace: str, limit: int = 20) -> list[EventInfo]: ...

    async def get_node_resources(self) -> list[NodeResources]: ...

    async def get_namespaces(self, label_selector: str | None = None) -> list[str]: ...

    async def get_pod_logs(self, name: str, namespace: str, tail: int = 50) -> str: ...

    async def get_pod_metrics(self, namespace: str) -> list[PodMetrics]: ...


# ---------------------------------------------------------------------------
# Parsers (used by K8sWatchdogSource and tests)
# ---------------------------------------------------------------------------


def _parse_metrics_cpu(cpu_str: str) -> int:
    """Parse a metrics.k8s.io CPU usage quantity into millicores."""
    if cpu_str.endswith("n"):
        return int(cpu_str[:-1]) // 1_000_000
    if cpu_str.endswith("m"):
        return int(cpu_str[:-1])
    return int(float(cpu_str) * 1000)


def _parse_metrics_memory(mem_str: str) -> int:
    """Parse a metrics.k8s.io memory usage quantity into MiB."""
    if mem_str.endswith("Ki"):
        return int(mem_str[:-2]) // 1024
    if mem_str.endswith("Mi"):
        return int(mem_str[:-2])
    if mem_str.endswith("Gi"):
        return int(mem_str[:-2]) * 1024
    return int(mem_str) // (1024 * 1024)


def _metrics_item_to_pod_metrics(item: dict[str, Any]) -> PodMetrics:
    """Convert one metrics.k8s.io pod entry into a PodMetrics dataclass."""
    name = item.get("metadata", {}).get("name", "")
    total_cpu = 0
    total_mem = 0
    for container in item.get("containers", []):
        usage = container.get("usage", {})
        total_cpu += _parse_metrics_cpu(usage.get("cpu", "0"))
        total_mem += _parse_metrics_memory(usage.get("memory", "0"))
    return PodMetrics(name=name, cpu_millicores=total_cpu, memory_mib=total_mem)


def _state_from_container_status(
    cs: V1ContainerStatus,
) -> tuple[str, str | None, str | None, int | None]:
    """Extract (state, reason, message, exit_code) from a V1ContainerStatus.

    V1ContainerStatus.state is a V1ContainerState with optional ``running``,
    ``waiting``, and ``terminated`` sub-objects.
    """
    state = cs.state if cs and cs.state is not None else None
    if state is None:
        return "unknown", None, None, None
    if state.running is not None:
        return "running", None, None, None
    if state.waiting is not None:
        w = state.waiting
        return "waiting", w.reason, w.message, None
    if state.terminated is not None:
        t = state.terminated
        return "terminated", t.reason, t.message, t.exit_code
    return "unknown", None, None, None


def _parse_container_state(
    state_dict: dict[str, Any],
) -> tuple[str, str | None, str | None, int | None]:
    """Extract state, reason, message, exit_code from a container state dict."""
    if "running" in state_dict:
        return "running", None, None, None
    if "waiting" in state_dict:
        w = state_dict["waiting"]
        return "waiting", w.get("reason"), w.get("message"), None
    if "terminated" in state_dict:
        t = state_dict["terminated"]
        return (
            "terminated",
            t.get("reason"),
            t.get("message"),
            t.get("exitCode"),
        )
    return "unknown", None, None, None

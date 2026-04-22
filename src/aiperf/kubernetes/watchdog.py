# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production benchmark watchdog for Kubernetes deployments.

Autonomous monitoring agent that runs as a background task alongside
benchmark deployments. Continuously watches the cluster, reasons about
pod state, detects problems early, and returns structured findings.

Monitors:
- Pod phase transitions with precise timing and role identification
- Container state analysis (CrashLoopBackOff, OOMKilled, ImagePullBackOff)
- K8s event stream for scheduling failures, volume issues, backoff errors
- Node resource pressure (CPU/memory/GPU allocation)
- Timeout prediction with escalating severity warnings
- Stale namespace detection (leaked resources from previous runs)
- Container exit code analysis for non-zero exits
- Restart count tracking and crash loop detection

Usage with kubernetes_asyncio::

    async with k8s_client() as api:
        source = K8sWatchdogSource(api)
        async with BenchmarkWatchdog(source, namespace, timeout=300) as wd:
            # ... run benchmark ...
        report = wd.report

Usage with custom data source (e.g. for testing)::

    async with BenchmarkWatchdog(my_source, namespace) as wd:
        ...
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE

logger = AIPerfLogger(__name__)


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
class PodInfo:
    """Snapshot of a Kubernetes pod's status."""

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

    async def get_pods(self, namespace: str) -> list[PodInfo]: ...

    async def get_events(self, namespace: str, limit: int = 20) -> list[EventInfo]: ...

    async def get_node_resources(self) -> list[NodeResources]: ...

    async def get_namespaces(self, label_selector: str | None = None) -> list[str]: ...

    async def get_pod_logs(self, name: str, namespace: str, tail: int = 50) -> str: ...

    async def get_pod_metrics(self, namespace: str) -> list[PodMetrics]: ...


# ---------------------------------------------------------------------------
# kubernetes_asyncio implementation
# ---------------------------------------------------------------------------


class K8sWatchdogSource:
    """WatchdogDataSource backed by kubernetes_asyncio."""

    def __init__(self, api: ApiClient) -> None:
        self._api = api

    @classmethod
    async def create(
        cls,
        *,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
    ) -> K8sWatchdogSource:
        """Create a source with explicit context/kubeconfig.

        Loads k8s config (in-cluster first, kubeconfig fallback) and
        constructs an ``ApiClient``. Caller owns the ApiClient lifecycle;
        the watchdog itself does not close it.
        """
        from kubernetes_asyncio import config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            await config.load_kube_config(config_file=kubeconfig, context=kube_context)
        return cls(ApiClient())

    async def get_pods(self, namespace: str) -> list[PodInfo]:
        """List pods in a namespace via CoreV1Api."""
        core = client.CoreV1Api(self._api)
        pod_list = await core.list_namespaced_pod(namespace)
        return [self._pod_to_info(p) for p in pod_list.items]

    async def get_events(self, namespace: str, limit: int = 20) -> list[EventInfo]:
        """List recent events in a namespace via CoreV1Api."""
        core = client.CoreV1Api(self._api)
        ev_list = await core.list_namespaced_event(namespace)
        result: list[EventInfo] = []
        for ev in ev_list.items:
            involved = ev.involved_object
            obj_name = involved.name if involved and involved.name else ""
            result.append(
                EventInfo(
                    type=ev.type or "Normal",
                    reason=ev.reason or "",
                    message=ev.message or "",
                    involved_object=obj_name,
                    last_timestamp=ev.last_timestamp,
                )
            )
        result.sort(
            key=lambda e: e.last_timestamp or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return result[:limit]

    async def get_node_resources(self) -> list[NodeResources]:
        """List node allocatable resources via CoreV1Api."""
        core = client.CoreV1Api(self._api)
        node_list = await core.list_node()
        result: list[NodeResources] = []
        for node in node_list.items:
            allocatable = (
                node.status.allocatable
                if node.status and node.status.allocatable
                else {}
            )
            gpu_str = allocatable.get("nvidia.com/gpu", "0")
            try:
                gpu_count = int(gpu_str)
            except (ValueError, TypeError):
                gpu_count = 0

            name = node.metadata.name if node.metadata and node.metadata.name else ""
            result.append(
                NodeResources(
                    name=name,
                    allocatable_cpu=allocatable.get("cpu", "0"),
                    allocatable_memory=allocatable.get("memory", "0"),
                    allocatable_gpu=gpu_count,
                )
            )
        return result

    async def get_namespaces(self, label_selector: str | None = None) -> list[str]:
        """List namespace names via CoreV1Api."""
        core = client.CoreV1Api(self._api)
        kwargs: dict[str, Any] = {}
        if label_selector:
            kwargs["label_selector"] = label_selector
        ns_list = await core.list_namespace(**kwargs)
        return [
            ns.metadata.name if ns.metadata and ns.metadata.name else ""
            for ns in ns_list.items
        ]

    async def get_pod_logs(self, name: str, namespace: str, tail: int = 50) -> str:
        """Fetch pod logs via CoreV1Api."""
        core = client.CoreV1Api(self._api)
        try:
            return await core.read_namespaced_pod_log(
                name=name,
                namespace=namespace,
                tail_lines=tail,
            )
        except ApiException:
            return ""
        except Exception:
            return ""

    async def get_pod_metrics(self, namespace: str) -> list[PodMetrics]:
        """Fetch pod metrics via metrics.k8s.io API."""
        try:
            resp = await self._api.call_api(
                f"/apis/metrics.k8s.io/v1beta1/namespaces/{namespace}/pods",
                "GET",
                response_type="object",
                auth_settings=["BearerToken"],
                _return_http_data_only=True,
            )
            data = resp if isinstance(resp, dict) else {}
            result: list[PodMetrics] = []
            for item in data.get("items", []):
                name = item.get("metadata", {}).get("name", "")
                total_cpu = 0
                total_mem = 0
                for container in item.get("containers", []):
                    usage = container.get("usage", {})
                    cpu_str = usage.get("cpu", "0")
                    mem_str = usage.get("memory", "0")
                    if cpu_str.endswith("n"):
                        total_cpu += int(cpu_str[:-1]) // 1_000_000
                    elif cpu_str.endswith("m"):
                        total_cpu += int(cpu_str[:-1])
                    else:
                        total_cpu += int(float(cpu_str) * 1000)
                    if mem_str.endswith("Ki"):
                        total_mem += int(mem_str[:-2]) // 1024
                    elif mem_str.endswith("Mi"):
                        total_mem += int(mem_str[:-2])
                    elif mem_str.endswith("Gi"):
                        total_mem += int(mem_str[:-2]) * 1024
                    else:
                        total_mem += int(mem_str) // (1024 * 1024)
                result.append(
                    PodMetrics(
                        name=name, cpu_millicores=total_cpu, memory_mib=total_mem
                    )
                )
            return result
        except Exception:
            return []

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _pod_to_info(pod: Any) -> PodInfo:
        """Convert a V1Pod into a PodInfo dataclass."""
        status = pod.status
        phase = status.phase if status and status.phase else "Unknown"

        container_statuses: list[ContainerInfo] = []
        all_ready = True
        total_restarts = 0

        primary = (status.container_statuses if status else None) or []
        for cs in primary:
            total_restarts += cs.restart_count or 0
            c_ready = bool(cs.ready)
            if not c_ready:
                all_ready = False

            c_state, c_reason, c_message, c_exit = _state_from_container_status(cs)

            container_statuses.append(
                ContainerInfo(
                    name=cs.name or "",
                    ready=c_ready,
                    state=c_state,
                    reason=c_reason,
                    message=c_message,
                    exit_code=c_exit,
                )
            )

        init_statuses = (status.init_container_statuses if status else None) or []
        for cs in init_statuses:
            total_restarts += cs.restart_count or 0
            c_ready = bool(cs.ready)
            if not c_ready:
                all_ready = False

            c_state, c_reason, c_message, c_exit = _state_from_container_status(cs)

            container_statuses.append(
                ContainerInfo(
                    name=cs.name or "",
                    ready=c_ready,
                    state=c_state,
                    reason=c_reason,
                    message=c_message,
                    exit_code=c_exit,
                )
            )

        if not container_statuses:
            all_ready = False

        metadata = pod.metadata
        parsed_creation: datetime | None = None
        if metadata and metadata.creation_timestamp:
            ts = metadata.creation_timestamp
            if isinstance(ts, datetime):
                parsed_creation = ts
            else:
                with contextlib.suppress(ValueError, TypeError):
                    parsed_creation = datetime.fromisoformat(
                        str(ts).replace("Z", "+00:00")
                    )

        return PodInfo(
            name=(metadata.name if metadata and metadata.name else ""),
            namespace=(metadata.namespace if metadata and metadata.namespace else ""),
            phase=phase,
            ready=all_ready and phase == "Running",
            restarts=total_restarts,
            container_statuses=container_statuses,
            creation_timestamp=parsed_creation,
        )


def _state_from_container_status(
    cs: Any,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m{secs:02d}s"


def _pod_role(name: str) -> str:
    """Identify pod role from its name."""
    if "controller" in name:
        return "controller"
    if "worker" in name:
        return "worker"
    return "unknown"


def _phase_icon(phase: str) -> str:
    """Map phase to a compact visual indicator."""
    return {
        "Pending": "...",
        "Running": ">>>",
        "Succeeded": "[OK]",
        "Failed": "[!!]",
        "Unknown": "[??]",
        "Completed": "[OK]",
    }.get(phase, "   ")


# Box drawing for dashboards
_W = 72
_LINE = "+" + "-" * _W + "+"
_DLINE = "+" + "=" * _W + "+"


def _row(text: str) -> str:
    """Format a line inside the box, padded to fixed width."""
    return f"| {text:<{_W - 2}} |"


def _progress_bar(pct: float, width: int = 30) -> str:
    """Render a text progress bar."""
    filled = int(width * min(pct, 100) / 100)
    empty = width - filled
    bar = "#" * filled + "-" * empty
    return f"[{bar}] {pct:.0f}%"


def _short_pod_name(name: str, max_len: int = 38) -> str:
    """Shorten pod name, keeping the unique suffix visible."""
    if len(name) <= max_len:
        return name
    return "..." + name[-(max_len - 3) :]


# ---------------------------------------------------------------------------
# BenchmarkWatchdog
# ---------------------------------------------------------------------------


class BenchmarkWatchdog:
    """Autonomous monitoring agent for benchmark deployments.

    Runs as a background async task that periodically:
    1. Tracks pod phase transitions with precise timing
    2. Detects CrashLoopBackOff, OOMKilled, ImagePullBackOff
    3. Monitors K8s events for scheduling failures
    4. Tracks time in Pending with escalating warnings
    5. Monitors restart counts and crash loops
    6. Checks node resource allocation
    7. Predicts timeouts with escalating urgency
    8. Detects stale namespaces from previous runs
    9. Analyzes container exit codes

    Usage::

        async with k8s_client() as api:
            source = K8sWatchdogSource(api)
            async with BenchmarkWatchdog(source, "my-ns", timeout=300) as wd:
                ...
            report = wd.report
    """

    def __init__(
        self,
        source: WatchdogDataSource,
        namespace: str,
        timeout: float | None = None,
        poll_interval: float = 5.0,
        status_interval: float = 10.0,
        pending_threshold: float = 30.0,
        pending_critical_threshold: float = 90.0,
        crashloop_threshold: int = 2,
        log: Any | None = None,
    ) -> None:
        self._log = log or logger
        self._source = source
        self.namespace = namespace
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.status_interval = status_interval
        self.pending_threshold = pending_threshold
        self.pending_critical_threshold = pending_critical_threshold
        self.crashloop_threshold = crashloop_threshold

        self._task: asyncio.Task[None] | None = None
        self._problems: list[WatchdogProblem] = []
        self._pod_timelines: dict[str, PodTimeline] = {}
        self._start_time: float = 0.0
        self._event_fingerprints: set[str] = set()
        self._stopped = False
        self._tick_count: int = 0
        self._last_status_time: float = 0.0
        self._last_pod_snapshot: list[PodInfo] = []
        self._completed_pods: set[str] = set()
        self._event_check_interval: int = 3  # Check events every Nth tick
        self._resource_check_interval: int = (
            6  # Check pod resources every Nth tick (~30s)
        )
        self._peak_memory: dict[str, int] = {}
        self._node_check_done: bool = False
        self._stale_ns_check_done: bool = False
        self._node_cpu_pct: int | None = None
        self._node_mem_pct: int | None = None
        self._stale_ns_count: int = 0

    async def __aenter__(self) -> BenchmarkWatchdog:
        self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def start(self) -> None:
        """Start the watchdog background task."""
        self._start_time = time.time()
        self._last_status_time = self._start_time
        self._stopped = False
        timeout_str = f"{self.timeout}s" if self.timeout else "none"
        self._log.info(
            f"[WATCHDOG] Monitoring started for {self.namespace} "
            f"| timeout={timeout_str} | poll={self.poll_interval}s "
            f"| status_interval={self.status_interval}s"
        )
        self._task = asyncio.create_task(
            self._monitor_loop(),
            name=f"watchdog-{self.namespace}",
        )

    async def stop(self) -> None:
        """Stop the watchdog and log final report."""
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._log_final_report()

    @property
    def problems(self) -> list[WatchdogProblem]:
        """All problems detected so far."""
        return list(self._problems)

    @property
    def has_critical(self) -> bool:
        """Whether any CRITICAL problems have been detected."""
        return any(p.severity == ProblemSeverity.CRITICAL for p in self._problems)

    @property
    def report(self) -> WatchdogReport:
        """Build a structured report of watchdog findings."""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        return WatchdogReport(
            namespace=self.namespace,
            duration=elapsed,
            timeout=self.timeout,
            problems=list(self._problems),
            pod_timelines=dict(self._pod_timelines),
            completed_pods=set(self._completed_pods),
            node_cpu_pct=self._node_cpu_pct,
            node_mem_pct=self._node_mem_pct,
            stale_ns_count=self._stale_ns_count,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while not self._stopped:
                self._tick_count += 1
                try:
                    pods = await self._fetch_pods()
                    if pods is not None:
                        self._last_pod_snapshot = pods
                        self._analyze_pods(pods)

                        now = time.time()
                        if now - self._last_status_time >= self.status_interval:
                            self._log_status_dashboard(pods)
                            self._last_status_time = now

                    if self._tick_count % self._event_check_interval == 0:
                        await self._check_events()
                    self._check_elapsed_time()

                    if not self._node_check_done and self._tick_count == 2:
                        await self._check_node_resources()
                        self._node_check_done = True

                    if not self._stale_ns_check_done and self._tick_count == 3:
                        await self._check_stale_namespaces()
                        self._stale_ns_check_done = True

                    if (
                        self._tick_count % self._resource_check_interval == 0
                        and self._tick_count > self._resource_check_interval
                    ):
                        await self._check_pod_resources()

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._log.debug(lambda e=e: f"[WATCHDOG] Monitor error: {e}")
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass

    async def _fetch_pods(self) -> list[PodInfo] | None:
        """Fetch pods, returning None on failure."""
        try:
            return await self._source.get_pods(self.namespace)
        except Exception as e:
            self._log.debug(lambda e=e: f"[WATCHDOG] Failed to fetch pods: {e}")
            return None

    # ------------------------------------------------------------------
    # Pod analysis
    # ------------------------------------------------------------------

    def _analyze_pods(self, pods: list[PodInfo]) -> None:
        """Run all pod checks."""
        for pod in pods:
            tl = self._get_or_create_timeline(pod)
            self._track_phase_transition(pod, tl)
            self._check_pending_too_long(pod, tl)
            self._check_crash_loop(pod, tl)
            self._check_container_states(pod)
            self._check_pod_completion(pod)
            tl.ready = pod.ready

    def _get_or_create_timeline(self, pod: PodInfo) -> PodTimeline:
        """Get existing timeline or create a new one for a pod."""
        if pod.name not in self._pod_timelines:
            self._pod_timelines[pod.name] = PodTimeline(
                name=pod.name,
                role=_pod_role(pod.name),
            )
        return self._pod_timelines[pod.name]

    def _track_phase_transition(self, pod: PodInfo, tl: PodTimeline) -> None:
        """Record phase changes with timing."""
        if pod.phase == tl.last_phase:
            return

        old_phase = tl.last_phase
        now = time.time()
        tl.phase_history.append((now, pod.phase))
        elapsed = now - self._start_time
        time_in_old = 0.0
        if len(tl.phase_history) >= 2:
            time_in_old = tl.phase_history[-1][0] - tl.phase_history[-2][0]

        short = pod.name.split("-")[-1] if "-" in pod.name else pod.name

        self._log.info(
            lambda: f"[WATCHDOG] {tl.role}({short}): "
            f"{old_phase} -> {pod.phase}  "
            f"(in {old_phase} for {_fmt_duration(time_in_old)}, "
            f"total +{_fmt_duration(elapsed)})"
        )

        tl.last_phase = pod.phase

    def _check_pending_too_long(self, pod: PodInfo, tl: PodTimeline) -> None:
        """Escalating warnings for pods stuck in Pending."""
        if pod.phase != "Pending":
            return

        pending_duration = time.time() - tl.first_seen

        if (
            pending_duration > self.pending_critical_threshold
            and not tl.pending_critical_warned
        ):
            tl.pending_critical_warned = True
            self._add_problem(
                ProblemSeverity.CRITICAL,
                "pod-pending-critical",
                f"Pod {pod.name} stuck Pending for "
                f"{_fmt_duration(pending_duration)}! "
                f"Likely resource exhaustion or scheduling constraint.",
                pod_name=pod.name,
                suggestion=(
                    f"1) kubectl describe pod -n {self.namespace} {pod.name} "
                    f"| tail -20\n"
                    f"  2) kubectl get ns | grep aiperf | wc -l  "
                    f"(check stale namespaces)\n"
                    f"  3) kubectl describe node | grep -A 10 'Allocated'"
                ),
            )
        elif pending_duration > self.pending_threshold and not tl.pending_warned:
            tl.pending_warned = True
            self._add_problem(
                ProblemSeverity.WARNING,
                "pod-pending",
                f"Pod {pod.name} Pending for "
                f"{_fmt_duration(pending_duration)} "
                f"(threshold: {_fmt_duration(self.pending_threshold)}). "
                f"May be waiting for resources.",
                pod_name=pod.name,
                suggestion=(
                    f"kubectl describe pod -n {self.namespace} {pod.name} | tail -20"
                ),
            )

    def _check_crash_loop(self, pod: PodInfo, tl: PodTimeline) -> None:
        """Detect restart count increases and crash loops."""
        if pod.restarts <= tl.last_restart_count:
            tl.last_restart_count = pod.restarts
            return

        old_count = tl.last_restart_count
        tl.last_restart_count = pod.restarts
        tl.restart_count = pod.restarts
        self._log.info(
            lambda: f"[WATCHDOG] Restart detected: "
            f"{tl.role}({pod.name.split('-')[-1]}) "
            f"restarts {old_count} -> {pod.restarts}"
        )

        if pod.restarts >= self.crashloop_threshold and not tl.crashloop_warned:
            tl.crashloop_warned = True
            self._add_problem(
                ProblemSeverity.CRITICAL,
                "crash-loop",
                f"Pod {pod.name} restarted {pod.restarts}x - likely CrashLoopBackOff.",
                pod_name=pod.name,
                suggestion=(f"kubectl -n {self.namespace} logs {pod.name} --previous"),
            )

    def _check_container_states(self, pod: PodInfo) -> None:
        """Detect problematic container states."""
        for c in pod.container_statuses:
            if c.state == "waiting" and c.reason in (
                "CrashLoopBackOff",
                "ImagePullBackOff",
                "ErrImagePull",
                "ErrImageNeverPull",
                "CreateContainerConfigError",
                "InvalidImageName",
            ):
                fp = f"{pod.name}/{c.name}/{c.reason}"
                if fp not in self._event_fingerprints:
                    self._event_fingerprints.add(fp)
                    msg_detail = (c.message or "N/A")[:100]
                    hint = ""
                    if c.reason in (
                        "ImagePullBackOff",
                        "ErrImagePull",
                        "ErrImageNeverPull",
                    ):
                        hint = (
                            " -- For locally built images, use: "
                            "--image-pull-policy IfNotPresent "
                            "(or imagePullPolicy: IfNotPresent in YAML)"
                        )
                    self._add_problem(
                        ProblemSeverity.CRITICAL,
                        f"container-{c.reason.lower()}",
                        f"{c.name} in {pod.name}: {c.reason} - {msg_detail}{hint}",
                        pod_name=pod.name,
                    )

            if c.state == "terminated" and c.reason == "OOMKilled":
                fp = f"{pod.name}/{c.name}/OOMKilled"
                if fp not in self._event_fingerprints:
                    self._event_fingerprints.add(fp)
                    self._add_problem(
                        ProblemSeverity.CRITICAL,
                        "oom-killed",
                        f"{c.name} in {pod.name}: OOMKilled. "
                        f"Process exceeded memory limits.",
                        pod_name=pod.name,
                        suggestion="Increase memory limits in benchmark config.",
                    )

            elif (
                c.state == "terminated"
                and c.exit_code == 137
                and c.reason != "OOMKilled"
            ):
                fp = f"{pod.name}/{c.name}/sigkill-137"
                if fp not in self._event_fingerprints:
                    self._event_fingerprints.add(fp)
                    self._add_problem(
                        ProblemSeverity.WARNING,
                        "sigkill",
                        f"{c.name} in {pod.name}: killed by SIGKILL (exit 137). "
                        f"May be pod eviction due to node memory pressure.",
                        pod_name=pod.name,
                        suggestion=(
                            f"1) kubectl describe pod -n {self.namespace} {pod.name}"
                            f" | grep -A5 'Status'\n"
                            f"  2) kubectl get events -n {self.namespace}"
                            f" --field-selector involvedObject.name={pod.name}"
                            f" | grep Evict\n"
                            f"  3) kubectl describe node | grep -A5 'Conditions'"
                        ),
                    )

            elif (
                c.state == "terminated" and c.exit_code is not None and c.exit_code != 0
            ):
                fp = f"{pod.name}/{c.name}/exit-{c.exit_code}"
                if fp not in self._event_fingerprints:
                    self._event_fingerprints.add(fp)
                    self._log.warning(
                        f"[WATCHDOG] {c.name} in {pod.name}: "
                        f"exit code {c.exit_code} "
                        f"(reason: {c.reason or 'Completed'})"
                    )

    def _check_pod_completion(self, pod: PodInfo) -> None:
        """Log when a pod reaches terminal state for the first time."""
        if pod.name in self._completed_pods:
            return
        if pod.phase not in ("Succeeded", "Failed"):
            return

        self._completed_pods.add(pod.name)
        elapsed = time.time() - self._start_time
        tl = self._pod_timelines.get(pod.name)
        pod_age = (time.time() - tl.first_seen) if tl else 0
        role = tl.role if tl else _pod_role(pod.name)
        short = pod.name.split("-")[-1] if "-" in pod.name else pod.name

        if pod.phase == "Succeeded":
            self._log.info(
                f"[WATCHDOG] {role}({short}) completed successfully "
                f"(age={_fmt_duration(pod_age)}, +{_fmt_duration(elapsed)})"
            )
        else:
            self._add_problem(
                ProblemSeverity.CRITICAL,
                "pod-failed",
                f"Pod {pod.name} FAILED after {_fmt_duration(pod_age)}.",
                pod_name=pod.name,
                suggestion=(
                    f"kubectl -n {self.namespace} logs {pod.name} --all-containers"
                ),
            )
            with contextlib.suppress(RuntimeError):
                asyncio.create_task(self._fetch_failure_logs(pod.name))

    async def _fetch_failure_logs(self, pod_name: str) -> None:
        """Best-effort fetch of logs from a failed pod."""
        try:
            logs = await self._source.get_pod_logs(pod_name, self.namespace, tail=20)
            if logs.strip():
                self._log.error(
                    lambda logs=logs, pod_name=pod_name: (
                        f"[WATCHDOG] Last logs from {pod_name}:\n{logs}"
                    )
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Event analysis
    # ------------------------------------------------------------------

    async def _check_events(self) -> None:
        """Watch K8s events for scheduling/resource problems."""
        try:
            events = await self._source.get_events(self.namespace)
            for event in events:
                self._process_event(event)
        except Exception:
            pass

    def _process_event(self, event: EventInfo) -> None:
        """Classify a single event and record problems."""
        fp = f"{event.type}/{event.involved_object}/{event.reason}/{event.message[:80]}"

        if fp in self._event_fingerprints:
            return

        if event.reason == "FailedScheduling":
            self._event_fingerprints.add(fp)
            severity = (
                ProblemSeverity.CRITICAL
                if "Insufficient" in event.message
                else ProblemSeverity.WARNING
            )
            self._add_problem(
                severity,
                "scheduling-failure",
                f"{event.involved_object}: FailedScheduling - {event.message[:120]}",
                pod_name=event.involved_object,
                suggestion=(
                    "kubectl get ns | grep aiperf- | wc -l  "
                    "(clean up stale namespaces if > 5)"
                ),
            )

        elif event.reason in ("FailedMount", "FailedAttachVolume"):
            self._event_fingerprints.add(fp)
            self._add_problem(
                ProblemSeverity.WARNING,
                "volume-issue",
                f"{event.involved_object}: {event.reason} - {event.message[:120]}",
                pod_name=event.involved_object,
            )

        elif event.reason == "BackOff" and event.type == "Warning":
            self._event_fingerprints.add(fp)
            self._add_problem(
                ProblemSeverity.WARNING,
                "container-backoff",
                f"{event.involved_object}: BackOff - {event.message[:120]}",
                pod_name=event.involved_object,
            )

        elif event.reason == "Evicted":
            self._event_fingerprints.add(fp)
            self._add_problem(
                ProblemSeverity.CRITICAL,
                "pod-evicted",
                f"{event.involved_object}: Evicted - {event.message[:120]}",
                pod_name=event.involved_object,
                suggestion=(
                    "Node under memory/disk pressure. "
                    "Reduce worker count or pod memory limits."
                ),
            )

        elif event.reason == "Killing":
            self._event_fingerprints.add(fp)
            self._log.info(
                f"[WATCHDOG] Event: {event.involved_object} being killed "
                f"- {event.message[:100]}"
            )

        elif event.reason == "Unhealthy" and event.type == "Warning":
            self._event_fingerprints.add(fp)
            short = (
                event.involved_object.split("-")[-1]
                if "-" in event.involved_object
                else event.involved_object
            )
            self._log.debug(
                lambda short=short, msg=event.message: (
                    f"[WATCHDOG] Probe failure on {short}: {msg[:80]}"
                )
            )

    # ------------------------------------------------------------------
    # Cluster-level checks
    # ------------------------------------------------------------------

    def _check_elapsed_time(self) -> None:
        """Escalating timeout warnings (only when timeout is set)."""
        if self.timeout is None:
            return
        elapsed = time.time() - self._start_time
        remaining = self.timeout - elapsed

        if remaining < 60 and not any(
            p.category == "timeout-warning" for p in self._problems
        ):
            self._add_problem(
                ProblemSeverity.WARNING,
                "timeout-warning",
                f"<60s remaining ({_fmt_duration(remaining)} of "
                f"{_fmt_duration(self.timeout)}). Should be completing.",
            )

        if remaining < 15 and not any(
            p.category == "timeout-imminent" for p in self._problems
        ):
            self._add_problem(
                ProblemSeverity.CRITICAL,
                "timeout-imminent",
                f"TIMEOUT IMMINENT: {remaining:.0f}s left! Will be killed.",
            )

    async def _check_node_resources(self) -> None:
        """Check node resource allocation levels."""
        try:
            nodes = await self._source.get_node_resources()
            if not nodes:
                return

            total_gpu = 0
            for node in nodes:
                total_gpu += node.allocatable_gpu

            if total_gpu > 0:
                self._log.info(
                    f"[WATCHDOG] Cluster GPUs: {total_gpu} allocatable "
                    f"across {len(nodes)} node(s)"
                )
        except Exception:
            pass

    async def _check_stale_namespaces(self) -> None:
        """Detect leftover aiperf-* namespaces from previous runs."""
        try:
            all_ns = await self._source.get_namespaces()

            excluded = {self.namespace, "aiperf-system", DEFAULT_BENCHMARK_NAMESPACE}
            stale = [
                ns for ns in all_ns if ns.startswith("aiperf-") and ns not in excluded
            ]
            self._stale_ns_count = len(stale)

            if len(stale) > 2:
                self._add_problem(
                    ProblemSeverity.WARNING,
                    "stale-namespaces",
                    f"Found {len(stale)} stale aiperf-* namespaces. "
                    f"These consume cluster resources.",
                    suggestion=(
                        "Clean up with: aiperf kube cleanup --all\n"
                        "  Or manually: kubectl get ns -o name | grep aiperf- | "
                        "xargs kubectl delete --wait=false"
                    ),
                )
            elif stale:
                self._log.info(
                    f"[WATCHDOG] Found {len(stale)} other aiperf-* "
                    f"namespace(s) (within normal range)"
                )
            else:
                self._log.info("[WATCHDOG] Cluster clean - no stale namespaces")
        except Exception:
            pass

    async def _check_pod_resources(self) -> None:
        """Check pod resource usage and warn on high memory."""
        try:
            metrics = await self._source.get_pod_metrics(self.namespace)
            for pm in metrics:
                prev_peak = self._peak_memory.get(pm.name, 0)
                if pm.memory_mib > prev_peak:
                    self._peak_memory[pm.name] = pm.memory_mib

                if prev_peak > 0 and pm.memory_mib > prev_peak * 1.2:
                    fp = f"{pm.name}/memory-growth/{pm.memory_mib // 100}"
                    if fp not in self._event_fingerprints:
                        self._event_fingerprints.add(fp)
                        self._add_problem(
                            ProblemSeverity.WARNING,
                            "memory-growth",
                            f"Pod {pm.name} memory growing: {pm.memory_mib}Mi "
                            f"(was {prev_peak}Mi peak)",
                            pod_name=pm.name,
                            suggestion="Check for memory leaks. Consider increasing memory limits.",
                        )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Status dashboard (structured log output)
    # ------------------------------------------------------------------

    def _log_status_dashboard(self, pods: list[PodInfo]) -> None:
        """Log a formatted status dashboard."""
        elapsed = time.time() - self._start_time

        lines = [
            _DLINE,
            _row(f"WATCHDOG  |  ns={self.namespace}"),
        ]

        if self.timeout is not None:
            remaining = max(0, self.timeout - elapsed)
            pct = min(100, (elapsed / self.timeout) * 100) if self.timeout > 0 else 0
            lines.append(
                _row(
                    f"time: {_fmt_duration(elapsed)} elapsed, "
                    f"{_fmt_duration(remaining)} remaining  "
                    f"{_progress_bar(pct, 20)}"
                )
            )
        else:
            lines.append(_row(f"time: {_fmt_duration(elapsed)} elapsed"))

        if self._node_cpu_pct is not None:
            node_line = f"node: cpu={self._node_cpu_pct}%"
            if self._node_mem_pct is not None:
                node_line += f"  mem={self._node_mem_pct}%"
            if self._stale_ns_count > 0:
                node_line += f"  stale_ns={self._stale_ns_count}"
            lines.append(_row(node_line))

        lines.append(_row("-" * (_W - 2)))
        lines.append(
            _row(f"{'':>4} {'POD':<36} {'PHASE':<12} {'RDY':<5} {'RST':>3} {'AGE':>6}")
        )
        lines.append(_row("-" * (_W - 2)))

        for pod in pods:
            tl = self._pod_timelines.get(pod.name)
            age = (time.time() - tl.first_seen) if tl else 0
            icon = _phase_icon(pod.phase)
            short = _short_pod_name(pod.name, 36)
            ready_str = "Y" if pod.ready else "N"

            lines.append(
                _row(
                    f"{icon} {short:<36} {pod.phase:<12} {ready_str:<5} "
                    f"{pod.restarts:>3} {_fmt_duration(age):>6}"
                )
            )

        crits = sum(1 for p in self._problems if p.severity == ProblemSeverity.CRITICAL)
        warns = sum(1 for p in self._problems if p.severity == ProblemSeverity.WARNING)
        if crits or warns:
            lines.append(_row("-" * (_W - 2)))
            parts = []
            if crits:
                parts.append(f"{crits} CRITICAL")
            if warns:
                parts.append(f"{warns} WARNING")
            lines.append(_row(f"issues: {', '.join(parts)}"))

        lines.append(_DLINE)
        self._log.info(lambda: "[WATCHDOG]\n" + "\n".join(lines))

    # ------------------------------------------------------------------
    # Problem tracking
    # ------------------------------------------------------------------

    def _add_problem(
        self,
        severity: ProblemSeverity,
        category: str,
        message: str,
        pod_name: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Record a problem and log it."""
        problem = WatchdogProblem(
            severity=severity,
            category=category,
            message=message,
            pod_name=pod_name,
            namespace=self.namespace,
            suggestion=suggestion,
        )
        self._problems.append(problem)

        if severity == ProblemSeverity.CRITICAL:
            self._log.error(f"[WATCHDOG:CRITICAL] {message}")
        elif severity == ProblemSeverity.WARNING:
            self._log.warning(f"[WATCHDOG:WARNING] {message}")
        else:
            self._log.info(f"[WATCHDOG:INFO] {message}")

        if suggestion:
            self._log.info(f"[WATCHDOG]  -> {suggestion}")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    def _log_final_report(self) -> None:
        """Log a comprehensive final watchdog report."""
        elapsed = time.time() - self._start_time
        total_pods = len(self._pod_timelines)
        succeeded = sum(
            1
            for tl in self._pod_timelines.values()
            if tl.last_phase in ("Succeeded", "Completed")
        )
        failed = sum(
            1 for tl in self._pod_timelines.values() if tl.last_phase == "Failed"
        )
        total_restarts = sum(tl.restart_count for tl in self._pod_timelines.values())

        lines = [
            "",
            _DLINE,
            _row("WATCHDOG FINAL REPORT"),
            _row(f"Namespace:  {self.namespace}"),
            _row(
                f"Duration:   {_fmt_duration(elapsed)}"
                + (
                    f" (timeout was {_fmt_duration(self.timeout)})"
                    if self.timeout is not None
                    else ""
                )
            ),
            _row(
                f"Pods:       {total_pods} tracked, "
                f"{succeeded} succeeded, {failed} failed, "
                f"{total_restarts} total restarts"
            ),
        ]

        if self._node_cpu_pct is not None:
            lines.append(
                _row(
                    f"Node:       CPU {self._node_cpu_pct}%, "
                    f"Memory {self._node_mem_pct or '?'}%"
                )
            )

        if self._pod_timelines:
            lines.append(_row("-" * (_W - 2)))
            lines.append(_row("POD LIFECYCLE:"))
            for tl in self._pod_timelines.values():
                short = tl.name.split("-")[-1] if "-" in tl.name else tl.name
                phase_times = []
                for i, (ts, phase) in enumerate(tl.phase_history):
                    if i + 1 < len(tl.phase_history):
                        dt = tl.phase_history[i + 1][0] - ts
                        phase_times.append(f"{phase}({_fmt_duration(dt)})")
                    else:
                        phase_times.append(phase)

                timing = " -> ".join(phase_times)
                rst_str = f" [{tl.restart_count}rst]" if tl.restart_count else ""
                lines.append(_row(f"  {tl.role[:4]}({short}): {timing}{rst_str}"))

        crits = [p for p in self._problems if p.severity == ProblemSeverity.CRITICAL]
        warns = [p for p in self._problems if p.severity == ProblemSeverity.WARNING]

        lines.append(_row("-" * (_W - 2)))
        if crits or warns:
            lines.append(_row(f"ISSUES: {len(crits)} critical, {len(warns)} warnings"))
            for p in crits:
                msg = p.message[: (_W - 14)]
                lines.append(_row(f"  [CRIT] {msg}"))
            for p in warns:
                msg = p.message[: (_W - 14)]
                lines.append(_row(f"  [WARN] {msg}"))
        else:
            lines.append(_row("STATUS: Clean run - no problems detected"))

        lines.append(_DLINE)
        self._log.info(lambda: "[WATCHDOG]" + "\n".join(lines))

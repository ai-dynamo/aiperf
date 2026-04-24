# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production benchmark watchdog for Kubernetes deployments.

Autonomous monitoring agent that runs as a background task alongside
benchmark deployments. Continuously watches the cluster, reasons about
pod state, detects problems early, and returns structured findings.

This module is the production in-cluster monitor invoked by the operator;
the sibling module ``aiperf.kubernetes.watch_models`` has parallel domain
models used by the ``aiperf kube watch`` CLI TUI.

Data models, event/pod-check helpers, rendering, and the kubernetes_asyncio
data source live in sibling modules (``watchdog_models``, ``watchdog_events``,
``watchdog_pod_checks``, ``watchdog_render``, ``watchdog_source``); their
public names are re-exported here for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import time

from kubernetes_asyncio import client  # noqa: F401 - re-exported for test patching

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.kubernetes.constants import DEFAULT_BENCHMARK_NAMESPACE
from aiperf.kubernetes.watchdog_events import classify_container_states, classify_event
from aiperf.kubernetes.watchdog_models import (
    ContainerInfo,
    EventInfo,
    NodeResources,
    PodMetrics,
    PodTimeline,
    ProblemSeverity,
    WatchdogDataSource,
    WatchdogPodSnapshot,
    WatchdogProblem,
    WatchdogReport,
    _metrics_item_to_pod_metrics,
    _parse_container_state,
    _parse_metrics_cpu,
    _parse_metrics_memory,
    _state_from_container_status,
)
from aiperf.kubernetes.watchdog_pod_checks import (
    check_crash_loop,
    check_pending_too_long,
    check_pod_completion,
    track_phase_transition,
)
from aiperf.kubernetes.watchdog_render import (
    _fmt_duration,
    _phase_icon,
    _short_pod_name,
    render_final_report,
    render_status_dashboard,
)
from aiperf.kubernetes.watchdog_source import K8sWatchdogSource

__all__ = [
    "BenchmarkWatchdog",
    "ContainerInfo",
    "EventInfo",
    "K8sWatchdogSource",
    "NodeResources",
    "PodMetrics",
    "PodTimeline",
    "ProblemSeverity",
    "WatchdogDataSource",
    "WatchdogPodSnapshot",
    "WatchdogProblem",
    "WatchdogReport",
]

logger = AIPerfLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pod_role(name: str) -> str:
    """Identify pod role from its name."""
    if "controller" in name:
        return "controller"
    if "worker" in name:
        return "worker"
    return "unknown"


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
        *,
        timeout: float | None = None,
        poll_interval: float = 5.0,
        status_interval: float = 10.0,
        pending_threshold: float = 30.0,
        pending_critical_threshold: float = 90.0,
        crashloop_threshold: int = 2,
        log: AIPerfLogger | None = None,
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
        self._last_pod_snapshot: list[WatchdogPodSnapshot] = []
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

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while not self._stopped:
                self._tick_count += 1
                try:
                    await self._run_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - watchdog must never die on a single-check failure
                    self._log.debug(lambda e=e: f"[WATCHDOG] Monitor error: {e}")
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass

    async def _run_tick(self) -> None:
        """Execute the checks scheduled for the current tick."""
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

    async def _fetch_pods(self) -> list[WatchdogPodSnapshot] | None:
        """Fetch pods, returning None on failure."""
        try:
            return await self._source.get_pods(self.namespace)
        except Exception as e:  # noqa: BLE001 - watchdog must never die on a single-check failure
            self._log.debug(lambda e=e: f"[WATCHDOG] Failed to fetch pods: {e}")
            return None

    def _analyze_pods(self, pods: list[WatchdogPodSnapshot]) -> None:
        """Run all pod checks."""
        for pod in pods:
            tl = self._get_or_create_timeline(pod)
            track_phase_transition(self, pod, tl)
            check_pending_too_long(self, pod, tl)
            check_crash_loop(self, pod, tl)
            self._check_container_states(pod)
            check_pod_completion(self, pod, _pod_role)
            tl.ready = pod.ready

    def _get_or_create_timeline(self, pod: WatchdogPodSnapshot) -> PodTimeline:
        """Get existing timeline or create a new one for a pod."""
        if pod.name not in self._pod_timelines:
            self._pod_timelines[pod.name] = PodTimeline(
                name=pod.name,
                role=_pod_role(pod.name),
            )
        return self._pod_timelines[pod.name]

    def _check_container_states(self, pod: WatchdogPodSnapshot) -> None:
        """Detect problematic container states."""
        classify_container_states(
            pod,
            namespace=self.namespace,
            seen_fingerprints=self._event_fingerprints,
            recorder=self._add_problem,
            log=self._log,
        )

    async def _check_events(self) -> None:
        """Watch K8s events for scheduling/resource problems."""
        try:
            events = await self._source.get_events(self.namespace)
            for event in events:
                self._process_event(event)
        except Exception:  # noqa: BLE001, S110 - watchdog must never die on a single-check failure
            pass

    def _process_event(self, event: EventInfo) -> None:
        """Classify a single event and record problems."""
        fp = f"{event.type}/{event.involved_object}/{event.reason}/{event.message[:80]}"
        if fp in self._event_fingerprints:
            return
        if classify_event(event, self._add_problem, self._log):
            self._event_fingerprints.add(fp)

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
        except Exception:  # noqa: BLE001, S110 - watchdog must never die on a single-check failure
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
        except Exception:  # noqa: BLE001, S110 - watchdog must never die on a single-check failure
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
        except Exception:  # noqa: BLE001, S110 - watchdog must never die on a single-check failure
            pass

    def _log_status_dashboard(self, pods: list[WatchdogPodSnapshot]) -> None:
        """Log a formatted status dashboard."""
        text = render_status_dashboard(
            namespace=self.namespace,
            start_time=self._start_time,
            timeout=self.timeout,
            node_cpu_pct=self._node_cpu_pct,
            node_mem_pct=self._node_mem_pct,
            stale_ns_count=self._stale_ns_count,
            pods=pods,
            pod_timelines=self._pod_timelines,
            problems=self._problems,
        )
        self._log.info(lambda: text)

    def _add_problem(
        self,
        severity: ProblemSeverity,
        category: str,
        message: str,
        *,
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

    def _log_final_report(self) -> None:
        """Log a comprehensive final watchdog report."""
        text = render_final_report(
            namespace=self.namespace,
            start_time=self._start_time,
            timeout=self.timeout,
            node_cpu_pct=self._node_cpu_pct,
            node_mem_pct=self._node_mem_pct,
            pod_timelines=self._pod_timelines,
            problems=self._problems,
        )
        self._log.info(lambda: text)


# Keep the private helpers re-exportable from this module so tests/other
# call sites that imported them from ``aiperf.kubernetes.watchdog`` continue
# to find them without change.
__all__ += [
    "_fmt_duration",
    "_metrics_item_to_pod_metrics",
    "_parse_container_state",
    "_parse_metrics_cpu",
    "_parse_metrics_memory",
    "_phase_icon",
    "_pod_role",
    "_short_pod_name",
    "_state_from_container_status",
]

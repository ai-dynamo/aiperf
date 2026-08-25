# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for aiperf.kubernetes.watchdog.BenchmarkWatchdog.

Complements ``test_watchdog.py`` by exercising paths that file leaves cold:

- Monitor-loop exception swallow + cancellation path through stop()
- Tick-scheduled checks (node@tick=2, stale-ns@tick=3, resources@tick%6==0)
- _check_stale_namespaces: 403 ApiException -> INFO, non-403 -> debug, generic exception
- _check_pod_resources: peak tracking, memory-growth dedup, exception swallow
- _check_node_resources: exception swallow on listing failure
- _check_events: exception swallow (already partially covered, but the silent path)
- report(): node_cpu_pct / node_mem_pct / stale_ns_count fields propagate
- _check_elapsed_time: no-op when timeout is None
- _add_problem: INFO severity branch (only WARNING/CRITICAL covered elsewhere)
- ContainerInfo healthy running state: produces no problems
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.watchdog import (
    BenchmarkWatchdog,
    ContainerInfo,
    EventInfo,
    NodeResources,
    PodMetrics,
    ProblemSeverity,
    WatchdogPodSnapshot,
)
from tests.harness.time_traveler import TimeTraveler

# ============================================================
# Helpers / Fakes
# ============================================================


class FakeDataSource:
    """In-memory WatchdogDataSource with per-call hookable errors."""

    def __init__(self) -> None:
        self.pods: list[WatchdogPodSnapshot] = []
        self.events: list[EventInfo] = []
        self.nodes: list[NodeResources] = []
        self.namespaces: list[str] = []
        self.pod_metrics: list[PodMetrics] = []
        self.pod_logs: dict[str, str] = {}
        self.get_pods_error: BaseException | None = None
        self.get_events_error: BaseException | None = None
        self.get_nodes_error: BaseException | None = None
        self.get_namespaces_error: BaseException | None = None
        self.get_pod_metrics_error: BaseException | None = None
        self.get_pods_calls: int = 0

    async def get_pods(self, namespace: str) -> list[WatchdogPodSnapshot]:
        self.get_pods_calls += 1
        if self.get_pods_error:
            raise self.get_pods_error
        return list(self.pods)

    async def get_events(self, namespace: str, limit: int = 20) -> list[EventInfo]:
        if self.get_events_error:
            raise self.get_events_error
        return list(self.events[:limit])

    async def get_node_resources(self) -> list[NodeResources]:
        if self.get_nodes_error:
            raise self.get_nodes_error
        return list(self.nodes)

    async def get_namespaces(self, label_selector: str | None = None) -> list[str]:
        if self.get_namespaces_error:
            raise self.get_namespaces_error
        return list(self.namespaces)

    async def get_pod_logs(self, name: str, namespace: str, tail: int = 50) -> str:
        return self.pod_logs.get(name, "")

    async def get_pod_metrics(self, namespace: str) -> list[PodMetrics]:
        if self.get_pod_metrics_error:
            raise self.get_pod_metrics_error
        return list(self.pod_metrics)


@pytest.fixture
def source() -> FakeDataSource:
    return FakeDataSource()


def _make_pod(
    name: str = "aiperf-worker-0",
    phase: str = "Running",
    ready: bool = True,
    restarts: int = 0,
    containers: list[ContainerInfo] | None = None,
) -> WatchdogPodSnapshot:
    return WatchdogPodSnapshot(
        name=name,
        namespace="aiperf-run-1",
        phase=phase,
        ready=ready,
        restarts=restarts,
        container_statuses=containers or [],
    )


# ============================================================
# Monitor loop integration
# ============================================================


class TestMonitorLoopExceptionSwallow:
    """The monitor loop must keep running on per-tick failures."""

    @pytest.mark.asyncio
    async def test_loop_continues_after_per_tick_exception(
        self, source: FakeDataSource
    ) -> None:
        """A failing data-source call doesn't kill the loop — next tick still runs."""
        source.get_pods_error = RuntimeError("transient apiserver hiccup")
        wd = BenchmarkWatchdog(source, "test-ns", poll_interval=0.0)

        async with wd:
            # Yield enough times to let several ticks fire under instant sleep.
            for _ in range(20):
                await asyncio.sleep(0)

        # Multiple ticks should have run despite the source raising every time.
        assert source.get_pods_calls >= 2

    @pytest.mark.asyncio
    async def test_stop_cancels_running_loop(self, source: FakeDataSource) -> None:
        """stop() cleanly cancels the in-flight task without raising."""
        wd = BenchmarkWatchdog(source, "test-ns", poll_interval=0.0)
        wd.start()
        for _ in range(3):
            await asyncio.sleep(0)
        await wd.stop()
        assert wd._stopped is True
        assert wd._task is not None
        assert wd._task.done()


class TestRunTickScheduling:
    """Verify which checks run on which tick numbers."""

    @pytest.mark.asyncio
    async def test_node_check_runs_on_tick_two(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", poll_interval=0.0)
        wd._start_time = time.time()

        # _run_tick reads tick_count without incrementing — _monitor_loop owns
        # the increment. Drive it directly here to pin the schedule.
        wd._tick_count = 1
        await wd._run_tick()
        assert wd._node_check_done is False

        wd._tick_count = 2
        await wd._run_tick()
        assert wd._node_check_done is True

        wd._tick_count = 3
        await wd._run_tick()
        assert wd._stale_ns_check_done is True

    @pytest.mark.asyncio
    async def test_resource_check_only_after_first_window(
        self, source: FakeDataSource
    ) -> None:
        """check_pod_resources only fires on tick%interval==0 AND tick>interval."""
        source.pod_metrics = [
            PodMetrics(name="aiperf-worker-0", cpu_millicores=100, memory_mib=512)
        ]
        wd = BenchmarkWatchdog(source, "test-ns", poll_interval=0.0)
        wd._start_time = time.time()
        wd._resource_check_interval = 2

        # tick == interval: modulo matches but tick !> interval -> still skipped.
        wd._tick_count = 2
        await wd._run_tick()
        assert wd._peak_memory == {}

        # tick > interval AND tick%interval==0 -> resource check fires.
        wd._tick_count = 4
        await wd._run_tick()
        assert wd._peak_memory.get("aiperf-worker-0") == 512


# ============================================================
# Stale namespace 403 / API errors
# ============================================================


class TestStaleNamespaceErrors:
    """The check must downgrade to INFO on 403 and never crash on other errors."""

    @pytest.mark.asyncio
    async def test_403_logs_info_and_records_no_problem(
        self, source: FakeDataSource
    ) -> None:
        source.get_namespaces_error = ApiException(status=403, reason="Forbidden")
        wd = BenchmarkWatchdog(source, "aiperf-run-1")
        wd._start_time = time.time()

        await wd._check_stale_namespaces()

        assert wd._stale_ns_count == 0
        assert wd._problems == []

    @pytest.mark.asyncio
    async def test_non_403_api_error_swallowed(self, source: FakeDataSource) -> None:
        """5xx / other status codes go to debug log without breaking."""
        source.get_namespaces_error = ApiException(status=500, reason="Server Error")
        wd = BenchmarkWatchdog(source, "aiperf-run-1")
        wd._start_time = time.time()

        await wd._check_stale_namespaces()

        assert wd._stale_ns_count == 0
        assert wd._problems == []

    @pytest.mark.asyncio
    async def test_generic_exception_swallowed(self, source: FakeDataSource) -> None:
        """Anything outside ApiException (timeouts, OSError, ...) is also swallowed."""
        source.get_namespaces_error = RuntimeError("boom")
        wd = BenchmarkWatchdog(source, "aiperf-run-1")
        wd._start_time = time.time()

        await wd._check_stale_namespaces()

        assert wd._problems == []

    @pytest.mark.asyncio
    async def test_ascending_stale_count_logs_only(
        self, source: FakeDataSource
    ) -> None:
        """1-2 stale -> info log only (no problem); >2 -> warning."""
        source.namespaces = [
            "aiperf-run-1",
            "aiperf-system",
            "aiperf-old-1",
            "aiperf-old-2",
            "default",
        ]
        wd = BenchmarkWatchdog(source, "aiperf-run-1")
        wd._start_time = time.time()
        await wd._check_stale_namespaces()

        # 2 stale namespaces is NOT enough to warn (threshold is >2).
        assert wd._stale_ns_count == 2
        assert wd._problems == []


# ============================================================
# Pod resources / memory growth
# ============================================================


class TestPodResourceMonitoring:
    """Memory peak tracking + 1.2x growth dedup + exception swallow."""

    @pytest.mark.asyncio
    async def test_first_observation_stores_peak_no_problem(
        self, source: FakeDataSource
    ) -> None:
        source.pod_metrics = [
            PodMetrics(name="aiperf-worker-0", cpu_millicores=100, memory_mib=200)
        ]
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        await wd._check_pod_resources()

        assert wd._peak_memory["aiperf-worker-0"] == 200
        assert wd._problems == []

    @pytest.mark.asyncio
    async def test_growth_above_1_2x_triggers_warning(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.pod_metrics = [
            PodMetrics(name="aiperf-worker-0", cpu_millicores=100, memory_mib=100)
        ]
        await wd._check_pod_resources()

        # 200 > 100 * 1.2 -> warning fires.
        source.pod_metrics = [
            PodMetrics(name="aiperf-worker-0", cpu_millicores=100, memory_mib=200)
        ]
        await wd._check_pod_resources()

        warns = [p for p in wd._problems if p.category == "memory-growth"]
        assert len(warns) == 1
        assert warns[0].pod_name == "aiperf-worker-0"
        assert "200Mi" in warns[0].message

    @pytest.mark.asyncio
    async def test_growth_below_1_2x_does_not_warn(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.pod_metrics = [
            PodMetrics(name="aiperf-worker-0", cpu_millicores=100, memory_mib=100)
        ]
        await wd._check_pod_resources()

        source.pod_metrics = [
            PodMetrics(name="aiperf-worker-0", cpu_millicores=100, memory_mib=110)
        ]
        await wd._check_pod_resources()

        assert [p for p in wd._problems if p.category == "memory-growth"] == []

    @pytest.mark.asyncio
    async def test_growth_warning_dedup_via_fingerprint(
        self, source: FakeDataSource
    ) -> None:
        """Same memory bucket (memory_mib//100) doesn't repeat the warning."""
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.pod_metrics = [
            PodMetrics(name="aiperf-worker-0", cpu_millicores=100, memory_mib=100)
        ]
        await wd._check_pod_resources()

        source.pod_metrics = [
            PodMetrics(name="aiperf-worker-0", cpu_millicores=100, memory_mib=210)
        ]
        await wd._check_pod_resources()

        # Same 100-mib bucket on a follow-up tick would re-warn; bump still
        # in same bucket (210//100 == 2) is silent.
        await wd._check_pod_resources()

        warns = [p for p in wd._problems if p.category == "memory-growth"]
        assert len(warns) == 1

    @pytest.mark.asyncio
    async def test_metrics_server_missing_swallowed(
        self, source: FakeDataSource
    ) -> None:
        """If metrics-server isn't installed, the API raises and we swallow."""
        source.get_pod_metrics_error = ApiException(
            status=404, reason="metrics.k8s.io not found"
        )
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        await wd._check_pod_resources()
        assert wd._problems == []


# ============================================================
# Node-resources / events failure paths
# ============================================================


class TestSecondaryCheckErrorSwallow:
    """All side-checks must keep the watchdog running on fetch failures."""

    @pytest.mark.asyncio
    async def test_check_node_resources_swallows_error(
        self, source: FakeDataSource
    ) -> None:
        source.get_nodes_error = ApiException(status=403, reason="Forbidden")
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        await wd._check_node_resources()
        assert wd._problems == []

    @pytest.mark.asyncio
    async def test_check_events_swallows_error(self, source: FakeDataSource) -> None:
        source.get_events_error = RuntimeError("apiserver TLS handshake failed")
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        await wd._check_events()
        assert wd._problems == []


# ============================================================
# Elapsed-time edge cases
# ============================================================


class TestElapsedTime:
    """The timeout-aware check is a no-op when timeout is unset."""

    def test_no_timeout_no_problem(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=None)
        wd._start_time = time.time() - 10_000  # arbitrarily large elapsed
        wd._check_elapsed_time()
        assert wd._problems == []

    def test_timeout_warning_then_imminent(
        self, source: FakeDataSource, time_traveler: TimeTraveler
    ) -> None:
        """Both warning + imminent are recorded once each across the threshold."""
        wd = BenchmarkWatchdog(source, "test-ns", timeout=100)
        wd._start_time = time.time()

        time_traveler.advance_time(50)  # 50s elapsed -> 50s remaining (< 60)
        wd._check_elapsed_time()

        time_traveler.advance_time(40)  # 90s elapsed -> 10s remaining (< 15)
        wd._check_elapsed_time()

        cats = [p.category for p in wd._problems]
        assert "timeout-warning" in cats
        assert "timeout-imminent" in cats


# ============================================================
# Add-problem severity / report fields
# ============================================================


class TestAddProblemSeverity:
    """All three severity branches log + record correctly."""

    @pytest.mark.parametrize(
        "severity",
        [
            param(ProblemSeverity.INFO, id="info"),
            param(ProblemSeverity.WARNING, id="warning"),
            param(ProblemSeverity.CRITICAL, id="critical"),
        ],
    )  # fmt: skip
    def test_add_problem_records(
        self, source: FakeDataSource, severity: ProblemSeverity
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._add_problem(severity, "test-cat", "msg", suggestion="try X")

        assert len(wd._problems) == 1
        recorded = wd._problems[0]
        assert recorded.severity is severity
        assert recorded.category == "test-cat"
        assert recorded.namespace == "test-ns"
        assert recorded.suggestion == "try X"


class TestReportSnapshotFields:
    """Report carries node_cpu_pct / node_mem_pct / stale_ns_count from state."""

    def test_report_includes_node_and_stale_fields(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=120)
        wd._start_time = time.time()
        wd._node_cpu_pct = 73
        wd._node_mem_pct = 41
        wd._stale_ns_count = 4

        report = wd.report
        assert report.namespace == "test-ns"
        assert report.timeout == 120.0
        assert report.node_cpu_pct == 73
        assert report.node_mem_pct == 41
        assert report.stale_ns_count == 4

    def test_report_duration_is_zero_when_not_started(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        # _start_time still 0.0; duration short-circuits to 0.0
        report = wd.report
        assert report.duration == 0.0


# ============================================================
# Container state: healthy paths produce no problems
# ============================================================


class TestContainerStatesHealthyPaths:
    """Running and Completed-with-zero-exit don't produce problems."""

    def test_running_container_no_problem(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        ok = ContainerInfo(name="main", ready=True, state="running")
        pod = _make_pod(containers=[ok])
        wd._analyze_pods([pod])

        assert wd._problems == []

    def test_terminated_zero_exit_no_problem(self, source: FakeDataSource) -> None:
        """Successful container terminations don't fabricate problems."""
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        done = ContainerInfo(
            name="main",
            ready=False,
            state="terminated",
            reason="Completed",
            exit_code=0,
        )
        pod = _make_pod(phase="Succeeded", containers=[done])
        wd._analyze_pods([pod])

        # pod-failed problems require phase=Failed; a Succeeded pod with a
        # zero-exit container records no problem.
        assert all(p.category != "pod-failed" for p in wd._problems)


# ============================================================
# Start logging branches
# ============================================================


class TestStartLogging:
    """Start() formats timeout='none' when unset and 'Ns' otherwise."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "timeout,token",
        [
            param(None, "none", id="unset"),
            param(60.0, "60.0s", id="set"),
        ],
    )  # fmt: skip
    async def test_start_logs_timeout_token(
        self,
        source: FakeDataSource,
        timeout: float | None,
        token: str,
        caplog: Any,
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=timeout, poll_interval=0.0)
        with caplog.at_level("INFO", logger="aiperf.kubernetes.watchdog"):
            wd.start()
            await wd.stop()

        text = "\n".join(rec.getMessage() for rec in caplog.records)
        assert token in text


class TestWatchdogThresholdEnvBinding:
    """Watchdog thresholds must be tunable via AIPERF_K8S_WATCHDOG_*.

    They were plain keyword defaults with no environment binding, so a cluster
    with slow image pulls or a restart-tolerant workload had no way to raise
    them. The old AIPERF_K8S_WATCH_* names went away with `aiperf kube watch`.
    """

    def test_defaults_come_from_environment(self):
        from aiperf.kubernetes.environment import K8sEnvironment
        from aiperf.kubernetes.watchdog import BenchmarkWatchdog

        wd = BenchmarkWatchdog(object(), "ns")
        env = K8sEnvironment.WATCHDOG
        assert wd.poll_interval == env.POLL_INTERVAL_SECONDS
        assert wd.status_interval == env.STATUS_INTERVAL_SECONDS
        assert wd.pending_threshold == env.PENDING_THRESHOLD_SECONDS
        assert wd.pending_critical_threshold == env.PENDING_CRITICAL_THRESHOLD_SECONDS
        assert wd.crashloop_threshold == env.CRASHLOOP_RESTART_THRESHOLD

    def test_explicit_kwargs_still_win(self):
        from aiperf.kubernetes.watchdog import BenchmarkWatchdog

        wd = BenchmarkWatchdog(object(), "ns", poll_interval=0.0, crashloop_threshold=9)
        assert wd.poll_interval == 0.0
        assert wd.crashloop_threshold == 9

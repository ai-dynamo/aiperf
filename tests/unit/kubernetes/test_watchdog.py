# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.watchdog module.

Focuses on:
- Data model construction and properties (WatchdogReport, PodTimeline, etc.)
- WatchdogDataSource protocol compliance
- K8sWatchdogSource parsing of typed kubernetes_asyncio objects
- BenchmarkWatchdog pod analysis, event processing, timeout detection
- Helper functions (_fmt_duration, _pod_role, _phase_icon, etc.)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.watchdog import (
    BenchmarkWatchdog,
    ContainerInfo,
    EventInfo,
    K8sWatchdogSource,
    NodeResources,
    PodMetrics,
    PodTimeline,
    ProblemSeverity,
    WatchdogDataSource,
    WatchdogPodSnapshot,
    WatchdogProblem,
    WatchdogReport,
    _fmt_duration,
    _parse_container_state,
    _phase_icon,
    _pod_role,
    _short_pod_name,
)
from tests.harness.time_traveler import TimeTraveler

# ============================================================
# Helpers for building typed kubernetes_asyncio fakes
# ============================================================


def _pod_obj(
    *,
    name: str = "pod-1",
    namespace: str = "ns",
    phase: str = "Running",
    creation_timestamp: datetime | None = None,
    container_statuses: list[Any] | None = None,
    init_container_statuses: list[Any] | None = None,
) -> MagicMock:
    """Build an object that looks like a V1Pod with the attributes we access."""
    pod = MagicMock()
    pod.metadata = MagicMock()
    pod.metadata.name = name
    pod.metadata.namespace = namespace
    pod.metadata.creation_timestamp = creation_timestamp
    pod.status = MagicMock()
    pod.status.phase = phase
    pod.status.container_statuses = container_statuses
    pod.status.init_container_statuses = init_container_statuses
    return pod


def _container_status(
    *,
    name: str = "main",
    ready: bool = True,
    restart_count: int = 0,
    running: bool = False,
    waiting_reason: str | None = None,
    waiting_message: str | None = None,
    terminated_reason: str | None = None,
    terminated_message: str | None = None,
    terminated_exit_code: int | None = None,
) -> MagicMock:
    """Build an object that looks like a V1ContainerStatus."""
    cs = MagicMock()
    cs.name = name
    cs.ready = ready
    cs.restart_count = restart_count
    cs.state = MagicMock()
    cs.state.running = MagicMock() if running else None
    if waiting_reason is not None or waiting_message is not None:
        cs.state.waiting = MagicMock()
        cs.state.waiting.reason = waiting_reason
        cs.state.waiting.message = waiting_message
    else:
        cs.state.waiting = None
    if (
        terminated_reason is not None
        or terminated_message is not None
        or terminated_exit_code is not None
    ):
        cs.state.terminated = MagicMock()
        cs.state.terminated.reason = terminated_reason
        cs.state.terminated.message = terminated_message
        cs.state.terminated.exit_code = terminated_exit_code
    else:
        cs.state.terminated = None
    return cs


def _event_obj(
    *,
    type_: str = "Normal",
    reason: str = "",
    message: str = "",
    involved_name: str = "pod-1",
    last_timestamp: datetime | None = None,
) -> MagicMock:
    """Build an object that looks like a V1Event."""
    ev = MagicMock()
    ev.type = type_
    ev.reason = reason
    ev.message = message
    ev.involved_object = MagicMock()
    ev.involved_object.name = involved_name
    ev.last_timestamp = last_timestamp
    return ev


def _node_obj(
    *,
    name: str = "node-1",
    allocatable: dict[str, str] | None = None,
) -> MagicMock:
    """Build an object that looks like a V1Node."""
    node = MagicMock()
    node.metadata = MagicMock()
    node.metadata.name = name
    node.status = MagicMock()
    node.status.allocatable = allocatable or {}
    return node


def _ns_obj(*, name: str = "default") -> MagicMock:
    """Build an object that looks like a V1Namespace."""
    ns = MagicMock()
    ns.metadata = MagicMock()
    ns.metadata.name = name
    return ns


def _list_result(items: list[Any]) -> MagicMock:
    """Build a list API response with .items."""
    result = MagicMock()
    result.items = items
    return result


def _patch_core_api(core_mock: MagicMock):
    """Patch kubernetes_asyncio.client.CoreV1Api inside watchdog module."""
    return patch("aiperf.kubernetes.watchdog.client.CoreV1Api", return_value=core_mock)


# ============================================================
# Fixtures
# ============================================================


class FakeDataSource:
    """In-memory WatchdogDataSource for testing BenchmarkWatchdog logic."""

    def __init__(self) -> None:
        self.pods: list[WatchdogPodSnapshot] = []
        self.events: list[EventInfo] = []
        self.nodes: list[NodeResources] = []
        self.namespaces: list[str] = []
        self.pod_logs: dict[str, str] = {}
        self.get_pods_error: Exception | None = None

    async def get_pods(self, namespace: str) -> list[WatchdogPodSnapshot]:
        if self.get_pods_error:
            raise self.get_pods_error
        return list(self.pods)

    async def get_events(self, namespace: str, limit: int = 20) -> list[EventInfo]:
        return list(self.events[:limit])

    async def get_node_resources(self) -> list[NodeResources]:
        return list(self.nodes)

    async def get_namespaces(self, label_selector: str | None = None) -> list[str]:
        return list(self.namespaces)

    async def get_pod_logs(self, name: str, namespace: str, tail: int = 50) -> str:
        return self.pod_logs.get(name, "")

    async def get_pod_metrics(self, namespace: str) -> list[PodMetrics]:
        return []


@pytest.fixture
def source() -> FakeDataSource:
    """Create a controllable in-memory data source."""
    return FakeDataSource()


def _make_pod(
    name: str = "aiperf-test-worker-abc",
    namespace: str = "aiperf-run-1",
    phase: str = "Running",
    ready: bool = True,
    restarts: int = 0,
    containers: list[ContainerInfo] | None = None,
) -> WatchdogPodSnapshot:
    """Build a WatchdogPodSnapshot with sensible defaults."""
    return WatchdogPodSnapshot(
        name=name,
        namespace=namespace,
        phase=phase,
        ready=ready,
        restarts=restarts,
        container_statuses=containers or [],
    )


def _make_event(
    reason: str = "Normal",
    message: str = "ok",
    involved_object: str = "pod-1",
    event_type: str = "Normal",
    last_timestamp: datetime | None = None,
) -> EventInfo:
    """Build an EventInfo with sensible defaults."""
    return EventInfo(
        type=event_type,
        reason=reason,
        message=message,
        involved_object=involved_object,
        last_timestamp=last_timestamp,
    )


# ============================================================
# Helper Functions
# ============================================================


class TestFmtDuration:
    """Verify human-readable duration formatting."""

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            param(0, "0s", id="zero"),
            param(30, "30s", id="30-seconds"),
            param(59, "59s", id="59-seconds"),
            param(60, "1m00s", id="one-minute"),
            param(90, "1m30s", id="90-seconds"),
            param(3661, "61m01s", id="over-one-hour"),
        ],
    )  # fmt: skip
    def test_fmt_duration_formats_correctly(
        self, seconds: float, expected: str
    ) -> None:
        assert _fmt_duration(seconds) == expected


class TestPodRole:
    """Verify pod role identification from name."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            param("aiperf-controller-abc", "controller", id="controller-prefix"),
            param("aiperf-worker-xyz", "worker", id="worker-prefix"),
            param("aiperf-random-pod", "unknown", id="unknown-role"),
            param("controller", "controller", id="bare-controller"),
            param("worker-0", "worker", id="bare-worker"),
        ],
    )  # fmt: skip
    def test_pod_role_returns_correct_role(self, name: str, expected: str) -> None:
        assert _pod_role(name) == expected


class TestPhaseIcon:
    """Verify phase-to-icon mapping."""

    @pytest.mark.parametrize(
        "phase,expected",
        [
            param("Pending", "...", id="pending"),
            param("Running", ">>>", id="running"),
            param("Succeeded", "[OK]", id="succeeded"),
            param("Failed", "[!!]", id="failed"),
            param("Unknown", "[??]", id="unknown"),
            param("Completed", "[OK]", id="completed"),
            param("SomethingElse", "   ", id="unrecognized"),
        ],
    )  # fmt: skip
    def test_phase_icon_returns_correct_icon(self, phase: str, expected: str) -> None:
        assert _phase_icon(phase) == expected


class TestShortPodName:
    """Verify pod name shortening."""

    def test_short_name_under_max_unchanged(self) -> None:
        assert _short_pod_name("short-name", max_len=38) == "short-name"

    def test_short_name_over_max_truncated_with_ellipsis(self) -> None:
        long_name = "aiperf-run-12345-controller-deployment-abcdef1234567890"
        result = _short_pod_name(long_name, max_len=20)
        assert result.startswith("...")
        assert len(result) == 20

    def test_short_name_exact_max_unchanged(self) -> None:
        name = "a" * 38
        assert _short_pod_name(name, max_len=38) == name


class TestParseContainerState:
    """Verify container state dict parsing."""

    def test_running_state(self) -> None:
        state, reason, message, exit_code = _parse_container_state({"running": {}})
        assert state == "running"
        assert reason is None
        assert exit_code is None

    def test_waiting_state_with_reason(self) -> None:
        state, reason, message, exit_code = _parse_container_state(
            {
                "waiting": {
                    "reason": "CrashLoopBackOff",
                    "message": "back-off restarting",
                }
            }
        )
        assert state == "waiting"
        assert reason == "CrashLoopBackOff"
        assert message == "back-off restarting"
        assert exit_code is None

    def test_terminated_state_with_exit_code(self) -> None:
        state, reason, message, exit_code = _parse_container_state(
            {"terminated": {"reason": "Completed", "exitCode": 0}}
        )
        assert state == "terminated"
        assert reason == "Completed"
        assert exit_code == 0

    def test_terminated_state_nonzero_exit(self) -> None:
        state, reason, message, exit_code = _parse_container_state(
            {"terminated": {"reason": "Error", "exitCode": 137, "message": "OOMKilled"}}
        )
        assert state == "terminated"
        assert exit_code == 137

    def test_empty_state_returns_unknown(self) -> None:
        state, reason, message, exit_code = _parse_container_state({})
        assert state == "unknown"


# ============================================================
# Data Models
# ============================================================


class TestWatchdogReport:
    """Verify WatchdogReport computed properties."""

    def _make_report(
        self,
        problems: list[WatchdogProblem] | None = None,
        timelines: dict[str, PodTimeline] | None = None,
    ) -> WatchdogReport:
        return WatchdogReport(
            namespace="test-ns",
            duration=100.0,
            timeout=300.0,
            problems=problems or [],
            pod_timelines=timelines or {},
            completed_pods=set(),
        )

    def test_has_critical_true_when_critical_problem(self) -> None:
        problems = [
            WatchdogProblem(
                severity=ProblemSeverity.CRITICAL, category="test", message="boom"
            ),
        ]
        report = self._make_report(problems=problems)
        assert report.has_critical is True

    def test_has_critical_false_when_only_warnings(self) -> None:
        problems = [
            WatchdogProblem(
                severity=ProblemSeverity.WARNING, category="test", message="hmm"
            ),
        ]
        report = self._make_report(problems=problems)
        assert report.has_critical is False

    def test_has_critical_false_when_no_problems(self) -> None:
        report = self._make_report()
        assert report.has_critical is False

    def test_succeeded_count(self) -> None:
        timelines = {
            "pod-1": PodTimeline(name="pod-1", last_phase="Succeeded"),
            "pod-2": PodTimeline(name="pod-2", last_phase="Running"),
            "pod-3": PodTimeline(name="pod-3", last_phase="Completed"),
        }
        report = self._make_report(timelines=timelines)
        assert report.succeeded_count == 2

    def test_failed_count(self) -> None:
        timelines = {
            "pod-1": PodTimeline(name="pod-1", last_phase="Failed"),
            "pod-2": PodTimeline(name="pod-2", last_phase="Running"),
            "pod-3": PodTimeline(name="pod-3", last_phase="Failed"),
        }
        report = self._make_report(timelines=timelines)
        assert report.failed_count == 2

    def test_total_restarts(self) -> None:
        timelines = {
            "pod-1": PodTimeline(name="pod-1", restart_count=3),
            "pod-2": PodTimeline(name="pod-2", restart_count=1),
        }
        report = self._make_report(timelines=timelines)
        assert report.total_restarts == 4

    def test_empty_report_zeroes(self) -> None:
        report = self._make_report()
        assert report.succeeded_count == 0
        assert report.failed_count == 0
        assert report.total_restarts == 0


# ============================================================
# WatchdogDataSource Protocol
# ============================================================


class TestWatchdogDataSourceProtocol:
    """Verify protocol compliance."""

    def test_k8s_source_satisfies_protocol(self) -> None:
        source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
        assert isinstance(source, WatchdogDataSource)

    def test_fake_source_satisfies_protocol(self) -> None:
        source = FakeDataSource()
        assert isinstance(source, WatchdogDataSource)


# ============================================================
# K8sWatchdogSource
# ============================================================


class TestK8sWatchdogSourceGetPods:
    """Verify pod parsing from V1Pod objects."""

    @pytest.mark.asyncio
    async def test_get_pods_parses_running_pod(self) -> None:
        pod = _pod_obj(
            name="aiperf-worker-abc",
            namespace="test-ns",
            phase="Running",
            creation_timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
            container_statuses=[
                _container_status(
                    name="main", ready=True, restart_count=0, running=True
                )
            ],
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_list_result([pod]))

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            pods = await source.get_pods("test-ns")

        assert len(pods) == 1
        result = pods[0]
        assert result.name == "aiperf-worker-abc"
        assert result.phase == "Running"
        assert result.ready is True
        assert result.restarts == 0
        assert len(result.container_statuses) == 1

    @pytest.mark.asyncio
    async def test_get_pods_sums_init_container_restarts(self) -> None:
        pod = _pod_obj(
            name="pod-1",
            namespace="ns",
            phase="Running",
            container_statuses=[
                _container_status(
                    name="main", ready=True, restart_count=1, running=True
                )
            ],
            init_container_statuses=[
                _container_status(
                    name="init",
                    ready=True,
                    restart_count=2,
                    terminated_exit_code=0,
                )
            ],
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_list_result([pod]))

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            pods = await source.get_pods("ns")

        assert pods[0].restarts == 3
        assert len(pods[0].container_statuses) == 2

    @pytest.mark.asyncio
    async def test_get_pods_not_ready_when_no_containers(self) -> None:
        pod = _pod_obj(
            name="pod-1", namespace="ns", phase="Pending", container_statuses=None
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_list_result([pod]))

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            pods = await source.get_pods("ns")

        assert pods[0].ready is False

    @pytest.mark.asyncio
    async def test_get_pods_not_ready_when_container_not_ready(self) -> None:
        pod = _pod_obj(
            name="pod-1",
            namespace="ns",
            phase="Running",
            container_statuses=[
                _container_status(
                    name="main",
                    ready=False,
                    restart_count=0,
                    waiting_reason="CrashLoopBackOff",
                )
            ],
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_pod = AsyncMock(return_value=_list_result([pod]))

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            pods = await source.get_pods("ns")

        assert pods[0].ready is False
        assert pods[0].container_statuses[0].reason == "CrashLoopBackOff"


class TestK8sWatchdogSourceGetEvents:
    """Verify event parsing and sorting."""

    @pytest.mark.asyncio
    async def test_get_events_parses_and_sorts_by_timestamp(self) -> None:
        older = _event_obj(
            type_="Warning",
            reason="FailedScheduling",
            message="no nodes available",
            involved_name="pod-1",
            last_timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        )
        newer = _event_obj(
            type_="Normal",
            reason="Scheduled",
            message="assigned to node",
            involved_name="pod-1",
            last_timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
        )
        mock_core = MagicMock()
        mock_core.list_namespaced_event = AsyncMock(
            return_value=_list_result([older, newer])
        )

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            events = await source.get_events("ns")

        assert len(events) == 2
        assert events[0].reason == "Scheduled"  # newer first
        assert events[1].reason == "FailedScheduling"

    @pytest.mark.asyncio
    async def test_get_events_respects_limit(self) -> None:
        raw_events = [
            _event_obj(
                type_="Normal",
                reason=f"Event{i}",
                message=f"msg{i}",
                involved_name="pod-1",
                last_timestamp=None,
            )
            for i in range(5)
        ]
        mock_core = MagicMock()
        mock_core.list_namespaced_event = AsyncMock(
            return_value=_list_result(raw_events)
        )

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            events = await source.get_events("ns", limit=2)

        assert len(events) == 2


class TestK8sWatchdogSourceGetNodeResources:
    """Verify node resource parsing."""

    @pytest.mark.asyncio
    async def test_get_node_resources_parses_gpu_count(self) -> None:
        node = _node_obj(
            name="gpu-node-1",
            allocatable={
                "cpu": "64",
                "memory": "256Gi",
                "nvidia.com/gpu": "8",
            },
        )
        mock_core = MagicMock()
        mock_core.list_node = AsyncMock(return_value=_list_result([node]))

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            nodes = await source.get_node_resources()

        assert len(nodes) == 1
        assert nodes[0].name == "gpu-node-1"
        assert nodes[0].allocatable_gpu == 8
        assert nodes[0].allocatable_cpu == "64"

    @pytest.mark.asyncio
    async def test_get_node_resources_handles_missing_gpu(self) -> None:
        node = _node_obj(name="cpu-node", allocatable={"cpu": "16", "memory": "64Gi"})
        mock_core = MagicMock()
        mock_core.list_node = AsyncMock(return_value=_list_result([node]))

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            nodes = await source.get_node_resources()

        assert nodes[0].allocatable_gpu == 0

    @pytest.mark.asyncio
    async def test_get_node_resources_handles_invalid_gpu_string(self) -> None:
        node = _node_obj(
            name="bad-node", allocatable={"nvidia.com/gpu": "not-a-number"}
        )
        mock_core = MagicMock()
        mock_core.list_node = AsyncMock(return_value=_list_result([node]))

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            nodes = await source.get_node_resources()

        assert nodes[0].allocatable_gpu == 0


class TestK8sWatchdogSourceGetNamespaces:
    """Verify namespace listing."""

    @pytest.mark.asyncio
    async def test_get_namespaces_returns_names(self) -> None:
        ns1 = _ns_obj(name="aiperf-run-1")
        ns2 = _ns_obj(name="default")
        mock_core = MagicMock()
        mock_core.list_namespace = AsyncMock(return_value=_list_result([ns1, ns2]))

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            result = await source.get_namespaces()

        assert result == ["aiperf-run-1", "default"]

    @pytest.mark.asyncio
    async def test_get_namespaces_passes_label_selector(self) -> None:
        mock_core = MagicMock()
        mock_core.list_namespace = AsyncMock(return_value=_list_result([]))

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            await source.get_namespaces(label_selector="app=aiperf")

        mock_core.list_namespace.assert_awaited_once()
        call_kwargs = mock_core.list_namespace.call_args.kwargs
        assert call_kwargs["label_selector"] == "app=aiperf"


class TestK8sWatchdogSourceGetPodLogs:
    """Verify pod log fetching."""

    @pytest.mark.asyncio
    async def test_get_pod_logs_returns_logs(self) -> None:
        mock_core = MagicMock()
        mock_core.read_namespaced_pod_log = AsyncMock(return_value="some log output")

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            result = await source.get_pod_logs("pod-1", "ns")

        assert result == "some log output"
        mock_core.read_namespaced_pod_log.assert_awaited_once_with(
            name="pod-1", namespace="ns", tail_lines=50
        )

    @pytest.mark.asyncio
    async def test_get_pod_logs_returns_empty_on_api_exception(self) -> None:
        mock_core = MagicMock()
        mock_core.read_namespaced_pod_log = AsyncMock(
            side_effect=ApiException(status=404, reason="Not Found")
        )

        with _patch_core_api(mock_core):
            source = K8sWatchdogSource(api=MagicMock(spec=ApiClient))
            result = await source.get_pod_logs("missing-pod", "ns")

        assert result == ""


# ============================================================
# BenchmarkWatchdog - Context Manager
# ============================================================


class TestBenchmarkWatchdogLifecycle:
    """Verify async context manager starts/stops background task."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=300, poll_interval=1.0)
        wd.start()
        assert wd._task is not None
        assert not wd._task.done()
        await wd.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_flag(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=300, poll_interval=1.0)
        wd._stopped = False
        wd._task = None
        await wd.stop()
        assert wd._stopped is True


# ============================================================
# BenchmarkWatchdog - Pod Phase Tracking
# ============================================================


class TestBenchmarkWatchdogPhaseTracking:
    """Verify pod timeline creation and phase transitions."""

    def test_analyze_pods_creates_timeline(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()
        pod = _make_pod(name="aiperf-worker-abc", phase="Pending")
        wd._analyze_pods([pod])

        assert "aiperf-worker-abc" in wd._pod_timelines
        tl = wd._pod_timelines["aiperf-worker-abc"]
        assert tl.role == "worker"

    def test_phase_transition_recorded(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        pod_pending = _make_pod(name="aiperf-worker-abc", phase="Pending")
        wd._analyze_pods([pod_pending])

        pod_running = _make_pod(name="aiperf-worker-abc", phase="Running")
        wd._analyze_pods([pod_running])

        tl = wd._pod_timelines["aiperf-worker-abc"]
        assert tl.last_phase == "Running"
        assert len(tl.phase_history) == 2

    def test_same_phase_not_duplicated(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Pending")
        wd._analyze_pods([pod])
        wd._analyze_pods([pod])

        tl = wd._pod_timelines["aiperf-worker-abc"]
        assert len(tl.phase_history) == 1


# ============================================================
# BenchmarkWatchdog - Pending Timeout Detection
# ============================================================


class TestBenchmarkWatchdogPendingTimeout:
    """Verify escalating warnings for pods stuck in Pending."""

    def test_pending_warning_after_threshold(
        self, source: FakeDataSource, time_traveler: TimeTraveler
    ) -> None:
        wd = BenchmarkWatchdog(
            source,
            "test-ns",
            timeout=60,
            pending_threshold=10.0,
            pending_critical_threshold=60.0,
        )
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Pending")
        wd._analyze_pods([pod])

        time_traveler.advance_time(15.0)

        wd._analyze_pods([pod])

        warnings = [p for p in wd._problems if p.severity == ProblemSeverity.WARNING]
        assert len(warnings) == 1
        assert "Pending" in warnings[0].message

    def test_pending_critical_after_critical_threshold(
        self, source: FakeDataSource, time_traveler: TimeTraveler
    ) -> None:
        wd = BenchmarkWatchdog(
            source,
            "test-ns",
            timeout=60,
            pending_threshold=10.0,
            pending_critical_threshold=30.0,
        )
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Pending")
        wd._analyze_pods([pod])

        time_traveler.advance_time(35.0)

        wd._analyze_pods([pod])

        crits = [p for p in wd._problems if p.severity == ProblemSeverity.CRITICAL]
        assert len(crits) == 1
        assert "stuck Pending" in crits[0].message

    def test_pending_warning_not_repeated(
        self, source: FakeDataSource, time_traveler: TimeTraveler
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=30, pending_threshold=5.0)
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Pending")
        wd._analyze_pods([pod])

        time_traveler.advance_time(10.0)

        wd._analyze_pods([pod])
        wd._analyze_pods([pod])

        warnings = [p for p in wd._problems if p.category == "pod-pending"]
        assert len(warnings) == 1

    def test_no_warning_when_not_pending(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", pending_threshold=5.0)
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Running")
        wd._analyze_pods([pod])

        assert len(wd._problems) == 0


# ============================================================
# BenchmarkWatchdog - Crash Loop Detection
# ============================================================


class TestBenchmarkWatchdogCrashLoop:
    """Verify restart count tracking and crash loop detection."""

    def test_crash_loop_detected_at_threshold(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", crashloop_threshold=3)
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Running", restarts=0)
        wd._analyze_pods([pod])

        pod_restarted = _make_pod(name="aiperf-worker-abc", phase="Running", restarts=3)
        wd._analyze_pods([pod_restarted])

        crits = [p for p in wd._problems if p.category == "crash-loop"]
        assert len(crits) == 1
        assert "restarted 3x" in crits[0].message

    def test_crash_loop_not_warned_below_threshold(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", crashloop_threshold=5)
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Running", restarts=0)
        wd._analyze_pods([pod])

        pod_restarted = _make_pod(name="aiperf-worker-abc", phase="Running", restarts=2)
        wd._analyze_pods([pod_restarted])

        assert len([p for p in wd._problems if p.category == "crash-loop"]) == 0

    def test_crash_loop_warning_not_repeated(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", crashloop_threshold=2)
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", restarts=0)
        wd._analyze_pods([pod])

        pod_r2 = _make_pod(name="aiperf-worker-abc", restarts=2)
        wd._analyze_pods([pod_r2])

        pod_r5 = _make_pod(name="aiperf-worker-abc", restarts=5)
        wd._analyze_pods([pod_r5])

        assert len([p for p in wd._problems if p.category == "crash-loop"]) == 1


# ============================================================
# BenchmarkWatchdog - Container State Analysis
# ============================================================


class TestBenchmarkWatchdogContainerStates:
    """Verify detection of problematic container states."""

    @pytest.mark.parametrize(
        "reason",
        [
            "CrashLoopBackOff",
            "ImagePullBackOff",
            "ErrImagePull",
            "ErrImageNeverPull",
            "CreateContainerConfigError",
            "InvalidImageName",
        ],
    )  # fmt: skip
    def test_waiting_container_with_bad_reason_creates_critical(
        self, source: FakeDataSource, reason: str
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        container = ContainerInfo(
            name="main", ready=False, state="waiting", reason=reason, message="details"
        )
        pod = _make_pod(name="pod-1", phase="Running", containers=[container])
        wd._analyze_pods([pod])

        crits = [p for p in wd._problems if p.severity == ProblemSeverity.CRITICAL]
        assert len(crits) == 1
        assert reason in crits[0].message or reason.lower() in crits[0].category

    def test_oomkilled_creates_critical(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        container = ContainerInfo(
            name="main",
            ready=False,
            state="terminated",
            reason="OOMKilled",
            exit_code=137,
        )
        pod = _make_pod(name="pod-1", phase="Running", containers=[container])
        wd._analyze_pods([pod])

        crits = [p for p in wd._problems if p.category == "oom-killed"]
        assert len(crits) == 1
        assert "OOMKilled" in crits[0].message

    def test_container_state_fingerprint_deduplication(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        container = ContainerInfo(
            name="main", ready=False, state="waiting", reason="CrashLoopBackOff"
        )
        pod = _make_pod(name="pod-1", phase="Running", containers=[container])

        wd._analyze_pods([pod])
        wd._analyze_pods([pod])

        crits = [p for p in wd._problems if p.severity == ProblemSeverity.CRITICAL]
        assert len(crits) == 1


# ============================================================
# BenchmarkWatchdog - Event Analysis
# ============================================================


class TestBenchmarkWatchdogEventAnalysis:
    """Verify K8s event classification and problem recording."""

    @pytest.mark.asyncio
    async def test_failed_scheduling_with_insufficient_creates_critical(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.events = [
            _make_event(
                reason="FailedScheduling",
                message="Insufficient nvidia.com/gpu",
                involved_object="pod-1",
                event_type="Warning",
            )
        ]
        await wd._check_events()

        crits = [p for p in wd._problems if p.severity == ProblemSeverity.CRITICAL]
        assert len(crits) == 1
        assert "FailedScheduling" in crits[0].message

    @pytest.mark.asyncio
    async def test_failed_scheduling_without_insufficient_creates_warning(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.events = [
            _make_event(
                reason="FailedScheduling",
                message="no matching node found",
                involved_object="pod-1",
            )
        ]
        await wd._check_events()

        warns = [p for p in wd._problems if p.severity == ProblemSeverity.WARNING]
        assert len(warns) == 1

    @pytest.mark.asyncio
    async def test_failed_mount_creates_warning(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.events = [
            _make_event(
                reason="FailedMount",
                message="volume not found",
                involved_object="pod-1",
            )
        ]
        await wd._check_events()

        warns = [p for p in wd._problems if p.category == "volume-issue"]
        assert len(warns) == 1

    @pytest.mark.asyncio
    async def test_backoff_warning_event(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.events = [
            _make_event(
                reason="BackOff",
                message="back-off restarting failed container",
                involved_object="pod-1",
                event_type="Warning",
            )
        ]
        await wd._check_events()

        warns = [p for p in wd._problems if p.category == "container-backoff"]
        assert len(warns) == 1

    @pytest.mark.asyncio
    async def test_event_fingerprint_deduplication(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        event = _make_event(
            reason="FailedScheduling",
            message="no nodes available",
            involved_object="pod-1",
        )
        source.events = [event]

        await wd._check_events()
        await wd._check_events()

        assert len(wd._problems) == 1

    @pytest.mark.asyncio
    async def test_normal_events_ignored(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.events = [
            _make_event(reason="Scheduled", message="assigned", event_type="Normal")
        ]
        await wd._check_events()

        assert len(wd._problems) == 0


# ============================================================
# BenchmarkWatchdog - Pod Completion
# ============================================================


class TestBenchmarkWatchdogPodCompletion:
    """Verify pod terminal state detection."""

    def test_succeeded_pod_tracked(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Succeeded")
        wd._analyze_pods([pod])

        assert "aiperf-worker-abc" in wd._completed_pods
        assert len(wd._problems) == 0

    def test_failed_pod_creates_critical(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Failed")
        wd._analyze_pods([pod])

        assert "aiperf-worker-abc" in wd._completed_pods
        crits = [p for p in wd._problems if p.category == "pod-failed"]
        assert len(crits) == 1

    def test_completed_pod_not_reported_twice(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        pod = _make_pod(name="aiperf-worker-abc", phase="Failed")
        wd._analyze_pods([pod])
        wd._analyze_pods([pod])

        crits = [p for p in wd._problems if p.category == "pod-failed"]
        assert len(crits) == 1


# ============================================================
# BenchmarkWatchdog - Timeout Prediction
# ============================================================


class TestBenchmarkWatchdogTimeoutPrediction:
    """Verify escalating timeout warnings."""

    def test_warning_when_less_than_60s_remaining(
        self, source: FakeDataSource, time_traveler: TimeTraveler
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=100)
        wd._start_time = time.time()
        time_traveler.advance_time(50)

        wd._check_elapsed_time()
        assert len(wd._problems) == 1
        assert wd._problems[0].category == "timeout-warning"

    def test_critical_when_less_than_15s_remaining(
        self, source: FakeDataSource, time_traveler: TimeTraveler
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=100)
        wd._start_time = time.time()
        time_traveler.advance_time(90)

        wd._check_elapsed_time()

        categories = {p.category for p in wd._problems}
        assert "timeout-imminent" in categories

    def test_no_warning_when_plenty_of_time(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=300)
        wd._start_time = time.time()

        wd._check_elapsed_time()
        assert len(wd._problems) == 0

    def test_timeout_warnings_not_repeated(
        self, source: FakeDataSource, time_traveler: TimeTraveler
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=100)
        wd._start_time = time.time()
        time_traveler.advance_time(50)

        wd._check_elapsed_time()
        wd._check_elapsed_time()

        warnings = [p for p in wd._problems if p.category == "timeout-warning"]
        assert len(warnings) == 1


# ============================================================
# BenchmarkWatchdog - Stale Namespaces
# ============================================================


class TestBenchmarkWatchdogStaleNamespaces:
    """Verify stale namespace detection."""

    @pytest.mark.asyncio
    async def test_many_stale_namespaces_creates_warning(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "aiperf-benchmarks")
        wd._start_time = time.time()

        source.namespaces = [
            "aiperf-benchmarks",
            "aiperf-system",
            "aiperf-old-1",
            "aiperf-old-2",
            "aiperf-old-3",
            "aiperf-old-4",
            "aiperf-old-5",
            "aiperf-old-6",
            "default",
        ]
        await wd._check_stale_namespaces()

        assert wd._stale_ns_count == 6
        warns = [p for p in wd._problems if p.category == "stale-namespaces"]
        assert len(warns) == 1

    @pytest.mark.asyncio
    async def test_few_stale_namespaces_no_warning(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "aiperf-benchmarks")
        wd._start_time = time.time()

        source.namespaces = [
            "aiperf-benchmarks",
            "aiperf-old-1",
            "aiperf-system",
            "default",
        ]
        await wd._check_stale_namespaces()

        assert wd._stale_ns_count == 1
        assert len(wd._problems) == 0

    @pytest.mark.asyncio
    async def test_no_stale_namespaces(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "aiperf-benchmarks")
        wd._start_time = time.time()

        source.namespaces = ["aiperf-benchmarks", "aiperf-system", "default"]
        await wd._check_stale_namespaces()

        assert wd._stale_ns_count == 0
        assert len(wd._problems) == 0

    @pytest.mark.asyncio
    async def test_default_benchmark_namespace_excluded(
        self, source: FakeDataSource
    ) -> None:
        """Verify aiperf-benchmarks is never counted as stale."""
        wd = BenchmarkWatchdog(source, "aiperf-custom")
        wd._start_time = time.time()

        source.namespaces = [
            "aiperf-custom",
            "aiperf-benchmarks",
            "aiperf-system",
            "default",
        ]
        await wd._check_stale_namespaces()

        assert wd._stale_ns_count == 0
        assert len(wd._problems) == 0


# ============================================================
# BenchmarkWatchdog - Node Resources
# ============================================================


class TestBenchmarkWatchdogNodeResources:
    """Verify node resource checking."""

    @pytest.mark.asyncio
    async def test_check_node_resources_logs_gpu_total(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.nodes = [
            NodeResources(
                name="node-1",
                allocatable_cpu="64",
                allocatable_memory="256Gi",
                allocatable_gpu=4,
            ),
            NodeResources(
                name="node-2",
                allocatable_cpu="64",
                allocatable_memory="256Gi",
                allocatable_gpu=4,
            ),
        ]
        await wd._check_node_resources()
        # No problems expected, just verifying it doesn't crash

    @pytest.mark.asyncio
    async def test_check_node_resources_handles_empty_nodes(
        self, source: FakeDataSource
    ) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        source.nodes = []
        await wd._check_node_resources()
        assert len(wd._problems) == 0


# ============================================================
# BenchmarkWatchdog - Report Generation
# ============================================================


class TestBenchmarkWatchdogReport:
    """Verify report generation from watchdog state."""

    def test_report_reflects_current_state(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=300)
        wd._start_time = time.time()

        pod_ok = _make_pod(name="pod-ok", phase="Succeeded")
        pod_fail = _make_pod(name="pod-fail", phase="Failed")
        wd._analyze_pods([pod_ok, pod_fail])

        report = wd.report
        assert report.namespace == "test-ns"
        assert report.timeout == 300.0
        assert report.succeeded_count == 1
        assert report.failed_count == 1
        assert "pod-fail" in report.completed_pods
        assert "pod-ok" in report.completed_pods
        assert report.has_critical is True  # pod-failed problem

    def test_report_has_critical_property(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        wd._start_time = time.time()

        assert wd.has_critical is False

        wd._add_problem(ProblemSeverity.CRITICAL, "test", "boom")
        assert wd.has_critical is True

    def test_problems_property_returns_copy(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns")
        problems = wd.problems
        problems.append(
            WatchdogProblem(severity=ProblemSeverity.INFO, category="x", message="y")
        )
        assert len(wd.problems) == 0


# ============================================================
# BenchmarkWatchdog - Monitor Loop Integration
# ============================================================


class TestBenchmarkWatchdogMonitorLoop:
    """Verify the monitor loop orchestrates checks correctly."""

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_none(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=300)
        wd._start_time = time.time()
        source.get_pods_error = RuntimeError("api down")

        result = await wd._fetch_pods()
        assert result is None
        assert len(wd._pod_timelines) == 0

    def test_full_lifecycle_with_pod_transitions(self, source: FakeDataSource) -> None:
        wd = BenchmarkWatchdog(source, "test-ns", timeout=300)
        wd._start_time = time.time()

        wd._analyze_pods(
            [_make_pod(name="aiperf-worker-abc", phase="Pending", ready=False)]
        )
        wd._analyze_pods([_make_pod(name="aiperf-worker-abc", phase="Running")])
        wd._analyze_pods([_make_pod(name="aiperf-worker-abc", phase="Succeeded")])

        report = wd.report
        assert "aiperf-worker-abc" in report.pod_timelines
        tl = report.pod_timelines["aiperf-worker-abc"]
        assert tl.last_phase == "Succeeded"
        assert "aiperf-worker-abc" in report.completed_pods

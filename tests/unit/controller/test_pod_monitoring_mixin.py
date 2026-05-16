# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``PodMonitoringMixin``.

Exercises the pure pod-tracking / threshold / status-query logic without
spinning up a full ``KubernetesServiceManager``. Methods that touch
``kubernetes_asyncio`` live on the host class and are intentionally NOT
tested here (covered by ``test_kubernetes_service_manager.py``).

A ``_FakeHost`` mimics the host attributes the mixin reads/writes: ``_pods``
dict, ``_restart_warned`` set, the ``pod_failure_abort_event`` /
``pod_failure_abort_reason`` slots, the ``required_services`` map, and the
logger-mixin facade methods (``error``/``warning``/``debug``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller._pod_monitoring_mixin import PodMonitoringMixin
from aiperf.controller.kubernetes_pod_helpers import PodInfo, PodSnapshot
from aiperf.kubernetes.enums import PodPhase
from aiperf.plugin.enums import ServiceType

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeHost(PodMonitoringMixin):
    """Minimal KubernetesServiceManager-shaped host for the mixin under test."""

    _pods: dict[str, PodInfo] = field(default_factory=dict)
    _restart_warned: set[str] = field(default_factory=set)
    required_services: dict[ServiceType, int] = field(default_factory=dict)
    pod_failure_abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    pod_failure_abort_reason: str = ""

    # Logger facade — record calls instead of asserting log content per-test.
    error_calls: list[str] = field(default_factory=list)
    warning_calls: list[str] = field(default_factory=list)
    debug_calls: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.error_calls.append(msg)

    def warning(self, msg: str) -> None:
        self.warning_calls.append(msg)

    def debug(self, msg: Any) -> None:  # noqa: ANN401  - matches LoggerMixin facade
        self.debug_calls.append(str(msg))


def _snapshot(
    pod_name: str,
    phase: PodPhase,
    *,
    container_statuses: list[dict] | None = None,
    status: dict | None = None,
) -> PodSnapshot:
    """Build a ``PodSnapshot`` 4-tuple for tests."""
    return (
        pod_name,
        phase,
        container_statuses or [],
        status or {"conditions": [], "containerStatuses": container_statuses or []},
    )


@pytest.fixture(autouse=True)
def _reset_service_registry() -> None:
    """ServiceRegistry is a module-level singleton; reset per test."""
    ServiceRegistry.reset()
    yield
    ServiceRegistry.reset()


# ---------------------------------------------------------------------------
# Pod-state queries
# ---------------------------------------------------------------------------


class TestPodStateQueries:
    """Verify the public ``get_*`` methods reflect mutations to ``_pods``."""

    def test_get_pod_info_returns_none_for_unknown_index(self) -> None:
        host = _FakeHost()
        assert host.get_pod_info("99") is None

    def test_get_pod_info_returns_tracked_entry(self) -> None:
        host = _FakeHost()
        info = PodInfo(pod_index="0", pod_name="pod-0")
        host._pods["0"] = info
        assert host.get_pod_info("0") is info

    def test_get_all_pod_info_returns_copy(self) -> None:
        host = _FakeHost()
        host._pods["0"] = PodInfo(pod_index="0", pod_name="pod-0")
        snapshot = host.get_all_pod_info()
        assert snapshot == host._pods
        # The returned dict is a copy — mutating it must not mutate the host.
        snapshot["1"] = PodInfo(pod_index="1", pod_name="pod-1")
        assert "1" not in host._pods

    def test_get_failed_pods_filters_to_failed_only(self) -> None:
        host = _FakeHost()
        host._pods = {
            "0": PodInfo(pod_index="0", pod_name="pod-0", failed=False),
            "1": PodInfo(pod_index="1", pod_name="pod-1", failed=True),
            "2": PodInfo(pod_index="2", pod_name="pod-2", failed=True),
        }
        failed = host.get_failed_pods()
        assert {p.pod_index for p in failed} == {"1", "2"}

    def test_get_failed_pods_empty_when_none_failed(self) -> None:
        host = _FakeHost()
        host._pods["0"] = PodInfo(pod_index="0", pod_name="pod-0")
        assert host.get_failed_pods() == []


class TestPodSummary:
    """``get_pod_summary`` should produce a friendly per-pod status string."""

    def test_phase_only_when_no_restarts_no_issues(self) -> None:
        host = _FakeHost()
        host._pods["0"] = PodInfo(
            pod_index="0",
            pod_name="pod-0",
            phase=PodPhase.RUNNING,
        )
        summary = host.get_pod_summary()
        assert summary == {"0": "Running"}

    def test_includes_restart_count_when_nonzero(self) -> None:
        host = _FakeHost()
        host._pods["0"] = PodInfo(
            pod_index="0",
            pod_name="pod-0",
            phase=PodPhase.RUNNING,
            restart_count=4,
        )
        assert "restarts=4" in host.get_pod_summary()["0"]

    def test_includes_container_issues_when_present(self) -> None:
        host = _FakeHost()
        host._pods["0"] = PodInfo(
            pod_index="0",
            pod_name="pod-0",
            phase=PodPhase.PENDING,
            container_issues=["OOMKilled", "CrashLoopBackOff"],
        )
        line = host.get_pod_summary()["0"]
        assert "Pending" in line
        assert "OOMKilled" in line
        assert "CrashLoopBackOff" in line
        assert "issues=" in line

    def test_empty_pods_yields_empty_summary(self) -> None:
        host = _FakeHost()
        assert host.get_pod_summary() == {}

    def test_summary_covers_all_tracked_pods(self) -> None:
        host = _FakeHost()
        host._pods = {
            "0": PodInfo(pod_index="0", pod_name="p0", phase=PodPhase.RUNNING),
            "1": PodInfo(pod_index="1", pod_name="p1", phase=PodPhase.PENDING),
        }
        summary = host.get_pod_summary()
        assert set(summary.keys()) == {"0", "1"}


# ---------------------------------------------------------------------------
# _check_pod_failure_threshold
# ---------------------------------------------------------------------------


class TestCheckPodFailureThreshold:
    """Verify the abort-event gating logic without env mutations beyond
    monkeypatch on ``Environment.SERVICE.POD_FAILURE_ABORT_THRESHOLD_PERCENT``.
    """

    def test_no_failed_pods_does_not_set_abort_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.SERVICE, "POD_FAILURE_ABORT_THRESHOLD_PERCENT", 50
        )
        host = _FakeHost(
            required_services={ServiceType.WORKER_GROUP_MANAGER: 4},
        )
        host._pods = {
            "0": PodInfo(pod_index="0", pod_name="p0", failed=False),
        }
        host._check_pod_failure_threshold()
        assert not host.pod_failure_abort_event.is_set()

    def test_threshold_zero_disables_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.SERVICE, "POD_FAILURE_ABORT_THRESHOLD_PERCENT", 0
        )
        host = _FakeHost(
            required_services={ServiceType.WORKER_GROUP_MANAGER: 2},
        )
        host._pods = {
            "0": PodInfo(pod_index="0", pod_name="p0", failed=True),
            "1": PodInfo(pod_index="1", pod_name="p1", failed=True),
        }
        host._check_pod_failure_threshold()
        assert not host.pod_failure_abort_event.is_set()

    def test_already_set_abort_event_returns_early(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.SERVICE, "POD_FAILURE_ABORT_THRESHOLD_PERCENT", 50
        )
        host = _FakeHost(
            required_services={ServiceType.WORKER_GROUP_MANAGER: 2},
            pod_failure_abort_reason="prior reason",
        )
        host.pod_failure_abort_event.set()
        host._pods = {
            "0": PodInfo(pod_index="0", pod_name="p0", failed=True),
            "1": PodInfo(pod_index="1", pod_name="p1", failed=True),
        }
        host._check_pod_failure_threshold()
        # Reason untouched, no error emitted.
        assert host.pod_failure_abort_reason == "prior reason"
        assert host.error_calls == []

    def test_aborts_when_percentage_meets_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.SERVICE, "POD_FAILURE_ABORT_THRESHOLD_PERCENT", 50
        )
        host = _FakeHost(
            required_services={ServiceType.WORKER_GROUP_MANAGER: 4},
        )
        host._pods = {
            "0": PodInfo(pod_index="0", pod_name="p0", failed=True),
            "1": PodInfo(pod_index="1", pod_name="p1", failed=True),
        }
        host._check_pod_failure_threshold()
        assert host.pod_failure_abort_event.is_set()
        assert "2/4" in host.pod_failure_abort_reason
        assert "50%" in host.pod_failure_abort_reason
        assert host.error_calls

    def test_does_not_abort_below_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.SERVICE, "POD_FAILURE_ABORT_THRESHOLD_PERCENT", 75
        )
        host = _FakeHost(
            required_services={ServiceType.WORKER_GROUP_MANAGER: 4},
        )
        host._pods = {
            "0": PodInfo(pod_index="0", pod_name="p0", failed=True),
            "1": PodInfo(pod_index="1", pod_name="p1", failed=True),
        }
        host._check_pod_failure_threshold()
        assert not host.pod_failure_abort_event.is_set()

    def test_falls_back_to_tracked_pod_count_when_no_expected_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``required_services`` lacks WORKER_GROUP_MANAGER, the mixin
        derives the denominator from ``len(self._pods)``."""
        monkeypatch.setattr(
            Environment.SERVICE, "POD_FAILURE_ABORT_THRESHOLD_PERCENT", 100
        )
        host = _FakeHost(required_services={})
        host._pods = {
            "0": PodInfo(pod_index="0", pod_name="p0", failed=True),
        }
        host._check_pod_failure_threshold()
        assert host.pod_failure_abort_event.is_set()
        assert "1/1" in host.pod_failure_abort_reason

    def test_zero_total_pods_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            Environment.SERVICE, "POD_FAILURE_ABORT_THRESHOLD_PERCENT", 100
        )
        host = _FakeHost(required_services={})
        host._pods = {}
        host._check_pod_failure_threshold()
        assert not host.pod_failure_abort_event.is_set()


# ---------------------------------------------------------------------------
# _update_pod_tracking
# ---------------------------------------------------------------------------


class TestUpdatePodTracking:
    """Per-pod upserts and warning-once behaviour."""

    def test_creates_new_pod_info_on_first_observation(self) -> None:
        host = _FakeHost()
        info = host._update_pod_tracking(
            "0",
            "pod-0",
            phase=PodPhase.RUNNING,
            container_statuses=[],
            now_ns=1_000,
        )
        assert host._pods["0"] is info
        assert info.pod_name == "pod-0"
        assert info.phase == PodPhase.RUNNING
        assert info.last_checked_ns == 1_000
        assert info.restart_count == 0
        assert info.container_issues == []

    def test_updates_existing_pod_info_in_place(self) -> None:
        host = _FakeHost()
        host._pods["0"] = PodInfo(
            pod_index="0",
            pod_name="pod-0-old",
            phase=PodPhase.PENDING,
        )
        info = host._update_pod_tracking(
            "0",
            "pod-0-new",
            phase=PodPhase.RUNNING,
            container_statuses=[{"name": "c", "restartCount": 1, "state": {}}],
            now_ns=5_000,
        )
        assert info is host._pods["0"]
        assert info.pod_name == "pod-0-new"
        assert info.phase == PodPhase.RUNNING
        assert info.restart_count == 1
        assert info.last_checked_ns == 5_000

    def test_aggregates_restart_count_across_containers(self) -> None:
        host = _FakeHost()
        info = host._update_pod_tracking(
            "0",
            "pod-0",
            phase=PodPhase.RUNNING,
            container_statuses=[
                {"name": "c1", "restartCount": 2, "state": {}},
                {"name": "c2", "restartCount": 3, "state": {}},
            ],
            now_ns=0,
        )
        assert info.restart_count == 5

    def test_extracts_container_issues(self) -> None:
        host = _FakeHost()
        info = host._update_pod_tracking(
            "0",
            "pod-0",
            phase=PodPhase.PENDING,
            container_statuses=[
                {
                    "name": "c1",
                    "restartCount": 0,
                    "state": {"waiting": {"reason": "ImagePullBackOff", "message": ""}},
                }
            ],
            now_ns=0,
        )
        assert "ImagePullBackOff" in info.container_issues

    def test_warns_once_at_three_restarts(self) -> None:
        host = _FakeHost()
        cs = [{"name": "c", "restartCount": 3, "state": {}}]
        host._update_pod_tracking(
            "0", "pod-0", phase=PodPhase.RUNNING, container_statuses=cs, now_ns=0
        )
        assert len(host.warning_calls) == 1
        # Second call with same pod_index must NOT re-warn.
        host._update_pod_tracking(
            "0", "pod-0", phase=PodPhase.RUNNING, container_statuses=cs, now_ns=1
        )
        assert len(host.warning_calls) == 1
        assert "0" in host._restart_warned

    def test_does_not_warn_below_three_restarts(self) -> None:
        host = _FakeHost()
        host._update_pod_tracking(
            "0",
            "pod-0",
            phase=PodPhase.RUNNING,
            container_statuses=[{"name": "c", "restartCount": 2, "state": {}}],
            now_ns=0,
        )
        assert host.warning_calls == []
        assert host._restart_warned == set()

    def test_logs_debug_when_running_with_issues(self) -> None:
        host = _FakeHost()
        host._update_pod_tracking(
            "0",
            "pod-0",
            phase=PodPhase.RUNNING,
            container_statuses=[
                {
                    "name": "c1",
                    "restartCount": 0,
                    "state": {"waiting": {"reason": "CrashLoopBackOff"}},
                }
            ],
            now_ns=0,
        )
        assert host.debug_calls
        assert any("CrashLoopBackOff" in m for m in host.debug_calls)


# ---------------------------------------------------------------------------
# _handle_terminal_pod / _process_pod_snapshots
# ---------------------------------------------------------------------------


class TestHandleTerminalPod:
    """Terminal pods get marked failed exactly once and have services failed."""

    def test_running_pod_is_not_marked_failed(self) -> None:
        host = _FakeHost()
        info = PodInfo(pod_index="0", pod_name="p0", phase=PodPhase.RUNNING)
        host._pods["0"] = info
        host._handle_terminal_pod(
            info,
            "0",
            "p0",
            phase=PodPhase.RUNNING,
            container_statuses=[],
            status={"conditions": []},
        )
        assert info.failed is False

    @pytest.mark.parametrize(
        "phase",
        [
            param(PodPhase.FAILED, id="failed"),
            param(PodPhase.UNKNOWN, id="unknown"),
        ],
    )  # fmt: skip
    def test_terminal_phases_mark_pod_failed(self, phase: PodPhase) -> None:
        host = _FakeHost()
        info = PodInfo(pod_index="0", pod_name="p0", phase=phase)
        host._pods["0"] = info
        with patch.object(host, "_fail_pod_services") as mock_fail:
            host._handle_terminal_pod(
                info,
                "0",
                "p0",
                phase=phase,
                container_statuses=[],
                status={"conditions": []},
            )
        assert info.failed is True
        mock_fail.assert_called_once_with("0", "p0", phase)

    def test_does_not_re_fail_already_failed_pod(self) -> None:
        host = _FakeHost()
        info = PodInfo(pod_index="0", pod_name="p0", phase=PodPhase.FAILED, failed=True)
        host._pods["0"] = info
        with patch.object(host, "_fail_pod_services") as mock_fail:
            host._handle_terminal_pod(
                info,
                "0",
                "p0",
                phase=PodPhase.FAILED,
                container_statuses=[],
                status={"conditions": []},
            )
        mock_fail.assert_not_called()


class TestProcessPodSnapshots:
    """End-to-end orchestration of upsert + terminal handling."""

    def test_processes_mixed_pending_and_running_pods(self) -> None:
        host = _FakeHost()
        snapshots: dict[str, PodSnapshot] = {
            "0": _snapshot("p0", PodPhase.PENDING),
            "1": _snapshot("p1", PodPhase.RUNNING),
        }
        with patch.object(host, "_fail_pod_services"):
            host._process_pod_snapshots(snapshots, now_ns=10)

        assert set(host._pods) == {"0", "1"}
        assert all(p.failed is False for p in host._pods.values())
        assert host._pods["0"].phase == PodPhase.PENDING
        assert host._pods["1"].phase == PodPhase.RUNNING

    def test_processes_all_failed_pods(self) -> None:
        host = _FakeHost()
        snapshots = {
            "0": _snapshot("p0", PodPhase.FAILED),
            "1": _snapshot("p1", PodPhase.FAILED),
        }
        with patch.object(host, "_fail_pod_services") as mock_fail:
            host._process_pod_snapshots(snapshots, now_ns=10)

        assert all(p.failed for p in host._pods.values())
        assert mock_fail.call_count == 2

    def test_empty_snapshots_no_op(self) -> None:
        host = _FakeHost()
        host._process_pod_snapshots({}, now_ns=0)
        assert host._pods == {}

    def test_pod_transitioning_running_then_failed_marks_failed(self) -> None:
        host = _FakeHost()
        with patch.object(host, "_fail_pod_services") as mock_fail:
            host._process_pod_snapshots(
                {"0": _snapshot("p0", PodPhase.RUNNING)}, now_ns=1
            )
            assert host._pods["0"].failed is False
            mock_fail.assert_not_called()

            host._process_pod_snapshots(
                {"0": _snapshot("p0", PodPhase.FAILED)}, now_ns=2
            )
            assert host._pods["0"].failed is True
            mock_fail.assert_called_once()


# ---------------------------------------------------------------------------
# _raise_for_any_failed_pod
# ---------------------------------------------------------------------------


class TestRaiseForAnyFailedPod:
    """Pre-PROFILE_START gate: any terminal pod immediately raises."""

    def test_no_terminal_pods_does_not_fail_any_services(self) -> None:
        host = _FakeHost()
        with patch.object(host, "_fail_pod_services") as mock_fail:
            host._raise_for_any_failed_pod(
                {
                    "0": _snapshot("p0", PodPhase.PENDING),
                    "1": _snapshot("p1", PodPhase.RUNNING),
                }
            )
        mock_fail.assert_not_called()

    def test_terminal_pod_invokes_fail_and_registry_raise(self) -> None:
        host = _FakeHost()
        with (
            patch.object(host, "_fail_pod_services") as mock_fail,
            patch.object(ServiceRegistry, "_raise_on_failure") as mock_raise,
        ):
            host._raise_for_any_failed_pod({"0": _snapshot("p0", PodPhase.FAILED)})
        mock_fail.assert_called_once_with("0")
        mock_raise.assert_called_once()
        assert host.error_calls
        assert "PROFILE_START" in host.error_calls[0]

    def test_unknown_phase_treated_as_terminal(self) -> None:
        host = _FakeHost()
        with (
            patch.object(host, "_fail_pod_services") as mock_fail,
            patch.object(ServiceRegistry, "_raise_on_failure"),
        ):
            host._raise_for_any_failed_pod({"0": _snapshot("p0", PodPhase.UNKNOWN)})
        mock_fail.assert_called_once()


# ---------------------------------------------------------------------------
# _fail_pod_services
# ---------------------------------------------------------------------------


class TestFailPodServices:
    """Translate pod_index -> registered service ids and fail them."""

    def test_no_services_for_pod_logs_warning(self) -> None:
        host = _FakeHost()
        host._fail_pod_services("0")
        assert host.warning_calls
        assert "pod_index=0" in host.warning_calls[0]

    def test_fails_each_registered_service(self) -> None:
        host = _FakeHost()
        from aiperf.common.enums import LifecycleState

        ServiceRegistry.expect_services({ServiceType.WORKER: 2})
        ServiceRegistry.register(
            "w_0",
            ServiceType.WORKER,
            first_seen_ns=1,
            state=LifecycleState.RUNNING,
            pod_index="0",
        )
        ServiceRegistry.register(
            "w_1",
            ServiceType.WORKER,
            first_seen_ns=2,
            state=LifecycleState.RUNNING,
            pod_index="0",
        )

        with patch.object(ServiceRegistry, "fail_service") as mock_fail:
            host._fail_pod_services("0", pod_name="pod-0", phase=PodPhase.FAILED)

        assert mock_fail.call_count == 2
        # Each call must include the service_id from the registry.
        failed_ids = {c.args[0] for c in mock_fail.call_args_list}
        assert failed_ids == {"w_0", "w_1"}
        # Warnings emitted (one per service marked failed).
        assert len(host.warning_calls) == 2
        assert all("pod-0" in m for m in host.warning_calls)

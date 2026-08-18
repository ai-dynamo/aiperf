# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A single unhealthy pod snapshot must not permanently evict a live producer.

Reaping is irreversible in practice: ``ResultJoinCoordinator.complete_domain``
returns early once a domain has been popped, so a pod dropped on one bad poll
never rejoins the run's results. The heartbeat watchdog has required two
consecutive strikes since the 285-worker false-batch-expiry incident; the pod
monitor did not, so an apiserver that momentarily could not reach a kubelet
(``phase == Unknown``) evicted a healthy, still-benchmarking pod.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.controller._pod_monitoring_mixin import PodMonitoringMixin
from aiperf.controller.kubernetes_pod_helpers import PodInfo
from aiperf.kubernetes.enums import PodPhase


class _Monitor(PodMonitoringMixin):
    """Minimal host for the mixin's pure health-evaluation logic."""

    def __init__(self) -> None:
        self._pods: dict[str, PodInfo] = {}
        self._restart_warned: set[str] = set()
        self.failed_pods: list[str] = []
        self.debug = MagicMock()
        self.info = MagicMock()
        self.warning = MagicMock()
        self.error = MagicMock()

    def _fail_pod_services(self, pod_index, pod_name=None, phase=None, *, fatal=False):  # noqa: ANN001, ANN202, ARG002
        self.failed_pods.append(pod_index)


def _observe(monitor: _Monitor, phase: PodPhase, *, waiting_reason: str = "") -> None:
    """Feed one pod snapshot through tracking + health evaluation."""
    container_statuses = (
        [{"name": "worker", "state": {"waiting": {"reason": waiting_reason}}}]
        if waiting_reason
        else [{"name": "worker", "state": {"running": {}}}]
    )
    pod_info = monitor._update_pod_tracking(
        "0",
        "aiperf-workers-0",
        phase=phase,
        container_statuses=container_statuses,
        now_ns=0,
    )
    monitor._evaluate_pod_health(
        pod_info,
        "0",
        "aiperf-workers-0",
        phase=phase,
        container_statuses=container_statuses,
        status={},
    )


def test_a_single_unknown_snapshot_does_not_evict() -> None:
    """The kubelet blip: one Unknown poll, then the pod reports Running again."""
    monitor = _Monitor()

    _observe(monitor, PodPhase.UNKNOWN)

    assert monitor.failed_pods == []
    assert monitor._pods["0"].failed is False


def test_a_recovered_pod_loses_its_strike() -> None:
    """Two Unknown polls separated by a healthy one are not two consecutive."""
    monitor = _Monitor()

    _observe(monitor, PodPhase.UNKNOWN)
    _observe(monitor, PodPhase.RUNNING)
    _observe(monitor, PodPhase.UNKNOWN)

    assert monitor.failed_pods == []


def test_two_consecutive_unknown_snapshots_evict() -> None:
    """Confirmation delays detection by one poll; it must not prevent it."""
    monitor = _Monitor()

    _observe(monitor, PodPhase.UNKNOWN)
    _observe(monitor, PodPhase.UNKNOWN)

    assert monitor.failed_pods == ["0"]
    assert monitor._pods["0"].failed is True


@pytest.mark.parametrize(
    "phase,waiting_reason",
    [
        param(PodPhase.FAILED, "", id="terminal_failed_phase"),
        param(PodPhase.PENDING, "ImagePullBackOff", id="blocked_image_pull"),
        param(PodPhase.RUNNING, "CrashLoopBackOff", id="blocked_crash_loop"),
    ],
)  # fmt: skip
def test_authoritative_failures_are_not_delayed(
    phase: PodPhase, waiting_reason: str
) -> None:
    """Only Unknown is sampling noise.

    ``Failed`` is terminal by definition and blocked container reasons are
    latched kubelet states, so both keep their original one-poll latency.
    """
    monitor = _Monitor()

    _observe(monitor, phase, waiting_reason=waiting_reason)

    assert monitor.failed_pods == ["0"]


def test_a_replacement_pod_generation_starts_with_a_clean_strike_count() -> None:
    """A new pod under the same index must not inherit its predecessor's strike."""
    monitor = _Monitor()
    _observe(monitor, PodPhase.UNKNOWN)

    pod_info = monitor._update_pod_tracking(
        "0",
        "aiperf-workers-0-replacement",
        phase=PodPhase.UNKNOWN,
        container_statuses=[],
        now_ns=1,
    )
    monitor._evaluate_pod_health(
        pod_info,
        "0",
        "aiperf-workers-0-replacement",
        phase=PodPhase.UNKNOWN,
        container_statuses=[],
        status={},
    )

    assert monitor.failed_pods == []

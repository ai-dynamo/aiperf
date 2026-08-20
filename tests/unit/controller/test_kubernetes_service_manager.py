# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kubernetes service manager and its pod-monitoring helpers."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import LifecycleState
from aiperf.common.exceptions import ServiceProcessDiedError
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.base_service_manager import BaseServiceManager
from aiperf.controller.kubernetes_pod_helpers import (
    PodInfo,
    aggregate_pods_by_index,
    extract_container_issues,
    format_pod_failure_reason,
)
from aiperf.controller.kubernetes_service_manager import (
    EXTERNAL_K8S_SERVICES,
    KubernetesServiceManager,
)
from aiperf.kubernetes.constants import JobSetLabels
from aiperf.kubernetes.enums import PodPhase
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType, ServiceType


def _fake_pod(
    name: str,
    pod_index: str,
    phase: str,
    *,
    replicated_job: str = "workers",
    restart_count: int = 0,
    waiting_reason: str | None = None,
    terminated_reason: str | None = None,
) -> MagicMock:
    """Build a kubernetes_asyncio-shaped pod double."""
    pod = MagicMock()
    pod.metadata.name = name
    pod.metadata.labels = {
        JobSetLabels.POD_INDEX: pod_index,
        JobSetLabels.REPLICATED_JOB_NAME: replicated_job,
    }
    pod.status.phase = phase
    cs = MagicMock()
    cs.name = "worker"
    cs.restart_count = restart_count
    cs.state.waiting = None
    cs.state.terminated = None
    cs.last_state = None
    if waiting_reason is not None:
        waiting = MagicMock()
        waiting.reason = waiting_reason
        waiting.message = "back-off restarting"
        cs.state.waiting = waiting
    if terminated_reason is not None:
        terminated = MagicMock()
        terminated.reason = terminated_reason
        terminated.message = "killed"
        terminated.exit_code = 137
        cs.state.terminated = terminated
    pod.status.container_statuses = [cs]
    pod.status.conditions = []
    return pod


class TestPluginRegistration:
    """The KUBERNETES run type is materialized by the plugin registration."""

    def test_kubernetes_run_type_is_registered(self) -> None:
        assert getattr(ServiceRunType, "KUBERNETES", None) is not None

    def test_kubernetes_run_type_resolves_to_kubernetes_manager(self) -> None:
        cls = plugins.get_class(PluginType.SERVICE_MANAGER, ServiceRunType.KUBERNETES)
        assert cls is KubernetesServiceManager

    def test_multiprocessing_manager_still_resolves(self) -> None:
        """Registering kubernetes must not disturb the default run type.

        The unit-test harness overrides this slot with a fake at a higher
        priority, so assert the slot resolves rather than pinning the class.
        """
        cls = plugins.get_class(
            PluginType.SERVICE_MANAGER, ServiceRunType.MULTIPROCESSING
        )
        assert cls is not KubernetesServiceManager
        assert issubclass(cls, BaseServiceManager)

    def test_system_controller_getattr_probe_sees_the_member(self) -> None:
        """The probe at system_controller.py must now resolve, not fall back."""
        from aiperf.plugin import enums

        assert getattr(enums.ServiceRunType, "KUBERNETES", None) == "kubernetes"


class TestExternalServiceHandling:
    @pytest.fixture
    def manager(self, benchmark_run) -> KubernetesServiceManager:
        return KubernetesServiceManager(
            required_services={ServiceType.WORKER: 2},
            run=benchmark_run,
        )

    def test_worker_is_external(self, manager: KubernetesServiceManager) -> None:
        assert manager._is_external_service(ServiceType.WORKER)
        assert ServiceType.RECORD_PROCESSOR in EXTERNAL_K8S_SERVICES

    @pytest.mark.asyncio
    async def test_run_service_does_not_spawn_a_process_for_external_services(
        self, manager: KubernetesServiceManager
    ) -> None:
        await manager.run_service(ServiceType.WORKER, num_replicas=3)
        assert manager.multi_process_info == []
        assert ServiceRegistry.expected_by_type[ServiceType.WORKER] == 3

    @pytest.mark.asyncio
    async def test_stop_service_is_a_noop_for_external_services(
        self, manager: KubernetesServiceManager
    ) -> None:
        assert await manager.stop_service(ServiceType.WORKER) == []

    @pytest.mark.asyncio
    async def test_registration_wait_uses_required_services_when_no_subprocesses(
        self, manager: KubernetesServiceManager
    ) -> None:
        """The K8s gate waits for the expected count, not one service per type."""
        await manager.run_service(ServiceType.WORKER, num_replicas=2)
        ServiceRegistry.register(
            "worker-0",
            ServiceType.WORKER,
            first_seen_ns=1,
            state=LifecycleState.RUNNING,
        )
        with pytest.raises(Exception, match="register"):
            await manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(), timeout_seconds=0.2
            )


class TestPodHelpers:
    def test_extract_container_issues_collects_waiting_and_terminated(self) -> None:
        statuses = [
            {
                "name": "worker",
                "restartCount": 4,
                "state": {"waiting": {"reason": "CrashLoopBackOff", "message": ""}},
                "lastState": {"terminated": {"reason": "OOMKilled"}},
            }
        ]
        assert extract_container_issues(statuses) == [
            "CrashLoopBackOff",
            "OOMKilled",
        ]

    def test_format_pod_failure_reason_includes_exit_code(self) -> None:
        reason = format_pod_failure_reason(
            "aiperf-w-0",
            PodPhase.FAILED,
            [
                {
                    "name": "worker",
                    "state": {
                        "terminated": {
                            "reason": "OOMKilled",
                            "message": "out of memory",
                            "exitCode": 137,
                        }
                    },
                }
            ],
            {"conditions": []},
        )
        assert "aiperf-w-0" in reason
        assert "exit_code=137" in reason
        assert "OOMKilled" in reason

    def test_aggregate_prefers_non_terminal_snapshot_per_index(self) -> None:
        pods = [
            _fake_pod("w-0-old", "0", "Failed"),
            _fake_pod("w-0-new", "0", "Running"),
        ]
        by_index = aggregate_pods_by_index(pods)
        assert by_index["0"][0] == "w-0-new"
        assert by_index["0"][1] == PodPhase.RUNNING

    def test_aggregate_skips_non_worker_replicated_jobs(self) -> None:
        pods = [_fake_pod("ctl-0", "0", "Running", replicated_job="controller")]
        assert aggregate_pods_by_index(pods) == {}


class TestPodMonitoring:
    @pytest.fixture
    def manager(self, benchmark_run) -> KubernetesServiceManager:
        mgr = KubernetesServiceManager(
            required_services={ServiceType.WORKER_GROUP_MANAGER: 2},
            run=benchmark_run,
        )
        mgr._pod_monitoring_active = True
        return mgr

    def test_terminal_pod_is_marked_failed_once(
        self, manager: KubernetesServiceManager
    ) -> None:
        snapshots = aggregate_pods_by_index([_fake_pod("w-0", "0", "Failed")])
        manager._process_pod_snapshots(snapshots, now_ns=1)
        pod = manager.get_pod_info("0")
        assert pod is not None and pod.failed
        assert manager.get_failed_pods() == [pod]

    def test_running_pod_is_not_failed(self, manager: KubernetesServiceManager) -> None:
        snapshots = aggregate_pods_by_index([_fake_pod("w-0", "0", "Running")])
        manager._process_pod_snapshots(snapshots, now_ns=1)
        assert manager.get_failed_pods() == []
        assert manager.get_pod_summary() == {"0": "Running"}

    def test_replacement_pod_clears_predecessor_failure(
        self, manager: KubernetesServiceManager
    ) -> None:
        manager._process_pod_snapshots(
            aggregate_pods_by_index([_fake_pod("w-0-old", "0", "Failed")]),
            now_ns=1,
        )
        assert manager.get_pod_info("0").failed

        manager._process_pod_snapshots(
            aggregate_pods_by_index([_fake_pod("w-0-new", "0", "Running")]),
            now_ns=2,
        )

        pod = manager.get_pod_info("0")
        assert pod is not None
        assert pod.pod_name == "w-0-new"
        assert not pod.failed

    def test_blocked_waiting_container_marks_pod_failed(
        self, manager: KubernetesServiceManager
    ) -> None:
        manager._process_pod_snapshots(
            aggregate_pods_by_index(
                [
                    _fake_pod(
                        "w-0",
                        "0",
                        "Pending",
                        waiting_reason="ImagePullBackOff",
                    )
                ]
            ),
            now_ns=1,
        )

        pod = manager.get_pod_info("0")
        assert pod is not None and pod.failed

    def test_same_pod_recovery_clears_failure(
        self, manager: KubernetesServiceManager
    ) -> None:
        manager._process_pod_snapshots(
            aggregate_pods_by_index(
                [
                    _fake_pod(
                        "w-0",
                        "0",
                        "Pending",
                        waiting_reason="CrashLoopBackOff",
                    )
                ]
            ),
            now_ns=1,
        )
        manager._process_pod_snapshots(
            aggregate_pods_by_index([_fake_pod("w-0", "0", "Running")]),
            now_ns=2,
        )

        assert not manager.get_pod_info("0").failed

    def test_restart_count_surfaces_in_summary(
        self, manager: KubernetesServiceManager
    ) -> None:
        snapshots = aggregate_pods_by_index(
            [
                _fake_pod(
                    "w-0", "0", "Running", restart_count=3, waiting_reason="OOMKilled"
                )
            ]
        )
        manager._process_pod_snapshots(snapshots, now_ns=1)
        summary = manager.get_pod_summary()["0"]
        assert "restarts=3" in summary
        assert "OOMKilled" in summary

    def test_failure_threshold_sets_abort_event(
        self, manager: KubernetesServiceManager
    ) -> None:
        manager._pods = {
            "0": PodInfo(pod_index="0", pod_name="w-0", failed=True),
            "1": PodInfo(pod_index="1", pod_name="w-1", failed=True),
        }
        manager._check_pod_failure_threshold()
        assert manager.pod_failure_abort_event.is_set()
        assert "2/2 worker pods failed" in manager.pod_failure_abort_reason

    def test_failure_threshold_escalates_recoverable_service_death(
        self, manager: KubernetesServiceManager
    ) -> None:
        ServiceRegistry.register(
            "worker_a",
            ServiceType.WORKER,
            first_seen_ns=1,
            state=LifecycleState.RUNNING,
            pod_index="0",
        )
        manager._pods = {
            "0": PodInfo(pod_index="0", pod_name="w-0", failed=True),
            "1": PodInfo(pod_index="1", pod_name="w-1"),
        }
        manager._fail_pod_services("0", "w-0", PodPhase.FAILED)

        assert ServiceRegistry.get_dead_services() == {"worker_a": ServiceType.WORKER}
        manager._check_pod_failure_threshold()

        assert manager.pod_failure_abort_event.is_set()
        with pytest.raises(ServiceProcessDiedError, match="worker_a"):
            ServiceRegistry._raise_on_failure()

    def test_failure_threshold_not_triggered_below_percentage(
        self, manager: KubernetesServiceManager
    ) -> None:
        """One failure out of four expected pods is 25%, under the 50% default."""
        manager.required_services[ServiceType.WORKER_GROUP_MANAGER] = 4
        manager._pods = {
            "0": PodInfo(pod_index="0", pod_name="w-0", failed=True),
            "1": PodInfo(pod_index="1", pod_name="w-1"),
        }
        manager._check_pod_failure_threshold()
        assert not manager.pod_failure_abort_event.is_set()

    @pytest.mark.asyncio
    async def test_check_pods_healthy_raises_for_failed_pod(
        self, manager: KubernetesServiceManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_NAMESPACE", "aiperf")
        monkeypatch.setenv("AIPERF_JOB_ID", "job-1")
        ServiceRegistry.register(
            "worker_a",
            ServiceType.WORKER,
            first_seen_ns=1,
            state=LifecycleState.RUNNING,
            pod_index="0",
        )
        pods = [_fake_pod("w-0", "0", "Failed")]

        @asynccontextmanager
        async def fake_client():
            yield MagicMock()

        with (
            patch(
                "aiperf.controller.kubernetes_service_manager.k8s_client",
                fake_client,
            ),
            patch(
                "aiperf.controller.kubernetes_service_manager.get_pods",
                AsyncMock(return_value=pods),
            ),
            pytest.raises(ServiceProcessDiedError),
        ):
            await manager.check_pods_healthy()

    @pytest.mark.asyncio
    async def test_check_pods_healthy_skips_without_env(
        self, manager: KubernetesServiceManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AIPERF_NAMESPACE", raising=False)
        monkeypatch.delenv("AIPERF_JOB_ID", raising=False)
        await manager.check_pods_healthy()

    @pytest.mark.asyncio
    async def test_monitor_loop_swallows_kubernetes_errors(
        self, manager: KubernetesServiceManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_NAMESPACE", "aiperf")
        monkeypatch.setenv("AIPERF_JOB_ID", "job-1")

        @asynccontextmanager
        async def failing_client():
            raise RuntimeError("api down")
            yield MagicMock()

        with patch(
            "aiperf.controller.kubernetes_service_manager.k8s_client",
            failing_client,
        ):
            await manager._monitor_worker_pods()
        assert manager._pods == {}

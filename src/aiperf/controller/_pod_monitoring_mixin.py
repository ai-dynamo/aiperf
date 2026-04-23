# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pod-monitoring state helpers for ``KubernetesServiceManager``.

Split out of ``kubernetes_service_manager.py`` to keep that module under the
file-size budget. The mixin supplies per-pod tracking, threshold detection,
and status-query methods. Methods that call into ``kubernetes_asyncio`` live
on ``KubernetesServiceManager`` itself so test patches against
``aiperf.controller.kubernetes_service_manager.<name>`` keep working.

Consumers must inherit from this BEFORE ``MultiProcessServiceManager`` so
``super().__init__`` chains correctly; ``KubernetesServiceManager`` does that.
"""

from __future__ import annotations

import asyncio

from kubernetes_asyncio.client import ApiClient

from aiperf.common.environment import Environment
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.kubernetes_pod_helpers import (
    PodInfo,
    PodSnapshot,
    extract_container_issues,
    format_pod_failure_reason,
)
from aiperf.kubernetes.enums import PodPhase
from aiperf.plugin.enums import ServiceType


class PodMonitoringMixin:
    """Mixin: worker-pod health tracking for Kubernetes deployments.

    Provides public pod-query methods, per-pod tracking updates, and the
    failure-threshold gate. The ``KubernetesServiceManager`` class holds the
    API-client accessor and the background monitoring loop so that tests can
    patch ``get_pods`` / ``config`` / ``ApiClient`` on the main module.
    """

    _pods: dict[str, PodInfo]
    _restart_warned: set[str]
    _kube_api: ApiClient | None
    _kube_client_lock: asyncio.Lock
    _pod_monitoring_active: bool
    _heartbeat_monitoring_active: bool
    _shutdown_complete: bool
    stop_requested: bool
    pod_failure_abort_event: asyncio.Event
    pod_failure_abort_reason: str

    # -- Pod state queries (for SystemController) --

    def get_pod_info(self, pod_index: str) -> PodInfo | None:
        """Get tracked state for a specific pod by index."""
        return self._pods.get(pod_index)

    def get_all_pod_info(self) -> dict[str, PodInfo]:
        """Get tracked state for all known worker pods."""
        return dict(self._pods)

    def get_failed_pods(self) -> list[PodInfo]:
        """Get pods that have been marked as failed."""
        return [p for p in self._pods.values() if p.failed]

    def get_pod_summary(self) -> dict[str, str]:
        """Get a summary dict of pod states for logging/diagnostics.

        Returns a dict mapping pod_index to a human-readable status string.
        """
        summary: dict[str, str] = {}
        for idx, pod in self._pods.items():
            parts = [pod.phase]
            if pod.restart_count > 0:
                parts.append(f"restarts={pod.restart_count}")
            if pod.container_issues:
                parts.append(f"issues=[{', '.join(pod.container_issues)}]")
            summary[idx] = " ".join(parts)
        return summary

    # -- Internal pod-state bookkeeping --

    def _raise_for_any_failed_pod(self, pods_by_index: dict[str, PodSnapshot]) -> None:
        """Fail services for any terminal pods and raise via the registry."""
        for pod_index, (
            pod_name,
            phase,
            container_statuses,
            status,
        ) in pods_by_index.items():
            if phase not in (PodPhase.FAILED, PodPhase.UNKNOWN):
                continue
            reason = format_pod_failure_reason(
                pod_name, phase, container_statuses, status
            )
            self.error(  # type: ignore[attr-defined]
                f"Pod health check failed before PROFILE_START: {reason}"
            )
            self._fail_pod_services(pod_index)
            ServiceRegistry._raise_on_failure()

    def _fail_pod_services(
        self,
        pod_index: str,
        pod_name: str | None = None,
        phase: PodPhase | None = None,
    ) -> None:
        """Mark all services on a pod as failed in the ServiceRegistry."""
        affected = ServiceRegistry.get_services_by_pod(pod_index)
        if not affected:
            self.warning(  # type: ignore[attr-defined]
                f"No services found for pod_index={pod_index} via registry — "
                f"services may not have registered with pod_index"
            )
            return
        for info in affected:
            context = ""
            if pod_name and phase:
                context = f" (pod '{pod_name}' is {phase})"
            self.warning(  # type: ignore[attr-defined]
                f"Marking service '{info.service_id}' as failed{context}"
            )
            ServiceRegistry.fail_service(info.service_id, info.service_type)

    def _check_pod_failure_threshold(self) -> None:
        """Check if failed pods exceed the abort threshold.

        When the percentage of failed worker pods reaches the configured
        threshold (AIPERF_SERVICE_POD_FAILURE_ABORT_THRESHOLD_PERCENT),
        signals pod_failure_abort_event so the system controller can
        cancel the benchmark.
        """
        if self.pod_failure_abort_event.is_set():
            return

        threshold = Environment.SERVICE.POD_FAILURE_ABORT_THRESHOLD_PERCENT
        if threshold == 0:
            return

        expected_total_pods = self.required_services.get(  # type: ignore[attr-defined]
            ServiceType.WORKER_GROUP_MANAGER, 0
        )
        total_pods = expected_total_pods or len(self._pods)
        if total_pods == 0:
            return

        failed_pods = sum(1 for p in self._pods.values() if p.failed)
        if failed_pods == 0:
            return

        failure_percent = (failed_pods / total_pods) * 100
        if failure_percent >= threshold:
            self.pod_failure_abort_reason = (
                f"{failed_pods}/{total_pods} worker pods failed "
                f"({failure_percent:.0f}% >= {threshold}% threshold)"
            )
            self.error(  # type: ignore[attr-defined]
                f"Pod failure threshold exceeded: {self.pod_failure_abort_reason}"
            )
            self.pod_failure_abort_event.set()

    def _update_pod_tracking(
        self,
        pod_index: str,
        pod_name: str,
        *,
        phase: PodPhase,
        container_statuses: list[dict],
        now_ns: int,
    ) -> PodInfo:
        """Upsert a PodInfo entry and log restart/issue warnings."""
        restart_count = sum(cs.get("restartCount", 0) for cs in container_statuses)
        issues = extract_container_issues(container_statuses)

        pod_info = self._pods.get(pod_index)
        if pod_info is None:
            pod_info = PodInfo(pod_index=pod_index, pod_name=pod_name)
            self._pods[pod_index] = pod_info

        pod_info.pod_name = pod_name
        pod_info.phase = phase
        pod_info.restart_count = restart_count
        pod_info.container_issues = issues
        pod_info.last_checked_ns = now_ns

        if restart_count >= 3 and pod_index not in self._restart_warned:
            self._restart_warned.add(pod_index)
            issue_detail = f" ({', '.join(issues)})" if issues else ""
            self.warning(  # type: ignore[attr-defined]
                f"Pod '{pod_name}' (index={pod_index}) has "
                f"{restart_count} container restarts{issue_detail}"
            )

        if issues and phase == PodPhase.RUNNING:
            self.debug(  # type: ignore[attr-defined]
                f"Pod '{pod_name}' is Running but has container issues: "
                f"{', '.join(issues)}"
            )

        return pod_info

    def _handle_terminal_pod(
        self,
        pod_info: PodInfo,
        pod_index: str,
        pod_name: str,
        *,
        phase: PodPhase,
        container_statuses: list[dict],
        status: dict,
    ) -> None:
        """Mark a terminal pod as failed (once) and fail its services."""
        if not pod_info.is_terminal or pod_info.failed:
            return
        pod_info.failed = True
        reason = format_pod_failure_reason(pod_name, phase, container_statuses, status)
        self.warning(reason)  # type: ignore[attr-defined]
        self._fail_pod_services(pod_index, pod_name, phase)

    def _process_pod_snapshots(
        self, pods_by_index: dict[str, PodSnapshot], now_ns: int
    ) -> None:
        """Update tracking state for each aggregated pod snapshot."""
        for pod_index, (
            pod_name,
            phase,
            container_statuses,
            status,
        ) in pods_by_index.items():
            pod_info = self._update_pod_tracking(
                pod_index,
                pod_name,
                phase=phase,
                container_statuses=container_statuses,
                now_ns=now_ns,
            )
            self._handle_terminal_pod(
                pod_info,
                pod_index,
                pod_name,
                phase=phase,
                container_statuses=container_statuses,
                status=status,
            )

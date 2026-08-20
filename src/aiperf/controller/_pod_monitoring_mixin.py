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
from collections.abc import Callable

from kubernetes_asyncio.client import ApiClient

from aiperf.common.environment import Environment
from aiperf.common.exceptions import ServiceProcessDiedError
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.kubernetes_pod_helpers import (
    PodInfo,
    PodSnapshot,
    dead_sibling_containers,
    extract_container_issues,
    format_pod_failure_reason,
)
from aiperf.kubernetes.enums import PodPhase
from aiperf.plugin.enums import ServiceType

POD_UNHEALTHY_STRIKES = 2
"""Consecutive unhealthy polls required before a pod's services are reaped.

Mirrors the heartbeat watchdog's two-strike rule in ``BaseServiceManager``.
Reaping is permanent -- ``ResultJoinCoordinator.complete_domain`` returns early
once a domain has been popped -- so a single sampling artifact must not be able
to drop a live producer out of the run.
"""

BLOCKED_CONTAINER_WAITING_REASONS = frozenset(
    {
        "CrashLoopBackOff",
        "CreateContainerConfigError",
        "CreateContainerError",
        "ErrImageNeverPull",
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
        "RunContainerError",
    }
)


class PodMonitoringMixin:
    """Mixin: worker-pod health tracking for Kubernetes deployments.

    Provides public pod-query methods, per-pod tracking updates, and the
    failure-threshold gate. The ``KubernetesServiceManager`` class holds the
    API-client accessor and the background monitoring loop so that tests can
    patch ``get_pods`` / ``config`` / ``ApiClient`` on the main module.
    """

    _pods: dict[str, PodInfo]
    _restart_warned: set[str]
    _pod_health_strikes: dict[str, int]
    _kube_api: ApiClient | None
    _kube_client_lock: asyncio.Lock
    _pod_monitoring_active: bool
    _shutdown_complete: bool
    stop_requested: bool
    pod_failure_abort_event: asyncio.Event
    pod_failure_abort_reason: str

    # Logging surface supplied by ``AIPerfLoggerMixin`` on the concrete
    # service manager. Declared so this mixin's log calls type-check on their
    # own, without an ``# type: ignore`` on every call site.
    debug: Callable[..., None]
    info: Callable[..., None]
    warning: Callable[..., None]
    error: Callable[..., None]

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

    def _raise_for_pod_failure_threshold(self) -> None:
        """Raise before profiling only when pod losses reach the threshold."""
        self._check_pod_failure_threshold()
        if not self.pod_failure_abort_event.is_set():
            unhealthy = self.get_failed_pods()
            if unhealthy:
                self.warning(
                    "Pod health check before PROFILE_START: "
                    f"{len(unhealthy)} unhealthy pod(s) "
                    f"({', '.join(p.pod_name for p in unhealthy)}) are below "
                    "the abort threshold; proceeding with surviving pods"
                )
            return

        self.error(
            "Pod health check failed before PROFILE_START: "
            f"{self.pod_failure_abort_reason}"
        )
        ServiceRegistry._raise_on_failure()
        failed = self.get_failed_pods()
        raise ServiceProcessDiedError(
            service_id=failed[0].pod_name if failed else "unknown-worker-pod",
            service_type=ServiceType.WORKER_GROUP_MANAGER,
        )

    def _fail_pod_services(
        self,
        pod_index: str,
        pod_name: str | None = None,
        phase: PodPhase | None = None,
        *,
        fatal: bool = False,
    ) -> None:
        """Mark services on a pod dead, recoverably by default."""
        affected = ServiceRegistry.get_services_by_pod(pod_index)
        if not affected:
            self.warning(
                f"No services found for pod_index={pod_index} via registry — "
                f"services may not have registered with pod_index"
            )
            return
        for info in affected:
            context = ""
            if pod_name and phase:
                context = f" (pod '{pod_name}' is {phase})"
            self.warning(f"Marking service '{info.service_id}' as failed{context}")
            ServiceRegistry.fail_service(
                info.service_id, info.service_type, fatal=fatal
            )
            # Also drop it from the result-join barrier, otherwise a pod that
            # died without emitting an error message keeps the barrier pending
            # forever. This method is synchronous, so it only records; the
            # heartbeat watchdog drains the queue on its next tick.
            record_reaped = getattr(self, "record_reaped_service", None)
            if record_reaped is not None:
                record_reaped(
                    info.service_id,
                    f"pod '{pod_name}' is {phase}"
                    if pod_name and phase
                    else "pod failed",
                )

    def _check_pod_failure_threshold(self) -> None:
        """Check if failed pods exceed the abort threshold.

        When the percentage of failed worker pods reaches the configured
        threshold (AIPERF_POD_FAILURE_ABORT_THRESHOLD_PERCENT),
        signals pod_failure_abort_event so the system controller can
        cancel the benchmark.
        """
        if self.pod_failure_abort_event.is_set():
            return

        threshold = Environment.POD.FAILURE_ABORT_THRESHOLD_PERCENT
        if threshold == 0:
            return

        expected_total_pods = self.required_services.get(
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
            self.error(
                f"Pod failure threshold exceeded: {self.pod_failure_abort_reason}"
            )
            ServiceRegistry.escalate_dead_services()
            self.pod_failure_abort_event.set()

    def _check_dead_sibling_containers(self, pods: list) -> None:
        """Abort when a service container in our own pod has died.

        The controller pod is in the same poll as the worker pods but is
        dropped by ``extract_pod_snapshot``, which keeps only the ``workers``
        replicated job. Without this check a sibling that dies before
        registering leaves the configure wait blocked for the full
        ``PROFILE_CONFIGURE_TIMEOUT`` and then reports a generic timeout that
        names nothing.

        Routed through ``pod_failure_abort_event`` -- the same path a
        worker-pod threshold breach takes -- so the controller cancels
        cleanly and still exports partial results.
        """
        if self.pod_failure_abort_event.is_set():
            return
        dead = dead_sibling_containers(pods)
        if not dead:
            return
        detail = ", ".join(
            f"{name} (reason={reason}, exitCode={code})" for name, reason, code in dead
        )
        self.pod_failure_abort_reason = f"controller-pod container died: {detail}"
        self.error(f"Sibling container failure: {self.pod_failure_abort_reason}")
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
        elif pod_info.pod_name != pod_name:
            self.info(
                f"Pod index {pod_index} replaced: '{pod_info.pod_name}' -> '{pod_name}'"
            )
            pod_info.failed = False
            self._restart_warned.discard(pod_index)
            self._unhealthy_pod_strikes().pop(pod_index, None)

        pod_info.pod_name = pod_name
        pod_info.phase = phase
        pod_info.restart_count = restart_count
        pod_info.container_issues = issues
        pod_info.last_checked_ns = now_ns

        if restart_count >= 3 and pod_index not in self._restart_warned:
            self._restart_warned.add(pod_index)
            issue_detail = f" ({', '.join(issues)})" if issues else ""
            self.warning(
                f"Pod '{pod_name}' (index={pod_index}) has "
                f"{restart_count} container restarts{issue_detail}"
            )

        if issues and phase == PodPhase.RUNNING:
            self.debug(
                f"Pod '{pod_name}' is Running but has container issues: "
                f"{', '.join(issues)}"
            )

        return pod_info

    @staticmethod
    def _blocked_container_reasons(container_statuses: list[dict]) -> list[str]:
        """Return active container waiting reasons that require intervention."""
        blocked: list[str] = []
        for container_status in container_statuses:
            waiting = (container_status.get("state") or {}).get("waiting") or {}
            reason = waiting.get("reason", "")
            if reason in BLOCKED_CONTAINER_WAITING_REASONS:
                blocked.append(f"{container_status.get('name', 'unknown')}: {reason}")
        return blocked

    def _unhealthy_pod_strikes(self) -> dict[str, int]:
        """Per-pod consecutive-unhealthy-poll counters, created on first use."""
        strikes: dict[str, int] | None = getattr(self, "_pod_health_strikes", None)
        if strikes is None:
            strikes = {}
            self._pod_health_strikes = strikes
        return strikes

    def _pod_failure_is_confirmed(
        self, pod_index: str, pod_name: str, *, phase: PodPhase, blocked: bool
    ) -> bool:
        """Require a second consecutive unhealthy poll before reaping a pod.

        Only ``Unknown`` needs the confirmation. It is the phase the apiserver
        reports when it simply cannot reach the node's kubelet, so a momentary
        control-plane blip publishes it for a healthy, still-benchmarking pod.
        ``Failed`` is terminal by definition and blocked container reasons
        (ImagePullBackOff, CrashLoopBackOff, ...) are latched kubelet states,
        not sampling noise, so both act on the first observation and keep the
        original detection latency.
        """
        strikes = self._unhealthy_pod_strikes()
        if blocked or phase != PodPhase.UNKNOWN:
            strikes.pop(pod_index, None)
            return True

        count = strikes.get(pod_index, 0) + 1
        if count >= POD_UNHEALTHY_STRIKES:
            strikes.pop(pod_index, None)
            return True

        strikes[pod_index] = count
        self.warning(
            f"Pod '{pod_name}' (index={pod_index}) reported {phase}; awaiting "
            f"a second consecutive poll before treating it as dead"
        )
        return False

    def _evaluate_pod_health(
        self,
        pod_info: PodInfo,
        pod_index: str,
        pod_name: str,
        *,
        phase: PodPhase,
        container_statuses: list[dict],
        status: dict,
    ) -> None:
        """Set or clear failure from the current pod generation's health."""
        blocked = self._blocked_container_reasons(container_statuses)
        if not pod_info.is_terminal and not blocked:
            self._unhealthy_pod_strikes().pop(pod_index, None)
            if pod_info.failed:
                self.info(f"Pod '{pod_name}' (index={pod_index}) recovered to {phase}")
                pod_info.failed = False
            return
        if pod_info.failed:
            return
        if not self._pod_failure_is_confirmed(
            pod_index, pod_name, phase=phase, blocked=bool(blocked)
        ):
            return
        pod_info.failed = True
        reason = format_pod_failure_reason(pod_name, phase, container_statuses, status)
        if blocked:
            reason += f" | blocked containers: {', '.join(blocked)}"
        self.warning(reason)
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
            self._evaluate_pod_health(
                pod_info,
                pod_index,
                pod_name,
                phase=phase,
                container_statuses=container_statuses,
                status=status,
            )

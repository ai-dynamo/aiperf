# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes service manager for AIPerf.

This module provides a hybrid ServiceManager implementation that:
- Treats control-plane services as sibling Kubernetes containers
- Treats workers and record processors as external Kubernetes pods
- Monitors pod health with container-level detail (OOMKilled, CrashLoopBackOff, etc.)

This enables Kubernetes mode to run one container per control-plane service
while workers remain separate worker pods managed by JobSet.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.exceptions import ServiceProcessDiedError
from aiperf.common.hooks import background_task
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT
from aiperf.controller.multiprocess_service_manager import MultiProcessServiceManager
from aiperf.kubernetes.constants import JobSetLabels
from aiperf.kubernetes.enums import PodPhase
from aiperf.plugin.enums import ServiceType

if TYPE_CHECKING:
    from aiperf.kubernetes.client import AIPerfKubeClient

# Services that are externally managed in Kubernetes mode (not spawned by the
# service manager as local subprocesses).
# In Kubernetes mode:
# - Control-plane services run in sibling containers in the controller pod
# - WORKER and RECORD_PROCESSOR run in sibling worker-pod containers
# - WORKER_POD_MANAGER is the shared pod-infrastructure container
EXTERNAL_K8S_SERVICES = frozenset(
    {
        ServiceType.API,
        ServiceType.DATASET_MANAGER,
        ServiceType.GPU_TELEMETRY_MANAGER,
        ServiceType.RECORDS_MANAGER,
        ServiceType.SERVER_METRICS_MANAGER,
        ServiceType.TIMING_MANAGER,
        ServiceType.WORKER,
        ServiceType.WORKER_MANAGER,
        ServiceType.RECORD_PROCESSOR,
        ServiceType.WORKER_POD_MANAGER,
    }
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


@dataclass
class PodInfo:
    """Tracked state for a single Kubernetes worker pod."""

    pod_index: str
    """JobSet pod index (from JobSetLabels.POD_INDEX label)."""

    pod_name: str
    """Kubernetes pod name."""

    phase: PodPhase = PodPhase.PENDING
    """Current pod phase."""

    restart_count: int = 0
    """Total container restart count across all containers."""

    container_issues: list[str] = field(default_factory=list)
    """Active container-level issues (e.g. 'OOMKilled', 'CrashLoopBackOff')."""

    last_checked_ns: int = 0
    """Timestamp of last health check (nanoseconds)."""

    failed: bool = False
    """Whether this pod has been marked as failed in the registry."""

    @property
    def is_terminal(self) -> bool:
        """Whether the pod is in a terminal failure state."""
        return self.phase in (PodPhase.FAILED, PodPhase.UNKNOWN)


class KubernetesServiceManager(MultiProcessServiceManager):
    """Service manager for Kubernetes distributed deployments.

    Treats control-plane services as sibling containers in the controller pod,
    while workers, record processors, and worker-pod-manager services are
    external Kubernetes containers/pods.

    Maintains a pod registry that tracks per-pod health, container states, and
    restart counts. The SystemController can query pod state for diagnostics
    and error reporting.

    Key differences from MultiProcessServiceManager:
    - run_service: No-op for externally managed K8s containers/pods
    - stop_service: No-op for externally managed K8s containers/pods
    - wait_for_*: Waits for external services to register via message bus
    - Pod health monitoring with container-level failure detection
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        run: BenchmarkRun,
        **kwargs,
    ):
        super().__init__(required_services, run, **kwargs)
        self._kube_client = None
        self._pods: dict[str, PodInfo] = {}
        self._restart_warned: set[str] = set()

    def _is_external_service(self, service_type: ServiceTypeT) -> bool:
        """Check if a service type is an external Kubernetes pod."""
        return service_type in EXTERNAL_K8S_SERVICES

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Register expectations for an externally managed Kubernetes service.

        Kubernetes manifests launch control-plane services as sibling containers
        and workers/record processors via worker pods, so the service manager
        only records how many instances must register.
        """
        if self._is_external_service(service_type):
            self.debug(
                f"Expecting {num_replicas} external {service_type} instance(s) to register"
            )
            ServiceRegistry.expect_services({service_type: num_replicas})
            return

        await super().run_service(service_type, num_replicas)

    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]:
        """Stop a service, either local subprocess or external Kubernetes runtime.

        Externally managed Kubernetes services receive shutdown over the control
        channel and exit on their own, so there is no subprocess to stop here.
        """
        if self._is_external_service(service_type):
            self.debug(
                f"stop_service called for {service_type} "
                "(no-op - externally managed in Kubernetes)"
            )
            return []

        return await super().stop_service(service_type, service_id)

    async def shutdown_all_services(self) -> list[BaseException | None]:
        """Stop any locally managed subprocesses except API.

        Normal Kubernetes-mode deployments launch sibling controller containers
        directly from the pod spec, so this usually has nothing to do. The
        fallback subprocess cleanup remains for defensive compatibility.
        """
        self._shutdown_complete = True
        self.debug(
            "Stopping any locally managed service processes "
            "(excluding API for results serving)"
        )

        to_stop = [
            info
            for info in self._subprocess_manager.subprocesses
            if info.service_type != ServiceType.API
        ]

        results = await asyncio.gather(
            *[self._subprocess_manager.stop_process(info) for info in to_stop],
            return_exceptions=True,
        )

        for info in to_stop:
            ServiceRegistry.unregister(info.service_id)
            self._subprocess_manager.remove(info)

        self._kube_client = None

        return results

    async def wait_for_api_subprocess(self) -> None:
        """Block until the API subprocess terminates, if one exists.

        When the API runs as its own container there is no local subprocess and
        this returns immediately. The fallback wait remains for compatibility
        with older single-container layouts.
        """
        api_infos = self._subprocess_manager.get_by_type(ServiceType.API)
        if not api_infos or not api_infos[0].process:
            self.debug("No API subprocess found to wait for")
            return

        api_process = api_infos[0].process
        self.info(
            f"Waiting for API subprocess (pid: {api_process.pid}) to serve results..."
        )

        while api_process.is_alive():
            await asyncio.sleep(1.0)

        self.info("API subprocess has terminated")

    async def wait_for_all_services_registration(
        self,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None:
        """Wait for all required services to register.

        This includes both:
        - Control-plane services spawned as subprocesses
        - External workers/record processors connecting via TCP

        Raises:
            ServiceProcessDiedError: If a subprocess dies before registering.
            ServiceRegistrationTimeoutError: If services don't register within timeout.
        """
        self.debug(
            "Waiting for all required services to register "
            "(subprocesses + external Kubernetes pods)..."
        )
        await ServiceRegistry.wait_for_all(timeout_seconds)

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

    async def check_pods_healthy(self) -> None:
        """Verify all tracked pods are in a healthy state before profiling.

        Performs a fresh pod status check and raises ServiceProcessDiedError
        if any worker pods are in a terminal failure state. Called by the
        SystemController as a gate before sending PROFILE_START.
        """
        namespace = os.environ.get("AIPERF_NAMESPACE")
        job_id = os.environ.get("AIPERF_JOB_ID")
        if not namespace or not job_id:
            self.warning(
                "Pod health check skipped: AIPERF_NAMESPACE and/or AIPERF_JOB_ID "
                "not set — cannot query Kubernetes API for pod statuses"
            )
            return

        try:
            client = await self._get_kube_client()
            pods = await client.get_pods(namespace, client.job_selector(job_id))

            for pod in pods:
                raw = pod.raw
                metadata = raw.get("metadata", {})
                labels = metadata.get("labels", {})
                replicated_job = labels.get(JobSetLabels.REPLICATED_JOB_NAME)
                if replicated_job and replicated_job != "workers":
                    continue
                pod_index = labels.get(JobSetLabels.POD_INDEX)
                if pod_index is None:
                    continue

                phase = PodPhase(raw.get("status", {}).get("phase", PodPhase.UNKNOWN))
                if phase in (PodPhase.FAILED, PodPhase.UNKNOWN):
                    pod_name = metadata.get("name", "unknown")
                    status = raw.get("status", {})
                    container_statuses = status.get("containerStatuses", [])
                    reason = _format_pod_failure_reason(
                        pod_name, phase, container_statuses, status
                    )
                    self.error(
                        f"Pod health check failed before PROFILE_START: {reason}"
                    )
                    self._fail_pod_services(pod_index)
                    ServiceRegistry._raise_on_failure()
        except ServiceProcessDiedError:
            raise
        except Exception as e:
            self.warning(f"Pod health check before PROFILE_START failed: {e!r}")

    # -- Kubernetes pod health monitoring --

    def _fail_pod_services(
        self,
        pod_index: str,
        pod_name: str | None = None,
        phase: PodPhase | None = None,
    ) -> None:
        """Mark all services on a pod as failed in the ServiceRegistry."""
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

        expected_total_pods = self.required_services.get(
            ServiceType.WORKER_POD_MANAGER, 0
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
            self.pod_failure_abort_event.set()

    async def _get_kube_client(self) -> AIPerfKubeClient:
        """Get or create a cached Kubernetes API client."""
        if self._kube_client is None:
            from aiperf.kubernetes.client import AIPerfKubeClient

            self._kube_client = await AIPerfKubeClient.create()
        return self._kube_client

    @background_task(
        interval=lambda self: Environment.SERVICE.PROCESS_MONITOR_INTERVAL,
        immediate=False,
    )
    async def _monitor_worker_pods(self) -> None:
        """Query the Kubernetes API for worker pod statuses.

        Detects pods that have entered a terminal failure state (Failed, Unknown)
        and marks them as failed in the ServiceRegistry so the system can react.
        Also tracks container-level issues (OOMKilled, CrashLoopBackOff,
        ImagePullBackOff) and restart counts for diagnostics.

        Runs when pod monitoring is active (enabled during registration/configuration)
        or when heartbeat monitoring is active (during profiling). Pod phase checks
        are safe during startup — unlike heartbeats, a pod in Failed/Unknown is
        always an error.
        """
        if self._shutdown_complete or self.stop_requested:
            return
        if not self._pod_monitoring_active and not self._heartbeat_monitoring_active:
            return

        namespace = os.environ.get("AIPERF_NAMESPACE")
        job_id = os.environ.get("AIPERF_JOB_ID")
        if not namespace or not job_id:
            self.warning(
                "Pod monitoring skipped: AIPERF_NAMESPACE and/or AIPERF_JOB_ID "
                "not set — cannot query Kubernetes API for pod statuses"
            )
            return

        try:
            client = await self._get_kube_client()
            pods = await client.get_pods(namespace, client.job_selector(job_id))
            now_ns = time.time_ns()

            pods_by_index: dict[str, tuple[str, PodPhase, list[dict], dict]] = {}
            for pod in pods:
                raw = pod.raw
                metadata = raw.get("metadata", {})
                pod_name = metadata.get("name", "unknown")
                labels = metadata.get("labels", {})
                replicated_job = labels.get(JobSetLabels.REPLICATED_JOB_NAME)
                if replicated_job and replicated_job != "workers":
                    continue
                pod_index = labels.get(JobSetLabels.POD_INDEX)
                if pod_index is None:
                    continue

                status = raw.get("status", {})
                phase = PodPhase(status.get("phase", PodPhase.UNKNOWN))
                existing = pods_by_index.get(pod_index)
                if existing is None or (
                    existing[1] in (PodPhase.FAILED, PodPhase.UNKNOWN)
                    and phase not in (PodPhase.FAILED, PodPhase.UNKNOWN)
                ):
                    pods_by_index[pod_index] = (
                        pod_name,
                        phase,
                        status.get("containerStatuses", []),
                        status,
                    )

            for pod_index, (
                pod_name,
                phase,
                container_statuses,
                status,
            ) in pods_by_index.items():
                restart_count = sum(
                    cs.get("restartCount", 0) for cs in container_statuses
                )
                issues = _extract_container_issues(container_statuses)

                # Update pod tracking
                pod_info = self._pods.get(pod_index)
                if pod_info is None:
                    pod_info = PodInfo(pod_index=pod_index, pod_name=pod_name)
                    self._pods[pod_index] = pod_info

                pod_info.pod_name = pod_name
                pod_info.phase = phase
                pod_info.restart_count = restart_count
                pod_info.container_issues = issues
                pod_info.last_checked_ns = now_ns

                # Warn on high restart count (once per pod)
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

                if not pod_info.is_terminal:
                    continue

                if pod_info.failed:
                    continue

                pod_info.failed = True
                reason = _format_pod_failure_reason(
                    pod_name, phase, container_statuses, status
                )
                self.warning(reason)

                self._fail_pod_services(pod_index, pod_name, phase)

            self._check_pod_failure_threshold()

        except Exception as e:
            self.warning(f"Failed to query Kubernetes pod statuses: {e!r}")


def _extract_container_issues(container_statuses: list[dict]) -> list[str]:
    """Extract actionable issue labels from container statuses.

    Inspects waiting and terminated container states for known failure
    patterns like OOMKilled, CrashLoopBackOff, and ImagePullBackOff.
    """
    issues: list[str] = []
    seen: set[str] = set()
    for cs in container_statuses:
        state = cs.get("state", {})

        waiting = state.get("waiting", {})
        if waiting:
            reason = waiting.get("reason", "")
            if reason and reason not in seen:
                seen.add(reason)
                issues.append(reason)

        terminated = state.get("terminated", {})
        if terminated:
            reason = terminated.get("reason", "")
            if reason and reason not in seen:
                seen.add(reason)
                issues.append(reason)

        last_state = cs.get("lastState", {})
        last_terminated = last_state.get("terminated", {})
        if last_terminated:
            reason = last_terminated.get("reason", "")
            if reason and reason not in seen:
                seen.add(reason)
                issues.append(reason)

    return issues


def _format_pod_failure_reason(
    pod_name: str,
    phase: PodPhase,
    container_statuses: list[dict],
    status: dict,
) -> str:
    """Build a detailed failure reason string for a failed pod.

    Includes the pod phase, container exit codes, termination reasons,
    and any waiting state reasons to help operators diagnose the failure.
    """
    parts = [f"K8s pod '{pod_name}' is {phase}"]

    for cs in container_statuses:
        container_name = cs.get("name", "unknown")
        state = cs.get("state", {})

        terminated = state.get("terminated", {})
        if terminated:
            reason = terminated.get("reason", "")
            exit_code = terminated.get("exitCode")
            detail = f"container '{container_name}': terminated"
            if reason:
                detail += f" ({reason})"
            if exit_code is not None:
                detail += f" exit_code={exit_code}"
            message = terminated.get("message", "")
            if message:
                detail += f" - {message[:200]}"
            parts.append(detail)

        waiting = state.get("waiting", {})
        if waiting:
            reason = waiting.get("reason", "")
            if reason:
                detail = f"container '{container_name}': waiting ({reason})"
                message = waiting.get("message", "")
                if message:
                    detail += f" - {message[:200]}"
                parts.append(detail)

    # Include pod-level conditions with useful messages
    conditions = status.get("conditions", [])
    for cond in conditions:
        if cond.get("status") == "False" and cond.get("message"):
            parts.append(f"condition {cond['type']}: {cond['message'][:200]}")

    return " | ".join(parts)

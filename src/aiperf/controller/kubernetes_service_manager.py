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

from aiperf.common.environment import Environment
from aiperf.common.exceptions import ServiceProcessDiedError
from aiperf.common.hooks import background_task
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT
from aiperf.controller._pod_monitoring_mixin import PodMonitoringMixin
from aiperf.controller.kubernetes_pod_helpers import (
    PodInfo,
    aggregate_pods_by_index,
)
from aiperf.controller.multiprocess_service_manager import MultiProcessServiceManager
from aiperf.kubernetes.client import get_pods, job_selector, k8s_client
from aiperf.plugin.enums import ServiceType

# Re-export PodInfo so importers of this module get the tracked-pod type too.
__all__ = ["EXTERNAL_K8S_SERVICES", "KubernetesServiceManager", "PodInfo"]

# Services that are externally managed in Kubernetes mode (not spawned by the
# service manager as local subprocesses).
# In Kubernetes mode:
# - Control-plane services run in sibling containers in the controller pod
# - WORKER and RECORD_PROCESSOR run in sibling worker-pod containers
# - WORKER_GROUP_MANAGER is the shared pod-infrastructure container
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
        ServiceType.WORKER_GROUP_MANAGER,
    }
)


class KubernetesServiceManager(PodMonitoringMixin, MultiProcessServiceManager):
    """Service manager for Kubernetes distributed deployments.

    Treats control-plane services as sibling containers in the controller pod,
    while workers, record processors, and worker-group-manager services are
    external Kubernetes containers/pods.

    Maintains a pod registry that tracks per-pod health, container states, and
    restart counts. The SystemController can query pod state for diagnostics
    and error reporting.

    Key differences from MultiProcessServiceManager:
    - ``run_service`` / ``stop_service``: no-ops for externally managed pods
    - Pod health monitoring with container-level failure detection

    Example:
        manager = KubernetesServiceManager(
            required_services={ServiceType.WORKER: 8}, run=run
        )
        await manager.run_service(ServiceType.WORKER, num_replicas=8)
        await manager.check_pods_healthy()
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        **kwargs,
    ) -> None:
        super().__init__(required_services, **kwargs)
        self._pods: dict[str, PodInfo] = {}
        self._restart_warned: set[str] = set()
        self._shutdown_complete = False
        # Pod-phase checks are safe from the moment the manager exists: unlike
        # heartbeats, a pod in Failed/Unknown is always an error, never a
        # not-yet-started service.
        self._pod_monitoring_active = True
        self.pod_failure_abort_event = asyncio.Event()
        self.pod_failure_abort_reason = ""

    def _is_external_service(self, service_type: ServiceTypeT) -> bool:
        """Check if a service type is externally managed by Kubernetes."""
        return service_type in EXTERNAL_K8S_SERVICES

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Register expectations for an externally managed Kubernetes service.

        For service types in ``EXTERNAL_K8S_SERVICES`` this spawns nothing:
        Kubernetes manifests launch control-plane services as sibling
        containers and workers/record processors via worker pods. Only the
        expected instance count is recorded with ``ServiceRegistry``.

        Any other service type delegates to
        ``MultiProcessServiceManager.run_service``, which spawns a subprocess.
        """
        if self._is_external_service(service_type):
            self.debug(
                lambda: f"Expecting {num_replicas} external {service_type} "
                "instance(s) to register"
            )
            ServiceRegistry.expect_services({service_type: num_replicas})
            return

        await super().run_service(service_type, num_replicas)

    async def wait_for_all_services_registration(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None:
        """Wait for every expected Kubernetes service instance, not just each type."""
        if stop_event.is_set():
            return
        wait_task = asyncio.create_task(ServiceRegistry.wait_for_all(timeout_seconds))
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if wait_task in done:
            await wait_task

    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]:
        """Stop a local subprocess, or no-op for externally managed pods.

        Externally managed Kubernetes services receive shutdown over the
        control channel and exit on their own, so there is nothing to stop
        here and an empty result list is returned.
        """
        if self._is_external_service(service_type):
            self.debug(
                lambda: f"stop_service called for {service_type} "
                "(no-op - externally managed in Kubernetes)"
            )
            return []

        return await super().stop_service(service_type, service_id)

    async def shutdown_all_services(self) -> list[BaseException | None]:
        """Stop any locally managed subprocesses.

        Normal Kubernetes-mode deployments launch sibling controller containers
        directly from the pod spec, so the subprocess half usually has nothing
        to do; it remains for defensive compatibility.
        """
        self._shutdown_complete = True
        return await super().shutdown_all_services()

    async def check_pods_healthy(self) -> None:
        """Verify all tracked pods are healthy before profiling starts.

        Performs a fresh pod status check and raises if any worker pod is in a
        terminal failure state. Intended as a gate before PROFILE_START.

        Raises:
            ServiceProcessDiedError: If any worker pod is Failed or Unknown.
        """
        namespace = os.environ.get("AIPERF_NAMESPACE")
        job_id = os.environ.get("AIPERF_JOB_ID")
        if not namespace or not job_id:
            self.warning(
                "Pod health check skipped: AIPERF_NAMESPACE and/or AIPERF_JOB_ID "
                "not set - cannot query Kubernetes API for pod statuses"
            )
            return

        try:
            async with k8s_client() as api:
                pods = await get_pods(api, namespace, job_selector(job_id))
            self._process_pod_snapshots(aggregate_pods_by_index(pods), time.time_ns())
            self._raise_for_pod_failure_threshold()
        except ServiceProcessDiedError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - advisory check must not abort profiling on a transient API error
            self.warning(f"Pod health check before PROFILE_START failed: {e!r}")

    @background_task(
        interval=lambda self: Environment.POD.MONITOR_INTERVAL,
        immediate=False,
    )
    async def _monitor_worker_pods(self) -> None:
        """Query the Kubernetes API for worker pod statuses.

        Detects pods that have entered a terminal failure state (Failed,
        Unknown) and marks their services as failed in the ServiceRegistry so
        the system can react. Also tracks container-level issues (OOMKilled,
        CrashLoopBackOff, ImagePullBackOff) and restart counts for diagnostics.
        """
        if self._shutdown_complete or self.stop_requested:
            return
        if not self._pod_monitoring_active:
            return

        namespace = os.environ.get("AIPERF_NAMESPACE")
        job_id = os.environ.get("AIPERF_JOB_ID")
        if not namespace or not job_id:
            self.warning(
                "Pod monitoring skipped: AIPERF_NAMESPACE and/or AIPERF_JOB_ID "
                "not set - cannot query Kubernetes API for pod statuses"
            )
            return

        try:
            async with k8s_client() as api:
                pods = await get_pods(api, namespace, job_selector(job_id))
            self._process_pod_snapshots(aggregate_pods_by_index(pods), time.time_ns())
            self._check_pod_failure_threshold()
            self._check_dead_sibling_containers(pods)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - monitoring loop must survive transient k8s API errors
            self.warning(f"Failed to query Kubernetes pod statuses: {e!r}")

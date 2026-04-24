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
from typing import TYPE_CHECKING

import aiohttp
from kubernetes_asyncio import config
from kubernetes_asyncio.client import ApiClient

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
from aiperf.kubernetes.client import get_pods, job_selector
from aiperf.plugin.enums import ServiceType

# Re-export PodInfo for backwards compatibility with existing importers.
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

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class KubernetesServiceManager(PodMonitoringMixin, MultiProcessServiceManager):
    """Service manager for Kubernetes distributed deployments.

    Treats control-plane services as sibling containers in the controller pod,
    while workers, record processors, and worker-group-manager services are
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
        self._kube_api: ApiClient | None = None
        # Serializes lazy init of _kube_api. Without it, two concurrent
        # _get_api callers both pass the None-check, both build an ApiClient,
        # and the first assignment is overwritten and its aiohttp session is
        # leaked.
        self._kube_client_lock = asyncio.Lock()
        self._pods: dict[str, PodInfo] = {}
        self._restart_warned: set[str] = set()

    def _is_external_service(self, service_type: ServiceTypeT) -> bool:
        """Check if a service type is an external Kubernetes pod."""
        return service_type in EXTERNAL_K8S_SERVICES

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Register expectations for an externally managed Kubernetes service.

        For service types listed in ``EXTERNAL_K8S_SERVICES``, this is a no-op
        for process spawning: Kubernetes manifests launch control-plane
        services as sibling containers and workers/record processors via
        worker pods. We only record how many instances must register with
        ``ServiceRegistry``.

        For any other (non-external) service types, delegates to the parent
        ``MultiProcessServiceManager.run_service`` which spawns a subprocess.

        Raises:
            (Non-external services only) Propagates any exceptions from the
            parent's subprocess spawn path.
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

        For service types listed in ``EXTERNAL_K8S_SERVICES``, this is a no-op:
        externally managed Kubernetes services receive shutdown over the
        control channel and exit on their own. Returns an empty list.

        For any other (non-external) service types, delegates to the parent
        ``MultiProcessServiceManager.stop_service`` which stops the local
        subprocess and returns a list of exception results.

        Raises:
            (Non-external services only) Propagates any exceptions from the
            parent's subprocess stop path (e.g. ``ServiceProcessDiedError``).
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

        api = self._kube_api
        self._kube_api = None
        if api is not None:
            try:
                await api.close()
            except (OSError, RuntimeError, aiohttp.ClientError) as e:
                self.debug(f"Error closing Kubernetes ApiClient: {e!r}")

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

    # -- Kubernetes API access --
    #
    # ``check_pods_healthy``, ``_get_api``, and ``_monitor_worker_pods`` live
    # here (not on ``PodMonitoringMixin``) so tests can patch ``get_pods``,
    # ``config``, and ``ApiClient`` on this module.

    async def _get_api(self) -> ApiClient:
        """Get or create a cached Kubernetes ApiClient."""
        async with self._kube_client_lock:
            if self._kube_api is None:
                from aiperf.common.noisy_loggers import suppress_noisy_http_loggers

                suppress_noisy_http_loggers()
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    await config.load_kube_config()
                self._kube_api = ApiClient()
            return self._kube_api

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
            api = await self._get_api()
            pods = await get_pods(api, namespace, job_selector(job_id))
            self._raise_for_any_failed_pod(aggregate_pods_by_index(pods))
        except ServiceProcessDiedError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - pod health check is advisory, must not raise
            self.warning(f"Pod health check before PROFILE_START failed: {e!r}")

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
            api = await self._get_api()
            pods = await get_pods(api, namespace, job_selector(job_id))
            self._process_pod_snapshots(aggregate_pods_by_index(pods), time.time_ns())
            self._check_pod_failure_threshold()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - pod monitoring loop must not crash on transient k8s errors
            self.warning(f"Failed to query Kubernetes pod statuses: {e!r}")

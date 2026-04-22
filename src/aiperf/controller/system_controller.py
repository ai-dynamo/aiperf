# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
import traceback
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import orjson
from msgspec import Struct
from rich.console import Console
from rich.panel import Panel

from aiperf.cli_utils import (
    print_developer_mode_warning,
    warn_osl_without_ignore_eos,
)
from aiperf.common.base_service import BaseService
from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
    CommandResponse,
    ControllerBoundMessage,
    Heartbeat,
    MemoryReport,
    Registration,
    RegistrationAck,
    ServerMetricsStatus,
    StatusUpdate,
    TelemetryStatus,
)
from aiperf.config.defaults import OutputDefaults
from aiperf.config.zmq import ZMQDualBindConfig

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
from pydantic import Field

from aiperf.common.enums import (
    CommAddress,
    CommandType,
    LifecycleState,
    MessageType,
    ServiceRegistrationStatus,
    WorkerStartupState,
)
from aiperf.common.environment import Environment
from aiperf.common.error_queue import (
    ErrorCollector,
    cleanup_global_error_queue,
)
from aiperf.common.exceptions import (
    LifecycleOperationError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.hooks import (
    AIPerfHook,
    on_command,
    on_init,
    on_message,
    on_start,
    on_stop,
)
from aiperf.common.logging import cleanup_global_log_queue, get_global_log_queue
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.memory_tracker import (
    MemoryPhase,
    MemoryReading,
    MemoryTracker,
    read_pss_self,
)
from aiperf.common.messages import (
    BenchmarkCompleteMessage,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
    ProcessTelemetryResultMessage,
    WorkerPodStateMessage,
    WorkerStatusSummaryMessage,
)
from aiperf.common.models import (
    AIPerfBaseModel,
    ErrorDetails,
    ProcessRecordsResult,
)
from aiperf.common.models.error_models import ExitErrorInfo
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT
from aiperf.controller.controller_utils import print_exit_errors
from aiperf.controller.protocols import ServiceManagerProtocol
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.controller.system_mixins import SignalHandlerMixin
from aiperf.credit.messages import CreditsCompleteMessage
from aiperf.exporters.exporter_manager import ExporterManager
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType, ServiceType, UIType
from aiperf.ui.protocols import AIPerfUIProtocol
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient


class AggregateWorkerStatus(AIPerfBaseModel):
    """Controller-authored aggregate worker-pod status snapshot."""

    ready: int = Field(default=0, description="Dispatch-ready worker count.")
    total: int = Field(default=0, description="Declared worker count.")
    dispatchable: int = Field(
        default=0,
        description="Workers eligible to receive credits.",
    )
    router_connected: int = Field(
        default=0,
        description="Workers connected to the credit router.",
    )
    ready_record_processors: int = Field(
        default=0,
        description="Record processors currently available across worker pods.",
    )
    declared_record_processors: int = Field(
        default=0,
        description="Declared record-processor count across worker pods.",
    )
    ready_pods: int = Field(
        default=0,
        description="Pods with usable worker capacity.",
    )
    total_pods: int = Field(
        default=0,
        description="Total worker pods seen by the controller.",
    )
    degraded_pods: int = Field(
        default=0,
        description="Pods that are usable but degraded.",
    )


@dataclass(frozen=True, slots=True)
class K8sServiceTopology:
    """Expected Kubernetes worker-pod topology derived from runtime config."""

    num_worker_pods: int
    """Number of Kubernetes worker pods to deploy."""

    workers_per_pod: int
    """Number of worker processes per pod."""

    record_processors_per_pod: int
    """Number of record processor processes per pod."""

    total_workers: int
    """Total worker count across all pods."""

    total_record_processors: int
    """Total record processor count across all pods."""


def build_aggregate_worker_status(
    pod_states: dict[str, WorkerPodStateMessage],
) -> AggregateWorkerStatus:
    """Summarize worker-pod snapshots into controller aggregate status."""
    pods = list(pod_states.values())
    return AggregateWorkerStatus(
        ready=sum(pod.ready_workers for pod in pods),
        total=sum(pod.declared_workers for pod in pods),
        dispatchable=sum(pod.dispatchable_workers for pod in pods),
        router_connected=sum(pod.router_connected_workers for pod in pods),
        ready_record_processors=sum(pod.ready_record_processors for pod in pods),
        declared_record_processors=sum(pod.declared_record_processors for pod in pods),
        ready_pods=sum(
            1
            for pod in pods
            if pod.dispatchable_workers >= 1 and pod.ready_record_processors >= 1
        ),
        total_pods=len(pods),
        degraded_pods=sum(
            1
            for pod in pods
            if pod.dispatchable_workers >= 1
            and pod.ready_record_processors >= 1
            and (pod.degraded_workers > 0 or pod.degraded_record_processors > 0)
        ),
    )


class SystemController(SignalHandlerMixin, BaseService):
    """System Controller service.

    This service is responsible for managing the lifecycle of all other services.
    It will start, stop, and configure all other services.
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )
        self.debug("Creating System Controller")
        if Environment.DEV.MODE:
            # Print a warning message to the console if developer mode is enabled, once at load time
            print_developer_mode_warning()

        # EOS may cause server to stop early, producing misleading OSL results
        if self._should_warn_osl_without_ignore_eos():
            warn_osl_without_ignore_eos()

        self._was_cancelled = False
        is_k8s_mode = self.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES
        self._k8s_topology: K8sServiceTopology | None = None
        self._declared_group_capacities: dict[str, tuple[int, int]] = {}

        self.required_services: dict[ServiceTypeT, int] = {
            ServiceType.DATASET_MANAGER: 1,
            ServiceType.TIMING_MANAGER: 1,
            ServiceType.RECORDS_MANAGER: 1,
            ServiceType.WORKER_GROUP_MANAGER: self.run.cfg.worker_group_service_count,
        }
        self.scale_record_processors_with_workers = False

        if is_k8s_mode:
            self._k8s_topology = self._build_k8s_service_topology()
            self.required_services[ServiceType.WORKER_GROUP_MANAGER] = (
                self._k8s_topology.num_worker_pods
            )

        event_bus_sidecar_enabled = False
        if is_k8s_mode:
            from aiperf.kubernetes.environment import K8sEnvironment

            event_bus_sidecar_enabled = K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED

        self.proxy_manager: ProxyManager = ProxyManager(
            run=self.run,
            enable_event_bus=not event_bus_sidecar_enabled,
            enable_dataset_manager=True,
            enable_raw_inference=False,
        )

        # Control ROUTER lives outside the comms lifecycle so it stays
        # alive after comms.stop() — child processes still need it during
        # their own shutdown sequence.
        additional_bind: str | None = None
        comm_config = self.run.resolved.comm_config or self.run.cfg.comm_config
        if (
            isinstance(comm_config, ZMQDualBindConfig)
            and not comm_config.controller_host
        ):
            additional_bind = comm_config.control_tcp_bind_address

        control_address = self.comms.get_address(CommAddress.CONTROL)
        self.info(
            f"Creating control ROUTER client: "
            f"address={control_address}, additional_bind={additional_bind}"
        )
        import zmq as _zmq

        self.control_router = ZMQStreamingRouterClient(
            address=control_address,
            bind=True,
            additional_bind_address=additional_bind,
            decode_type=ControllerBoundMessage,
            socket_ops={_zmq.ROUTER_MANDATORY: 1},
        )

        ServiceManagerClass = plugins.get_class(
            PluginType.SERVICE_MANAGER, self.run.cfg.runtime.service_run_type
        )

        using_dashboard = self.run.cfg.ui_type == UIType.DASHBOARD
        log_queue = get_global_log_queue() if using_dashboard else None
        self._error_collector = ErrorCollector(
            logger=self, exit_errors=self._exit_errors
        )

        self.service_manager: ServiceManagerProtocol = ServiceManagerClass(
            required_services=self.required_services,
            run=self.run,
            log_queue=log_queue,
            error_queue=self._error_collector.error_queue,
        )
        UIClass = plugins.get_class(PluginType.UI, self.run.cfg.ui_type)
        self.ui: AIPerfUIProtocol = UIClass(
            run=self.run,
            log_queue=log_queue,
            controller=self,
        )
        self.attach_child_lifecycle(self.ui)
        self._profile_results: ProcessRecordsResult | None = None
        self._exit_errors: list[ExitErrorInfo] = []
        self._telemetry_results: TelemetryExportData | None = None
        self._server_metrics_results: ServerMetricsResults | None = None
        self._profile_results_received = False
        self._should_wait_for_telemetry = False
        self._should_wait_for_server_metrics = False

        self._shutdown_triggered = False
        self._shutdown_lock = asyncio.Lock()
        self._results_exported = False
        self._exporter_manager: ExporterManager | None = None
        self._memory_tracker = MemoryTracker()

        # Configure-on-register: when enabled, each service receives
        # PROFILE_CONFIGURE immediately upon registration instead of
        # waiting for all services to register first.
        self._auto_configure: bool = False
        self._configuring_ids: set[str] = set()
        self._configured_ids: set[str] = set()
        self._all_configured_event: asyncio.Event = asyncio.Event()
        self._configure_errors: list[CommandResponse | ErrorDetails] = []
        self._configure_scheduler: LoopScheduler | None = None

        self._telemetry_endpoints_configured: list[str] = []
        self._telemetry_endpoints_reachable: list[str] = []
        self._server_metrics_endpoints_configured: list[str] = []
        self._worker_startup_states: dict[str, str] = {}
        self._pod_states: dict[str, WorkerPodStateMessage] = {}
        self._all_workers_ready_event: asyncio.Event = asyncio.Event()
        self._server_metrics_endpoints_reachable: list[str] = []
        self._pod_failure_watcher_task: asyncio.Task | None = None
        self.debug("System Controller created")

    def _build_k8s_service_topology(self) -> K8sServiceTopology:
        """Derive the full Kubernetes worker-pod topology from runtime config.

        Kubernetes deployments are pod-based: each worker pod runs a fixed
        number of worker and record-processor service containers. Startup must wait
        for the full expanded topology rather than the requested logical worker
        count, because the last pod is not partially filled.
        """
        import math

        runtime = self.run.cfg.runtime

        workers_per_pod = (
            runtime.workers_per_pod or Environment.WORKER.DEFAULT_WORKERS_PER_POD
        )
        requested_workers = runtime.workers or workers_per_pod
        num_worker_pods = max(1, math.ceil(requested_workers / workers_per_pod))
        total_workers = num_worker_pods * workers_per_pod

        if runtime.record_processors_per_pod is not None:
            record_processors_per_pod = runtime.record_processors_per_pod
        elif runtime.record_processors is not None:
            record_processors_per_pod = max(
                1, math.ceil(runtime.record_processors / num_worker_pods)
            )
        else:
            record_processors_per_pod = max(
                1, workers_per_pod // Environment.RECORD.PROCESSOR_SCALE_FACTOR
            )
        total_record_processors = num_worker_pods * record_processors_per_pod

        return K8sServiceTopology(
            num_worker_pods=num_worker_pods,
            workers_per_pod=workers_per_pod,
            record_processors_per_pod=record_processors_per_pod,
            total_workers=total_workers,
            total_record_processors=total_record_processors,
        )

    def _should_warn_osl_without_ignore_eos(self) -> bool:
        """Check if --osl is used without ignore_eos or min_tokens in extra inputs."""
        dataset = self.run.cfg.get_default_dataset()
        prompts = getattr(dataset, "prompts", None)
        osl = getattr(prompts, "osl", None) if prompts else None
        if osl is None:
            return False

        extra_inputs = self.run.cfg.endpoint.extra
        if not extra_inputs:
            return True

        # Check if ignore_eos or min_tokens is set with a truthy value
        return not (extra_inputs.get("ignore_eos") or extra_inputs.get("min_tokens"))

    async def request_realtime_metrics(self) -> None:
        """Request real-time metrics from the RecordsManager."""
        rm_ids = [
            s.service_id
            for s in ServiceRegistry.get_services(ServiceType.RECORDS_MANAGER)
        ]
        for sid in rm_ids:
            await self._send_control_command(
                sid, CommandType.REALTIME_METRICS, timeout=5.0
            )

    async def start_realtime_telemetry(self) -> None:
        """Send START_REALTIME_TELEMETRY command to GPUTelemetryManager(s)."""
        gpu_ids = [
            s.service_id
            for s in ServiceRegistry.get_services(ServiceType.GPU_TELEMETRY_MANAGER)
        ]
        for sid in gpu_ids:
            await self._send_control_command(
                sid, CommandType.START_REALTIME_TELEMETRY, timeout=5.0
            )

    async def initialize(self) -> None:
        """We need to override the initialize method to run the proxy manager before the base service initialize.
        This is because the proxies need to be running before we can subscribe to the message bus.
        """
        self.debug("Running ZMQ Proxy Manager Before Initialize")
        await self.proxy_manager.initialize_and_start()
        # Once the proxies are running, call the original initialize method
        await super().initialize()

    @on_init
    async def _initialize_system_controller(self) -> None:
        self.debug("Initializing System Controller")

        # Register the unified receiver that dispatches by message type.
        self.control_router.register_receiver(self._handle_control_message)

        # Initialize and start the control ROUTER independently of comms.
        self.info("Initializing control ROUTER client")
        await self.control_router.initialize()
        self.info(
            f"Control ROUTER initialized (state={self.control_router.state}), starting..."
        )
        await self.control_router.start()
        self.info(f"Control ROUTER started (state={self.control_router.state})")

        self.setup_signal_handlers(self._handle_signal)
        self.debug("Setup signal handlers")

        async with self.try_operation_or_stop("Initialize Service Manager"):
            await self.service_manager.initialize()

        self.debug("System Controller initialized successfully")

    @on_start
    async def _start_services(self) -> None:
        """Bootstrap the system services.

        Services are configured immediately upon registration rather than
        waiting for all services to register first. This overlaps the
        registration and configuration phases for faster startup.
        """
        self.debug("System Controller is bootstrapping services")
        self._controller_pss_at_start = read_pss_self()

        # Enable auto-configure so that each service receives
        # PROFILE_CONFIGURE as soon as it registers.
        self._configure_scheduler = LoopScheduler()
        self._auto_configure = True

        # Flush any services that registered BEFORE auto-configure was
        # enabled (e.g. k8s worker pods whose Registration arrived during
        # initialize/control-router bind, before _start_services ran).
        # Without this flush those services never receive PROFILE_CONFIGURE.
        for sid, info in list(ServiceRegistry.services.items()):
            if (
                info.registration_status == ServiceRegistrationStatus.REGISTERED
                and sid not in self._configuring_ids
            ):
                self.info(
                    f"Flushing pre-CONFIGURING registration for '{sid}' "
                    f"({info.service_type})"
                )
                self._configure_scheduler.execute_async(
                    self._configure_single_service(sid)
                )

        # Collect optional services to spawn alongside required services
        optional_services: list[ServiceTypeT] = []
        if self.run.cfg.gpu_telemetry.enabled:
            optional_services.append(ServiceType.GPU_TELEMETRY_MANAGER)
        else:
            self.info("GPU telemetry disabled via --no-gpu-telemetry")
            self._should_wait_for_telemetry = False

        if self.run.cfg.server_metrics.enabled:
            optional_services.append(ServiceType.SERVER_METRICS_MANAGER)
        else:
            self.info("Server metrics disabled via --no-server-metrics")
            self._should_wait_for_server_metrics = False

        # Start AIPerf API if enabled
        api_port = self.run.cfg.runtime.api_port or Environment.API_SERVER.PORT
        api_host = self.run.cfg.runtime.api_host or Environment.API_SERVER.HOST
        if api_port is not None and api_host is not None:
            self.info(f"Starting AIPerf API server at http://{api_host}:{api_port}/")
            optional_services.append(ServiceType.API)

        total_services = sum(self.required_services.values()) + len(optional_services)
        types_summary = ", ".join(
            f"{st}: {n}" for st, n in self.required_services.items()
        )
        if optional_services:
            types_summary += ", " + ", ".join(f"{st}: 1" for st in optional_services)
        if self._k8s_topology is not None:
            topo = self._k8s_topology
            self.info(
                "Kubernetes startup topology: "
                f"{topo.num_worker_pods} worker pod(s) x "
                f"{topo.workers_per_pod} workers + "
                f"{topo.record_processors_per_pod} record processors per pod "
                f"({topo.total_workers} workers, "
                f"{topo.total_record_processors} record processors total)"
            )
        self.info(f"Preparing {total_services} services ({types_summary})")
        spawn_start = time.perf_counter()

        async with self.try_operation_or_stop("Start Service Manager"):
            await self.service_manager.start()

        startup_tasks = [
            self.service_manager.run_service(st) for st in optional_services
        ]
        if startup_tasks:
            await asyncio.gather(*startup_tasks)

        spawn_elapsed = time.perf_counter() - spawn_start
        self.info(f"All {total_services} services prepared in {spawn_elapsed:.2f}s")

        # Enable pod monitoring early so failed pods are detected during
        # registration/configuration rather than waiting for timeout.
        self.service_manager.activate_pod_monitoring()

        self.info("AIPerf System is CONFIGURING")
        async with self.try_operation_or_stop("Configure Services"):
            await self._wait_for_all_configured(
                timeout=Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT,
            )
        self.info("AIPerf System is CONFIGURED")
        self._auto_configure = False
        self.service_manager.activate_heartbeat_monitoring()

        self.info("Post-configure startup flow: checking pod health")
        # Verify pod health before starting profiling. A pod could have
        # registered its services but since crashed (e.g. OOMKilled).
        async with self.try_operation_or_stop("Pod Health Check"):
            await self.service_manager.check_pods_healthy()

        self.info(
            "Post-configure startup flow: waiting for sufficient worker pod readiness "
            f"(timeout={Environment.SERVICE.PROFILE_START_TIMEOUT}s)"
        )
        async with self.try_operation_or_stop("Wait For Worker Pods Ready"):
            await self._wait_for_sufficient_worker_pods(
                timeout=Environment.SERVICE.PROFILE_START_TIMEOUT,
            )

        self.info("Post-configure startup flow: sending PROFILE_START to all services")
        await self._start_profiling_all_services()
        self.info("AIPerf System is PROFILING")

        # Watch for pod failure threshold breach during profiling
        self._pod_failure_watcher_task = asyncio.create_task(
            self._watch_pod_failure_abort()
        )

    async def _configure_single_service(self, service_id: str) -> None:
        """Send PROFILE_CONFIGURE to a single service and track completion.

        Called as a fire-and-forget task from the registration handler so that
        each service begins configuration immediately upon registering.

        Retries on transient ZMQ errors ("stream is closed") which occur when
        the TCP connection between pods drops during idle periods. The DEALER
        auto-reconnects (ZMQ_RECONNECT_IVL) and the ROUTER accepts the new
        connection (ZMQ_ROUTER_HANDOVER), so a retry on a new connection works.
        """
        self._configuring_ids.add(service_id)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            self.debug(
                lambda sid=service_id,
                att=attempt: f"Sending PROFILE_CONFIGURE to '{sid}' (attempt {att}/{max_retries})"
            )
            try:
                response = await self._send_control_command(
                    service_id,
                    CommandType.PROFILE_CONFIGURE,
                    timeout=Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT,
                )
                if isinstance(response, CommandErr):
                    self._configure_errors.append(response)
                    self._all_configured_event.set()
                    return
                break  # success
            except Exception as e:
                is_stream_error = (
                    "stream" in str(e).lower() or "closed" in str(e).lower()
                )
                if is_stream_error and attempt < max_retries:
                    self.warning(
                        f"PROFILE_CONFIGURE to '{service_id}' failed (attempt {attempt}): {e}. "
                        f"Retrying in 3s (DEALER will auto-reconnect)..."
                    )
                    await asyncio.sleep(3)
                    continue
                self.error(f"PROFILE_CONFIGURE to '{service_id}' failed: {e}")
                self._configure_errors.append(ErrorDetails.from_exception(e))
                self._all_configured_event.set()
                return

        self._configured_ids.add(service_id)
        total = len(self._configuring_ids)
        self.info(f"Configured '{service_id}' ({len(self._configured_ids)}/{total})")
        if self._all_expected_configured():
            self._all_configured_event.set()

    def _all_expected_configured(self) -> bool:
        """Check if every expected service has been configured.

        Verifies both:
        - All individually-expected service IDs (from expect_service) are configured
        - All type-count expectations (from expect_services) are met
        """
        expected_ids = ServiceRegistry.expected_ids
        expected_by_type = ServiceRegistry.expected_by_type
        if not expected_ids and not expected_by_type:
            return False
        if not expected_ids.issubset(self._configured_ids):
            return False
        for stype, expected_count in expected_by_type.items():
            configured_count = sum(
                1
                for sid in self._configured_ids
                if sid in ServiceRegistry.services
                and ServiceRegistry.services[sid].service_type == stype
            )
            if configured_count < expected_count:
                return False
        return True

    def _get_pending_type_counts(self) -> dict[str, str]:
        """Get type counts that haven't reached their expected configured count."""
        pending: dict[str, str] = {}
        for stype, expected_count in ServiceRegistry.expected_by_type.items():
            configured_count = sum(
                1
                for sid in self._configured_ids
                if sid in ServiceRegistry.services
                and ServiceRegistry.services[sid].service_type == stype
            )
            if configured_count < expected_count:
                pending[str(stype)] = f"{configured_count}/{expected_count}"
        return pending

    def _cancel_configure_tasks(self) -> None:
        """Cancel in-flight configure tasks and clear tracking."""
        if self._configure_scheduler is not None:
            self._configure_scheduler.cancel_all()

    async def _wait_for_all_configured(self, timeout: float) -> None:
        """Wait until all expected services have been configured.

        Uses fail-fast: if any service returns an error during configuration,
        we abort immediately.

        The _all_configured_event is set by:
        - _configure_single_service: on success (all expected done) or error
        - _cancel_profiling: on Ctrl+C signal
        - ServiceRegistry.fail_service wakes this via _failure_event
        """
        begin = time.perf_counter()

        if not self._all_expected_configured():
            # Ensure ServiceRegistry has a failure event we can watch
            if ServiceRegistry._failure_event is None:
                ServiceRegistry._failure_event = asyncio.Event()
            failure_event = ServiceRegistry._failure_event

            progress_task = asyncio.create_task(
                self._log_configure_progress(begin, timeout)
            )
            try:
                # Wait for any of: all configured, service failure, or timeout
                config_waiter = asyncio.create_task(self._all_configured_event.wait())
                failure_waiter = asyncio.create_task(failure_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        {config_waiter, failure_waiter},
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    config_waiter.cancel()
                    failure_waiter.cancel()

                if not done:
                    # Timeout -- neither event fired
                    self._cancel_configure_tasks()
                    pending_ids = ServiceRegistry.expected_ids - self._configured_ids
                    pending_types = self._get_pending_type_counts()
                    startup_summary = self._summarize_pending_worker_startup_states(
                        pending_ids
                    )
                    startup_detail = (
                        f", Pending worker startup: {startup_summary}"
                        if startup_summary
                        else ""
                    )
                    raise ServiceRegistrationTimeoutError(
                        f"Timed out waiting for services to configure "
                        f"({len(self._configured_ids)} configured). "
                        f"Pending IDs: {pending_ids}, "
                        f"Pending types: {pending_types}"
                        f"{startup_detail}",
                        missing={},
                    ) from None

                # Something woke us -- check what
                self._cancel_configure_tasks()

                # Cancellation (Ctrl+C)
                if self._was_cancelled:
                    raise asyncio.CancelledError(
                        "Configuration interrupted by shutdown"
                    )

                # Service process died
                ServiceRegistry._raise_on_failure()

                # Configure task returned an error
                self._parse_control_responses_for_errors(
                    self._configure_errors, "Configure Profiling"
                )

                # Verify all expected services are actually configured.
                if not self._all_expected_configured():
                    pending_ids = ServiceRegistry.expected_ids - self._configured_ids
                    pending_types = self._get_pending_type_counts()
                    startup_summary = self._summarize_pending_worker_startup_states(
                        pending_ids
                    )
                    startup_detail = (
                        f", Pending worker startup: {startup_summary}"
                        if startup_summary
                        else ""
                    )
                    raise ServiceRegistrationTimeoutError(
                        f"Configuration wait ended but not all services "
                        f"configured. Pending IDs: {pending_ids}, "
                        f"Pending types: {pending_types}"
                        f"{startup_detail}",
                        missing={},
                    ) from None

            finally:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task
        else:
            self._cancel_configure_tasks()
            self._parse_control_responses_for_errors(
                self._configure_errors, "Configure Profiling"
            )

        self.info(
            f"All services configured in {time.perf_counter() - begin:.2f} seconds"
        )

        if not Environment.HTTP.SSL_VERIFY:
            self.warning(
                "SSL certificate verification is DISABLED - this is insecure. "
                "This should only be used for testing in a trusted environment."
            )

    async def _log_configure_progress(self, begin: float, timeout: float) -> None:
        """Log periodic progress during configuration wait."""
        interval = 5.0
        while True:
            await asyncio.sleep(interval)
            elapsed = time.perf_counter() - begin
            pending_types = self._get_pending_type_counts()
            pending_ids = ServiceRegistry.expected_ids - self._configured_ids
            configured = len(self._configured_ids)
            total = ServiceRegistry._total_expected
            msg = (
                f"Waiting for configuration: {configured}/{total} "
                f"({elapsed:.1f}s elapsed). "
                f"Pending IDs: {pending_ids}, Pending types: {pending_types}"
            )
            startup_summary = self._summarize_pending_worker_startup_states(pending_ids)
            if startup_summary:
                msg += f", Pending worker startup: {startup_summary}"
            pod_summary = self.service_manager.get_pod_summary()
            if pod_summary:
                msg += f", Pod states: {pod_summary}"
            self.info(msg)

    def _summarize_pending_worker_startup_states(
        self, pending_ids: set[str]
    ) -> dict[str, int]:
        """Summarize startup states for workers still pending configuration."""
        summary: dict[str, int] = {}
        for worker_id in pending_ids:
            state = self._worker_startup_states.get(worker_id)
            if state is None:
                continue
            summary[state] = summary.get(state, 0) + 1
        return summary

    def _all_expected_workers_ready(self) -> bool:
        """Check whether all expected workers are in app-level READY state."""
        expected_workers = self.required_services.get(ServiceType.WORKER, 0)
        if expected_workers <= 0:
            return True
        ready_workers = [
            worker_id
            for worker_id, state in self._worker_startup_states.items()
            if state == str(WorkerStartupState.READY)
        ]
        return len(ready_workers) >= expected_workers

    def get_aggregate_worker_status(self) -> AggregateWorkerStatus:
        """Return the controller-authored aggregate worker-pod status snapshot."""
        return build_aggregate_worker_status(self._pod_states)

    def _ready_worker_pod_count(self) -> int:
        """Count worker pods that are currently dispatchable."""
        return sum(
            1
            for pod in self._pod_states.values()
            if pod.dispatchable_workers >= 1 and pod.ready_record_processors >= 1
        )

    def _all_target_worker_pods_ready(self) -> bool:
        """Check whether the full desired worker-pod topology is ready."""
        if self._k8s_topology is None:
            return self._ready_worker_pod_count() >= 1
        return self._ready_worker_pod_count() >= self._k8s_topology.num_worker_pods

    def _has_sufficient_ready_worker_pods(self) -> bool:
        """Check whether enough worker pods are dispatchable to start profiling."""
        if self.run.cfg.runtime.service_run_type != ServiceRunType.KUBERNETES:
            return True
        return self._ready_worker_pod_count() >= 1

    async def _wait_for_sufficient_worker_pods(self, timeout: float) -> None:
        """Wait until enough worker pods are dispatchable to start profiling."""
        if self.run.cfg.runtime.service_run_type != ServiceRunType.KUBERNETES:
            return
        begin = time.perf_counter()
        grace_period = min(5.0, timeout)
        self._all_workers_ready_event.clear()
        while True:
            elapsed = time.perf_counter() - begin
            if self._all_target_worker_pods_ready():
                return
            if elapsed >= grace_period and self._has_sufficient_ready_worker_pods():
                return
            remaining = timeout - elapsed
            if remaining <= 0:
                raise ServiceRegistrationTimeoutError(
                    "Timed out waiting for sufficient worker pod readiness",
                    missing={},
                ) from None
            group_manager_ids = [
                service.service_id
                for service in ServiceRegistry.get_services(
                    ServiceType.WORKER_GROUP_MANAGER
                )
            ]
            for service_id in group_manager_ids:
                with contextlib.suppress(Exception):
                    await self._send_control_command(
                        service_id,
                        CommandType.REPORT_WORKER_STATUS_SUMMARY,
                        timeout=min(remaining, 5.0),
                    )
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._all_workers_ready_event.wait(),
                    timeout=min(remaining, 5.0),
                )
            self._all_workers_ready_event.clear()

    @on_message(MessageType.WORKER_STATUS_SUMMARY)
    async def _on_worker_status_summary(
        self, message: WorkerStatusSummaryMessage
    ) -> None:
        """Track worker startup states for diagnostics."""
        for worker_id, startup_state in message.worker_startup_states.items():
            self._worker_startup_states[worker_id] = str(startup_state)

    @on_message(MessageType.WORKER_POD_STATE)
    async def _on_worker_pod_state(self, message: WorkerPodStateMessage) -> None:
        """Track aggregate worker-pod snapshots for Kubernetes startup gating."""
        self._pod_states[message.pod_index] = message
        if self._has_sufficient_ready_worker_pods():
            self._all_workers_ready_event.set()

    async def _wait_for_endpoint_ready(self) -> None:
        """Deprecated. Endpoint readiness is now a CLI preflight; see
        ``cli_runner._preflight_endpoint_ready``. This stub remains only so
        older tests that stub this method don't break — remove once tests
        are updated.
        """
        return

    async def _start_profiling_all_services(self) -> None:
        """Tell all services to start profiling.

        Uses fail-fast behavior: if any service returns an error,
        we abort immediately without waiting for the remaining services.
        """
        self.debug("Sending PROFILE_START command to all services")
        responses = await self._send_control_command_to_all_fail_fast(
            CommandType.PROFILE_START,
            list(ServiceRegistry.get_all_registered_ids()),
            timeout=Environment.SERVICE.PROFILE_START_TIMEOUT,
        )
        self._parse_control_responses_for_errors(responses, "Start Profiling")
        self.info("All services started profiling successfully")

    def _parse_control_responses_for_errors(
        self,
        responses: list[CommandResponse | ErrorDetails],
        operation: str,
    ) -> None:
        """Parse control channel command responses for errors."""
        for response in responses:
            if isinstance(response, ErrorDetails):
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=response, operation=operation, service_id=None
                    )
                )
            elif isinstance(response, CommandErr):
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=ErrorDetails(
                            type="CommandError",
                            message=response.error,
                            cause=response.traceback or None,
                        ),
                        operation=operation,
                        service_id=response.sid,
                    )
                )
        if self._exit_errors:
            raise LifecycleOperationError(
                operation=operation,
                original_exception=None,
                lifecycle_id=self.id,
            )

    @on_message(MessageType.CREDITS_COMPLETE)
    async def _process_credits_complete_message(
        self, message: CreditsCompleteMessage
    ) -> None:
        """Log receipt of credits complete message from a service.

        Args:
            message: The credits complete message to process
        """
        service_id = message.service_id
        self.info(f"Received credits complete from '{service_id}'")

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _handle_profile_complete_relay(self, message: Command) -> None:
        """Relay PROFILE_COMPLETE from RecordsManager to GPU telemetry and server metrics services."""
        target_types = [
            ServiceType.GPU_TELEMETRY_MANAGER,
            ServiceType.SERVER_METRICS_MANAGER,
            ServiceType.WORKER_GROUP_MANAGER,
        ]
        target_ids = []
        for stype in target_types:
            target_ids.extend(s.service_id for s in ServiceRegistry.get_services(stype))

        if target_ids:
            await self._send_control_command_to_all(
                CommandType.PROFILE_COMPLETE, target_ids, timeout=10.0
            )

    @on_message(MessageType.PROCESS_RECORDS_RESULT)
    async def _on_process_records_result_message(
        self, message: ProcessRecordsResultMessage
    ) -> None:
        """Handle a profile results message."""
        self.trace_or_debug(
            lambda: f"Received profile results message: {message}",
            lambda: (
                f"Received profile results message: {len(message.results.results.records) if message.results.results else 0} records"
            ),
        )
        if message.results.errors:
            self.error(
                f"Received process records result message with errors: {message.results.errors}"
            )

        self.debug(
            lambda: (
                f"Error summary: {message.results.results.error_summary if message.results.results else 'N/A'}"
            )
        )

        self._profile_results = message.results

        if not message.results.results:
            self.error(
                f"Received process records result message with no records: {message.results.results}"
            )

        self._profile_results_received = True
        # Coordinate with telemetry results before shutdown
        await self._check_and_trigger_shutdown()

    @on_message(MessageType.PROCESS_TELEMETRY_RESULT)
    async def _on_process_telemetry_result_message(
        self, message: ProcessTelemetryResultMessage
    ) -> None:
        """Handle a telemetry results message."""
        try:
            self.trace_or_debug(
                lambda: f"Received telemetry results message: {message}",
                lambda: (
                    f"Received telemetry results message: {len(message.telemetry_result.results.endpoints) if message.telemetry_result.results else 0} endpoints"
                ),
            )

            telemetry_results = message.telemetry_result.results
            if not telemetry_results:
                self.error(
                    f"Received process telemetry result message with no records: {telemetry_results}"
                )
            else:
                # Update endpoint info in the summary (TelemetryExportData structure)
                telemetry_results.summary.endpoints_configured = (
                    self._telemetry_endpoints_configured
                )
                telemetry_results.summary.endpoints_successful = (
                    self._telemetry_endpoints_reachable
                )

            self._telemetry_results = telemetry_results
        except Exception as e:
            self.exception(f"Error processing telemetry results message: {e!r}")
        finally:
            self._should_wait_for_telemetry = False
            await self._check_and_trigger_shutdown()

    @on_message(MessageType.PROCESS_SERVER_METRICS_RESULT)
    async def _on_process_server_metrics_result_message(
        self, message: ProcessServerMetricsResultMessage
    ) -> None:
        """Handle a server metrics results message."""
        try:
            self.trace_or_debug(
                lambda: f"Received server metrics results message: {message}",
                lambda: (
                    f"Received server metrics results message: {len(message.server_metrics_result.results.endpoint_summaries or {}) if message.server_metrics_result.results else 0} endpoints"
                ),
            )

            self.debug(
                lambda: (
                    f"Server metrics error summary: {message.server_metrics_result.results.error_summary if message.server_metrics_result.results else 'N/A'}"
                )
            )

            server_metrics_results = message.server_metrics_result.results

            if not server_metrics_results:
                self.debug(
                    f"Received process server metrics result message with no results: {server_metrics_results}"
                )
            else:
                server_metrics_results.endpoints_configured = (
                    self._server_metrics_endpoints_configured
                )
                server_metrics_results.endpoints_successful = (
                    self._server_metrics_endpoints_reachable
                )

            self._server_metrics_results = server_metrics_results
        except Exception as e:
            self.exception(f"Error processing server metrics results message: {e!r}")
        finally:
            self._should_wait_for_server_metrics = False
            await self._check_and_trigger_shutdown()

    async def _check_and_trigger_shutdown(self) -> None:
        """Check if all required results are received and trigger unified export + shutdown.

        Coordination logic:
        1. Always wait for profile results (ProcessRecordsResultMessage)
        2. If telemetry disabled OR telemetry results received → proceed
        3. If server metrics disabled OR server metrics results received → proceed
        4. Otherwise → wait (results arrive nearly simultaneously and will call this method again)

        Thread safety:
        Uses self._shutdown_lock to prevent race conditions when ProcessRecordsResultMessage,
        ProcessTelemetryResultMessage, and ProcessServerMetricsResultMessage arrive concurrently.
        The lock ensures atomic check-and-set of _shutdown_triggered, preventing double-triggering of stop().
        """
        self.debug(
            lambda: f"_check_and_trigger_shutdown: profile_received={self._profile_results_received}, "
            f"wait_telemetry={self._should_wait_for_telemetry}, telemetry_results={self._telemetry_results is not None}, "
            f"wait_server_metrics={self._should_wait_for_server_metrics}, server_metrics_results={self._server_metrics_results is not None}, "
            f"shutdown_triggered={self._shutdown_triggered}"
        )
        # Check if we should trigger shutdown (with lock protection)
        should_shutdown = False
        async with self._shutdown_lock:
            if self._shutdown_triggered:
                self.debug(
                    "_check_and_trigger_shutdown: shutdown already triggered, returning"
                )
                return

            if not self._profile_results_received:
                self.debug(
                    "_check_and_trigger_shutdown: profile results not received yet"
                )
                return

            telemetry_ready_for_shutdown = (
                not self._should_wait_for_telemetry
                or self._telemetry_results is not None
            )

            server_metrics_ready_for_shutdown = (
                not self._should_wait_for_server_metrics
                or self._server_metrics_results is not None
            )

            if telemetry_ready_for_shutdown and server_metrics_ready_for_shutdown:
                self._shutdown_triggered = True
                should_shutdown = True
                self.info("All results received, initiating shutdown")
            else:
                if not telemetry_ready_for_shutdown:
                    self.info("Waiting for telemetry results...")
                if not server_metrics_ready_for_shutdown:
                    self.info("Waiting for server metrics results...")

        # Call stop() OUTSIDE the lock to prevent deadlock
        if should_shutdown:
            # Export results BEFORE shutdown — files must exist on disk before
            # the API reports "complete" and before any external consumer
            # (operator, CLI) tries to fetch them. Previously the export ran
            # inside @on_stop, creating a race where the operator fetched
            # partial results before the export finished.
            if self._profile_results and self._profile_results.results.records:
                await self._export_results_data()
            self.debug("Calling self.stop()...")
            await asyncio.shield(self.stop())
            self.debug("self.stop() completed")

    async def _handle_signal(self, sig: int) -> None:
        """Handle received signals with two-stage cancellation.

        First Ctrl+C: Graceful cancel - stops issuing new credits, cancels
        in-flight requests, and writes results to files.

        Second Ctrl+C: Force quit - immediately terminates all processes.
        Results may be incomplete or not written.

        Args:
            sig: The signal number received
        """
        if self._was_cancelled:
            # SECOND Ctrl+C - Force quit immediately
            self._print_force_quit_warning()
            self.warning(f"Force quit requested (signal {sig})")
            await self._kill()
            return

        # FIRST Ctrl+C - Graceful cancel with warning
        self._print_cancel_warning()
        self.warning(f"Graceful shutdown requested (signal {sig})")
        await self._cancel_profiling()

    def _print_cancel_warning(self) -> None:
        """Print prominent warning panel on first Ctrl+C.

        Informs user that the benchmark is being cancelled gracefully and
        results are being processed. Also instructs how to force quit.

        Uses stderr to ensure visibility even when stdout is redirected or
        captured by the UI.
        """
        console = Console(file=sys.stderr, force_terminal=True)
        console.print()
        console.print(
            Panel(
                "[bold yellow]⚠️  BENCHMARK CANCELLED[/bold yellow]\n\n"
                "Stopping credit issuance and cancelling in-flight requests...\n"
                "Results will be written to files.\n\n"
                "[dim]Press Ctrl+C again to force quit immediately[/dim]\n"
                "[dim](results may be incomplete or not written)[/dim]",
                border_style="yellow",
                padding=(1, 2),
                title="[bold yellow]Cancellation in Progress[/bold yellow]",
            )
        )
        console.print()
        console.file.flush()

    def _print_force_quit_warning(self) -> None:
        """Print warning panel on second Ctrl+C (force quit).

        Warns user that results may be incomplete due to immediate termination.

        Uses stderr to ensure visibility even when stdout is redirected or
        captured by the UI.
        """
        console = Console(file=sys.stderr, force_terminal=True)
        console.print()
        console.print(
            Panel(
                "[bold red]🛑 FORCE QUIT[/bold red]\n\n"
                "Terminating all processes immediately.\n"
                "Results may be incomplete or not written to files.",
                border_style="red",
                padding=(1, 2),
                title="[bold red]Force Quit[/bold red]",
            )
        )
        console.print()
        console.file.flush()

    async def _watch_pod_failure_abort(self) -> None:
        """Watch for pod failure threshold breach and cancel profiling."""
        await self.service_manager.pod_failure_abort_event.wait()
        if self._was_cancelled or self._shutdown_triggered:
            return
        reason = self.service_manager.pod_failure_abort_reason
        self.error(f"Aborting benchmark: {reason}")
        await self._cancel_profiling()

    async def _cancel_profiling(self) -> None:
        self.debug("Cancelling profiling of all services")
        self._was_cancelled = True
        if self._pod_failure_watcher_task and not self._pod_failure_watcher_task.done():
            self._pod_failure_watcher_task.cancel()
        self._cancel_configure_tasks()
        self._all_configured_event.set()
        self.service_manager.notify_shutdown()

        # Mark shutdown as triggered FIRST to prevent _check_and_trigger_shutdown()
        # from also calling stop() when results arrive during cancellation.
        should_call_stop = False
        async with self._shutdown_lock:
            if not self._shutdown_triggered:
                self._shutdown_triggered = True
                should_call_stop = True
            else:
                self.debug("Shutdown already triggered, skipping stop() call")

        # Send cancel to all registered services. Wait only for RecordsManager
        # response since it returns ProcessRecordsResult.
        all_ids = list(ServiceRegistry.get_all_registered_ids())
        records_manager_ids = {
            s.service_id
            for s in ServiceRegistry.get_services(ServiceType.RECORDS_MANAGER)
        }
        self.debug(
            f"Sending cancel to {len(all_ids)} services, waiting for {len(records_manager_ids)} RecordsManager(s)"
        )

        try:
            # Fire-and-forget cancel to non-RecordsManager services
            non_rm_ids = [sid for sid in all_ids if sid not in records_manager_ids]
            for sid in non_rm_ids:
                with contextlib.suppress(Exception):
                    await self.control_router.send_to(
                        sid,
                        Command(
                            cid=uuid.uuid4().hex,
                            cmd=CommandType.PROFILE_CANCEL,
                        ),
                    )

            # Wait for RecordsManager responses (they return ProcessRecordsResult)
            responses = await self._send_control_command_to_all(
                CommandType.PROFILE_CANCEL,
                list(records_manager_ids),
                timeout=Environment.SERVICE.PROFILE_CANCEL_TIMEOUT,
            )

            for response in responses:
                if isinstance(response, ErrorDetails):
                    self.warning(
                        f"Cancel command error (timeout or service unavailable): {response}"
                    )
                elif isinstance(response, CommandErr):
                    self.warning(
                        f"Cancel command failed from {response.sid}: {response.error}"
                    )

            # Extract ProcessRecordsResult from RecordsManager's CommandOk response
            for response in responses:
                if isinstance(response, CommandOk) and response.payload:
                    try:
                        data = orjson.loads(response.payload)
                        result = ProcessRecordsResult.model_validate(data)
                        self.debug(
                            f"Received ProcessRecordsResult from cancel command: {result}"
                        )
                        self._profile_results = result
                        self._profile_results_received = True
                        break
                    except Exception as e:
                        self.warning(f"Failed to parse cancel response payload: {e}")
        except Exception as e:
            self.warning(f"Exception during cancel command (proceeding to stop): {e!r}")

        if should_call_stop:
            self.debug("Stopping system controller after profiling cancelled")
            await asyncio.shield(self.stop())

    @on_stop
    async def _stop_system_controller(self) -> None:
        """Stop the system controller and all running services."""
        # Check if we're in Kubernetes mode with API enabled
        is_k8s_mode = self.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES
        keep_api_running = is_k8s_mode and self.run.cfg.runtime.api_port

        if keep_api_running:
            # In Kubernetes mode with API: signal benchmark completion to API service
            # so it can continue serving results after other services shut down
            await self.publish(
                BenchmarkCompleteMessage(
                    service_id=self.service_id,
                    was_cancelled=self._was_cancelled,
                )
            )

        # Suppress heartbeat/process monitors before broadcasting shutdown
        self.service_manager.notify_shutdown()

        # Send shutdown command to all registered services via ROUTER (fire-and-forget)
        all_ids = list(ServiceRegistry.get_all_registered_ids())
        if keep_api_running:
            api_ids = {
                s.service_id for s in ServiceRegistry.get_services(ServiceType.API)
            }
            all_ids = [sid for sid in all_ids if sid not in api_ids]
        for sid in all_ids:
            try:
                await self.control_router.send_to(
                    sid,
                    Command(
                        cid=uuid.uuid4().hex,
                        cmd=CommandType.SHUTDOWN,
                    ),
                )
            except Exception as e:
                self.debug(f"Failed to send shutdown to {sid}: {e}")

        # Brief delay for messages to propagate before tearing down services
        await asyncio.sleep(Environment.SERVICE.SHUTDOWN_PROPAGATION_DELAY)

        await self.service_manager.shutdown_all_services()

        # In K8s mode with RAW export, wait for worker pods to upload raw records
        # to the API before stopping comms. Workers upload during their shutdown
        # sequence (after flushing RecordProcessor buffers to disk).
        if is_k8s_mode and self._should_wait_for_raw_records():
            await self._wait_for_raw_record_uploads()

        await self.comms.stop()
        await self.proxy_manager.stop()
        self.info(f"Stopping control ROUTER client (state={self.control_router.state})")
        await self.control_router.stop()
        self.info("Control ROUTER client stopped")

        # Drain subprocess errors reported via the error queue backchannel
        self._error_collector.drain_into()

        # Post-benchmark reporting (after services and comms are stopped).
        # Bound with a timeout: the Dashboard UI can hang when the parent
        # process runs under PIPE'd stdio (integration tests under xdist)
        # because Textual's driver waits on a terminal that never arrives.
        try:
            await asyncio.wait_for(self.ui.stop(), timeout=5.0)
        except (asyncio.TimeoutError, Exception) as e:
            self.warning(f"UI stop did not complete cleanly: {e!r}")
        try:
            await asyncio.wait_for(self.ui.wait_for_tasks(), timeout=5.0)
        except (asyncio.TimeoutError, Exception) as e:
            self.warning(f"UI task drain did not complete cleanly: {e!r}")
        await asyncio.sleep(0.1)

        if not self._exit_errors:
            if self._profile_results and self._profile_results.results.records:
                await self._print_post_benchmark_info_and_metrics()
            elif self._was_cancelled:
                self.warning("Benchmark was cancelled before results were collected")
            else:
                self.error("No profile results to export")
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=ErrorDetails(
                            type="NO_RESULTS",
                            message="No profile results to export",
                        ),
                        operation="profile",
                    )
                )
                self._print_exit_errors_and_log_file()
        else:
            self._print_exit_errors_and_log_file()

        self._print_process_memory_summary()

        if Environment.DEV.MODE:
            print_developer_mode_warning()

        # Signal benchmark completion to the operator via CR annotation.
        # This triggers the kopf handler immediately instead of waiting
        # for the next monitor poll cycle.
        # Only signal when the benchmark actually ran — if startup failed
        # (e.g. tokenizer resolution error), there are no results to fetch
        # and signaling completion would cause the operator to incorrectly
        # mark the job as Completed.
        has_results = self._profile_results and self._profile_results.results.records
        if keep_api_running and (has_results or self._was_cancelled):
            from aiperf.kubernetes.completion_signal import signal_benchmark_complete

            await signal_benchmark_complete()

        # Clean up global queues to prevent semaphore leaks. Bound each
        # cleanup with a hard timeout: multiprocessing.Queue.join_thread can
        # block indefinitely when the feeder thread cannot flush pending
        # items (e.g. pipe buffer contention under heavy xdist load).
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(cleanup_global_log_queue(), timeout=2.0)
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(cleanup_global_error_queue(), timeout=2.0)

        is_k8s_mode = self.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES
        keep_api_running = is_k8s_mode and self.run.cfg.runtime.api_port

        if keep_api_running:
            if has_results or self._was_cancelled:
                # Benchmark ran successfully — keep API alive for results fetch.
                self.info(
                    "Kubernetes mode: API service continues running to serve results"
                )
                await self.service_manager.wait_for_api_subprocess()
                self.info("API service has stopped, exiting")
                self._force_exit(0)
            else:
                # Benchmark failed during startup — no results to serve.
                # Exit with error code so the JobSet reports failure.
                self.info("Kubernetes mode: benchmark failed, not waiting for API")

        self._force_exit(1 if self._exit_errors else 0)

    async def _handle_control_message(
        self, identity: str, message: ControllerBoundMessage
    ) -> Struct | None:
        """Dispatch control channel messages from child services.

        Returns a Struct response for request-reply patterns (Registration, Command).
        Returns None for fire-and-forget messages (Heartbeat, StatusUpdate, etc.).
        """
        match message:
            case Registration():
                self.debug(
                    lambda: f"Received registration from {message.stype} service: {message.sid}"
                )
                already_configuring = message.sid in self._configuring_ids
                ServiceRegistry.register(
                    service_id=message.sid,
                    service_type=message.stype,
                    first_seen_ns=time.time_ns(),
                    state=LifecycleState(message.state),
                    pod_name=message.pod_name,
                    pod_index=message.pod_index,
                )
                if (
                    not already_configuring
                    and message.declared_worker_capacity is not None
                    and message.declared_record_processor_capacity is not None
                ):
                    self._declared_group_capacities[message.sid] = (
                        message.declared_worker_capacity,
                        message.declared_record_processor_capacity,
                    )
                    self.info(
                        f"Pod '{message.sid}' reports capacity: "
                        f"{message.declared_worker_capacity} workers, "
                        f"{message.declared_record_processor_capacity} record processors"
                    )
                    if self._k8s_topology is not None and (
                        message.declared_worker_capacity
                        != self._k8s_topology.workers_per_pod
                        or message.declared_record_processor_capacity
                        != self._k8s_topology.record_processors_per_pod
                    ):
                        self.warning(
                            f"Pod '{message.sid}' reported unexpected capacity "
                            f"({message.declared_worker_capacity} workers, "
                            f"{message.declared_record_processor_capacity} record processors). "
                            f"Expected {self._k8s_topology.workers_per_pod} workers and "
                            f"{self._k8s_topology.record_processors_per_pod} "
                            "record processors per pod."
                        )
                if self._auto_configure and not already_configuring:
                    self._configure_scheduler.execute_async(
                        self._configure_single_service(message.sid)
                    )
                return RegistrationAck(rid=message.rid)
            case Heartbeat():
                self.debug(
                    lambda msg=message: f"Received heartbeat from {msg.stype} service: {msg.sid}"
                )
                ServiceRegistry.update_service(
                    service_id=message.sid,
                    service_type=message.stype,
                    last_seen_ns=time.time_ns(),
                    state=LifecycleState(message.state),
                )
                return None
            case StatusUpdate():
                self.debug(
                    lambda msg=message: f"Received status from {msg.stype} service: {msg.sid}"
                )
                ServiceRegistry.update_service(
                    service_id=message.sid,
                    service_type=message.stype,
                    last_seen_ns=time.time_ns(),
                    state=LifecycleState(message.state),
                )
                return None
            case MemoryReport():
                self._memory_tracker.record(
                    label=message.sid,
                    group=message.stype,
                    pid=message.pid,
                    phase=MemoryPhase(message.phase),
                    reading=MemoryReading(
                        pss=message.pss_bytes,
                        rss=message.rss_bytes,
                        uss=message.uss_bytes,
                        shared=message.shared_bytes,
                    ),
                )
                return None
            case TelemetryStatus():
                self._telemetry_endpoints_configured = list(
                    message.endpoints_configured
                )
                self._telemetry_endpoints_reachable = list(message.endpoints_reachable)
                self._should_wait_for_telemetry = message.enabled
                if not message.enabled:
                    reason_msg = f" - {message.reason}" if message.reason else ""
                    self.info(f"GPU telemetry disabled{reason_msg}")
                    ServiceRegistry.forget(message.sid)
                else:
                    self.info(
                        f"GPU telemetry enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable"
                    )
                await self._check_and_trigger_shutdown()
                return None
            case ServerMetricsStatus():
                self._server_metrics_endpoints_configured = list(
                    message.endpoints_configured
                )
                self._server_metrics_endpoints_reachable = list(
                    message.endpoints_reachable
                )
                self._should_wait_for_server_metrics = message.enabled
                if not message.enabled:
                    reason_msg = f" - {message.reason}" if message.reason else ""
                    self.info(f"Server metrics disabled{reason_msg}")
                    ServiceRegistry.forget(message.sid)
                else:
                    self.info(
                        f"Server metrics enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable."
                    )
                    unreachable = set(message.endpoints_configured) - set(
                        message.endpoints_reachable
                    )
                    if unreachable:
                        self.warning(f"Unreachable endpoints: {', '.join(unreachable)}")
                await self._check_and_trigger_shutdown()
                return None
            case Command():
                return await self._dispatch_control_command(identity, message)
            case CommandAck() | CommandOk() | CommandErr():
                # Responses to pending requests are handled by _pending_requests
                # matching in the ROUTER receive loop. If we get here, it's
                # an unexpected response.
                self.debug(
                    f"Unexpected command response from {identity}: {type(message).__name__}"
                )
                return None

    # -------------------------------------------------------------------------
    # Control channel: command dispatch and sending helpers
    # -------------------------------------------------------------------------

    async def _dispatch_control_command(
        self, identity: str, message: Command
    ) -> Struct | None:
        """Dispatch an incoming Command from a service to local @on_command hooks.

        Returns a CommandAck/CommandOk/CommandErr response struct.
        """
        for hook in self.get_hooks(AIPerfHook.ON_COMMAND):
            resolved = hook.resolve_params(self)
            if isinstance(resolved, Iterable) and message.cmd in resolved:
                try:
                    result = await hook.func(message)
                    if result is None:
                        return CommandAck(cid=message.cid, sid=self.service_id)
                    from pydantic import BaseModel

                    if isinstance(result, BaseModel):
                        payload = result.model_dump_json().encode()
                    elif isinstance(result, bytes):
                        payload = result
                    elif isinstance(result, dict):
                        payload = orjson.dumps(result)
                    else:
                        payload = orjson.dumps(result)
                    return CommandOk(
                        cid=message.cid, sid=self.service_id, payload=payload
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    self.error(
                        f"Failed to handle command {message.cmd} from {identity}: {e}"
                    )
                    return CommandErr(
                        cid=message.cid,
                        sid=self.service_id,
                        error=str(e),
                        traceback=tb,
                    )

        self.debug(f"No handler for command {message.cmd} from {identity}")
        return CommandAck(cid=message.cid, sid=self.service_id)

    async def _send_control_command(
        self,
        identity: str,
        cmd: str,
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> CommandResponse:
        """Send a command to a specific service via ROUTER and wait for response."""
        command = Command(cid=uuid.uuid4().hex, cmd=cmd, payload=payload)
        return await self.control_router.request_to(identity, command, timeout)

    async def _send_control_command_to_all(
        self,
        cmd: str,
        service_ids: list[str],
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> list[CommandResponse | ErrorDetails]:
        """Send a command to all specified services and wait for all responses."""
        tasks = {
            sid: asyncio.create_task(
                self._send_control_command(sid, cmd, payload, timeout)
            )
            for sid in service_ids
        }
        results: list[CommandResponse | ErrorDetails] = []
        for sid, task in tasks.items():
            try:
                results.append(await task)
            except asyncio.TimeoutError:
                results.append(
                    ErrorDetails(
                        type="TimeoutError",
                        message=f"Command {cmd} timed out for {sid}",
                    )
                )
            except Exception as e:
                results.append(ErrorDetails.from_exception(e))
        return results

    async def _send_control_command_to_all_fail_fast(
        self,
        cmd: str,
        service_ids: list[str],
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> list[CommandResponse | ErrorDetails]:
        """Send command to all services, aborting on first error."""
        tasks = {
            sid: asyncio.create_task(
                self._send_control_command(sid, cmd, payload, timeout)
            )
            for sid in service_ids
        }
        results: list[CommandResponse | ErrorDetails] = []
        try:
            for coro in asyncio.as_completed(tasks.values()):
                try:
                    response = await coro
                    results.append(response)
                    if isinstance(response, CommandErr):
                        self.debug(
                            f"Received error from {response.sid}, aborting wait for "
                            f"remaining {len(service_ids) - len(results)} service(s)"
                        )
                        break
                except asyncio.TimeoutError:
                    results.append(
                        ErrorDetails(
                            type="TimeoutError", message=f"Command {cmd} timed out"
                        )
                    )
                    break
                except Exception as e:
                    results.append(ErrorDetails.from_exception(e))
                    break
        finally:
            for task in tasks.values():
                task.cancel()
        return results

    @staticmethod
    def _force_exit(code: int) -> None:
        """Flush stdio and exit. Falls back to os._exit if sys.exit hangs
        (e.g. ZMQ context blocking in atexit)."""
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    def _should_wait_for_raw_records(self) -> bool:
        """Check if we need to wait for raw record uploads from worker pods."""
        from aiperf.common.enums import ExportLevel

        return self.run.cfg.output.export_level == ExportLevel.RAW

    async def _wait_for_raw_record_uploads(self) -> None:
        """Wait for worker pods to upload raw record files to the API.

        Polls the raw_records subdirectory until we have at least one file
        per worker group manager, or the timeout expires.
        """
        raw_records_dir = (
            self.run.cfg.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
        )
        timeout = Environment.SERVICE.RAW_RECORD_UPLOAD_TIMEOUT
        poll_interval = 1.0
        deadline = time.monotonic() + timeout

        wgm_count = len(ServiceRegistry.get_services(ServiceType.WORKER_GROUP_MANAGER))
        if wgm_count == 0:
            self.debug("No worker group managers registered, skipping raw record wait")
            return

        self.info(f"Waiting for raw record uploads from {wgm_count} worker group(s)...")

        while time.monotonic() < deadline:
            if raw_records_dir.exists():
                files = list(raw_records_dir.glob("raw_records_*.jsonl"))
                if len(files) >= wgm_count:
                    self.info(
                        f"Received {len(files)} raw record file(s) from "
                        f"{wgm_count} group(s), proceeding with export"
                    )
                    return
                if files:
                    self.debug(
                        f"Have {len(files)}/{wgm_count} raw record file(s), "
                        "waiting for remaining pods..."
                    )
            await asyncio.sleep(poll_interval)

        # Check what we got before warning
        actual = 0
        if raw_records_dir.exists():
            actual = len(list(raw_records_dir.glob("raw_records_*.jsonl")))
        if actual > 0:
            self.warning(
                f"Timed out after {timeout}s: received {actual}/{wgm_count} "
                "raw record file(s). Proceeding with partial data."
            )
        else:
            self.warning(
                f"Timed out waiting for raw record uploads after {timeout}s. "
                "Raw records may be missing from export."
            )

    def _print_exit_errors_and_log_file(self) -> None:
        """Print post exit errors and log file info to the console."""
        console = Console()
        print_exit_errors(self._exit_errors, console=console)
        self._print_log_file_info(console)
        console.print()
        console.file.flush()

    async def _shutdown_record_processors_and_wait_for_flush(self) -> None:
        """Shut down WorkerGroupManager(s) and poll for flushed raw record files.

        In local multiprocessing mode, RPs are WGM subprocesses — outside the
        controller's service_manager — so stop_service(RECORD_PROCESSOR) on
        the controller is a no-op. Instead, send SHUTDOWN to each WGM over the
        control router; each WGM cascades shutdown to its child RPs, whose
        @on_stop hooks (BufferedJSONLWriterMixin._close_file) flush the
        raw_records_*.jsonl files before the aggregator reads them.
        """
        wgm_services = ServiceRegistry.get_services(ServiceType.WORKER_GROUP_MANAGER)
        if not wgm_services:
            return

        for svc in wgm_services:
            try:
                await self.control_router.send_to(
                    svc.service_id,
                    Command(cid=uuid.uuid4().hex, cmd=CommandType.SHUTDOWN),
                )
            except Exception as e:
                self.debug(f"Failed to send shutdown to {svc.service_id}: {e}")

        raw_records_dir = (
            self.run.cfg.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
        )
        expected = len(wgm_services)
        deadline = time.monotonic() + Environment.SERVICE.RAW_RECORD_UPLOAD_TIMEOUT
        stable_snapshots: list[tuple[int, int]] = []
        # Wait until at least `expected` files exist, each with size > 0, and
        # the (count, total_size) tuple is unchanged across two consecutive
        # samples ~300ms apart. Mere existence is insufficient: RPs flush in
        # batches, so the first write of a partial batch can lead the
        # aggregator to read a truncated file.
        while time.monotonic() < deadline:
            if raw_records_dir.exists():
                files = list(raw_records_dir.glob("raw_records_*.jsonl"))
                if len(files) >= expected and all(f.stat().st_size > 0 for f in files):
                    snapshot = (len(files), sum(f.stat().st_size for f in files))
                    stable_snapshots.append(snapshot)
                    if (
                        len(stable_snapshots) >= 2
                        and stable_snapshots[-1] == stable_snapshots[-2]
                    ):
                        self.debug(
                            f"Raw record files stable at {raw_records_dir} "
                            f"(files={snapshot[0]}, bytes={snapshot[1]})"
                        )
                        return
                else:
                    stable_snapshots.clear()
            await asyncio.sleep(0.3)

        self.warning(
            f"Timed out waiting for record processors to flush raw records to "
            f"{raw_records_dir}; export may be incomplete."
        )

    async def _export_results_data(self) -> None:
        """Write result files (CSV, JSON, Parquet) to the artifacts directory.

        Called from ``_check_and_trigger_shutdown`` BEFORE ``self.stop()`` so
        that files exist on disk before the API reports completion and before
        any external consumer (operator, CLI) fetches them.

        Sets ``_results_exported`` to True so ``@on_stop`` skips re-export.
        """
        # Stop record processors first so their buffered per-record writers
        # (raw_record_writer, record_export_csv, ...) flush to disk before
        # the aggregator/exporter reads those files. In real K8s mode, the
        # WGM/record-processor pods flush as part of their shutdown before
        # uploading; in local/fake mode, the aggregator otherwise races
        # against the RP's on_stop flush and sees an empty file.
        await self.service_manager.stop_service(ServiceType.RECORD_PROCESSOR)

        # In local multiprocessing mode the RPs are subprocesses of the WGM,
        # so stop_service(RECORD_PROCESSOR) on the controller's manager is a
        # no-op. Directly send SHUTDOWN to each registered RP over the
        # control router and poll the raw_records dir until files have been
        # flushed by RP @on_stop hooks, so the aggregator sees complete data.
        if (
            self.run.cfg.runtime.service_run_type == ServiceRunType.MULTIPROCESSING
            and self._should_wait_for_raw_records()
        ):
            await self._shutdown_record_processors_and_wait_for_flush()

        self._exporter_manager = ExporterManager(
            results=self._profile_results.results,
            config=self.run.cfg,
            telemetry_results=self._telemetry_results,
            server_metrics_results=self._server_metrics_results,
        )
        await self._exporter_manager.export_data()
        if self.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES:
            from aiperf.kubernetes.results_sidecar import write_ready_marker

            write_ready_marker(
                self.run.cfg.artifacts.artifact_directory,
                was_cancelled=self._was_cancelled,
            )
        self._results_exported = True
        self.info("Results exported to disk")

    async def _print_post_benchmark_info_and_metrics(self) -> None:
        """Print post benchmark info and metrics to the console."""
        console = Console()
        if console.width < 100:
            console.width = 100

        if not self._results_exported:
            # Non-K8s path or export didn't happen yet — do it now
            self._exporter_manager = ExporterManager(
                results=self._profile_results.results,
                config=self.run.cfg,
                telemetry_results=self._telemetry_results,
                server_metrics_results=self._server_metrics_results,
            )
            await self._exporter_manager.export_data()
            self._results_exported = True

        await self._exporter_manager.export_console(console=console)

        console.print()
        self._print_cli_command(console)
        self._print_benchmark_duration(console)
        self._print_exported_file_infos(self._exporter_manager, console)
        self._print_log_file_info(console)
        if self._was_cancelled:
            console.print(
                "[italic yellow]The profile run was cancelled early. Results shown may be incomplete or inaccurate.[/italic yellow]"
            )

        console.print()
        console.file.flush()

    def _print_log_file_info(self, console: Console) -> None:
        """Print the log file info."""
        log_file = (
            self.run.cfg.artifacts.dir
            / OutputDefaults.LOG_FOLDER
            / OutputDefaults.LOG_FILE
        )
        console.print(
            f"[bold green]Log File:[/bold green] [cyan]{log_file.resolve()}[/cyan]"
        )

    def _print_exported_file_infos(
        self, exporter_manager: ExporterManager, console: Console
    ) -> None:
        """Print the exported file infos."""
        file_infos = exporter_manager.get_exported_file_infos()
        for file_info in file_infos:
            console.print(
                f"[bold green]{file_info.export_type}[/bold green]: [cyan]{file_info.file_path.resolve()}[/cyan]"
            )

    def _print_cli_command(self, console: Console) -> None:
        """Print the CLI command that was used to run the benchmark."""
        cli_command = self.run.cfg.artifacts.cli_command or "N/A"
        console.print(
            f"[bold green]CLI Command:[/bold green] [italic]{cli_command}[/italic]"
        )

    def _print_benchmark_duration(self, console: Console) -> None:
        """Print the duration of the benchmark."""
        from aiperf.metrics.types.benchmark_duration_metric import (
            BenchmarkDurationMetric,
        )

        # Metrics are already in display units from summarize()
        duration = self._profile_results.get(BenchmarkDurationMetric.tag)
        if duration:
            duration_str = f"[bold green]{BenchmarkDurationMetric.header}[/bold green]: {duration.avg:.2f} {duration.unit}"
            if self._was_cancelled:
                duration_str += " [italic yellow](cancelled early)[/italic yellow]"
            console.print(duration_str)

    def _print_process_memory_summary(self) -> None:
        """Print memory summary for all AIPerf processes."""
        controller_pss_start = getattr(self, "_controller_pss_at_start", None)
        if controller_pss_start is not None:
            self._memory_tracker.record(
                label="SystemController",
                group="controller",
                pid=os.getpid(),
                phase=MemoryPhase.STARTUP,
                reading=MemoryReading(pss=controller_pss_start),
            )
        self._memory_tracker.capture(
            label="SystemController",
            group="controller",
            pid=os.getpid(),
            phase=MemoryPhase.SHUTDOWN,
        )

        self._memory_tracker.print_summary(title="AIPerf Process Memory")

    async def _kill(self) -> None:
        """Kill the system controller."""
        try:
            await self.service_manager.kill_all_services()
        except Exception as e:
            raise self._service_error("Failed to stop all services") from e

        await super()._kill()


def main() -> None:
    """Main entry point for the system controller."""

    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.SYSTEM_CONTROLLER)


if __name__ == "__main__":
    main()

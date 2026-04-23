# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import TYPE_CHECKING

import orjson

from aiperf.cli_utils import (
    print_developer_mode_warning,
    warn_osl_without_ignore_eos,
)
from aiperf.common.base_service import BaseService
from aiperf.common.control_structs import (
    Command,
    CommandErr,
    CommandOk,
    CommandResponse,
    ControllerBoundMessage,
)
from aiperf.config.zmq import ZMQDualBindConfig

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun

from aiperf.common.enums import (
    CommAddress,
    CommandType,
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
    on_command,
    on_init,
    on_message,
    on_start,
    on_stop,
)
from aiperf.common.logging import cleanup_global_log_queue, get_global_log_queue
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.memory_tracker import (
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
    ErrorDetails,
    ProcessRecordsResult,
)
from aiperf.common.models.error_models import ExitErrorInfo
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT
from aiperf.controller.protocols import ServiceManagerProtocol
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.controller.system_controller_commands import SystemControllerCommandMixin
from aiperf.controller.system_controller_dispatch import SystemControllerDispatchMixin
from aiperf.controller.system_controller_models import (
    AggregateWorkerStatus,
    K8sServiceTopology,
    build_aggregate_worker_status,
)
from aiperf.controller.system_controller_output import SystemControllerOutputMixin
from aiperf.controller.system_controller_raw_records import (
    SystemControllerRawRecordsMixin,
)
from aiperf.controller.system_mixins import SignalHandlerMixin
from aiperf.credit.messages import CreditsCompleteMessage
from aiperf.exporters.exporter_manager import ExporterManager
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType, ServiceType, UIType
from aiperf.ui.protocols import AIPerfUIProtocol
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient


class SystemController(
    SystemControllerDispatchMixin,
    SystemControllerOutputMixin,
    SystemControllerCommandMixin,
    SystemControllerRawRecordsMixin,
    SignalHandlerMixin,
    BaseService,
):
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

        is_k8s_mode = self.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES
        self._init_topology_and_required_services(is_k8s_mode)
        self._init_proxy_manager(is_k8s_mode)
        self._init_control_router()
        self._init_service_manager_and_ui()
        self._init_runtime_state()
        self.debug("System Controller created")

    def _init_topology_and_required_services(self, is_k8s_mode: bool) -> None:
        """Populate K8s topology and required-service counts."""
        self._was_cancelled = False
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

    def _init_proxy_manager(self, is_k8s_mode: bool) -> None:
        """Create the ZMQ proxy manager, respecting K8s sidecar mode."""
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

    def _init_control_router(self) -> None:
        """Create the control ROUTER client outside the comms lifecycle.

        The control ROUTER lives outside the comms lifecycle so it stays
        alive after comms.stop() — child processes still need it during
        their own shutdown sequence.
        """
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

    def _init_service_manager_and_ui(self) -> None:
        """Instantiate the service manager, error collector, and UI plugin."""
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

    def _init_runtime_state(self) -> None:
        """Reset per-run state holders (results, events, tracking sets)."""
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

        self._flush_pre_configuring_registrations()

        optional_services = self._collect_optional_services()

        total_services = sum(self.required_services.values()) + len(optional_services)
        self._log_startup_summary(total_services, optional_services)
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

        await self._configure_all_services_and_start_profiling()

    def _flush_pre_configuring_registrations(self) -> None:
        """Kick off PROFILE_CONFIGURE for services registered before auto-configure.

        e.g. k8s worker pods whose Registration arrived during
        initialize/control-router bind, before _start_services ran.
        Without this flush those services never receive PROFILE_CONFIGURE.
        """
        assert self._configure_scheduler is not None
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

    def _collect_optional_services(self) -> list[ServiceTypeT]:
        """Return the optional services to spawn alongside required services."""
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

        api_port = self.run.cfg.runtime.api_port or Environment.API_SERVER.PORT
        api_host = self.run.cfg.runtime.api_host or Environment.API_SERVER.HOST
        if api_port is not None and api_host is not None:
            self.info(f"Starting AIPerf API server at http://{api_host}:{api_port}/")
            optional_services.append(ServiceType.API)
        return optional_services

    def _log_startup_summary(
        self, total_services: int, optional_services: list[ServiceTypeT]
    ) -> None:
        """Emit the informational banner describing the service topology."""
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

    async def _configure_all_services_and_start_profiling(self) -> None:
        """Drive configure -> pod-health -> worker-ready -> profile-start sequence."""
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
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - service cmd dispatch boundary
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
            await self._wait_and_handle_configure_events(begin, timeout)
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

    async def _wait_and_handle_configure_events(
        self, begin: float, timeout: float
    ) -> None:
        """Wait for configure completion / failure / timeout and react."""
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
                done, _pending = await asyncio.wait(
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
                raise self._build_configure_timeout_error(
                    "Timed out waiting for services to configure "
                    f"({len(self._configured_ids)} configured). "
                )

            # Something woke us -- check what
            self._cancel_configure_tasks()

            # Cancellation (Ctrl+C)
            if self._was_cancelled:
                raise asyncio.CancelledError("Configuration interrupted by shutdown")

            # Service process died
            ServiceRegistry._raise_on_failure()

            # Configure task returned an error
            self._parse_control_responses_for_errors(
                self._configure_errors, "Configure Profiling"
            )

            # Verify all expected services are actually configured.
            if not self._all_expected_configured():
                raise self._build_configure_timeout_error(
                    "Configuration wait ended but not all services configured. "
                )

        finally:
            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task

    def _build_configure_timeout_error(
        self, prefix: str
    ) -> ServiceRegistrationTimeoutError:
        """Compose a ServiceRegistrationTimeoutError describing pending services."""
        pending_ids = ServiceRegistry.expected_ids - self._configured_ids
        pending_types = self._get_pending_type_counts()
        startup_summary = self._summarize_pending_worker_startup_states(pending_ids)
        startup_detail = (
            f", Pending worker startup: {startup_summary}" if startup_summary else ""
        )
        return ServiceRegistrationTimeoutError(
            f"{prefix}Pending IDs: {pending_ids}, Pending types: {pending_types}"
            f"{startup_detail}",
            missing={},
        )

    async def _check_sibling_containers_alive(self) -> None:
        """Fail registration if a sibling container in our own pod has died.

        Without this check, a crashed or OOMKilled sibling (e.g. the
        ``server-metrics-manager`` container hitting its memory limit
        before it can register) would leave ``_wait_and_handle_configure
        _events`` blocked for the full ``PROFILE_CONFIGURE_TIMEOUT`` (300
        s by default). By reading our own pod's container statuses and
        calling ``ServiceRegistry.fail_service`` for any terminated non-
        control-plane container, we fail fast with an actionable error.

        Quietly no-ops outside Kubernetes (required env vars not set) or
        if the API call fails — the configure-wait's own timeout is a
        safe fallback.

        Finding our own pod is slightly indirect: ``HOSTNAME`` is set to
        the JobSet's deterministic pod-hostname (e.g.
        ``aiperf-chaos-baseline-controller-0-0``) which is a PREFIX of
        the real pod name (Kubernetes appends a random suffix like
        ``-k52d5``). We list controller-labeled pods in our namespace and
        match by ``metadata.name.startswith(HOSTNAME)``.
        """
        import os

        pod_hostname = os.environ.get("HOSTNAME")
        namespace = os.environ.get("AIPERF_NAMESPACE")
        job_id = os.environ.get("AIPERF_JOB_ID")
        if not pod_hostname or not namespace or not job_id:
            return

        try:
            from kubernetes_asyncio import client

            from aiperf.kubernetes.client import k8s_client

            async with k8s_client() as api:
                pod_list = await client.CoreV1Api(api).list_namespaced_pod(
                    namespace=namespace,
                    label_selector=(
                        f"aiperf.nvidia.com/job-id={job_id},"
                        f"jobset.sigs.k8s.io/replicatedjob-name=controller"
                    ),
                )
        except Exception as e:  # noqa: BLE001 - sibling-check is best-effort; never raise into configure loop
            self.debug(
                lambda: f"Sibling-container check skipped (pod list failed): {e}"
            )
            return

        own_pod = next(
            (
                p
                for p in (pod_list.items or [])
                if (p.metadata.name if p.metadata else "").startswith(pod_hostname)
            ),
            None,
        )
        if own_pod is None or own_pod.status is None:
            return

        # Containers we never want to fail on: us (control-plane) and
        # infrastructure sidecars that aren't aiperf services.
        INFRA_CONTAINERS = {
            "control-plane",
            "event-bus-proxy",
            "results-sidecar",
            "worker-manager",  # legacy name for worker-group-manager
        }

        for cs in own_pod.status.container_statuses or []:
            container_name = cs.name or ""
            if container_name in INFRA_CONTAINERS:
                continue
            terminated = cs.state.terminated if cs.state and cs.state.terminated else None
            if not terminated:
                continue
            reason = terminated.reason or "Terminated"
            exit_code = terminated.exit_code
            if exit_code == 0 and reason not in ("OOMKilled",):
                continue  # Peaceful exit of an optional worker; not a failure

            # Convert container name ("dataset-manager") to the service_id the
            # registry uses ("dataset_manager").
            service_id = container_name.replace("-", "_")
            service_info = ServiceRegistry.services.get(service_id)
            service_type = (
                service_info.service_type
                if service_info
                else service_id  # type: ignore[assignment] — registry accepts string fallbacks
            )

            self.error(
                f"Sibling container '{container_name}' terminated "
                f"(reason={reason}, exitCode={exit_code}) before registration — "
                f"failing {service_id}"
            )
            ServiceRegistry.fail_service(service_id, service_type)

    async def _log_configure_progress(self, begin: float, timeout: float) -> None:
        """Log periodic progress during configuration wait."""
        interval = 5.0
        while True:
            await asyncio.sleep(interval)
            elapsed = time.perf_counter() - begin
            # Detect sibling containers that died before registering so we fail
            # fast instead of waiting out PROFILE_CONFIGURE_TIMEOUT.
            await self._check_sibling_containers_alive()
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
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - on_message handler boundary, must not crash bus
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
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - on_message handler boundary, must not crash bus
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
            await self._fire_and_forget_cancel_non_rm(all_ids, records_manager_ids)
            responses = await self._send_control_command_to_all(
                CommandType.PROFILE_CANCEL,
                list(records_manager_ids),
                timeout=Environment.SERVICE.PROFILE_CANCEL_TIMEOUT,
            )
            self._log_cancel_response_errors(responses)
            self._capture_cancel_records_result(responses)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - cancel path must always reach stop()
            self.warning(f"Exception during cancel command (proceeding to stop): {e!r}")

        if should_call_stop:
            self.debug("Stopping system controller after profiling cancelled")
            await asyncio.shield(self.stop())

    async def _fire_and_forget_cancel_non_rm(
        self, all_ids: list[str], records_manager_ids: set[str]
    ) -> None:
        """Send PROFILE_CANCEL to non-RecordsManager services without awaiting reply."""
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

    def _log_cancel_response_errors(
        self, responses: list[CommandResponse | ErrorDetails]
    ) -> None:
        """Warn on any cancel-command errors or timeouts."""
        for response in responses:
            if isinstance(response, ErrorDetails):
                self.warning(
                    f"Cancel command error (timeout or service unavailable): {response}"
                )
            elif isinstance(response, CommandErr):
                self.warning(
                    f"Cancel command failed from {response.sid}: {response.error}"
                )

    def _capture_cancel_records_result(
        self, responses: list[CommandResponse | ErrorDetails]
    ) -> None:
        """Extract ProcessRecordsResult from a RecordsManager CommandOk payload, if present."""
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
                    return
                except (orjson.JSONDecodeError, ValueError) as e:
                    self.warning(f"Failed to parse cancel response payload: {e}")

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

        await self._broadcast_shutdown_to_services(
            keep_api_running=bool(keep_api_running)
        )

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

        await self._teardown_ui_with_timeout()
        await asyncio.sleep(0.1)

        await self._emit_post_benchmark_report_or_errors()
        self._print_process_memory_summary()

        if Environment.DEV.MODE:
            print_developer_mode_warning()

        has_results = self._profile_results and self._profile_results.results.records
        await self._maybe_signal_k8s_completion(keep_api_running, has_results)

        # Clean up global queues to prevent semaphore leaks. Bound each
        # cleanup with a hard timeout: multiprocessing.Queue.join_thread can
        # block indefinitely when the feeder thread cannot flush pending
        # items (e.g. pipe buffer contention under heavy xdist load).
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(cleanup_global_log_queue(), timeout=2.0)
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(cleanup_global_error_queue(), timeout=2.0)

        await self._exit_after_optional_api_wait(is_k8s_mode, has_results)

    async def _broadcast_shutdown_to_services(self, *, keep_api_running: bool) -> None:
        """Fire-and-forget SHUTDOWN command to all registered services via ROUTER.

        When ``keep_api_running`` is True the API service is excluded so it can
        keep serving results after the rest of the system tears down.
        """
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
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - shutdown is best-effort per service
                self.debug(f"Failed to send shutdown to {sid}: {e}")

    async def _teardown_ui_with_timeout(self) -> None:
        """Stop the UI and drain its tasks, each bounded by a 5s timeout.

        The Dashboard UI can hang when the parent process runs under PIPE'd
        stdio (integration tests under xdist) because Textual's driver waits
        on a terminal that never arrives.
        """
        try:
            await asyncio.wait_for(self.ui.stop(), timeout=5.0)
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 - UI stop must not block controller teardown
            self.warning(f"UI stop did not complete cleanly: {e!r}")
        try:
            await asyncio.wait_for(self.ui.wait_for_tasks(), timeout=5.0)
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 - UI wait must not block controller teardown
            self.warning(f"UI task drain did not complete cleanly: {e!r}")

    async def _emit_post_benchmark_report_or_errors(self) -> None:
        """Print results/metrics, or the exit-error summary if startup failed."""
        if self._exit_errors:
            self._print_exit_errors_and_log_file()
            return
        if self._profile_results and self._profile_results.results.records:
            await self._print_post_benchmark_info_and_metrics()
            return
        if self._was_cancelled:
            self.warning("Benchmark was cancelled before results were collected")
            return
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

    async def _maybe_signal_k8s_completion(
        self, keep_api_running: object, has_results: object
    ) -> None:
        """Signal benchmark completion to the operator via CR annotation.

        Triggers the kopf handler immediately instead of waiting for the next
        monitor poll cycle. Only signal when the benchmark actually ran — if
        startup failed (e.g. tokenizer resolution error), there are no results
        to fetch and signaling completion would cause the operator to
        incorrectly mark the job as Completed.
        """
        if keep_api_running and (has_results or self._was_cancelled):
            from aiperf.kubernetes.completion_signal import signal_benchmark_complete

            await signal_benchmark_complete()

    async def _exit_after_optional_api_wait(
        self, is_k8s_mode: bool, has_results: object
    ) -> None:
        """Block on the K8s API subprocess if it must keep serving results, then exit."""
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

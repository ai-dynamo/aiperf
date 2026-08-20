# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import sys
import time
from typing import TYPE_CHECKING, cast

from rich.console import Console
from rich.panel import Panel

from aiperf.accuracy.models import AccuracySummary
from aiperf.cli_utils import (
    print_developer_mode_warning,
    warn_accuracy_temperature,
    warn_osl_without_ignore_eos,
)
from aiperf.common.base_service import BaseService
from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    ExportLevel,
    LifecycleState,
    MessageType,
    ServiceRegistrationStatus,
    SystemState,
    parse_result_producer_capability,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.hooks import on_command, on_init, on_message, on_start, on_stop
from aiperf.common.logging import cleanup_global_log_queue, get_global_log_queue
from aiperf.common.messages import (
    BaseServiceErrorMessage,
    BenchmarkCompleteMessage,
    CommandAcknowledgedResponse,
    CommandErrorResponse,
    CommandResponse,
    CommandSuccessResponse,
    CommandUnhandledResponse,
    FinalizeArtifactsCommand,
    GetPodStatesCommand,
    HeartbeatMessage,
    ProcessAccuracyResultMessage,
    ProcessAllResultsMessage,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
    ProcessTelemetryResultMessage,
    ProfileCancelCommand,
    ProfileConfigureCommand,
    ProfileStartCommand,
    RealtimeMetricsCommand,
    RegisterServiceCommand,
    ResultsExportedMessage,
    ServerMetricsStatusMessage,
    ShutdownCommand,
    ShutdownWorkersCommand,
    SpawnWorkersCommand,
    StatusMessage,
    SystemStateChangedMessage,
    TelemetryStatusMessage,
)
from aiperf.common.mixins import PodStateTrackerMixin
from aiperf.common.models import (
    ErrorDetails,
    ProcessRecordsResult,
    ServiceRunInfo,
)
from aiperf.common.models.error_models import ExitErrorInfo
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT
from aiperf.config.artifacts import OutputDefaults
from aiperf.controller.controller_utils import print_exit_errors
from aiperf.controller.protocols import (
    KubernetesServiceManagerProtocol,
    LocalProcessServiceManagerProtocol,
    ServiceManagerProtocol,
)
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.controller.result_join_coordinator import ResultJoinCoordinator
from aiperf.controller.system_controller_models import (
    AggregateWorkerStatus,
    K8sServiceTopology,
    PodStateSnapshot,
    build_aggregate_worker_status,
)
from aiperf.controller.system_mixins import SignalHandlerMixin
from aiperf.credit.messages import CreditsCompleteMessage
from aiperf.exporters.exporter_manager import ExporterFailure, ExporterManager
from aiperf.common.results_markers import (
    write_processing_marker,
    write_ready_marker,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType, ServiceType, UIType
from aiperf.ui.protocols import AIPerfUIProtocol

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


_PRE_BENCHMARK_STATES = frozenset(
    {SystemState.INITIALIZING, SystemState.CONFIGURING, SystemState.READY}
)
"""States in which no benchmark result can legitimately have been produced yet."""


class SystemController(PodStateTrackerMixin, SignalHandlerMixin, BaseService):
    """System Controller service.

    This service is responsible for managing the lifecycle of all other services.
    It will start, stop, and configure all other services.
    """

    def __init__(
        self,
        run: "BenchmarkRun",
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

        if self._should_warn_accuracy_temperature():
            warn_accuracy_temperature()

        self._was_cancelled = False
        # List of required service types, in no particular order
        # These are services that must be running before the system controller can start profiling
        self.required_services: dict[ServiceTypeT, int] = {
            ServiceType.DATASET_MANAGER: 1,
            ServiceType.TIMING_MANAGER: 1,
            ServiceType.WORKER_MANAGER: 1,
            ServiceType.RECORDS_MANAGER: 1,
        }
        if self.run.cfg.record_processor_service_count is not None:
            self.required_services[ServiceType.RECORD_PROCESSOR] = (
                self.run.cfg.record_processor_service_count
            )
            self.scale_record_processors_with_workers = False
        else:
            self.scale_record_processors_with_workers = True

        # In Kubernetes mode, workers are external pods that connect via TCP.
        # We must wait for at least one worker to register before starting profiling.
        # In Multi-Process mode, workers are spawned locally and register automatically.
        # KUBERNETES is registered in plugins.yaml only when the operator/k8s
        # service-manager is present; in this build it is intentionally
        # absent, so probe via getattr rather than referencing the enum
        # member directly.
        if self._is_kubernetes():
            self.required_services[ServiceType.WORKER] = 1
            # Nothing runs a WorkerManager container in Kubernetes: worker pods
            # run one WorkerGroupManager each as their pod-infrastructure
            # process, so requiring WORKER_MANAGER here guarantees a
            # registration timeout and a dead control plane.
            del self.required_services[ServiceType.WORKER_MANAGER]
            # One WorkerGroupManager per worker pod. Hardcoding 1 here makes
            # expected_pods == ready_pods == 1, so profiling starts against
            # whichever pod registers first.
            self._k8s_topology = self._build_k8s_service_topology()
            self.required_services[ServiceType.WORKER_GROUP_MANAGER] = (
                self._k8s_topology.num_worker_pods
            )

        self.proxy_manager: ProxyManager = ProxyManager(
            run=self.run, enable_event_bus=not self._event_bus_proxy_is_external()
        )
        service_run_type = self.run.cfg.runtime.service_run_type
        ServiceManagerClass = plugins.get_class(
            PluginType.SERVICE_MANAGER, service_run_type
        )

        using_dashboard = self.run.cfg.ui_type == UIType.DASHBOARD
        log_queue = get_global_log_queue() if using_dashboard else None

        self.service_manager: ServiceManagerProtocol = ServiceManagerClass(
            required_services=self.required_services,
            run=self.run,
            log_queue=log_queue,
        )
        UIClass = plugins.get_class(PluginType.UI, self.run.cfg.ui_type)
        self.ui: AIPerfUIProtocol = UIClass(
            run=self.run,
            log_queue=log_queue,
            controller=self,
        )
        self.attach_child_lifecycle(self.ui)
        self._stop_tasks: set[asyncio.Task] = set()
        self._profile_results: ProcessRecordsResult | None = None
        self._exit_errors: list[ExitErrorInfo] = []
        self._export_failed = False
        self._telemetry_results: TelemetryExportData | None = None
        self._server_metrics_results: ServerMetricsResults | None = None
        self._accuracy_results: AccuracySummary | None = None
        self._accuracy_results_injected = False
        # Shared shutdown barrier: every result-producing service that advertises a
        # ``result_producer:<domain>`` capability at registration joins here, and
        # each domain must complete before shutdown. A producer that dies (service
        # error) or reports itself disabled (telemetry/server-metrics status)
        # unregisters, so it stops blocking. Replaces the per-gate boolean flags.
        self._result_join_coordinator = ResultJoinCoordinator()
        # Services the watchdog confirmed dead. Kept alongside the coordinator
        # because command fan-out targets must exclude them even after the
        # barrier has released.
        self._reaped_service_ids: set[str] = set()
        # A producer that dies abruptly never sends SERVICE_ERROR, so the
        # heartbeat/pod reapers are the only signal. Without this hook they mark
        # the service failed in the registry but leave it in the barrier, and
        # the controller waits for it forever.
        self.service_manager.on_service_reaped = self._on_service_reaped
        # Set when the accuracy result message lands (even with results=None). The
        # cancel path awaits this to give a graded summary a bounded chance to
        # arrive; the normal path relies on the barrier alone.
        self._accuracy_result_arrived = asyncio.Event()
        self._server_metrics_result_arrived = asyncio.Event()

        self._shutdown_triggered = False
        self._pod_failure_watcher_task: asyncio.Task | None = None
        self._pod_failure_watch_disarmed = False
        self._system_state: SystemState = SystemState.INITIALIZING
        self._replacement_configuring_ids: set[str] = set()
        self._shutdown_lock = asyncio.Lock()
        self._api_enabled = False
        self._telemetry_endpoints_configured: list[str] = []
        self._telemetry_endpoints_reachable: list[str] = []
        self._server_metrics_endpoints_configured: list[str] = []
        self._server_metrics_endpoints_reachable: list[str] = []
        self.debug("System Controller created")

    def get_aggregate_worker_status(self) -> AggregateWorkerStatus:
        """Return the controller-authored aggregate worker-pod status snapshot.

        ``/api/progress`` resolves ``progress.workers`` by calling this on the
        in-process controller handle; without it the endpoint falls back to an
        empty snapshot and every worker field the operator mirrors into
        ``AIPerfJob.status.workers`` reads zero even on a healthy run.
        """
        return build_aggregate_worker_status(self._pod_state_tracker.pod_states)

    def get_pod_state_snapshot(self) -> PodStateSnapshot:
        """Return one authoritative copy of controller-owned worker state."""
        return PodStateSnapshot(
            pod_states=dict(self._pod_state_tracker.pod_states),
            worker_startup_states=dict(self._pod_state_tracker.worker_startup_states),
        )

    def _ready_worker_pod_count(self) -> int:
        """Count worker pods that can actually be routed a credit.

        Keyed on ``dispatchable_workers`` alone. ``ready_record_processors``
        is deliberately NOT required here, unlike ``build_aggregate_worker_status``:
        record-processor peer registration is asynchronous and may lag worker
        readiness, and a pod whose workers can be routed a credit is enough to
        start profiling. Requiring it would stall the start gate behind
        bookkeeping that has no bearing on dispatchability.
        """
        return sum(
            1
            for pod in self._pod_state_tracker.pod_states.values()
            if pod.dispatchable_workers >= 1
        )

    def _has_sufficient_ready_worker_pods(self) -> bool:
        """Check whether enough worker pods are dispatchable to start profiling."""
        if not self._is_kubernetes():
            return True
        return self._ready_worker_pod_count() >= 1

    def _is_kubernetes(self) -> bool:
        """Whether this controller runs under the Kubernetes operator."""
        kubernetes_run_type = getattr(ServiceRunType, "KUBERNETES", None)
        return (
            kubernetes_run_type is not None
            and self.run.cfg.runtime.service_run_type == kubernetes_run_type
        )

    def _event_bus_proxy_is_external(self) -> bool:
        """Report whether something outside this process already binds the event bus.

        In Kubernetes mode the JobSet may run the XPUB/XSUB proxy as a dedicated
        ``aiperf proxy --kind event_bus`` sidecar container in the controller pod.
        Both processes bind the same two addresses, so the controller must not
        start its own copy or the second bind fails with ``Address already in
        use`` and the whole control plane dies on init.

        The operator stamps ``AIPERF_K8S_EVENT_BUS_SIDECAR_ENABLED`` into the
        control-plane container (see
        ``aiperf.kubernetes.jobset_builder.AIPerfJobSetBuilder._create_control_plane_containers``).
        Any non-Kubernetes run returns False, so the multiprocessing path keeps
        hosting all three proxies exactly as before.
        """
        if not self._is_kubernetes():
            return False
        from aiperf.kubernetes.environment import K8sEnvironment

        return K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED

    def _build_k8s_service_topology(self) -> K8sServiceTopology:
        """Derive the worker-pod topology from runtime config.

        Kubernetes deployments are pod-based: each worker pod runs a fixed
        number of worker and record-processor containers, and the last pod is
        not partially filled. Startup must therefore wait for the full expanded
        topology rather than the requested logical worker count -- expecting a
        single WorkerGroupManager lets profiling begin as soon as the first pod
        reports in, running against a fraction of the requested load with no
        error raised anywhere.

        The fan-out itself exists because a node holds only ~65k ephemeral
        ports, capping concurrent connections per node; see
        ``RuntimeConfig.workers_per_pod`` for the sizing ratios that shaped the
        defaults consumed here.
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

        return K8sServiceTopology(
            num_worker_pods=num_worker_pods,
            workers_per_pod=workers_per_pod,
            record_processors_per_pod=record_processors_per_pod,
            total_workers=total_workers,
            total_record_processors=num_worker_pods * record_processors_per_pod,
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
        extra_dict = dict(extra_inputs)
        return not (extra_dict.get("ignore_eos") or extra_dict.get("min_tokens"))

    def _should_warn_accuracy_temperature(self) -> bool:
        """Check if accuracy mode is active without temperature=0 in extra inputs."""
        accuracy = self.run.cfg.accuracy
        if accuracy is None or not accuracy.enabled:
            return False
        extra_inputs = self.run.cfg.endpoint.extra
        if not extra_inputs:
            return True
        val = dict(extra_inputs).get("temperature")
        try:
            return float(val) != 0.0
        except (TypeError, ValueError):
            return True

    async def request_realtime_metrics(self) -> None:
        """Request real-time metrics from the RecordsManager."""
        await self.send_command_and_wait_for_response(
            RealtimeMetricsCommand(
                service_id=self.service_id,
                target_service_type=ServiceType.RECORDS_MANAGER,
            )
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

        await self._begin_results_export_transaction()

        self.setup_signal_handlers(self._handle_signal)
        self.debug("Setup signal handlers")

        async with self.try_operation_or_stop("Initialize Service Manager"):
            await self.service_manager.initialize()

        self.debug("System Controller initialized successfully")

    @on_start
    async def _start_services(self) -> None:
        """Bootstrap the system services.

        This method will:
        - Initialize all required services
        - Wait for all required services to be registered
        - Start all required services
        """
        self.debug("System Controller is bootstrapping services")

        # Start all required services
        async with self.try_operation_or_stop("Start Service Manager"):
            await self.service_manager.start()

        # Start optional services before waiting for registration so they can participate in configuration
        if self.run.cfg.gpu_telemetry.enabled:
            await self.service_manager.run_service(ServiceType.GPU_TELEMETRY_MANAGER)
        else:
            self.info("GPU telemetry disabled via --no-gpu-telemetry")

        if self.run.cfg.server_metrics.enabled:
            self.debug("Starting optional ServerMetricsManager service")
            await self.service_manager.run_service(ServiceType.SERVER_METRICS_MANAGER)
        else:
            self.info("Server metrics disabled via --no-server-metrics")

        if self.run.cfg.network_latency.should_probe:
            self.debug("Starting optional NetworkLatencyManager service")
            await self.service_manager.run_service(ServiceType.NETWORK_LATENCY_MANAGER)

        # Start AIPerf API if enabled
        api_port = self.run.cfg.runtime.api_port or Environment.API_SERVER.PORT
        api_host = self.run.cfg.runtime.api_host or Environment.API_SERVER.HOST
        if api_port is not None and api_host is not None:
            self.info(f"Starting AIPerf API server at http://{api_host}:{api_port}/")
            await self.service_manager.run_service(ServiceType.API)
            self._api_enabled = True

        async with self.try_operation_or_stop("Register Services"):
            await self.service_manager.wait_for_all_services_registration(
                stop_event=self._stop_requested_event,
            )

        # Services are silent while starting up, so staleness only becomes
        # meaningful once every one of them has registered.
        self.service_manager.activate_heartbeat_monitoring()

        await self._set_system_state(SystemState.CONFIGURING)
        self.info("AIPerf System is CONFIGURING")
        await self._profile_configure_all_services()
        await self._set_system_state(SystemState.READY)
        self.info("AIPerf System is CONFIGURED")
        await self._verify_pods_healthy()
        await self._wait_for_dispatchable_worker_pods()
        if isinstance(self.service_manager, KubernetesServiceManagerProtocol):
            self._pod_failure_watcher_task = self.execute_async(
                self._watch_pod_failure_abort()
            )
        await self._start_profiling_all_services()
        await self._set_system_state(SystemState.PROFILING)
        self.info("AIPerf System is PROFILING")
        # A very short run can publish its terminal result while PROFILE_START
        # acknowledgements are still being collected. Re-check after leaving
        # the startup states so an earlier, deliberately ignored readiness
        # notification cannot strand the controller.
        await self._check_and_trigger_shutdown()

    async def _verify_pods_healthy(self) -> None:
        """Gate PROFILE_START on worker-pod health.

        A pod can register all its services and then die (OOMKilled, evicted)
        before profiling begins. Without this gate the run proceeds with fewer
        workers than it believes it has and reports results that silently omit
        that pod's share of the load. Managers without the Kubernetes
        capability are skipped.
        """
        if not isinstance(self.service_manager, KubernetesServiceManagerProtocol):
            return
        async with self.try_operation_or_stop("Pod Health Check"):
            await self.service_manager.check_pods_healthy()

    async def _wait_for_dispatchable_worker_pods(self) -> None:
        """Gate PROFILE_START on worker pods being able to *serve* credits.

        Distinct from ``_verify_pods_healthy``, which only asks whether a pod is
        alive. A pod can be perfectly healthy and still have no dataset: the
        dataset arrives over one-shot broadcasts, and a pod whose containers
        subscribe late receives neither. Its workers hold themselves out of the
        routing pool (see ``Worker._send_worker_ready_message``), so starting
        here would run the benchmark at a fraction of the requested load.

        Waits for every registered worker pod to become dispatchable, then falls
        back to "at least one" after a short grace period. Proceeding with a
        subset is safe because an undispatchable worker is not in the routing
        pool at all, and it rejoins mid-run once its dataset poll succeeds --
        but it changes the effective load, so say so loudly.

        A no-op outside Kubernetes mode.
        """
        if not self._is_kubernetes():
            return
        timeout = Environment.DATASET.CONFIGURATION_TIMEOUT
        grace_period = min(5.0, timeout)
        poll_interval = Environment.WORKER.STATUS_SUMMARY_INTERVAL
        begin = time.perf_counter()
        while True:
            expected_pods = len(
                self.service_manager.service_map.get(
                    ServiceType.WORKER_GROUP_MANAGER, []
                )
            )
            ready_pods = self._ready_worker_pod_count()
            if expected_pods and ready_pods >= expected_pods:
                self.info(f"All {ready_pods} worker pod(s) are dispatchable")
                return

            elapsed = time.perf_counter() - begin
            if elapsed >= grace_period and ready_pods >= 1:
                self.warning(
                    f"Starting profiling with {ready_pods} of {expected_pods} worker "
                    f"pod(s) dispatchable after {elapsed:.1f}s. The remaining pod(s) "
                    "have no dataset yet; they receive no credits until they do, so "
                    "the load starts below the requested level."
                )
                return
            if elapsed >= timeout:
                raise LifecycleOperationError(
                    f"No worker pod became dispatchable within {timeout:.0f}s "
                    f"({ready_pods} of {expected_pods} ready). Worker pods are "
                    "running but have no dataset, so every credit would fail."
                )
            await asyncio.sleep(poll_interval)

    async def _watch_pod_failure_abort(self) -> None:
        """Cancel the benchmark when failed worker pods breach the threshold.

        The Kubernetes service manager's monitoring loop sets the event; the
        controller reacts through the same ``_cancel_profiling`` path a Ctrl+C
        takes, so partial results are still exported.

        Only armed for the load phase. Worker pods exit legitimately once
        credits are complete, so a breach observed after that is normal
        teardown, not a failure -- see ``_disarm_pod_failure_watcher``.
        """
        if not isinstance(self.service_manager, KubernetesServiceManagerProtocol):
            return
        await self.service_manager.pod_failure_abort_event.wait()
        if (
            self._pod_failure_watch_disarmed
            or self._was_cancelled
            or self._shutdown_triggered
        ):
            return
        self.error(
            f"Aborting benchmark: {self.service_manager.pod_failure_abort_reason}"
        )
        await self._cancel_profiling()

    def _disarm_pod_failure_watcher(self) -> None:
        """Stop treating worker-pod exits as benchmark failures.

        Called on entry to every teardown path (credits complete, the shutdown
        readiness check, and cancellation). The flag is set in addition to
        cancelling the task because ``pod_failure_abort_event`` may already be
        set with the waiter scheduled to resume: cancellation alone loses that
        race, and the run reports as cancelled even though it succeeded.
        """
        self._pod_failure_watch_disarmed = True
        task = self._pod_failure_watcher_task
        if task is None or task.done():
            return
        # The abort path runs *inside* this task: _watch_pod_failure_abort ->
        # _cancel_profiling -> here. Cancelling ourselves would deliver
        # CancelledError at the next await and silently skip the rest of the
        # teardown, so the benchmark would keep running after logging that it
        # was aborting. The disarm flag above is what stops the waiter in that
        # case; only a caller on another task needs the cancel.
        if task is asyncio.current_task():
            return
        task.cancel()

    async def _set_system_state(self, state: SystemState) -> None:
        """Advance the controller's outer-lifecycle ``SystemState`` and notify
        subscribers via the message bus.

        Idempotent: a no-op (no log line, no publish) when ``state`` already
        matches the current state, so callers can re-stamp without flooding the
        bus. The ProgressRouter mirrors the published state onto
        ``/api/progress`` so the operator can stamp
        ``AIPerfJob.status.subPhase``.

        Example:
            await self._set_system_state(SystemState.PROFILING)
        """
        if state == self._system_state:
            return
        # Forward-only. _cancel_profiling sets STOPPING and then blocks on
        # ProfileCancelCommand; during that window the cancelled
        # PhaseOrchestrator's finally: publishes CreditsCompleteMessage, which
        # would stamp PROCESSING and walk status.subPhase
        # stopping -> processing -> shutdown, breaking every consumer that
        # assumes the sequence only advances.
        if state.rank < self._system_state.rank:
            self.debug(
                lambda: f"Ignoring backwards system state {self._system_state} -> {state}"
            )
            return
        self.info(f"System state: {self._system_state} -> {state}")
        self._system_state = state
        await self.publish(
            SystemStateChangedMessage(service_id=self.service_id, state=state)
        )

    async def _profile_configure_all_services(self) -> None:
        """Configure all services to start profiling.

        This is a blocking call that will wait for all services to be configured
        before returning. Uses fail-fast behavior: if any service returns an error,
        we abort immediately without waiting for the remaining services.
        """
        self.info("Configuring all services to start profiling")
        begin = time.perf_counter()
        responses = await self.send_command_and_wait_until_first_error(
            ProfileConfigureCommand(
                service_id=self.service_id,
            ),
            list(self.service_manager.service_id_map.keys()),
            timeout=Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT,
        )
        duration = time.perf_counter() - begin
        self._parse_responses_for_errors(responses, "Configure Profiling")
        self.info(f"All services configured in {duration:.2f} seconds")

        if not Environment.HTTP.SSL_VERIFY:
            self.warning(
                "SSL certificate verification is DISABLED - this is insecure. This should only be used for testing in a trusted environment."
            )

    async def _start_profiling_all_services(self) -> None:
        """Tell all services to start profiling.

        Uses fail-fast behavior: if any service returns an error,
        we abort immediately without waiting for the remaining services.
        """
        self.debug("Sending PROFILE_START command to all services")
        responses = await self.send_command_and_wait_until_first_error(
            ProfileStartCommand(
                service_id=self.service_id,
            ),
            list(self.service_manager.service_id_map.keys()),
            timeout=Environment.SERVICE.PROFILE_START_TIMEOUT,
        )
        self._parse_responses_for_errors(responses, "Start Profiling")
        self.info("All services started profiling successfully")

    def _parse_responses_for_errors(
        self, responses: list[CommandResponse | ErrorDetails], operation: str
    ) -> None:
        """Parse the responses for errors."""
        for response in responses:
            if isinstance(response, ErrorDetails):
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=response, operation=operation, service_id=None
                    )
                )
            elif isinstance(response, CommandErrorResponse):
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=response.error,
                        operation=operation,
                        service_id=response.service_id,
                    )
                )
        if self._exit_errors:
            raise LifecycleOperationError(
                operation=operation,
                original_exception=None,
                lifecycle_id=self.id,
            )

    @on_command(CommandType.REGISTER_SERVICE)
    async def _handle_register_service_command(
        self, message: RegisterServiceCommand
    ) -> None:
        """Process a registration message from a service.

        Adds the service to the service manager's tracking maps (service_id_map and
        service_map) so it can participate in lifecycle coordination.

        Args:
            message: The registration message to process
        """

        self.debug(
            lambda: (
                f"Processing registration from {message.service_type} with ID: {message.service_id}"
            )
        )

        prior_info = ServiceRegistry.get_service(message.service_id)
        was_registered = ServiceRegistry.is_registered(message.service_id)
        is_replacement = self._is_replacement_worker_group_registration(
            message, prior_info, was_registered
        )
        now_ns = time.time_ns()
        ServiceRegistry.register(
            service_id=message.service_id,
            service_type=message.service_type,
            first_seen_ns=now_ns,
            state=message.state,
            pod_name=message.pod_name,
            pod_index=message.pod_index,
        )
        service_info = ServiceRegistry.get_service(message.service_id)
        if service_info is None:
            raise RuntimeError(
                f"Service registry lost registration for '{message.service_id}'"
            )

        # A replacement pod reusing a deterministic service ID is alive again,
        # so it must stop being excluded from command fan-out.
        self._reaped_service_ids.discard(message.service_id)

        previous = self.service_manager.service_id_map.get(message.service_id)
        if previous is not None and previous.service_type != message.service_type:
            self.service_manager.service_map[previous.service_type] = [
                info
                for info in self.service_manager.service_map.get(
                    previous.service_type, []
                )
                if info.service_id != message.service_id
            ]
        self.service_manager.service_id_map[message.service_id] = service_info
        services = self.service_manager.service_map.setdefault(message.service_type, [])
        for index, existing in enumerate(services):
            if existing.service_id == message.service_id:
                services[index] = service_info
                break
        else:
            services.append(service_info)

        # Join every result domain this service advertises into the shutdown
        # barrier. A telemetry/server-metrics producer may later report it is
        # disabled via its status message, which unregisters the domain again.
        for capability in message.capabilities:
            domain = parse_result_producer_capability(capability)
            if domain is not None:
                self._result_join_coordinator.register(domain, message.service_id)

        try:
            type_name = ServiceType(message.service_type).name.title().replace("_", " ")
        except (TypeError, ValueError):
            type_name = message.service_type
        self.info(lambda: f"Registered {type_name} (id: '{message.service_id}')")

        if (
            is_replacement
            and self._system_state != SystemState.INITIALIZING
            and message.service_id not in self._replacement_configuring_ids
        ):
            self._replacement_configuring_ids.add(message.service_id)
            self.execute_async(
                self._configure_replacement_worker_group(message.service_id)
            )

    def _is_replacement_worker_group_registration(
        self,
        message: RegisterServiceCommand,
        prior_info: ServiceRunInfo | None,
        was_registered: bool,
    ) -> bool:
        """Return whether a JobSet replacement reused a worker-group ID."""
        if (
            message.service_type != ServiceType.WORKER_GROUP_MANAGER
            or prior_info is None
        ):
            return False
        if prior_info.state == LifecycleState.FAILED:
            return True
        return (
            was_registered
            and prior_info.pod_name is not None
            and message.pod_name is not None
            and prior_info.pod_name != message.pod_name
        )

    async def _configure_replacement_worker_group(self, service_id: str) -> None:
        """Configure a replacement pod after its registration ACK can be sent."""
        try:
            response = await self.send_command_and_wait_for_response(
                ProfileConfigureCommand(
                    service_id=self.service_id,
                    target_service_id=service_id,
                ),
                timeout=Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT,
            )
            if isinstance(response, ErrorDetails):
                self.error(
                    f"Replacement worker pod '{service_id}' did not configure: "
                    f"{response.message}"
                )
                ServiceRegistry.fail_service(
                    service_id, ServiceType.WORKER_GROUP_MANAGER, fatal=False
                )
                return
            if isinstance(response, CommandErrorResponse):
                self.error(
                    f"Replacement worker pod '{service_id}' rejected configure: "
                    f"{response.error.message}"
                )
                ServiceRegistry.fail_service(
                    service_id, ServiceType.WORKER_GROUP_MANAGER, fatal=False
                )
                return
            self.info(f"Configured replacement worker pod '{service_id}'")
        finally:
            self._replacement_configuring_ids.discard(service_id)

    @on_message(MessageType.HEARTBEAT)
    async def _process_heartbeat_message(self, message: HeartbeatMessage) -> None:
        """Process a heartbeat message from a service. It will
        update the last seen timestamp and state of the service.

        Args:
            message: The heartbeat message to process
        """
        service_id = message.service_id
        service_type = message.service_type
        timestamp = message.request_ns or time.time_ns()

        # Update the last heartbeat timestamp if the component exists
        try:
            service_info = self.service_manager.service_id_map[service_id]
            service_info.last_seen_ns = timestamp
            service_info.state = message.state
            ServiceRegistry.update_service(
                service_id, service_type, timestamp, message.state
            )
            self.debug(lambda: f"Updated heartbeat for '{service_id}' to {timestamp}")
        except Exception:
            self.warning(
                f"Received heartbeat from unknown service: '{service_id}' ('{service_type}')"
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
        # Credits exhausted means request dispatch is done and the remaining
        # work is record aggregation, which is what PROCESSING denotes.
        await self._set_system_state(SystemState.PROCESSING)
        self._disarm_pod_failure_watcher()

    @on_message(MessageType.SERVICE_ERROR)
    async def _process_service_error_message(
        self, message: BaseServiceErrorMessage
    ) -> None:
        """Record a service-reported failure so the run exits non-zero.

        Sources include ``BaseService._kill`` (FAILED-state self-kill) and
        TimingManager's phase-orchestrator done-callback. Without this
        handler the failure logs but ``_exit_errors`` stays empty, so
        ``os._exit(0)`` masks the failure — particularly visible when
        FixedScheduleStrategy rejects a dataset whose first-turn timestamp
        was filtered out by the offset window.
        """
        self.error(
            f"Received service error from '{message.service_id}': "
            f"{message.error.message}"
        )
        self._exit_errors.append(
            ExitErrorInfo(
                error_details=message.error,
                operation="service_runtime",
                service_id=message.service_id,
            )
        )
        # A dead producer can no longer join its result domain; drop it from the
        # barrier so shutdown isn't blocked forever, then re-check readiness.
        self._result_join_coordinator.unregister_service(message.service_id)
        if self._system_state not in {SystemState.STOPPING, SystemState.SHUTDOWN}:
            await self._cancel_profiling()
            return
        await self._check_and_trigger_shutdown()

    async def _on_service_reaped(self, service_id: str, reason: str) -> None:
        """Evict a reaped producer from the result-join barrier, degrading the run.

        Invoked by the service manager's watchdog after a service has been
        confirmed dead. The registry's fail_service only flips state and wakes
        startup-time waiters, so without this the barrier keeps waiting on a
        process that will never report -- the controller hangs after profiling.

        Eviction is safe here because the watchdog already applies a two-strike
        confirmation before declaring death, so a briefly-slow service is not
        dropped. A result that still arrives afterwards is accepted by the
        coordinator rather than discarded.

        Releasing the barrier is not the same as succeeding. The surviving
        producers' records are still exported, but a run missing one of eight
        worker pods reports throughput computed from seven, so the eviction is
        recorded in ``_exit_errors`` -- the sole input to the exit code. A CI
        job or sweep reading that exit code must not be told the benchmark
        succeeded.
        """
        if not self._result_join_coordinator.evict_service(service_id, reason):
            return
        self.warning(
            f"Evicted '{service_id}' from the result-join barrier ({reason}); "
            "results for this producer will be missing from the run"
        )
        self._reaped_service_ids.add(service_id)
        self._exit_errors.append(
            ExitErrorInfo(
                error_details=ErrorDetails(
                    message=(
                        f"Result producer '{service_id}' died before reporting "
                        f"({reason}). The exported results are missing this "
                        "producer's share of the run and are not comparable to "
                        "a complete benchmark."
                    ),
                    type="ProducerReaped",
                ),
                operation="result_producer_reaped",
                service_id=service_id,
            )
        )
        self._forget_reaped_service(service_id)
        await self._check_and_trigger_shutdown()

    def _forget_reaped_service(self, service_id: str) -> None:
        """Drop a confirmed-dead service from the command-target maps.

        Eviction from the join barrier alone is not enough: artifact
        finalization builds its target list from ``service_id_map``, so a
        reaped service would still be commanded and the barrier would sit on
        the full command timeout before failing on a peer that is already known
        to be gone.
        """
        info = self.service_manager.service_id_map.pop(service_id, None)
        if info is None:
            return
        peers = self.service_manager.service_map.get(info.service_type)
        if peers is None:
            return
        self.service_manager.service_map[info.service_type] = [
            peer for peer in peers if peer.service_id != service_id
        ]

    @on_message(MessageType.STATUS)
    async def _process_status_message(self, message: StatusMessage) -> None:
        """Process a generic service lifecycle status message.

        Updates the service registry with lifecycle state changes (initializing,
        running, stopping, etc.).

        Args:
            message: The status message to process
        """
        service_id = message.service_id
        service_type = message.service_type
        state = message.state

        self.debug(
            lambda: (
                f"Received status update from '{service_type}' (ID: '{service_id}'): {state}"
            )
        )

        # Update the component state if the component exists
        if service_id not in self.service_manager.service_id_map:
            self.debug(
                lambda: (
                    f"Received status update from un-registered service: {service_id} ({service_type})"
                )
            )
            return

        service_info = self.service_manager.service_id_map.get(service_id)
        if service_info is None:
            return

        service_info.state = message.state
        ServiceRegistry.update_service(
            service_id,
            service_type,
            message.request_ns or time.time_ns(),
            message.state,
        )

        self.debug(f"Updated state for {service_id} to {message.state}")

    @on_message(MessageType.TELEMETRY_STATUS)
    async def _on_telemetry_status_message(
        self, message: TelemetryStatusMessage
    ) -> None:
        """Handle telemetry status from TelemetryManager.

        TelemetryStatusMessage informs SystemController if telemetry results will be available.
        """

        self._telemetry_endpoints_configured = message.endpoints_configured
        self._telemetry_endpoints_reachable = message.endpoints_reachable

        if not message.enabled:
            # Advertised the capability at registration but won't actually produce;
            # drop the domain from the barrier.
            self._result_join_coordinator.unregister("telemetry", message.service_id)
            reason_msg = f": {message.reason}" if message.reason else ""
            self.info(f"DCGM telemetry skipped{reason_msg}")
        else:
            self.info(
                f"DCGM telemetry enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable"
            )

        # Re-check shutdown readiness in case results arrived before status message
        await self._check_and_trigger_shutdown()

    @on_message(MessageType.SERVER_METRICS_STATUS)
    async def _on_server_metrics_status_message(
        self, message: ServerMetricsStatusMessage
    ) -> None:
        """Handle server metrics status from ServerMetricsManager.

        ServerMetricsStatusMessage informs SystemController if server metrics results will be available.
        """

        self._server_metrics_endpoints_configured = message.endpoints_configured
        self._server_metrics_endpoints_reachable = message.endpoints_reachable

        if not message.enabled:
            self._result_join_coordinator.unregister(
                "server_metrics", message.service_id
            )
            reason_msg = f" - {message.reason}" if message.reason else ""
            self.info(f"Server metrics disabled{reason_msg}")
        else:
            self.info(
                f"Server metrics enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable."
            )
            unreachable_endpoints = set(message.endpoints_configured) - set(
                message.endpoints_reachable
            )
            if unreachable_endpoints:
                self.warning(
                    f"Unreachable endpoints: {', '.join(unreachable_endpoints)}"
                )

        # Re-check shutdown readiness in case results arrived before status message
        await self._check_and_trigger_shutdown()

    @on_message(MessageType.COMMAND_RESPONSE)
    async def _process_command_response_message(self, message: CommandResponse) -> None:
        """Process a command response message."""
        self.debug(lambda: f"Received command response message: {message}")
        if message.status == CommandResponseStatus.SUCCESS:
            self.debug(f"Command {message.command} succeeded from {message.service_id}")
        elif message.status == CommandResponseStatus.ACKNOWLEDGED:
            self.debug(
                f"Command {message.command} acknowledged from {message.service_id}"
            )
        elif message.status == CommandResponseStatus.UNHANDLED:
            self.debug(f"Command {message.command} unhandled from {message.service_id}")
        elif message.status == CommandResponseStatus.FAILURE:
            message = cast(CommandErrorResponse, message)
            self.error(
                f"Command {message.command} failed from {message.service_id}: {message.error}"
            )

    @on_command(CommandType.SPAWN_WORKERS)
    async def _handle_spawn_workers_command(self, message: SpawnWorkersCommand) -> None:
        """Handle a spawn workers command."""
        self.debug(lambda: f"Received spawn workers command: {message}")
        # Spawn the workers
        await self.service_manager.run_service(ServiceType.WORKER, message.num_workers)
        # If we are scaling the record processor service count with the number of workers, spawn the record processors
        if self.scale_record_processors_with_workers:
            await self.service_manager.run_service(
                ServiceType.RECORD_PROCESSOR,
                max(
                    1, message.num_workers // Environment.RECORD.PROCESSOR_SCALE_FACTOR
                ),
            )

    @on_command(CommandType.GET_POD_STATES)
    async def _handle_get_pod_states_command(
        self, _message: GetPodStatesCommand
    ) -> dict[str, object]:
        """Serve the controller's authoritative worker-state cache."""
        return self.get_pod_state_snapshot().model_dump(mode="json")

    @on_command(CommandType.SHUTDOWN_WORKERS)
    async def _handle_shutdown_workers_command(
        self, message: ShutdownWorkersCommand
    ) -> None:
        """Handle a shutdown workers command."""
        self.debug(lambda: f"Received shutdown workers command: {message}")
        # TODO: Handle individual worker shutdowns via worker id
        await self.service_manager.stop_service(ServiceType.WORKER)
        if self.scale_record_processors_with_workers:
            await self.service_manager.stop_service(ServiceType.RECORD_PROCESSOR)

    @on_message(MessageType.PROCESS_ALL_RESULTS)
    async def _on_process_all_results_message(
        self, message: ProcessAllResultsMessage
    ) -> None:
        """Receive the unified results message from RecordsManager.

        Supplements the per-stream PROCESS_RECORDS_RESULT / PROCESS_TELEMETRY_RESULT
        / PROCESS_SERVER_METRICS_RESULT handlers — those still own the shutdown
        trigger.
        """
        self.trace_or_debug(
            lambda: f"Received unified results message: {message}",
            lambda: "Received unified results message",
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
            # Aggregation-side diagnostics, NOT a verdict on the export. A
            # GPU-telemetry drain timeout or one malformed record must not
            # withhold a fully valid inference result set, so this deliberately
            # does not set ``_export_failed`` (which gates the Kubernetes
            # results-ready marker and ResultsExportedMessage). The run is still
            # reported and still exits non-zero: these entries reach
            # ``print_exit_errors`` and ``os._exit(1 if self._exit_errors ...)``.
            self._exit_errors.extend(
                ExitErrorInfo(
                    error_details=error,
                    operation="process_records",
                    service_id=message.service_id,
                )
                for error in message.results.errors
            )

        self.debug(
            lambda: (
                f"Error summary: {message.results.results.error_summary if message.results.results else 'N/A'}"
            )
        )

        self._profile_results = message.results
        self._merge_server_metric_phase_results(self._server_metrics_results)

        if not message.results.results:
            self.error(
                f"Received process records result message with no records: {message.results.results}"
            )

        self._result_join_coordinator.complete_domain("profile")
        # Coordinate with the remaining result domains before shutdown
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
            self._result_join_coordinator.complete_domain("telemetry")
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
            self._merge_server_metric_phase_results(server_metrics_results)
        except Exception as e:
            self.exception(f"Error processing server metrics results message: {e!r}")
        finally:
            self._server_metrics_result_arrived.set()
            self._result_join_coordinator.complete_domain("server_metrics")
            await self._check_and_trigger_shutdown()

    def _merge_server_metric_phase_results(
        self, server_metrics_results: ServerMetricsResults | None
    ) -> None:
        """Attach manager-owned named-phase summaries to profile phase rows."""
        if (
            server_metrics_results is None
            or not server_metrics_results.phase_results
            or self._profile_results is None
            or self._profile_results.results.phase_records is None
        ):
            return
        by_index = {
            result.phase_index: result
            for result in server_metrics_results.phase_results
            if result.phase_index is not None
        }
        for phase_result in self._profile_results.results.phase_records:
            matched = by_index.get(phase_result.phase_index)
            if matched is not None:
                phase_result.server_metrics_results = matched

    @on_message(MessageType.PROCESS_ACCURACY_RESULT)
    async def _on_process_accuracy_result_message(
        self, message: ProcessAccuracyResultMessage
    ) -> None:
        """Handle an accuracy results message."""
        try:
            self._accuracy_results = message.accuracy_result.results
        except Exception as e:
            self.exception(f"Error processing accuracy results message: {e!r}")
        finally:
            self._accuracy_result_arrived.set()
            self._result_join_coordinator.complete_domain("accuracy")
            await self._check_and_trigger_shutdown()

    def _is_api_service_alive(self) -> bool:
        """Return True iff the API service is registered and its process is live.

        Used to gate the POST_COMPLETE_GRACE extension at shutdown: if the API
        never registered (startup failure) or has transitioned to FAILED/STOPPED,
        there is no listener for clients to reach, so the extended wait would
        only delay shutdown without serving anyone.

        BaseComponentService._on_state_change suppresses StatusMessage publishes
        once stop_requested is set, so service_map[ServiceType.API][*].state
        stays frozen at RUNNING even after the API process self-stopped, crashed,
        or transitioned to FAILED. On the multiprocess backend we cross-check
        process.is_alive() as the authoritative signal; other backends fall back
        to the registration/state check.
        """
        api_services = self.service_manager.service_map.get(ServiceType.API, [])
        terminal_states = (LifecycleState.STOPPED, LifecycleState.FAILED)
        registered = any(
            info.registration_status == ServiceRegistrationStatus.REGISTERED
            and info.state not in terminal_states
            for info in api_services
        )
        if not registered:
            return False
        manager = self.service_manager
        mp_info = (
            cast(LocalProcessServiceManagerProtocol, manager).multi_process_info
            if hasattr(manager, "multi_process_info")
            else None
        )
        # An empty list means this backend runs no local processes at all --
        # under Kubernetes the API is its own pod, and KubernetesServiceManager
        # inherits an always-empty list from MultiProcessServiceManager. Polling
        # it would report the API dead on every k8s run and collapse the
        # post-complete grace to the bare delivery window.
        if not isinstance(mp_info, list) or not mp_info:
            return True
        return any(
            rec.service_type == ServiceType.API
            and rec.process is not None
            and rec.process.is_alive()
            for rec in mp_info
        )

    async def _check_and_trigger_shutdown(self) -> None:
        """Trigger unified export + shutdown once every result domain has joined.

        Readiness is owned by the ``ResultJoinCoordinator`` shutdown barrier: each
        result-producing service that advertised a ``result_producer:<domain>``
        capability at registration must ``complete_domain`` (its result arrived) or
        drop out (service error / reported-disabled) before the barrier is
        ``ready``. This is called on every result/status/error message so it
        re-checks as domains complete.

        Thread safety: ``self._shutdown_lock`` makes the check-and-set of
        ``_shutdown_triggered`` atomic against concurrently-arriving result
        messages, preventing double-triggering of stop().
        """
        self.debug(
            lambda: (
                "_check_and_trigger_shutdown: "
                f"pending_domains={self._result_join_coordinator.pending_domains}, "
                f"shutdown_triggered={self._shutdown_triggered}"
            )
        )
        # Registration is concurrent. An early optional producer can unregister
        # before the profile producer registers, making the empty barrier read as
        # ready even though profiling has not begun.
        if self._system_state in _PRE_BENCHMARK_STATES:
            self.debug(
                lambda: (
                    "_check_and_trigger_shutdown: ignoring an early barrier "
                    f"in state {self._system_state}"
                )
            )
            return

        should_shutdown = False
        async with self._shutdown_lock:
            if self._shutdown_triggered:
                self.debug(
                    "_check_and_trigger_shutdown: shutdown already triggered, returning"
                )
                return

            if self._result_join_coordinator.ready:
                self._shutdown_triggered = True
                should_shutdown = True
                # Teardown starts here: worker pods exiting from now on are
                # expected, not a failure the benchmark should be cancelled for.
                self._disarm_pod_failure_watcher()
                await self._set_system_state(SystemState.STOPPING)
                self.info("All results received, initiating shutdown")
            elif (
                pending_domains
                := self._result_join_coordinator.pending_domains_changed()
            ) is not None:
                self.info(f"Waiting for result domains: {', '.join(pending_domains)}")

        # Call stop() OUTSIDE the lock to prevent deadlock
        if should_shutdown:
            await self._finalize_kubernetes_raw_artifacts()
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
                "[bold yellow]BENCHMARK CANCELLED[/bold yellow]\n\n"
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
                "[bold red]FORCE QUIT[/bold red]\n\n"
                "Terminating all processes immediately.\n"
                "Results may be incomplete or not written to files.",
                border_style="red",
                padding=(1, 2),
                title="[bold red]Force Quit[/bold red]",
            )
        )
        console.print()
        console.file.flush()

    async def _cancel_profiling(self) -> None:
        self.debug("Cancelling profiling of all services")
        self._was_cancelled = True
        self._disarm_pod_failure_watcher()
        await self._set_system_state(SystemState.STOPPING)

        # Mark shutdown as triggered FIRST to prevent _check_and_trigger_shutdown()
        # from also calling stop() when results arrive during cancellation.
        # This prevents the race condition that causes SIGKILL (exit code -9).
        # Also track if shutdown was already triggered to avoid double-stop.
        should_call_stop = False
        async with self._shutdown_lock:
            if not self._shutdown_triggered:
                self._shutdown_triggered = True
                should_call_stop = True
            else:
                self.debug("Shutdown already triggered, skipping stop() call")

        # Only wait for RecordsManager's response since it returns ProcessRecordsResult.
        # Other services receive the broadcast cancel command but we don't wait for them.
        # This avoids blocking if a service has exited early (e.g., TelemetryManager).
        records_manager_ids = [
            service_id
            for service_id, info in self.service_manager.service_id_map.items()
            if info.service_type == ServiceType.RECORDS_MANAGER
        ]
        self.debug(
            f"Sending cancel to all services, waiting for {len(records_manager_ids)} RecordsManager(s)"
        )

        try:
            responses = await self.send_command_and_wait_for_all_responses(
                ProfileCancelCommand(
                    service_id=self.service_id,
                ),
                records_manager_ids,
                timeout=Environment.SERVICE.PROFILE_CANCEL_TIMEOUT,
            )
            self._consume_profile_cancel_responses(responses)
        except Exception as e:
            # Catch ANY exception during cancellation - we must always proceed to stop().
            self.warning(f"Exception during cancel command (proceeding to stop): {e!r}")

        # The normal completion path holds stop() on the accuracy result domain
        # until RecordsManager publishes ProcessAccuracyResultMessage; the cancel
        # path bypasses the barrier, so wait a bounded time here for the graded
        # summary to arrive before exporting (otherwise a cancelled accuracy run
        # can export with accuracy.* rows missing).
        await self._await_cancel_result_domains(should_call_stop)
        self._merge_server_metric_phase_results(self._server_metrics_results)

        # Only call stop() if we were the first to trigger shutdown
        if should_call_stop:
            await self._finalize_kubernetes_raw_artifacts()
            self.debug("Stopping system controller after profiling cancelled")
            await asyncio.shield(self.stop())

    def _consume_profile_cancel_responses(
        self, responses: list[CommandResponse | ErrorDetails]
    ) -> None:
        """Record cancellation errors and retain any final records summary."""
        for response in responses:
            if isinstance(response, ErrorDetails):
                self.warning(
                    f"Cancel command error (timeout or service unavailable): {response}"
                )
            elif isinstance(response, CommandErrorResponse):
                self.warning(
                    f"Cancel command failed from {response.service_id}: {response.error}"
                )

        for response in responses:
            if (
                isinstance(response, CommandSuccessResponse)
                and response.command == CommandType.PROFILE_CANCEL
                and isinstance(response.data, ProcessRecordsResult)
            ):
                self.debug(
                    lambda r=response: (
                        f"Received ProcessRecordsResult from cancel command: {r.data}"
                    )
                )
                self._profile_results = response.data
                return

    async def _await_cancel_result_domains(self, should_wait: bool) -> None:
        """Wait for result producers that may still be finalizing on cancellation."""
        if not should_wait:
            return
        if "accuracy" in self._result_join_coordinator.pending_domains:
            await self._await_accuracy_results_for_cancel()
        if "server_metrics" in self._result_join_coordinator.pending_domains:
            await self._await_server_metrics_results_for_cancel()

    async def _await_accuracy_results_for_cancel(self) -> None:
        """Bounded wait for the accuracy summary on the cancel (Ctrl+C) path.

        Waits up to ``Environment.ACCURACY.CANCEL_RESULT_WAIT_SEC`` for
        RecordsManager's ``ProcessAccuracyResultMessage`` to land, returning as
        soon as it does. Awaits ``_accuracy_result_arrived`` (set even on a
        ``results=None`` empty run) so the wait ends on message arrival, not on
        the results field.
        """
        timeout = Environment.ACCURACY.CANCEL_RESULT_WAIT_SEC
        try:
            await asyncio.wait_for(self._accuracy_result_arrived.wait(), timeout)
            self.debug("Accuracy results arrived during cancel wait")
        except TimeoutError:
            self.warning(
                "Accuracy results did not arrive within "
                f"{timeout}s of cancellation; export may omit accuracy metrics"
            )

    async def _await_server_metrics_results_for_cancel(self) -> None:
        """Bound cancellation until manager-owned server metrics are published."""
        timeout = Environment.SERVER_METRICS.PROFILE_COMPLETE_RELAY_TIMEOUT
        try:
            await asyncio.wait_for(self._server_metrics_result_arrived.wait(), timeout)
            self.debug("Server metrics results arrived during cancel wait")
        except TimeoutError:
            self.warning(
                "Server metrics results did not arrive within "
                f"{timeout}s of cancellation; export may omit server metrics"
            )

    @on_command(CommandType.FINALIZE_ARTIFACTS)
    async def _handle_finalize_artifacts_command(
        self,
        message: FinalizeArtifactsCommand,  # noqa: ARG002
    ) -> None:
        """Coordinate an exact durability barrier for every record processor.

        Fails closed: every failure is recorded in ``_exit_errors`` (so the
        controller's own exit code reflects it even if the caller drops the
        response) and then raised, which ``CommandHandlerMixin`` turns into a
        ``CommandErrorResponse`` for the caller. Reaped services are excluded
        from the target list -- commanding a process the watchdog already
        confirmed dead only buys a full command timeout before failing on a
        peer that is known to be gone.
        """
        if self._is_kubernetes():
            await self._finalize_kubernetes_raw_artifacts()
            if self._export_failed:
                raise RuntimeError("Kubernetes RAW artifact finalization failed")
            return

        service_ids = sorted(
            service_id
            for service_id, info in self.service_manager.service_id_map.items()
            if info.service_type == ServiceType.RECORD_PROCESSOR
            and service_id not in self._reaped_service_ids
        )
        if not service_ids:
            raise self._record_and_build_finalize_failure(
                service_id=self.service_id,
                message=(
                    "Cannot finalize record artifacts: no live record "
                    "processors are registered"
                ),
            )

        command = FinalizeArtifactsCommand(
            service_id=self.service_id,
            target_service_type=ServiceType.RECORD_PROCESSOR,
            request_ns=time.time_ns(),
        )
        responses = await self.send_command_and_wait_for_all_responses(
            command,
            service_ids,
            timeout=Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
        )
        failures: list[Exception] = []
        for service_id, response in zip(service_ids, responses, strict=True):
            if (
                isinstance(response, CommandAcknowledgedResponse)
                and response.command == CommandType.FINALIZE_ARTIFACTS
                and response.service_id == service_id
            ):
                continue
            if isinstance(response, ErrorDetails):
                detail = response
            elif isinstance(response, CommandErrorResponse):
                detail = response.error
            elif isinstance(response, CommandUnhandledResponse):
                detail = ErrorDetails(
                    message=(
                        f"Record processor '{service_id}' does not handle "
                        f"{CommandType.FINALIZE_ARTIFACTS}"
                    )
                )
            else:
                detail = ErrorDetails(
                    message=(
                        "Unexpected artifact finalization response from "
                        f"'{service_id}': {response!r}"
                    )
                )
            failures.append(
                self._record_and_build_finalize_failure(
                    service_id=service_id, message=detail.message
                )
            )
        if failures:
            raise ExceptionGroup(
                f"Failed to finalize {len(failures)} record processor(s)", failures
            )

    def _record_and_build_finalize_failure(
        self, *, service_id: str, message: str
    ) -> RuntimeError:
        """Record an artifact-finalization failure and return it for raising.

        Recording is what makes the failure reportable: the caller may drop the
        command response, but ``_exit_errors`` still forces a non-zero exit and
        a printed error panel instead of a silent success.
        """
        self._exit_errors.append(
            ExitErrorInfo(
                error_details=ErrorDetails(message=message, type="FinalizeArtifacts"),
                operation="finalize_artifacts",
                service_id=service_id,
            )
        )
        self.error(message)
        return RuntimeError(message)

    def _record_raw_artifact_finalize_failure(
        self, *, service_id: str, error: ErrorDetails
    ) -> None:
        """Make an incomplete Kubernetes RAW artifact set fail closed."""
        self._export_failed = True
        self._exit_errors.append(
            ExitErrorInfo(
                error_details=error,
                operation="finalize_raw_artifacts",
                service_id=service_id,
            )
        )

    @staticmethod
    def _raw_artifact_finalize_response_error(
        service_id: str, response: CommandResponse | ErrorDetails
    ) -> ErrorDetails | None:
        """Return an error unless ``response`` is the expected peer ACK."""
        if (
            isinstance(response, CommandAcknowledgedResponse)
            and response.command == CommandType.FINALIZE_ARTIFACTS
            and response.service_id == service_id
        ):
            return None
        if isinstance(response, ErrorDetails):
            return response
        if isinstance(response, CommandErrorResponse):
            return response.error
        if isinstance(response, CommandUnhandledResponse):
            return ErrorDetails(
                message=(
                    f"Worker-group manager '{service_id}' does not handle "
                    f"{CommandType.FINALIZE_ARTIFACTS}"
                )
            )
        return ErrorDetails(
            message=(
                f"Unexpected RAW artifact finalization response from "
                f"'{service_id}': {response!r}"
            )
        )

    def _raw_finalize_membership_is_acceptable(
        self, expected: int, service_ids: list[str]
    ) -> bool:
        """Decide whether the surviving worker groups can finalize RAW artifacts.

        The finalize barrier must agree with the pod-loss tolerance policy it
        runs under. ``POD.FAILURE_ABORT_THRESHOLD_PERCENT`` deliberately lets a
        4-pod run continue on 3, so demanding exact equality here failed a run
        the rest of the system had already chosen to keep -- and did so without
        contacting the three healthy pods at all.

        Within tolerance the barrier proceeds against the pods that are
        genuinely expected to be alive and marks the run degraded (non-zero
        exit, named producers) rather than discarding it. Outside tolerance, or
        with nothing left to ask, it still fails closed and withholds
        readiness: an incomplete RAW set must never be published as complete.
        """
        missing = expected - len(service_ids)
        if missing <= 0:
            return True

        threshold = Environment.POD.FAILURE_ABORT_THRESHOLD_PERCENT
        missing_percent = (missing / expected) * 100 if expected else 100.0
        # threshold == 0 disables pod-failure aborts entirely, so any loss is
        # tolerated -- but a barrier with no members left proves nothing.
        tolerated = service_ids and (threshold == 0 or missing_percent < threshold)
        if not tolerated:
            error = RuntimeError(
                "Cannot finalize Kubernetes RAW artifacts: expected "
                f"{expected} registered worker-group manager(s), found "
                f"{len(service_ids)} ({', '.join(service_ids) or 'none'})"
            )
            self._record_raw_artifact_finalize_failure(
                service_id=self.service_id,
                error=ErrorDetails.from_exception(error),
            )
            return False

        message = (
            f"Finalizing Kubernetes RAW artifacts on {len(service_ids)} of "
            f"{expected} worker-group manager(s): {missing} pod(s) were lost "
            f"within the {threshold:.0f}% abort threshold, so their RAW shards "
            "are absent from this run"
        )
        self.warning(message)
        self._exit_errors.append(
            ExitErrorInfo(
                error_details=ErrorDetails(
                    message=message, type="DegradedRawArtifactSet"
                ),
                operation="finalize_raw_artifacts_degraded",
                service_id=self.service_id,
            )
        )
        return True

    async def _finalize_kubernetes_raw_artifacts(self) -> None:
        """Wait for every exact worker group to flush and upload RAW artifacts.

        This command runs before the shutdown broadcast, while the controller,
        worker-group managers, and record processors can still acknowledge
        failures. Registered service identities are authoritative; filename
        counts cannot distinguish an idle processor from a missing upload.
        """
        if (
            not self._is_kubernetes()
            or self.run.cfg.artifacts.export_level != ExportLevel.RAW
        ):
            return

        expected = self._k8s_topology.num_worker_pods
        service_ids = sorted(
            info.service_id
            for info in ServiceRegistry.get_services(ServiceType.WORKER_GROUP_MANAGER)
            if info.service_id not in self._reaped_service_ids
        )
        if not self._raw_finalize_membership_is_acceptable(expected, service_ids):
            return

        self.info(f"Finalizing RAW artifacts on {len(service_ids)} worker group(s)...")
        try:
            responses = await self.send_command_and_wait_for_all_responses(
                FinalizeArtifactsCommand(
                    service_id=self.service_id,
                    target_service_type=ServiceType.WORKER_GROUP_MANAGER,
                ),
                service_ids,
                timeout=Environment.WORKER.RAW_RECORD_UPLOAD_TIMEOUT,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._record_raw_artifact_finalize_failure(
                service_id=self.service_id,
                error=ErrorDetails.from_exception(e),
            )
            return

        for service_id, response in zip(service_ids, responses, strict=True):
            error = self._raw_artifact_finalize_response_error(service_id, response)
            if error is None:
                continue
            self._record_raw_artifact_finalize_failure(
                service_id=service_id, error=error
            )

        if not self._export_failed:
            self.info("All Kubernetes RAW artifacts finalized and uploaded")

    def _surface_export_failures(self, failures: list[ExporterFailure]) -> bool:
        """Record local export failures and report whether readiness is blocked.

        Deferred exporters only upload already-complete local artifacts, so a
        remote service outage is warning-worthy but does not make the local
        result set unsafe to serve.
        """
        marker_blocking = False
        for failure in failures:
            if failure.is_deferred:
                self.warning(
                    f"Deferred exporter '{failure.exporter}' failed; local "
                    f"results are unaffected: {failure.error!r}"
                )
                continue
            marker_blocking = True
            self._exit_errors.append(
                ExitErrorInfo(
                    error_details=ErrorDetails.from_exception(failure.error),
                    operation=f"export:{failure.exporter}",
                    service_id=self.service_id,
                )
            )
        return marker_blocking

    async def _announce_benchmark_complete(self) -> None:
        """Tell the API service the benchmark is over.

        The results endpoint reports "running" until this arrives, and
        POST /api/shutdown -- the operator's graceful-exit handshake -- answers
        409 until the same flag is set. Without the announcement every
        completion fell through to the operator's hard pod-delete fallback.
        """
        if not self._api_enabled:
            return
        await self.publish(
            BenchmarkCompleteMessage(
                service_id=self.service_id,
                was_cancelled=self._was_cancelled,
            )
        )

    @on_stop
    async def _stop_system_controller(self) -> None:
        """Stop the system controller and all running services."""
        await self._set_system_state(SystemState.SHUTDOWN)

        # Stop the local UI before console rendering, but keep services and the
        # message bus alive through export. In Kubernetes the API router must
        # receive ResultsExportedMessage after the durable ready marker commits;
        # exporting after comms.stop() made that live-bus contract impossible.
        await self.ui.stop()
        await self.ui.wait_for_tasks()

        # Reporting must never prevent the service shutdown below. At this
        # point all result domains and the RAW artifact barrier are complete,
        # but the API and event bus remain live for the export notification.
        try:
            # "Degraded but has results" and "no results at all" are different
            # outcomes and must not share a gate. Any recorded error used to
            # skip the export entirely, so a single aggregation diagnostic or a
            # reaped producer threw away profile_export.csv/.json, the console
            # summary, auto-plot, the Kubernetes ready marker and
            # ResultsExportedMessage for a run that had complete records in
            # hand. Export whenever there is something to export; the errors are
            # still printed below and still drive the non-zero exit code.
            if self._has_exportable_results() or not self._exit_errors:
                await self._print_post_benchmark_info_and_metrics()
            if self._exit_errors:
                self._print_exit_errors_and_log_file()

            if Environment.DEV.MODE:
                print_developer_mode_warning()
        except (UnicodeEncodeError, OSError) as e:
            self.error(f"Pre-shutdown reporting failed (continuing to exit): {e!r}")
        except BaseException:
            self.exception(
                "Unexpected pre-shutdown reporting failure (continuing to exit)"
            )

        # BenchmarkComplete makes the API shutdown endpoint eligible only
        # after export/ready publication has finished (or failed closed).
        await self._announce_benchmark_complete()
        await self.publish(ShutdownCommand(service_id=self.service_id))

        # ShutdownCommand is fire-and-forget on the pub/sub bus: BaseComponentService's
        # SHUTDOWN handler raises asyncio.CancelledError instead of returning, so the
        # CommandHandlerMixin wrapper never publishes an ack we could await. Child
        # processes also ignore SIGTERM (see bootstrap.py), so
        # MultiProcessServiceManager._wait_for_process skips terminate()+join and
        # goes straight to kill() for any process still alive after this grace —
        # only a successful message-bus delivery here results in graceful
        # shutdown rather than SIGKILL. This grace period gives ZMQ inproc/IPC
        # pub/sub time to deliver the broadcast to every subscriber before we
        # start killing stragglers. 500ms is empirically sufficient under normal
        # load.
        # When the API server is enabled AND still alive, extend the wait so the
        # API process can honor its POST_COMPLETE_GRACE window before
        # _wait_for_process SIGKILLs it. If the API never registered or has
        # already failed/stopped, the extension would only delay shutdown without
        # serving any client, so skip it.
        delivery_grace = 0.5
        if self._api_enabled and self._is_api_service_alive():
            delivery_grace = max(
                delivery_grace, Environment.API_SERVER.POST_COMPLETE_GRACE
            )
        await asyncio.sleep(delivery_grace)

        await self.service_manager.shutdown_all_services()
        await self.comms.stop()
        await self.proxy_manager.stop()

        # Clean up the global log queue to prevent semaphore leaks
        await cleanup_global_log_queue()

        # Exit the process in a more explicit way, to ensure that it stops
        os._exit(1 if self._exit_errors else 0)

    def _has_exportable_results(self) -> bool:
        """Whether the run produced records worth writing out.

        This is the question the shutdown reporting path must ask, as opposed
        to "did anything go wrong": a run can be degraded (a producer died, an
        aggregation stage reported an error) and still hold a complete,
        exportable record set.
        """
        return bool(self._profile_results and self._profile_results.results.records)

    def _print_degraded_producers(self, console: Console) -> None:
        """Name the producers whose results are missing from this export.

        The result-join barrier releases when a dead producer is evicted, so
        without this the console table is indistinguishable from a complete
        run -- the eviction only appeared in a log line.
        """
        evicted = self._result_join_coordinator.evicted
        if not evicted:
            return
        console.print(
            f"[bold yellow]DEGRADED:[/bold yellow] {len(evicted)} result "
            "producer(s) died before reporting; the metrics above are computed "
            "from the surviving producers only."
        )
        for service_id, reason in sorted(evicted.items()):
            console.print(f"  [yellow]•[/yellow] {service_id}: {reason}")

    def _print_exit_errors_and_log_file(self) -> None:
        """Print post exit errors and log file info to the console."""
        console = Console()
        print_exit_errors(self._exit_errors, console=console)
        self._print_log_file_info(console)
        console.print()
        console.file.flush()

    def _inject_accuracy_results_into_records(self) -> None:
        """Materialize the dedicated-channel accuracy summary into the profile records.

        The accuracy computation/transport lives on its own dedicated channel
        (``AccuracyAccumulator`` -> ``AccuracySummary``), but legacy exporters
        (perf CSV/JSON + the accuracy CSV/console) read ``accuracy.*`` MetricResults
        from ``ProfileResults.records``. Convert the summary to those MetricResults
        and append them at the END (so JSON key order matches legacy: accuracy.*
        after all perf metrics). Guarded so a re-export cannot double-append.
        """
        if self._accuracy_results is None or self._accuracy_results_injected:
            return
        if not self._profile_results or self._profile_results.results.records is None:
            return
        self._profile_results.results.records.extend(
            self._accuracy_results.to_metric_results()
        )
        self._accuracy_results_injected = True

    async def _print_post_benchmark_info_and_metrics(self) -> None:
        """Print post benchmark info and metrics to the console."""
        if not await self._begin_results_export_transaction():
            return

        if not self._profile_results or not self._profile_results.results.records:
            self.error("No profile results to export")
            # Record the failure in _exit_errors so the caller's
            # ``os._exit(1 if self._exit_errors else 0)`` exits non-zero.
            # ``sys.exit(1)`` here is swallowed because we run inside an
            # asyncio task hook, leaving the process to exit cleanly.
            self._exit_errors.append(
                ExitErrorInfo(
                    error_details=ErrorDetails(
                        message="No profile results to export. "
                        "A required service likely failed before any "
                        "records could be collected — see prior log output.",
                    ),
                    operation="export_results",
                    service_id=self.id,
                )
            )
            return

        results = self._profile_results.results
        if results.successful_request_count == 0 and results.error_request_count > 0:
            self.error(
                f"All {results.error_request_count} inference request(s) failed; "
                "no successful responses were collected."
            )
            self._exit_errors.append(
                ExitErrorInfo(
                    error_details=ErrorDetails(
                        message=(
                            f"All {results.error_request_count} inference "
                            "request(s) failed. No successful responses were "
                            "collected — check the server URL, endpoint path, "
                            "and response format. See prior log output for "
                            "per-request error details."
                        ),
                    ),
                    operation="export_results",
                    service_id=self.id,
                )
            )
            return

        console = Console()
        if console.width < 100:
            console.width = 100

        self._inject_accuracy_results_into_records()

        exporter_manager = ExporterManager(
            results=self._profile_results.results,
            run=self.run,
            telemetry_results=self._telemetry_results,
            server_metrics_results=self._server_metrics_results,
        )

        # Export data files (CSV, JSON) with complete dataset including telemetry
        export_failures = await exporter_manager.export_data()
        local_export_failed = self._surface_export_failures(export_failures)
        self._export_failed = self._export_failed or local_export_failed

        # Export console output with complete dataset including telemetry
        await exporter_manager.export_console(console=console)

        console.print()
        self._print_cli_command(console)
        self._print_benchmark_duration(console)
        self._print_exported_file_infos(exporter_manager, console)
        self._print_log_file_info(console)
        self._print_degraded_producers(console)
        if self._was_cancelled:
            console.print(
                "[italic yellow]The profile run was cancelled early. Results shown may be incomplete or inaccurate.[/italic yellow]"
            )

        console.print()
        console.file.flush()

        await self._run_kubernetes_auto_plot()
        await self._announce_results_exported()

    async def _begin_results_export_transaction(self) -> bool:
        """Hide stale exports before a Kubernetes result export begins."""
        if not self._is_kubernetes():
            return True

        artifact_dir = self.run.cfg.artifacts.dir
        try:
            await asyncio.to_thread(write_processing_marker, artifact_dir)
        except OSError as error:
            self._export_failed = True
            self._exit_errors.append(
                ExitErrorInfo(
                    error_details=ErrorDetails.from_exception(error),
                    operation="export:ResultsProcessingMarker",
                    service_id=self.service_id,
                )
            )
            self.error(
                f"Failed to begin the results export transaction in {artifact_dir}: "
                f"{error!r}; withholding results readiness"
            )
            return False
        return True

    async def _run_kubernetes_auto_plot(self) -> None:
        """Render configured plots before publishing Kubernetes readiness."""
        artifacts = self.run.cfg.artifacts
        if not self._is_kubernetes() or not artifacts.auto_plot or self._export_failed:
            return

        from aiperf.plot.auto_plot import run_auto_plot_async

        try:
            await run_auto_plot_async(
                artifact_dir=artifacts.dir,
                plot_required=artifacts.plot_required,
                plot_envelope=self.run.plot,
            )
        except Exception as e:  # noqa: BLE001 - strict plotting is an export failure
            self._export_failed = True
            self._exit_errors.append(
                ExitErrorInfo(
                    error_details=ErrorDetails.from_exception(e),
                    operation="auto_plot",
                    service_id=self.service_id,
                )
            )
            self.error(
                f"Required auto-plot failed in {artifacts.dir}: {e!r}; "
                "withholding results readiness"
            )

    async def _announce_results_exported(self) -> None:
        """Commit readiness, publish locally, then notify the Kubernetes operator.

        These are the handshakes the result consumers wait on. The results sidecar
        refuses to serve top-level artifacts until
        ``.aiperf_results_ready.json`` exists, and ProgressRouter only reports
        ``is_complete`` after ResultsExportedMessage. The controller then patches
        the parent AIPerfJob's benchmark-complete annotation so kopf can harvest
        immediately; its timer remains the recovery path for a failed patch.
        """
        if getattr(self, "_export_failed", False):
            self.error(
                "Local result export failed; withholding results-ready and "
                "ResultsExportedMessage"
            )
            return

        artifact_dir = self.run.cfg.artifacts.dir
        if self._is_kubernetes():
            try:
                await asyncio.to_thread(
                    write_ready_marker,
                    artifact_dir,
                    was_cancelled=self._was_cancelled,
                )
            except OSError as e:
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=ErrorDetails.from_exception(e),
                        operation="export:ResultsReadyMarker",
                        service_id=self.service_id,
                    )
                )
                self.error(
                    f"Failed to write the results-ready marker in {artifact_dir}: "
                    f"{e!r}; withholding ResultsExportedMessage"
                )
                return
        try:
            await self.publish(ResultsExportedMessage(service_id=self.service_id))
        except asyncio.CancelledError:
            # The durable marker remains authoritative if the live-bus
            # optimization disappears during teardown.
            self.warning(
                "Message bus stopped during ResultsExportedMessage publication; "
                "the durable results-ready marker remains authoritative"
            )
        except Exception as e:  # noqa: BLE001 - a late bus failure must not mask exported results
            self.warning(f"Failed to publish ResultsExportedMessage: {e!r}")

        if self._is_kubernetes():
            from aiperf.kubernetes.completion_signal import signal_benchmark_complete

            try:
                await signal_benchmark_complete()
            except asyncio.CancelledError:
                self.warning(
                    "Benchmark-complete notification was cancelled; the durable "
                    "results marker and operator timer remain authoritative"
                )
            except Exception as e:  # noqa: BLE001 - notification is a latency optimization
                self.warning(
                    f"Failed to notify the operator of benchmark completion: {e!r}; "
                    "the operator timer will recover"
                )

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
        cli_command = self.run.cli_command
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

    async def _kill(self):
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

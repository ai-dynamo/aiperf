# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.control_structs import CommandErr, Registration, RegistrationAck
from aiperf.common.enums import LifecycleState, WorkerStartupState, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.exceptions import (
    LifecycleOperationError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.models import ErrorDetails, ExitErrorInfo
from aiperf.common.service_registry import ServiceRegistry
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceRunType, ServiceType
from tests.unit.controller.conftest import MockTestException


def assert_exit_error(
    system_controller: SystemController,
    expected_error_or_exception: ErrorDetails | LifecycleOperationError,
    operation: str,
    service_id: str | None,
) -> None:
    """Assert that an exit error was recorded with the proper details."""
    assert len(system_controller._exit_errors) == 1
    exit_error = system_controller._exit_errors[0]
    assert isinstance(exit_error, ExitErrorInfo)

    # Handle both ErrorDetails objects and LifecycleOperationError objects
    if isinstance(expected_error_or_exception, ErrorDetails):
        expected_error_details = expected_error_or_exception
    else:
        expected_error_details = ErrorDetails.from_exception(
            expected_error_or_exception
        )

    assert exit_error.error_details == expected_error_details
    assert exit_error.operation == operation
    assert exit_error.service_id == service_id


class TestSystemController:
    """Test SystemController."""

    @pytest.mark.asyncio
    async def test_system_controller_no_error_on_initialize_success(
        self, system_controller: SystemController, mock_service_manager: AsyncMock
    ):
        """Test that SystemController does not exit when initialize succeeds."""
        mock_service_manager.initialize.return_value = None
        await system_controller._initialize_system_controller()
        # Verify that no exit errors were recorded
        assert len(system_controller._exit_errors) == 0

    @pytest.mark.asyncio
    async def test_system_controller_no_error_on_start_success(
        self, system_controller: SystemController, mock_service_manager: AsyncMock
    ):
        """Test that SystemController does not exit when start services succeeds."""
        mock_service_manager.start.return_value = None
        system_controller._start_profiling_all_services = AsyncMock(return_value=None)
        system_controller._wait_for_all_configured = AsyncMock(return_value=None)

        await system_controller._start_services()
        # Verify that no exit errors were recorded
        assert len(system_controller._exit_errors) == 0

        assert mock_service_manager.start.called
        assert system_controller._wait_for_all_configured.called
        assert system_controller._start_profiling_all_services.called


class TestSystemControllerExitScenarios:
    """Test exit scenarios for the SystemController."""

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_profile_configure_error_response(
        self,
        system_controller: SystemController,
        mock_exception: MockTestException,
    ):
        """Test that SystemController records configure errors when a service returns CommandErr."""
        error_response = CommandErr(
            cid="test-cid",
            sid="test-service",
            error=str(mock_exception),
            traceback="",
        )
        system_controller._send_control_command = AsyncMock(return_value=error_response)

        await system_controller._configure_single_service("test-service")

        assert len(system_controller._configure_errors) == 1
        assert isinstance(system_controller._configure_errors[0], CommandErr)
        assert system_controller._configure_errors[0].error == str(mock_exception)

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_profile_start_error_response(
        self,
        system_controller: SystemController,
        mock_exception: MockTestException,
    ):
        """Test that SystemController exits when receiving a CommandErr for profile_start."""
        error_responses = [
            CommandErr(
                cid="test-cid",
                sid="test-service",
                error=str(mock_exception),
                traceback="",
            )
        ]
        system_controller._send_control_command_to_all_fail_fast = AsyncMock(
            return_value=error_responses
        )

        with pytest.raises(
            LifecycleOperationError,
            match="Failed to perform operation 'Start Profiling'",
        ):
            await system_controller._start_profiling_all_services()

        assert len(system_controller._exit_errors) == 1
        assert system_controller._exit_errors[0].operation == "Start Profiling"
        assert system_controller._exit_errors[0].service_id == "test-service"

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_service_manager_initialize_error(
        self,
        system_controller: SystemController,
        mock_service_manager: AsyncMock,
        mock_exception: MockTestException,
    ):
        """Test that SystemController exits when the service manager initialize fails."""
        mock_service_manager.initialize.side_effect = mock_exception
        with pytest.raises(LifecycleOperationError, match=str(mock_exception)):
            await system_controller._initialize_system_controller()

        # Verify that exit errors were recorded
        assert_exit_error(
            system_controller,
            mock_exception,
            "Initialize Service Manager",
            system_controller.id,
        )

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_service_manager_start_error(
        self,
        system_controller: SystemController,
        mock_service_manager: AsyncMock,
        mock_exception: MockTestException,
    ):
        """Test that SystemController exits when the service manager start fails."""
        mock_service_manager.start.side_effect = LifecycleOperationError(
            operation="Start Service",
            original_exception=mock_exception,
            lifecycle_id=system_controller.id,
        )
        with pytest.raises(LifecycleOperationError, match="Test error"):
            await system_controller._start_services()

        # Verify that exit errors were recorded
        assert_exit_error(
            system_controller,
            LifecycleOperationError(
                operation="Start Service",
                original_exception=mock_exception,
                lifecycle_id=system_controller.id,
            ),
            "Start Service Manager",
            system_controller.id,
        )

    @pytest.mark.asyncio
    async def test_system_controller_exits_on_wait_for_all_configured_error(
        self,
        system_controller: SystemController,
        mock_service_manager: AsyncMock,
        mock_exception: MockTestException,
    ):
        """Test that SystemController exits when _wait_for_all_configured fails."""
        mock_service_manager.start.return_value = None
        system_controller._wait_for_all_configured = AsyncMock(
            side_effect=LifecycleOperationError(
                operation="Configure Profiling",
                original_exception=mock_exception,
                lifecycle_id=system_controller.id,
            )
        )
        with pytest.raises(LifecycleOperationError, match="Test error"):
            await system_controller._start_services()

        # Verify that exit errors were recorded
        assert_exit_error(
            system_controller,
            LifecycleOperationError(
                operation="Configure Profiling",
                original_exception=mock_exception,
                lifecycle_id=system_controller.id,
            ),
            "Configure Services",
            system_controller.id,
        )


# =============================================================================
# Signal Handling Tests (Two-Stage Ctrl+C)
# =============================================================================


class TestSignalHandling:
    """Tests for two-stage Ctrl+C signal handling."""

    @pytest.mark.asyncio
    async def test_first_signal_calls_cancel_profiling(
        self, system_controller: SystemController
    ):
        """First Ctrl+C calls _cancel_profiling for graceful shutdown."""
        system_controller._cancel_profiling = AsyncMock()
        system_controller._kill = AsyncMock()

        # First signal - should trigger graceful cancel
        with patch.object(system_controller, "_print_cancel_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        system_controller._cancel_profiling.assert_called_once()
        system_controller._kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_signal_calls_kill(self, system_controller: SystemController):
        """Second Ctrl+C calls _kill for immediate termination."""
        system_controller._cancel_profiling = AsyncMock()
        system_controller._kill = AsyncMock()
        system_controller._was_cancelled = (
            True  # Simulate first Ctrl+C already happened
        )

        # Second signal - should trigger force quit
        with patch.object(system_controller, "_print_force_quit_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        system_controller._kill.assert_called_once()
        system_controller._cancel_profiling.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_signal_sets_was_cancelled_flag(
        self, system_controller: SystemController, mock_service_manager: AsyncMock
    ):
        """First Ctrl+C sets _was_cancelled flag via _cancel_profiling."""
        system_controller._send_control_command_to_all = AsyncMock(return_value=[])
        system_controller.control_router = AsyncMock()
        system_controller.stop = AsyncMock()

        assert system_controller._was_cancelled is False

        with patch.object(system_controller, "_print_cancel_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        assert system_controller._was_cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_profiling_sends_profile_cancel_command(
        self, system_controller: SystemController, mock_service_manager: AsyncMock
    ):
        """_cancel_profiling sends PROFILE_CANCEL to all services."""
        system_controller._send_control_command_to_all = AsyncMock(return_value=[])
        system_controller.control_router = AsyncMock()
        system_controller.stop = AsyncMock()

        await system_controller._cancel_profiling()

        system_controller._send_control_command_to_all.assert_called_once()

    def test_print_cancel_warning_uses_console(
        self, system_controller: SystemController
    ):
        """_print_cancel_warning prints to console."""
        with patch("aiperf.controller.system_controller.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            system_controller._print_cancel_warning()

            # Should have printed something
            assert mock_console.print.call_count >= 2  # Panel and newlines
            mock_console.file.flush.assert_called_once()

    def test_print_force_quit_warning_uses_console(
        self, system_controller: SystemController
    ):
        """_print_force_quit_warning prints to console."""
        with patch("aiperf.controller.system_controller.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            system_controller._print_force_quit_warning()

            # Should have printed something
            assert mock_console.print.call_count >= 2  # Panel and newlines
            mock_console.file.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_sequential_signals_go_graceful_then_force(
        self, system_controller: SystemController
    ):
        """Sequential signals: first graceful cancel, second force quit."""

        # Mock _cancel_profiling to set _was_cancelled flag (mimicking real behavior)
        async def cancel_side_effect():
            system_controller._was_cancelled = True

        system_controller._cancel_profiling = AsyncMock(side_effect=cancel_side_effect)
        system_controller._kill = AsyncMock()

        # First signal
        with patch.object(system_controller, "_print_cancel_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        assert system_controller._was_cancelled is True
        system_controller._cancel_profiling.assert_called_once()
        system_controller._kill.assert_not_called()

        # Second signal
        with patch.object(system_controller, "_print_force_quit_warning"):
            await system_controller._handle_signal(signal.SIGINT)

        system_controller._kill.assert_called_once()


class TestKubernetesMode:
    """Test Kubernetes-specific behavior in SystemController."""

    def _create_system_controller(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> tuple[SystemController, MagicMock]:
        """Create a SystemController with custom config, mimicking the conftest pattern.

        Returns:
            Tuple of (controller, mock_proxy_class) so callers can inspect ProxyManager kwargs.
        """
        from pathlib import Path

        run = BenchmarkRun(
            benchmark_id="test",
            cfg=config,
            artifact_dir=Path("/tmp/test"),
        )
        mock_ui = AsyncMock()
        mock_comm = AsyncMock()
        mock_comm.create_reply_client = MagicMock(return_value=AsyncMock())

        def mock_get_class(protocol, name):
            if protocol == "service_manager":
                return lambda **kwargs: mock_service_manager
            if protocol == "ui":
                return lambda **kwargs: mock_ui
            if protocol == "communication":
                return lambda **kwargs: mock_comm
            raise ValueError(f"Unknown protocol: {protocol}")

        with (
            patch(
                "aiperf.controller.system_controller.plugins.get_class",
                side_effect=mock_get_class,
            ),
            patch("aiperf.controller.system_controller.ProxyManager") as mock_proxy,
            patch(
                "aiperf.controller.system_controller.ZMQStreamingRouterClient",
                return_value=AsyncMock(),
            ),
            patch(
                "aiperf.common.mixins.communication_mixin.plugins.get_class",
                side_effect=mock_get_class,
            ),
        ):  # fmt: skip
            mock_proxy.return_value = AsyncMock()
            controller = SystemController(
                run=run,
                service_id="test_controller",
            )
            controller.stop = AsyncMock()
            return controller, mock_proxy

    def test_kubernetes_mode_includes_workers_and_rps_in_required_services(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """In K8s mode, the full worker-pod topology is in required_services.

        The controller must wait for worker pod managers, workers, and record
        processors derived from pod capacity, not just a placeholder worker.
        """
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.workers = 12
        config.runtime.workers_per_pod = 5
        config.runtime.record_processors_per_pod = 2
        controller, _ = self._create_system_controller(config, mock_service_manager)
        assert controller.required_services[ServiceType.WORKER_POD_MANAGER] == 3
        assert controller.required_services[ServiceType.WORKER] == 15
        assert controller.required_services[ServiceType.RECORD_PROCESSOR] == 6
        assert ServiceType.WORKER in controller.required_services
        assert ServiceType.RECORD_PROCESSOR in controller.required_services
        assert ServiceType.WORKER_POD_MANAGER in controller.required_services

    def test_multiprocessing_mode_does_not_include_worker_in_required(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """In local mode, WORKER is not in required_services.

        Workers are spawned by MultiprocessServiceManager, not tracked
        as required services that the controller waits for.
        """
        config.runtime.service_run_type = ServiceRunType.MULTIPROCESSING
        controller, _ = self._create_system_controller(config, mock_service_manager)
        assert ServiceType.WORKER not in controller.required_services

    def test_keep_api_running_true_in_kubernetes_with_api_port(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """In K8s mode with api_port set, keep_api_running should be True."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.api_port = 9090
        controller, _ = self._create_system_controller(config, mock_service_manager)
        is_k8s_mode = (
            controller.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES
        )
        keep_api_running = is_k8s_mode and controller.run.cfg.runtime.api_port
        assert keep_api_running

    def test_keep_api_running_false_in_local_mode(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """In local mode, keep_api_running should be False."""
        config.runtime.service_run_type = ServiceRunType.MULTIPROCESSING
        controller, _ = self._create_system_controller(config, mock_service_manager)
        is_k8s_mode = (
            controller.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES
        )
        keep_api_running = is_k8s_mode and controller.run.cfg.runtime.api_port
        assert not keep_api_running

    def test_keep_api_running_false_in_kubernetes_without_api_port(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """In K8s mode without api_port, keep_api_running should be False."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.api_port = None
        controller, _ = self._create_system_controller(config, mock_service_manager)
        is_k8s_mode = (
            controller.run.cfg.runtime.service_run_type == ServiceRunType.KUBERNETES
        )
        keep_api_running = is_k8s_mode and controller.run.cfg.runtime.api_port
        assert not keep_api_running

    @pytest.mark.asyncio
    async def test_kubernetes_start_services_does_not_re_run_required_services(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """Kubernetes startup should let the service manager handle required services once."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.api_port = None
        config.gpu_telemetry.enabled = False
        config.server_metrics.enabled = False
        controller, _ = self._create_system_controller(config, mock_service_manager)
        controller._wait_for_all_configured = AsyncMock(return_value=None)
        controller._wait_for_all_workers_ready = AsyncMock(return_value=None)
        controller._start_profiling_all_services = AsyncMock(return_value=None)
        controller._wait_for_endpoint_ready = AsyncMock(return_value=None)

        await controller._start_services()

        mock_service_manager.start.assert_awaited_once()
        mock_service_manager.run_service.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_local_start_services_starts_manager_before_spawning_workers(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """Local startup should start the service manager before spawning workers."""
        config.runtime.service_run_type = ServiceRunType.MULTIPROCESSING
        config.runtime.api_port = None
        config.gpu_telemetry.enabled = False
        config.server_metrics.enabled = False
        call_order: list[str] = []

        async def start_side_effect() -> None:
            call_order.append("start")

        async def run_service_side_effect(service_type: ServiceType, *_args) -> None:
            call_order.append(str(service_type))

        mock_service_manager.start.side_effect = start_side_effect
        mock_service_manager.run_service.side_effect = run_service_side_effect
        controller, _ = self._create_system_controller(config, mock_service_manager)
        controller._wait_for_all_configured = AsyncMock(return_value=None)
        controller._wait_for_all_workers_ready = AsyncMock(return_value=None)
        controller._start_profiling_all_services = AsyncMock(return_value=None)
        controller._wait_for_endpoint_ready = AsyncMock(return_value=None)

        await controller._start_services()

        assert call_order[0] == "start"
        assert str(ServiceType.WORKER) in call_order[1:]

    @pytest.mark.asyncio
    async def test_kubernetes_start_services_waits_for_all_workers_ready(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """Kubernetes startup should wait for app-level worker READY before profiling."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.api_port = None
        config.gpu_telemetry.enabled = False
        config.server_metrics.enabled = False
        controller, _ = self._create_system_controller(config, mock_service_manager)
        controller._wait_for_all_configured = AsyncMock(return_value=None)
        controller._wait_for_all_workers_ready = AsyncMock(return_value=None)
        controller._start_profiling_all_services = AsyncMock(return_value=None)
        controller._wait_for_endpoint_ready = AsyncMock(return_value=None)

        await controller._start_services()

        controller._wait_for_all_workers_ready.assert_awaited_once_with(
            timeout=Environment.SERVICE.PROFILE_START_TIMEOUT,
        )
        controller._start_profiling_all_services.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_configure_single_worker_marks_worker_ready(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """Successful worker configure should mark the worker READY for the barrier."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.workers = 1
        config.runtime.workers_per_pod = 1
        controller, _ = self._create_system_controller(config, mock_service_manager)
        ServiceRegistry.expect_service("worker_a", ServiceType.WORKER)
        ServiceRegistry.register(
            service_id="worker_a",
            service_type=ServiceType.WORKER,
            first_seen_ns=1,
            state=LifecycleState.RUNNING,
        )
        controller._send_control_command = AsyncMock(return_value=None)

        await controller._configure_single_service("worker_a")

        assert controller._worker_startup_states["worker_a"] == str(
            WorkerStartupState.READY
        )
        assert controller._all_expected_workers_ready() is True

    @pytest.mark.asyncio
    async def test_wait_for_all_workers_ready_succeeds_when_expected_workers_ready(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """Worker READY barrier should pass once all expected workers report READY."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.workers = 2
        config.runtime.workers_per_pod = 2
        controller, _ = self._create_system_controller(config, mock_service_manager)
        controller._worker_startup_states = {
            "worker_a": str(WorkerStartupState.READY),
            "worker_b": str(WorkerStartupState.READY),
        }

        await controller._wait_for_all_workers_ready(timeout=1.0)

    @pytest.mark.asyncio
    async def test_wait_for_all_workers_ready_times_out_when_workers_not_ready(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """Worker READY barrier should time out when expected workers never reach READY."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.workers = 2
        config.runtime.workers_per_pod = 2
        controller, _ = self._create_system_controller(config, mock_service_manager)
        controller._worker_startup_states = {
            "worker_a": str(WorkerStartupState.WAITING_FOR_DATASET),
            "worker_b": str(WorkerStartupState.ROUTER_PROBING),
        }

        with pytest.raises(
            ServiceRegistrationTimeoutError, match="workers to become READY"
        ):
            await controller._wait_for_all_workers_ready(timeout=0.01)

    def test_kubernetes_mode_disables_raw_inference_proxy(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """In K8s mode, raw inference proxy is disabled (worker pods run it locally)."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        _, mock_proxy_cls = self._create_system_controller(config, mock_service_manager)
        call_kwargs = mock_proxy_cls.call_args[1]
        assert call_kwargs["enable_event_bus"] is True
        assert call_kwargs["enable_dataset_manager"] is True
        assert call_kwargs["enable_raw_inference"] is False

    def test_multiprocessing_mode_enables_all_proxies(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """In local mode, all proxies including raw inference are enabled."""
        config.runtime.service_run_type = ServiceRunType.MULTIPROCESSING
        _, mock_proxy_cls = self._create_system_controller(config, mock_service_manager)
        call_kwargs = mock_proxy_cls.call_args[1]
        assert call_kwargs["enable_event_bus"] is True
        assert call_kwargs["enable_dataset_manager"] is True
        assert call_kwargs["enable_raw_inference"] is True

    @pytest.mark.asyncio
    async def test_wpm_registration_logs_capacity(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """WPM Registration with capacity fields logs pod capacity info."""
        from aiperf.common.service_registry import ServiceRegistry

        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.workers = 8
        config.runtime.workers_per_pod = 4
        config.runtime.record_processors_per_pod = 1
        controller, _ = self._create_system_controller(config, mock_service_manager)

        msg = Registration(
            sid="wpm_pod0",
            rid="r1",
            stype="worker_pod_manager",
            state="running",
            num_workers=4,
            num_record_processors=1,
        )
        result = await controller._handle_control_message("identity_0", msg)

        assert isinstance(result, RegistrationAck)
        assert ServiceRegistry.is_registered("wpm_pod0")

    @pytest.mark.asyncio
    async def test_wpm_registration_warns_on_capacity_mismatch(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """WPM registrations that disagree with configured pod capacity should warn."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        config.runtime.workers = 8
        config.runtime.workers_per_pod = 4
        config.runtime.record_processors_per_pod = 1
        controller, _ = self._create_system_controller(config, mock_service_manager)

        msg = Registration(
            sid="wpm_pod0",
            rid="r1",
            stype="worker_pod_manager",
            state="running",
            num_workers=3,
            num_record_processors=2,
        )

        with patch.object(controller, "warning") as mock_warning:
            result = await controller._handle_control_message("identity_0", msg)

        assert isinstance(result, RegistrationAck)
        mock_warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_registration_without_capacity_does_not_modify_expectations(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """Regular service Registration (no capacity fields) should not change expectations."""
        from aiperf.common.service_registry import ServiceRegistry

        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        controller, _ = self._create_system_controller(config, mock_service_manager)

        workers_before = ServiceRegistry.expected_by_type.get(ServiceType.WORKER, 0)
        rps_before = ServiceRegistry.expected_by_type.get(
            ServiceType.RECORD_PROCESSOR, 0
        )

        msg = Registration(
            sid="timing_manager_0",
            rid="r1",
            stype="timing_manager",
            state="running",
        )
        await controller._handle_control_message("id_0", msg)

        assert (
            ServiceRegistry.expected_by_type.get(ServiceType.WORKER, 0)
            == workers_before
        )
        assert (
            ServiceRegistry.expected_by_type.get(ServiceType.RECORD_PROCESSOR, 0)
            == rps_before
        )

    @pytest.mark.asyncio
    async def test_worker_status_summary_tracks_pending_startup_states(
        self,
        config: AIPerfConfig,
        mock_service_manager: AsyncMock,
    ) -> None:
        """Worker status summaries should surface pending worker startup phases."""
        config.runtime.service_run_type = ServiceRunType.KUBERNETES
        controller, _ = self._create_system_controller(config, mock_service_manager)

        await controller._on_worker_status_summary(
            WorkerStatusSummaryMessage(
                service_id="worker-manager",
                worker_statuses={
                    "worker_a": WorkerStatus.IDLE,
                    "worker_b": WorkerStatus.IDLE,
                },
                worker_startup_states={
                    "worker_a": WorkerStartupState.WAITING_FOR_DATASET,
                    "worker_b": WorkerStartupState.ROUTER_PROBING,
                },
            )
        )

        assert controller._summarize_pending_worker_startup_states(
            {"worker_a", "worker_b", "worker_c"}
        ) == {
            "waiting_for_dataset": 1,
            "router_probing": 1,
        }


class TestSSLVerificationWarning:
    """Test SSL verification warning in SystemController."""

    @pytest.mark.asyncio
    async def test_warning_logged_when_ssl_verify_disabled(
        self, system_controller: SystemController, monkeypatch
    ):
        """Test that a warning is logged when SSL verification is disabled."""
        monkeypatch.setattr(Environment.HTTP, "SSL_VERIFY", False)

        with (
            patch.object(
                system_controller, "_all_expected_configured", return_value=True
            ),
            patch.object(system_controller, "warning") as mock_warning,
        ):
            await system_controller._wait_for_all_configured(timeout=5.0)

            mock_warning.assert_called_once()
            warning_message = mock_warning.call_args[0][0]
            assert "SSL certificate verification is DISABLED" in warning_message

    @pytest.mark.asyncio
    async def test_no_warning_logged_when_ssl_verify_enabled(
        self, system_controller: SystemController, monkeypatch
    ):
        """Test that no warning is logged when SSL verification is enabled."""
        monkeypatch.setattr(Environment.HTTP, "SSL_VERIFY", True)

        with (
            patch.object(
                system_controller, "_all_expected_configured", return_value=True
            ),
            patch.object(system_controller, "warning") as mock_warning,
        ):
            await system_controller._wait_for_all_configured(timeout=5.0)

            mock_warning.assert_not_called()

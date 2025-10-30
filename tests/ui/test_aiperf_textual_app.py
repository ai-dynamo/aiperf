# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aiperf.common.enums import GPUTelemetryMode
from aiperf.common.messages import StartRealtimeTelemetryCommand
from aiperf.common.models import MetricResult
from aiperf.ui.dashboard.aiperf_textual_app import AIPerfTextualApp


class TestAIPerfTextualAppInitialization:
    """Test AIPerfTextualApp initialization."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller."""
        controller = MagicMock()
        controller.service_id = "test_controller"
        controller.user_config = MagicMock()
        controller.user_config.gpu_telemetry_mode = GPUTelemetryMode.SUMMARY
        return controller

    @pytest.fixture
    def app(self, service_config, mock_controller):
        """Create AIPerfTextualApp instance."""
        return AIPerfTextualApp(
            service_config=service_config, controller=mock_controller
        )

    def test_init_sets_title(self, app):
        """Test that initialization sets the correct title."""
        assert app.title == "NVIDIA AIPerf"

    def test_init_widget_references_none(self, app):
        """Test that widget references start as None."""
        assert app.log_viewer is None
        assert app.progress_dashboard is None
        assert app.progress_header is None
        assert app.worker_dashboard is None
        assert app.realtime_metrics_dashboard is None
        assert app.realtime_telemetry_dashboard is None


class TestAIPerfTextualAppActions:
    """Test action handlers."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller."""
        controller = AsyncMock()
        controller.service_id = "test_controller"
        controller.user_config = MagicMock()
        controller.user_config.gpu_telemetry_mode = GPUTelemetryMode.SUMMARY
        controller.publish = AsyncMock()
        return controller

    @pytest.fixture
    def app(self, service_config, mock_controller):
        """Create AIPerfTextualApp instance."""
        return AIPerfTextualApp(
            service_config=service_config, controller=mock_controller
        )

    @pytest.mark.asyncio
    async def test_action_quit_cleanup(self, app):
        """Test that action_quit clears widget references and signals."""
        app.worker_dashboard = Mock()
        app.progress_dashboard = Mock()
        app.progress_header = Mock()
        app.realtime_metrics_dashboard = Mock()
        app.log_viewer = Mock()

        with (
            patch("os.kill") as mock_kill,
            patch("os.getpid", return_value=12345),
        ):
            await app.action_quit()

            assert app.worker_dashboard is None
            assert app.progress_dashboard is None
            assert app.progress_header is None
            assert app.realtime_metrics_dashboard is None
            assert app.log_viewer is None

            mock_kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_toggle_hide_log_viewer(self, app):
        """Test that toggle_hide_log_viewer toggles the hidden class."""
        mock_logs_section = Mock()

        with patch.object(app, "query_one", return_value=mock_logs_section):
            await app.action_toggle_hide_log_viewer()

            mock_logs_section.toggle_class.assert_called_once_with("hidden")

    @pytest.mark.asyncio
    async def test_action_restore_all_panels(self, app):
        """Test that restore_all_panels minimizes screen and unhides logs."""
        mock_screen = Mock()
        mock_logs_section = Mock()

        with (
            patch.object(
                type(app),
                "screen",
                new_callable=lambda: property(lambda self: mock_screen),
            ),
            patch.object(app, "query_one", return_value=mock_logs_section),
        ):
            await app.action_restore_all_panels()

            mock_screen.minimize.assert_called_once()
            mock_logs_section.remove_class.assert_called_once_with("hidden")

    @pytest.mark.asyncio
    async def test_action_minimize_all_panels(self, app):
        """Test that minimize_all_panels minimizes the screen."""
        mock_screen = Mock()

        with patch.object(
            type(app), "screen", new_callable=lambda: property(lambda self: mock_screen)
        ):
            await app.action_minimize_all_panels()

            mock_screen.minimize.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_toggle_maximize_regular_panel(self, app):
        """Test toggle_maximize for a regular panel."""
        mock_screen = Mock()
        mock_panel = Mock()
        mock_panel.is_maximized = False

        with (
            patch.object(
                type(app),
                "screen",
                new_callable=lambda: property(lambda self: mock_screen),
            ),
            patch.object(app, "query_one", return_value=mock_panel),
        ):
            await app.action_toggle_maximize("progress")

            mock_screen.maximize.assert_called_once_with(mock_panel)

    @pytest.mark.asyncio
    async def test_action_toggle_maximize_minimizes_when_maximized(self, app):
        """Test toggle_maximize minimizes when panel is already maximized."""
        mock_screen = Mock()
        mock_panel = Mock()
        mock_panel.is_maximized = True

        with (
            patch.object(
                type(app),
                "screen",
                new_callable=lambda: property(lambda self: mock_screen),
            ),
            patch.object(app, "query_one", return_value=mock_panel),
        ):
            await app.action_toggle_maximize("progress")

            mock_screen.minimize.assert_called_once()
            mock_screen.maximize.assert_not_called()

    @pytest.mark.asyncio
    async def test_action_toggle_maximize_telemetry_enables_mode(
        self, app, mock_controller
    ):
        """Test that toggling telemetry panel enables telemetry and publishes command."""
        mock_screen = Mock()
        mock_panel = Mock()
        mock_panel.is_maximized = False
        app.realtime_telemetry_dashboard = Mock()

        with (
            patch.object(
                type(app),
                "screen",
                new_callable=lambda: property(lambda self: mock_screen),
            ),
            patch.object(app, "query_one", return_value=mock_panel),
        ):
            await app.action_toggle_maximize_telemetry()

            assert (
                mock_controller.user_config.gpu_telemetry_mode
                == GPUTelemetryMode.REALTIME_DASHBOARD
            )
            app.realtime_telemetry_dashboard.set_status_message.assert_called_once_with(
                "Enabling live GPU telemetry..."
            )
            mock_controller.publish.assert_called_once()

            call_args = mock_controller.publish.call_args[0][0]
            assert isinstance(call_args, StartRealtimeTelemetryCommand)


class TestAIPerfTextualAppProgressHandlers:
    """Test progress update handlers."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller."""
        controller = MagicMock()
        controller.service_id = "test_controller"
        controller.user_config = MagicMock()
        return controller

    @pytest.fixture
    def app(self, service_config, mock_controller):
        """Create AIPerfTextualApp instance."""
        return AIPerfTextualApp(
            service_config=service_config, controller=mock_controller
        )

    @pytest.mark.asyncio
    async def test_on_warmup_progress(self, app):
        """Test on_warmup_progress updates dashboard and header."""
        app.progress_dashboard = Mock()
        app.progress_dashboard.batch = MagicMock()
        app.progress_header = Mock()
        app._has_result_data = True
        mock_section = Mock()

        warmup_stats = Mock()
        warmup_stats.finished = 50
        warmup_stats.total_expected_requests = 100

        with patch.object(app, "query_one", return_value=mock_section):
            await app.on_warmup_progress(warmup_stats)

            app.progress_dashboard.on_warmup_progress.assert_called_once_with(
                warmup_stats
            )
            app.progress_header.update_progress.assert_called_once_with(
                header="Warmup", progress=50, total=100
            )

    @pytest.mark.asyncio
    async def test_on_profiling_progress(self, app):
        """Test on_profiling_progress updates dashboard and header."""
        app.progress_dashboard = Mock()
        app.progress_dashboard.batch = MagicMock()
        app.progress_header = Mock()
        app._has_result_data = True
        mock_section = Mock()

        profiling_stats = Mock()
        profiling_stats.finished = 75
        profiling_stats.total_expected_requests = 150

        with patch.object(app, "query_one", return_value=mock_section):
            await app.on_profiling_progress(profiling_stats)

            app.progress_dashboard.on_profiling_progress.assert_called_once_with(
                profiling_stats
            )
            app.progress_header.update_progress.assert_called_once_with(
                header="Profiling", progress=75, total=150
            )

    @pytest.mark.asyncio
    async def test_on_records_progress(self, app):
        """Test on_records_progress updates dashboard."""
        app.progress_dashboard = Mock()
        app.progress_dashboard.batch = MagicMock()
        app.progress_header = Mock()
        app._profiling_stats = Mock()
        app._profiling_stats.finished = 100
        app._profiling_stats.total_expected_requests = 100
        app._profiling_stats.is_complete = True

        records_stats = Mock()

        await app.on_records_progress(records_stats)

        app.progress_dashboard.on_records_progress.assert_called_once_with(
            records_stats
        )
        app.progress_header.update_progress.assert_called_once_with(
            header="Records", progress=100, total=100
        )


class TestAIPerfTextualAppMetricsHandlers:
    """Test metrics and telemetry handlers."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller."""
        controller = MagicMock()
        controller.service_id = "test_controller"
        controller.user_config = MagicMock()
        return controller

    @pytest.fixture
    def app(self, service_config, mock_controller):
        """Create AIPerfTextualApp instance."""
        return AIPerfTextualApp(
            service_config=service_config, controller=mock_controller
        )

    @pytest.mark.asyncio
    async def test_on_worker_update(self, app):
        """Test on_worker_update forwards to worker_dashboard."""
        app.worker_dashboard = Mock()
        app.worker_dashboard.batch = MagicMock()

        worker_stats = Mock()

        await app.on_worker_update("worker1", worker_stats)

        app.worker_dashboard.on_worker_update.assert_called_once_with(
            "worker1", worker_stats
        )

    @pytest.mark.asyncio
    async def test_on_realtime_metrics(self, app):
        """Test on_realtime_metrics forwards to realtime_metrics_dashboard."""
        app.realtime_metrics_dashboard = Mock()
        app.realtime_metrics_dashboard.batch = MagicMock()

        metrics = [
            MetricResult(tag="test_metric", header="Test Metric", unit="ms", avg=10.5)
        ]

        await app.on_realtime_metrics(metrics)

        app.realtime_metrics_dashboard.on_realtime_metrics.assert_called_once_with(
            metrics
        )

    @pytest.mark.asyncio
    async def test_on_realtime_telemetry_metrics(self, app):
        """Test on_realtime_telemetry_metrics forwards to realtime_telemetry_dashboard."""
        app.realtime_telemetry_dashboard = Mock()
        app.realtime_telemetry_dashboard.batch = MagicMock()

        metrics = [
            MetricResult(
                tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                unit="%",
                avg=75.0,
            )
        ]

        await app.on_realtime_telemetry_metrics(metrics)

        app.realtime_telemetry_dashboard.on_realtime_telemetry_metrics.assert_called_once_with(
            metrics
        )

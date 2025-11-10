# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.ui.dashboard.aiperf_dashboard_ui import AIPerfDashboardUI


class TestAIPerfDashboardUIInitialization:
    """Test AIPerfDashboardUI initialization."""

    @pytest.fixture
    def mock_dependencies(self, service_config, user_config):
        """Create mocked dependencies for AIPerfDashboardUI."""
        mock_log_queue = Mock()
        mock_controller = MagicMock()
        mock_controller.service_id = "test_controller"

        return {
            "log_queue": mock_log_queue,
            "service_config": service_config,
            "user_config": user_config,
            "controller": mock_controller,
        }

    @pytest.fixture
    def dashboard_ui(self, mock_dependencies):
        """Create AIPerfDashboardUI instance with mocked dependencies."""
        with (
            patch(
                "aiperf.ui.dashboard.aiperf_dashboard_ui.AIPerfTextualApp"
            ) as mock_app_class,
            patch(
                "aiperf.ui.dashboard.aiperf_dashboard_ui.LogConsumer"
            ) as mock_consumer_class,
        ):
            mock_app_instance = Mock()
            mock_app_class.return_value = mock_app_instance

            mock_consumer_instance = Mock()
            mock_consumer_class.return_value = mock_consumer_instance

            dashboard = AIPerfDashboardUI(**mock_dependencies)

            dashboard.mock_app = mock_app_instance
            dashboard.mock_consumer = mock_consumer_instance

            return dashboard

    def test_init_creates_log_consumer(self, mock_dependencies):
        """Test that initialization creates LogConsumer with log_queue."""
        with (
            patch(
                "aiperf.ui.dashboard.aiperf_dashboard_ui.AIPerfTextualApp"
            ) as mock_app_class,
            patch(
                "aiperf.ui.dashboard.aiperf_dashboard_ui.LogConsumer"
            ) as mock_consumer_class,
        ):
            mock_app_instance = Mock()
            mock_app_class.return_value = mock_app_instance

            AIPerfDashboardUI(**mock_dependencies)

            mock_consumer_class.assert_called_once()
            call_kwargs = mock_consumer_class.call_args[1]
            assert call_kwargs["log_queue"] == mock_dependencies["log_queue"]
            assert call_kwargs["app"] == mock_app_instance

    def test_init_attaches_child_lifecycle(self, dashboard_ui):
        """Test that log consumer lifecycle is attached."""
        assert dashboard_ui.log_consumer == dashboard_ui.mock_consumer

    def test_init_attaches_all_hooks(self, dashboard_ui):
        """Test that all required hooks are attached correctly to the app."""
        assert hasattr(dashboard_ui.app, "on_records_progress")
        assert hasattr(dashboard_ui.app, "on_profiling_progress")
        assert hasattr(dashboard_ui.app, "on_warmup_progress")
        assert hasattr(dashboard_ui.app, "on_worker_update")
        assert hasattr(dashboard_ui.app, "on_worker_status_summary")
        assert hasattr(dashboard_ui.app, "on_realtime_metrics")
        assert hasattr(dashboard_ui.app, "on_realtime_telemetry_metrics")

    def test_init_stores_references(self, dashboard_ui, mock_dependencies):
        """Test that controller and service_config are stored."""
        assert dashboard_ui.controller == mock_dependencies["controller"]
        assert dashboard_ui.service_config == mock_dependencies["service_config"]

    def test_init_creates_textual_app(self, mock_dependencies):
        """Test that AIPerfTextualApp is created with correct parameters."""
        with (
            patch(
                "aiperf.ui.dashboard.aiperf_dashboard_ui.AIPerfTextualApp"
            ) as mock_app_class,
            patch("aiperf.ui.dashboard.aiperf_dashboard_ui.LogConsumer"),
        ):
            AIPerfDashboardUI(**mock_dependencies)

            mock_app_class.assert_called_once()
            call_kwargs = mock_app_class.call_args[1]
            assert call_kwargs["service_config"] == mock_dependencies["service_config"]
            assert call_kwargs["controller"] == mock_dependencies["controller"]


class TestAIPerfDashboardUILifecycle:
    """Test lifecycle hooks."""

    @pytest.fixture
    def dashboard_ui(self, service_config, user_config):
        """Create AIPerfDashboardUI instance with mocked dependencies."""
        mock_log_queue = Mock()
        mock_controller = MagicMock()

        with (
            patch(
                "aiperf.ui.dashboard.aiperf_dashboard_ui.AIPerfTextualApp"
            ) as mock_app_class,
            patch("aiperf.ui.dashboard.aiperf_dashboard_ui.LogConsumer"),
        ):
            mock_app_instance = Mock()
            mock_app_class.return_value = mock_app_instance

            dashboard = AIPerfDashboardUI(
                log_queue=mock_log_queue,
                service_config=service_config,
                user_config=user_config,
                controller=mock_controller,
            )

            dashboard.mock_app = mock_app_instance

            return dashboard

    @pytest.mark.asyncio
    async def test_run_app_starts_textual(self, dashboard_ui):
        """Test that _run_app starts the Textual app via execute_async."""
        mock_coroutine = Mock()
        dashboard_ui.app.run_async.return_value = mock_coroutine

        with patch.object(dashboard_ui, "execute_async") as mock_execute:
            await dashboard_ui._run_app()

            mock_execute.assert_called_once_with(mock_coroutine)

    @pytest.mark.asyncio
    async def test_on_stop_exits_app(self, dashboard_ui):
        """Test that _on_stop calls app.exit with return_code=0."""
        await dashboard_ui._on_stop()

        dashboard_ui.app.exit.assert_called_once_with(return_code=0)

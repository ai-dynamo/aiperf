# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.enums import CommandType
from aiperf.common.messages import ProfileConfigureCommand, ProfileStartCommand
from aiperf.common.messages.server_metrics_messages import ServerMetricsRecordMessage
from aiperf.common.models import ErrorDetails
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.manager import ServerMetricsManager


@pytest.fixture
def user_config_with_endpoint() -> UserConfig:
    """Create UserConfig with inference endpoint."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat"],
        ),
    )


@pytest.fixture
def user_config_with_server_metrics_urls() -> UserConfig:
    """Create UserConfig with custom server metrics URLs."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat"],
        ),
        server_metrics=[
            "http://custom-endpoint:9400/metrics",
            "http://another-endpoint:8081",
        ],
    )


class TestServerMetricsManagerInitialization:
    """Test ServerMetricsManager initialization and endpoint discovery."""

    def test_initialization_basic(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test basic initialization with inference endpoint."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        assert manager._collectors == {}
        # Should include inference endpoint with /metrics appended
        assert manager._server_metrics_endpoints == [
            "http://localhost:8000/v1/chat/metrics"
        ]
        assert manager._collection_interval == 0.333  # SERVER_METRICS default (333ms)

    def test_endpoint_discovery_from_inference_url(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that inference endpoint port is discovered by default."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        # Should include inference port (localhost:8000) by default
        assert len(manager._server_metrics_endpoints) == 1
        assert "localhost:8000" in manager._server_metrics_endpoints[0]

    def test_custom_server_metrics_urls_added(
        self,
        service_config: ServiceConfig,
        user_config_with_server_metrics_urls: UserConfig,
    ):
        """Test that user-specified server metrics URLs are added to endpoint list."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_server_metrics_urls,
        )

        assert (
            "http://custom-endpoint:9400/metrics" in manager._server_metrics_endpoints
        )
        assert (
            "http://another-endpoint:8081/metrics" in manager._server_metrics_endpoints
        )

    def test_duplicate_urls_avoided(
        self,
        service_config: ServiceConfig,
        user_config_with_server_metrics_urls: UserConfig,
    ):
        """Test that duplicate URLs are deduplicated."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_server_metrics_urls,
        )

        endpoint_counts = {}
        for endpoint in manager._server_metrics_endpoints:
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1

        for count in endpoint_counts.values():
            assert count == 1


class TestProfileConfigureCommand:
    """Test profile configuration and endpoint reachability checking."""

    @pytest.mark.asyncio
    async def test_configure_with_reachable_endpoints(
        self,
        service_config: ServiceConfig,
        user_config_with_server_metrics_urls: UserConfig,
    ):
        """Test configuration when all endpoints are reachable."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_server_metrics_urls,
        )

        with patch(
            "aiperf.server_metrics.manager.ServerMetricsDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                ProfileConfigureCommand(
                    service_id=manager.id,
                    command=CommandType.PROFILE_CONFIGURE,
                    config={},
                )
            )

            assert len(manager._collectors) > 0

    @pytest.mark.asyncio
    async def test_configure_with_unreachable_endpoints(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test configuration when no endpoints are reachable."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        with patch(
            "aiperf.server_metrics.manager.ServerMetricsDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=False)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                ProfileConfigureCommand(
                    service_id=manager.id,
                    command=CommandType.PROFILE_CONFIGURE,
                    config={},
                )
            )

            assert len(manager._collectors) == 0

    @pytest.mark.asyncio
    async def test_configure_clears_existing_collectors(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that configuration clears previous collectors."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager._collectors["old_collector"] = AsyncMock()

        with patch(
            "aiperf.server_metrics.manager.ServerMetricsDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                ProfileConfigureCommand(
                    service_id=manager.id,
                    command=CommandType.PROFILE_CONFIGURE,
                    config={},
                )
            )

            assert "old_collector" not in manager._collectors


class TestProfileStartCommand:
    """Test profile start functionality."""

    @pytest.mark.asyncio
    async def test_start_initializes_and_starts_collectors(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that start command starts all collectors.

        Note: Collectors are initialized during configure phase, not start phase.
        This test only verifies that start() is called.
        """
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        mock_collector = AsyncMock()
        manager._collectors["http://localhost:8081/metrics"] = mock_collector

        await manager._on_start_profiling(
            ProfileStartCommand(
                service_id=manager.id, command=CommandType.PROFILE_START
            )
        )

        mock_collector.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_triggers_delayed_shutdown_when_no_collectors(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that start triggers delayed shutdown when no collectors available.

        When no endpoints are reachable, the manager should use delayed shutdown
        to allow the command response to be sent before stopping. This prevents
        timeout errors in the SystemController.
        """

        def close_coroutine(coro):
            coro.close()
            return MagicMock()

        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )
        manager._collectors = {}  # No collectors

        with patch(
            "asyncio.create_task", side_effect=close_coroutine
        ) as mock_create_task:
            await manager._on_start_profiling(
                ProfileStartCommand(
                    service_id=manager.id, command=CommandType.PROFILE_START
                )
            )

            # Verify delayed shutdown was scheduled via asyncio.create_task
            mock_create_task.assert_called_once()
            assert hasattr(manager, "_shutdown_task")

    @pytest.mark.asyncio
    async def test_start_handles_initialization_failure(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test start command handles collector initialization failures."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        mock_collector = AsyncMock()
        mock_collector.initialize.side_effect = Exception("Initialization failed")
        manager._collectors["http://localhost:8081/metrics"] = mock_collector

        await manager._on_start_profiling(
            ProfileStartCommand(
                service_id=manager.id, command=CommandType.PROFILE_START
            )
        )

    @pytest.mark.asyncio
    async def test_start_triggers_delayed_shutdown_when_all_collectors_fail(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that start triggers delayed shutdown when all collectors fail to start.

        When all collectors fail to start, the manager should use delayed shutdown
        to allow the command response to be sent before stopping.
        """

        def close_coroutine(coro):
            coro.close()
            return MagicMock()

        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        mock_collector = AsyncMock()
        mock_collector.start.side_effect = Exception("Start failed")
        manager._collectors["http://localhost:8081/metrics"] = mock_collector

        with patch(
            "asyncio.create_task", side_effect=close_coroutine
        ) as mock_create_task:
            await manager._on_start_profiling(
                ProfileStartCommand(
                    service_id=manager.id, command=CommandType.PROFILE_START
                )
            )

            # Verify delayed shutdown was scheduled via asyncio.create_task
            mock_create_task.assert_called_once()
            assert hasattr(manager, "_shutdown_task")


class TestManagerCallbackFunctionality:
    """Test callback handling for records and errors."""

    @pytest.mark.asyncio
    async def test_record_callback_sends_message(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that record callback sends ServerMetricsRecordMessage."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager.records_push_client.push = AsyncMock()

        test_record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        await manager._on_server_metrics_records([test_record], "test_collector")

        manager.records_push_client.push.assert_called_once()
        call_args = manager.records_push_client.push.call_args[0][0]
        assert isinstance(call_args, ServerMetricsRecordMessage)
        assert call_args.record == test_record

    @pytest.mark.asyncio
    async def test_error_callback_logs_error(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that error callback logs the error."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        test_error = ErrorDetails.from_exception(ValueError("Test error"))

        await manager._on_server_metrics_error(test_error, "test_collector")

    @pytest.mark.asyncio
    async def test_record_callback_handles_send_failure(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that record callback handles message send failures gracefully."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager.records_push_client.push = AsyncMock(
            side_effect=Exception("Send failed")
        )

        test_records = [
            ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={},
            )
        ]

        await manager._on_server_metrics_records(test_records, "test_collector")


class TestDisabledServerMetrics:
    """Test server metrics disabled scenarios."""

    @pytest.mark.asyncio
    async def test_configure_when_server_metrics_disabled(
        self,
        service_config: ServiceConfig,
    ):
        """Test configuration when server metrics are disabled via CLI flag."""
        user_config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                urls=["http://localhost:8000/v1/chat"],
            ),
            no_server_metrics=True,  # Disable server metrics
        )
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config,
        )

        manager.publish = AsyncMock()

        await manager._profile_configure_command(
            ProfileConfigureCommand(
                service_id=manager.id,
                command=CommandType.PROFILE_CONFIGURE,
                config={},
            )
        )

        # Should not create any collectors
        assert len(manager._collectors) == 0
        # Should publish disabled status
        manager.publish.assert_called_once()


class TestExceptionHandling:
    """Test exception handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_exception_during_reachability_check(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that exceptions during reachability check are handled."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        with patch(
            "aiperf.server_metrics.manager.ServerMetricsDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable.side_effect = Exception("Network error")
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                ProfileConfigureCommand(
                    service_id=manager.id,
                    command=CommandType.PROFILE_CONFIGURE,
                    config={},
                )
            )

            # Should handle exception and not add collector
            assert len(manager._collectors) == 0

    @pytest.mark.asyncio
    async def test_exception_during_baseline_capture(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that exceptions during baseline capture are logged but don't fail configuration."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        with patch(
            "aiperf.server_metrics.manager.ServerMetricsDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector.initialize = AsyncMock()
            mock_collector.collect_and_process_metrics.side_effect = Exception(
                "Baseline failed"
            )
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                ProfileConfigureCommand(
                    service_id=manager.id,
                    command=CommandType.PROFILE_CONFIGURE,
                    config={},
                )
            )

            # Collector should still be added despite baseline failure
            assert len(manager._collectors) > 0


class TestPartialStartup:
    """Test partial collector startup scenarios."""

    @pytest.mark.asyncio
    async def test_partial_collector_startup(
        self,
        service_config: ServiceConfig,
        user_config_with_server_metrics_urls: UserConfig,
    ):
        """Test scenario where some collectors start successfully and some fail."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_server_metrics_urls,
        )

        # Create 2 collectors: one succeeds, one fails
        mock_collector1 = AsyncMock()
        mock_collector1.start = AsyncMock()  # Succeeds

        mock_collector2 = AsyncMock()
        mock_collector2.start.side_effect = Exception("Start failed")  # Fails

        manager._collectors = {
            "endpoint1": mock_collector1,
            "endpoint2": mock_collector2,
        }

        await manager._on_start_profiling(
            ProfileStartCommand(
                service_id=manager.id, command=CommandType.PROFILE_START
            )
        )

        # Both should be called
        mock_collector1.start.assert_called_once()
        mock_collector2.start.assert_called_once()


class TestProfileCompleteAndCancel:
    """Test profile completion and cancellation scenarios."""

    @pytest.mark.asyncio
    async def test_profile_complete_triggers_final_scrape(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that profile complete triggers final metrics scrape."""
        from aiperf.common.messages import ProfileCompleteCommand

        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        mock_collector = AsyncMock()
        manager._collectors = {"endpoint1": mock_collector}

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )

        # Should call final scrape
        mock_collector.collect_and_process_metrics.assert_called_once()
        # Should stop collector after final scrape
        mock_collector.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_profile_complete_handles_final_scrape_failure(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that profile complete handles final scrape failures gracefully."""
        from aiperf.common.messages import ProfileCompleteCommand

        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        mock_collector = AsyncMock()
        mock_collector.collect_and_process_metrics.side_effect = Exception(
            "Final scrape failed"
        )
        manager._collectors = {"endpoint1": mock_collector}

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )

        # Should still stop collector even if final scrape fails
        mock_collector.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_profile_complete_when_already_stopped(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that profile complete is idempotent when collectors already stopped."""
        from aiperf.common.messages import ProfileCompleteCommand

        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager._collectors = {}  # Already stopped

        # Should not raise exception
        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )

    @pytest.mark.asyncio
    async def test_profile_cancel(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that profile cancel stops all collectors."""
        from aiperf.common.messages import ProfileCancelCommand

        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        mock_collector = AsyncMock()
        manager._collectors = {"endpoint1": mock_collector}

        await manager._handle_profile_cancel_command(
            ProfileCancelCommand(
                service_id=manager.id, command=CommandType.PROFILE_CANCEL
            )
        )

        mock_collector.stop.assert_called_once()


class TestLifecycleHooks:
    """Test lifecycle hook handlers."""

    @pytest.mark.asyncio
    async def test_on_stop_hook(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that on_stop hook stops all collectors."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        mock_collector = AsyncMock()
        manager._collectors = {"endpoint1": mock_collector}

        await manager._server_metrics_manager_stop()

        mock_collector.stop.assert_called_once()


class TestStopAllCollectors:
    """Test stopping all collectors."""

    @pytest.mark.asyncio
    async def test_stop_all_collectors_calls_stop(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that stop_all_collectors stops each collector."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        mock_collector1 = AsyncMock()
        mock_collector2 = AsyncMock()
        manager._collectors = {
            "endpoint1": mock_collector1,
            "endpoint2": mock_collector2,
        }

        await manager._stop_all_collectors()

        mock_collector1.stop.assert_called_once()
        mock_collector2.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all_collectors_handles_failure(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that stop_all_collectors handles failures gracefully."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        mock_collector = AsyncMock()
        mock_collector.stop.side_effect = Exception("Stop failed")
        manager._collectors = {"endpoint1": mock_collector}

        await manager._stop_all_collectors()

    @pytest.mark.asyncio
    async def test_stop_all_collectors_when_no_collectors(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that stop_all_collectors handles empty collectors dict."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager._collectors = {}

        # Should not raise exception
        await manager._stop_all_collectors()


class TestDelayedShutdown:
    """Test delayed shutdown functionality."""

    @pytest.mark.asyncio
    async def test_delayed_shutdown(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that delayed shutdown sleeps and then stops service."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager.stop = AsyncMock()

        with (
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch("asyncio.shield", new_callable=AsyncMock) as mock_shield,
        ):
            await manager._delayed_shutdown()

            # Should sleep before stopping
            mock_sleep.assert_called_once()
            # Should call stop with shield
            mock_shield.assert_called_once()


class TestCallbackEdgeCases:
    """Test callback edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_record_callback_with_empty_list(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that record callback handles empty record list."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager.records_push_client.push = AsyncMock()

        await manager._on_server_metrics_records([], "test_collector")

        # Should not push anything for empty list
        manager.records_push_client.push.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_callback_handles_send_failure(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that error callback handles message send failures gracefully."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager.records_push_client.push = AsyncMock(
            side_effect=Exception("Send failed")
        )

        test_error = ErrorDetails.from_exception(ValueError("Test error"))

        # Should not raise exception
        await manager._on_server_metrics_error(test_error, "test_collector")

    @pytest.mark.asyncio
    async def test_status_send_failure(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that status send failures are handled gracefully."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager.publish = AsyncMock(side_effect=Exception("Publish failed"))

        # Should not raise exception
        await manager._send_server_metrics_status(
            enabled=True,
            reason=None,
            endpoints_configured=[],
            endpoints_reachable=[],
        )

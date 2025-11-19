# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.enums import CommandType, EndpointType
from aiperf.common.messages import ProfileConfigureCommand, ProfileStartCommand
from aiperf.common.messages.server_metrics_messages import ServerMetricsRecordsMessage
from aiperf.common.models import ErrorDetails
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.server_metrics.server_metrics_manager import ServerMetricsManager


@pytest.fixture
def user_config_with_endpoint() -> UserConfig:
    """Create UserConfig with inference endpoint."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            url="http://localhost:8000/v1/chat",
        ),
    )


@pytest.fixture
def user_config_with_server_metrics_urls() -> UserConfig:
    """Create UserConfig with custom server metrics URLs."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            url="http://localhost:8000/v1/chat",
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
        assert len(manager._server_metrics_endpoints) > 0
        assert manager._collection_interval == 0.1

    def test_endpoint_discovery_from_inference_url(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that endpoints are discovered from inference URL hostname."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        assert any(
            "localhost" in endpoint for endpoint in manager._server_metrics_endpoints
        )

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
        user_config_with_endpoint: UserConfig,
    ):
        """Test configuration when all endpoints are reachable."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        with patch(
            "aiperf.server_metrics.server_metrics_manager.ServerMetricsDataCollector"
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
            "aiperf.server_metrics.server_metrics_manager.ServerMetricsDataCollector"
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
            "aiperf.server_metrics.server_metrics_manager.ServerMetricsDataCollector"
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
        """Test that start command initializes and starts all collectors."""
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

        mock_collector.initialize.assert_called_once()
        mock_collector.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_with_no_collectors(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test start command when no collectors configured."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        await manager._on_start_profiling(
            ProfileStartCommand(
                service_id=manager.id, command=CommandType.PROFILE_START
            )
        )

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


class TestCallbackFunctionality:
    """Test callback handling for records and errors."""

    @pytest.mark.asyncio
    async def test_record_callback_sends_message(
        self,
        service_config: ServiceConfig,
        user_config_with_endpoint: UserConfig,
    ):
        """Test that record callback sends ServerMetricsRecordsMessage."""
        manager = ServerMetricsManager(
            service_config=service_config,
            user_config=user_config_with_endpoint,
        )

        manager.records_push_client.push = AsyncMock()

        test_records = [
            ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={},
            )
        ]

        await manager._on_server_metrics_records(test_records, "test_collector")

        manager.records_push_client.push.assert_called_once()
        call_args = manager.records_push_client.push.call_args[0][0]
        assert isinstance(call_args, ServerMetricsRecordsMessage)
        assert call_args.records == test_records

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

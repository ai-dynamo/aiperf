# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType, CreditPhase
from aiperf.common.messages import (
    RealtimeServerMetricsMessage,
)
from aiperf.common.models import CreditPhaseStats, ErrorDetails
from aiperf.common.models._server_metrics_export import (
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
)
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.credit.messages import CreditPhaseStartMessage
from aiperf.plugin.enums import EndpointType, TimingMode
from aiperf.server_metrics.manager import ServerMetricsManager
from aiperf.timing.config import CreditPhaseConfig
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def cfg_with_endpoint() -> CLIConfig:
    """Create CLIConfig with inference endpoint."""
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        urls=["http://localhost:8000/v1/chat"],
    )


@pytest.fixture
def cfg_with_server_metrics_urls() -> CLIConfig:
    """Create CLIConfig with custom server metrics URLs."""
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        urls=["http://localhost:8000/v1/chat"],
        server_metrics=[
            "http://custom-endpoint:9400/metrics",
            "http://another-endpoint:8081",
        ],
    )


class TestManagerOwnedServerMetricsProcessors:
    def test_initializes_local_server_metrics_processors_and_accumulator(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        assert manager._processors
        assert manager._accumulator is not None
        assert manager._accumulator in manager._processors

    @pytest.mark.asyncio
    async def test_record_callback_fans_out_locally_without_records_manager_push(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        processor = AsyncMock()
        manager._processors = [processor]
        manager.records_push_client = MagicMock()
        manager.records_push_client.push = AsyncMock()
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8000/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        await manager._on_server_metrics_records(
            [record], "http://localhost:8000/metrics"
        )

        processor.process_server_metrics_record.assert_awaited_once_with(record)
        manager.records_push_client.push.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_record_callback_publishes_realtime_server_metrics_from_accumulator(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8000/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )
        summary = ServerMetricsEndpointSummary(
            endpoint_url="http://localhost:8000/metrics",
            info=ServerMetricsEndpointInfo(
                total_fetches=1,
                first_fetch_ns=1_000_000_000,
                last_fetch_ns=1_000_000_000,
                avg_fetch_latency_ms=5.0,
                unique_updates=1,
                first_update_ns=1_000_000_000,
                last_update_ns=1_000_000_000,
                duration_seconds=0.0,
                avg_update_interval_ms=0.0,
            ),
            metrics={},
        )
        endpoint_summaries = {"localhost:8000": summary}
        processor = AsyncMock()
        accumulator = MagicMock()
        accumulator.compute_endpoint_summaries.return_value = endpoint_summaries
        manager._processors = [processor]
        manager._accumulator = accumulator
        manager.publish = AsyncMock()
        manager._profiling_started = True
        manager._last_realtime_publish_ns = 0

        await manager._on_server_metrics_records(
            [record], "http://localhost:8000/metrics"
        )

        manager.publish.assert_awaited_once()
        message = manager.publish.await_args.args[0]
        assert isinstance(message, RealtimeServerMetricsMessage)
        assert message.endpoint_summaries == endpoint_summaries

    @pytest.mark.asyncio
    async def test_record_callback_skips_realtime_publish_before_profiling(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8000/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )
        processor = AsyncMock()
        accumulator = MagicMock()
        manager._processors = [processor]
        manager._accumulator = accumulator
        manager.publish = AsyncMock()

        await manager._on_server_metrics_records(
            [record], "http://localhost:8000/metrics"
        )

        accumulator.compute_endpoint_summaries.assert_not_called()
        manager.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_record_callback_isolates_realtime_publish_failure(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8000/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )
        summary = ServerMetricsEndpointSummary(
            endpoint_url="http://localhost:8000/metrics",
            info=ServerMetricsEndpointInfo(
                total_fetches=1,
                first_fetch_ns=1_000_000_000,
                last_fetch_ns=1_000_000_000,
                avg_fetch_latency_ms=5.0,
                unique_updates=1,
                first_update_ns=1_000_000_000,
                last_update_ns=1_000_000_000,
                duration_seconds=0.0,
                avg_update_interval_ms=0.0,
            ),
            metrics={},
        )
        processor = AsyncMock()
        accumulator = MagicMock()
        accumulator.compute_endpoint_summaries.return_value = {
            "localhost:8000": summary
        }
        manager._processors = [processor]
        manager._accumulator = accumulator
        manager.publish = AsyncMock(side_effect=RuntimeError("bus down"))
        manager._profiling_started = True
        manager._last_realtime_publish_ns = 0

        await manager._on_server_metrics_records(
            [record], "http://localhost:8000/metrics"
        )

        processor.process_server_metrics_record.assert_awaited_once_with(record)
        manager.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_profile_complete_scrapes_final_metrics_before_publishing_result(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        events: list[str] = []
        collector = AsyncMock()

        async def collect_and_process_metrics() -> None:
            events.append("final-scrape")

        async def publish_result(*, start_ns: int | None, end_ns: int | None) -> None:
            events.append("publish")

        collector.collect_and_process_metrics.side_effect = collect_and_process_metrics
        manager._collectors = {"http://localhost:8000/metrics": collector}
        manager._publish_server_metrics_result = publish_result

        await manager._handle_profile_complete_command(
            Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
        )

        assert events == ["final-scrape", "publish"]
        collector.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_profile_complete_scrapes_final_metrics_once(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        collector = AsyncMock()

        async def collect_and_process_metrics() -> None:
            await asyncio.sleep(0.01)

        collector.collect_and_process_metrics.side_effect = collect_and_process_metrics
        manager._collectors = {"http://localhost:8000/metrics": collector}

        with patch(
            "aiperf.server_metrics.result_publisher.publish_server_metrics_result",
            new_callable=AsyncMock,
        ) as publish_result:
            await asyncio.gather(
                manager._handle_profile_complete_command(
                    Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
                ),
                manager._handle_profile_complete_command(
                    Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
                ),
            )

        collector.collect_and_process_metrics.assert_awaited_once()
        collector.stop.assert_awaited_once()
        publish_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publish_failure_leaves_result_retryable(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        with patch(
            "aiperf.server_metrics.result_publisher.publish_server_metrics_result",
            new_callable=AsyncMock,
        ) as publish_result:
            publish_result.side_effect = [RuntimeError("publish failed"), None]

            with pytest.raises(RuntimeError, match="publish failed"):
                await manager._publish_server_metrics_result(start_ns=1, end_ns=2)

            assert manager._result_published is False

            await manager._publish_server_metrics_result(start_ns=1, end_ns=2)

        assert manager._result_published is True
        assert publish_result.await_count == 2

    def test_parse_profile_complete_window_treats_invalid_values_as_none(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        message = MagicMock(payload=b'{"start_ns":"1","end_ns":{"bad":true}}')

        assert manager._parse_profile_complete_window(message) == (None, None)

    @pytest.mark.asyncio
    async def test_profiling_phase_start_captures_boundary_metrics(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        collector = AsyncMock()
        manager._collectors = {"http://localhost:8000/metrics": collector}

        await manager._on_credit_phase_start(
            CreditPhaseStartMessage(
                service_id=manager.id,
                config=CreditPhaseConfig(
                    phase=CreditPhase.PROFILING,
                    timing_mode=TimingMode.REQUEST_RATE,
                ),
                stats=CreditPhaseStats(phase=CreditPhase.PROFILING),
            )
        )

        collector.collect_and_process_metrics.assert_awaited_once()


class TestServerMetricsManagerInitialization:
    """Test ServerMetricsManager initialization and endpoint discovery."""

    def test_initialization_basic(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test basic initialization with inference endpoint."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        assert manager._collectors == {}
        # Should include inference endpoint with /metrics appended
        assert manager._server_metrics_endpoints == [
            "http://localhost:8000/v1/chat/metrics"
        ]
        assert manager._collection_interval == 0.333  # SERVER_METRICS default (333ms)

    def test_endpoint_discovery_from_inference_url(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that inference endpoint port is discovered by default."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        # Should include inference port (localhost:8000) by default
        assert len(manager._server_metrics_endpoints) == 1
        assert "localhost:8000" in manager._server_metrics_endpoints[0]

    def test_custom_server_metrics_urls_added(
        self,
        cli_config: CLIConfig,
        cfg_with_server_metrics_urls: CLIConfig,
    ):
        """Test that user-specified server metrics URLs are added to endpoint list."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_server_metrics_urls),
        )

        assert (
            "http://custom-endpoint:9400/metrics" in manager._server_metrics_endpoints
        )
        assert (
            "http://another-endpoint:8081/metrics" in manager._server_metrics_endpoints
        )

    def test_duplicate_urls_avoided(
        self,
        cli_config: CLIConfig,
        cfg_with_server_metrics_urls: CLIConfig,
    ):
        """Test that duplicate URLs are deduplicated."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_server_metrics_urls),
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
        cli_config: CLIConfig,
        cfg_with_server_metrics_urls: CLIConfig,
    ):
        """Test configuration when all endpoints are reachable."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_server_metrics_urls),
        )

        with patch(
            "aiperf.server_metrics.manager.ServerMetricsDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

            assert len(manager._collectors) > 0

    @pytest.mark.asyncio
    async def test_configure_with_unreachable_endpoints(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test configuration when no endpoints are reachable."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        with patch(
            "aiperf.server_metrics.manager.ServerMetricsDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=False)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

            assert len(manager._collectors) == 0

    @pytest.mark.asyncio
    async def test_configure_clears_existing_collectors(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that configuration clears previous collectors."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        manager._collectors["old_collector"] = AsyncMock()

        with patch(
            "aiperf.server_metrics.manager.ServerMetricsDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

            assert "old_collector" not in manager._collectors


class TestProfileStartCommand:
    """Test profile start functionality."""

    @pytest.mark.asyncio
    async def test_start_initializes_and_starts_collectors(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that start command starts all collectors.

        Note: Collectors are initialized during configure phase, not start phase.
        This test only verifies that start() is called.
        """
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        mock_collector = AsyncMock()
        manager._collectors["http://localhost:8081/metrics"] = mock_collector

        await manager._on_start_profiling(
            Command(cid="test", cmd=CommandType.PROFILE_START)
        )

        mock_collector.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_triggers_delayed_shutdown_when_no_collectors(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
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
            run=make_run_from_cli(cfg_with_endpoint),
        )
        manager._collectors = {}  # No collectors

        with patch(
            "asyncio.create_task", side_effect=close_coroutine
        ) as mock_create_task:
            await manager._on_start_profiling(
                Command(cid="test", cmd=CommandType.PROFILE_START)
            )

            # Verify delayed shutdown was scheduled via asyncio.create_task
            mock_create_task.assert_called_once()
            assert hasattr(manager, "_shutdown_task")

    @pytest.mark.asyncio
    async def test_start_handles_initialization_failure(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test start command handles collector initialization failures."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        mock_collector = AsyncMock()
        mock_collector.initialize.side_effect = Exception("Initialization failed")
        manager._collectors["http://localhost:8081/metrics"] = mock_collector

        await manager._on_start_profiling(
            Command(cid="test", cmd=CommandType.PROFILE_START)
        )

    @pytest.mark.asyncio
    async def test_start_triggers_delayed_shutdown_when_all_collectors_fail(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that start triggers delayed shutdown when all collectors fail to start.

        When all collectors fail to start, the manager should use delayed shutdown
        to allow the command response to be sent before stopping.
        """

        def close_coroutine(coro):
            coro.close()
            return MagicMock()

        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        mock_collector = AsyncMock()
        mock_collector.start.side_effect = Exception("Start failed")
        manager._collectors["http://localhost:8081/metrics"] = mock_collector

        with patch(
            "asyncio.create_task", side_effect=close_coroutine
        ) as mock_create_task:
            await manager._on_start_profiling(
                Command(cid="test", cmd=CommandType.PROFILE_START)
            )

            # Verify delayed shutdown was scheduled via asyncio.create_task
            mock_create_task.assert_called_once()
            assert hasattr(manager, "_shutdown_task")


class TestManagerCallbackFunctionality:
    """Test callback handling for records and errors."""

    @pytest.mark.asyncio
    async def test_record_callback_fans_out_to_local_processors(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that record callback fans out records to local processors."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        first_processor = AsyncMock()
        second_processor = AsyncMock()
        manager._processors = [first_processor, second_processor]

        test_record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        await manager._on_server_metrics_records([test_record], "test_collector")

        first_processor.process_server_metrics_record.assert_awaited_once_with(
            test_record
        )
        second_processor.process_server_metrics_record.assert_awaited_once_with(
            test_record
        )

    @pytest.mark.asyncio
    async def test_error_callback_logs_error(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that error callback logs the error."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        test_error = ErrorDetails.from_exception(ValueError("Test error"))

        await manager._on_server_metrics_error(test_error, "test_collector")

    @pytest.mark.asyncio
    async def test_record_callback_handles_processor_failure(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that record callback tracks processor failures gracefully."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        processor = AsyncMock()
        processor.process_server_metrics_record.side_effect = RuntimeError(
            "processor failed"
        )
        manager._processors = [processor]

        test_records = [
            ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={},
            )
        ]

        await manager._on_server_metrics_records(test_records, "test_collector")

        processor.process_server_metrics_record.assert_awaited_once_with(
            test_records[0]
        )
        assert sum(manager._error_state.error_counts.values()) == 1
        recorded_error = next(iter(manager._error_state.error_counts))
        assert recorded_error.type == "RuntimeError"
        assert "processor failed" in recorded_error.message


class TestDisabledServerMetrics:
    """Test server metrics disabled scenarios."""

    @pytest.mark.asyncio
    async def test_configure_when_server_metrics_disabled(
        self,
        cli_config: CLIConfig,
    ):
        """Test configuration when server metrics are disabled via CLI flag."""
        cli_config = CLIConfig(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            urls=["http://localhost:8000/v1/chat"],
            no_server_metrics=True,  # Disable server metrics
        )
        manager = ServerMetricsManager(
            run=make_run_from_cli(cli_config),
        )

        manager.publish = AsyncMock()

        await manager._profile_configure_command(
            Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
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
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that exceptions during reachability check are handled."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        with patch(
            "aiperf.server_metrics.manager.ServerMetricsDataCollector"
        ) as mock_collector_class:
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable.side_effect = Exception("Network error")
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

            # Should handle exception and not add collector
            assert len(manager._collectors) == 0

    @pytest.mark.asyncio
    async def test_exception_during_baseline_capture(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that exceptions during baseline capture are logged but don't fail configuration."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
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
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

            # Collector should still be added despite baseline failure
            assert len(manager._collectors) > 0


class TestPartialStartup:
    """Test partial collector startup scenarios."""

    @pytest.mark.asyncio
    async def test_partial_collector_startup(
        self,
        cli_config: CLIConfig,
        cfg_with_server_metrics_urls: CLIConfig,
    ):
        """Test scenario where some collectors start successfully and some fail."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_server_metrics_urls),
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
            Command(cid="test", cmd=CommandType.PROFILE_START)
        )

        # Both should be called
        mock_collector1.start.assert_called_once()
        mock_collector2.start.assert_called_once()


class TestProfileCompleteAndCancel:
    """Test profile completion and cancellation scenarios."""

    @pytest.mark.asyncio
    async def test_profile_complete_triggers_final_scrape(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that profile complete triggers final metrics scrape."""

        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        mock_collector = AsyncMock()
        manager._collectors = {"endpoint1": mock_collector}

        await manager._handle_profile_complete_command(
            Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
        )

        # Should call final scrape
        mock_collector.collect_and_process_metrics.assert_called_once()
        # Should stop collector after final scrape
        mock_collector.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_profile_complete_handles_final_scrape_failure(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that profile complete handles final scrape failures gracefully."""

        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        mock_collector = AsyncMock()
        mock_collector.collect_and_process_metrics.side_effect = Exception(
            "Final scrape failed"
        )
        manager._collectors = {"endpoint1": mock_collector}

        await manager._handle_profile_complete_command(
            Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
        )

        # Should still stop collector even if final scrape fails
        mock_collector.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_profile_complete_when_already_stopped(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that profile complete is idempotent when collectors already stopped."""

        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        manager._collectors = {}  # Already stopped

        # Should not raise exception
        await manager._handle_profile_complete_command(
            Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
        )

    @pytest.mark.asyncio
    async def test_profile_cancel(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that profile cancel stops all collectors."""

        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        mock_collector = AsyncMock()
        manager._collectors = {"endpoint1": mock_collector}

        await manager._handle_profile_cancel_command(
            Command(cid="test", cmd=CommandType.PROFILE_CANCEL)
        )

        mock_collector.stop.assert_called_once()


class TestLifecycleHooks:
    """Test lifecycle hook handlers."""

    @pytest.mark.asyncio
    async def test_on_stop_hook(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that on_stop hook stops all collectors."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
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
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that stop_all_collectors stops each collector."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
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
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that stop_all_collectors handles failures gracefully."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        mock_collector = AsyncMock()
        mock_collector.stop.side_effect = Exception("Stop failed")
        manager._collectors = {"endpoint1": mock_collector}

        await manager._stop_all_collectors()

    @pytest.mark.asyncio
    async def test_stop_all_collectors_when_no_collectors(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that stop_all_collectors handles empty collectors dict."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        manager._collectors = {}

        # Should not raise exception
        await manager._stop_all_collectors()


class TestDelayedShutdown:
    """Test delayed shutdown functionality."""

    @pytest.mark.asyncio
    async def test_delayed_shutdown(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that delayed shutdown sleeps and then stops service."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
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
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that record callback skips processors for an empty record list."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        processor = AsyncMock()
        manager._processors = [processor]

        await manager._on_server_metrics_records([], "test_collector")

        processor.process_server_metrics_record.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_callback_records_error_locally(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that error callback records collection errors locally."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        test_error = ErrorDetails.from_exception(ValueError("Test error"))

        await manager._on_server_metrics_error(test_error, "test_collector")

        assert manager._error_state.error_counts[test_error] == 1

    @pytest.mark.asyncio
    async def test_status_send_failure(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that status send failures are handled gracefully."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        manager.publish = AsyncMock(side_effect=Exception("Publish failed"))

        # Should not raise exception
        await manager._send_server_metrics_status(
            enabled=True,
            reason=None,
            endpoints_configured=[],
            endpoints_reachable=[],
        )

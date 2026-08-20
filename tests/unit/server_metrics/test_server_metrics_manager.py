# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import BaselineKind, CommandType, CreditPhase
from aiperf.common.messages import (
    PhaseBaselineRequestMessage,
    ProfileConfigureCommand,
    ProfileStartCommand,
)
from aiperf.common.models import CreditPhaseStats, ErrorDetails
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseStartMessage,
)
from aiperf.plugin.enums import EndpointType, TimingMode
from aiperf.server_metrics.manager import (
    ServerMetricsManager,
    _ServerMetricsPhaseIdentity,
)
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
            ProfileStartCommand(
                service_id=manager.id, command=CommandType.PROFILE_START
            )
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
            ProfileStartCommand(
                service_id=manager.id, command=CommandType.PROFILE_START
            )
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
    async def test_record_callback_processes_locally(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Raw records stay local to the manager-owned processors."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        processor = AsyncMock()
        manager._processors = [processor]
        manager.publish = AsyncMock()

        test_record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        await manager._on_server_metrics_records([test_record], "test_collector")

        processor.process_record.assert_awaited_once_with(test_record)
        manager.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_record_callback_tags_active_phase(
        self,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that server metric records are tagged with the active phase."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        manager._active_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.WARMUP,
            phase_name="cache-prime",
            phase_kind="warmup",
        )
        processor = AsyncMock()
        manager._processors = [processor]

        test_record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        await manager._on_server_metrics_records([test_record], "test_collector")

        processed = processor.process_record.await_args.args[0]
        assert processed.benchmark_phase == CreditPhase.WARMUP
        assert processed.phase_name == "cache-prime"

    @pytest.mark.asyncio
    async def test_scoped_collect_tags_records_with_captured_phase(
        self,
        cfg_with_endpoint: CLIConfig,
    ):
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        manager._active_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.PROFILING,
            phase_name="main",
            phase_kind="profiling",
        )
        processor = AsyncMock()
        manager._processors = [processor]

        test_record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        class _Collector:
            async def collect_and_process_metrics(self):
                await manager._on_server_metrics_records(
                    [test_record], "test_collector"
                )

        await manager._collect_and_process_metrics_for_phase(
            _Collector(),
            _ServerMetricsPhaseIdentity(
                phase=CreditPhase.WARMUP,
                phase_name="cache-prime",
                phase_kind="warmup",
            ),
        )

        processed = processor.process_record.await_args.args[0]
        assert processed.benchmark_phase == CreditPhase.WARMUP
        assert processed.phase_name == "cache-prime"
        assert manager._active_phase is not None
        assert manager._active_phase.phase == CreditPhase.PROFILING

    @pytest.mark.asyncio
    async def test_collect_snapshot_tags_records_with_start_phase_after_phase_flip(
        self,
        cfg_with_endpoint: CLIConfig,
    ):
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        manager._active_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.WARMUP,
            phase_name="cache-prime",
            phase_kind="warmup",
        )
        processor = AsyncMock()
        manager._processors = [processor]

        test_record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={},
        )

        class _Collector:
            async def collect_and_process_metrics(self):
                manager._active_phase = _ServerMetricsPhaseIdentity(
                    phase=CreditPhase.PROFILING,
                    phase_name="main",
                    phase_kind="profiling",
                )
                await manager._on_server_metrics_records(
                    [test_record], "test_collector"
                )

        collector = _Collector()
        manager._attach_phase_scoped_collection(collector)

        await collector.collect_and_process_metrics()

        processed = processor.process_record.await_args.args[0]
        assert processed.benchmark_phase == CreditPhase.WARMUP
        assert processed.phase_name == "cache-prime"
        assert manager._active_phase is not None
        assert manager._active_phase.phase == CreditPhase.PROFILING

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
    async def test_record_callback_tracks_processor_failure(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """One local processor failure is counted without escaping callback."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        processor = AsyncMock()
        processor.process_record.side_effect = Exception("process failed")
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
        assert sum(manager._error_state.error_counts.values()) == 1


class TestPhaseTransitionRace:
    """Phase-tagging transitions must be compare-and-set: message handlers run
    as independent tasks, so CREDIT_PHASE_START(PROFILING) can interleave with
    the awaited warmup-final scrapes inside _on_credit_phase_complete."""

    @pytest.mark.asyncio
    async def test_profiling_start_during_warmup_final_scrape_survives(
        self,
        cfg_with_endpoint: CLIConfig,
    ):
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        manager._active_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.WARMUP, phase_name="warmup", phase_kind="warmup"
        )

        scrape_started = asyncio.Event()
        release_scrape = asyncio.Event()

        async def slow_scrape():
            scrape_started.set()
            await release_scrape.wait()

        mock_collector = MagicMock()
        mock_collector.collect_and_process_metrics = AsyncMock(side_effect=slow_scrape)
        manager._collectors = {"http://localhost:8000/metrics": mock_collector}

        complete_task = asyncio.create_task(
            manager._on_credit_phase_complete(
                CreditPhaseCompleteMessage(
                    service_id="timing-manager",
                    stats=CreditPhaseStats(phase=CreditPhase.WARMUP),
                )
            )
        )
        await scrape_started.wait()
        # Profiling starts while the warmup-final scrape is still awaited.
        await manager._on_credit_phase_start(
            CreditPhaseStartMessage(
                service_id="timing-manager",
                stats=CreditPhaseStats(phase=CreditPhase.PROFILING),
                config=CreditPhaseConfig(
                    phase=CreditPhase.PROFILING,
                    timing_mode=TimingMode.REQUEST_RATE,
                ),
            )
        )
        release_scrape.set()
        await complete_task

        assert manager._active_phase is not None
        assert manager._active_phase.phase == CreditPhase.PROFILING

    @pytest.mark.asyncio
    async def test_warmup_complete_without_race_clears_phase(
        self,
        cfg_with_endpoint: CLIConfig,
    ):
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        manager._active_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.WARMUP, phase_name="warmup", phase_kind="warmup"
        )
        manager._collectors = {"http://localhost:8000/metrics": AsyncMock()}

        await manager._on_credit_phase_complete(
            CreditPhaseCompleteMessage(
                service_id="timing-manager",
                stats=CreditPhaseStats(phase=CreditPhase.WARMUP),
            )
        )

        assert manager._active_phase is None

    @pytest.mark.asyncio
    async def test_profiling_start_during_start_baseline_scrape_survives(
        self,
        cfg_with_endpoint: CLIConfig,
    ):
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        scrape_started = asyncio.Event()
        release_scrape = asyncio.Event()

        async def slow_scrape():
            scrape_started.set()
            await release_scrape.wait()

        mock_collector = MagicMock()
        mock_collector.collect_and_process_metrics = AsyncMock(side_effect=slow_scrape)
        manager._collectors = {"http://localhost:8000/metrics": mock_collector}

        baseline_task = asyncio.create_task(
            manager.collect_baseline(
                PhaseBaselineRequestMessage(
                    service_id="timing-manager",
                    phase_id="phase-0",
                    phase_index=0,
                    profiling_index=0,
                    phase_name="first",
                    phase_kind="profiling",
                    kind=BaselineKind.START,
                )
            )
        )
        await scrape_started.wait()
        await manager._on_credit_phase_start(
            CreditPhaseStartMessage(
                service_id="timing-manager",
                stats=CreditPhaseStats(phase=CreditPhase.PROFILING),
                config=CreditPhaseConfig(
                    phase=CreditPhase.PROFILING,
                    timing_mode=TimingMode.REQUEST_RATE,
                ),
            )
        )
        release_scrape.set()
        await baseline_task

        assert manager._active_phase is not None
        assert manager._active_phase.phase == CreditPhase.PROFILING

    @pytest.mark.asyncio
    async def test_warmup_start_during_start_baseline_scrape_survives_kind_change(
        self,
        cfg_with_endpoint: CLIConfig,
    ):
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )
        manager._active_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.PROFILING,
            phase_name="profiling",
            phase_kind="profiling",
        )

        scrape_started = asyncio.Event()
        release_scrape = asyncio.Event()

        async def slow_scrape():
            scrape_started.set()
            await release_scrape.wait()

        mock_collector = MagicMock()
        mock_collector.collect_and_process_metrics = AsyncMock(side_effect=slow_scrape)
        manager._collectors = {"http://localhost:8000/metrics": mock_collector}

        baseline_task = asyncio.create_task(
            manager.collect_baseline(
                PhaseBaselineRequestMessage(
                    service_id="timing-manager",
                    phase_id="phase-1",
                    phase_index=1,
                    profiling_index=None,
                    phase_name="gap",
                    phase_kind="warmup",
                    kind=BaselineKind.START,
                )
            )
        )
        await scrape_started.wait()
        await manager._on_credit_phase_start(
            CreditPhaseStartMessage(
                service_id="timing-manager",
                stats=CreditPhaseStats(phase=CreditPhase.WARMUP),
                config=CreditPhaseConfig(
                    phase=CreditPhase.WARMUP,
                    timing_mode=TimingMode.REQUEST_RATE,
                ),
            )
        )
        release_scrape.set()
        await baseline_task

        assert manager._active_phase is not None
        assert manager._active_phase.phase == CreditPhase.WARMUP


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
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that profile complete triggers final metrics scrape."""
        from aiperf.common.messages import ProfileCompleteCommand

        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        mock_collector = AsyncMock()
        manager._collectors = {"endpoint1": mock_collector}
        manager.publish = AsyncMock()

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )

        # Should call final scrape
        mock_collector.collect_and_process_metrics.assert_called_once()
        # Should stop collector after final scrape
        mock_collector.stop.assert_called_once()
        published = manager.publish.await_args.args[0]
        assert published.message_type == "process_server_metrics_result"
        assert manager._result_published is True

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )
        manager.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_profile_complete_skips_scrape_after_trailing_warmup_phase(
        self,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        """A cooldown must not be reattributed to the preceding named profile."""
        from aiperf.common.messages import ProfileCompleteCommand

        manager = ServerMetricsManager(run=make_run_from_cli(cfg_with_endpoint))
        profiling = manager.run.cfg.phases[0].model_copy(
            update={"name": "profile-a", "kind": "profiling"}
        )
        cooldown = profiling.model_copy(update={"name": "cooldown", "kind": "warmup"})
        manager.run.cfg.phases = [profiling, cooldown]
        manager._last_profiling_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.PROFILING,
            phase_index=0,
            profiling_index=0,
            phase_name="profile-a",
            phase_kind="profiling",
        )

        collector = AsyncMock()
        manager._collectors = {"endpoint1": collector}
        manager.publish = AsyncMock()

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id,
                command=CommandType.PROFILE_COMPLETE,
                end_ns=1,
            )
        )

        collector.collect_and_process_metrics.assert_not_awaited()
        collector.stop.assert_awaited_once()
        manager.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_profile_complete_handles_final_scrape_failure(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that profile complete handles final scrape failures gracefully."""
        from aiperf.common.messages import ProfileCompleteCommand

        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
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
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that profile complete is idempotent when collectors already stopped."""
        from aiperf.common.messages import ProfileCompleteCommand

        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        manager._collectors = {}  # Already stopped

        # Should not raise exception
        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )

    @pytest.mark.asyncio
    async def test_profile_complete_flushes_local_jsonl_before_export(
        self,
        cfg_with_endpoint: CLIConfig,
    ) -> None:
        """The final result is the barrier for manager-owned raw artifacts."""
        from aiperf.common.messages import ProfileCompleteCommand

        manager = ServerMetricsManager(run=make_run_from_cli(cfg_with_endpoint))
        exporter = AsyncMock()
        accumulator = AsyncMock()
        accumulator.export_results.return_value = None
        manager._stream_exporters = [exporter]
        manager._accumulator = accumulator
        manager._collectors = {}
        manager.publish = AsyncMock()

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id,
                start_ns=100,
                end_ns=200,
                warmup_start_ns=10,
                warmup_end_ns=90,
            )
        )

        exporter.finalize.assert_awaited_once_with()
        context = accumulator.export_results.await_args.args[0]
        assert context.start_ns == 100
        assert context.end_ns == 200
        assert context.warmup_start_ns == 10
        assert context.warmup_end_ns == 90
        manager.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_profile_cancel(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Test that profile cancel stops all collectors."""
        from aiperf.common.messages import ProfileCancelCommand

        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        mock_collector = AsyncMock()
        manager._collectors = {"endpoint1": mock_collector}
        manager.publish = AsyncMock()

        await manager._handle_profile_cancel_command(
            ProfileCancelCommand(
                service_id=manager.id, command=CommandType.PROFILE_CANCEL
            )
        )

        mock_collector.stop.assert_called_once()
        assert manager._result_published is True


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
        """Test that record callback handles empty record list."""
        manager = ServerMetricsManager(
            run=make_run_from_cli(cfg_with_endpoint),
        )

        processor = AsyncMock()
        manager._processors = [processor]

        await manager._on_server_metrics_records([], "test_collector")

        processor.process_record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_error_callback_records_error_locally(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Collector failures are retained for the final local export."""
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


class TestWarmupPhaseCompleteScrape:
    """End-of-warmup scrape behavior on CREDIT_PHASE_COMPLETE."""

    def _make_manager(self, cfg_with_endpoint: CLIConfig) -> ServerMetricsManager:
        return ServerMetricsManager(run=make_run_from_cli(cfg_with_endpoint))

    def _phase_complete(self, phase: CreditPhase) -> CreditPhaseCompleteMessage:
        return CreditPhaseCompleteMessage(
            service_id="timing-manager",
            stats=CreditPhaseStats(phase=phase, start_ns=1_000_000_000),
        )

    @pytest.mark.asyncio
    async def test_warmup_complete_scrapes_all_collectors(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """Warmup completion triggers a final scrape of every collector."""
        manager = self._make_manager(cfg_with_endpoint)
        collector_a = MagicMock()
        collector_a.collect_and_process_metrics = AsyncMock()
        collector_b = MagicMock()
        collector_b.collect_and_process_metrics = AsyncMock()
        manager._collectors = {
            "http://a:8081/metrics": collector_a,
            "http://b:8081/metrics": collector_b,
        }
        manager._active_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.WARMUP, phase_name="warmup", phase_kind="warmup"
        )

        await manager._on_credit_phase_complete(
            self._phase_complete(CreditPhase.WARMUP)
        )

        collector_a.collect_and_process_metrics.assert_awaited_once()
        collector_b.collect_and_process_metrics.assert_awaited_once()
        # WARMUP is a non-profiling phase, so the active phase is retired.
        assert manager._active_phase is None

    @pytest.mark.asyncio
    async def test_warmup_complete_one_endpoint_failure_does_not_skip_rest(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """A failing endpoint scrape must not prevent the remaining scrapes."""
        manager = self._make_manager(cfg_with_endpoint)
        failing = MagicMock()
        failing.collect_and_process_metrics = AsyncMock(
            side_effect=ConnectionError("scrape failed")
        )
        healthy = MagicMock()
        healthy.collect_and_process_metrics = AsyncMock()
        manager._collectors = {
            "http://bad:8081/metrics": failing,
            "http://good:8081/metrics": healthy,
        }

        await manager._on_credit_phase_complete(
            self._phase_complete(CreditPhase.WARMUP)
        )

        failing.collect_and_process_metrics.assert_awaited_once()
        healthy.collect_and_process_metrics.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_warmup_complete_without_collectors_skips_scrape(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """No collectors -> no scrape attempt, phase still retired."""
        manager = self._make_manager(cfg_with_endpoint)
        manager._collectors = {}
        manager._active_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.WARMUP, phase_name="warmup", phase_kind="warmup"
        )

        await manager._on_credit_phase_complete(
            self._phase_complete(CreditPhase.WARMUP)
        )

        assert manager._active_phase is None

    @pytest.mark.asyncio
    async def test_profiling_complete_preserves_active_phase(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """PROFILE_COMPLETE owns the final profiling scrape; phase is kept."""
        manager = self._make_manager(cfg_with_endpoint)
        manager._collectors = {}
        manager._active_phase = _ServerMetricsPhaseIdentity(
            phase=CreditPhase.PROFILING,
            phase_name="profiling",
            phase_kind="profiling",
        )

        await manager._on_credit_phase_complete(
            self._phase_complete(CreditPhase.PROFILING)
        )

        assert manager._active_phase is not None
        assert manager._active_phase.phase == CreditPhase.PROFILING


class TestKubernetesDiscoveryIntegration:
    """Manager discovery honors mode, timeout, merge, and dedup semantics."""

    @pytest.mark.asyncio
    async def test_forced_discovery_preserves_custom_path_and_deduplicates(
        self, cfg_with_endpoint: CLIConfig
    ) -> None:
        manager = ServerMetricsManager(run=make_run_from_cli(cfg_with_endpoint))
        manager.run.cfg.server_metrics.discovery.mode = "kubernetes"
        with (
            patch(
                "aiperf.server_metrics.manager.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.manager.discover_kubernetes_endpoints",
                new=AsyncMock(return_value=["http://discovered:9090/vllm/stats"]),
            ) as discover,
        ):
            await manager._merge_discovered_endpoints()
            await manager._merge_discovered_endpoints()

        assert (
            manager._server_metrics_endpoints.count("http://discovered:9090/vllm/stats")
            == 1
        )
        assert "http://discovered:9090/vllm/stats/metrics" not in (
            manager._server_metrics_endpoints
        )
        discover.assert_awaited()

    @pytest.mark.asyncio
    async def test_auto_discovery_skips_outside_cluster(
        self, cfg_with_endpoint: CLIConfig
    ) -> None:
        manager = ServerMetricsManager(run=make_run_from_cli(cfg_with_endpoint))
        with (
            patch(
                "aiperf.server_metrics.manager.is_running_in_kubernetes",
                return_value=False,
            ),
            patch(
                "aiperf.server_metrics.manager.discover_kubernetes_endpoints",
                new_callable=AsyncMock,
            ) as discover,
        ):
            assert await manager._run_metrics_discovery() == []
        discover.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_discovery_timeout_degrades_to_explicit_endpoints(
        self, cfg_with_endpoint: CLIConfig
    ) -> None:
        manager = ServerMetricsManager(run=make_run_from_cli(cfg_with_endpoint))
        manager.run.cfg.server_metrics.discovery.mode = "kubernetes"
        manager.run.cfg.server_metrics.discovery.timeout_seconds = 0.001

        async def never_returns(**_: object) -> list[str]:
            await asyncio.Future()
            return []

        manager.warning = MagicMock()
        with (
            patch(
                "aiperf.server_metrics.manager.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.manager.discover_kubernetes_endpoints",
                new=never_returns,
            ),
        ):
            assert await manager._run_metrics_discovery() == []
        assert "timed out" in manager.warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_phase_start_tracks_active_phase(
        self,
        cli_config: CLIConfig,
        cfg_with_endpoint: CLIConfig,
    ):
        """CREDIT_PHASE_START updates the phase used to tag scrapes."""
        manager = ServerMetricsManager(run=make_run_from_cli(cfg_with_endpoint))
        assert manager._active_phase is None

        await manager._on_credit_phase_start(
            CreditPhaseStartMessage(
                service_id="timing-manager",
                stats=CreditPhaseStats(
                    phase=CreditPhase.PROFILING, start_ns=1_000_000_000
                ),
                config=CreditPhaseConfig(
                    phase=CreditPhase.PROFILING,
                    timing_mode=TimingMode.REQUEST_RATE,
                ),
            )
        )

        assert manager._active_phase is not None
        assert manager._active_phase.phase == CreditPhase.PROFILING

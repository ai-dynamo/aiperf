# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType, CreditPhase
from aiperf.common.models import CreditPhaseStats, ErrorDetails
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.credit.messages import CreditPhaseStartMessage
from aiperf.plugin.enums import TimingMode
from aiperf.server_metrics.manager import ServerMetricsManager
from aiperf.timing.config import CreditPhaseConfig

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    datasets={
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


def _make_run(config: AIPerfConfig) -> BenchmarkRun:
    """Wrap an AIPerfConfig in a BenchmarkRun."""
    from pathlib import Path

    return BenchmarkRun(
        benchmark_id="test",
        cfg=config,
        artifact_dir=Path("/tmp/test"),
    )


@pytest.fixture
def config_with_endpoint() -> BenchmarkRun:
    """Create BenchmarkRun with inference endpoint."""
    return _make_run(AIPerfConfig(**_BASE))


@pytest.fixture
def config_with_server_metrics_urls() -> BenchmarkRun:
    """Create BenchmarkRun with custom server metrics URLs."""
    return _make_run(
        AIPerfConfig(
            **_BASE,
            server_metrics={
                "urls": [
                    "http://custom-endpoint:9400/metrics",
                    "http://another-endpoint:8081",
                ],
            },
        )
    )


class TestServerMetricsManagerInitialization:
    """Test ServerMetricsManager initialization and endpoint discovery."""

    def test_initialization_basic(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test basic initialization with inference endpoint."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        assert manager._collectors == {}
        # Should include inference endpoint with /metrics appended
        assert manager._server_metrics_endpoints == [
            "http://localhost:8000/v1/chat/completions/metrics"
        ]
        assert manager._collection_interval == 0.333  # SERVER_METRICS default (333ms)

    def test_endpoint_discovery_from_inference_url(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that inference endpoint port is discovered by default."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        # Should include inference port (localhost:8000) by default
        assert len(manager._server_metrics_endpoints) == 1
        assert "localhost:8000" in manager._server_metrics_endpoints[0]

    def test_custom_server_metrics_urls_added(
        self,
        config_with_server_metrics_urls: BenchmarkRun,
    ):
        """Test that user-specified server metrics URLs are added to endpoint list."""
        manager = ServerMetricsManager(
            run=config_with_server_metrics_urls,
        )

        assert (
            "http://custom-endpoint:9400/metrics" in manager._server_metrics_endpoints
        )
        assert (
            "http://another-endpoint:8081/metrics" in manager._server_metrics_endpoints
        )

    def test_duplicate_urls_avoided(
        self,
        config_with_server_metrics_urls: BenchmarkRun,
    ):
        """Test that duplicate URLs are deduplicated."""
        manager = ServerMetricsManager(
            run=config_with_server_metrics_urls,
        )

        endpoint_counts = {}
        for endpoint in manager._server_metrics_endpoints:
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1

        for count in endpoint_counts.values():
            assert count == 1


class TestProfileConfigure:
    """Test profile configuration and endpoint reachability checking."""

    @pytest.mark.asyncio
    async def test_configure_with_reachable_endpoints(
        self,
        config_with_server_metrics_urls: BenchmarkRun,
    ):
        """Test configuration when all endpoints are reachable."""
        manager = ServerMetricsManager(
            run=config_with_server_metrics_urls,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test configuration when no endpoints are reachable."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that configuration clears previous collectors."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that start command starts all collectors.

        Note: Collectors are initialized during configure phase, not start phase.
        This test only verifies that start() is called.
        """
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
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
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test start command handles collector initialization failures."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that start triggers delayed shutdown when all collectors fail to start.

        When all collectors fail to start, the manager should use delayed shutdown
        to allow the command response to be sent before stopping.
        """

        def close_coroutine(coro):
            coro.close()
            return MagicMock()

        manager = ServerMetricsManager(
            run=config_with_endpoint,
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


class TestDisabledServerMetrics:
    """Test server metrics disabled scenarios."""

    @pytest.mark.asyncio
    async def test_configure_when_server_metrics_disabled(
        self,
    ):
        """Test configuration when server metrics are disabled via CLI flag."""
        config = AIPerfConfig(
            **_BASE,
            server_metrics={"enabled": False},
        )
        manager = ServerMetricsManager(
            run=_make_run(config),
        )

        manager.control_client = AsyncMock()

        await manager._profile_configure_command(
            Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
        )

        # Should not create any collectors
        assert len(manager._collectors) == 0
        # Should send disabled status via health check client
        manager.control_client.send.assert_called_once()


class TestExceptionHandling:
    """Test exception handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_exception_during_reachability_check(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that exceptions during reachability check are handled."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that exceptions during baseline capture are logged but don't fail configuration."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_server_metrics_urls: BenchmarkRun,
    ):
        """Test scenario where some collectors start successfully and some fail."""
        manager = ServerMetricsManager(
            run=config_with_server_metrics_urls,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that profile complete triggers final metrics scrape."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that profile complete handles final scrape failures gracefully."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that profile complete is idempotent when collectors already stopped."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        manager._collectors = {}  # Already stopped

        # Should not raise exception
        await manager._handle_profile_complete_command(
            Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
        )

    @pytest.mark.asyncio
    async def test_profile_cancel(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that profile cancel stops all collectors."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that on_stop hook stops all collectors."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that stop_all_collectors stops each collector."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that stop_all_collectors handles failures gracefully."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        mock_collector = AsyncMock()
        mock_collector.stop.side_effect = Exception("Stop failed")
        manager._collectors = {"endpoint1": mock_collector}

        await manager._stop_all_collectors()

    @pytest.mark.asyncio
    async def test_stop_all_collectors_when_no_collectors(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that stop_all_collectors handles empty collectors dict."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        manager._collectors = {}

        # Should not raise exception
        await manager._stop_all_collectors()


class TestDelayedShutdown:
    """Test delayed shutdown functionality."""

    @pytest.mark.asyncio
    async def test_delayed_shutdown(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that delayed shutdown sleeps and then stops service."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
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
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that record callback handles empty record list."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        # Empty list should be a no-op (no processor fan-out, no error tracking).
        await manager._on_server_metrics_records([], "test_collector")

        assert sum(manager._error_state.error_counts.values()) == 0

    @pytest.mark.asyncio
    async def test_error_callback_records_error(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that error callback records the error against local state."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        test_error = ErrorDetails.from_exception(ValueError("Test error"))

        # Should not raise exception
        await manager._on_server_metrics_error(test_error, "test_collector")

        assert sum(manager._error_state.error_counts.values()) == 1

    @pytest.mark.asyncio
    async def test_status_send_failure(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that status send failures are handled gracefully."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        manager.control_client = AsyncMock()
        manager.control_client.send = AsyncMock(side_effect=Exception("Publish failed"))

        # Should not raise exception
        await manager._send_server_metrics_status(
            enabled=True,
            reason=None,
            endpoints_configured=[],
            endpoints_reachable=[],
        )


class TestCreditPhaseStart:
    """Test boundary scrape at warmup→profiling phase transition."""

    @staticmethod
    def _make_phase_start_message(phase: CreditPhase) -> CreditPhaseStartMessage:
        """Create a CreditPhaseStartMessage with real Pydantic models."""
        return CreditPhaseStartMessage(
            service_id="timing_manager",
            stats=CreditPhaseStats(phase=phase),
            config=CreditPhaseConfig(
                phase=phase,
                timing_mode=TimingMode.REQUEST_RATE,
                total_expected_requests=100,
            ),
        )

    @pytest.mark.asyncio
    async def test_profiling_phase_triggers_boundary_scrape(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that PROFILING phase triggers scrape without stopping collectors."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        mock_collector = AsyncMock()
        manager._collectors = {"endpoint1": mock_collector}

        await manager._on_credit_phase_start(
            self._make_phase_start_message("profiling")
        )

        mock_collector.collect_and_process_metrics.assert_called_once()
        mock_collector.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_warmup_phase_does_not_trigger_scrape(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that WARMUP phase does not trigger a boundary scrape."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        mock_collector = AsyncMock()
        manager._collectors = {"endpoint1": mock_collector}

        await manager._on_credit_phase_start(self._make_phase_start_message("warmup"))

        mock_collector.collect_and_process_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_boundary_scrape_with_no_collectors(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that boundary scrape with empty collectors does not raise."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        manager._collectors = {}

        await manager._on_credit_phase_start(
            self._make_phase_start_message("profiling")
        )

    @pytest.mark.asyncio
    async def test_boundary_scrape_handles_collector_failure(
        self,
        config_with_endpoint: BenchmarkRun,
    ):
        """Test that one collector failure doesn't prevent scraping the other."""
        manager = ServerMetricsManager(
            run=config_with_endpoint,
        )

        mock_collector_ok = AsyncMock()
        mock_collector_fail = AsyncMock()
        mock_collector_fail.collect_and_process_metrics.side_effect = Exception(
            "Scrape failed"
        )
        manager._collectors = {
            "endpoint_fail": mock_collector_fail,
            "endpoint_ok": mock_collector_ok,
        }

        await manager._on_credit_phase_start(
            self._make_phase_start_message("profiling")
        )

        mock_collector_fail.collect_and_process_metrics.assert_called_once()
        mock_collector_ok.collect_and_process_metrics.assert_called_once()


class TestMetricsDiscovery:
    """Test _run_metrics_discovery integration with ServerMetricsManager."""

    @pytest.mark.asyncio
    async def test_disabled_mode_returns_empty(self):
        run = _make_run(
            AIPerfConfig(
                **_BASE,
                server_metrics={"discovery": {"mode": "disabled"}},
            )
        )
        manager = ServerMetricsManager(run=run)
        result = await manager._run_metrics_discovery()
        assert result == []

    @pytest.mark.asyncio
    async def test_kubernetes_mode_not_in_cluster(self):
        run = _make_run(
            AIPerfConfig(
                **_BASE,
                server_metrics={"discovery": {"mode": "kubernetes"}},
            )
        )
        manager = ServerMetricsManager(run=run)
        with patch(
            "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
            return_value=False,
        ):
            result = await manager._run_metrics_discovery()
        assert result == []

    @pytest.mark.asyncio
    async def test_kubernetes_mode_in_cluster(self):
        run = _make_run(
            AIPerfConfig(
                **_BASE,
                server_metrics={
                    "discovery": {
                        "mode": "kubernetes",
                        "namespace": "inference",
                        "label_selector": "app=vllm",
                    },
                },
            )
        )
        manager = ServerMetricsManager(run=run)
        with (
            patch(
                "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.endpoint_resolver.discover_kubernetes_endpoints",
                new_callable=AsyncMock,
                return_value=["http://pod-1:8080/metrics"],
            ) as mock_discover,
        ):
            result = await manager._run_metrics_discovery()

        assert result == ["http://pod-1:8080/metrics"]
        mock_discover.assert_called_once_with(
            namespace="inference",
            label_selector="app=vllm",
        )

    @pytest.mark.asyncio
    async def test_auto_mode_in_cluster(self):
        run = _make_run(AIPerfConfig(**_BASE))
        manager = ServerMetricsManager(run=run)
        with (
            patch(
                "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.endpoint_resolver.discover_kubernetes_endpoints",
                new_callable=AsyncMock,
                return_value=["http://auto:8080/metrics"],
            ),
        ):
            result = await manager._run_metrics_discovery()
        assert result == ["http://auto:8080/metrics"]

    @pytest.mark.asyncio
    async def test_auto_mode_in_cluster_derives_namespace_from_default_endpoint(self):
        run = _make_run(
            AIPerfConfig(
                **{
                    **_BASE,
                    "endpoint": {
                        "urls": [
                            "http://gateway.inference.svc.cluster.local/v1/chat/completions"
                        ]
                    },
                }
            )
        )
        manager = ServerMetricsManager(run=run)
        with (
            patch(
                "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.endpoint_resolver.discover_kubernetes_endpoints",
                new_callable=AsyncMock,
                return_value=["http://pod-1:8080/metrics"],
            ) as mock_discover,
        ):
            result = await manager._run_metrics_discovery()

        assert result == ["http://pod-1:8080/metrics"]
        mock_discover.assert_called_once_with(
            namespace="inference",
            label_selector=None,
        )

    @pytest.mark.asyncio
    async def test_auto_mode_not_in_cluster(self):
        run = _make_run(AIPerfConfig(**_BASE))
        manager = ServerMetricsManager(run=run)
        with patch(
            "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
            return_value=False,
        ):
            result = await manager._run_metrics_discovery()
        assert result == []

    @pytest.mark.asyncio
    async def test_discovery_exception_handled_gracefully(self):
        run = _make_run(
            AIPerfConfig(
                **_BASE,
                server_metrics={"discovery": {"mode": "kubernetes"}},
            )
        )
        manager = ServerMetricsManager(run=run)
        with (
            patch(
                "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.endpoint_resolver.discover_kubernetes_endpoints",
                new_callable=AsyncMock,
                side_effect=Exception("K8s API error"),
            ),
        ):
            result = await manager._run_metrics_discovery()
        assert result == []

    @pytest.mark.asyncio
    async def test_configure_integrates_discovered_endpoints(self):
        run = _make_run(
            AIPerfConfig(
                **_BASE,
                server_metrics={"discovery": {"mode": "kubernetes"}},
            )
        )
        manager = ServerMetricsManager(run=run)
        initial_count = len(manager._server_metrics_endpoints)

        with (
            patch(
                "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.endpoint_resolver.discover_kubernetes_endpoints",
                new_callable=AsyncMock,
                return_value=["http://discovered:8080/metrics"],
            ),
            patch(
                "aiperf.server_metrics.manager.ServerMetricsDataCollector"
            ) as mock_collector_class,
        ):
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

        assert initial_count == 1
        assert manager._server_metrics_endpoints == ["http://discovered:8080/metrics"]
        assert (
            "http://localhost:8000/v1/chat/completions/metrics"
            not in manager._server_metrics_endpoints
        )

    @pytest.mark.asyncio
    async def test_configure_auto_in_cluster_replaces_default_endpoint(self):
        run = _make_run(AIPerfConfig(**_BASE))
        manager = ServerMetricsManager(run=run)

        with (
            patch(
                "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.endpoint_resolver.discover_kubernetes_endpoints",
                new_callable=AsyncMock,
                return_value=["http://pod-1:8080/metrics"],
            ),
            patch(
                "aiperf.server_metrics.manager.ServerMetricsDataCollector"
            ) as mock_collector_class,
        ):
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

        assert manager._server_metrics_endpoints == ["http://pod-1:8080/metrics"]
        assert (
            "http://localhost:8000/v1/chat/completions/metrics"
            not in manager._server_metrics_endpoints
        )

    @pytest.mark.asyncio
    async def test_configure_deduplicates_discovered_endpoints(self):
        run = _make_run(
            AIPerfConfig(
                **_BASE,
                server_metrics={
                    "urls": ["http://already-there:8080/metrics"],
                    "discovery": {"mode": "kubernetes"},
                },
            )
        )
        manager = ServerMetricsManager(run=run)

        with (
            patch(
                "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.endpoint_resolver.discover_kubernetes_endpoints",
                new_callable=AsyncMock,
                return_value=["http://already-there:8080/metrics"],
            ),
            patch(
                "aiperf.server_metrics.manager.ServerMetricsDataCollector"
            ) as mock_collector_class,
        ):
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

        assert manager._server_metrics_endpoints == [
            "http://already-there:8080/metrics"
        ]

    @pytest.mark.asyncio
    async def test_configure_kubernetes_discovery_keeps_explicit_urls(self):
        run = _make_run(
            AIPerfConfig(
                **_BASE,
                server_metrics={
                    "urls": ["http://custom-endpoint:9400/metrics"],
                    "discovery": {"mode": "kubernetes"},
                },
            )
        )
        manager = ServerMetricsManager(run=run)

        with (
            patch(
                "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
                return_value=True,
            ),
            patch(
                "aiperf.server_metrics.endpoint_resolver.discover_kubernetes_endpoints",
                new_callable=AsyncMock,
                return_value=["http://pod-1:8080/metrics"],
            ),
            patch(
                "aiperf.server_metrics.manager.ServerMetricsDataCollector"
            ) as mock_collector_class,
        ):
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

        assert manager._server_metrics_endpoints == [
            "http://custom-endpoint:9400/metrics",
            "http://pod-1:8080/metrics",
        ]
        assert (
            "http://localhost:8000/v1/chat/completions/metrics"
            not in manager._server_metrics_endpoints
        )

    @pytest.mark.asyncio
    async def test_configure_auto_not_in_cluster_keeps_default_endpoint(self):
        run = _make_run(AIPerfConfig(**_BASE))
        manager = ServerMetricsManager(run=run)

        with (
            patch(
                "aiperf.server_metrics.endpoint_resolver.is_running_in_kubernetes",
                return_value=False,
            ),
            patch(
                "aiperf.server_metrics.manager.ServerMetricsDataCollector"
            ) as mock_collector_class,
        ):
            mock_collector = AsyncMock()
            mock_collector.is_url_reachable = AsyncMock(return_value=True)
            mock_collector_class.return_value = mock_collector

            await manager._profile_configure_command(
                Command(cid="test", cmd=CommandType.PROFILE_CONFIGURE)
            )

        assert manager._server_metrics_endpoints == [
            "http://localhost:8000/v1/chat/completions/metrics"
        ]

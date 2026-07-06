# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NetworkLatencyManager command handlers + probe-collector wiring.

Mirrors ``tests/unit/server_metrics/test_server_metrics_manager.py``: the manager
is constructed directly with a real ``BenchmarkRun`` (comms/push client come from
the auto-mocked singleton communication) and the ``@on_command`` handlers are
invoked with command messages. ``NetworkLatencyProbeCollector`` /
``asyncio.open_connection`` are patched so no real network I/O happens.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CommandType
from aiperf.common.environment import Environment
from aiperf.common.messages import (
    NetworkLatencyRecordMessage,
    ProfileCancelCommand,
    ProfileCompleteCommand,
    ProfileStartCommand,
)
from aiperf.common.models import ErrorDetails, NetworkLatencySample
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.network_latency.manager import NetworkLatencyManager
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def cfg_automatic() -> CLIConfig:
    """CLIConfig with active RTT probing enabled (no manual mean)."""
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        urls=["http://localhost:8000/v1/chat"],
        network_latency_automatic=True,
    )


@pytest.fixture
def cfg_automatic_multi_url() -> CLIConfig:
    """CLIConfig with two distinct host:port targets plus a duplicate."""
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        urls=[
            "http://localhost:8000/v1/chat",
            "http://localhost:8000/v1/completions",  # dupe host:port
            "http://other-host:9000/v1/chat",
        ],
        network_latency_automatic=True,
    )


def _make_manager(cli_config: CLIConfig) -> NetworkLatencyManager:
    return NetworkLatencyManager(run=make_run_from_cli(cli_config))


class _StubProbeCollector:
    """Per-instance stand-in for a probe collector in PROFILE_COMPLETE top-up tests.

    Uses a real instance (not the shared AsyncMock type) so ``successful_samples``
    can be a property without leaking onto every AsyncMock process-wide.
    """

    def __init__(
        self,
        *,
        samples: int = 0,
        bump_per_probe: int = 0,
        probe_error: Exception | None = None,
    ) -> None:
        self._samples = samples
        self._bump_per_probe = bump_per_probe
        self._probe_error = probe_error
        self.probe_calls = 0
        self.stop = AsyncMock()

    @property
    def successful_samples(self) -> int:
        return self._samples

    async def probe_once(self) -> None:
        self.probe_calls += 1
        if self._probe_error is not None:
            raise self._probe_error
        self._samples += self._bump_per_probe


def _success_sample() -> NetworkLatencySample:
    return NetworkLatencySample(
        timestamp_ns=1_000,
        target_url="http://localhost:8000/v1/chat",
        target_host="localhost",
        target_port=8000,
        probe_type="tcp_connect",
        rtt_ns=1_234,
        success=True,
    )


class TestNetworkLatencyManagerInitialization:
    """Target discovery + deduplication at construction time."""

    def test_init_discovers_single_target(self, cfg_automatic: CLIConfig) -> None:
        manager = _make_manager(cfg_automatic)
        assert "localhost:8000" in manager._targets
        assert manager._collectors == {}

    def test_init_dedupes_targets_by_host_port(
        self, cfg_automatic_multi_url: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic_multi_url)
        # Two URLs share localhost:8000; only one target key for that host:port.
        assert set(manager._targets) == {"localhost:8000", "other-host:9000"}

    def test_derive_target_applies_scheme_default_port(self) -> None:
        assert NetworkLatencyManager._derive_target("https://example.com/v1") == (
            "example.com",
            443,
        )
        assert NetworkLatencyManager._derive_target("http://example.com/v1") == (
            "example.com",
            80,
        )

    def test_derive_target_explicit_port_wins(self) -> None:
        assert NetworkLatencyManager._derive_target("http://h:1234/x") == ("h", 1234)

    def test_derive_target_no_host_returns_none(self) -> None:
        assert (
            NetworkLatencyManager._derive_target("not-a-url-with-empty-host://") is None
        )


class TestProfileStartCommand:
    """PROFILE_START builds, resolves, and starts one collector per target."""

    @pytest.mark.asyncio
    async def test_start_builds_one_collector_per_unique_target(
        self, cfg_automatic_multi_url: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic_multi_url)

        with patch(
            "aiperf.network_latency.manager.NetworkLatencyProbeCollector",
            side_effect=[AsyncMock(), AsyncMock()],
        ):
            await manager._on_start_profiling(
                ProfileStartCommand(
                    service_id=manager.id, command=CommandType.PROFILE_START
                )
            )

        # One collector per unique host:port target.
        assert set(manager._collectors) == {"localhost:8000", "other-host:9000"}
        for collector in manager._collectors.values():
            collector.initialize.assert_awaited_once()
            collector.resolve.assert_awaited_once()
            collector.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_tolerates_partial_collector_failure(
        self, cfg_automatic_multi_url: CLIConfig
    ) -> None:
        """One bad target must not abort the others."""
        manager = _make_manager(cfg_automatic_multi_url)

        good = AsyncMock()
        bad = AsyncMock()
        bad.start.side_effect = RuntimeError("boom")

        with patch(
            "aiperf.network_latency.manager.NetworkLatencyProbeCollector",
            side_effect=[good, bad],
        ):
            await manager._on_start_profiling(
                ProfileStartCommand(
                    service_id=manager.id, command=CommandType.PROFILE_START
                )
            )

        # Only the collector that started successfully is retained.
        assert len(manager._collectors) == 1
        assert good in manager._collectors.values()

    @pytest.mark.asyncio
    async def test_start_with_no_targets_schedules_delayed_shutdown(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        manager._targets = {}

        def close_coroutine(coro):
            coro.close()
            return MagicMock()

        with patch("asyncio.create_task", side_effect=close_coroutine) as mock_create:
            await manager._on_start_profiling(
                ProfileStartCommand(
                    service_id=manager.id, command=CommandType.PROFILE_START
                )
            )

        mock_create.assert_called_once()
        assert manager._collectors == {}

    @pytest.mark.asyncio
    async def test_start_with_all_collectors_failing_schedules_delayed_shutdown(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)

        bad = AsyncMock()
        bad.initialize.side_effect = RuntimeError("init failed")

        def close_coroutine(coro):
            coro.close()
            return MagicMock()

        with (
            patch(
                "aiperf.network_latency.manager.NetworkLatencyProbeCollector",
                return_value=bad,
            ),
            patch("asyncio.create_task", side_effect=close_coroutine) as mock_create,
        ):
            await manager._on_start_profiling(
                ProfileStartCommand(
                    service_id=manager.id, command=CommandType.PROFILE_START
                )
            )

        mock_create.assert_called_once()
        assert manager._collectors == {}


class TestProfileCompleteCommand:
    """PROFILE_COMPLETE tops up to MIN_SAMPLES then stops all collectors."""

    @pytest.mark.asyncio
    async def test_complete_tops_up_to_min_samples_then_stops(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        min_samples = Environment.NETWORK_LATENCY.MIN_SAMPLES

        # Below the floor; each probe_once bumps the count by one.
        collector = _StubProbeCollector(bump_per_probe=1)
        manager._collectors = {"localhost:8000": collector}

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )

        # Topped up exactly to the floor, then stopped + cleared.
        assert collector.successful_samples == min_samples
        collector.stop.assert_awaited_once()
        assert manager._collectors == {}

    @pytest.mark.asyncio
    async def test_complete_no_topup_when_already_at_floor(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)

        collector = _StubProbeCollector(samples=Environment.NETWORK_LATENCY.MIN_SAMPLES)
        manager._collectors = {"localhost:8000": collector}

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )

        assert collector.probe_calls == 0
        collector.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_topup_probe_failure_breaks_loop_and_still_stops(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)

        collector = _StubProbeCollector(probe_error=RuntimeError("probe boom"))
        manager._collectors = {"localhost:8000": collector}

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )

        # The failing probe breaks the top-up loop after one attempt; stop still runs.
        assert collector.probe_calls == 1
        collector.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_topup_respects_wallclock_budget(
        self, cfg_automatic: CLIConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A zero budget means the deadline is already passed: no top-up probes are
        # issued (so a slow endpoint can't stall PROFILE_COMPLETE), but stop still runs.
        monkeypatch.setattr(Environment.NETWORK_LATENCY, "COMPLETE_TOPUP_TIMEOUT", 0.0)
        manager = _make_manager(cfg_automatic)
        collector = _StubProbeCollector(bump_per_probe=1)
        manager._collectors = {"localhost:8000": collector}

        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )

        assert collector.probe_calls == 0
        collector.stop.assert_awaited_once()
        assert manager._collectors == {}

    @pytest.mark.asyncio
    async def test_complete_is_idempotent_when_already_stopped(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        manager._collectors = {}

        # Must not raise.
        await manager._handle_profile_complete_command(
            ProfileCompleteCommand(
                service_id=manager.id, command=CommandType.PROFILE_COMPLETE
            )
        )


class TestProfileCancelAndStop:
    @pytest.mark.asyncio
    async def test_cancel_stops_all_collectors(self, cfg_automatic: CLIConfig) -> None:
        manager = _make_manager(cfg_automatic)
        collector = AsyncMock()
        manager._collectors = {"localhost:8000": collector}

        await manager._handle_profile_cancel_command(
            ProfileCancelCommand(
                service_id=manager.id, command=CommandType.PROFILE_CANCEL
            )
        )

        collector.stop.assert_awaited_once()
        assert manager._collectors == {}

    @pytest.mark.asyncio
    async def test_on_stop_hook_stops_all_collectors(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        collector = AsyncMock()
        manager._collectors = {"localhost:8000": collector}

        await manager._network_latency_manager_stop()

        collector.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_all_collectors_tolerates_stop_failure(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        collector = AsyncMock()
        collector.stop.side_effect = RuntimeError("stop boom")
        manager._collectors = {"localhost:8000": collector}

        # Must not raise; collectors cleared.
        await manager._stop_all_collectors()
        assert manager._collectors == {}

    @pytest.mark.asyncio
    async def test_stop_all_collectors_noop_when_empty(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        manager._collectors = {}
        await manager._stop_all_collectors()

    @pytest.mark.asyncio
    async def test_delayed_shutdown_sleeps_then_stops(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        manager.stop = AsyncMock()

        with (
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch("asyncio.shield", new_callable=AsyncMock) as mock_shield,
        ):
            await manager._delayed_shutdown()

        mock_sleep.assert_awaited_once()
        mock_shield.assert_awaited_once()


class TestSampleAndErrorCallbacks:
    """Sample/error callbacks push NetworkLatencyRecordMessage to RECORDS."""

    @pytest.mark.asyncio
    async def test_sample_callback_pushes_record_message(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        manager.records_push_client.push = AsyncMock()

        sample = _success_sample()
        await manager._on_network_latency_samples([sample], "localhost:8000")

        manager.records_push_client.push.assert_awaited_once()
        message = manager.records_push_client.push.call_args[0][0]
        assert isinstance(message, NetworkLatencyRecordMessage)
        assert message.sample == sample
        assert message.collector_id == "localhost:8000"
        assert message.error is None

    @pytest.mark.asyncio
    async def test_sample_callback_empty_list_pushes_nothing(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        manager.records_push_client.push = AsyncMock()

        await manager._on_network_latency_samples([], "localhost:8000")

        manager.records_push_client.push.assert_not_called()

    @pytest.mark.asyncio
    async def test_sample_callback_push_failure_falls_back_to_error_message(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        # First push (sample) fails; fallback push (error) succeeds.
        manager.records_push_client.push = AsyncMock(
            side_effect=[RuntimeError("push boom"), None]
        )

        await manager._on_network_latency_samples([_success_sample()], "localhost:8000")

        assert manager.records_push_client.push.await_count == 2
        fallback = manager.records_push_client.push.call_args_list[1][0][0]
        assert isinstance(fallback, NetworkLatencyRecordMessage)
        assert fallback.sample is None
        assert fallback.error is not None

    @pytest.mark.asyncio
    async def test_sample_callback_nested_failure_does_not_raise(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        manager.records_push_client.push = AsyncMock(
            side_effect=RuntimeError("always fails")
        )

        # Both the sample push and the error fallback push fail; must not raise.
        await manager._on_network_latency_samples([_success_sample()], "localhost:8000")
        assert manager.records_push_client.push.await_count == 2

    @pytest.mark.asyncio
    async def test_error_callback_pushes_error_message(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        manager.records_push_client.push = AsyncMock()

        error = ErrorDetails.from_exception(ConnectionRefusedError("refused"))
        await manager._on_network_latency_error(error, "localhost:8000")

        manager.records_push_client.push.assert_awaited_once()
        message = manager.records_push_client.push.call_args[0][0]
        assert isinstance(message, NetworkLatencyRecordMessage)
        assert message.sample is None
        assert message.error == error

    @pytest.mark.asyncio
    async def test_error_callback_push_failure_does_not_raise(
        self, cfg_automatic: CLIConfig
    ) -> None:
        manager = _make_manager(cfg_automatic)
        manager.records_push_client.push = AsyncMock(
            side_effect=RuntimeError("push boom")
        )

        error = ErrorDetails.from_exception(ConnectionRefusedError("refused"))
        # Must not raise.
        await manager._on_network_latency_error(error, "localhost:8000")

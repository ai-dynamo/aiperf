# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the network-latency handler + RTT delivery in RecordsManager.

Mirrors the ``RecordsManager.__new__(RecordsManager)`` dispatch-test pattern in
``test_records_manager.py``: only the attributes a method touches are populated,
so each method is exercised in isolation without spinning up the full service.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.messages import NetworkLatencyRecordMessage
from aiperf.common.models import ErrorDetails, NetworkLatencySample
from aiperf.records.records_manager import ErrorTrackingState, RecordsManager


def _sample(rtt_ns: int = 1_500_000, success: bool = True) -> NetworkLatencySample:
    return NetworkLatencySample(
        timestamp_ns=1_000,
        target_url="http://localhost:8000/v1/chat",
        target_host="localhost",
        target_port=8000,
        probe_type="tcp_connect",
        rtt_ns=rtt_ns if success else None,
        success=success,
    )


class TestOnNetworkLatencyRecords:
    @pytest.mark.asyncio
    async def test_valid_sample_accumulates_and_forwards(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._network_latency_accumulator = MagicMock()
        manager._send_network_latency_to_results_processors = AsyncMock()

        sample = _sample()
        message = NetworkLatencyRecordMessage(
            service_id="net-mgr", collector_id="localhost:8000", sample=sample
        )

        await manager._on_network_latency_records(message)

        manager._network_latency_accumulator.add_sample.assert_called_once_with(sample)
        manager._send_network_latency_to_results_processors.assert_awaited_once_with(
            sample
        )

    @pytest.mark.asyncio
    async def test_valid_sample_without_accumulator_still_forwards(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._network_latency_accumulator = None
        manager._send_network_latency_to_results_processors = AsyncMock()

        sample = _sample()
        await manager._on_network_latency_records(
            NetworkLatencyRecordMessage(
                service_id="net-mgr", collector_id="localhost:8000", sample=sample
            )
        )

        manager._send_network_latency_to_results_processors.assert_awaited_once_with(
            sample
        )

    @pytest.mark.asyncio
    async def test_error_message_increments_error_count_and_does_not_forward(
        self,
    ) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._network_latency_accumulator = MagicMock()
        manager._network_latency_state = ErrorTrackingState()
        manager._send_network_latency_to_results_processors = AsyncMock()

        error = ErrorDetails.from_exception(ConnectionRefusedError("refused"))
        message = NetworkLatencyRecordMessage(
            service_id="net-mgr",
            collector_id="localhost:8000",
            sample=None,
            error=error,
        )

        await manager._on_network_latency_records(message)

        assert manager._network_latency_state.error_counts[error] == 1
        manager._network_latency_accumulator.add_sample.assert_not_called()
        manager._send_network_latency_to_results_processors.assert_not_awaited()


class TestSendNetworkLatencyToResultsProcessors:
    @pytest.mark.asyncio
    async def test_empty_processor_list_is_noop(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._network_latency_processors = []
        manager.exception = MagicMock()

        await manager._send_network_latency_to_results_processors(_sample())

        manager.exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_forwards_sample_to_each_processor(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        p1 = MagicMock()
        p1.process_network_latency_sample = AsyncMock()
        p2 = MagicMock()
        p2.process_network_latency_sample = AsyncMock()
        manager._network_latency_processors = [p1, p2]
        manager.exception = MagicMock()

        sample = _sample()
        await manager._send_network_latency_to_results_processors(sample)

        p1.process_network_latency_sample.assert_awaited_once_with(sample)
        p2.process_network_latency_sample.assert_awaited_once_with(sample)
        manager.exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_processor_failure_is_tracked_not_raised(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        ok = MagicMock()
        ok.process_network_latency_sample = AsyncMock()
        failing = MagicMock()
        failing.process_network_latency_sample = AsyncMock(
            side_effect=RuntimeError("processor boom")
        )
        manager._network_latency_processors = [ok, failing]
        manager._network_latency_state = ErrorTrackingState()
        manager.exception = MagicMock()

        # Must not raise.
        await manager._send_network_latency_to_results_processors(_sample())

        manager.exception.assert_called_once()
        assert sum(manager._network_latency_state.error_counts.values()) == 1


class TestDeliverNetworkRttToProcessors:
    def _make_manager(self, network_cfg) -> RecordsManager:
        manager = RecordsManager.__new__(RecordsManager)
        manager.run = SimpleNamespace(cfg=SimpleNamespace(network_latency=network_cfg))
        manager.notice = MagicMock()
        manager.warning = MagicMock()
        manager._network_latency_accumulator = None
        manager._metric_results_processors = []
        return manager

    def test_disabled_is_noop(self) -> None:
        processor = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=False, mean_ms=None))
        manager._metric_results_processors = [processor]

        manager._deliver_network_rtt_to_processors()

        processor.set_network_rtt_ns.assert_not_called()
        manager.notice.assert_not_called()

    def test_manual_mean_sets_rtt_ns_and_logs_notice(self) -> None:
        processor = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=2.5))
        manager._metric_results_processors = [processor]

        manager._deliver_network_rtt_to_processors()

        # 2.5 ms -> 2.5e6 ns delivered to the processor.
        processor.set_network_rtt_ns.assert_called_once_with(2.5 * 1e6)
        manager.notice.assert_called_once()

    def test_measured_mean_from_accumulator_sets_rtt_ns(self) -> None:
        processor = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=None))
        manager._metric_results_processors = [processor]
        manager._network_latency_accumulator = MagicMock(
            mean_rtt_ns=1_750_000.0, successful_sample_count=12
        )

        manager._deliver_network_rtt_to_processors()

        processor.set_network_rtt_ns.assert_called_once_with(1_750_000.0)
        manager.notice.assert_called_once()
        manager.warning.assert_not_called()

    def test_zero_successful_samples_warns_and_applies_no_adjustment(self) -> None:
        processor = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=None))
        manager._metric_results_processors = [processor]
        manager._network_latency_accumulator = MagicMock(mean_rtt_ns=None)

        manager._deliver_network_rtt_to_processors()

        # No measurable RTT: warn and leave processors at their default (no injection).
        manager.warning.assert_called_once()
        processor.set_network_rtt_ns.assert_not_called()

    def test_zero_mean_override_is_noop(self) -> None:
        # mean_ms=0 would emit network_adjusted_* identical to the raw metrics; skip it.
        processor = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=0.0))
        manager._metric_results_processors = [processor]

        manager._deliver_network_rtt_to_processors()

        processor.set_network_rtt_ns.assert_not_called()
        manager.notice.assert_not_called()

    def test_processor_without_setter_is_skipped(self) -> None:
        # A processor lacking set_network_rtt_ns must not break delivery.
        processor = MagicMock(spec=[])
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=2.5))
        manager._metric_results_processors = [processor]

        # Must not raise.
        manager._deliver_network_rtt_to_processors()
        manager.notice.assert_called_once()

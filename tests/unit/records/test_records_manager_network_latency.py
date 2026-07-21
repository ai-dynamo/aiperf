# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the network-latency handler + RTT delivery in RecordsManager."""

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
    async def test_valid_sample_accumulates_and_dispatches(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._network_latency_accumulator = MagicMock()
        manager._network_latency_state = ErrorTrackingState()
        manager._dispatch_record = AsyncMock(return_value=[])

        sample = _sample()
        message = NetworkLatencyRecordMessage(
            service_id="net-mgr", collector_id="localhost:8000", sample=sample
        )

        await manager._on_network_latency_records(message)

        manager._network_latency_accumulator.add_sample.assert_called_once_with(sample)
        manager._dispatch_record.assert_awaited_once_with(sample)
        assert manager._network_latency_state.error_counts == {}

    @pytest.mark.asyncio
    async def test_valid_sample_without_accumulator_still_dispatches(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._network_latency_accumulator = None
        manager._network_latency_state = ErrorTrackingState()
        manager._dispatch_record = AsyncMock(return_value=[])

        sample = _sample()
        await manager._on_network_latency_records(
            NetworkLatencyRecordMessage(
                service_id="net-mgr", collector_id="localhost:8000", sample=sample
            )
        )

        manager._dispatch_record.assert_awaited_once_with(sample)

    @pytest.mark.asyncio
    async def test_dispatch_errors_are_tracked(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._network_latency_accumulator = MagicMock()
        manager._network_latency_state = ErrorTrackingState()
        dispatch_error = RuntimeError("writer failed")
        manager._dispatch_record = AsyncMock(return_value=[dispatch_error])

        await manager._on_network_latency_records(
            NetworkLatencyRecordMessage(
                service_id="net-mgr", collector_id="localhost:8000", sample=_sample()
            )
        )

        tracked = ErrorDetails.from_exception(dispatch_error)
        assert manager._network_latency_state.error_counts[tracked] == 1

    @pytest.mark.asyncio
    async def test_error_message_increments_error_count_and_does_not_dispatch(
        self,
    ) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._network_latency_accumulator = MagicMock()
        manager._network_latency_state = ErrorTrackingState()
        manager._dispatch_record = AsyncMock(return_value=[])

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
        manager._dispatch_record.assert_not_awaited()


class TestDeliverNetworkRttToAccumulators:
    def _make_manager(self, network_cfg) -> RecordsManager:
        manager = RecordsManager.__new__(RecordsManager)
        manager.run = SimpleNamespace(cfg=SimpleNamespace(network_latency=network_cfg))
        manager.notice = MagicMock()
        manager.warning = MagicMock()
        manager._network_latency_accumulator = None
        manager._metric_record_accumulators = []
        return manager

    def test_disabled_is_noop(self) -> None:
        accumulator = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=False, mean_ms=None))
        manager._metric_record_accumulators = [accumulator]

        manager._deliver_network_rtt_to_accumulators()

        accumulator.set_network_rtt_ns.assert_not_called()
        manager.notice.assert_not_called()

    def test_manual_mean_sets_rtt_ns_and_logs_notice(self) -> None:
        accumulator = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=2.5))
        manager._metric_record_accumulators = [accumulator]

        manager._deliver_network_rtt_to_accumulators()

        accumulator.set_network_rtt_ns.assert_called_once_with(2.5 * 1e6)
        manager.notice.assert_called_once()

    def test_measured_mean_from_accumulator_sets_rtt_ns(self) -> None:
        accumulator = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=None))
        manager._metric_record_accumulators = [accumulator]
        manager._network_latency_accumulator = MagicMock(
            mean_rtt_ns=1_750_000.0, successful_sample_count=12
        )

        manager._deliver_network_rtt_to_accumulators()

        accumulator.set_network_rtt_ns.assert_called_once_with(1_750_000.0)
        manager.notice.assert_called_once()
        manager.warning.assert_not_called()

    def test_zero_successful_samples_warns_and_applies_no_adjustment(self) -> None:
        accumulator = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=None))
        manager._metric_record_accumulators = [accumulator]
        manager._network_latency_accumulator = MagicMock(mean_rtt_ns=None)

        manager._deliver_network_rtt_to_accumulators()

        manager.warning.assert_called_once()
        accumulator.set_network_rtt_ns.assert_not_called()

    def test_zero_mean_override_is_noop(self) -> None:
        accumulator = MagicMock()
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=0.0))
        manager._metric_record_accumulators = [accumulator]

        manager._deliver_network_rtt_to_accumulators()

        accumulator.set_network_rtt_ns.assert_not_called()
        manager.notice.assert_not_called()

    def test_accumulator_without_setter_is_skipped(self) -> None:
        accumulator = MagicMock(spec=[])
        manager = self._make_manager(SimpleNamespace(enabled=True, mean_ms=2.5))
        manager._metric_record_accumulators = [accumulator]

        manager._deliver_network_rtt_to_accumulators()

        manager.notice.assert_called_once()

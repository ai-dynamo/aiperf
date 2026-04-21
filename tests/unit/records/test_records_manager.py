# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import msgspec
import pytest

from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsBatchWireMessage,
    MetricRecordsData,
    TelemetryRecordsWireMessage,
    _error_to_wire,
)
from aiperf.common.models import (
    ErrorDetails,
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
    TelemetryHierarchy,
    TelemetryRecord,
)
from aiperf.common.types import MetricTagT
from aiperf.records.records_manager import RecordsManager


# Helper functions
def create_mock_records_manager(
    start_time_ns: int,
    expected_duration_sec: float | None,
    grace_period_sec: float = 0.0,
) -> MagicMock:
    """Create a mock RecordsManager instance for testing filtering logic."""
    instance = MagicMock()
    instance.expected_duration_sec = expected_duration_sec
    instance.start_time_ns = start_time_ns
    instance.user_config.loadgen.benchmark_grace_period = grace_period_sec
    instance.debug = MagicMock()
    return instance


def create_metric_record_data(
    request_start_ns: int,
    request_end_ns: int,
    metrics: dict[MetricTagT, int | float] | None = None,
) -> MetricRecordsData:
    """Create a MetricRecordsData object with sensible defaults for testing."""
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            request_num=0,
            session_num=0,
            conversation_id="test",
            turn_index=0,
            request_start_ns=request_start_ns,
            request_end_ns=request_end_ns,
            worker_id="worker-1",
            record_processor_id="processor-1",
            benchmark_phase="profiling",
        ),
        metrics=metrics or {},
    )


class TestRecordsManagerTelemetry:
    """Test RecordsManager telemetry handling with mocked components."""

    @pytest.mark.asyncio
    async def test_on_telemetry_records_valid(self):
        """Test handling valid telemetry records."""
        # Create sample telemetry records
        records = [
            TelemetryRecord(
                timestamp_ns=1000000,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=0,
                gpu_uuid="GPU-123",
                gpu_model_name="Test GPU",
                telemetry_data={
                    "gpu_power_usage": 100.0,
                },
            )
        ]

        message = TelemetryRecordsWireMessage(
            service_id="test_service",
            collector_id="test_collector",
            dcgm_url="http://localhost:9400/metrics",
            records=tuple(records),
        )
        assert isinstance(message, TelemetryRecordsWireMessage)

        # Mock the hierarchy
        mock_hierarchy = MagicMock(spec=TelemetryHierarchy)
        mock_hierarchy.add_record = MagicMock()
        mock_send_to_processors = AsyncMock()

        # Test the logic directly without instantiating the full service
        for record in message.records:
            mock_hierarchy.add_record(record)

        if message.records:
            await mock_send_to_processors(message.records)

        # Verify behavior
        assert mock_hierarchy.add_record.call_count == len(records)
        mock_send_to_processors.assert_called_once_with(message.records)

    @pytest.mark.asyncio
    async def test_on_telemetry_records_invalid(self):
        """Test handling invalid telemetry records with errors."""
        error = ErrorDetails(message="Test error", code=500)

        message = TelemetryRecordsWireMessage(
            service_id="test_service",
            collector_id="test_collector",
            dcgm_url="http://localhost:9400/metrics",
            error=_error_to_wire(error),
        )

        mock_send_to_processors = AsyncMock()
        error_counts = {}

        # Test the logic: errors should be tracked, not sent to processors
        if message.error:
            error_counts[message.error] = error_counts.get(message.error, 0) + 1
        else:
            await mock_send_to_processors(message.records)

        # Should not send to processors
        mock_send_to_processors.assert_not_called()

        # Error should be tracked
        assert message.error in error_counts
        assert error_counts[message.error] == 1

    @pytest.mark.asyncio
    async def test_send_telemetry_to_results_processors(self):
        """Test sending telemetry records to processors."""
        # Create mock telemetry processor
        mock_processor = Mock()
        mock_processor.process_telemetry_record = AsyncMock()

        records = [
            TelemetryRecord(
                timestamp_ns=1000000,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=0,
                gpu_uuid="GPU-123",
                gpu_model_name="Test GPU",
                telemetry_data={},
            ),
            TelemetryRecord(
                timestamp_ns=1000001,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=1,
                gpu_uuid="GPU-456",
                gpu_model_name="Test GPU",
                telemetry_data={},
            ),
        ]

        # Test the logic: each record should be sent to processor
        for record in records:
            await mock_processor.process_telemetry_record(record)

        # Processor should be called for each record
        assert mock_processor.process_telemetry_record.call_count == len(records)

    def test_telemetry_hierarchy_add_record(self):
        """Test that telemetry hierarchy adds records correctly."""
        hierarchy = TelemetryHierarchy()

        record = TelemetryRecord(
            timestamp_ns=1000000,
            dcgm_url="http://localhost:9400/metrics",
            gpu_index=0,
            gpu_uuid="GPU-123",
            gpu_model_name="Test GPU",
            telemetry_data={
                "gpu_power_usage": 100.0,
            },
        )

        # Add record to hierarchy
        hierarchy.add_record(record)

        # Verify hierarchy structure
        assert "http://localhost:9400/metrics" in hierarchy.dcgm_endpoints
        assert "GPU-123" in hierarchy.dcgm_endpoints["http://localhost:9400/metrics"]


class TestRecordsManagerBatchWire:
    @pytest.mark.asyncio
    async def test_on_metric_records_handles_batch_wire_message(self) -> None:
        record_data = create_metric_record_data(100, 200, {"request_latency": 1.0})
        batch_message = MetricRecordsBatchWireMessage(
            service_id="record-processor-1",
            records=[record_data, record_data],
        )
        manager = SimpleNamespace(
            is_trace_enabled=False,
            trace=MagicMock(),
            _records_tracker=MagicMock(),
            _send_results_to_results_processors=AsyncMock(),
            _error_tracker=MagicMock(),
            _handle_all_records_received=AsyncMock(),
            _process_metric_record_data=AsyncMock(),
        )
        manager._records_tracker.is_phase_excluded.return_value = False
        manager._records_tracker.check_and_set_all_records_received_for_phase.side_effect = [
            False,
            False,
        ]

        await RecordsManager._on_metric_records(manager, batch_message)

        assert manager._process_metric_record_data.await_count == 2


class TestRecordsManagerTimeslice:
    """Test cases for RecordsManager timeslice functionality."""

    @pytest.mark.asyncio
    async def test_process_records_result_with_both_records_and_timeslice(self):
        """Test that ProcessRecordsResult can contain both records and timeslice results."""

        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        timeslice_results = {
            0: [metric_result],
            1: [metric_result],
        }

        # Create a ProcessRecordsResult with both types of results
        result = ProcessRecordsResult(
            results=ProfileResults(
                records=[metric_result, metric_result],
                timeslice_metric_results=timeslice_results,
                completed=2,
                start_ns=1000000000,
                end_ns=2000000000,
            )
        )

        assert result.results.records is not None
        assert len(result.results.records) == 2
        assert result.results.timeslice_metric_results is not None
        assert len(result.results.timeslice_metric_results) == 2

    @pytest.mark.asyncio
    async def test_profile_results_serialization_with_timeslice(self):
        """Test that ProfileResults with timeslice data can be serialized."""
        metric_result = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=100.0,
            count=10,
        )

        timeslice_results = {
            0: [metric_result],
            1: [metric_result],
        }

        profile_results = ProfileResults(
            records=[metric_result],
            timeslice_metric_results=timeslice_results,
            completed=1,
            start_ns=1000000000,
            end_ns=2000000000,
        )

        # Test that it can be converted to dict (for JSON serialization)
        result_dict = msgspec.to_builtins(profile_results)

        assert "records" in result_dict
        assert "timeslice_metric_results" in result_dict
        assert result_dict["timeslice_metric_results"] is not None
        assert 0 in result_dict["timeslice_metric_results"]
        assert 1 in result_dict["timeslice_metric_results"]


class TestRecordsManagerServerMetricsErrorHandling:
    """T1 regression: server-metrics processor exceptions must be caught and
    counted, matching the telemetry path. Before the fix these propagated out
    of the PULL handler and were silently swallowed by the dispatcher, losing
    the error signal entirely.
    """

    @pytest.mark.asyncio
    async def test_processor_exception_is_counted_not_propagated(self) -> None:
        from collections import defaultdict

        from aiperf.common.metric_records_wire import (
            ServerMetricsRecordWireMessage,
        )
        from aiperf.common.models import ServerMetricsRecord

        manager = SimpleNamespace(
            _send_server_metrics_to_results_processors=AsyncMock(
                side_effect=ValueError("processor failure")
            ),
            _server_metrics_state=SimpleNamespace(
                error_counts=defaultdict(int),
            ),
            debug=MagicMock(),
        )

        message = ServerMetricsRecordWireMessage(
            service_id="server-metrics-manager",
            collector_id="col-1",
            record=ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                metrics={},
            ),
        )

        # Must NOT raise — processor failures must be absorbed by the handler.
        await RecordsManager._on_server_metrics_records(manager, message)

        # Exactly one error recorded; processor was attempted once.
        assert sum(manager._server_metrics_state.error_counts.values()) == 1, (
            "Processor exception should have been counted"
        )
        manager._send_server_metrics_to_results_processors.assert_awaited_once()

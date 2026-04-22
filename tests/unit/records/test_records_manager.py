# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest

from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsBatchWireMessage,
    MetricRecordsData,
)
from aiperf.common.models import (
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
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

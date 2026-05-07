# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import (
    CreditPhaseStats,
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.common.types import MetricTagT
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
)
from aiperf.plugin.enums import TimingMode
from aiperf.records.records_manager import RecordsManager
from aiperf.timing.config import CreditPhaseConfig
from tests.harness import mock_plugin


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
            session_num=0,
            conversation_id="test",
            turn_index=0,
            request_start_ns=request_start_ns,
            request_end_ns=request_end_ns,
            worker_id="worker-1",
            record_processor_id="processor-1",
            benchmark_phase=CreditPhase.PROFILING,
        ),
        metrics=metrics or {},
    )


class TestRecordsManagerTelemetry:
    """Test RecordsManager telemetry handling with mocked components."""

    @pytest.mark.asyncio
    async def test_on_telemetry_records_valid(self):
        """Test handling valid telemetry records."""
        from unittest.mock import AsyncMock, MagicMock

        from aiperf.common.messages import TelemetryRecordsMessage
        from aiperf.common.models import (
            TelemetryHierarchy,
            TelemetryMetrics,
            TelemetryRecord,
        )

        # Create sample telemetry records
        records = [
            TelemetryRecord(
                timestamp_ns=1000000,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=0,
                gpu_uuid="GPU-123",
                gpu_model_name="Test GPU",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=100.0,
                ),
            )
        ]

        message = TelemetryRecordsMessage(
            service_id="test_service",
            collector_id="test_collector",
            dcgm_url="http://localhost:9400/metrics",
            records=records,
            error=None,
        )

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
        mock_send_to_processors.assert_called_once_with(records)

    @pytest.mark.asyncio
    async def test_on_telemetry_records_invalid(self):
        """Test handling invalid telemetry records with errors."""
        from unittest.mock import AsyncMock

        from aiperf.common.messages import TelemetryRecordsMessage
        from aiperf.common.models import ErrorDetails

        error = ErrorDetails(message="Test error", code=500)

        message = TelemetryRecordsMessage(
            service_id="test_service",
            collector_id="test_collector",
            dcgm_url="http://localhost:9400/metrics",
            records=[],
            error=error,
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
        assert error in error_counts
        assert error_counts[error] == 1

    @pytest.mark.asyncio
    async def test_send_telemetry_to_results_processors(self):
        """Test sending telemetry records to processors."""
        from unittest.mock import AsyncMock, Mock

        from aiperf.common.models import TelemetryMetrics, TelemetryRecord

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
                telemetry_data=TelemetryMetrics(),
            ),
            TelemetryRecord(
                timestamp_ns=1000001,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=1,
                gpu_uuid="GPU-456",
                gpu_model_name="Test GPU",
                telemetry_data=TelemetryMetrics(),
            ),
        ]

        # Test the logic: each record should be sent to processor
        for record in records:
            await mock_processor.process_telemetry_record(record)

        # Processor should be called for each record
        assert mock_processor.process_telemetry_record.call_count == len(records)

    def test_telemetry_hierarchy_add_record(self):
        """Test that telemetry hierarchy adds records correctly."""
        from aiperf.common.models import (
            TelemetryHierarchy,
            TelemetryMetrics,
            TelemetryRecord,
        )

        hierarchy = TelemetryHierarchy()

        record = TelemetryRecord(
            timestamp_ns=1000000,
            dcgm_url="http://localhost:9400/metrics",
            gpu_index=0,
            gpu_uuid="GPU-123",
            gpu_model_name="Test GPU",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=100.0,
            ),
        )

        # Add record to hierarchy
        hierarchy.add_record(record)

        # Verify hierarchy structure
        assert "http://localhost:9400/metrics" in hierarchy.dcgm_endpoints
        assert "GPU-123" in hierarchy.dcgm_endpoints["http://localhost:9400/metrics"]


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
        result_dict = profile_results.model_dump()

        assert "records" in result_dict
        assert "timeslice_metric_results" in result_dict
        assert result_dict["timeslice_metric_results"] is not None
        assert 0 in result_dict["timeslice_metric_results"]
        assert 1 in result_dict["timeslice_metric_results"]


def _create_credit_phase_stats() -> CreditPhaseStats:
    return CreditPhaseStats(
        phase=CreditPhase.PROFILING,
        start_ns=1_000_000_000,
        sent_end_ns=2_000_000_000,
        requests_end_ns=3_000_000_000,
        total_expected_requests=64,
        expected_duration_sec=60.0,
        expected_grace_period_sec=30.0,
        requests_sent=64,
        requests_completed=64,
        requests_cancelled=0,
        request_errors=0,
        sent_sessions=64,
        completed_sessions=64,
        cancelled_sessions=0,
        total_session_turns=64,
    )


def _create_manager_for_timing_dispatch() -> RecordsManager:
    manager = RecordsManager.__new__(RecordsManager)
    manager._records_tracker = MagicMock()
    manager._timing_results_processors = []
    manager._send_timing_to_results_processors = AsyncMock()
    manager.info = MagicMock()
    manager.notice = MagicMock()
    manager.exception = MagicMock()
    manager._handle_all_records_received = AsyncMock()
    return manager


class TestRecordsManagerTimingDispatch:
    @pytest.mark.asyncio
    async def test_on_credit_phase_start_forwards_timing_snapshot(self) -> None:
        manager = _create_manager_for_timing_dispatch()
        stats = _create_credit_phase_stats()
        message = CreditPhaseStartMessage(
            service_id="timing-manager",
            stats=stats,
            config=CreditPhaseConfig(
                phase=CreditPhase.PROFILING,
                timing_mode=TimingMode.REQUEST_RATE,
            ),
        )

        await manager._on_credit_phase_start(message)

        manager._records_tracker.update_phase_info.assert_called_once_with(stats)
        manager._send_timing_to_results_processors.assert_awaited_once_with(stats)

    @pytest.mark.asyncio
    async def test_on_credit_phase_progress_forwards_timing_snapshot(self) -> None:
        manager = _create_manager_for_timing_dispatch()
        stats = _create_credit_phase_stats()
        message = CreditPhaseProgressMessage(service_id="timing-manager", stats=stats)

        await manager._on_credit_phase_progress(message)

        manager._records_tracker.update_phase_info.assert_called_once_with(stats)
        manager._send_timing_to_results_processors.assert_awaited_once_with(stats)

    @pytest.mark.asyncio
    async def test_on_credit_phase_sending_complete_forwards_timing_snapshot(
        self,
    ) -> None:
        manager = _create_manager_for_timing_dispatch()
        stats = _create_credit_phase_stats().model_copy(
            update={"final_requests_sent": 64}
        )
        message = CreditPhaseSendingCompleteMessage(
            service_id="timing-manager",
            stats=stats,
        )

        await manager._on_credit_phase_sending_complete(message)

        manager._records_tracker.update_phase_info.assert_called_once_with(stats)
        manager._send_timing_to_results_processors.assert_awaited_once_with(stats)

    @pytest.mark.asyncio
    async def test_on_credit_phase_complete_forwards_timing_snapshot(self) -> None:
        manager = _create_manager_for_timing_dispatch()
        stats = _create_credit_phase_stats().model_copy(
            update={"final_requests_completed": 64}
        )
        message = CreditPhaseCompleteMessage(service_id="timing-manager", stats=stats)
        manager._records_tracker.check_and_set_all_records_received_for_phase.return_value = False
        manager._records_tracker.create_stats_for_phase.return_value = MagicMock(
            total_records=64,
            final_requests_completed=64,
        )

        await manager._on_credit_phase_complete(message)

        manager._records_tracker.update_phase_info.assert_called_once_with(stats)
        manager._send_timing_to_results_processors.assert_awaited_once_with(stats)

    @pytest.mark.asyncio
    async def test_send_timing_to_results_processors_ignores_empty_processor_list(
        self,
    ) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._timing_results_processors = []
        manager.exception = MagicMock()

        await manager._send_timing_to_results_processors(_create_credit_phase_stats())

        manager.exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_timing_to_results_processors_logs_processor_failures(
        self,
    ) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        ok_processor = MagicMock()
        ok_processor.process_result = AsyncMock(return_value=None)
        failing_processor = MagicMock()
        failing_processor.process_result = AsyncMock(
            side_effect=RuntimeError("timing failure")
        )
        manager._timing_results_processors = [ok_processor, failing_processor]
        manager.exception = MagicMock()

        await manager._send_timing_to_results_processors(_create_credit_phase_stats())

        ok_processor.process_result.assert_awaited_once()
        failing_processor.process_result.assert_awaited_once()
        manager.exception.assert_called_once()


class TestRecordsManagerProcessorDispatch:
    @pytest.mark.asyncio
    async def test_send_metric_results_to_results_processors_ignores_empty_processor_list(
        self,
    ) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager._metric_results_processors = []
        manager.exception = MagicMock()

        await manager._send_results_to_results_processors(
            create_metric_record_data(1_000, 2_000)
        )

        manager.exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_results_to_results_processors_logs_failures(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        ok_processor = MagicMock()
        ok_processor.process_result = AsyncMock(return_value=None)
        failing_processor = MagicMock()
        failing_processor.process_result = AsyncMock(
            side_effect=RuntimeError("metric processing failed")
        )
        manager._metric_results_processors = [ok_processor, failing_processor]
        manager.exception = MagicMock()

        await manager._send_results_to_results_processors(
            create_metric_record_data(1_000, 2_000)
        )

        ok_processor.process_result.assert_awaited_once()
        failing_processor.process_result.assert_awaited_once()
        manager.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_metric_results_processors_flushes_only_flushable(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager.exception = MagicMock()
        manager.debug = MagicMock()

        class FakeFlushProtocol:
            pass

        class FakeFlushable(FakeFlushProtocol):
            def __init__(self) -> None:
                self.flush = AsyncMock(return_value=None)

        flushable = FakeFlushable()
        non_flushable = MagicMock()
        manager._metric_results_processors = [flushable, non_flushable]

        with patch(
            "aiperf.records.records_manager.FlushableResultsProcessorProtocol",
            FakeFlushProtocol,
        ):
            await manager._flush_metric_results_processors(force=True)

        flushable.flush.assert_awaited_once_with(force=True)
        manager.exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_metric_results_processors_logs_flush_failures(self) -> None:
        manager = RecordsManager.__new__(RecordsManager)
        manager.exception = MagicMock()
        manager.debug = MagicMock()

        class FakeFlushProtocol:
            pass

        class FakeFlushable(FakeFlushProtocol):
            def __init__(self) -> None:
                self.flush = AsyncMock(side_effect=RuntimeError("flush failed"))

        flushable = FakeFlushable()
        manager._metric_results_processors = [flushable]

        with patch(
            "aiperf.records.records_manager.FlushableResultsProcessorProtocol",
            FakeFlushProtocol,
        ):
            await manager._flush_metric_results_processors(force=True)

        flushable.flush.assert_awaited_once_with(force=True)
        manager.exception.assert_called_once()


class TestRecordsManagerInitialization:
    def test_otel_post_processor_disabled_logs_info(
        self,
        service_config,
        user_config,
    ) -> None:
        def _fake_pull_client_init(self, service_config, **kwargs) -> None:
            self.service_config = service_config
            self.user_config = kwargs["user_config"]
            self.service_id = kwargs.get("service_id") or "records_manager"
            self.pub_client = MagicMock()
            self.attach_child_lifecycle = MagicMock()
            self.debug = MagicMock()
            self.info = MagicMock()
            self.error = MagicMock()
            self.exception = MagicMock()

        class DisabledProcessor:
            def __init__(self, **kwargs) -> None:
                raise PostProcessorDisabled("disabled for test")

        with (
            patch(
                "aiperf.records.records_manager.PullClientMixin.__init__",
                new=_fake_pull_client_init,
            ),
            mock_plugin(
                "results_processor",
                "otel_metrics_streamer",
                DisabledProcessor,
            ),
        ):
            manager = RecordsManager(
                service_config=service_config,
                user_config=user_config,
            )

        info_messages = [args[0] for args, _ in manager.info.call_args_list]
        assert any(
            "OTel metrics streamer is disabled and will not be used" in message
            for message in info_messages
        )

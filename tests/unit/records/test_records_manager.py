# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest

from aiperf.common.messages import BaseServiceErrorMessage
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
        dataset_configured_event = asyncio.Event()
        dataset_configured_event.set()
        manager = SimpleNamespace(
            is_trace_enabled=False,
            trace=MagicMock(),
            _dataset_configured_event=dataset_configured_event,
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


class TestRecordsManagerDatasetConfiguredBarrier:
    """The records manager must not run metric records through its results
    processors until the DatasetConfiguredNotification has been applied.

    Metric records (PULL socket) and the notification (SUB socket) arrive on
    independent channels with no ordering guarantee, so processing must block
    on an explicit barrier that _on_dataset_configured releases.
    """

    @pytest.mark.asyncio
    async def test_on_dataset_configured_sets_event(self):
        """_on_dataset_configured must release the barrier once processors are configured."""
        mock_self = MagicMock(spec=RecordsManager)
        mock_self._dataset_configured_event = asyncio.Event()
        mock_self._metric_results_processors = []

        await RecordsManager._on_dataset_configured(mock_self, MagicMock())

        assert mock_self._dataset_configured_event.is_set()

    @pytest.mark.asyncio
    async def test_on_metric_records_waits_for_dataset_configured(self):
        """_on_metric_records must block until the dataset is configured, then proceed."""
        mock_self = MagicMock(spec=RecordsManager)
        mock_self._dataset_configured_event = asyncio.Event()
        mock_self.is_trace_enabled = False
        # First downstream step after the barrier; raising proves the barrier was passed.
        mock_self._process_metric_record_data = AsyncMock(
            side_effect=RuntimeError("REACHED_PROCESSING")
        )
        record_data = create_metric_record_data(100, 200, {"request_latency": 1.0})
        message = MetricRecordsBatchWireMessage(
            service_id="record-processor-rp-7f2a",
            records=[record_data],
        )

        task = asyncio.create_task(
            RecordsManager._on_metric_records(mock_self, message)
        )
        for _ in range(3):
            await asyncio.sleep(0)

        # Barrier not released: processing has not started.
        assert not task.done()
        assert not mock_self._process_metric_record_data.called

        # Barrier released: processing proceeds past the wait.
        mock_self._dataset_configured_event.set()
        with pytest.raises(RuntimeError, match="REACHED_PROCESSING"):
            await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_on_metric_records_fails_run_on_config_timeout(self, monkeypatch):
        """On dataset-config timeout, abort the run (report error + kill) rather
        than process the record without a configured dataset."""
        mock_self = MagicMock(spec=RecordsManager)
        mock_self.service_id = "rm-test"
        mock_self._dataset_configured_event = asyncio.Event()
        mock_self.is_trace_enabled = False
        mock_self.publish = AsyncMock()
        mock_self._kill = AsyncMock()
        mock_self._process_metric_record_data = AsyncMock()
        record_data = create_metric_record_data(100, 200, {"request_latency": 1.0})
        message = MetricRecordsBatchWireMessage(
            service_id="record-processor-rp-7f2a",
            records=[record_data],
        )

        async def _raise_timeout(coro, *args, **kwargs):
            coro.close()  # avoid "coroutine was never awaited" warning
            raise asyncio.TimeoutError

        monkeypatch.setattr(
            "aiperf.records.dataset_gate.asyncio.wait_for", _raise_timeout
        )

        await RecordsManager._on_metric_records(mock_self, message)

        # Run is failed loudly ...
        mock_self._kill.assert_awaited_once()
        published = mock_self.publish.await_args.args[0]
        assert isinstance(published, BaseServiceErrorMessage)
        # ... and the record is not processed.
        mock_self._process_metric_record_data.assert_not_called()


class TestMaybeHintMissingCacheReporting:
    """One-shot mid-run hint when the server reports token usage but never
    reports prompt-cache reads (cache-capable server not told to report
    ``cached_tokens``). Restored from origin/main; wired on the branch's
    ``_process_metric_record_data`` path."""

    def _bound_manager(self) -> MagicMock:
        mgr = MagicMock()
        mgr._warned_missing_cache_reporting = False
        mgr.warning = MagicMock()
        mgr._maybe_hint_missing_cache_reporting = (
            RecordsManager._maybe_hint_missing_cache_reporting.__get__(mgr)
        )
        return mgr

    def test_fires_once_when_usage_without_cache(self) -> None:
        mgr = self._bound_manager()
        record = create_metric_record_data(
            request_start_ns=0, request_end_ns=1, metrics={"usage_prompt_tokens": 128}
        )

        mgr._maybe_hint_missing_cache_reporting(record)
        mgr._maybe_hint_missing_cache_reporting(record)  # second record: no re-warn

        assert mgr._warned_missing_cache_reporting is True
        mgr.warning.assert_called_once()

    def test_no_hint_when_cache_reads_reported(self) -> None:
        mgr = self._bound_manager()
        record = create_metric_record_data(
            request_start_ns=0,
            request_end_ns=1,
            metrics={"usage_prompt_tokens": 128, "usage_prompt_cache_read_tokens": 0},
        )

        mgr._maybe_hint_missing_cache_reporting(record)

        assert mgr._warned_missing_cache_reporting is False
        mgr.warning.assert_not_called()

    def test_no_hint_when_no_usage(self) -> None:
        mgr = self._bound_manager()
        record = create_metric_record_data(
            request_start_ns=0, request_end_ns=1, metrics={}
        )

        mgr._maybe_hint_missing_cache_reporting(record)

        assert mgr._warned_missing_cache_reporting is False
        mgr.warning.assert_not_called()

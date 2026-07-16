# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.accuracy.models import AccuracyRecordsData
from aiperf.common.enums import CreditPhase, ExportLevel
from aiperf.common.messages import (
    BaseServiceErrorMessage,
    MetricRecordsData,
    RecordsMessage,
)
from aiperf.common.utils import compute_time_ns
from aiperf.records.record_processor_service import RecordProcessor
from tests.unit.post_processors.conftest import create_metric_metadata


def _make_accuracy_record(session_num: int = 0) -> AccuracyRecordsData:
    return AccuracyRecordsData(
        session_num=session_num,
        worker_id="w1",
        benchmark_phase=CreditPhase.PROFILING,
        timestamp_ns=1_000,
        task=None,
        grader_name="multiple_choice",
        passed=True,
        unparsed=False,
        confidence=1.0,
        expected="A",
        actual="A",
        explanation="ok",
    )


def _make_processor_mock(producers, observers=None):
    """Build a RecordProcessor MagicMock wired for the 2-stage generic ship."""
    mock_self = MagicMock(spec=RecordProcessor)
    mock_self.service_id = "rp"
    mock_self.run = MagicMock()
    mock_self.run.cfg.artifacts.export_level = ExportLevel.RECORDS
    mock_self.records_push_client = AsyncMock()
    mock_self.inference_result_parser = MagicMock()
    mock_self.inference_result_parser.parse_request_record = AsyncMock(
        return_value=MagicMock()
    )
    mock_self._create_metric_record_metadata = MagicMock(
        return_value=create_metric_metadata(session_num=0)
    )
    mock_self._free_record_data = MagicMock(return_value=(None, None))
    mock_self._producers = producers
    mock_self._observers = observers or []
    return mock_self


class TestRecordProcessorGenericShip:
    """Stage 1 producers emit finished typed records grouped by their declared
    record_type; stage 2 observers view them via a RecordObserverContext; then
    exactly ONE generic RecordsMessage per inference record ships every produced
    record on the envelope (no per-type message class, no builder map)."""

    @pytest.mark.asyncio
    async def test_process_and_forward_ships_single_generic_message(self):
        """A metric producer + an accuracy producer flatten into one RecordsMessage."""
        metric_data = MetricRecordsData(
            metadata=create_metric_metadata(session_num=0), metrics={"some_metric": 1.0}
        )
        accuracy_record = _make_accuracy_record()

        metric_producer = MagicMock()
        metric_producer.process_record = AsyncMock(return_value=metric_data)
        accuracy_producer = MagicMock()
        accuracy_producer.process_record = AsyncMock(return_value=accuracy_record)

        mock_self = _make_processor_mock(
            producers=[
                ("metric_records", metric_producer),
                ("accuracy", accuracy_producer),
            ]
        )

        await RecordProcessor._process_and_forward_record(
            mock_self, MagicMock(service_id="w1"), MagicMock(), None
        )

        mock_self.records_push_client.push.assert_awaited_once()
        msg = mock_self.records_push_client.push.await_args.args[0]
        assert isinstance(msg, RecordsMessage)
        assert metric_data in msg.records
        assert accuracy_record in msg.records
        assert len(msg.records) == 2

    @pytest.mark.asyncio
    async def test_observers_receive_context_with_grouped_produced(self):
        """Observers get a RecordObserverContext exposing ctx.metrics and
        ctx.get(record_type), grouped by the producer's declared record_type."""
        metric_data = MetricRecordsData(
            metadata=create_metric_metadata(session_num=0), metrics={"some_metric": 1.0}
        )
        accuracy_record = _make_accuracy_record()

        metric_producer = MagicMock()
        metric_producer.process_record = AsyncMock(return_value=metric_data)
        accuracy_producer = MagicMock()
        accuracy_producer.process_record = AsyncMock(return_value=accuracy_record)

        observer = MagicMock()
        observer.observe = AsyncMock()

        mock_self = _make_processor_mock(
            producers=[
                ("metric_records", metric_producer),
                ("accuracy", accuracy_producer),
            ],
            observers=[observer],
        )

        await RecordProcessor._process_and_forward_record(
            mock_self, MagicMock(service_id="w1"), MagicMock(), None
        )

        observer.observe.assert_awaited_once()
        ctx = observer.observe.await_args.args[0]
        assert ctx.metrics is metric_data
        assert ctx.get("accuracy") == [accuracy_record]

    @pytest.mark.asyncio
    async def test_producer_exception_does_not_block_ship(self):
        """A failing producer is logged and skipped; the record still ships."""
        accuracy_record = _make_accuracy_record()
        bad_producer = MagicMock()
        bad_producer.process_record = AsyncMock(side_effect=RuntimeError("boom"))
        accuracy_producer = MagicMock()
        accuracy_producer.process_record = AsyncMock(return_value=accuracy_record)

        mock_self = _make_processor_mock(
            producers=[
                ("metric_records", bad_producer),
                ("accuracy", accuracy_producer),
            ]
        )

        await RecordProcessor._process_and_forward_record(
            mock_self, MagicMock(service_id="w1"), MagicMock(), None
        )

        mock_self.records_push_client.push.assert_awaited_once()
        msg = mock_self.records_push_client.push.await_args.args[0]
        assert msg.records == [accuracy_record]


class TestRecordProcessorCreateMetricRecordMetadata:
    """Test the RecordProcessor._create_metric_record_metadata method."""

    @pytest.fixture
    def mock_record_processor(self, cli_config):
        """Create a mock RecordProcessor instance for testing."""
        instance = MagicMock(spec=RecordProcessor)
        instance.service_id = "test-processor-id"
        instance.info = MagicMock()
        return instance

    def test_create_metadata_without_end_and_no_responses(
        self, mock_record_processor, sample_request_record
    ):
        """Test creating metadata when RequestRecord has no end_perf_ns and no responses."""
        sample_request_record.end_perf_ns = None
        sample_request_record.responses = []
        sample_request_record.credit_num = 1
        sample_request_record.credit_phase = CreditPhase.PROFILING
        sample_request_record.recv_start_perf_ns = (
            sample_request_record.start_perf_ns + 10_000
        )

        worker_id = "worker-1"

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, worker_id
        )

        # When no end_perf_ns and no responses, should use start_perf_ns as fallback
        expected_end_ns = sample_request_record.timestamp_ns
        assert metadata.request_start_ns == sample_request_record.timestamp_ns
        assert metadata.request_end_ns == expected_end_ns
        assert metadata.worker_id == worker_id
        assert metadata.record_processor_id == "test-processor-id"

    def test_create_metadata_last_response_perf_ns_takes_precedence(
        self, mock_record_processor, sample_request_record
    ):
        """Test that last_response_perf_ns takes precedence over end_perf_ns."""
        last_response_perf_ns = sample_request_record.start_perf_ns + 150_000
        sample_request_record.end_perf_ns = (
            sample_request_record.start_perf_ns + 200_000
        )
        sample_request_record.credit_num = 2

        worker_id = "worker-2"

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor,
            sample_request_record,
            worker_id,
            last_response_perf_ns=last_response_perf_ns,
        )

        # Should use last_response_perf_ns (not end_perf_ns)
        expected_end_ns = compute_time_ns(
            sample_request_record.timestamp_ns,
            sample_request_record.start_perf_ns,
            last_response_perf_ns,
        )
        assert metadata.request_end_ns == expected_end_ns
        assert metadata.worker_id == worker_id

    def test_create_metadata_with_cancellation(
        self, mock_record_processor, sample_request_record
    ):
        """Test creating metadata for a cancelled request."""
        cancellation_perf_ns = sample_request_record.start_perf_ns + 75_000
        sample_request_record.end_perf_ns = (
            sample_request_record.start_perf_ns + 100_000
        )
        sample_request_record.cancellation_perf_ns = cancellation_perf_ns
        sample_request_record.credit_num = 3

        worker_id = "worker-3"

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, worker_id
        )

        expected_cancellation_time = compute_time_ns(
            sample_request_record.timestamp_ns,
            sample_request_record.start_perf_ns,
            cancellation_perf_ns,
        )
        assert metadata.was_cancelled is True
        assert metadata.cancellation_time_ns == expected_cancellation_time
        assert metadata.worker_id == worker_id

    @pytest.mark.parametrize(
        "field_name,field_value,expected_metadata_field",
        [
            ("recv_start_perf_ns", None, "request_ack_ns"),
        ],
    )
    def test_create_metadata_with_optional_fields_none(
        self,
        mock_record_processor,
        sample_request_record,
        field_name: str,
        field_value,
        expected_metadata_field: str,
    ):
        """Test creating metadata when optional fields are None."""
        setattr(sample_request_record, field_name, field_value)
        sample_request_record.credit_num = 4

        worker_id = "worker-4"

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, worker_id
        )

        assert getattr(metadata, expected_metadata_field) is None
        assert metadata.worker_id == worker_id


class TestRecordProcessorDatasetConfiguredBarrier:
    """The record processor must not process inference results until the
    DatasetConfiguredNotification has been applied to its processors.

    Records (PULL socket) and the notification (SUB socket) arrive on
    independent channels with no ordering guarantee, so processing must block
    on an explicit barrier that _on_dataset_configured releases.
    """

    @pytest.mark.asyncio
    async def test_on_dataset_configured_sets_event(self):
        """_on_dataset_configured must release the barrier once processors are configured."""
        mock_self = MagicMock(spec=RecordProcessor)
        mock_self._dataset_configured_event = asyncio.Event()
        mock_self._producers = []
        mock_self._observers = []

        await RecordProcessor._on_dataset_configured(mock_self, MagicMock())

        assert mock_self._dataset_configured_event.is_set()

    @pytest.mark.asyncio
    async def test_on_inference_results_waits_for_dataset_configured(self):
        """_on_inference_results must block until the dataset is configured, then proceed."""
        mock_self = MagicMock(spec=RecordProcessor)
        mock_self._dataset_configured_event = asyncio.Event()
        # First downstream step after the barrier; the handler swallows
        # processing exceptions (lockstep contract), so assert on the call
        # instead of a raised error.
        mock_self._process_and_forward_record = AsyncMock()

        task = asyncio.create_task(
            RecordProcessor._on_inference_results(mock_self, MagicMock())
        )
        for _ in range(3):
            await asyncio.sleep(0)

        # Barrier not released: processing has not started.
        assert not task.done()
        assert not mock_self._process_and_forward_record.called

        # Barrier released: processing proceeds past the wait.
        mock_self._dataset_configured_event.set()
        await asyncio.wait_for(task, timeout=1.0)
        mock_self._process_and_forward_record.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_inference_results_fails_run_on_config_timeout(self, monkeypatch):
        """On dataset-config timeout, abort the run (report error + kill) rather
        than process the record without a configured dataset."""
        mock_self = MagicMock(spec=RecordProcessor)
        mock_self.service_id = "rp-test"
        mock_self._dataset_configured_event = asyncio.Event()
        mock_self.publish = AsyncMock()
        mock_self._kill = AsyncMock()
        mock_self.inference_result_parser = MagicMock()
        mock_self.inference_result_parser.parse_request_record = AsyncMock()

        async def _raise_timeout(coro, *args, **kwargs):
            coro.close()  # avoid "coroutine was never awaited" warning
            raise TimeoutError

        monkeypatch.setattr(
            "aiperf.records.dataset_gate.asyncio.wait_for", _raise_timeout
        )

        await RecordProcessor._on_inference_results(mock_self, MagicMock())

        # Run is failed loudly ...
        mock_self._kill.assert_awaited_once()
        published = mock_self.publish.await_args.args[0]
        assert isinstance(published, BaseServiceErrorMessage)
        # ... and the record is not processed.
        mock_self.inference_result_parser.parse_request_record.assert_not_called()


class TestRecordProcessorLockstepGuard:
    """The lockstep contract requires that every received inference result
    forwards exactly one RecordsMessage. The error-forward path itself
    must therefore never drop the record, even when metadata creation fails or
    the forward call raises -- otherwise the timeout-less RecordsManager
    completion barrier hangs the run at end-of-phase.
    """

    @pytest.mark.asyncio
    async def test_forward_failed_record_metadata_creation_raises_still_pushes_error(
        self, sample_request_record
    ):
        """If _create_metric_record_metadata raises (e.g. request_info None
        triggering the original failure), _forward_failed_record must fall back
        to minimal metadata and still push exactly one error record."""
        mock_self = MagicMock(spec=RecordProcessor)
        mock_self.service_id = "rp"
        mock_self.records_push_client = AsyncMock()
        mock_self._create_metric_record_metadata = MagicMock(
            side_effect=AttributeError("request_info None")
        )

        await RecordProcessor._forward_failed_record(
            mock_self,
            MagicMock(service_id="w1"),
            sample_request_record,
            None,
            RuntimeError("boom"),
        )

        mock_self.records_push_client.push.assert_awaited_once()
        pushed = mock_self.records_push_client.push.await_args.args[0]
        assert isinstance(pushed, RecordsMessage)
        assert len(pushed.records) == 1
        assert pushed.records[0].metrics == {}
        assert pushed.error is not None

    @pytest.mark.asyncio
    async def test_on_inference_results_forward_failed_record_raises_does_not_escape(
        self,
    ):
        """A failure inside the error-forward path must be swallowed by the
        handler's last-resort guard so it cannot escape _on_inference_results."""
        mock_self = MagicMock(spec=RecordProcessor)
        mock_self._dataset_configured_event = asyncio.Event()
        mock_self._dataset_configured_event.set()
        mock_self._process_and_forward_record = AsyncMock(
            side_effect=RuntimeError("process boom")
        )
        mock_self._forward_failed_record = AsyncMock(
            side_effect=RuntimeError("forward boom")
        )

        await RecordProcessor._on_inference_results(mock_self, MagicMock())

        mock_self._forward_failed_record.assert_awaited_once()

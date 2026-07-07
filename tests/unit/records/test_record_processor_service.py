# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import BaseServiceErrorMessage
from aiperf.common.utils import compute_time_ns
from aiperf.records.record_processor_service import RecordProcessor


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
        mock_self.records_processors = []

        await RecordProcessor._on_dataset_configured(mock_self, MagicMock())

        assert mock_self._dataset_configured_event.is_set()

    @pytest.mark.asyncio
    async def test_on_inference_results_waits_for_dataset_configured(self):
        """_on_inference_results must block until the dataset is configured, then proceed."""
        mock_self = MagicMock(spec=RecordProcessor)
        mock_self._dataset_configured_event = asyncio.Event()
        mock_self.inference_result_parser = MagicMock()
        # First downstream step after the barrier; raising proves the barrier was passed.
        mock_self.inference_result_parser.parse_request_record = AsyncMock(
            side_effect=RuntimeError("REACHED_PROCESSING")
        )

        task = asyncio.create_task(
            RecordProcessor._on_inference_results(mock_self, MagicMock())
        )
        for _ in range(3):
            await asyncio.sleep(0)

        # Barrier not released: processing has not started.
        assert not task.done()
        assert not mock_self.inference_result_parser.parse_request_record.called

        # Barrier released: processing proceeds past the wait.
        mock_self._dataset_configured_event.set()
        with pytest.raises(RuntimeError, match="REACHED_PROCESSING"):
            await asyncio.wait_for(task, timeout=1.0)

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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase, ExportLevel, MemoryMapFormat
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    InferenceResultsMessage,
)
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    MemoryMapClientMetadata,
    MetricInputs,
    ParsedResponseRecord,
    RequestRecord,
    TurnMetadata,
)
from aiperf.common.utils import compute_time_ns
from aiperf.plugin.enums import DatasetSamplingStrategy
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
        # Pre-3a.4 the test set ``sample_request_record.credit_num = 3``; that
        # attribute never existed on RequestRecord (Pydantic ``extra="allow"``
        # silently swallowed it). msgspec.Struct rejects unknown attribute
        # writes -- drop the line, the test never read credit_num back.

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

        worker_id = "worker-4"

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, worker_id
        )

        assert getattr(metadata, expected_metadata_field) is None
        assert metadata.worker_id == worker_id


def _empty_dataset_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        conversations=[],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _metadata_with_turns() -> DatasetMetadata:
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="c1",
                turns=[TurnMetadata(delay_ms=1), TurnMetadata(delay_ms=2)],
            ),
            ConversationMetadata(
                conversation_id="c2",
                turns=[TurnMetadata(delay_ms=3)],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _mmap_client_metadata(fmt: MemoryMapFormat) -> MemoryMapClientMetadata:
    return MemoryMapClientMetadata(
        format=fmt,
        data_file_path=Path("/tmp/test_data.mmap"),
        index_file_path=Path("/tmp/test_index.mmap"),
        conversation_count=0,
        total_size_bytes=0,
    )


class TestRecordProcessorInferenceResults:
    @pytest.mark.asyncio
    async def test_merges_metric_results_before_push_and_warns_on_duplicates(
        self,
    ) -> None:
        service = RecordProcessor.__new__(RecordProcessor)
        service.service_id = "rp1"
        service.run = MagicMock()
        service.run.cfg.artifacts.export_level = ExportLevel.RECORDS
        service.warning = MagicMock()
        service.error = MagicMock()
        service.records_push_client = MagicMock()
        service.records_push_client.push = AsyncMock()
        service.inference_result_parser = MagicMock()
        service.inference_result_parser.parse_request_record = AsyncMock()
        service._process_record = AsyncMock(
            return_value=[{"latency": 1.0}, {"latency": 2.0}, {"tokens": 3}]
        )
        service._free_record_data = MagicMock(return_value=(None, None))

        record = RequestRecord(
            metric_inputs=MetricInputs(
                credit_num=1,
                credit_phase=CreditPhase.PROFILING,
                conversation_id="c1",
                turn_index=0,
                x_request_id="req",
                x_correlation_id="corr",
            )
        )
        parsed = ParsedResponseRecord(request=record, responses=[])
        service.inference_result_parser.parse_request_record.return_value = parsed

        await RecordProcessor._on_inference_results(
            service,
            InferenceResultsMessage(service_id="worker1", record=record),
        )

        pushed = service.records_push_client.push.await_args.args[0]
        assert pushed.metrics == {"latency": 2.0, "tokens": 3}
        assert not hasattr(pushed, "results")
        service.warning.assert_called_once()
        assert "Duplicate metric tag 'latency'" in service.warning.call_args.args[0]


class TestRecordProcessorDatasetConfigured:
    """``RecordProcessor._on_dataset_configured`` opens its own mmap client for
    PAYLOAD_BYTES datasets and skips for CONVERSATION mode.

    Uses ``__new__`` to bypass heavy service construction — we only exercise
    the handler itself, which is the new code under test.
    """

    @pytest.mark.asyncio
    async def test_turn_metadata_index_is_grouped_by_conversation(self) -> None:
        service = RecordProcessor.__new__(RecordProcessor)
        service._dataset_client = None
        service.records_processors = []
        service.debug = MagicMock()
        service.inference_result_parser = MagicMock()

        metadata = _metadata_with_turns()
        client_metadata = _mmap_client_metadata(MemoryMapFormat.CONVERSATION)

        await RecordProcessor._on_dataset_configured(
            service,
            DatasetConfiguredNotification(
                service_id="dataset-manager",
                metadata=metadata,
                client_metadata=client_metadata,
            ),
        )

        expected_turn_metadata = {
            "c1": tuple(metadata.conversations[0].turns),
            "c2": tuple(metadata.conversations[1].turns),
        }
        assert not hasattr(service, "_turn_metadata_by_conversation")
        service.inference_result_parser.on_dataset_configured.assert_called_once_with(
            turn_metadata_by_conversation=expected_turn_metadata,
            dataset_client=None,
        )

    @pytest.mark.asyncio
    async def test_payload_bytes_mmap_opens_dataset_client(self) -> None:
        service = RecordProcessor.__new__(RecordProcessor)
        service._dataset_client = None
        service.records_processors = []
        service.debug = MagicMock()
        service.inference_result_parser = MagicMock()

        fake_client = MagicMock()
        fake_client.initialize = AsyncMock()
        fake_client_cls = MagicMock(return_value=fake_client)

        metadata = _empty_dataset_metadata()
        client_metadata = _mmap_client_metadata(MemoryMapFormat.PAYLOAD_BYTES)

        with patch(
            "aiperf.records.record_processor_service.plugins.get_class",
            return_value=fake_client_cls,
        ):
            await RecordProcessor._on_dataset_configured(
                service,
                DatasetConfiguredNotification(
                    service_id="dataset-manager",
                    metadata=metadata,
                    client_metadata=client_metadata,
                ),
            )

        assert service._dataset_client is fake_client
        fake_client_cls.assert_called_once_with(client_metadata=client_metadata)
        fake_client.initialize.assert_awaited_once_with()
        service.inference_result_parser.on_dataset_configured.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_format_does_not_open_dataset_client(self) -> None:
        service = RecordProcessor.__new__(RecordProcessor)
        service._dataset_client = None
        service.records_processors = []
        service.debug = MagicMock()
        service.inference_result_parser = MagicMock()

        metadata = _empty_dataset_metadata()
        client_metadata = _mmap_client_metadata(MemoryMapFormat.CONVERSATION)

        with patch(
            "aiperf.records.record_processor_service.plugins.get_class"
        ) as mock_get_class:
            await RecordProcessor._on_dataset_configured(
                service,
                DatasetConfiguredNotification(
                    service_id="dataset-manager",
                    metadata=metadata,
                    client_metadata=client_metadata,
                ),
            )

        assert service._dataset_client is None
        mock_get_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_payload_bytes_notification_does_not_reopen_client(
        self,
    ) -> None:
        """Idempotent: a second PAYLOAD_BYTES notification leaves the existing client in place."""
        service = RecordProcessor.__new__(RecordProcessor)
        existing_client = MagicMock()
        existing_client.initialize = AsyncMock()
        service._dataset_client = existing_client
        service.records_processors = []
        service.debug = MagicMock()
        service.inference_result_parser = MagicMock()

        metadata = _empty_dataset_metadata()
        client_metadata = _mmap_client_metadata(MemoryMapFormat.PAYLOAD_BYTES)

        with patch(
            "aiperf.records.record_processor_service.plugins.get_class"
        ) as mock_get_class:
            await RecordProcessor._on_dataset_configured(
                service,
                DatasetConfiguredNotification(
                    service_id="dataset-manager",
                    metadata=metadata,
                    client_metadata=client_metadata,
                ),
            )

        assert service._dataset_client is existing_client
        mock_get_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_dataset_client_releases_resources(self) -> None:
        service = RecordProcessor.__new__(RecordProcessor)
        client = MagicMock()
        client.stop = AsyncMock()
        service._dataset_client = client

        await RecordProcessor._stop_dataset_client(service)

        assert service._dataset_client is None
        client.stop.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_stop_dataset_client_noop_when_no_client(self) -> None:
        service = RecordProcessor.__new__(RecordProcessor)
        service._dataset_client = None

        # Should not raise.
        await RecordProcessor._stop_dataset_client(service)

        assert service._dataset_client is None

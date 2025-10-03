# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock, patch

import orjson
import pytest

from aiperf.common.config import (
    EndpointConfig,
    OutputConfig,
    ServiceConfig,
    UserConfig,
)
from aiperf.common.enums import CreditPhase, EndpointType
from aiperf.common.enums.data_exporter_enums import ExportLevel
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.messages import MetricRecordsMessage
from aiperf.common.models.record_models import (
    MetricRecordInfo,
    MetricRecordMetadata,
    MetricValue,
)
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.post_processors.record_export_results_processor import (
    RecordExportResultsProcessor,
)
from tests.post_processors.conftest import create_metric_records_message


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    """Create a temporary artifact directory for testing."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


@pytest.fixture
def user_config_records_export(tmp_artifact_dir: Path) -> UserConfig:
    """Create a UserConfig with RECORDS export level."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
        ),
        output=OutputConfig(
            artifact_directory=tmp_artifact_dir,
        ),
    )


@pytest.fixture
def service_config() -> ServiceConfig:
    """Create a ServiceConfig for testing."""
    return ServiceConfig()


@pytest.fixture
def sample_metric_records_message():
    """Create a sample MetricRecordsMessage for testing."""
    return create_metric_records_message(
        service_id="processor-1",
        x_request_id="test-record-123",
        conversation_id="conv-456",
        x_correlation_id="test-correlation-123",
        results=[
            {"request_latency_ns": 1_000_000, "output_token_count": 10},
            {"ttft_ns": 500_000},
        ],
    )


class TestRecordExportResultsProcessorInitialization:
    """Test RecordExportResultsProcessor initialization."""

    @pytest.mark.parametrize(
        "export_level, raise_exception",
        [
            (ExportLevel.SUMMARY, True),
            (ExportLevel.RECORDS, False),
            (ExportLevel.RAW, False),
        ],
    )
    def test_init_with_export_level(
        self,
        monkeypatch,
        export_level: ExportLevel,
        raise_exception: bool,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test initialization with various export levels enable or disable the processor."""
        monkeypatch.setattr(
            type(user_config_records_export.output),
            "export_level",
            property(lambda self: export_level),
        )
        if raise_exception:
            with pytest.raises(PostProcessorDisabled):
                _ = RecordExportResultsProcessor(
                    service_id="records-manager",
                    service_config=service_config,
                    user_config=user_config_records_export,
                )
        else:
            processor = RecordExportResultsProcessor(
                service_id="records-manager",
                service_config=service_config,
                user_config=user_config_records_export,
            )

            assert processor.record_count == 0
            assert processor.output_file.name == "profile_export.jsonl"
            assert processor.output_file.parent.exists()

    def test_init_with_raw_export_level(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test initialization with RAW export level enables the processor."""
        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        assert processor.record_count == 0
        assert processor.output_file.name == "profile_export.jsonl"
        assert processor.output_file.parent.exists()

    def test_init_creates_output_directory(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that initialization creates the output directory."""
        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        assert processor.output_file.parent.exists()
        assert processor.output_file.parent.is_dir()

    def test_init_clears_existing_file(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that initialization clears existing output file."""
        # Create a file with existing content
        output_file = (
            user_config_records_export.output.artifact_directory
            / "profile_export.jsonl"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("existing content\n")

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        # File should be cleared or not exist
        if processor.output_file.exists():
            content = processor.output_file.read_text()
            assert content == ""
        else:
            assert not processor.output_file.exists()

    def test_init_sets_show_internal_in_dev_mode(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that show_internal is set based on dev mode."""
        with patch(
            "aiperf.post_processors.record_export_results_processor.AIPERF_DEV_MODE",
            True,
        ):
            service_config.developer.show_internal_metrics = True
            processor = RecordExportResultsProcessor(
                service_id="records-manager",
                service_config=service_config,
                user_config=user_config_records_export,
            )

            assert processor.show_internal is True


class TestRecordExportResultsProcessorProcessResult:
    """Test RecordExportResultsProcessor process_result method."""

    @pytest.mark.asyncio
    async def test_process_result_writes_valid_data(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        sample_metric_records_message: MetricRecordsMessage,
        mock_metric_registry: Mock,
    ):
        """Test that process_result writes valid data to file."""
        mock_display_dict = {
            "request_latency": MetricValue(value=1.0, unit="ms"),
            "output_token_count": MetricValue(value=10, unit="tokens"),
        }

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        with patch.object(
            MetricRecordDict,
            "to_display_dict",
            return_value=mock_display_dict,
        ):
            await processor.process_result(sample_metric_records_message)

        assert processor.record_count == 2  # Two result dicts in the message
        assert processor.output_file.exists()

        # Verify file content
        with open(processor.output_file, "rb") as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            record_dict = orjson.loads(line)
            record = MetricRecordInfo.model_validate(record_dict)
            assert record.metadata.x_request_id == "test-record-123"
            assert record.metadata.conversation_id == "conv-456"
            assert record.metadata.turn_index == 0
            assert record.metadata.worker_id == "worker-1"
            assert record.metadata.record_processor_id == "processor-1"
            assert record.metadata.credit_phase == CreditPhase.PROFILING
            assert record.metadata.timestamp_ns == 1_000_000_000
            assert record.error is None
            assert "request_latency" in record.metrics
            assert "output_token_count" in record.metrics

    @pytest.mark.asyncio
    async def test_process_result_with_empty_display_metrics(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        sample_metric_records_message: MetricRecordsMessage,
        mock_metric_registry: Mock,
    ):
        """Test that process_result skips records with empty display metrics."""
        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        # Mock to_display_dict to return empty dict
        with patch.object(MetricRecordDict, "to_display_dict", return_value={}):
            await processor.process_result(sample_metric_records_message)

        # Should not write anything since display_metrics is empty
        assert processor.record_count == 0
        if processor.output_file.exists():
            content = processor.output_file.read_text()
            assert content == ""

    @pytest.mark.asyncio
    async def test_process_result_handles_errors_gracefully(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        sample_metric_records_message: MetricRecordsMessage,
        mock_metric_registry: Mock,
    ):
        """Test that errors during processing don't raise exceptions."""
        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        # Mock to_display_dict to raise an exception
        with (
            patch.object(
                MetricRecordDict, "to_display_dict", side_effect=Exception("Test error")
            ),
            patch.object(processor, "error") as mock_error,
        ):
            # Should not raise
            await processor.process_result(sample_metric_records_message)

            # Should log the error
            assert mock_error.call_count >= 1

        # Record count should not increment
        assert processor.record_count == 0

    @pytest.mark.asyncio
    async def test_process_result_multiple_messages(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        sample_metric_records_message: MetricRecordsMessage,
        mock_metric_registry: Mock,
    ):
        """Test processing multiple messages accumulates records."""
        mock_display_dict = {
            "request_latency": MetricValue(value=1.0, unit="ms"),
        }

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        with patch.object(
            MetricRecordDict, "to_display_dict", return_value=mock_display_dict
        ):
            # Process 5 messages, each with 2 results
            for i in range(5):
                message = create_metric_records_message(
                    x_request_id=f"record-{i}",
                    conversation_id=f"conv-{i}",
                    turn_index=i,
                    timestamp_ns=1_000_000_000 + i,
                    results=[{"metric1": 100}, {"metric2": 200}],
                )
                await processor.process_result(message)

        assert processor.record_count == 10  # 5 messages * 2 results each
        assert processor.output_file.exists()

        with open(processor.output_file, "rb") as f:
            lines = f.readlines()

        assert len(lines) == 10

        # Verify each line is a valid MetricRecordInfo
        for line in lines:
            record_dict = orjson.loads(line)
            record = MetricRecordInfo.model_validate(record_dict)
            assert isinstance(record, MetricRecordInfo)
            assert record.metadata.x_request_id.startswith("record-")  # type: ignore[union-attr]
            assert "request_latency" in record.metrics


class TestRecordExportResultsProcessorFileFormat:
    """Test RecordExportResultsProcessor file format."""

    @pytest.mark.asyncio
    async def test_output_is_valid_jsonl(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        sample_metric_records_message: MetricRecordsMessage,
        mock_metric_registry: Mock,
    ):
        """Test that output file is valid JSONL format."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        with patch.object(
            MetricRecordDict, "to_display_dict", return_value=mock_display_dict
        ):
            await processor.process_result(sample_metric_records_message)

        with open(processor.output_file, "rb") as f:
            lines = f.readlines()

        for line in lines:
            # Each line should be valid JSON
            record_dict = orjson.loads(line)
            assert isinstance(record_dict, dict)
            # Validate it can be parsed as MetricRecordInfo
            record = MetricRecordInfo.model_validate(record_dict)
            assert isinstance(record, MetricRecordInfo)
            # Check for newline at end
            assert line.endswith(b"\n")

    @pytest.mark.asyncio
    async def test_record_structure_is_complete(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        sample_metric_records_message: MetricRecordsMessage,
        mock_metric_registry: Mock,
    ):
        """Test that each record has the expected structure."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        with patch.object(
            MetricRecordDict, "to_display_dict", return_value=mock_display_dict
        ):
            await processor.process_result(sample_metric_records_message)

        with open(processor.output_file, "rb") as f:
            record_dict = orjson.loads(f.readline())

        # Validate using Pydantic model
        record = MetricRecordInfo.model_validate(record_dict)

        # Check top-level structure
        assert isinstance(record.metadata, MetricRecordMetadata)
        assert isinstance(record.metrics, dict)

        # Check metadata structure
        assert record.metadata.conversation_id is not None
        assert isinstance(record.metadata.turn_index, int)
        assert isinstance(record.metadata.timestamp_ns, int)
        assert isinstance(record.metadata.worker_id, str)
        assert isinstance(record.metadata.record_processor_id, str)
        assert isinstance(record.metadata.credit_phase, CreditPhase)

        # Check metrics structure
        assert "test_metric" in record.metrics
        assert isinstance(record.metrics["test_metric"], MetricValue)
        assert record.metrics["test_metric"].value == 42
        assert record.metrics["test_metric"].unit == "ms"


class TestRecordExportResultsProcessorLogging:
    """Test RecordExportResultsProcessor logging behavior."""

    @pytest.mark.asyncio
    async def test_periodic_debug_logging(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test that debug logging occurs every 100 records."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        with (
            patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ),
            patch.object(processor, "debug") as mock_debug,
        ):
            # Process 100 records (50 messages with 2 results each)
            for i in range(50):
                message = create_metric_records_message(
                    x_request_id=f"record-{i}",
                    conversation_id=f"conv-{i}",
                    turn_index=i,
                    timestamp_ns=1_000_000_000 + i,
                    results=[{"metric1": 100}, {"metric2": 200}],
                )
                await processor.process_result(message)

            # Should be called once at record 100
            assert mock_debug.call_count == 1
            call_args = str(mock_debug.call_args)
            assert "100" in call_args

    @pytest.mark.asyncio
    async def test_error_logging_on_write_failure(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        sample_metric_records_message: MetricRecordsMessage,
        mock_metric_registry: Mock,
    ):
        """Test that errors are logged when write fails."""
        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        with (
            patch.object(
                MetricRecordDict, "to_display_dict", side_effect=OSError("Disk full")
            ),
            patch.object(processor, "error") as mock_error,
        ):
            await processor.process_result(sample_metric_records_message)

            # Should log errors (one per result in message)
            assert mock_error.call_count >= 1
            call_args = str(mock_error.call_args_list[0])
            assert "Failed to write record metrics" in call_args


class TestRecordExportResultsProcessorShutdown:
    """Test RecordExportResultsProcessor shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_logs_statistics(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        sample_metric_records_message: MetricRecordsMessage,
        mock_metric_registry: Mock,
    ):
        """Test that shutdown logs final statistics."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        with patch.object(
            MetricRecordDict, "to_display_dict", return_value=mock_display_dict
        ):
            # Process some records
            for i in range(3):
                message = create_metric_records_message(
                    x_request_id=f"record-{i}",
                    conversation_id=f"conv-{i}",
                    turn_index=i,
                    timestamp_ns=1_000_000_000 + i,
                    results=[{"metric1": 100}],
                )
                await processor.process_result(message)

        with patch.object(processor, "info") as mock_info:
            await processor._shutdown()

            mock_info.assert_called_once()
            call_args = str(mock_info.call_args)
            assert "3 records written" in call_args or "3" in call_args


class TestRecordExportResultsProcessorSummarize:
    """Test RecordExportResultsProcessor summarize method."""

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_list(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that summarize returns an empty list (no aggregation needed)."""
        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        result = await processor.summarize()

        assert result == []
        assert isinstance(result, list)

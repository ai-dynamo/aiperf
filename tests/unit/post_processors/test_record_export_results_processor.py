# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
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
from aiperf.common.enums import CreditPhase, ExportLevel
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.messages import MetricRecordsMessage
from aiperf.common.models.record_models import (
    MetricRecordInfo,
    MetricRecordMetadata,
    MetricValue,
)
from aiperf.common.models.trace_models import AioHttpTraceData
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.record_export_results_processor import (
    RecordExportResultsProcessor,
)
from tests.unit.post_processors.conftest import (
    aiperf_lifecycle,
    create_metric_records_message,
)


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
        """Test init with various export levels enable or disable the processor."""
        user_config_records_export.output.export_level = export_level
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

            assert processor.lines_written == 0
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

        assert processor.lines_written == 0
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
        with (
            patch.object(Environment.DEV, "MODE", True),
            patch.object(Environment.DEV, "SHOW_INTERNAL_METRICS", True),
            patch.object(Environment.DEV, "SHOW_EXPERIMENTAL_METRICS", False),
        ):
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

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict,
                "to_display_dict",
                return_value=mock_display_dict,
            ):
                await processor.process_result(sample_metric_records_message.to_data())

        lines = processor.output_file.read_text().splitlines()

        assert len(lines) == 1
        record_dict = orjson.loads(lines[0])
        record = MetricRecordInfo.model_validate(record_dict)
        assert record.metadata.x_request_id == "test-record-123"
        assert record.metadata.conversation_id == "conv-456"
        assert record.metadata.turn_index == 0
        assert record.metadata.worker_id == "worker-1"
        assert record.metadata.record_processor_id == "processor-1"
        assert record.metadata.benchmark_phase == CreditPhase.PROFILING
        assert record.metadata.request_start_ns == 1_000_000_000
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
            await processor.process_result(sample_metric_records_message.to_data())

        # Should not write anything since display_metrics is empty
        assert processor.lines_written == 0
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
            await processor.process_result(sample_metric_records_message.to_data())

            # Should log the error
            assert mock_error.call_count >= 1

        # Record count should not increment
        assert processor.lines_written == 0

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

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                for i in range(5):
                    message = create_metric_records_message(
                        x_request_id=f"record-{i}",
                        conversation_id=f"conv-{i}",
                        turn_index=i,
                        request_start_ns=1_000_000_000 + i,
                        results=[{"metric1": 100}, {"metric2": 200}],
                    )
                    await processor.process_result(message.to_data())

        assert processor.lines_written == 5
        assert processor.output_file.exists()

        lines = processor.output_file.read_text().splitlines()

        assert len(lines) == 5

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

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(sample_metric_records_message.to_data())

        lines = processor.output_file.read_text().splitlines()

        for line in lines:
            if line.strip():
                record_dict = orjson.loads(line)
                assert isinstance(record_dict, dict)
                record = MetricRecordInfo.model_validate(record_dict)
                assert isinstance(record, MetricRecordInfo)

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

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(sample_metric_records_message.to_data())

        lines = processor.output_file.read_text().splitlines()

        for line in lines:
            record_dict = orjson.loads(line)
            record = MetricRecordInfo.model_validate(record_dict)

            assert isinstance(record.metadata, MetricRecordMetadata)
            assert isinstance(record.metrics, dict)

            assert record.metadata.conversation_id is not None
            assert isinstance(record.metadata.turn_index, int)
            assert isinstance(record.metadata.request_start_ns, int)
            assert isinstance(record.metadata.worker_id, str)
            assert isinstance(record.metadata.record_processor_id, str)
            assert isinstance(record.metadata.benchmark_phase, CreditPhase)

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
        caplog,
    ):
        """Test that debug logging occurs when buffer is flushed."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                with caplog.at_level(logging.DEBUG):
                    for i in range(processor._batch_size):
                        message = create_metric_records_message(
                            x_request_id=f"record-{i}",
                            conversation_id=f"conv-{i}",
                            turn_index=i,
                            request_start_ns=1_000_000_000 + i,
                            results=[{"metric1": 100}, {"metric2": 200}],
                        )
                        await processor.process_result(message.to_data())

                    # Wait for async flush task to complete
                    await processor.wait_for_tasks()

                # Check that flushing debug message was logged
                assert any("Flushing" in record.message for record in caplog.records)

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
            await processor.process_result(sample_metric_records_message.to_data())

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

        await processor.initialize()
        await processor.start()

        try:
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                for i in range(3):
                    message = create_metric_records_message(
                        x_request_id=f"record-{i}",
                        conversation_id=f"conv-{i}",
                        turn_index=i,
                        request_start_ns=1_000_000_000 + i,
                        results=[{"metric1": 100}],
                    )
                    await processor.process_result(message.to_data())

                # Wait for any pending flush tasks
                await processor.wait_for_tasks()

            await processor.stop()

            # Check stats were logged during shutdown by verifying lines_written
            assert processor.lines_written == 3, (
                f"Expected 3 records written, but got {processor.lines_written}"
            )
        except Exception:
            await processor.stop()
            raise


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


class TestRecordExportResultsProcessorHttpTrace:
    """Test RecordExportResultsProcessor HTTP trace export functionality."""

    @pytest.fixture
    def user_config_with_http_trace(self, tmp_artifact_dir: Path) -> UserConfig:
        """Create a UserConfig with export_http_trace enabled."""
        return UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
            ),
            output=OutputConfig(
                artifact_directory=tmp_artifact_dir,
                export_http_trace=True,
            ),
        )

    @pytest.fixture
    def sample_trace_data(self) -> AioHttpTraceData:
        """Create a sample AioHttpTraceData object for testing.

        This creates a realistic trace data object with all phases populated:
        - Request send: 1000000000 -> 1000100000 (100us sending)
        - Waiting: 1000100000 -> 1050100000 (50ms TTFB)
        - Response receive: 1050100000 -> 1100000000 (49.9ms receiving)
        """
        base_perf_ns = 1000000000
        return AioHttpTraceData(
            trace_type="aiohttp",
            # Reference timestamps for wall-clock conversion
            reference_time_ns=1700000000000000000,  # Wall-clock reference
            reference_perf_ns=base_perf_ns,
            # Request send phase
            request_send_start_perf_ns=base_perf_ns,
            request_headers={"Content-Type": "application/json"},
            request_headers_sent_perf_ns=base_perf_ns + 50000,
            request_chunks=[
                (base_perf_ns + 100000, 1024)
            ],  # 100us after start, 1KB sent
            # Response receive phase
            response_status_code=200,
            response_reason="OK",
            response_headers_received_perf_ns=base_perf_ns + 50000000,
            response_receive_start_perf_ns=base_perf_ns + 50100000,
            response_chunks=[
                (base_perf_ns + 50100000, 512),  # First chunk at 50.1ms
                (base_perf_ns + 100000000, 256),  # Last chunk at 100ms
            ],
            response_receive_end_perf_ns=base_perf_ns + 100000000,
            # Connection info
            local_ip="127.0.0.1",
            local_port=54321,
            remote_ip="127.0.0.1",
            remote_port=8000,
        )

    def test_init_default_http_trace_disabled(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that export_http_trace defaults to False."""
        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        assert processor.export_http_trace is False

    def test_init_http_trace_enabled(
        self,
        user_config_with_http_trace: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that export_http_trace can be enabled via config."""
        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_http_trace,
        )

        assert processor.export_http_trace is True

    def test_init_logs_when_http_trace_enabled(
        self,
        user_config_with_http_trace: UserConfig,
        service_config: ServiceConfig,
        caplog,
    ):
        """Test that initialization logs when HTTP trace export is enabled."""
        with caplog.at_level(logging.INFO):
            _ = RecordExportResultsProcessor(
                service_id="records-manager",
                service_config=service_config,
                user_config=user_config_with_http_trace,
            )

        assert any("--export-http-trace" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_trace_data_excluded_when_disabled(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
        sample_trace_data: AioHttpTraceData,
    ):
        """Test that trace_data is NOT in output when export_http_trace=False."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        # Create message with trace_data
        message = create_metric_records_message(
            x_request_id="test-record-with-trace",
            conversation_id="conv-trace-1",
            results=[{"test_metric": 42}],
            trace_data=sample_trace_data,
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(message.to_data())

        lines = processor.output_file.read_text().splitlines()
        assert len(lines) == 1

        record_dict = orjson.loads(lines[0])
        record = MetricRecordInfo.model_validate(record_dict)

        # Verify trace_data is NOT in the output
        assert record.trace_data is None
        # But metrics are still present
        assert "test_metric" in record.metrics

    @pytest.mark.asyncio
    async def test_trace_data_included_when_enabled(
        self,
        user_config_with_http_trace: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
        sample_trace_data: AioHttpTraceData,
    ):
        """Test that trace_data IS included in output when export_http_trace=True."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_http_trace,
        )

        # Create message with trace_data
        message = create_metric_records_message(
            x_request_id="test-record-with-trace",
            conversation_id="conv-trace-2",
            results=[{"test_metric": 42}],
            trace_data=sample_trace_data,
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(message.to_data())

        lines = processor.output_file.read_text().splitlines()
        assert len(lines) == 1

        record_dict = orjson.loads(lines[0])
        record = MetricRecordInfo.model_validate(record_dict)

        # Verify trace_data IS in the output
        assert record.trace_data is not None
        assert record.trace_data.trace_type == "aiohttp"
        # sending_ns = request_send_end - request_send_start = 100000 ns
        assert record.trace_data.sending_ns == 100000
        # Metrics are also present
        assert "test_metric" in record.metrics

    @pytest.mark.asyncio
    async def test_metrics_always_present_regardless_of_trace_flag(
        self,
        user_config_records_export: UserConfig,
        user_config_with_http_trace: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
        sample_trace_data: AioHttpTraceData,
    ):
        """Test metrics are always included regardless of export_http_trace setting."""
        mock_display_dict = {
            "request_latency": MetricValue(value=100.5, unit="ms"),
            "output_token_count": MetricValue(value=50, unit="tokens"),
        }

        # Test with trace disabled
        processor_disabled = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        # Test with trace enabled
        processor_enabled = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_http_trace,
        )

        for processor in [processor_disabled, processor_enabled]:
            message = create_metric_records_message(
                x_request_id="test-record-metrics",
                conversation_id="conv-metrics",
                results=[{"request_latency_ns": 100_500_000, "output_token_count": 50}],
                trace_data=sample_trace_data,
            )

            async with aiperf_lifecycle(processor):
                with patch.object(
                    MetricRecordDict, "to_display_dict", return_value=mock_display_dict
                ):
                    await processor.process_result(message.to_data())

            lines = processor.output_file.read_text().splitlines()
            assert len(lines) == 1

            record_dict = orjson.loads(lines[0])
            record = MetricRecordInfo.model_validate(record_dict)

            # Metrics should always be present
            assert "request_latency" in record.metrics
            assert "output_token_count" in record.metrics
            assert record.metrics["request_latency"].value == 100.5
            assert record.metrics["output_token_count"].value == 50

    @pytest.mark.asyncio
    async def test_no_trace_data_when_record_has_none(
        self,
        user_config_with_http_trace: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test trace_data is null when record has no trace data (even if enabled)."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_with_http_trace,
        )

        # Create message WITHOUT trace_data
        message = create_metric_records_message(
            x_request_id="test-record-no-trace",
            conversation_id="conv-no-trace",
            results=[{"test_metric": 42}],
            # No trace_data provided
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(message.to_data())

        lines = processor.output_file.read_text().splitlines()
        assert len(lines) == 1

        record_dict = orjson.loads(lines[0])
        record = MetricRecordInfo.model_validate(record_dict)

        # trace_data should be None since the record had no trace data
        assert record.trace_data is None


class TestRecordExportResultsProcessorLifecycle:
    """Test RecordExportResultsProcessor lifecycle."""

    @pytest.mark.asyncio
    async def test_lifecycle(
        self,
        user_config_records_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
        mock_aiofiles_stringio,
    ):
        """Test that the processor can be initialized, processed, and shutdown."""
        processor = RecordExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_records_export,
        )

        assert processor._file_handle is None
        await processor.initialize()
        assert processor._file_handle is not None
        await processor.start()

        mock_display_dict = {"inter_token_latency": MetricValue(value=100, unit="ms")}

        try:
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                for i in range(Environment.RECORD.EXPORT_BATCH_SIZE * 2):
                    await processor.process_result(
                        create_metric_records_message(
                            x_request_id=f"record-{i}",
                            conversation_id=f"conv-{i}",
                            turn_index=0,
                            request_start_ns=1_000_000_000 + i,
                            results=[{"inter_token_latency": 100}],
                        ).to_data()
                    )

                # Wait for all async flush tasks to complete
                await processor.wait_for_tasks()
        finally:
            await processor.stop()

        assert processor.lines_written == Environment.RECORD.EXPORT_BATCH_SIZE * 2

        contents = mock_aiofiles_stringio.getvalue()
        lines = contents.splitlines()
        assert contents.endswith(b"\n"), (
            f"Contents should end with newline but got: {repr(contents[-20:])}"
        )
        assert len(lines) == Environment.RECORD.EXPORT_BATCH_SIZE * 2

        for i, line in enumerate(lines):
            record = MetricRecordInfo.model_validate_json(line)
            assert record.metadata.x_request_id == f"record-{i}"
            assert record.metadata.conversation_id == f"conv-{i}"
            assert record.metadata.turn_index == 0
            assert "inter_token_latency" in record.metrics

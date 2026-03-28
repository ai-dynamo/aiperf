# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import csv
import io
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.record_models import MetricValue
from aiperf.config import AIPerfConfig
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.record_export_csv_processor import (
    RecordExportCSVProcessor,
)
from tests.unit.post_processors.conftest import (
    _make_run,
    aiperf_lifecycle,
    create_metric_records_message,
)

_MINIMAL_DATASETS = {
    "default": {
        "type": "synthetic",
        "entries": 100,
        "prompts": {"isl": 128, "osl": 64},
    }
}
_MINIMAL_PHASES = {"default": {"type": "concurrency", "requests": 10, "concurrency": 1}}


def _make_csv_config(tmp_artifact_dir: Path, **artifacts_overrides) -> AIPerfConfig:
    """Create an AIPerfConfig with CSV records export enabled."""
    artifacts = {
        "dir": str(tmp_artifact_dir),
        "records": ["csv", "jsonl"],
        **artifacts_overrides,
    }
    return AIPerfConfig(
        models=["test-model"],
        endpoint={
            "urls": ["http://localhost:8000/v1/chat/completions"],
            "type": EndpointType.CHAT,
        },
        datasets=_MINIMAL_DATASETS,
        phases=_MINIMAL_PHASES,
        artifacts=artifacts,
    )


async def wait_for_flush_tasks(processor) -> None:
    """Wait for one-shot flush tasks without blocking on periodic background tasks."""
    while True:
        flush_tasks = [
            task
            for task in processor.tasks
            if task.get_coro().__name__ == "_csv_flush_buffer"
        ]
        if not flush_tasks:
            return
        await asyncio.gather(*flush_tasks)


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    """Create a temporary artifact directory for testing."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


@pytest.fixture
def csv_config(tmp_artifact_dir: Path) -> AIPerfConfig:
    """Create an AIPerfConfig with CSV records export enabled."""
    return _make_csv_config(tmp_artifact_dir)


@pytest.fixture
def sample_metric_records_message():
    """Create a sample MetricRecordsData for testing."""
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


def _parse_csv_output(output_file: Path) -> list[dict[str, str]]:
    """Parse a CSV output file into a list of dicts."""
    content = output_file.read_text()
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


class TestRecordExportCSVProcessorInitialization:
    """Test RecordExportCSVProcessor initialization."""

    @pytest.mark.parametrize(
        "records,raise_exception",
        [
            (False, True),
            (["csv"], False),
            (["csv", "jsonl"], False),
        ],
    )  # fmt: skip
    def test_init_with_records_config(
        self,
        records,
        raise_exception: bool,
        tmp_artifact_dir: Path,
    ):
        """Test init with various records config enable or disable the processor."""
        config = _make_csv_config(tmp_artifact_dir, records=records)
        if raise_exception:
            with pytest.raises(PostProcessorDisabled):
                _ = RecordExportCSVProcessor(
                    service_id="records-manager",
                    run=_make_run(config),
                )
        else:
            processor = RecordExportCSVProcessor(
                service_id="records-manager",
                run=_make_run(config),
            )

            assert processor.rows_written == 0
            assert processor.output_file.name == "profile_export_records.csv"
            assert processor.output_file.parent.exists()

    def test_init_disabled_when_csv_not_in_formats(
        self,
        tmp_artifact_dir: Path,
    ):
        """Test that processor is disabled when csv is not in records formats."""
        config = _make_csv_config(tmp_artifact_dir, records=["jsonl"])
        with pytest.raises(PostProcessorDisabled):
            _ = RecordExportCSVProcessor(
                service_id="records-manager",
                run=_make_run(config),
            )

    @pytest.mark.asyncio
    async def test_init_creates_output_directory(
        self,
        csv_config: AIPerfConfig,
    ):
        """Test that @on_init creates the output directory."""
        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )
        await processor.initialize()
        assert processor.output_file.parent.exists()
        assert processor.output_file.parent.is_dir()

    @pytest.mark.asyncio
    async def test_init_clears_existing_file(
        self,
        csv_config: AIPerfConfig,
    ):
        """Test that @on_init clears existing output file."""
        output_file = csv_config.artifacts.dir / "profile_export_records.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("existing content\n")

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )
        await processor.initialize()

        assert processor.output_file.exists()
        content = processor.output_file.read_text()
        assert content == ""


class TestRecordExportCSVProcessorProcessResult:
    """Test RecordExportCSVProcessor process_result method."""

    @pytest.mark.asyncio
    async def test_process_result_writes_valid_csv(
        self,
        csv_config: AIPerfConfig,
        sample_metric_records_message,
        mock_metric_registry: Mock,
    ):
        """Test that process_result writes valid CSV data."""
        mock_display_dict = {
            "request_latency": MetricValue(value=1.0, unit="ms"),
            "output_token_count": MetricValue(value=10, unit="tokens"),
        }

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict,
                "to_display_dict",
                return_value=mock_display_dict,
            ):
                await processor.process_result(sample_metric_records_message)

        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 1
        row = rows[0]
        assert row["x_request_id"] == "test-record-123"
        assert row["conversation_id"] == "conv-456"
        assert row["worker_id"] == "worker-1"
        assert row["record_processor_id"] == "processor-1"
        assert row["benchmark_phase"] == "profiling"
        assert row["request_start_ns"] == "1000000000"
        assert "request_latency (ms)" in row
        assert "output_token_count (tokens)" in row
        assert row["request_latency (ms)"] == "1"
        assert row["output_token_count (tokens)"] == "10"

    @pytest.mark.asyncio
    async def test_process_result_with_empty_display_metrics(
        self,
        csv_config: AIPerfConfig,
        sample_metric_records_message,
        mock_metric_registry: Mock,
    ):
        """Test that process_result skips records with empty display metrics."""
        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        with patch.object(MetricRecordDict, "to_display_dict", return_value={}):
            await processor.process_result(sample_metric_records_message)

        assert processor.rows_written == 0

    @pytest.mark.asyncio
    async def test_process_result_handles_errors_gracefully(
        self,
        csv_config: AIPerfConfig,
        sample_metric_records_message,
        mock_metric_registry: Mock,
    ):
        """Test that errors during processing don't raise exceptions."""
        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        with (
            patch.object(
                MetricRecordDict, "to_display_dict", side_effect=Exception("Test error")
            ),
            patch.object(processor, "error") as mock_error,
        ):
            await processor.process_result(sample_metric_records_message)
            assert mock_error.call_count >= 1

        assert processor.rows_written == 0

    @pytest.mark.asyncio
    async def test_process_result_multiple_messages(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test processing multiple messages accumulates records."""
        mock_display_dict = {
            "request_latency": MetricValue(value=1.0, unit="ms"),
        }

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
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
                        results=[{"metric1": 100}],
                    )
                    await processor.process_result(message)

        assert processor.rows_written == 5
        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 5

        for i, row in enumerate(rows):
            assert row["x_request_id"] == f"record-{i}"
            assert row["conversation_id"] == f"conv-{i}"
            assert "request_latency (ms)" in row


class TestRecordExportCSVProcessorFileFormat:
    """Test RecordExportCSVProcessor file format."""

    @pytest.mark.asyncio
    async def test_output_is_valid_csv(
        self,
        csv_config: AIPerfConfig,
        sample_metric_records_message,
        mock_metric_registry: Mock,
    ):
        """Test that output file is valid CSV format."""
        mock_display_dict = {"request_latency": MetricValue(value=42, unit="ms")}

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(sample_metric_records_message)

        content = processor.output_file.read_text()
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 data row
        assert "request_latency (ms)" in rows[0]

    @pytest.mark.asyncio
    async def test_csv_has_correct_metadata_columns(
        self,
        csv_config: AIPerfConfig,
        sample_metric_records_message,
        mock_metric_registry: Mock,
    ):
        """Test that CSV has all expected metadata columns."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(sample_metric_records_message)

        content = processor.output_file.read_text()
        reader = csv.reader(io.StringIO(content))
        headers = next(reader)

        expected_metadata = [
            "request_num",
            "session_num",
            "x_request_id",
            "x_correlation_id",
            "conversation_id",
            "turn_index",
            "benchmark_phase",
            "worker_id",
            "record_processor_id",
            "credit_issued_ns",
            "request_start_ns",
            "request_ack_ns",
            "request_end_ns",
            "was_cancelled",
            "cancellation_time_ns",
            "error_code",
            "error_message",
        ]
        for col in expected_metadata:
            assert col in headers

    @pytest.mark.asyncio
    async def test_csv_request_num_and_session_num_values(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test that request_num and session_num values appear correctly in CSV rows."""
        mock_display_dict = {"test_metric": MetricValue(value=1.0, unit="ms")}

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                for i in range(3):
                    msg = create_metric_records_message(
                        request_num=i,
                        session_num=i
                        // 2,  # session 0 for req 0,1; session 1 for req 2
                    )
                    await processor.process_result(msg)

        content = processor.output_file.read_text()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["request_num"] == "0"
        assert rows[0]["session_num"] == "0"
        assert rows[1]["request_num"] == "1"
        assert rows[1]["session_num"] == "0"
        assert rows[2]["request_num"] == "2"
        assert rows[2]["session_num"] == "1"


class TestRecordExportCSVProcessorErrorRecords:
    """Test CSV export of error records."""

    @pytest.mark.asyncio
    async def test_error_record_exported_with_error_columns(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test that error records include error_code and error_message columns."""
        error = ErrorDetails(code=500, message="Internal server error")
        mock_display_dict = {"request_latency": MetricValue(value=0.0, unit="ms")}

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        message = create_metric_records_message(
            x_request_id="error-record-1",
            conversation_id="conv-error",
            results=[{"request_latency_ns": 0}],
            error=error,
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(message)

        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 1
        row = rows[0]
        assert row["error_code"] == "500"
        assert row["error_message"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_error_record_exported_even_with_no_metrics(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test that error records are exported even when display_metrics is empty."""
        error = ErrorDetails(code=503, message="Service unavailable")

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        message = create_metric_records_message(
            x_request_id="error-no-metrics",
            conversation_id="conv-error-2",
            results=[],
            error=error,
        )

        async with aiperf_lifecycle(processor):
            with patch.object(MetricRecordDict, "to_display_dict", return_value={}):
                await processor.process_result(message)

        assert processor.rows_written == 1
        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 1
        assert rows[0]["error_code"] == "503"
        assert rows[0]["error_message"] == "Service unavailable"


class TestRecordExportCSVProcessorSummarize:
    """Test RecordExportCSVProcessor summarize method."""

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_list(
        self,
        csv_config: AIPerfConfig,
    ):
        """Test that summarize returns an empty list (no aggregation needed)."""
        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        result = await processor.summarize()
        assert result == []
        assert isinstance(result, list)


class TestRecordExportCSVProcessorShutdown:
    """Test RecordExportCSVProcessor shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_flushes_remaining_records(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test that shutdown flushes remaining buffered records."""
        mock_display_dict = {"test_metric": MetricValue(value=42, unit="ms")}

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
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
                        turn_index=0,
                        request_start_ns=1_000_000_000 + i,
                        results=[{"test_metric": 42}],
                    )
                    await processor.process_result(message)

                await wait_for_flush_tasks(processor)
        finally:
            await processor.stop()

        assert processor.rows_written == 3


class TestRecordExportCSVProcessorLifecycle:
    """Test RecordExportCSVProcessor lifecycle."""

    @pytest.mark.asyncio
    async def test_lifecycle(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
        mock_aiofiles_stringio,
    ):
        """Test that the processor can be initialized, processed, and shutdown."""
        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        assert processor._csv_file_handle is None
        await processor.initialize()
        assert processor._csv_file_handle is not None
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
                        )
                    )

                await wait_for_flush_tasks(processor)
        finally:
            await processor.stop()

        assert processor.rows_written == Environment.RECORD.EXPORT_BATCH_SIZE * 2

        contents = mock_aiofiles_stringio.getvalue()
        assert contents.endswith(b"\n")

        lines = contents.decode("utf-8").strip().split("\n")
        # 1 header + N data rows
        assert len(lines) == Environment.RECORD.EXPORT_BATCH_SIZE * 2 + 1

        # Verify header
        reader = csv.reader(io.StringIO(lines[0]))
        header = next(reader)
        assert "session_num" in header
        assert "inter_token_latency (ms)" in header


class TestRecordExportCSVProcessorMetricFormatting:
    """Test metric value formatting in CSV output."""

    @pytest.mark.asyncio
    async def test_float_metric_formatting(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test that float metric values are formatted with reasonable precision."""
        mock_display_dict = {
            "request_latency": MetricValue(value=123.456789, unit="ms"),
            "time_to_first_token": MetricValue(value=0.001234, unit="ms"),
        }

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        message = create_metric_records_message(
            x_request_id="format-test",
            conversation_id="conv-format",
            results=[{"request_latency_ns": 123_456_789}],
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(message)

        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 1
        assert rows[0]["request_latency (ms)"] == "123.457"
        assert rows[0]["time_to_first_token (ms)"] == "0.001234"

    @pytest.mark.asyncio
    async def test_integer_metric_formatting(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test that integer metric values are formatted without decimal points."""
        mock_display_dict = {
            "output_token_count": MetricValue(value=42, unit="tokens"),
        }

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        message = create_metric_records_message(
            x_request_id="int-test",
            conversation_id="conv-int",
            results=[{"output_token_count": 42}],
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(message)

        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 1
        assert rows[0]["output_token_count (tokens)"] == "42"


class TestRecordExportCSVProcessorPerChunkData:
    """Test CSV export per-chunk data filtering."""

    @pytest.mark.asyncio
    async def test_list_metrics_excluded_by_default(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test that list-valued metrics are excluded when per_chunk_data is False (default)."""
        mock_display_dict = {
            "request_latency": MetricValue(value=1.0, unit="ms"),
            "inter_chunk_latency": MetricValue(value=[0.1, 0.2, 0.3], unit="ms"),
        }

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )
        assert processor.export_per_chunk_data is False

        message = create_metric_records_message(
            x_request_id="no-chunk-test",
            conversation_id="conv-no-chunk",
            results=[{"request_latency_ns": 1_000_000}],
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(message)

        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 1
        headers = list(rows[0].keys())
        assert "request_latency (ms)" in headers
        assert "inter_chunk_latency (ms)" not in headers

    @pytest.mark.asyncio
    async def test_list_metrics_included_when_enabled(
        self,
        tmp_artifact_dir: Path,
        mock_metric_registry: Mock,
    ):
        """Test that list-valued metrics are included when per_chunk_data is True."""
        config = _make_csv_config(tmp_artifact_dir, per_chunk_data=True)

        mock_display_dict = {
            "request_latency": MetricValue(value=1.0, unit="ms"),
            "inter_chunk_latency": MetricValue(value=[0.1, 0.2, 0.3], unit="ms"),
        }

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(config),
        )
        assert processor.export_per_chunk_data is True

        message = create_metric_records_message(
            x_request_id="chunk-test",
            conversation_id="conv-chunk",
            results=[{"request_latency_ns": 1_000_000}],
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=mock_display_dict
            ):
                await processor.process_result(message)

        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 1
        headers = list(rows[0].keys())
        assert "request_latency (ms)" in headers
        assert "inter_chunk_latency (ms)" in headers
        assert rows[0]["inter_chunk_latency (ms)"] == "[0.1, 0.2, 0.3]"


class TestRecordExportCSVProcessorMixedMetrics:
    """Test CSV export handles records with different metric sets."""

    @pytest.mark.asyncio
    async def test_error_record_before_success_includes_all_columns(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test that an error record (fewer metrics) followed by a success record
        produces a CSV where both rows have all metric columns from the registry."""
        error_display = {
            "request_latency": MetricValue(value=0.0, unit="ms"),
        }
        success_display = {
            "request_latency": MetricValue(value=5.0, unit="ms"),
            "time_to_first_token": MetricValue(value=1.5, unit="ms"),
            "output_token_count": MetricValue(value=42, unit="tokens"),
        }

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        error_msg = create_metric_records_message(
            x_request_id="error-1",
            conversation_id="conv-err",
            results=[{"request_latency_ns": 0}],
            error=ErrorDetails(code=500, message="fail"),
        )
        success_msg = create_metric_records_message(
            x_request_id="success-1",
            conversation_id="conv-ok",
            results=[
                {"request_latency_ns": 5_000_000},
                {"time_to_first_token_ns": 1_500_000},
                {"output_token_count": 42},
            ],
        )

        async with aiperf_lifecycle(processor):
            # Error record arrives first with only request_latency
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=error_display
            ):
                await processor.process_result(error_msg)
            # Success record arrives with more metrics
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=success_display
            ):
                await processor.process_result(success_msg)

        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 2

        # All columns come from the registry, so both rows have the same headers
        headers = list(rows[0].keys())
        assert "request_latency (ms)" in headers
        assert "time_to_first_token (ms)" in headers
        assert "output_token_count (tokens)" in headers

        # Error record should have empty cells for metrics it didn't have
        assert rows[0]["request_latency (ms)"] == "0"
        assert rows[0]["time_to_first_token (ms)"] == ""
        assert rows[0]["output_token_count (tokens)"] == ""

        # Success record should have all metric values
        assert rows[1]["request_latency (ms)"] == "5"
        assert rows[1]["time_to_first_token (ms)"] == "1.5"
        assert rows[1]["output_token_count (tokens)"] == "42"

    @pytest.mark.asyncio
    async def test_success_record_before_error_includes_all_columns(
        self,
        csv_config: AIPerfConfig,
        mock_metric_registry: Mock,
    ):
        """Test that a success record followed by an error record (fewer metrics)
        produces a CSV where both rows have all metric columns."""
        success_display = {
            "request_latency": MetricValue(value=5.0, unit="ms"),
            "time_to_first_token": MetricValue(value=1.5, unit="ms"),
        }
        error_display = {
            "request_latency": MetricValue(value=0.0, unit="ms"),
        }

        processor = RecordExportCSVProcessor(
            service_id="records-manager",
            run=_make_run(csv_config),
        )

        success_msg = create_metric_records_message(
            x_request_id="success-1",
            conversation_id="conv-ok",
            results=[
                {"request_latency_ns": 5_000_000},
                {"time_to_first_token_ns": 1_500_000},
            ],
        )
        error_msg = create_metric_records_message(
            x_request_id="error-1",
            conversation_id="conv-err",
            results=[{"request_latency_ns": 0}],
            error=ErrorDetails(code=503, message="unavailable"),
        )

        async with aiperf_lifecycle(processor):
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=success_display
            ):
                await processor.process_result(success_msg)
            with patch.object(
                MetricRecordDict, "to_display_dict", return_value=error_display
            ):
                await processor.process_result(error_msg)

        rows = _parse_csv_output(processor.output_file)
        assert len(rows) == 2

        headers = list(rows[0].keys())
        assert "request_latency (ms)" in headers
        assert "time_to_first_token (ms)" in headers

        # Success record has all metrics
        assert rows[0]["request_latency (ms)"] == "5"
        assert rows[0]["time_to_first_token (ms)"] == "1.5"

        # Error record has only request_latency, time_to_first_token is empty
        assert rows[1]["request_latency (ms)"] == "0"
        assert rows[1]["time_to_first_token (ms)"] == ""
        assert rows[1]["error_code"] == "503"

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest

from aiperf.common.config import EndpointConfig, OutputConfig, UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.records.raw_record_writer import RawRecordWriter


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    """Create a temporary artifact directory for testing."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


@pytest.fixture
def user_config_with_tmp_dir(tmp_artifact_dir: Path) -> UserConfig:
    """Create a UserConfig with temporary artifact directory."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
        ),
        output=OutputConfig(artifact_directory=tmp_artifact_dir),
    )


@pytest.fixture
def sample_metrics() -> list[MetricRecordDict]:
    """Create sample metric results for testing."""
    return [
        MetricRecordDict({"request_latency_ns": 1_000_000}),
        MetricRecordDict({"output_token_throughput": 50.5}),
    ]


class TestRawRecordWriterInitialization:
    """Test RawRecordWriter initialization."""

    def test_init_with_service_id(self, user_config_with_tmp_dir: UserConfig):
        """Test initialization with a provided service_id."""
        writer = RawRecordWriter(
            service_id="test-processor-1",
            user_config=user_config_with_tmp_dir,
        )

        assert writer.service_id == "test-processor-1"
        assert writer.record_count == 0
        assert writer.output_file.name == "raw_records_test-processor-1.jsonl"
        assert writer.output_file.parent.exists()
        assert writer.output_file.parent.name == "raw_records"

    def test_init_with_none_service_id(self, user_config_with_tmp_dir: UserConfig):
        """Test initialization with None service_id defaults to 'processor'."""
        writer = RawRecordWriter(
            service_id=None,
            user_config=user_config_with_tmp_dir,
        )

        assert writer.service_id == "processor"
        assert writer.output_file.name == "raw_records_processor.jsonl"

    def test_init_sanitizes_special_characters(
        self, user_config_with_tmp_dir: UserConfig
    ):
        """Test that special characters in service_id are sanitized for filename."""
        writer = RawRecordWriter(
            service_id="worker/process:123 test",
            user_config=user_config_with_tmp_dir,
        )

        assert writer.output_file.name == "raw_records_worker_process_123_test.jsonl"

    def test_init_creates_output_directory(self, user_config_with_tmp_dir: UserConfig):
        """Test that initialization creates the output directory."""
        raw_records_dir = (
            user_config_with_tmp_dir.output.artifact_directory / "raw_records"
        )
        assert not raw_records_dir.exists()

        _ = RawRecordWriter(
            service_id="test",
            user_config=user_config_with_tmp_dir,
        )

        assert raw_records_dir.exists()
        assert raw_records_dir.is_dir()


class TestRawRecordWriterWriteRecord:
    """Test RawRecordWriter write_record method."""

    @pytest.mark.asyncio
    async def test_write_record_with_valid_data(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
        sample_metrics: list[MetricRecordDict],
    ):
        """Test writing a record with valid data."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        await writer.write_record(
            parsed_record=sample_parsed_record,
            metric_results=sample_metrics,
            worker_id="worker-1",
        )

        assert writer.record_count == 1
        assert writer.output_file.exists()

        with open(writer.output_file, "rb") as f:
            line = f.readline()
            record = orjson.loads(line)

        assert record["worker_id"] == "worker-1"
        assert record["processor_id"] == "test-processor"
        assert "parsed_record" in record
        assert "metrics" in record
        assert record["metrics"]["request_latency_ns"] == 1_000_000
        assert record["metrics"]["output_token_throughput"] == 50.5

    @pytest.mark.asyncio
    async def test_write_record_filters_exceptions(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ):
        """Test that exceptions in metric_results are filtered out."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        metric_results: list[MetricRecordDict | BaseException] = [
            MetricRecordDict({"valid_metric": 100}),
            ValueError("This should be filtered"),
            MetricRecordDict({"another_metric": 200}),
            RuntimeError("Also filtered"),
        ]

        await writer.write_record(
            parsed_record=sample_parsed_record,
            metric_results=metric_results,
            worker_id="worker-1",
        )

        assert writer.record_count == 1

        with open(writer.output_file, "rb") as f:
            line = f.readline()
            record = orjson.loads(line)

        assert len(record["metrics"]) == 2
        assert record["metrics"]["valid_metric"] == 100
        assert record["metrics"]["another_metric"] == 200

    @pytest.mark.asyncio
    async def test_write_record_with_empty_metrics(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ):
        """Test writing a record with empty metrics list."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        await writer.write_record(
            parsed_record=sample_parsed_record,
            metric_results=[],
            worker_id="worker-1",
        )

        assert writer.record_count == 1

        with open(writer.output_file, "rb") as f:
            line = f.readline()
            record = orjson.loads(line)

        assert record["metrics"] == {}

    @pytest.mark.asyncio
    async def test_write_record_handles_errors_gracefully(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
        sample_metrics: list[MetricRecordDict],
    ):
        """Test that write errors don't raise exceptions (fail gracefully)."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        with patch("aiofiles.open", side_effect=OSError("Disk full")):
            # Should not raise an exception
            await writer.write_record(
                parsed_record=sample_parsed_record,
                metric_results=sample_metrics,
                worker_id="worker-1",
            )

        assert writer.record_count == 0

    @pytest.mark.asyncio
    async def test_write_multiple_records(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
        sample_metrics: list[MetricRecordDict],
    ):
        """Test writing multiple records and record counting."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        num_records = 5
        for i in range(num_records):
            await writer.write_record(
                parsed_record=sample_parsed_record,
                metric_results=sample_metrics,
                worker_id=f"worker-{i}",
            )

        assert writer.record_count == num_records
        assert writer.output_file.exists()

        with open(writer.output_file, "rb") as f:
            lines = f.readlines()

        assert len(lines) == num_records
        for i, line in enumerate(lines):
            record = orjson.loads(line)
            assert record["worker_id"] == f"worker-{i}"


class TestRawRecordWriterFileFormat:
    """Test RawRecordWriter file format and structure."""

    @pytest.mark.asyncio
    async def test_output_is_valid_jsonl(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
        sample_metrics: list[MetricRecordDict],
    ):
        """Test that output file is valid JSONL format."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        await writer.write_record(
            parsed_record=sample_parsed_record,
            metric_results=sample_metrics,
            worker_id="worker-1",
        )
        await writer.write_record(
            parsed_record=sample_parsed_record,
            metric_results=sample_metrics,
            worker_id="worker-2",
        )

        with open(writer.output_file, "rb") as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            # Each line should be valid JSON
            record = orjson.loads(line)
            assert isinstance(record, dict)
            # Check for newline at end
            assert line.endswith(b"\n")

    @pytest.mark.asyncio
    async def test_record_structure_is_complete(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
        sample_metrics: list[MetricRecordDict],
    ):
        """Test that each record has the expected structure."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        await writer.write_record(
            parsed_record=sample_parsed_record,
            metric_results=sample_metrics,
            worker_id="worker-1",
        )

        with open(writer.output_file, "rb") as f:
            record = orjson.loads(f.readline())

        assert set(record.keys()) == {
            "worker_id",
            "processor_id",
            "parsed_record",
            "metrics",
        }
        assert isinstance(record["worker_id"], str)
        assert isinstance(record["processor_id"], str)
        assert isinstance(record["parsed_record"], dict)
        assert isinstance(record["metrics"], dict)

        # Verify parsed_record has expected structure
        assert "request" in record["parsed_record"]
        assert "responses" in record["parsed_record"]
        assert "input_token_count" in record["parsed_record"]
        assert "output_token_count" in record["parsed_record"]


class TestRawRecordWriterClose:
    """Test RawRecordWriter close method."""

    @pytest.mark.asyncio
    async def test_close_logs_statistics(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
        sample_metrics: list[MetricRecordDict],
    ):
        """Test that close method logs final statistics."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        # Write some records
        for i in range(3):
            await writer.write_record(
                parsed_record=sample_parsed_record,
                metric_results=sample_metrics,
                worker_id=f"worker-{i}",
            )

        with patch.object(writer, "info") as mock_info:
            await writer.close()

            mock_info.assert_called_once()
            call_args = str(mock_info.call_args)
            assert "3 records written" in call_args or "3" in call_args

    @pytest.mark.asyncio
    async def test_close_with_zero_records(
        self,
        user_config_with_tmp_dir: UserConfig,
    ):
        """Test that close works correctly with zero records written."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        with patch.object(writer, "info") as mock_info:
            await writer.close()

            mock_info.assert_called_once()
            call_args = str(mock_info.call_args)
            assert "0 records written" in call_args or "0" in call_args


class TestRawRecordWriterLogging:
    """Test RawRecordWriter logging behavior."""

    @pytest.mark.asyncio
    async def test_periodic_debug_logging(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
        sample_metrics: list[MetricRecordDict],
    ):
        """Test that debug logging occurs every 100 records."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        with patch.object(writer, "debug") as mock_debug:
            # Write 100 records
            for i in range(100):
                await writer.write_record(
                    parsed_record=sample_parsed_record,
                    metric_results=sample_metrics,
                    worker_id=f"worker-{i}",
                )

            # Should be called once at record 100
            assert mock_debug.call_count == 1
            call_args = str(mock_debug.call_args)
            assert "100" in call_args

        with patch.object(writer, "debug") as mock_debug:
            # Write 50 more records (total 150)
            for i in range(50):
                await writer.write_record(
                    parsed_record=sample_parsed_record,
                    metric_results=sample_metrics,
                    worker_id=f"worker-{i}",
                )

            # Should not be called (not at a multiple of 100)
            assert mock_debug.call_count == 0

    @pytest.mark.asyncio
    async def test_error_logging_on_write_failure(
        self,
        user_config_with_tmp_dir: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
        sample_metrics: list[MetricRecordDict],
    ):
        """Test that errors are logged when write fails."""
        writer = RawRecordWriter(
            service_id="test-processor",
            user_config=user_config_with_tmp_dir,
        )

        with (
            patch.object(writer, "error") as mock_error,
            patch("aiofiles.open", side_effect=OSError("Disk full")),
        ):
            await writer.write_record(
                parsed_record=sample_parsed_record,
                metric_results=sample_metrics,
                worker_id="worker-1",
            )

            mock_error.assert_called_once()
            call_args = str(mock_error.call_args)
            assert "Failed to write raw record" in call_args

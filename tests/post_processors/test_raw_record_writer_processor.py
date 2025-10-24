# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.common.config import UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import CreditPhase
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.models.record_models import RawRecordInfo
from aiperf.post_processors.raw_record_writer_processor import (
    RawRecordAggregator,
    RawRecordWriterProcessor,
)
from tests.post_processors.conftest import (
    create_exporter_config,
    create_metric_metadata,
    raw_record_processor,
)


# Alias for backward compatibility and clearer intent
@pytest.fixture
def sample_parsed_record(sample_parsed_record_with_raw_responses: ParsedResponseRecord):
    """Alias for sample_parsed_record_with_raw_responses."""
    return sample_parsed_record_with_raw_responses


class TestRawRecordWriterProcessorInitialization:
    """Test RawRecordWriterProcessor initialization."""

    def test_init_creates_output_directory(self, user_config_raw: UserConfig):
        """Test that initialization creates the raw_records directory."""
        processor = RawRecordWriterProcessor(
            service_id="processor-1",
            user_config=user_config_raw,
        )

        expected_dir = (
            user_config_raw.output.artifact_directory
            / OutputDefaults.RAW_RECORDS_FOLDER
        )
        assert expected_dir.exists()
        assert expected_dir.is_dir()
        assert processor.output_file.parent == expected_dir

    @pytest.mark.parametrize(
        "service_id,expected_filename",
        [
            ("simple", "raw_records_simple.jsonl"),
            ("with/slash", "raw_records_with_slash.jsonl"),
            ("with:colon", "raw_records_with_colon.jsonl"),
            ("with space", "raw_records_with_space.jsonl"),
            ("complex/mix:of chars", "raw_records_complex_mix_of_chars.jsonl"),
        ],
    )
    def test_filename_sanitization(
        self,
        user_config_raw: UserConfig,
        service_id: str,
        expected_filename: str,
    ):
        """Test various service_id sanitization scenarios."""
        processor = RawRecordWriterProcessor(
            service_id=service_id,
            user_config=user_config_raw,
        )
        assert processor.output_file.name == expected_filename

    def test_init_with_none_service_id(self, user_config_raw: UserConfig):
        """Test initialization with None service_id defaults to 'processor'."""
        processor = RawRecordWriterProcessor(
            service_id=None,
            user_config=user_config_raw,
        )

        assert processor.service_id == "processor"
        assert processor.output_file.name == "raw_records_processor.jsonl"


class TestRawRecordWriterProcessorProcessRecord:
    """Test RawRecordWriterProcessor process_record method."""

    @pytest.mark.asyncio
    async def test_process_record_writes_valid_data(
        self,
        user_config_raw: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ):
        """Test that process_record writes valid raw data to file."""
        async with raw_record_processor("processor-1", user_config_raw) as processor:
            metadata = create_metric_metadata(
                session_num=0,
                conversation_id="conv-123",
                x_request_id="req-123",
                x_correlation_id="corr-123",
            )

            await processor.process_record(sample_parsed_record, metadata)

        assert processor.output_file.exists()
        lines = processor.output_file.read_text().splitlines()

        assert len(lines) == 1
        record_dict = orjson.loads(lines[0])
        record = RawRecordInfo.model_validate(record_dict)

        assert record.metadata.conversation_id == "conv-123"
        assert record.metadata.x_request_id == "req-123"
        assert record.metadata.x_correlation_id == "corr-123"
        assert record.status == 200
        assert record.payload is not None
        assert record.request_headers == {"Content-Type": "application/json"}
        assert record.error is None
        assert len(record.responses) == 2

    @pytest.mark.asyncio
    async def test_process_record_with_error(
        self,
        user_config_raw: UserConfig,
        error_parsed_record: ParsedResponseRecord,
    ):
        """Test that process_record handles error records correctly."""
        async with raw_record_processor("processor-1", user_config_raw) as processor:
            metadata = create_metric_metadata(
                session_num=0,
                conversation_id="conv-error",
            )

            await processor.process_record(error_parsed_record, metadata)

        record_dict = orjson.loads(processor.output_file.read_text().splitlines()[0])

        record = RawRecordInfo.model_validate(record_dict)
        assert record.metadata.conversation_id == "conv-error"
        assert record.status == 500
        assert record.error is not None
        assert record.error.code == 500
        assert record.error.message == "Internal server error"
        assert len(record.responses) == 0

    @pytest.mark.asyncio
    async def test_process_multiple_records(
        self,
        user_config_raw: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ):
        """Test processing multiple records."""
        async with raw_record_processor("processor-1", user_config_raw) as processor:
            for i in range(5):
                metadata = create_metric_metadata(
                    session_num=i,
                    conversation_id=f"conv-{i}",
                    x_request_id=f"req-{i}",
                )
                await processor.process_record(sample_parsed_record, metadata)

        assert processor.lines_written == 5
        lines = processor.output_file.read_text().splitlines()

        assert len(lines) == 5
        for i, line in enumerate(lines):
            record = RawRecordInfo.model_validate(orjson.loads(line))
            assert record.metadata.session_num == i
            assert record.metadata.conversation_id == f"conv-{i}"
            assert record.metadata.x_request_id == f"req-{i}"


class TestRawRecordWriterProcessorFileFormat:
    """Test RawRecordWriterProcessor file format."""

    @pytest.mark.asyncio
    async def test_output_is_valid_jsonl_and_record_structure(
        self,
        user_config_raw: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ):
        """Test that output is valid JSONL format and record structure is complete."""
        async with raw_record_processor("processor-1", user_config_raw) as processor:
            metadata = create_metric_metadata(
                conversation_id="test-conv",
                turn_index=2,
            )

            await processor.process_record(sample_parsed_record, metadata)

        lines = processor.output_file.read_text().splitlines()

        assert len(lines) == 1
        line = lines[0]

        # Verify format: valid JSON and ends with newline
        record_dict = orjson.loads(line)
        assert isinstance(record_dict, dict)

        # Verify structure
        record = RawRecordInfo.model_validate(record_dict)
        assert record.metadata.conversation_id == "test-conv"
        assert record.metadata.turn_index == 2
        assert isinstance(record.metadata.request_start_ns, int)
        assert isinstance(record.metadata.benchmark_phase, CreditPhase)
        assert isinstance(record.start_perf_ns, int)
        assert isinstance(record.payload, dict)
        assert isinstance(record.status, int)
        assert isinstance(record.responses, list)


class TestRawRecordAggregator:
    """Test RawRecordAggregator functionality."""

    @pytest.mark.asyncio
    async def test_aggregator_combines_multiple_files(
        self,
        user_config_raw: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ):
        """Test that aggregator combines multiple processor files."""
        # Create multiple processor files
        for i in range(3):
            async with raw_record_processor(
                f"processor-{i}", user_config_raw
            ) as processor:
                for j in range(2):
                    metadata = create_metric_metadata(
                        session_num=i * 2 + j,
                        conversation_id=f"conv-{i}-{j}",
                    )
                    await processor.process_record(sample_parsed_record, metadata)

        # Run aggregator
        exporter_config = create_exporter_config(user_config_raw)
        aggregator = RawRecordAggregator(exporter_config=exporter_config)

        await aggregator.export()

        # Verify aggregated output
        assert aggregator.output_file.exists()
        lines = aggregator.output_file.read_text().splitlines()

        assert len(lines) == 6  # 3 processors * 2 records each

        # Verify raw_records directory is cleaned up
        raw_records_dir = (
            user_config_raw.output.artifact_directory
            / OutputDefaults.RAW_RECORDS_FOLDER
        )
        assert not raw_records_dir.exists()

    @pytest.mark.asyncio
    async def test_aggregator_with_no_files(self, user_config_raw: UserConfig):
        """Test that aggregator handles no input files gracefully."""
        exporter_config = create_exporter_config(user_config_raw)
        aggregator = RawRecordAggregator(exporter_config=exporter_config)

        # Should not raise
        await aggregator.export()

        # Output file should not be created
        assert not aggregator.output_file.exists()

    @pytest.mark.asyncio
    async def test_aggregator_skips_empty_lines(
        self,
        user_config_raw: UserConfig,
    ):
        """Test that aggregator skips empty lines in input files."""
        # Create a processor file with some empty lines
        raw_records_dir = (
            user_config_raw.output.artifact_directory
            / OutputDefaults.RAW_RECORDS_FOLDER
        )
        raw_records_dir.mkdir(parents=True, exist_ok=True)

        test_file = raw_records_dir / "raw_records_test.jsonl"
        with open(test_file, "w") as f:
            f.write('{"metadata": {"session_num": 0}}\n')
            f.write("\n")
            f.write('{"metadata": {"session_num": 1}}\n')
            f.write("   \n")
            f.write('{"metadata": {"session_num": 2}}\n')

        exporter_config = create_exporter_config(user_config_raw)
        aggregator = RawRecordAggregator(exporter_config=exporter_config)

        await aggregator.export()

        # Verify only non-empty lines are counted
        assert aggregator.output_file.exists()
        lines = aggregator.output_file.read_text().splitlines()

        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_aggregator_clears_existing_output(
        self,
        user_config_raw: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ):
        """Test that aggregator clears existing output file."""
        # Create existing output file
        output_file = user_config_raw.output.profile_export_raw_jsonl_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("old content\n")

        # Create a processor file
        async with raw_record_processor("processor-1", user_config_raw) as processor:
            metadata = create_metric_metadata()
            await processor.process_record(sample_parsed_record, metadata)

        # Run aggregator
        exporter_config = create_exporter_config(user_config_raw)
        aggregator = RawRecordAggregator(exporter_config=exporter_config)
        await aggregator.export()

        # Verify old content is gone
        content = output_file.read_text()
        assert "old content" not in content
        assert content.strip()  # Has new content

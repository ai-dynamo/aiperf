# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import orjson
import pytest

from aiperf.common.config import (
    EndpointConfig,
    OutputConfig,
    ServiceConfig,
    UserConfig,
)
from aiperf.common.environment import Environment
from aiperf.common.models.telemetry_models import (
    TelemetryMetrics,
    TelemetryRecord,
)
from aiperf.gpu_telemetry.jsonl_writer import (
    GPUTelemetryJSONLWriter,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.post_processors.conftest import aiperf_lifecycle, make_telemetry_record


@pytest.fixture
def user_config_telemetry_export(tmp_artifact_dir: Path) -> UserConfig:
    """Create a UserConfig for telemetry export testing."""
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
def sample_telemetry_record() -> TelemetryRecord:
    """Create a sample TelemetryRecord with all fields populated."""
    return make_telemetry_record(
        timestamp_ns=1_000_000_000,
        dcgm_url="http://node1:9401/metrics",
        gpu_index=0,
        gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
        gpu_model_name="NVIDIA RTX 6000 Ada Generation",
        pci_bus_id="00000000:02:00.0",
        device="nvidia0",
        hostname="node1",
        gpu_power_usage=75.5,
        energy_consumption=1000.0,
        gpu_utilization=85.0,
        gpu_memory_used=15.26,
        gpu_temperature=70.0,
        xid_errors=None,
        power_violation=0.0,
    )


@pytest.fixture
def sample_telemetry_record_partial() -> TelemetryRecord:
    """Create a sample TelemetryRecord with some fields None."""
    return make_telemetry_record(
        timestamp_ns=2_000_000_000,
        dcgm_url="http://node2:9401/metrics",
        gpu_index=1,
        gpu_uuid="GPU-a1b2c3d4-e5f6-7890-abcd-1234567890ab",
        gpu_model_name="NVIDIA H100 80GB HBM3",
        pci_bus_id=None,
        device=None,
        hostname="node2",
        gpu_power_usage=150.0,
        energy_consumption=None,
        gpu_utilization=95.0,
        gpu_memory_used=70.0,
        gpu_temperature=85.0,
        xid_errors=None,
        power_violation=None,
    )


class TestGPUTelemetryJSONLWriterInitialization:
    """Test GPUTelemetryJSONLWriter initialization."""

    def test_initialization(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that processor initializes with correct output file path."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        assert processor.lines_written == 0
        assert processor.output_file.name == "gpu_telemetry_export.jsonl"
        assert processor.output_file.parent.exists()

    def test_creates_output_directory(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that initialization creates the output directory."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        assert processor.output_file.parent.exists()
        assert processor.output_file.parent.is_dir()

    @pytest.mark.asyncio
    async def test_clears_existing_file(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that initialization clears existing output file."""
        # Create a file with existing content
        output_file = (
            user_config_telemetry_export.output.artifact_directory
            / "gpu_telemetry_export.jsonl"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("existing content\n")

        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )
        await processor.initialize()

        try:
            # File should be cleared or not exist
            if processor.output_file.exists():
                content = processor.output_file.read_text()
                assert content == ""
            else:
                assert not processor.output_file.exists()
        finally:
            await processor.stop()

    def test_sets_batch_size_from_environment(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that batch_size is set from Environment.RECORD.EXPORT_BATCH_SIZE."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        assert processor._batch_size == Environment.RECORD.EXPORT_BATCH_SIZE

    def test_logs_initialization_message(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        caplog,
    ):
        """Test that initialization logs info message about telemetry export."""
        with caplog.at_level(logging.INFO):
            GPUTelemetryJSONLWriter(
                service_id="records-manager",
                service_config=service_config,
                user_config=user_config_telemetry_export,
            )

            assert any(
                "GPU telemetry export enabled" in record.message
                for record in caplog.records
            )
            assert any(
                "gpu_telemetry_export.jsonl" in record.message
                for record in caplog.records
            )


class TestGPUTelemetryJSONLWriterProcessing:
    """Test GPUTelemetryJSONLWriter processing methods."""

    @pytest.mark.asyncio
    async def test_process_telemetry_record_writes_to_buffer(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that process_telemetry_record buffers the record correctly."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record)

        assert processor.lines_written == 1
        lines = processor.output_file.read_text().splitlines()
        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_process_telemetry_record_with_complete_data(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test processing a telemetry record with all fields populated."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record)

        lines = processor.output_file.read_text().splitlines()
        record_dict = orjson.loads(lines[0])
        record = TelemetryRecord.model_validate(record_dict)

        assert record.timestamp_ns == 1_000_000_000
        assert record.dcgm_url == "http://node1:9401/metrics"
        assert record.gpu_index == 0
        assert record.gpu_uuid == "GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc"
        assert record.gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert record.telemetry_data.gpu_power_usage == 75.5
        assert record.telemetry_data.gpu_utilization == 85.0

    @pytest.mark.asyncio
    async def test_process_telemetry_record_with_partial_data(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record_partial: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test processing a telemetry record with some fields None."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record_partial)

        lines = processor.output_file.read_text().splitlines()
        record_dict = orjson.loads(lines[0])
        record = TelemetryRecord.model_validate(record_dict)

        assert record.timestamp_ns == 2_000_000_000
        assert record.pci_bus_id is None
        assert record.device is None
        assert record.telemetry_data.energy_consumption is None
        assert record.telemetry_data.power_violation is None

    @pytest.mark.asyncio
    async def test_process_multiple_telemetry_records(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test processing multiple telemetry records sequentially."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        records = [
            make_telemetry_record(
                timestamp_ns=1_000_000_000 + i * 1_000_000,
                gpu_uuid="GPU-test-uuid",
                gpu_power_usage=100.0 + i,
                gpu_utilization=80.0 + i,
            )
            for i in range(5)
        ]

        async with aiperf_lifecycle(processor):
            for record in records:
                await processor.process_telemetry_record(record)

        assert processor.lines_written == 5
        lines = processor.output_file.read_text().splitlines()
        assert len(lines) == 5

    @pytest.mark.asyncio
    async def test_process_telemetry_record_handles_exceptions(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that exceptions during processing are caught and logged."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        with (
            patch.object(
                processor, "buffered_write", side_effect=Exception("Test error")
            ),
            patch.object(processor, "error") as mock_error,
        ):
            await processor.process_telemetry_record(sample_telemetry_record)
            assert mock_error.call_count >= 1
            call_args = str(mock_error.call_args_list[0])
            assert "Failed to write GPU telemetry record" in call_args

        # lines_written should not increment
        assert processor.lines_written == 0

    @pytest.mark.asyncio
    async def test_buffer_auto_flush_at_batch_size(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test that buffer auto-flushes when batch_size is reached."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        batch_size = processor._batch_size

        async with aiperf_lifecycle(processor):
            for i in range(batch_size * 2):
                record = make_telemetry_record(
                    timestamp_ns=1_000_000_000 + i,
                    gpu_power_usage=100.0 + i,
                )
                await processor.process_telemetry_record(record)

            await processor.wait_for_tasks()

        assert processor.lines_written == batch_size * 2

    @pytest.mark.asyncio
    async def test_multiple_gpus_same_endpoint(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test processing records from multiple GPUs on same endpoint."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        records = [
            make_telemetry_record(
                gpu_index=i,
                gpu_uuid=f"GPU-{i}",
                gpu_model_name=f"GPU {i}",
                gpu_power_usage=100.0 + i,
            )
            for i in range(4)
        ]

        async with aiperf_lifecycle(processor):
            for record in records:
                await processor.process_telemetry_record(record)

        assert processor.lines_written == 4
        lines = processor.output_file.read_text().splitlines()

        # Verify each GPU is represented
        gpu_indices = [orjson.loads(line)["gpu_index"] for line in lines]
        assert gpu_indices == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_multiple_endpoints(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test processing records from different DCGM endpoints."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        records = [
            make_telemetry_record(
                dcgm_url=f"http://node{i}:9401/metrics",
                gpu_uuid=f"GPU-node{i}",
                hostname=f"node{i}",
            )
            for i in range(3)
        ]

        async with aiperf_lifecycle(processor):
            for record in records:
                await processor.process_telemetry_record(record)

        lines = processor.output_file.read_text().splitlines()
        dcgm_urls = [orjson.loads(line)["dcgm_url"] for line in lines]

        assert "http://node0:9401/metrics" in dcgm_urls
        assert "http://node1:9401/metrics" in dcgm_urls
        assert "http://node2:9401/metrics" in dcgm_urls

    @pytest.mark.asyncio
    async def test_records_written_count(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test that lines_written counter increments correctly."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        assert processor.lines_written == 0

        async with aiperf_lifecycle(processor):
            for i in range(10):
                record = make_telemetry_record(
                    timestamp_ns=1_000_000_000 + i,
                    gpu_power_usage=100.0 + i,
                )
                await processor.process_telemetry_record(record)

            await processor.wait_for_tasks()

        assert processor.lines_written == 10


class TestGPUTelemetryJSONLWriterFileFormat:
    """Test GPUTelemetryJSONLWriter file format validation."""

    @pytest.mark.asyncio
    async def test_output_is_valid_jsonl(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that output file is valid JSONL format."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record)

        lines = processor.output_file.read_text().splitlines()

        for line in lines:
            if line.strip():
                record_dict = orjson.loads(line)
                assert isinstance(record_dict, dict)
                record = TelemetryRecord.model_validate(record_dict)
                assert isinstance(record, TelemetryRecord)

    @pytest.mark.asyncio
    async def test_record_structure_is_complete(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that each record has the expected structure."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record)

        lines = processor.output_file.read_text().splitlines()

        for line in lines:
            record_dict = orjson.loads(line)
            record = TelemetryRecord.model_validate(record_dict)

            assert isinstance(record.timestamp_ns, int)
            assert isinstance(record.dcgm_url, str)
            assert isinstance(record.gpu_index, int)
            assert isinstance(record.gpu_uuid, str)
            assert isinstance(record.gpu_model_name, str)
            assert isinstance(record.telemetry_data, TelemetryMetrics)

    @pytest.mark.asyncio
    async def test_preserves_all_telemetry_fields(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that all telemetry fields are serialized correctly."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record)

        lines = processor.output_file.read_text().splitlines()
        record_dict = orjson.loads(lines[0])
        record = TelemetryRecord.model_validate(record_dict)

        # Check all metadata fields
        assert record.timestamp_ns == sample_telemetry_record.timestamp_ns
        assert record.dcgm_url == sample_telemetry_record.dcgm_url
        assert record.gpu_index == sample_telemetry_record.gpu_index
        assert record.gpu_uuid == sample_telemetry_record.gpu_uuid
        assert record.gpu_model_name == sample_telemetry_record.gpu_model_name
        assert record.pci_bus_id == sample_telemetry_record.pci_bus_id
        assert record.device == sample_telemetry_record.device
        assert record.hostname == sample_telemetry_record.hostname

        # Check telemetry data fields
        assert (
            record.telemetry_data.gpu_power_usage
            == sample_telemetry_record.telemetry_data.gpu_power_usage
        )
        assert (
            record.telemetry_data.gpu_utilization
            == sample_telemetry_record.telemetry_data.gpu_utilization
        )
        assert (
            record.telemetry_data.gpu_memory_used
            == sample_telemetry_record.telemetry_data.gpu_memory_used
        )

    @pytest.mark.asyncio
    async def test_handles_none_values(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record_partial: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that None values are handled correctly in serialization."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record_partial)

        lines = processor.output_file.read_text().splitlines()
        record_dict = orjson.loads(lines[0])

        # Verify None values are not present in the dict
        # Verify None values are not present in the serialized dict
        assert "pci_bus_id" not in record_dict
        assert "device" not in record_dict
        assert "energy_consumption" not in record_dict["telemetry_data"]

    @pytest.mark.asyncio
    async def test_timestamp_precision(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test that nanosecond timestamps are preserved with full precision."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        precise_timestamp = 1_234_567_890_123_456_789
        record = make_telemetry_record(timestamp_ns=precise_timestamp)

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(record)

        lines = processor.output_file.read_text().splitlines()
        record_dict = orjson.loads(lines[0])

        assert record_dict["timestamp_ns"] == precise_timestamp

    @pytest.mark.asyncio
    async def test_metadata_fields_present(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that all required metadata fields are present in output."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record)

        lines = processor.output_file.read_text().splitlines()
        record_dict = orjson.loads(lines[0])

        # Check all required metadata fields
        assert "timestamp_ns" in record_dict
        assert "dcgm_url" in record_dict
        assert "gpu_index" in record_dict
        assert "gpu_uuid" in record_dict
        assert "gpu_model_name" in record_dict
        assert "hostname" in record_dict
        assert "telemetry_data" in record_dict

    @pytest.mark.asyncio
    async def test_hierarchical_identifiers(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that records contain proper identifiers for hierarchical access."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record)

        lines = processor.output_file.read_text().splitlines()
        record_dict = orjson.loads(lines[0])

        # Verify hierarchical keys are present
        assert record_dict["dcgm_url"] == "http://node1:9401/metrics"
        assert record_dict["gpu_uuid"] == "GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc"
        assert record_dict["timestamp_ns"] == 1_000_000_000


class TestGPUTelemetryJSONLWriterSummarize:
    """Test GPUTelemetryJSONLWriter summarize method."""

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_list(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that summarize returns an empty list (no aggregation needed)."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        result = await processor.summarize()

        assert result == []
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_summarize_after_processing_records(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that summarize returns empty list even after processing records."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record)

        result = await processor.summarize()

        assert result == []
        assert isinstance(result, list)


class TestGPUTelemetryJSONLWriterLifecycle:
    """Test GPUTelemetryJSONLWriter lifecycle."""

    @pytest.mark.asyncio
    async def test_lifecycle_with_mock_aiofiles(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
        mock_aiofiles_stringio,
    ):
        """Test full lifecycle using mock_aiofiles_stringio."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        assert processor._file_handle is None
        await processor.initialize()
        assert processor._file_handle is not None
        await processor.start()

        try:
            for i in range(Environment.RECORD.EXPORT_BATCH_SIZE * 2):
                record = make_telemetry_record(
                    timestamp_ns=1_000_000_000 + i,
                    gpu_power_usage=100.0 + i,
                    gpu_utilization=80.0,
                )
                await processor.process_telemetry_record(record)

            await processor.wait_for_tasks()
        finally:
            await processor.stop()

        assert processor.lines_written == Environment.RECORD.EXPORT_BATCH_SIZE * 2

        contents = mock_aiofiles_stringio.getvalue()
        lines = contents.splitlines()
        assert contents.endswith(b"\n")
        assert len(lines) == Environment.RECORD.EXPORT_BATCH_SIZE * 2

        for i, line in enumerate(lines):
            record = TelemetryRecord.model_validate_json(line)
            assert record.timestamp_ns == 1_000_000_000 + i
            assert record.gpu_uuid == "GPU-test"
            assert record.telemetry_data.gpu_power_usage == 100.0 + i

    @pytest.mark.asyncio
    async def test_file_handle_lifecycle(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that _file_handle is managed correctly through lifecycle."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        # Initially None
        assert processor._file_handle is None

        # After initialize, should be open
        await processor.initialize()
        assert processor._file_handle is not None

        await processor.start()
        assert processor._file_handle is not None

        # After stop, should be closed (None)
        await processor.stop()
        # Note: _file_handle may still be set but closed

    @pytest.mark.asyncio
    async def test_flush_on_shutdown(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test that remaining buffer is flushed on shutdown."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        # Process fewer records than batch size
        num_records = processor._batch_size // 2

        async with aiperf_lifecycle(processor):
            for i in range(num_records):
                record = make_telemetry_record(
                    timestamp_ns=1_000_000_000 + i,
                    gpu_power_usage=100.0 + i,
                )
                await processor.process_telemetry_record(record)

        assert processor.lines_written == num_records
        lines = processor.output_file.read_text().splitlines()
        assert len(lines) == num_records

    @pytest.mark.asyncio
    async def test_wait_for_async_tasks(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test that wait_for_tasks waits for async flush operations."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            for i in range(processor._batch_size * 3):
                record = make_telemetry_record(
                    timestamp_ns=1_000_000_000 + i,
                    gpu_power_usage=100.0 + i,
                )
                await processor.process_telemetry_record(record)

            await processor.wait_for_tasks()

        assert processor.lines_written == processor._batch_size * 3

    @pytest.mark.asyncio
    async def test_statistics_logged_on_shutdown(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that lines_written is correct on shutdown."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        await processor.initialize()
        await processor.start()

        try:
            for i in range(5):
                record = make_telemetry_record(
                    timestamp_ns=1_000_000_000 + i,
                    gpu_power_usage=100.0 + i,
                )
                await processor.process_telemetry_record(record)

            await processor.wait_for_tasks()
            await processor.stop()

            assert processor.lines_written == 5
        except Exception:
            await processor.stop()
            raise


class TestGPUTelemetryJSONLWriterIntegration:
    """Test GPUTelemetryJSONLWriter integration scenarios."""

    @pytest.mark.asyncio
    async def test_integration_with_real_files(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test end-to-end with actual file I/O."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_telemetry_record(sample_telemetry_record)

        # Verify file exists and is readable
        assert processor.output_file.exists()
        assert processor.output_file.is_file()

        # Verify content
        content = processor.output_file.read_text()
        assert len(content) > 0
        assert content.count("\n") == 1

        # Verify parseable
        lines = content.splitlines()
        record_dict = orjson.loads(lines[0])
        record = TelemetryRecord.model_validate(record_dict)
        assert record.gpu_uuid == sample_telemetry_record.gpu_uuid

    @pytest.mark.asyncio
    async def test_concurrent_writes(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test processing many records concurrently."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        num_records = 100

        async with aiperf_lifecycle(processor):
            for i in range(num_records):
                record = make_telemetry_record(
                    timestamp_ns=1_000_000_000 + i,
                    dcgm_url=f"http://node{i % 3}:9401/metrics",
                    gpu_index=i % 4,
                    gpu_uuid=f"GPU-{i}",
                    hostname=f"node{i % 3}",
                    gpu_power_usage=100.0 + i,
                    gpu_utilization=80.0,
                )
                await processor.process_telemetry_record(record)

            await processor.wait_for_tasks()

        assert processor.lines_written == num_records
        lines = processor.output_file.read_text().splitlines()
        assert len(lines) == num_records

    @pytest.mark.asyncio
    async def test_large_batch_processing(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test processing multiple batches worth of records."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        num_batches = 5
        total_records = processor._batch_size * num_batches

        async with aiperf_lifecycle(processor):
            for i in range(total_records):
                record = make_telemetry_record(
                    timestamp_ns=1_000_000_000 + i,
                    gpu_power_usage=100.0 + i,
                )
                await processor.process_telemetry_record(record)

            await processor.wait_for_tasks()

        assert processor.lines_written == total_records

    @pytest.mark.asyncio
    async def test_interleaved_gpu_records(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test processing records from multiple GPUs in interleaved fashion."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        num_gpus = 4
        records_per_gpu = 10

        async with aiperf_lifecycle(processor):
            for cycle in range(records_per_gpu):
                for gpu_idx in range(num_gpus):
                    record = make_telemetry_record(
                        timestamp_ns=1_000_000_000 + (cycle * num_gpus + gpu_idx),
                        gpu_index=gpu_idx,
                        gpu_uuid=f"GPU-{gpu_idx}",
                        gpu_model_name=f"GPU {gpu_idx}",
                        gpu_power_usage=100.0 + gpu_idx,
                        gpu_utilization=80.0 + cycle,
                    )
                    await processor.process_telemetry_record(record)

            await processor.wait_for_tasks()

        assert processor.lines_written == num_gpus * records_per_gpu

        # Verify records are in order
        lines = processor.output_file.read_text().splitlines()
        timestamps = [orjson.loads(line)["timestamp_ns"] for line in lines]
        assert timestamps == sorted(timestamps)


class TestGPUTelemetryJSONLWriterErrorHandling:
    """Test GPUTelemetryJSONLWriter error handling."""

    @pytest.mark.asyncio
    async def test_logs_error_on_write_failure(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        sample_telemetry_record: TelemetryRecord,
        mock_metric_registry: Mock,
    ):
        """Test that errors are logged when write fails."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        with (
            patch.object(processor, "buffered_write", side_effect=OSError("Disk full")),
            patch.object(processor, "error") as mock_error,
        ):
            await processor.process_telemetry_record(sample_telemetry_record)
            assert mock_error.call_count >= 1

    @pytest.mark.asyncio
    async def test_continues_after_write_error(
        self,
        user_config_telemetry_export: UserConfig,
        service_config: ServiceConfig,
        mock_metric_registry: Mock,
    ):
        """Test that processor continues after encountering an error."""
        processor = GPUTelemetryJSONLWriter(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_telemetry_export,
        )

        call_count = 0

        def side_effect_once(record):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("Temporary error")
            return AsyncMock()

        async with aiperf_lifecycle(processor):
            with patch.object(
                processor, "buffered_write", side_effect=side_effect_once
            ):
                with patch.object(processor, "error"):
                    for i in range(3):
                        record = make_telemetry_record(timestamp_ns=1_000_000_000 + i)
                        await processor.process_telemetry_record(record)

        assert call_count == 3

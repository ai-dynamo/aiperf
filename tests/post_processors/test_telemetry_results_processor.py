# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult
from aiperf.common.models.telemetry_models import (
    GpuMetadata,
    GpuTelemetryData,
    TelemetryHierarchy,
    TelemetryMetrics,
    TelemetryRecord,
)
from aiperf.post_processors.telemetry_results_processor import (
    TelemetryResultsProcessor,
)


@pytest.fixture
def mock_user_config() -> UserConfig:
    """Provide minimal UserConfig for testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
        )
    )


@pytest.fixture
def sample_telemetry_record() -> TelemetryRecord:
    """Create a sample TelemetryRecord with typical values."""
    return TelemetryRecord(
        timestamp_ns=1000000000,
        dcgm_url="http://node1:9401/metrics",
        gpu_index=0,
        gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
        gpu_model_name="NVIDIA RTX 6000 Ada Generation",
        pci_bus_id="00000000:02:00.0",
        device="nvidia0",
        hostname="node1",
        telemetry_data=TelemetryMetrics(
            gpu_power_usage=75.5,
            energy_consumption=1000.0,
            gpu_utilization=85.0,
            gpu_memory_used=15.26,
            sm_clock_frequency=1500.0,
            memory_clock_frequency=800.0,
            memory_temperature=65.0,
            gpu_temperature=70.0,
        ),
    )


class TestTelemetryResultsProcessor:
    """Test cases for TelemetryResultsProcessor."""

    def test_initialization(self, mock_user_config: UserConfig) -> None:
        """Test processor initialization sets up hierarchy and metric units."""
        processor = TelemetryResultsProcessor(mock_user_config)

        assert isinstance(processor._telemetry_hierarchy, TelemetryHierarchy)

    @pytest.mark.asyncio
    async def test_process_telemetry_record(
        self, mock_user_config: UserConfig, sample_telemetry_record: TelemetryRecord
    ) -> None:
        """Test processing a telemetry record adds it to the hierarchy."""
        processor = TelemetryResultsProcessor(mock_user_config)

        await processor.process_telemetry_record(sample_telemetry_record)

        dcgm_url = sample_telemetry_record.dcgm_url
        gpu_uuid = sample_telemetry_record.gpu_uuid

        assert dcgm_url in processor._telemetry_hierarchy.dcgm_endpoints
        assert gpu_uuid in processor._telemetry_hierarchy.dcgm_endpoints[dcgm_url]

    @pytest.mark.asyncio
    async def test_summarize_with_valid_data(
        self, mock_user_config: UserConfig, sample_telemetry_record: TelemetryRecord
    ) -> None:
        """Test summarize generates MetricResults for all metrics with data."""
        processor = TelemetryResultsProcessor(mock_user_config)

        # Add multiple records to have enough data for statistics
        for i in range(5):
            record = TelemetryRecord(
                timestamp_ns=1000000000 + i * 1000000,
                dcgm_url=sample_telemetry_record.dcgm_url,
                gpu_index=sample_telemetry_record.gpu_index,
                gpu_uuid=sample_telemetry_record.gpu_uuid,
                gpu_model_name=sample_telemetry_record.gpu_model_name,
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=75.0 + i,
                    energy_consumption=1000.0 + i * 10,
                    gpu_utilization=80.0 + i,
                    gpu_memory_used=15.0 + i * 0.1,
                ),
            )
            await processor.process_telemetry_record(record)

        results = await processor.summarize()

        # Should have results for all metrics that had data
        assert len(results) > 0
        assert all(isinstance(r, MetricResult) for r in results)

        # Check that metrics are properly tagged
        result_tags = [r.tag for r in results]
        assert any("gpu_power_usage" in tag for tag in result_tags)
        assert any("energy_consumption" in tag for tag in result_tags)

    @pytest.mark.asyncio
    async def test_summarize_handles_no_metric_value(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test summarize logs debug message when metric has no data and continues."""
        processor = TelemetryResultsProcessor(mock_user_config)

        # Create a telemetry hierarchy with a GPU but no metric data
        mock_metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-12345678",
            model_name="Test GPU",
        )
        mock_telemetry_data = GpuTelemetryData(metadata=mock_metadata)
        processor._telemetry_hierarchy.dcgm_endpoints = {
            "http://test:9401/metrics": {
                "GPU-12345678": mock_telemetry_data,
            }
        }

        with patch.object(processor, "debug") as mock_debug:
            results = await processor.summarize()

            # Should have logged debug messages for missing metrics
            assert mock_debug.call_count > 0
            debug_messages = [call[0][0] for call in mock_debug.call_args_list]
            assert any("No data available" in msg for msg in debug_messages)

            # Should return empty list when no data available
            assert results == []

    @pytest.mark.asyncio
    async def test_summarize_handles_unexpected_exception(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test summarize logs exception with stack trace on unexpected errors."""
        processor = TelemetryResultsProcessor(mock_user_config)

        # Create a mock telemetry data that raises unexpected exception
        mock_metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-87654321",
            model_name="Test GPU",
        )
        mock_telemetry_data = Mock(spec=GpuTelemetryData)
        mock_telemetry_data.metadata = mock_metadata
        mock_telemetry_data.get_metric_result.side_effect = RuntimeError(
            "Unexpected error"
        )

        processor._telemetry_hierarchy.dcgm_endpoints = {
            "http://test:9401/metrics": {
                "GPU-87654321": mock_telemetry_data,
            }
        }

        with patch.object(processor, "exception") as mock_exception:
            results = await processor.summarize()

            # Should have logged exception with context
            assert mock_exception.call_count > 0
            exception_messages = [call[0][0] for call in mock_exception.call_args_list]
            assert any(
                "Unexpected error generating metric result" in msg
                for msg in exception_messages
            )
            assert any(
                "GPU-87654321" in msg for msg in exception_messages
            )  # First 12 chars

            # Should return empty list when all metrics fail
            assert results == []

    @pytest.mark.asyncio
    async def test_summarize_continues_after_errors(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test summarize continues processing other metrics after encountering errors."""
        processor = TelemetryResultsProcessor(mock_user_config)

        # Create mock telemetry data where some metrics fail
        mock_metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-mixed-results",
            model_name="Test GPU",
        )

        mock_telemetry_data = Mock(spec=GpuTelemetryData)
        mock_telemetry_data.metadata = mock_metadata

        # First metric raises NoMetricValue, second succeeds, third raises unexpected error
        call_count = 0

        def side_effect_func(_metric_name, tag, header, unit):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise NoMetricValue("No data for first metric")
            elif call_count == 2:
                return MetricResult(
                    tag=tag, header=header, unit=unit, avg=50.0, count=10
                )
            else:
                raise ValueError("Unexpected error")

        mock_telemetry_data.get_metric_result.side_effect = side_effect_func

        processor._telemetry_hierarchy.dcgm_endpoints = {
            "http://test:9401/metrics": {
                "GPU-mixed-results": mock_telemetry_data,
            }
        }

        with (
            patch.object(processor, "debug") as mock_debug,
            patch.object(processor, "exception") as mock_exception,
        ):
            results = await processor.summarize()

            # Should have logged both types of errors
            assert mock_debug.call_count > 0
            assert mock_exception.call_count > 0

            # Should have one successful result despite errors
            assert len(results) == 1
            assert results[0].avg == 50.0

    @pytest.mark.asyncio
    async def test_summarize_generates_correct_tags(
        self, mock_user_config: UserConfig, sample_telemetry_record: TelemetryRecord
    ) -> None:
        """Test summarize generates properly formatted tags with DCGM URL and GPU info."""
        processor = TelemetryResultsProcessor(mock_user_config)

        # Add records
        for i in range(3):
            record = TelemetryRecord(
                timestamp_ns=1000000000 + i * 1000000,
                dcgm_url="http://node1:9401/metrics",
                gpu_index=0,
                gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
                gpu_model_name="NVIDIA RTX 6000",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=75.0 + i,
                ),
            )
            await processor.process_telemetry_record(record)

        results = await processor.summarize()

        # Check tag format: metric_name_dcgm_TAG_gpuINDEX_UUID
        power_results = [r for r in results if "gpu_power_usage" in r.tag]
        assert len(power_results) > 0

        tag = power_results[0].tag
        assert "gpu_power_usage" in tag
        assert "dcgm_http" in tag  # URL gets sanitized
        assert "node1" in tag
        assert "gpu0" in tag
        assert "GPU-ef6ef310" in tag  # First 12 chars of UUID

    @pytest.mark.asyncio
    async def test_summarize_multiple_gpus(self, mock_user_config: UserConfig) -> None:
        """Test summarize handles multiple GPUs correctly."""
        processor = TelemetryResultsProcessor(mock_user_config)

        # Add records for two different GPUs
        for gpu_index in range(2):
            for i in range(3):
                record = TelemetryRecord(
                    timestamp_ns=1000000000 + i * 1000000,
                    dcgm_url="http://node1:9401/metrics",
                    gpu_index=gpu_index,
                    gpu_uuid=f"GPU-0000000{gpu_index}-0000-0000-0000-000000000000",
                    gpu_model_name="NVIDIA RTX 6000",
                    telemetry_data=TelemetryMetrics(
                        gpu_power_usage=75.0 + gpu_index * 10 + i,
                    ),
                )
                await processor.process_telemetry_record(record)

        results = await processor.summarize()

        # Should have results for both GPUs
        gpu0_results = [r for r in results if "gpu0" in r.tag]
        gpu1_results = [r for r in results if "gpu1" in r.tag]

        assert len(gpu0_results) > 0
        assert len(gpu1_results) > 0

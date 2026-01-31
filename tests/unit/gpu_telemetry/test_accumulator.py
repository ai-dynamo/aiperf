# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult
from aiperf.common.models.telemetry_models import (
    GpuMetadata,
    GpuTelemetryData,
    TelemetryHierarchy,
    TelemetryRecord,
)
from aiperf.gpu_telemetry.accumulator import (
    GPUTelemetryAccumulator,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.post_processors.conftest import make_telemetry_record


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
def mock_service_config() -> ServiceConfig:
    """Provide minimal ServiceConfig for testing."""
    return ServiceConfig()


@pytest.fixture
def mock_pub_client():
    """Provide mock pub client for testing."""
    mock = Mock()
    mock.publish = AsyncMock()
    return mock


@pytest.fixture
def sample_telemetry_record() -> TelemetryRecord:
    """Create a sample TelemetryRecord with typical values."""
    return make_telemetry_record(
        timestamp_ns=1000000000,
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
        xid_errors=0.0,
        power_violation=120.0,
    )


class TestGPUTelemetryAccumulator:
    """Test cases for GPUTelemetryAccumulator."""

    def test_initialization(
        self,
        mock_user_config: UserConfig,
        mock_service_config: ServiceConfig,
        mock_pub_client,
    ) -> None:
        """Test processor initialization sets up hierarchy and metric units."""
        processor = GPUTelemetryAccumulator(
            user_config=mock_user_config,
            service_config=mock_service_config,
            pub_client=mock_pub_client,
        )

        assert isinstance(processor._hierarchy, TelemetryHierarchy)

    @pytest.mark.asyncio
    async def test_process_telemetry_record(
        self,
        mock_user_config: UserConfig,
        mock_service_config: ServiceConfig,
        mock_pub_client,
        sample_telemetry_record: TelemetryRecord,
    ) -> None:
        """Test processing a telemetry record adds it to the hierarchy."""
        processor = GPUTelemetryAccumulator(
            user_config=mock_user_config,
            service_config=mock_service_config,
            pub_client=mock_pub_client,
        )

        await processor.process_telemetry_record(sample_telemetry_record)

        dcgm_url = sample_telemetry_record.dcgm_url
        gpu_uuid = sample_telemetry_record.gpu_uuid

        assert dcgm_url in processor._hierarchy.dcgm_endpoints
        assert gpu_uuid in processor._hierarchy.dcgm_endpoints[dcgm_url]

    @pytest.mark.asyncio
    async def test_get_hierarchy(
        self,
        mock_user_config: UserConfig,
        mock_service_config: ServiceConfig,
        mock_pub_client,
        sample_telemetry_record: TelemetryRecord,
    ) -> None:
        """Test get_hierarchy returns accumulated data."""
        processor = GPUTelemetryAccumulator(
            user_config=mock_user_config,
            service_config=mock_service_config,
            pub_client=mock_pub_client,
        )

        # Add some records
        await processor.process_telemetry_record(sample_telemetry_record)

        # Get hierarchy
        hierarchy = processor._hierarchy

        assert isinstance(hierarchy, TelemetryHierarchy)
        assert sample_telemetry_record.dcgm_url in hierarchy.dcgm_endpoints
        assert (
            sample_telemetry_record.gpu_uuid
            in hierarchy.dcgm_endpoints[sample_telemetry_record.dcgm_url]
        )

    @pytest.mark.asyncio
    async def test_summarize_with_valid_data(
        self,
        mock_user_config: UserConfig,
        mock_service_config: ServiceConfig,
        mock_pub_client,
        sample_telemetry_record: TelemetryRecord,
    ) -> None:
        """Test summarize generates MetricResults for all metrics with data."""
        processor = GPUTelemetryAccumulator(
            user_config=mock_user_config,
            service_config=mock_service_config,
            pub_client=mock_pub_client,
        )

        for i in range(5):
            record = make_telemetry_record(
                timestamp_ns=1000000000 + i * 1000000,
                dcgm_url=sample_telemetry_record.dcgm_url,
                gpu_index=sample_telemetry_record.gpu_index,
                gpu_uuid=sample_telemetry_record.gpu_uuid,
                gpu_model_name=sample_telemetry_record.gpu_model_name,
                gpu_power_usage=75.0 + i,
                energy_consumption=1000.0 + i * 10,
                gpu_utilization=80.0 + i,
                gpu_memory_used=15.0 + i * 0.1,
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
        self,
        mock_user_config: UserConfig,
        mock_service_config: ServiceConfig,
        mock_pub_client,
    ) -> None:
        """Test summarize logs debug message when metric has no data and continues."""
        processor = GPUTelemetryAccumulator(
            user_config=mock_user_config,
            service_config=mock_service_config,
            pub_client=mock_pub_client,
        )

        mock_metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-12345678",
            gpu_model_name="Test GPU",
        )
        mock_telemetry_data = GpuTelemetryData(metadata=mock_metadata)
        processor._hierarchy.dcgm_endpoints = {
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
        self,
        mock_user_config: UserConfig,
        mock_service_config: ServiceConfig,
        mock_pub_client,
    ) -> None:
        """Test summarize logs exception with stack trace on unexpected errors."""
        processor = GPUTelemetryAccumulator(
            user_config=mock_user_config,
            service_config=mock_service_config,
            pub_client=mock_pub_client,
        )

        mock_metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-87654321",
            gpu_model_name="Test GPU",
        )
        mock_telemetry_data = Mock(spec=GpuTelemetryData)
        mock_telemetry_data.metadata = mock_metadata
        mock_telemetry_data.get_metric_result.side_effect = RuntimeError(
            "Unexpected error"
        )

        processor._hierarchy.dcgm_endpoints = {
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
        self,
        mock_user_config: UserConfig,
        mock_service_config: ServiceConfig,
        mock_pub_client,
    ) -> None:
        """Test summarize continues processing other metrics after encountering errors."""
        processor = GPUTelemetryAccumulator(
            user_config=mock_user_config,
            service_config=mock_service_config,
            pub_client=mock_pub_client,
        )

        mock_metadata = GpuMetadata(
            gpu_index=0,
            gpu_uuid="GPU-mixed-results",
            gpu_model_name="Test GPU",
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

        processor._hierarchy.dcgm_endpoints = {
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
        self,
        mock_user_config: UserConfig,
        mock_service_config: ServiceConfig,
        mock_pub_client,
        sample_telemetry_record: TelemetryRecord,
    ) -> None:
        """Test summarize generates properly formatted tags with DCGM URL and GPU info."""
        processor = GPUTelemetryAccumulator(
            user_config=mock_user_config,
            service_config=mock_service_config,
            pub_client=mock_pub_client,
        )

        for i in range(3):
            record = make_telemetry_record(
                timestamp_ns=1000000000 + i * 1000000,
                gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
                gpu_model_name="NVIDIA RTX 6000",
                gpu_power_usage=75.0 + i,
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
    async def test_summarize_multiple_gpus(
        self,
        mock_user_config: UserConfig,
        mock_service_config: ServiceConfig,
        mock_pub_client,
    ) -> None:
        """Test summarize handles multiple GPUs correctly."""
        processor = GPUTelemetryAccumulator(
            user_config=mock_user_config,
            service_config=mock_service_config,
            pub_client=mock_pub_client,
        )

        for gpu_index in range(2):
            for i in range(3):
                record = make_telemetry_record(
                    timestamp_ns=1000000000 + i * 1000000,
                    gpu_index=gpu_index,
                    gpu_uuid=f"GPU-0000000{gpu_index}-0000-0000-0000-000000000000",
                    gpu_model_name="NVIDIA RTX 6000",
                    gpu_power_usage=75.0 + gpu_index * 10 + i,
                )
                await processor.process_telemetry_record(record)

        results = await processor.summarize()

        # Should have results for both GPUs
        gpu0_results = [r for r in results if "gpu0" in r.tag]
        gpu1_results = [r for r in results if "gpu1" in r.tag]

        assert len(gpu0_results) > 0
        assert len(gpu1_results) > 0

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for GPU telemetry collection pipeline.

Tests the end-to-end flow from DCGMTelemetryCollector through GPUTelemetryAccumulator
with realistic mock data and callback mechanisms.
"""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.gpu_telemetry.accumulator import GPUTelemetryAccumulator
from aiperf.gpu_telemetry.dcgm_collector import DCGMTelemetryCollector


class TestGPUTelemetryIntegration:
    """Integration tests for complete telemetry collection and processing pipeline.

    This test class verifies the end-to-end integration between:
    - DCGMTelemetryCollector (DCGM data collection)
    - GPUTelemetryAccumulator (hierarchical data organization)
    - Multi-node telemetry aggregation
    - Error handling across the pipeline

    Key integration scenarios tested:
    - Multi-node collection with different GPU configurations
    - Callback pipeline error propagation and handling
    - Unit scaling consistency across components
    - Empty/invalid response handling throughout the pipeline
    """

    @pytest.fixture
    def mock_dcgm_response_node1(self):
        """Mock DCGM metrics response for node1 with 2 GPUs."""

        return """# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (in %).
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-ef6ef310-1234-5678-9abc-def012345678",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="node1"} 75.0
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-a1b2c3d4-5678-9abc-def0-123456789abc",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="node1"} 85.0
# HELP DCGM_FI_DEV_POWER_USAGE Power draw (in W).
# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-ef6ef310-1234-5678-9abc-def012345678",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="node1"} 120.5
DCGM_FI_DEV_POWER_USAGE{gpu="1",UUID="GPU-a1b2c3d4-5678-9abc-def0-123456789abc",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="node1"} 135.2
# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB).
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-ef6ef310-1234-5678-9abc-def012345678",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="node1"} 8192.0
DCGM_FI_DEV_FB_USED{gpu="1",UUID="GPU-a1b2c3d4-5678-9abc-def0-123456789abc",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="node1"} 12288.0
# HELP DCGM_FI_DEV_FB_TOTAL Framebuffer memory total (in MiB).
# TYPE DCGM_FI_DEV_FB_TOTAL gauge
DCGM_FI_DEV_FB_TOTAL{gpu="0",UUID="GPU-ef6ef310-1234-5678-9abc-def012345678",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="node1"} 49152.0
DCGM_FI_DEV_FB_TOTAL{gpu="1",UUID="GPU-a1b2c3d4-5678-9abc-def0-123456789abc",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="node1"} 49152.0
"""

    @pytest.fixture
    def mock_dcgm_response_node2(self):
        """Mock DCGM metrics response for node2 with 2 GPUs."""

        return """# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (in %).
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-f5e6d7c8-9abc-def0-1234-56789abcdef0",device="nvidia0",modelName="NVIDIA H100 PCIe",Hostname="node2"} 90.0
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-9876fedc-ba09-8765-4321-fedcba098765",device="nvidia1",modelName="NVIDIA H100 PCIe",Hostname="node2"} 95.0
# HELP DCGM_FI_DEV_POWER_USAGE Power draw (in W).
# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-f5e6d7c8-9abc-def0-1234-56789abcdef0",device="nvidia0",modelName="NVIDIA H100 PCIe",Hostname="node2"} 250.0
DCGM_FI_DEV_POWER_USAGE{gpu="1",UUID="GPU-9876fedc-ba09-8765-4321-fedcba098765",device="nvidia1",modelName="NVIDIA H100 PCIe",Hostname="node2"} 275.3
# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB).
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-f5e6d7c8-9abc-def0-1234-56789abcdef0",device="nvidia0",modelName="NVIDIA H100 PCIe",Hostname="node2"} 65536.0
DCGM_FI_DEV_FB_USED{gpu="1",UUID="GPU-9876fedc-ba09-8765-4321-fedcba098765",device="nvidia1",modelName="NVIDIA H100 PCIe",Hostname="node2"} 70000.0
# HELP DCGM_FI_DEV_FB_TOTAL Framebuffer memory total (in MiB).
# TYPE DCGM_FI_DEV_FB_TOTAL gauge
DCGM_FI_DEV_FB_TOTAL{gpu="0",UUID="GPU-f5e6d7c8-9abc-def0-1234-56789abcdef0",device="nvidia0",modelName="NVIDIA H100 PCIe",Hostname="node2"} 81920.0
DCGM_FI_DEV_FB_TOTAL{gpu="1",UUID="GPU-9876fedc-ba09-8765-4321-fedcba098765",device="nvidia1",modelName="NVIDIA H100 PCIe",Hostname="node2"} 81920.0
"""

    @pytest.fixture
    def user_config(self):
        """User configuration for telemetry processing."""
        return UserConfig(
            endpoint=EndpointConfig(
                url="http://localhost:8000", model_names=["test-model"]
            )
        )

    @pytest.fixture
    def service_config(self):
        """Service configuration for telemetry processing."""
        return ServiceConfig()

    @pytest.fixture
    def mock_pub_client(self):
        """Mock pub client for telemetry processing."""
        mock = Mock()
        mock.publish = AsyncMock()
        return mock

    def setup_method(self):
        """Set up test fixtures for each test."""

        self.collected_records = []
        self.collection_errors = []

        async def record_callback(records, collector_id):
            """Callback to collect telemetry records."""
            self.collected_records.extend(records)

        async def error_callback(error, collector_id):
            """Callback to collect errors."""
            self.collection_errors.append(error)

        self.record_callback = record_callback
        self.error_callback = error_callback

    @pytest.mark.asyncio
    async def test_multi_node_telemetry_collection_and_processing(
        self,
        mock_dcgm_response_node1,
        mock_dcgm_response_node2,
        user_config,
        service_config,
        mock_pub_client,
    ):
        """
        Integration test for multi-node telemetry collection through processing pipeline.

        Tests the complete flow:
        1. DCGMTelemetryCollector fetches from multiple DCGM endpoints
        2. Records are processed through callbacks
        3. GPUTelemetryAccumulator stores in hierarchical structure
        4. Statistical aggregation produces MetricResult objects
        """

        # Mock aiohttp responses for different DCGM endpoints
        def mock_aiohttp_get(url, **kwargs):
            mock_context_manager = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = Mock()

            if "node1" in str(url):
                mock_response.text.return_value = mock_dcgm_response_node1
            elif "node2" in str(url):
                mock_response.text.return_value = mock_dcgm_response_node2
            else:
                mock_response.status = 404
                mock_response.raise_for_status.side_effect = aiohttp.ClientError(
                    "Not found"
                )

            mock_context_manager.__aenter__.return_value = mock_response
            return mock_context_manager

        with patch("aiohttp.ClientSession.get", side_effect=mock_aiohttp_get):
            collector1 = DCGMTelemetryCollector(
                dcgm_url="http://node1:9401/metrics",
                collection_interval=0.05,
                record_callback=self.record_callback,
                error_callback=self.error_callback,
                collector_id="node1_collector",
            )

            collector2 = DCGMTelemetryCollector(
                dcgm_url="http://node2:9401/metrics",
                collection_interval=0.05,
                record_callback=self.record_callback,
                error_callback=self.error_callback,
                collector_id="node2_collector",
            )

            processor = GPUTelemetryAccumulator(
                user_config=user_config,
                service_config=service_config,
                pub_client=mock_pub_client,
            )

            await collector1.initialize()
            await collector2.initialize()

            await collector1._collect_and_process_metrics()
            await collector2._collect_and_process_metrics()

            await collector1.stop()
            await collector2.stop()

            assert len(self.collected_records) > 0
            assert len(self.collection_errors) == 0

            for record in self.collected_records:
                await processor.process_telemetry_record(record)
            metric_results = await processor.summarize()

            assert len(processor._hierarchy.dcgm_endpoints) == 2

            node1_data = processor._hierarchy.dcgm_endpoints.get(
                "http://node1:9401/metrics", {}
            )
            node2_data = processor._hierarchy.dcgm_endpoints.get(
                "http://node2:9401/metrics", {}
            )

            assert len(node1_data) == 2
            assert len(node2_data) == 2

            node1_gpu0 = node1_data.get("GPU-ef6ef310-1234-5678-9abc-def012345678")
            assert node1_gpu0 is not None
            assert node1_gpu0.metadata.gpu_index == 0
            assert (
                node1_gpu0.metadata.gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
            )
            assert node1_gpu0.metadata.hostname == "node1"

            node2_gpu0 = node2_data.get("GPU-f5e6d7c8-9abc-def0-1234-56789abcdef0")
            assert node2_gpu0 is not None
            assert node2_gpu0.metadata.gpu_index == 0
            assert node2_gpu0.metadata.gpu_model_name == "NVIDIA H100 PCIe"
            assert node2_gpu0.metadata.hostname == "node2"

            assert len(metric_results) > 0

            power_metrics = [r for r in metric_results if "gpu_power_usage" in r.tag]
            util_metrics = [r for r in metric_results if "gpu_utilization" in r.tag]
            memory_metrics = [r for r in metric_results if "gpu_memory_used" in r.tag]

            assert len(power_metrics) > 0
            assert len(util_metrics) > 0
            assert len(memory_metrics) > 0

            sample_power_metric = power_metrics[0]
            assert "dcgm_" in sample_power_metric.tag
            assert "gpu" in sample_power_metric.tag

            power_metric_units = {r.unit for r in power_metrics}
            util_metric_units = {r.unit for r in util_metrics}
            memory_metric_units = {r.unit for r in memory_metrics}

            assert "W" in power_metric_units
            assert "%" in util_metric_units
            assert "GB" in memory_metric_units

            for metric in power_metrics:
                assert metric.min > 0
                assert metric.max >= metric.min
                assert 0 <= metric.avg <= 1000
                assert metric.count > 0

            for metric in util_metrics:
                assert 0 <= metric.min <= 100
                assert 0 <= metric.max <= 100
                assert metric.count > 0

    @pytest.mark.asyncio
    async def test_callback_pipeline_error_handling(
        self, mock_dcgm_response_node1, user_config, service_config, mock_pub_client
    ):
        """Test error handling in the callback pipeline during processing.

        Tests that errors during record processing are captured properly
        by directly calling collection method.
        """

        # Create a processor that will fail during processing
        faulty_processor = GPUTelemetryAccumulator(
            user_config=user_config,
            service_config=service_config,
            pub_client=mock_pub_client,
        )

        # Mock the process_telemetry_record method to raise an exception
        original_process = faulty_processor.process_telemetry_record

        async def failing_process_result(record):
            if record.gpu_index == 0:  # Fail on first GPU only
                raise ValueError("Simulated processing error")
            return await original_process(record)

        faulty_processor.process_telemetry_record = failing_process_result

        async def failing_record_callback(records, collector_id):
            """Async callback that processes records and may encounter errors."""
            try:
                self.collected_records.extend(records)
                # Process each record (this would normally be done by RecordsManager)
                for record in records:
                    try:
                        await faulty_processor.process_telemetry_record(record)
                    except Exception as e:
                        self.collection_errors.append(f"Processing error: {e}")
            except Exception as e:
                self.collection_errors.append(f"Callback error: {e}")

        def mock_aiohttp_get_error(url, **kwargs):
            mock_context_manager = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = Mock()
            mock_response.text.return_value = mock_dcgm_response_node1

            mock_context_manager.__aenter__.return_value = mock_response
            return mock_context_manager

        with patch("aiohttp.ClientSession.get", side_effect=mock_aiohttp_get_error):
            collector = DCGMTelemetryCollector(
                dcgm_url="http://node1:9401/metrics",
                collection_interval=0.05,
                record_callback=failing_record_callback,
                error_callback=self.error_callback,
                collector_id="test_collector",
            )

            # Initialize collector but don't start background task
            await collector.initialize()

            # Call collection method directly (simulates multiple collection cycles)
            await collector._collect_and_process_metrics()
            await collector._collect_and_process_metrics()

            await collector.stop()

            # Verify records and errors were collected
            assert len(self.collected_records) > 0, "Expected records to be collected"
            assert all(hasattr(r, "gpu_index") for r in self.collected_records), (
                "All records should be TelemetryRecord objects"
            )

            # Verify errors were captured from the faulty processor
            assert len(self.collection_errors) > 0, (
                "Expected processing errors to be captured"
            )
            assert all(isinstance(e, str) for e in self.collection_errors), (
                "All errors should be string messages"
            )

            # Verify we captured the simulated processing error
            processing_errors = [
                e
                for e in self.collection_errors
                if "simulated processing error" in e.lower()
            ]
            assert len(processing_errors) > 0, (
                "Should have captured simulated processing errors"
            )

    @pytest.mark.asyncio
    async def test_empty_dcgm_response_handling(
        self, user_config, service_config, mock_pub_client
    ):
        def mock_aiohttp_get_empty(url, **kwargs):
            mock_context_manager = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = Mock()
            mock_response.text.return_value = "# No metrics available\n"

            mock_context_manager.__aenter__.return_value = mock_response
            return mock_context_manager

        with patch("aiohttp.ClientSession.get", side_effect=mock_aiohttp_get_empty):
            collector = DCGMTelemetryCollector(
                dcgm_url="http://empty-node:9401/metrics",
                collection_interval=0.1,
                record_callback=self.record_callback,
                error_callback=self.error_callback,
                collector_id="empty_collector",
            )

            processor = GPUTelemetryAccumulator(
                user_config=user_config,
                service_config=service_config,
                pub_client=mock_pub_client,
            )

            await collector.initialize()
            await collector._collect_and_process_metrics()
            await collector.stop()

            for record in self.collected_records:
                await processor.process_telemetry_record(record)
            metric_results = await processor.summarize()

            assert len(self.collected_records) == 0
            assert len(metric_results) == 0
            assert len(self.collection_errors) == 0

    @pytest.mark.asyncio
    async def test_metric_unit_scaling_in_pipeline(
        self, user_config, service_config, mock_pub_client
    ):
        mock_response = """# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB).
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-test-1234",device="nvidia0",modelName="Test GPU",Hostname="testnode"} 1024.0
# HELP DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION Total energy consumption since last reset (in mJ).
# TYPE DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION counter
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0",UUID="GPU-test-1234",device="nvidia0",modelName="Test GPU",Hostname="testnode"} 1000000.0
"""

        def mock_aiohttp_get_scaling(url, **kwargs):
            mock_context_manager = AsyncMock()
            mock_response_obj = AsyncMock()
            mock_response_obj.status = 200
            mock_response_obj.raise_for_status = Mock()
            mock_response_obj.text.return_value = mock_response

            mock_context_manager.__aenter__.return_value = mock_response_obj
            return mock_context_manager

        with patch("aiohttp.ClientSession.get", side_effect=mock_aiohttp_get_scaling):
            collector = DCGMTelemetryCollector(
                dcgm_url="http://testnode:9401/metrics",
                collection_interval=0.1,
                record_callback=self.record_callback,
                error_callback=self.error_callback,
                collector_id="scaling_test",
            )

            processor = GPUTelemetryAccumulator(
                user_config=user_config,
                service_config=service_config,
                pub_client=mock_pub_client,
            )

            await collector.initialize()
            await collector._collect_and_process_metrics()
            await collector.stop()

            for record in self.collected_records:
                await processor.process_telemetry_record(record)
            metric_results = await processor.summarize()

            memory_metrics = [r for r in metric_results if "gpu_memory_used" in r.tag]
            energy_metrics = [
                r for r in metric_results if "energy_consumption" in r.tag
            ]

            assert len(memory_metrics) > 0
            memory_metric = memory_metrics[0]
            expected_gb = 1024 * 0.001048576
            assert abs(memory_metric.avg - expected_gb) < 0.01
            assert memory_metric.unit == "GB"

            assert len(energy_metrics) > 0
            energy_metric = energy_metrics[0]
            expected_mj = 1000000 * 1e-9
            assert abs(energy_metric.avg - expected_mj) < 0.001
            assert energy_metric.unit == "MJ"

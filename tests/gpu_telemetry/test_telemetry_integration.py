# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for GPU telemetry collection pipeline.

Tests the end-to-end flow from TelemetryDataCollector through TelemetryResultsProcessor
with realistic mock data and callback mechanisms.
"""

import asyncio
import json
import time
from unittest.mock import Mock, patch, create_autospec
from io import StringIO

import pytest
import requests

from aiperf.gpu_telemetry.telemetry_data_collector import TelemetryDataCollector
from aiperf.post_processors.telemetry_results_processor import TelemetryResultsProcessor
from aiperf.common.config import UserConfig
from aiperf.common.models.telemetry_models import TelemetryRecord
from aiperf.common.models.telemetry_models import TelemetryHierarchy
from aiperf.common.models import MetricResult


class TestTelemetryIntegration:
    """Integration tests for complete telemetry collection and processing pipeline.
    
    This test class verifies the end-to-end integration between:
    - TelemetryDataCollector (DCGM data collection)
    - TelemetryResultsProcessor (hierarchical data organization)
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
        """Mock user configuration for telemetry processing."""

        config = create_autospec(UserConfig, instance=True)
        config.log_level = "INFO"
        config.enable_trace = False
        return config

    def setup_method(self):
        """Set up test fixtures for each test."""

        self.collected_records = []
        self.collection_errors = []
        
        def record_callback(records):
            """Callback to collect telemetry records."""
            self.collected_records.extend(records)
            
        def error_callback(error):
            """Callback to collect errors."""
            self.collection_errors.append(error)
            
        self.record_callback = record_callback
        self.error_callback = error_callback

    def test_multi_node_telemetry_collection_and_processing(
        self, mock_dcgm_response_node1, mock_dcgm_response_node2, user_config
    ):
        """
        Integration test for multi-node telemetry collection through processing pipeline.

        Tests the complete flow:
        1. TelemetryDataCollector fetches from multiple DCGM endpoints
        2. Records are processed through callbacks
        3. TelemetryResultsProcessor stores in hierarchical structure
        4. Statistical aggregation produces MetricResult objects
        """

        # Mock HTTP responses for different DCGM endpoints
        def mock_requests_get(url, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            
            if "node1" in url:
                mock_response.text = mock_dcgm_response_node1
            elif "node2" in url:
                mock_response.text = mock_dcgm_response_node2
            else:
                mock_response.status_code = 404
                mock_response.raise_for_status.side_effect = requests.RequestException("Not found")
            
            return mock_response

        with patch('requests.get', side_effect=mock_requests_get):
            # Set up telemetry collectors for two DCGM endpoints
            collector1 = TelemetryDataCollector(
                dcgm_url="http://node1:9401/metrics",
                collection_interval=0.1,
                record_callback=self.record_callback,
                error_callback=self.error_callback,
                collector_id="node1_collector"
            )
            
            collector2 = TelemetryDataCollector(
                dcgm_url="http://node2:9401/metrics", 
                collection_interval=0.1,
                record_callback=self.record_callback,
                error_callback=self.error_callback,
                collector_id="node2_collector"
            )
            
            # Set up telemetry results processor
            processor = TelemetryResultsProcessor(user_config=user_config)
            
            # Start collectors and collect for a short period
            collector1.start()
            collector2.start()
            
            # Allow time for collection (should get 2-3 samples)
            time.sleep(0.35)
            
            # Stop collectors
            collector1.stop()
            collector2.stop()
            
            # Verify records were collected
            assert len(self.collected_records) > 0, "No telemetry records were collected"
            assert len(self.collection_errors) == 0, f"Collection errors occurred: {self.collection_errors}"
            
            # Process all collected records through the processor
            async def process_all_records():
                for record in self.collected_records:
                    await processor.process_telemetry_record(record)
                return await processor.summarize()
            
            # Run the async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                metric_results = loop.run_until_complete(process_all_records())
            finally:
                loop.close()
            
            # Verify hierarchical structure was created
            assert len(processor._telemetry_hierarchy.dcgm_endpoints) == 2, f"Expected 2 DCGM endpoints, got {len(processor._telemetry_hierarchy.dcgm_endpoints)}"
            
            node1_data = processor._telemetry_hierarchy.dcgm_endpoints.get("http://node1:9401/metrics", {})
            node2_data = processor._telemetry_hierarchy.dcgm_endpoints.get("http://node2:9401/metrics", {})
            
            assert len(node1_data) == 2, f"Expected 2 GPUs for node1, got {len(node1_data)}. Available GPUs: {list(node1_data.keys())}"
            assert len(node2_data) == 2, f"Expected 2 GPUs for node2, got {len(node2_data)}. Available GPUs: {list(node2_data.keys())}"
            
            # Verify GPU metadata was captured correctly
            node1_gpu0 = node1_data.get("GPU-ef6ef310-1234-5678-9abc-def012345678")
            assert node1_gpu0 is not None, "Node1 GPU0 not found"
            assert node1_gpu0.metadata.gpu_index == 0
            assert node1_gpu0.metadata.model_name == "NVIDIA RTX 6000 Ada Generation"
            assert node1_gpu0.metadata.hostname == "node1"
            
            node2_gpu0 = node2_data.get("GPU-f5e6d7c8-9abc-def0-1234-56789abcdef0")
            assert node2_gpu0 is not None, "Node2 GPU0 not found"
            assert node2_gpu0.metadata.gpu_index == 0
            assert node2_gpu0.metadata.model_name == "NVIDIA H100 PCIe"
            assert node2_gpu0.metadata.hostname == "node2"
            
            # Verify metric results were generated
            assert len(metric_results) > 0, "No metric results generated"
            
            # Find specific metrics to validate
            power_metrics = [r for r in metric_results if "gpu_power_usage" in r.tag]
            util_metrics = [r for r in metric_results if "gpu_utilization" in r.tag]
            memory_metrics = [r for r in metric_results if "gpu_memory_used" in r.tag]
            
            assert len(power_metrics) > 0, "No power usage metrics found"
            assert len(util_metrics) > 0, "No utilization metrics found"
            assert len(memory_metrics) > 0, "No memory usage metrics found"
            
            # Verify metric tag format (hierarchical naming)
            sample_power_metric = power_metrics[0]
            assert "dcgm_" in sample_power_metric.tag, "Metric tag should include DCGM endpoint identifier"
            assert "gpu" in sample_power_metric.tag, "Metric tag should include GPU identifier"
            
            # Verify units are correctly assigned
            power_metric_units = {r.unit for r in power_metrics}
            util_metric_units = {r.unit for r in util_metrics}
            memory_metric_units = {r.unit for r in memory_metrics}
            
            assert "W" in power_metric_units, "Power metrics should have Watts unit"
            assert "%" in util_metric_units, "Utilization metrics should have percentage unit"
            assert "GB" in memory_metric_units, "Memory metrics should have GB unit"
            
            # Verify statistical computation (should have realistic values)
            for metric in power_metrics:
                assert metric.min > 0, f"Power metric {metric.tag} should have positive minimum"
                assert metric.max >= metric.min, f"Power metric {metric.tag} max should be >= min"
                assert 0 <= metric.avg <= 1000, f"Power metric {metric.tag} average should be reasonable"
                assert metric.count > 0, f"Power metric {metric.tag} should have data points"
            
            for metric in util_metrics:
                assert 0 <= metric.min <= 100, f"Utilization metric {metric.tag} minimum should be 0-100%"
                assert 0 <= metric.max <= 100, f"Utilization metric {metric.tag} maximum should be 0-100%"
                assert metric.count > 0, f"Utilization metric {metric.tag} should have data points"

    def test_callback_pipeline_error_handling(self, mock_dcgm_response_node1, user_config):
        """Test error handling in the callback pipeline during processing."""

        # Create a processor that will fail during processing
        faulty_processor = TelemetryResultsProcessor(user_config=user_config)
        
        # Mock the process_telemetry_record method to raise an exception
        original_process = faulty_processor.process_telemetry_record
        async def failing_process_result(record):
            if record.gpu_index == 0:  # Fail on first GPU only
                raise ValueError("Simulated processing error")
            return await original_process(record)
        
        faulty_processor.process_telemetry_record = failing_process_result
        
        def failing_record_callback(records):
            """Callback that processes records and may encounter errors."""
            try:
                self.collected_records.extend(records)
                # Simulate processing each record (this would normally be done by RecordsManager)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    for record in records:
                        loop.run_until_complete(faulty_processor.process_telemetry_record(record))
                except Exception as e:
                    self.collection_errors.append(f"Processing error: {e}")
                finally:
                    loop.close()
            except Exception as e:
                self.collection_errors.append(f"Callback error: {e}")
        
        def mock_requests_get(url, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = mock_dcgm_response_node1
            mock_response.raise_for_status = Mock()
            return mock_response

        with patch('requests.get', side_effect=mock_requests_get):
            collector = TelemetryDataCollector(
                dcgm_url="http://node1:9401/metrics",
                collection_interval=0.1,
                record_callback=failing_record_callback,
                error_callback=self.error_callback,
                collector_id="test_collector"
            )
            
            collector.start()
            time.sleep(0.25)  # Collect for a short time
            collector.stop()
            
            # Verify that records were collected but processing errors occurred
            assert len(self.collected_records) > 0, "Records should still be collected despite processing errors"
            assert len(self.collection_errors) > 0, "Processing errors should be captured"
            
            # Verify the error message format
            processing_errors = [e for e in self.collection_errors if "processing error" in e.lower()]
            assert len(processing_errors) > 0, "Should have processing errors"

    def test_empty_dcgm_response_handling(self, user_config):
        """Test handling of empty or invalid DCGM responses."""

        def mock_requests_get(url, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "# No metrics available\n"  # Empty response
            mock_response.raise_for_status = Mock()
            return mock_response

        with patch('requests.get', side_effect=mock_requests_get):
            collector = TelemetryDataCollector(
                dcgm_url="http://empty-node:9401/metrics",
                collection_interval=0.1,
                record_callback=self.record_callback,
                error_callback=self.error_callback,
                collector_id="empty_collector"
            )
            
            processor = TelemetryResultsProcessor(user_config=user_config)
            
            collector.start()
            time.sleep(0.25)
            collector.stop()
            
            # Process any records that were collected
            async def process_records():
                for record in self.collected_records:
                    await processor.process_telemetry_record(record)
                return await processor.summarize()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                metric_results = loop.run_until_complete(process_records())
            finally:
                loop.close()
            
            # With empty responses, we should have no records and no metric results
            assert len(self.collected_records) == 0, "No records should be collected from empty responses"
            assert len(metric_results) == 0, "No metric results should be generated from empty data"
            assert len(self.collection_errors) == 0, "Empty responses should not generate errors"

    def test_metric_unit_scaling_in_pipeline(self, user_config):
        """Test that metric unit scaling is applied correctly through the pipeline."""

        # Mock response with specific values to test scaling
        mock_response = """# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB).
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-test-1234",device="nvidia0",modelName="Test GPU",Hostname="testnode"} 1024.0
# HELP DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION Total energy consumption since last reset (in mJ).
# TYPE DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION counter
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0",UUID="GPU-test-1234",device="nvidia0",modelName="Test GPU",Hostname="testnode"} 1000000.0
"""
        
        def mock_requests_get(url, **kwargs):
            mock_response_obj = Mock()
            mock_response_obj.status_code = 200
            mock_response_obj.text = mock_response
            mock_response_obj.raise_for_status = Mock()
            return mock_response_obj

        with patch('requests.get', side_effect=mock_requests_get):
            collector = TelemetryDataCollector(
                dcgm_url="http://testnode:9401/metrics",
                collection_interval=0.1,
                record_callback=self.record_callback,
                error_callback=self.error_callback,
                collector_id="scaling_test"
            )
            
            processor = TelemetryResultsProcessor(user_config=user_config)
            
            collector.start()
            time.sleep(0.25)
            collector.stop()
            
            # Process records and generate metrics
            async def process_records():
                for record in self.collected_records:
                    await processor.process_telemetry_record(record)
                return await processor.summarize()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                metric_results = loop.run_until_complete(process_records())
            finally:
                loop.close()
            
            # Find memory and energy metrics to verify scaling
            memory_metrics = [r for r in metric_results if "gpu_memory_used" in r.tag]
            energy_metrics = [r for r in metric_results if "energy_consumption" in r.tag]
            
            if memory_metrics:
                memory_metric = memory_metrics[0]
                # 1024 MiB should be scaled to 1.073741824 GB (1024 * 0.001048576)
                expected_gb = 1024 * 0.001048576  # MiB to GB scaling factor
                assert abs(memory_metric.avg - expected_gb) < 0.01, f"Memory scaling incorrect: expected {expected_gb:.3f} GB, got {memory_metric.avg}"
                assert memory_metric.unit == "GB", "Memory metric should have GB unit"
            
            if energy_metrics:
                energy_metric = energy_metrics[0]
                # Verify energy scaling: 1000000 mJ scaled to 0.001 MJ
                expected_mj = 1000000 * 1e-9  # mJ to MJ scaling factor
                assert abs(energy_metric.avg - expected_mj) < 0.001, f"Energy scaling incorrect: expected {expected_mj:.6f} MJ, got {energy_metric.avg}"
                assert energy_metric.unit == "MJ", "Energy metric should have MJ unit"
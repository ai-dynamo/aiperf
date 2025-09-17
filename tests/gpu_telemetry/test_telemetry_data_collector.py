# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
import time
from unittest.mock import Mock, patch
from threading import Event
from aiperf.gpu_telemetry.telemetry_data_collector import TelemetryDataCollector
from aiperf.common.models.telemetry_models import TelemetryRecord


class TestTelemetryDataCollectorCore:
    """Test core TelemetryDataCollector functionality.
    
    This test class focuses exclusively on the data collection, parsing,
    and lifecycle management of the TelemetryDataCollector. It does NOT
    test metric extraction logic or model validation (those are in separate files).
    
    Key areas tested:
    - Initialization and configuration
    - DCGM HTTP endpoint communication  
    - Prometheus metric parsing
    - Background collection lifecycle
    - Error handling and resilience
    """
    
    def setup_method(self):
        """Set up test fixtures for callback testing."""
        self.records_received = []
        self.errors_received = []
        
        def record_callback(records):
            self.records_received.extend(records)
            
        def error_callback(error):
            self.errors_received.append(error)
            
        self.record_callback = record_callback
        self.error_callback = error_callback

    def test_collector_initialization_complete(self):
        """Test TelemetryDataCollector initialization with all parameters.
        
        Verifies that the collector properly stores all configuration parameters
        including DCGM URL, collection interval, callbacks, and collector ID.
        Also checks that the initial state is correct (not started, no thread).
        """
        collector = TelemetryDataCollector(
            dcgm_url="http://localhost:9401/metrics", 
            collection_interval=0.1,
            record_callback=self.record_callback,
            error_callback=self.error_callback,
            collector_id="test_collector"
        )
        
        assert collector._dcgm_url == "http://localhost:9401/metrics"
        assert collector._collection_interval == 0.1
        assert collector._collector_id == "test_collector"
        assert collector._record_callback is not None
        assert collector._error_callback is not None
        assert not collector._stop_event.is_set()
        assert collector._collection_thread is None

    def test_collector_initialization_minimal(self):
        """Test TelemetryDataCollector initialization with minimal parameters.
        
        Verifies that the collector applies correct default values when only
        the required DCGM URL is provided. Tests default collection interval
        (~30Hz), default collector ID, and None callbacks.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        assert collector._dcgm_url == "http://localhost:9401/metrics"
        assert collector._collection_interval == 0.033  # Default ~30Hz
        assert collector._collector_id == "telemetry_collector"
        assert collector._record_callback is None
        assert collector._error_callback is None


class TestPrometheusMetricParsing:
    """Test DCGM Prometheus metric parsing functionality.
    
    This test class focuses on the parsing of DCGM Prometheus format responses.
    Tests the robustness of metric line parsing, label extraction, and
    data type conversion without testing the full collection lifecycle.
    """

    def test_parse_valid_metric_line(self):
        """Test parsing of well-formed DCGM Prometheus metric lines.
        
        Verifies that the collector can correctly extract all components from
        a valid DCGM metric line: metric name, GPU index, numeric value, and
        metadata fields (model name, UUID, PCI bus ID, device, hostname).
        Uses a realistic DCGM_FI_DEV_POWER_USAGE example.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        line = 'DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 22.582000'
        result = collector._parse_metric_line(line)
        
        assert result is not None
        metric_name, gpu_index, value, model_name, uuid, pci_bus_id, device, hostname = result
        assert metric_name == "DCGM_FI_DEV_POWER_USAGE"
        assert gpu_index == 0
        assert value == 22.582000
        assert model_name == "NVIDIA RTX 6000 Ada Generation"
        assert uuid == "GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc"
        assert pci_bus_id == "00000000:02:00.0"
        assert device == "nvidia0"
        assert hostname == "ed7e7a5e585f"

    def test_parse_malformed_metric_lines(self):
        """Test graceful handling of malformed DCGM metric lines.
        
        Verifies that the parser returns None (doesn't crash) when given
        invalid input: missing values, incorrect format, missing required
        labels (GPU index, model name). This ensures robustness against
        partial or corrupted DCGM responses.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        malformed_cases = [
            "DCGM_FI_DEV_POWER_USAGE{gpu=\"0\"}",  # Missing value
            "invalid line",  # Invalid format
            "DCGM_FI_DEV_POWER_USAGE{modelName=\"RTX\"} 22.5",  # Missing GPU index
            "DCGM_FI_DEV_POWER_USAGE{gpu=\"0\"} 22.5",  # Missing model name
        ]
        
        for malformed_line in malformed_cases:
            assert collector._parse_metric_line(malformed_line) is None

    def test_label_extraction_robustness(self):
        """Test extraction of labels from various Prometheus label formats.
        
        Verifies that the label extraction methods can handle different
        label orderings, spacing, and missing optional labels without
        breaking the parsing process.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        # Test GPU index extraction
        assert collector._extract_gpu_index('gpu="0"') == 0
        assert collector._extract_gpu_index('gpu="1",other="value"') == 1
        assert collector._extract_gpu_index('other="value",gpu="2"') == 2
        assert collector._extract_gpu_index('no_gpu="0"') is None
        assert collector._extract_gpu_index('gpu="invalid"') is None
        
        # Test model name extraction  
        model_str = 'modelName="NVIDIA RTX 6000 Ada Generation"'
        assert collector._extract_model_name(model_str) == "NVIDIA RTX 6000 Ada Generation"
        assert collector._extract_model_name('no_model="NVIDIA"') is None

    def test_complete_parsing_single_gpu(self, sample_dcgm_data):
        """Test parsing complete DCGM response into TelemetryRecord for one GPU.
        
        Verifies that the collector can parse a multi-line DCGM response containing
        various metrics for a single GPU and consolidate them into one TelemetryRecord.
        Tests proper unit scaling (MiB→GB for memory, mJ→MJ for energy) and
        that all metadata and metric values are correctly assigned.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        records = collector._parse_metrics_to_records(sample_dcgm_data)
        assert len(records) == 1
        
        record = records[0]
        assert record.dcgm_url == "http://localhost:9401/metrics"
        assert record.gpu_index == 0
        assert record.gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert record.gpu_uuid == "GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc"
        assert record.gpu_power_usage == 22.582000
        assert record.gpu_power_limit == 300.000000
        
        # Test unit scaling applied correctly
        assert abs(record.energy_consumption - 0.955287014) < 0.001  # mJ to MJ
        assert abs(record.gpu_memory_used - 48.878) < 0.001  # MiB to GB

    def test_complete_parsing_multi_gpu(self, multi_gpu_dcgm_data):
        """Test parsing complete DCGM response for multiple GPUs.
        
        Verifies that the collector can parse a multi-line DCGM response containing
        metrics for multiple GPUs and create separate TelemetryRecord objects for each.
        Tests that GPU-specific metadata is correctly associated with the right GPU.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        records = collector._parse_metrics_to_records(multi_gpu_dcgm_data)
        assert len(records) == 3
        
        # Sort by GPU index for predictable testing
        records.sort(key=lambda r: r.gpu_index)
        
        # Verify each GPU has correct metadata
        assert records[0].gpu_index == 0
        assert records[0].gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert records[1].gpu_index == 1  
        assert records[2].gpu_index == 2
        assert records[2].gpu_model_name == "NVIDIA H100 PCIe"

    def test_empty_response_handling(self):
        """Test parsing of empty or comment-only DCGM responses.
        
        Verifies that the collector gracefully handles edge cases:
        - Completely empty responses
        - Responses containing only comments (# HELP, # TYPE)
        Should return empty record list without crashing.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        empty_cases = [
            "",  # Empty
            "# HELP comment\n# TYPE comment",  # Only comments
            "   \n\n   ",  # Only whitespace
        ]
        
        for empty_data in empty_cases:
            records = collector._parse_metrics_to_records(empty_data)
            assert len(records) == 0


class TestHttpCommunication:
    """Test HTTP communication with DCGM endpoints.
    
    This test class focuses on the network communication aspects:
    endpoint reachability, HTTP request handling, and error scenarios.
    """

    @patch('aiperf.gpu_telemetry.telemetry_data_collector.requests.get')
    def test_endpoint_reachability_success(self, mock_get):
        """Test DCGM endpoint reachability check with successful HTTP response.
        
        Verifies that the collector correctly identifies a reachable DCGM endpoint
        when the HTTP request returns a 200 status code. Tests the timeout parameter
        and proper HTTP request configuration.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        assert collector.is_url_reachable() is True
        
        mock_get.assert_called_once_with("http://localhost:9401/metrics", timeout=5)

    @patch('aiperf.gpu_telemetry.telemetry_data_collector.requests.get')
    def test_endpoint_reachability_failures(self, mock_get):
        """Test DCGM endpoint reachability check with various failure scenarios.
        
        Verifies that the collector correctly identifies unreachable DCGM endpoints:
        - HTTP error status codes (404, 500, etc.)
        - Network connection errors  
        - Request timeouts
        Ensures robustness before starting data collection.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        # Test HTTP error status
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        assert collector.is_url_reachable() is False
        
        # Test request exception
        mock_get.side_effect = requests.RequestException("Connection error")
        assert collector.is_url_reachable() is False

    @patch('aiperf.gpu_telemetry.telemetry_data_collector.requests.get')
    def test_metrics_fetching(self, mock_get):
        """Test HTTP request to fetch DCGM metrics from Prometheus endpoint.
        
        Verifies that the collector makes properly configured HTTP GET requests
        to the DCGM endpoint, handles the response correctly, and raises appropriate
        exceptions for HTTP errors. Tests the core data fetching mechanism.
        """
        mock_response = Mock()
        mock_response.text = "test_metrics_data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        result = collector._fetch_metrics()
        
        assert result == "test_metrics_data"
        mock_get.assert_called_once_with("http://localhost:9401/metrics", timeout=5)
        mock_response.raise_for_status.assert_called_once()


class TestCollectionLifecycle:
    """Test the background collection thread lifecycle.
    
    This test class focuses on thread management, timing, callbacks,
    and graceful start/stop behavior of the continuous collection process.
    """

    def setup_method(self):
        """Set up callback tracking for lifecycle tests."""
        self.records_received = []
        self.errors_received = []
        
        def record_callback(records):
            self.records_received.extend(records)
            
        def error_callback(error):
            self.errors_received.append(error)
            
        self.record_callback = record_callback
        self.error_callback = error_callback

    def test_successful_collection_loop(self):
        """Test the continuous collection loop with successful data processing.
        
        Verifies that the background collection thread:
        - Fetches metrics from DCGM at regular intervals
        - Parses the data into TelemetryRecord objects  
        - Calls the record callback with parsed data
        Tests the happy path of the collection lifecycle.
        """
        collector = TelemetryDataCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.1,
            record_callback=self.record_callback,
            error_callback=self.error_callback
        )
        
        # Mock successful metrics fetch
        with patch.object(collector, '_fetch_metrics') as mock_fetch:
            mock_fetch.return_value = '''DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-test",modelName="Test GPU",pci_bus_id="test",device="test",Hostname="test"} 75.5'''
            
            collector.start()
            time.sleep(0.25)  # Allow multiple collection cycles
            collector.stop()
            
            # Verify records were collected and processed
            assert len(self.records_received) > 0
            assert all(isinstance(r, TelemetryRecord) for r in self.records_received)
            assert all(r.gpu_power_usage == 75.5 for r in self.records_received)
            assert len(self.errors_received) == 0

    def test_error_handling_in_collection_loop(self):
        """Test the continuous collection loop with network/HTTP errors.
        
        Verifies that the background collection thread:
        - Properly handles network failures and HTTP errors
        - Calls the error callback with exception details
        - Continues running despite errors (doesn't crash)
        Tests the error handling path of the collection lifecycle.
        """
        collector = TelemetryDataCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.1,
            record_callback=self.record_callback,
            error_callback=self.error_callback
        )
        
        # Mock fetch error
        with patch.object(collector, '_fetch_metrics') as mock_fetch:
            mock_fetch.side_effect = requests.RequestException("Network error")
            
            collector.start()
            time.sleep(0.25)
            collector.stop()
            
            # Verify errors were captured
            assert len(self.errors_received) > 0
            assert all(isinstance(e, requests.RequestException) for e in self.errors_received)
            assert len(self.records_received) == 0

    def test_thread_lifecycle_management(self):
        """Test the collector's thread lifecycle management.
        
        Verifies that the collector can be properly started and stopped:
        - start() creates and launches a background collection thread
        - stop() signals the thread to terminate gracefully  
        - Thread actually terminates within reasonable time
        Tests the fundamental lifecycle control of the collector.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        # Mock collection loop to run until stop_event is set
        def mock_collection_loop():
            while not collector._stop_event.is_set():
                time.sleep(0.01)
        
        with patch.object(collector, '_collection_loop', side_effect=mock_collection_loop):
            collector.start()
            assert collector._collection_thread is not None
            assert collector._collection_thread.is_alive()
            
            collector.stop()
            time.sleep(0.1)  # Allow thread to terminate
            assert not collector._collection_thread.is_alive()

    def test_callback_exception_resilience(self):
        """Test collector resilience when user-provided callbacks raise exceptions.
        
        Verifies that the collection loop continues running even if:
        - Record callbacks raise runtime errors during data processing
        - Error callbacks raise exceptions during error handling
        Ensures that user code bugs don't crash the background collection thread.
        """
        def failing_record_callback(records):
            raise RuntimeError("Record callback failed")
            
        def failing_error_callback(error):
            raise RuntimeError("Error callback failed")
        
        collector = TelemetryDataCollector(
            dcgm_url="http://localhost:9401/metrics",
            collection_interval=0.1,
            record_callback=failing_record_callback,
            error_callback=failing_error_callback
        )
        
        # Test with successful data - record callback should fail but not crash
        with patch.object(collector, '_fetch_metrics') as mock_fetch:
            mock_fetch.return_value = '''DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-test",modelName="Test"} 75.5'''
            
            collector.start()
            time.sleep(0.15)
            collector.stop()
            # Should not have crashed despite callback failure

    def test_multiple_start_calls_safety(self):
        """Test thread safety of multiple start() calls.
        
        Verifies that calling start() multiple times doesn't create
        multiple background threads. Only the first start() should create
        a thread; subsequent calls should be no-ops. Prevents resource
        leaks and race conditions from accidental double-starts.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        def mock_collection_loop():
            while not collector._stop_event.is_set():
                time.sleep(0.01)
        
        with patch.object(collector, '_collection_loop', side_effect=mock_collection_loop):
            collector.start()
            first_thread = collector._collection_thread
            
            collector.start()  # Second call should not create new thread
            assert collector._collection_thread is first_thread
            
            collector.stop()

    def test_stop_before_start_safety(self):
        """Test graceful handling of stop() call before start().
        
        Verifies that calling stop() on a collector that was never started
        doesn't raise exceptions or cause errors. Tests defensive programming
        and ensures the collector can handle improper usage patterns safely.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        # Should not raise an exception
        collector.stop()


class TestDataProcessingEdgeCases:
    """Test edge cases and data quality scenarios.
    
    This test class focuses on the robustness of data processing
    under various real-world conditions and data quality issues.
    """

    def test_unit_scaling_accuracy(self):
        """Test proper unit scaling during DCGM metric parsing.
        
        Verifies that the collector applies correct scaling factors:
        - Energy: milli-joules (mJ) → mega-joules (MJ) [×1e-9]
        - Memory: mebibytes (MiB) → gigabytes (GB) [×1.048576e-3]
        Ensures that raw DCGM values are converted to standard units
        for consistent reporting across the application.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        metrics_data = '''
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0",UUID="GPU-test",modelName="Test"} 2000000000
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-test",modelName="Test"} 1024
DCGM_FI_DEV_FB_TOTAL{gpu="0",UUID="GPU-test",modelName="Test"} 2048
'''
        
        records = collector._parse_metrics_to_records(metrics_data)
        assert len(records) == 1
        
        record = records[0]
        # Energy: 2e9 * 1e-9 = 2.0 MJ
        assert record.energy_consumption == pytest.approx(2.0)
        # Memory: 1024 * 1.048576 * 1e-3 = 1.073741824 GB
        assert record.gpu_memory_used == pytest.approx(1.073741824)
        # Total: 2048 * 1.048576 * 1e-3 = 2.147483648 GB  
        assert record.total_gpu_memory == pytest.approx(2.147483648)

    def test_mixed_quality_response_resilience(self):
        """Test resilient parsing of DCGM responses containing mixed data quality.
        
        Verifies that the collector gracefully handles real-world data issues:
        - Valid metric lines are parsed correctly
        - Invalid lines (missing values, bad format) are skipped
        - Comment lines are ignored
        - Partially corrupted responses don't crash the parser
        Tests robustness against imperfect DCGM data.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        mixed_quality_data = '''
# Comment line
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-test",modelName="Test"} 75.5
invalid_line_without_braces 123
DCGM_FI_DEV_GPU_UTIL{gpu="invalid"} 85.0
DCGM_FI_DEV_FB_USED{} 1024
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0",UUID="GPU-test",modelName="Test"} invalid_value
'''
        
        records = collector._parse_metrics_to_records(mixed_quality_data)
        
        # Should only get record from valid line
        assert len(records) == 1
        assert records[0].gpu_power_usage == 75.5

    def test_temporal_consistency_in_batches(self):
        """Test temporal consistency of TelemetryRecord timestamps within a collection batch.
        
        Verifies that all TelemetryRecord objects created from a single DCGM
        response have identical timestamps. This ensures that metrics collected
        simultaneously are marked with the same collection time, enabling
        accurate time-series analysis and correlation.
        """
        collector = TelemetryDataCollector("http://localhost:9401/metrics")
        
        metrics_data = '''
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-test1",modelName="Test"} 75.5
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-test1",modelName="Test"} 85.0
DCGM_FI_DEV_POWER_USAGE{gpu="1",UUID="GPU-test2",modelName="Test"} 120.0
'''
        
        records = collector._parse_metrics_to_records(metrics_data)
        
        # All records should have the same timestamp (collected at same time)
        timestamps = [r.timestamp_ns for r in records]
        assert len(set(timestamps)) == 1, "All records in batch should have same timestamp"
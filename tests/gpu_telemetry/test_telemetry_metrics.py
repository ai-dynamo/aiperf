# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums.metric_enums import MetricType
from aiperf.common.models.telemetry_models import TelemetryMetrics, TelemetryRecord
from aiperf.metrics.base_telemetry_metric import BaseTelemetryMetric
from aiperf.metrics.gpu_telemetry_types.gpu_power_usage_metric import (
    GpuPowerUsageMetric,
)
from aiperf.metrics.gpu_telemetry_types.gpu_utilization_metric import (
    GpuUtilizationMetric,
)


class TestBaseTelemetryMetric:
    """Test the abstract base class for telemetry metrics.

    This test class focuses on the base metric functionality including
    abstract method enforcement, batch processing logic, and the generic
    metric interface. It does NOT test specific metric implementations.
    """

    def test_base_telemetry_metric_is_abstract(self):
        """Test that BaseTelemetryMetric cannot be instantiated directly.

        Verifies that the abstract base class properly enforces implementation
        of required methods (_extract_value) and cannot be used directly.
        This ensures proper inheritance patterns for metric implementations.
        """

        with pytest.raises(TypeError):
            BaseTelemetryMetric()

    def test_base_telemetry_metric_type_classification(self):
        """Test that BaseTelemetryMetric has correct metric type classification.

        Verifies that all telemetry metrics are properly classified as
        MetricType.TELEMETRY, enabling correct routing and processing
        in the metrics pipeline.
        """

        assert BaseTelemetryMetric.type == MetricType.TELEMETRY

    def test_concrete_implementation_interface(self):
        """Test that concrete metric implementations work with the base interface.

        Verifies that a properly implemented concrete metric class can be
        instantiated and provides all required attributes (tag, header, unit, type).
        Tests the contract that concrete metrics must fulfill.
        """

        class TestTelemetryMetric(BaseTelemetryMetric[float]):
            tag = "test_metric"
            header = "Test Metric"
            unit = "units"

            def _extract_value(self, record: TelemetryRecord) -> float | None:
                return record.telemetry_data.gpu_power_usage

        metric = TestTelemetryMetric()
        assert metric.type == MetricType.TELEMETRY
        assert metric.tag == "test_metric"
        assert metric.header == "Test Metric"
        assert metric.unit == "units"

    def test_batch_processing_single_gpu_aggregation(self):
        """Test batch processing logic for single GPU scenarios.

        Verifies that the base metric class correctly aggregates multiple
        time-series data points for a single GPU into a list of values.
        Tests the core batch processing mechanism used for statistical analysis.
        """

        class TestMetric(BaseTelemetryMetric[float]):
            tag = "test_single_gpu"
            header = "Test Metric"
            unit = "W"

            def _extract_value(self, record: TelemetryRecord) -> float | None:
                return record.telemetry_data.gpu_power_usage

        metric = TestMetric()

        # Create sample data from conftest.py pattern
        records = [
            TelemetryRecord(
                timestamp_ns=1000000000 + i * 33000000,  # 33ms apart (~30Hz)
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=0,
                gpu_model_name="Test GPU",
                gpu_uuid="GPU-test-uuid",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=100.0 + i * 10.0,
                ),
            )
            for i in range(3)
        ]

        gpu_values = metric.process_telemetry_batch(records)

        assert len(gpu_values) == 1  # Single GPU
        assert "GPU-test-uuid" in gpu_values
        assert gpu_values["GPU-test-uuid"] == [100.0, 110.0, 120.0]

    def test_batch_processing_none_value_filtering(self):
        """Test that None values are properly filtered during batch processing.

        Verifies that the base metric class correctly handles missing data
        by filtering out None values while preserving valid measurements.
        This ensures robust statistical analysis with incomplete data.
        """

        class TestMetric(BaseTelemetryMetric[float]):
            tag = "test_none_filtering"
            header = "Test Metric"
            unit = "W"

            def _extract_value(self, record: TelemetryRecord) -> float | None:
                return record.telemetry_data.gpu_power_usage

        metric = TestMetric()

        records = [
            TelemetryRecord(
                timestamp_ns=1000000000,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=0,
                gpu_model_name="Test GPU",
                gpu_uuid="GPU-test-uuid",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=100.0,
                ),
            ),
            TelemetryRecord(
                timestamp_ns=1000033000,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=0,
                gpu_model_name="Test GPU",
                gpu_uuid="GPU-test-uuid",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=None,  # Missing data point
                ),
            ),
            TelemetryRecord(
                timestamp_ns=1000066000,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=0,
                gpu_model_name="Test GPU",
                gpu_uuid="GPU-test-uuid",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=125.0,
                ),
            ),
        ]

        gpu_values = metric.process_telemetry_batch(records)

        assert len(gpu_values) == 1
        assert "GPU-test-uuid" in gpu_values
        assert gpu_values["GPU-test-uuid"] == [100.0, 125.0]  # None value filtered out

    def test_batch_processing_multi_node_separation(self):
        """Test that telemetry from different nodes with same GPU index stays separate.

        Verifies that when multiple nodes each have a GPU with the same gpu_index
        (e.g., both have "GPU 0"), the telemetry data is kept separate by using
        gpu_uuid as the dictionary key. This prevents incorrectly merging data
        from different physical GPUs that happen to have the same index on their
        respective nodes.
        """

        class TestMetric(BaseTelemetryMetric[float]):
            tag = "test_multi_node"
            header = "Test Metric"
            unit = "W"

            def _extract_value(self, record: TelemetryRecord) -> float | None:
                return record.telemetry_data.gpu_power_usage

        metric = TestMetric()

        # Create records from two different nodes, both with gpu_index=0
        records = [
            # Node 1, GPU 0
            TelemetryRecord(
                timestamp_ns=1000000000,
                dcgm_url="http://node1:9401/metrics",
                gpu_index=0,
                gpu_model_name="NVIDIA A100",
                gpu_uuid="GPU-node1-uuid-0000",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=100.0,
                ),
            ),
            TelemetryRecord(
                timestamp_ns=1000033000,
                dcgm_url="http://node1:9401/metrics",
                gpu_index=0,
                gpu_model_name="NVIDIA A100",
                gpu_uuid="GPU-node1-uuid-0000",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=110.0,
                ),
            ),
            # Node 2, GPU 0 (same index, different UUID)
            TelemetryRecord(
                timestamp_ns=1000000000,
                dcgm_url="http://node2:9401/metrics",
                gpu_index=0,
                gpu_model_name="NVIDIA A100",
                gpu_uuid="GPU-node2-uuid-0000",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=200.0,
                ),
            ),
            TelemetryRecord(
                timestamp_ns=1000033000,
                dcgm_url="http://node2:9401/metrics",
                gpu_index=0,
                gpu_model_name="NVIDIA A100",
                gpu_uuid="GPU-node2-uuid-0000",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=210.0,
                ),
            ),
        ]

        gpu_values = metric.process_telemetry_batch(records)

        # Verify data is kept separate by UUID, not merged by gpu_index
        assert len(gpu_values) == 2  # Two GPUs despite same gpu_index
        assert "GPU-node1-uuid-0000" in gpu_values
        assert "GPU-node2-uuid-0000" in gpu_values
        assert gpu_values["GPU-node1-uuid-0000"] == [100.0, 110.0]
        assert gpu_values["GPU-node2-uuid-0000"] == [200.0, 210.0]

    def test_abstract_method_enforcement(self):
        """Test that abstract method _extract_value must be implemented.

        Verifies that concrete classes without proper _extract_value implementation
        cannot be instantiated. This enforces the metric implementation contract
        and prevents incomplete metric classes.
        """

        class IncompleteMetric(BaseTelemetryMetric[float]):
            tag = "incomplete"
            header = "Incomplete Metric"
            unit = "units"
            # Missing _extract_value implementation

        with pytest.raises(TypeError):
            IncompleteMetric()


class TestSpecificTelemetryMetrics:
    """Test specific telemetry metric implementations.

    This test class focuses on concrete metric implementations and their
    value extraction logic. Tests are kept minimal since the main functionality
    is already tested in the base class tests above.
    """

    def test_gpu_power_usage_metric_properties(self):
        """Test GpuPowerUsageMetric configuration and metadata.

        Verifies that the power usage metric has correct identifying properties
        for dashboard display and data processing pipeline routing.
        """

        metric = GpuPowerUsageMetric()

        assert metric.tag == "gpu_power_usage"
        assert metric.header == "GPU Power Usage"
        assert metric.type == MetricType.TELEMETRY
        assert metric.unit == "W"  # Watts

    def test_gpu_power_usage_value_extraction(self):
        """Test power usage value extraction from TelemetryRecord.

        Verifies that the metric correctly extracts the gpu_power_usage field
        from TelemetryRecord objects and handles None values appropriately.
        """

        metric = GpuPowerUsageMetric()

        # Test successful extraction
        record_with_value = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="Test GPU",
            gpu_uuid="GPU-test-uuid",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=22.582,
            ),
        )
        assert metric._extract_value(record_with_value) == 22.582

        # Test None handling
        record_with_none = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="Test GPU",
            gpu_uuid="GPU-test-uuid",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=None,
            ),
        )
        assert metric._extract_value(record_with_none) is None

    def test_gpu_utilization_metric_properties(self):
        """Test GpuUtilizationMetric configuration and metadata.

        Verifies that the utilization metric has correct identifying properties
        for dashboard display and data processing pipeline routing.
        """

        metric = GpuUtilizationMetric()

        assert metric.tag == "gpu_utilization"
        assert metric.header == "GPU Utilization"
        assert metric.type == MetricType.TELEMETRY
        assert metric.unit == "%"  # Percentage

    def test_gpu_utilization_value_extraction(self):
        """Test GPU utilization value extraction from TelemetryRecord.

        Verifies that the metric correctly extracts the gpu_utilization field
        from TelemetryRecord objects and handles None values appropriately.
        """

        metric = GpuUtilizationMetric()

        # Test successful extraction
        record_with_value = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="Test GPU",
            gpu_uuid="GPU-test-uuid",
            telemetry_data=TelemetryMetrics(
                gpu_utilization=85.0,
            ),
        )
        assert metric._extract_value(record_with_value) == 85.0

        # Test None handling
        record_with_none = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="Test GPU",
            gpu_uuid="GPU-test-uuid",
            telemetry_data=TelemetryMetrics(
                gpu_utilization=None,
            ),
        )
        assert metric._extract_value(record_with_none) is None

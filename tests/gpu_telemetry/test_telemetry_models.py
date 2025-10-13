# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.telemetry_models import (
    GpuMetricTimeSeries,
    GpuTelemetryData,
    GpuTelemetrySnapshot,
    TelemetryMetrics,
    TelemetryRecord,
)


class TestTelemetryRecord:
    """Test TelemetryRecord model validation and data structure integrity.

    This test class focuses on Pydantic model validation, field requirements,
    and data structure correctness. It does NOT test parsing logic or metric
    extraction - those belong in other test files.
    """

    def test_telemetry_record_complete_creation(self):
        """Test creating a TelemetryRecord with all fields populated.

        Verifies that a fully-populated TelemetryRecord stores all fields correctly
        including both required fields (timestamp, dcgm_url, gpu_index, etc.) and
        optional metadata fields (pci_bus_id, device, hostname).
        """

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA RTX 6000 Ada Generation",
            gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
            pci_bus_id="00000000:02:00.0",
            device="nvidia0",
            hostname="ed7e7a5e585f",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=75.5,
                energy_consumption=1000000000,
                gpu_utilization=85.0,
                gpu_memory_used=15.26,
            ),
        )

        assert record.timestamp_ns == 1000000000
        assert record.dcgm_url == "http://localhost:9401/metrics"
        assert record.gpu_index == 0
        assert record.gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert record.gpu_uuid == "GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc"

        assert record.pci_bus_id == "00000000:02:00.0"
        assert record.device == "nvidia0"
        assert record.hostname == "ed7e7a5e585f"

        assert record.telemetry_data.gpu_power_usage == 75.5
        assert record.telemetry_data.energy_consumption == 1000000000
        assert record.telemetry_data.gpu_utilization == 85.0
        assert record.telemetry_data.gpu_memory_used == 15.26

    def test_telemetry_record_minimal_creation(self):
        """Test creating a TelemetryRecord with only required fields.

        Verifies that TelemetryRecord can be created with minimal required fields
        and that optional fields default to None. This tests the flexibility
        needed for varying DCGM response completeness.
        """

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://node2:9401/metrics",
            gpu_index=1,
            gpu_model_name="NVIDIA H100",
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
            telemetry_data=TelemetryMetrics(),
        )

        # Verify required fields are set
        assert record.timestamp_ns == 1000000000
        assert record.dcgm_url == "http://node2:9401/metrics"
        assert record.gpu_index == 1
        assert record.gpu_model_name == "NVIDIA H100"
        assert record.gpu_uuid == "GPU-00000000-0000-0000-0000-000000000001"

        assert record.pci_bus_id is None
        assert record.device is None
        assert record.hostname is None
        assert record.telemetry_data.gpu_power_usage is None
        assert record.telemetry_data.energy_consumption is None
        assert record.telemetry_data.gpu_utilization is None
        assert record.telemetry_data.gpu_memory_used is None

    def test_telemetry_record_field_validation(self):
        """Test Pydantic validation of required fields.

        Verifies that TelemetryRecord enforces required field validation
        and raises appropriate validation errors when required fields
        are missing. Tests the data integrity guarantees.
        """

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA RTX 6000",
            gpu_uuid="GPU-test-uuid",
            telemetry_data=TelemetryMetrics(),
        )
        assert record.timestamp_ns == 1000000000

        with pytest.raises(ValidationError):  # Pydantic validation error
            TelemetryRecord()  # No fields provided

    def test_telemetry_record_metadata_structure(self):
        """Test the hierarchical metadata structure for GPU identification.

        Verifies that TelemetryRecord properly supports the hierarchical
        identification structure needed for telemetry organization:
        dcgm_url -> gpu_uuid -> metadata. This structure enables proper
        grouping and filtering in the dashboard.
        """

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://gpu-node-01:9401/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA RTX 6000 Ada Generation",
            gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
            pci_bus_id="00000000:02:00.0",
            device="nvidia0",
            hostname="gpu-node-01",
            telemetry_data=TelemetryMetrics(),
        )

        # Verify hierarchical identification works
        # Level 1: DCGM endpoint identification
        assert record.dcgm_url == "http://gpu-node-01:9401/metrics"

        # Level 2: Unique GPU identification
        assert record.gpu_uuid == "GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc"

        # Level 3: Human-readable metadata
        assert record.gpu_index == 0  # For display ordering
        assert record.gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert record.hostname == "gpu-node-01"

        # Level 4: Hardware-specific metadata
        assert record.pci_bus_id == "00000000:02:00.0"
        assert record.device == "nvidia0"


class TestGpuTelemetrySnapshot:
    """Test GpuTelemetrySnapshot model for grouped metric collection."""

    def test_snapshot_creation_with_metrics(self):
        """Test creating a snapshot with multiple metrics."""
        snapshot = GpuTelemetrySnapshot(
            timestamp_ns=1000000000,
            metrics={
                "gpu_power_usage": 75.5,
                "gpu_utilization": 85.0,
                "gpu_memory_used": 15.26,
            },
        )

        assert snapshot.timestamp_ns == 1000000000
        assert len(snapshot.metrics) == 3
        assert snapshot.metrics["gpu_power_usage"] == 75.5
        assert snapshot.metrics["gpu_utilization"] == 85.0
        assert snapshot.metrics["gpu_memory_used"] == 15.26

    def test_snapshot_empty_metrics(self):
        """Test creating a snapshot with no metrics."""
        snapshot = GpuTelemetrySnapshot(timestamp_ns=2000000000, metrics={})

        assert snapshot.timestamp_ns == 2000000000
        assert len(snapshot.metrics) == 0


class TestGpuMetricTimeSeries:
    """Test GpuMetricTimeSeries model for grouped time series data."""

    def test_append_snapshot(self):
        """Test adding snapshots to time series."""
        time_series = GpuMetricTimeSeries()

        metrics1 = {"power": 100.0, "util": 80.0}
        metrics2 = {"power": 110.0, "util": 85.0}

        time_series.append_snapshot(metrics1, 1000000000)
        time_series.append_snapshot(metrics2, 2000000000)

        assert len(time_series.snapshots) == 2
        assert time_series.snapshots[0].timestamp_ns == 1000000000
        assert time_series.snapshots[0].metrics == metrics1
        assert time_series.snapshots[1].timestamp_ns == 2000000000
        assert time_series.snapshots[1].metrics == metrics2

    def test_get_metric_values(self):
        """Test extracting values for a specific metric."""
        time_series = GpuMetricTimeSeries()

        time_series.append_snapshot({"power": 100.0, "util": 80.0}, 1000000000)
        time_series.append_snapshot({"power": 110.0, "util": 85.0}, 2000000000)
        time_series.append_snapshot({"util": 90.0}, 3000000000)  # Missing power

        power_values = time_series.get_metric_values("power")
        util_values = time_series.get_metric_values("util")

        assert power_values == [(100.0, 1000000000), (110.0, 2000000000)]
        assert util_values == [
            (80.0, 1000000000),
            (85.0, 2000000000),
            (90.0, 3000000000),
        ]

    def test_to_metric_result_success(self):
        """Test converting time series to MetricResult."""
        time_series = GpuMetricTimeSeries()

        time_series.append_snapshot({"power": 100.0}, 1000000000)
        time_series.append_snapshot({"power": 120.0}, 2000000000)
        time_series.append_snapshot({"power": 80.0}, 3000000000)

        result = time_series.to_metric_result("power", "gpu_power", "GPU Power", "W")

        assert result.tag == "gpu_power"
        assert result.header == "GPU Power"
        assert result.unit == "W"
        assert result.min == 80.0
        assert result.max == 120.0
        assert result.avg == 100.0  # (100 + 120 + 80) / 3
        assert result.count == 3

    def test_to_metric_result_no_data(self):
        """Test MetricResult conversion with no data for specified metric."""
        time_series = GpuMetricTimeSeries()

        with pytest.raises(NoMetricValue) as exc_info:
            time_series.to_metric_result("nonexistent", "tag", "header", "unit")

        assert "No telemetry data available for metric 'nonexistent'" in str(
            exc_info.value
        )


class TestGpuTelemetryData:
    """Test GpuTelemetryData model with grouped approach."""

    def test_add_record_grouped(self):
        """Test adding TelemetryRecord creates grouped snapshots."""
        from aiperf.common.models.telemetry_models import GpuMetadata

        metadata = GpuMetadata(
            gpu_index=0, gpu_uuid="GPU-test-uuid", model_name="Test GPU"
        )

        telemetry_data = GpuTelemetryData(metadata=metadata)

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="Test GPU",
            gpu_uuid="GPU-test-uuid",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=100.0,
                gpu_utilization=80.0,
                gpu_memory_used=15.0,
            ),
        )

        telemetry_data.add_record(record)

        assert len(telemetry_data.time_series.snapshots) == 1
        snapshot = telemetry_data.time_series.snapshots[0]
        assert snapshot.timestamp_ns == 1000000000
        assert len(snapshot.metrics) == 3
        assert snapshot.metrics["gpu_power_usage"] == 100.0
        assert snapshot.metrics["gpu_utilization"] == 80.0
        assert snapshot.metrics["gpu_memory_used"] == 15.0

    def test_add_record_filters_none_values(self):
        """Test that None metric values are filtered out."""
        from aiperf.common.models.telemetry_models import GpuMetadata

        metadata = GpuMetadata(
            gpu_index=0, gpu_uuid="GPU-test-uuid", model_name="Test GPU"
        )

        telemetry_data = GpuTelemetryData(metadata=metadata)

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="Test GPU",
            gpu_uuid="GPU-test-uuid",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=100.0,
                gpu_utilization=None,  # Should be filtered out
                gpu_memory_used=15.0,
            ),
        )

        telemetry_data.add_record(record)

        snapshot = telemetry_data.time_series.snapshots[0]
        assert len(snapshot.metrics) == 2
        assert "gpu_power_usage" in snapshot.metrics
        assert "gpu_memory_used" in snapshot.metrics
        assert "gpu_utilization" not in snapshot.metrics

    def test_get_metric_result(self):
        """Test getting MetricResult for a specific metric."""
        from aiperf.common.models.telemetry_models import GpuMetadata

        metadata = GpuMetadata(
            gpu_index=0, gpu_uuid="GPU-test-uuid", model_name="Test GPU"
        )

        telemetry_data = GpuTelemetryData(metadata=metadata)

        # Add multiple records
        for i, power in enumerate([100.0, 120.0, 80.0]):
            record = TelemetryRecord(
                timestamp_ns=1000000000 + i * 1000000,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=0,
                gpu_model_name="Test GPU",
                gpu_uuid="GPU-test-uuid",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=power,
                ),
            )
            telemetry_data.add_record(record)

        result = telemetry_data.get_metric_result(
            "gpu_power_usage", "power_tag", "GPU Power", "W"
        )

        assert result.tag == "power_tag"
        assert result.header == "GPU Power"
        assert result.unit == "W"
        assert result.min == 80.0
        assert result.max == 120.0
        assert result.avg == 100.0

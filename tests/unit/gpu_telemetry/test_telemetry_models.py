# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pydantic import ValidationError

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.common.models.telemetry_models import (
    GpuMetadata,
    GpuMetricTimeSeries,
    GpuTelemetryData,
    GpuTelemetrySnapshot,
    TelemetryMetrics,
    TelemetryRecord,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_record(timestamp_ns: int, **metrics: float) -> TelemetryRecord:
    """Create TelemetryRecord with minimal boilerplate for testing."""
    return TelemetryRecord(
        timestamp_ns=timestamp_ns,
        dcgm_url="http://localhost:9401/metrics",
        gpu_index=0,
        gpu_model_name="Test GPU",
        gpu_uuid="GPU-test-uuid",
        telemetry_data=TelemetryMetrics(**metrics),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gpu_telemetry_data() -> GpuTelemetryData:
    """GpuTelemetryData with standard test metadata."""
    return GpuTelemetryData(
        metadata=GpuMetadata(
            gpu_index=0, gpu_uuid="GPU-test-uuid", gpu_model_name="Test GPU"
        )
    )


@pytest.fixture
def time_series_4pt() -> GpuMetricTimeSeries:
    """Time series with 4 data points at 1s intervals for filtering tests."""
    ts = GpuMetricTimeSeries()
    ts.append_snapshot({"power": 100.0}, 1_000_000_000)
    ts.append_snapshot({"power": 110.0}, 2_000_000_000)
    ts.append_snapshot({"power": 120.0}, 3_000_000_000)
    ts.append_snapshot({"power": 130.0}, 4_000_000_000)
    return ts


@pytest.fixture
def counter_time_series() -> GpuMetricTimeSeries:
    """Time series with counter data: baseline at 1s, profiling from 2s-4s."""
    ts = GpuMetricTimeSeries()
    ts.append_snapshot({"energy": 1000.0}, 1_000_000_000)  # baseline
    ts.append_snapshot({"energy": 1200.0}, 2_000_000_000)  # profiling start
    ts.append_snapshot({"energy": 1500.0}, 3_000_000_000)  # profiling
    ts.append_snapshot({"energy": 1800.0}, 4_000_000_000)  # profiling end
    return ts


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

        time_series.append_snapshot({"power": 100.0, "util": 80.0}, 1000000000)
        time_series.append_snapshot({"power": 110.0, "util": 85.0}, 2000000000)

        assert len(time_series) == 2
        assert list(time_series.timestamps) == [1000000000, 2000000000]
        assert list(time_series.get_metric_array("power")) == [100.0, 110.0]
        assert list(time_series.get_metric_array("util")) == [80.0, 85.0]

    def test_consistent_metric_schema(self):
        """Static-schema collectors (DCGM, PyNVML) emit the same keys every
        scrape; the time series stores them column-by-column with no NaN
        backfill.
        """
        time_series = GpuMetricTimeSeries()

        time_series.append_snapshot({"power": 100.0, "util": 80.0}, 1000000000)
        time_series.append_snapshot({"power": 110.0, "util": 85.0}, 2000000000)
        time_series.append_snapshot({"power": 120.0, "util": 90.0}, 3000000000)

        power = time_series.get_metric_array("power")
        util = time_series.get_metric_array("util")

        # All values present
        assert list(power) == [100.0, 110.0, 120.0]
        assert list(util) == [80.0, 85.0, 90.0]

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

    # Columnar storage tests

    def test_len(self):
        """Test __len__ returns correct size."""
        time_series = GpuMetricTimeSeries()
        assert len(time_series) == 0

        time_series.append_snapshot({"power": 100.0}, 1_000_000_000)
        assert len(time_series) == 1

        time_series.append_snapshot({"power": 110.0}, 2_000_000_000)
        assert len(time_series) == 2

    def test_timestamps_property(self):
        """Test timestamps property returns correct array view."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0}, 1_000_000_000)
        time_series.append_snapshot({"power": 110.0}, 2_000_000_000)

        timestamps = time_series.timestamps
        assert list(timestamps) == [1_000_000_000, 2_000_000_000]

    def test_get_metric_array(self):
        """Test get_metric_array returns correct array view."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0, "util": 80.0}, 1_000_000_000)
        time_series.append_snapshot({"power": 110.0, "util": 85.0}, 2_000_000_000)

        power = time_series.get_metric_array("power")
        util = time_series.get_metric_array("util")
        unknown = time_series.get_metric_array("unknown")

        assert list(power) == [100.0, 110.0]
        assert list(util) == [80.0, 85.0]
        assert unknown is None

    def test_stats_computation(self):
        """Test stats computation on consistent metric data."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0}, 1_000_000_000)
        time_series.append_snapshot({"power": 150.0}, 2_000_000_000)
        time_series.append_snapshot({"power": 200.0}, 3_000_000_000)

        result = time_series.to_metric_result("power", "tag", "header", "W")

        assert result.count == 3
        assert result.avg == 150.0  # (100 + 150 + 200) / 3
        assert result.min == 100.0
        assert result.max == 200.0
        assert result.current == 200.0  # Last value

    @pytest.mark.parametrize(
        ("values", "expected_std"),
        [
            pytest.param([100.0], 0.0, id="single_sample"),
            pytest.param(
                [100.0, 150.0, 200.0],
                np.std([100.0, 150.0, 200.0], ddof=1),
                id="multiple_samples",
            ),
        ],
    )
    def test_stats_uses_sample_std(self, values: list[float], expected_std: float):
        """Test std uses sample std (ddof=1) with edge case handling."""
        time_series = GpuMetricTimeSeries()
        for i, val in enumerate(values):
            time_series.append_snapshot({"power": val}, (i + 1) * 1_000_000_000)

        result = time_series.to_metric_result("power", "tag", "header", "W")

        assert result.std == expected_std
        assert result.count == len(values)

    def test_grow_preserves_data(self):
        """Test array growth preserves existing data."""
        time_series = GpuMetricTimeSeries()
        # Add more than initial capacity (128)
        for i in range(200):
            time_series.append_snapshot({"power": float(i)}, i * 1_000_000_000)

        assert len(time_series) == 200
        assert time_series.get_metric_array("power")[0] == 0.0
        assert time_series.get_metric_array("power")[199] == 199.0

    def test_insert_sorted_out_of_order(self):
        """Test insert-sorted handles out-of-order timestamps."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0}, 1_000_000_000)
        time_series.append_snapshot({"power": 300.0}, 3_000_000_000)
        time_series.append_snapshot({"power": 200.0}, 2_000_000_000)  # Out of order!

        # Data should be sorted by timestamp
        assert list(time_series.timestamps) == [
            1_000_000_000,
            2_000_000_000,
            3_000_000_000,
        ]
        assert list(time_series.get_metric_array("power")) == [100.0, 200.0, 300.0]

    def test_insert_sorted_preserves_all_metrics(self):
        """Test insert-sorted correctly preserves all metric values during shift."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0, "util": 80.0}, 1_000_000_000)
        time_series.append_snapshot({"power": 120.0, "util": 90.0}, 3_000_000_000)
        time_series.append_snapshot(
            {"power": 150.0, "util": 85.0}, 2_000_000_000
        )  # Out of order!

        # After sorting: ts1, ts2(inserted), ts3
        assert list(time_series.timestamps) == [
            1_000_000_000,
            2_000_000_000,
            3_000_000_000,
        ]

        power = time_series.get_metric_array("power")
        assert list(power) == [100.0, 150.0, 120.0]

        util = time_series.get_metric_array("util")
        assert list(util) == [80.0, 85.0, 90.0]

    def test_multiple_metrics_columnar_access(self):
        """Test columnar access for multiple metrics."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0, "util": 80.0}, 1_000_000_000)
        time_series.append_snapshot({"power": 110.0, "util": 85.0}, 2_000_000_000)

        assert len(time_series) == 2

        power = time_series.get_metric_array("power")
        util = time_series.get_metric_array("util")
        unknown = time_series.get_metric_array("unknown")

        assert list(power) == [100.0, 110.0]
        assert list(util) == [80.0, 85.0]
        assert unknown is None

    def test_timestamps_and_metrics_aligned(self):
        """Test timestamps and metric values remain aligned."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0}, 1_000_000_000)
        time_series.append_snapshot({"power": 150.0}, 2_000_000_000)
        time_series.append_snapshot({"power": 200.0}, 3_000_000_000)

        power = time_series.get_metric_array("power")
        timestamps = time_series.timestamps

        # Verify alignment
        assert len(power) == len(timestamps) == 3
        assert list(zip(timestamps, power, strict=False)) == [
            (1_000_000_000, 100.0),
            (2_000_000_000, 150.0),
            (3_000_000_000, 200.0),
        ]

    def test_unknown_metric_raises_no_metric_value(self):
        """Test NoMetricValue raised for unknown metric."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0}, 1_000_000_000)

        # Unknown metric returns None from get_metric_array
        assert time_series.get_metric_array("unknown") is None

        # to_metric_result raises NoMetricValue for unknown metric
        with pytest.raises(NoMetricValue):
            time_series.to_metric_result("unknown", "tag", "header", "unit")

    # Time filtering tests

    @pytest.mark.parametrize(
        ("time_filter", "expected_mask"),
        [
            pytest.param(None, [True, True, True, True], id="no_filter"),
            pytest.param(TimeRangeFilter(start_ns=2_000_000_000), [False, True, True, True], id="start_only"),
            pytest.param(TimeRangeFilter(end_ns=2_500_000_000), [True, True, False, False], id="end_only"),
            pytest.param(TimeRangeFilter(start_ns=1_500_000_000, end_ns=3_500_000_000), [False, True, True, False], id="range"),
        ],
    )  # fmt: skip
    def test_get_time_mask(
        self,
        time_series_4pt: GpuMetricTimeSeries,
        time_filter: TimeRangeFilter | None,
        expected_mask: list[bool],
    ):
        """Test get_time_mask with various filter configurations."""
        mask = time_series_4pt.get_time_mask(time_filter)
        assert list(mask) == expected_mask

    @pytest.mark.parametrize(
        ("time_filter", "expected_idx"),
        [
            pytest.param(None, None, id="no_filter"),
            pytest.param(TimeRangeFilter(end_ns=3_000_000_000), None, id="no_start"),
            pytest.param(TimeRangeFilter(start_ns=2_000_000_000, end_ns=5_000_000_000), 0, id="baseline_at_0"),
            pytest.param(TimeRangeFilter(start_ns=3_000_000_000, end_ns=5_000_000_000), 1, id="baseline_at_1"),
        ],
    )  # fmt: skip
    def test_get_reference_idx(
        self,
        time_series_4pt: GpuMetricTimeSeries,
        time_filter: TimeRangeFilter | None,
        expected_idx: int | None,
    ):
        """Test get_reference_idx with various filter configurations."""
        assert time_series_4pt.get_reference_idx(time_filter) == expected_idx

    def test_get_reference_idx_no_baseline_before_start(self):
        """Test get_reference_idx returns None when no data before start_ns."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0}, 2_000_000_000)
        time_series.append_snapshot({"power": 110.0}, 3_000_000_000)

        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=4_000_000_000)
        assert time_series.get_reference_idx(time_filter) is None

    def test_to_metric_result_filtered_gauge(self):
        """Test gauge stats computed only on filtered data."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 50.0}, 1_000_000_000)  # warmup - excluded
        time_series.append_snapshot({"power": 100.0}, 2_000_000_000)  # profiling
        time_series.append_snapshot({"power": 120.0}, 3_000_000_000)  # profiling
        time_series.append_snapshot({"power": 80.0}, 4_000_000_000)  # profiling

        time_filter = TimeRangeFilter(start_ns=2_000_000_000, end_ns=5_000_000_000)
        result = time_series.to_metric_result_filtered(
            "power", "tag", "header", "W", time_filter, is_counter=False
        )

        # Stats should only include values 100, 120, 80 (not 50)
        assert result.count == 3
        assert result.min == 80.0
        assert result.max == 120.0
        assert result.avg == 100.0  # (100 + 120 + 80) / 3

    def test_to_metric_result_filtered_counter_with_baseline(
        self, counter_time_series: GpuMetricTimeSeries
    ):
        """Test counter delta computed from baseline before start_ns."""
        time_filter = TimeRangeFilter(start_ns=2_000_000_000, end_ns=5_000_000_000)
        result = counter_time_series.to_metric_result_filtered(
            "energy", "tag", "header", "MJ", time_filter, is_counter=True
        )

        # Delta: final (1800) - baseline (1000) = 800
        # Counters only set avg, other fields are None
        assert result.avg == 800.0
        assert result.min is None
        assert result.max is None
        assert result.current is None

    def test_to_metric_result_filtered_counter_no_baseline(self):
        """Test counter delta uses first filtered value when no baseline exists."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"energy": 1000.0}, 2_000_000_000)
        time_series.append_snapshot({"energy": 1300.0}, 3_000_000_000)
        time_series.append_snapshot({"energy": 1600.0}, 4_000_000_000)

        # Filter starts before any data
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=5_000_000_000)
        result = time_series.to_metric_result_filtered(
            "energy", "tag", "header", "MJ", time_filter, is_counter=True
        )

        # No baseline: delta = final (1600) - first filtered (1000) = 600
        assert result.avg == 600.0

    def test_to_metric_result_filtered_counter_reset_clamped_to_zero(self):
        """Test counter reset (e.g., DCGM restart) clamps delta to 0."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"energy": 5000.0}, 1_000_000_000)  # baseline
        time_series.append_snapshot({"energy": 5500.0}, 2_000_000_000)
        time_series.append_snapshot({"energy": 100.0}, 3_000_000_000)  # after reset
        time_series.append_snapshot({"energy": 300.0}, 4_000_000_000)

        time_filter = TimeRangeFilter(start_ns=2_000_000_000, end_ns=5_000_000_000)
        result = time_series.to_metric_result_filtered(
            "energy", "tag", "header", "MJ", time_filter, is_counter=True
        )

        # Raw delta: 300 - 5000 = -4700, clamped to 0
        assert result.avg == 0.0

    def test_to_metric_result_filtered_empty_range(self):
        """Test NoMetricValue raised when filtered range has no data."""
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0}, 1_000_000_000)
        time_series.append_snapshot({"power": 110.0}, 2_000_000_000)

        # Filter for range with no data
        time_filter = TimeRangeFilter(start_ns=5_000_000_000, end_ns=6_000_000_000)

        with pytest.raises(NoMetricValue) as exc_info:
            time_series.to_metric_result_filtered(
                "power", "tag", "header", "W", time_filter, is_counter=False
            )
        assert "No data in time range" in str(exc_info.value)

    # -- Dynamic-schema regression coverage (PR #908 dynamo-ops review) --
    # AMDSMI emits a different set of fields per scrape when sensor reads
    # transiently fail. The following tests pin down the NaN-aware semantics
    # added to append_snapshot / _grow / to_metric_result*.

    def test_late_arriving_metric_backfilled_with_nan(self):
        """A field absent from snapshot 1 then present in snapshot 2 must
        allocate a new array NaN-backfilled rather than raising KeyError.
        """
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"power": 100.0}, 1_000_000_000)
        time_series.append_snapshot(
            {"power": 110.0, "temperature": 50.0}, 2_000_000_000
        )

        temp = time_series.get_metric_array("temperature")
        assert np.isnan(temp[0])
        assert temp[1] == 50.0

        # Stats over the late-arriving metric ignore the NaN slot.
        result = time_series.to_metric_result("temperature", "t", "h", "C")
        assert result.avg == 50.0
        assert result.min == 50.0
        assert result.max == 50.0

    def test_disappearing_metric_writes_nan(self):
        """A field present in snapshot 1 then absent from snapshot 2 must
        leave NaN at the new index, not garbage from np.empty.
        """
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot(
            {"power": 100.0, "temperature": 50.0}, 1_000_000_000
        )
        time_series.append_snapshot({"power": 110.0}, 2_000_000_000)

        temp = time_series.get_metric_array("temperature")
        assert temp[0] == 50.0
        assert np.isnan(temp[1])

        # Stats reflect only the real value.
        result = time_series.to_metric_result("temperature", "t", "h", "C")
        assert result.avg == 50.0
        assert result.count == 2  # count = number of scrapes, not non-NaN samples

    def test_intermittent_metric_only_uses_real_values(self):
        """Mixed present/absent across many snapshots: nan-aware stats use
        only the real samples.
        """
        time_series = GpuMetricTimeSeries()
        # Five scrapes; b is present at indices 0, 2, 4 (values 10, 30, 50).
        snapshots = [
            ({"a": 1.0, "b": 10.0}, 1_000_000_000),
            ({"a": 2.0}, 2_000_000_000),
            ({"a": 3.0, "b": 30.0}, 3_000_000_000),
            ({"a": 4.0}, 4_000_000_000),
            ({"a": 5.0, "b": 50.0}, 5_000_000_000),
        ]
        for metrics, ts in snapshots:
            time_series.append_snapshot(metrics, ts)

        b = time_series.get_metric_array("b")
        assert b[0] == 10.0
        assert np.isnan(b[1])
        assert b[2] == 30.0
        assert np.isnan(b[3])
        assert b[4] == 50.0

        result = time_series.to_metric_result("b", "t", "h", "u")
        assert result.avg == 30.0  # mean(10, 30, 50)
        assert result.min == 10.0
        assert result.max == 50.0

    def test_to_metric_result_all_nan_raises(self):
        """If every sample for a metric is NaN, raise NoMetricValue rather
        than emitting NaN stats and a numpy RuntimeWarning.
        """
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"a": 1.0, "b": 10.0}, 1_000_000_000)
        time_series.append_snapshot({"a": 2.0}, 2_000_000_000)

        # Filter so only the second scrape is in-range; b is NaN there.
        time_filter = TimeRangeFilter(start_ns=1_500_000_000, end_ns=3_000_000_000)
        with pytest.raises(NoMetricValue, match="All in-range samples"):
            time_series.to_metric_result_filtered(
                "b", "t", "h", "u", time_filter, is_counter=False
            )

    def test_to_metric_result_filtered_counter_nan_baseline_clamps_to_zero(self):
        """A counter whose reference sample is NaN (the metric arrived after
        the baseline scrape) and that has *no* in-window movement falls back
        to first-valid-in-window: delta = 0.0.
        """
        time_series = GpuMetricTimeSeries()
        # Baseline: only "a" is present.
        time_series.append_snapshot({"a": 10.0}, 1_000_000_000)
        # In-window: "energy" arrives at the same value twice (no movement).
        time_series.append_snapshot({"a": 20.0, "energy": 100.0}, 2_500_000_000)
        time_series.append_snapshot({"a": 30.0, "energy": 100.0}, 3_500_000_000)

        time_filter = TimeRangeFilter(start_ns=2_000_000_000, end_ns=4_000_000_000)
        result = time_series.to_metric_result_filtered(
            "energy", "t", "h", "MJ", time_filter, is_counter=True
        )
        # Reference = first valid in-window = 100; last valid in-window = 100.
        assert result.avg == 0.0

    def test_counter_delta_walks_back_for_valid_baseline(self):
        """If the chosen reference index is NaN but an earlier scrape had a
        valid value, walk back to it instead of zeroing the delta.
        """
        time_series = GpuMetricTimeSeries()
        # Two pre-window scrapes: scrape 0 has "energy", scrape 1 doesn't.
        time_series.append_snapshot({"energy": 100.0}, 1_000_000_000)
        time_series.append_snapshot({"a": 5.0}, 1_500_000_000)  # baseline NaN
        # In-window: monotonic counter movement.
        time_series.append_snapshot({"energy": 150.0}, 2_500_000_000)
        time_series.append_snapshot({"energy": 200.0}, 3_500_000_000)

        time_filter = TimeRangeFilter(start_ns=2_000_000_000, end_ns=4_000_000_000)
        result = time_series.to_metric_result_filtered(
            "energy", "t", "h", "MJ", time_filter, is_counter=True
        )
        # reference_idx = 1 (NaN); walk back to scrape 0 (= 100).
        # filtered_last = 200. delta = 200 - 100 = 100.
        assert result.avg == 100.0

    def test_counter_delta_skips_nan_final_sample(self):
        """If the last in-window sample is NaN but an earlier in-window scrape
        had a valid value, use that as ``filtered_last``.
        """
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"energy": 100.0}, 1_000_000_000)  # baseline
        time_series.append_snapshot({"energy": 200.0}, 2_500_000_000)
        time_series.append_snapshot({"a": 5.0}, 3_500_000_000)  # final NaN

        time_filter = TimeRangeFilter(start_ns=2_000_000_000, end_ns=4_000_000_000)
        result = time_series.to_metric_result_filtered(
            "energy", "t", "h", "MJ", time_filter, is_counter=True
        )
        # reference = 100, filtered_last = 200 (not NaN). delta = 100.
        assert result.avg == 100.0

    def test_counter_delta_filtered_all_nan_raises(self):
        """A filtered window with no valid samples raises NoMetricValue
        rather than silently reporting delta=0 from the all-NaN reference.
        """
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"energy": 100.0}, 1_000_000_000)
        time_series.append_snapshot({"a": 5.0}, 2_500_000_000)
        time_series.append_snapshot({"a": 6.0}, 3_500_000_000)

        time_filter = TimeRangeFilter(start_ns=2_000_000_000, end_ns=4_000_000_000)
        with pytest.raises(NoMetricValue, match="No valid"):
            time_series.to_metric_result_filtered(
                "energy", "t", "h", "MJ", time_filter, is_counter=True
            )

    def test_gauge_std_uses_non_nan_count_for_ddof_guard(self):
        """3 scrapes, only 1 has the metric → std=0.0 instead of NaN+warning
        (ddof=1 with one valid sample is degrees-of-freedom 0).
        """
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot({"a": 1.0, "b": 50.0}, 1_000_000_000)
        time_series.append_snapshot({"a": 2.0}, 2_000_000_000)
        time_series.append_snapshot({"a": 3.0}, 3_000_000_000)

        result = time_series.to_metric_result("b", "t", "h", "u")
        assert result.std == 0.0

        # Same check for the filtered path.
        result_f = time_series.to_metric_result_filtered(
            "b", "t", "h", "u", time_filter=None, is_counter=False
        )
        assert result_f.std == 0.0

    def test_current_uses_last_non_nan_sample(self):
        """``current`` should be the most recent *valid* sample so the
        realtime dashboard doesn't display NaN (and so change-detection
        doesn't republish every interval because NaN != NaN).
        """
        time_series = GpuMetricTimeSeries()
        time_series.append_snapshot(
            {"power": 100.0, "temperature": 50.0}, 1_000_000_000
        )
        time_series.append_snapshot({"power": 110.0}, 2_000_000_000)  # temp NaN
        time_series.append_snapshot({"power": 120.0}, 3_000_000_000)  # temp NaN

        result = time_series.to_metric_result("temperature", "t", "h", "C")
        assert result.current == 50.0

    def test_current_is_none_when_all_filtered_nan(self):
        """If a metric is registered but never had a valid value, ``current``
        is None rather than NaN. (The all-NaN guard raises NoMetricValue
        before we get here, so this exercises the per-metric helper directly.)
        """
        from aiperf.common.models.telemetry_models import _last_valid

        assert _last_valid(np.array([np.nan, np.nan, np.nan])) is None
        assert _last_valid(np.array([1.0, np.nan, np.nan])) == 1.0
        assert _last_valid(np.array([np.nan, 2.0, np.nan])) == 2.0
        assert _last_valid(np.array([1.0, 2.0, 3.0])) == 3.0


class TestGpuTelemetryData:
    """Test GpuTelemetryData model with grouped approach."""

    def test_add_record_grouped(self, gpu_telemetry_data: GpuTelemetryData):
        """Test adding TelemetryRecord creates grouped snapshots."""
        record = _make_record(
            1_000_000_000,
            gpu_power_usage=100.0,
            gpu_utilization=80.0,
            gpu_memory_used=15.0,
        )
        gpu_telemetry_data.add_record(record)

        ts = gpu_telemetry_data.time_series
        assert len(ts) == 1
        assert ts.timestamps[0] == 1_000_000_000
        assert ts.get_metric_array("gpu_power_usage")[0] == 100.0
        assert ts.get_metric_array("gpu_utilization")[0] == 80.0
        assert ts.get_metric_array("gpu_memory_used")[0] == 15.0

    def test_add_record_filters_none_values(self, gpu_telemetry_data: GpuTelemetryData):
        """Test that None metric values are filtered out."""
        record = _make_record(
            1_000_000_000,
            gpu_power_usage=100.0,
            gpu_memory_used=15.0,
            # gpu_utilization intentionally omitted (will be None)
        )
        gpu_telemetry_data.add_record(record)

        ts = gpu_telemetry_data.time_series
        assert len(ts) == 1
        assert ts.get_metric_array("gpu_power_usage") is not None
        assert ts.get_metric_array("gpu_memory_used") is not None
        assert ts.get_metric_array("gpu_utilization") is None

    def test_get_metric_result(self, gpu_telemetry_data: GpuTelemetryData):
        """Test getting MetricResult for a specific metric."""
        for i, power in enumerate([100.0, 120.0, 80.0]):
            gpu_telemetry_data.add_record(
                _make_record(1_000_000_000 + i * 1_000_000, gpu_power_usage=power)
            )

        result = gpu_telemetry_data.get_metric_result(
            "gpu_power_usage", "power_tag", "GPU Power", "W"
        )

        assert result.tag == "power_tag"
        assert result.header == "GPU Power"
        assert result.unit == "W"
        assert result.min == 80.0
        assert result.max == 120.0
        assert result.avg == 100.0

    def test_get_metric_result_with_time_filter(
        self, gpu_telemetry_data: GpuTelemetryData
    ):
        """Test getting MetricResult with time filtering."""
        # Add records: warmup + profiling
        for ts, power in [(1, 50.0), (2, 100.0), (3, 120.0), (4, 80.0)]:
            gpu_telemetry_data.add_record(
                _make_record(ts * 1_000_000_000, gpu_power_usage=power)
            )

        # Exclude warmup at 1s
        time_filter = TimeRangeFilter(start_ns=2_000_000_000, end_ns=5_000_000_000)
        result = gpu_telemetry_data.get_metric_result(
            "gpu_power_usage", "power_tag", "GPU Power", "W", time_filter=time_filter
        )

        # Stats should exclude warmup value of 50.0
        assert result.count == 3
        assert result.min == 80.0
        assert result.max == 120.0
        assert result.avg == 100.0  # (100 + 120 + 80) / 3

    def test_get_metric_result_counter_with_time_filter(
        self, gpu_telemetry_data: GpuTelemetryData
    ):
        """Test getting MetricResult for counter metric with delta calculation."""
        # Add records: baseline + profiling
        for ts, energy in [(1, 1000.0), (2, 1200.0), (3, 1500.0), (4, 1800.0)]:
            gpu_telemetry_data.add_record(
                _make_record(ts * 1_000_000_000, energy_consumption=energy)
            )

        time_filter = TimeRangeFilter(start_ns=2_000_000_000, end_ns=5_000_000_000)
        result = gpu_telemetry_data.get_metric_result(
            "energy_consumption",
            "energy_tag",
            "Energy",
            "MJ",
            time_filter=time_filter,
            is_counter=True,
        )

        # Delta: final (1800) - baseline (1000) = 800
        assert result.avg == 800.0

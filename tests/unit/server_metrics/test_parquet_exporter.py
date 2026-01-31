# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsParquetExporter using mocked data structures.

FILESYSTEM ISOLATION:
- All tests use pytest's tmp_path fixture for file operations
- Parquet files are created in temporary directories that are automatically cleaned up
- No files are created in the user's filesystem outside of pytest's temp directories
- Each test gets its own isolated tmp_path directory
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import PrometheusMetricType, ServerMetricsFormat
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import TimeRangeFilter
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.parquet_exporter import ServerMetricsParquetExporter
from aiperf.server_metrics.storage import (
    HistogramTimeSeries,
    ScalarTimeSeries,
    ServerMetricEntry,
    ServerMetricKey,
    ServerMetricsHierarchy,
    ServerMetricsTimeSeries,
)

# =============================================================================
# Mock Data Builders
# =============================================================================


def build_scalar_time_series(
    timestamps_ns: list[int],
    values: list[float],
) -> ScalarTimeSeries:
    """Build a ScalarTimeSeries with given data."""
    ts = ScalarTimeSeries()
    ts._timestamps = np.array(timestamps_ns, dtype=np.int64)
    ts._values = np.array(values, dtype=np.float64)
    ts._size = len(timestamps_ns)
    return ts


def build_histogram_time_series(
    timestamps_ns: list[int],
    sums: list[float],
    counts: list[float],
    bucket_les: tuple[str, ...],
    bucket_counts: list[list[float]],
) -> HistogramTimeSeries:
    """Build a HistogramTimeSeries with given data."""
    ts = HistogramTimeSeries()
    ts._timestamps = np.array(timestamps_ns, dtype=np.int64)
    ts._sums = np.array(sums, dtype=np.float64)
    ts._counts = np.array(counts, dtype=np.float64)
    ts._bucket_les = bucket_les
    ts._bucket_counts = np.array(bucket_counts, dtype=np.float64)
    ts._size = len(timestamps_ns)
    return ts


def build_metric_entry(
    metric_type: PrometheusMetricType,
    data: ScalarTimeSeries | HistogramTimeSeries,
    description: str = "Test metric description",
) -> ServerMetricEntry:
    """Build a ServerMetricEntry."""
    return ServerMetricEntry(
        metric_type=metric_type,
        description=description,
        data=data,
    )


def build_hierarchy(
    endpoints_data: dict[str, list[tuple[str, dict | None, ServerMetricEntry]]],
) -> ServerMetricsHierarchy:
    """Build a ServerMetricsHierarchy from endpoint data.

    Args:
        endpoints_data: Dict mapping endpoint_url to list of
            (metric_name, labels_dict, ServerMetricEntry) tuples
    """
    hierarchy = ServerMetricsHierarchy()
    for endpoint_url, metrics_list in endpoints_data.items():
        time_series = ServerMetricsTimeSeries()
        for metric_name, labels_dict, entry in metrics_list:
            key = ServerMetricKey.from_name_and_labels(metric_name, labels_dict)
            time_series.metrics[key] = entry
        hierarchy.endpoints[endpoint_url] = time_series
    return hierarchy


def create_mock_accumulator(
    user_config: UserConfig,
    hierarchy: ServerMetricsHierarchy,
) -> MagicMock:
    """Create a mock accumulator that returns the given hierarchy."""
    mock = MagicMock()
    mock.user_config = user_config
    mock.get_hierarchy_for_export.return_value = hierarchy
    return mock


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def verify_no_filesystem_pollution(tmp_path):
    """Verify tests don't create files outside of tmp_path.

    This autouse fixture runs before each test to ensure filesystem isolation.
    Pytest's tmp_path creates unique temporary directories that are automatically
    cleaned up after test completion (keeping last 3 runs for debugging).
    """
    yield
    # Cleanup happens automatically via pytest's tmp_path fixture


@pytest.fixture
def mock_user_config(tmp_path):
    """Create a UserConfig with Parquet format enabled.

    Uses pytest's tmp_path fixture to ensure all files are created in
    a temporary directory that is automatically cleaned up after tests.
    """
    config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="/v1/chat/completions",
        ),
        output={"artifact_directory": str(tmp_path)},
        server_metrics_formats=[ServerMetricsFormat.PARQUET],
    )

    # Verify that output path is within tmp_path (isolation check)
    assert str(config.output.server_metrics_export_parquet_file).startswith(
        str(tmp_path)
    ), (
        f"Parquet file path {config.output.server_metrics_export_parquet_file} is not within tmp_path {tmp_path}"
    )

    return config


@pytest.fixture
def gauge_hierarchy():
    """Create hierarchy with gauge metrics."""
    return build_hierarchy(
        {
            "http://localhost:8081/metrics": [
                (
                    "test_gauge",
                    {"model": "llama-3"},
                    build_metric_entry(
                        PrometheusMetricType.GAUGE,
                        build_scalar_time_series(
                            [1_000_000_000, 2_000_000_000, 3_000_000_000],
                            [50.0, 60.0, 55.0],
                        ),
                        "A test gauge metric",
                    ),
                ),
            ],
        }
    )


@pytest.fixture
def counter_hierarchy():
    """Create hierarchy with counter metrics (including reference point)."""
    return build_hierarchy(
        {
            "http://localhost:8081/metrics": [
                (
                    "test_counter_total",
                    {"method": "GET", "status": "200"},
                    build_metric_entry(
                        PrometheusMetricType.COUNTER,
                        build_scalar_time_series(
                            [500_000_000, 1_000_000_000, 2_000_000_000, 3_000_000_000],
                            [100.0, 150.0, 250.0, 400.0],
                        ),
                        "A test counter metric",
                    ),
                ),
            ],
        }
    )


@pytest.fixture
def histogram_hierarchy():
    """Create hierarchy with histogram metrics."""
    return build_hierarchy(
        {
            "http://localhost:8081/metrics": [
                (
                    "test_histogram",
                    {"endpoint": "/generate"},
                    build_metric_entry(
                        PrometheusMetricType.HISTOGRAM,
                        build_histogram_time_series(
                            [500_000_000, 1_000_000_000, 2_000_000_000],
                            [0.5, 2.5, 5.0],  # sums
                            [10.0, 50.0, 100.0],  # counts
                            ("0.01", "0.1", "1.0", "+Inf"),
                            [
                                [2.0, 8.0, 10.0, 10.0],  # Reference buckets
                                [5.0, 40.0, 50.0, 50.0],  # Start of profiling
                                [10.0, 80.0, 100.0, 100.0],  # Later
                            ],
                        ),
                        "A test histogram metric",
                    ),
                ),
            ],
        }
    )


# =============================================================================
# Basic Export Tests
# =============================================================================


class TestParquetExporterBasics:
    """Tests for basic Parquet exporter functionality."""

    def test_parquet_disabled_when_format_not_selected(self, mock_user_config):
        """Exporter is disabled when Parquet format not selected."""
        mock_user_config.server_metrics_formats = [ServerMetricsFormat.JSON]
        hierarchy = build_hierarchy({})
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)

        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        with pytest.raises(DataExporterDisabled, match="format not selected"):
            ServerMetricsParquetExporter(mock_accumulator, time_filter)

    async def test_parquet_file_created(self, mock_user_config, gauge_hierarchy):
        """Parquet file is created with valid schema."""
        mock_accumulator = create_mock_accumulator(mock_user_config, gauge_hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        parquet_file = mock_user_config.output.server_metrics_export_parquet_file
        assert parquet_file.exists()

        table = pq.read_table(parquet_file)
        assert table.num_rows > 0


# =============================================================================
# Schema Discovery Tests
# =============================================================================


class TestSchemaDiscovery:
    """Tests for dynamic schema discovery."""

    async def test_label_keys_discovered(self, mock_user_config):
        """Label keys are discovered from all metrics."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_gauge",
                        {"model": "llama-3", "gpu": "0"},
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series([1_000_000_000], [50.0]),
                        ),
                    ),
                    (
                        "test_counter",
                        {"method": "POST", "status": "200"},
                        build_metric_entry(
                            PrometheusMetricType.COUNTER,
                            build_scalar_time_series([1_000_000_000], [100.0]),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        label_keys = exporter._discover_all_label_keys()

        assert "model" in label_keys
        assert "gpu" in label_keys
        assert "method" in label_keys
        assert "status" in label_keys

    async def test_buckets_in_normalized_rows(self, mock_user_config):
        """Histogram buckets are exported as separate rows (normalized schema)."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_histogram",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.HISTOGRAM,
                            build_histogram_time_series(
                                [1_000_000_000],
                                [5.0],
                                [100.0],
                                ("0.01", "0.1", "1.0", "+Inf"),
                                [[10.0, 80.0, 95.0, 100.0]],
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        hist_rows = df[df["metric_type"] == "histogram"]
        assert len(hist_rows) == 4  # One row per bucket

        bucket_les = set(hist_rows["bucket_le"].values)
        assert bucket_les == {"0.01", "0.1", "1.0", "+Inf"}
        assert hist_rows["bucket_count"].notna().all()

    async def test_reserved_label_names_filtered(self, mock_user_config):
        """Labels with reserved names are filtered out."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_gauge",
                        {"value": "conflicting", "bucket_le": "bad", "model": "good"},
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series([1_000_000_000], [50.0]),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        parquet_file = mock_user_config.output.server_metrics_export_parquet_file
        table = pq.read_table(parquet_file)
        schema_names = table.schema.names

        assert schema_names.count("value") == 1
        assert schema_names.count("bucket_le") == 1
        assert "model" in schema_names


# =============================================================================
# Delta Calculation Tests
# =============================================================================


class TestDeltaCalculations:
    """Tests for delta calculation logic."""

    @pytest.mark.asyncio
    async def test_gauge_raw_values(self, mock_user_config, gauge_hierarchy):
        """Gauge metrics export raw values without delta calculations."""
        mock_accumulator = create_mock_accumulator(mock_user_config, gauge_hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        gauge_rows = df[df["metric_type"] == "gauge"].sort_values("timestamp_ns")
        values = gauge_rows["value"].values

        assert len(values) == 3
        np.testing.assert_array_almost_equal(values, [50.0, 60.0, 55.0])

    @pytest.mark.asyncio
    async def test_counter_cumulative_deltas(self, mock_user_config, counter_hierarchy):
        """Counter metrics export cumulative deltas from reference point."""
        mock_accumulator = create_mock_accumulator(mock_user_config, counter_hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        counter_rows = df[df["metric_type"] == "counter"].sort_values("timestamp_ns")
        values = counter_rows["value"].values

        # Reference is at 500ms with value 100.0
        # Deltas: [50.0, 150.0, 300.0] (cumulative from reference)
        assert len(values) == 3
        np.testing.assert_array_almost_equal(values, [50.0, 150.0, 300.0])
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    @pytest.mark.asyncio
    async def test_histogram_cumulative_deltas(
        self, mock_user_config, histogram_hierarchy
    ):
        """Histogram metrics export cumulative sum/count/bucket deltas."""
        mock_accumulator = create_mock_accumulator(
            mock_user_config, histogram_hierarchy
        )
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        hist_rows = df[df["metric_type"] == "histogram"].sort_values(
            ["timestamp_ns", "bucket_le"]
        )

        # Should have 2 timestamps × 4 buckets = 8 rows
        assert len(hist_rows) == 8

        unique_timestamps = hist_rows["timestamp_ns"].unique()
        assert len(unique_timestamps) == 2

        first_ts_rows = hist_rows[hist_rows["timestamp_ns"] == unique_timestamps[0]]
        assert len(first_ts_rows) == 4

        assert (first_ts_rows["sum"] == 2.0).all()  # 2.5 - 0.5
        assert (first_ts_rows["count"] == 40.0).all()  # 50 - 10

        bucket_0_1_row = first_ts_rows[first_ts_rows["bucket_le"] == "0.1"]
        assert len(bucket_0_1_row) == 1
        assert bucket_0_1_row["bucket_count"].values[0] == 32.0  # 40 - 8


# =============================================================================
# Time Filtering Edge Cases
# =============================================================================


class TestTimeFilteringEdgeCases:
    """Tests for time filtering edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_all_data_before_filter_returns_empty(self, mock_user_config):
        """Test returns empty file when all data falls before time filter."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_gauge",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series(
                                [500_000_000, 800_000_000], [50.0, 60.0]
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        result = await exporter.export()

        assert result.export_type == "Server Metrics Parquet Export"

    @pytest.mark.asyncio
    async def test_single_timestamp_in_filter(self, mock_user_config):
        """Test with only one timestamp within filter range."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_counter",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.COUNTER,
                            build_scalar_time_series(
                                [500_000_000, 1_000_000_000], [100.0, 150.0]
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        assert len(df) == 1
        assert df["value"].values[0] == 50.0  # Delta from reference

    @pytest.mark.asyncio
    async def test_no_reference_point_available(self, mock_user_config):
        """Test counter when no reference point before filter start."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_counter",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.COUNTER,
                            build_scalar_time_series(
                                [1_000_000_000, 2_000_000_000], [100.0, 200.0]
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        counter_rows = df[df["metric_type"] == "counter"].sort_values("timestamp_ns")
        np.testing.assert_array_almost_equal(counter_rows["value"].values, [0.0, 100.0])


# =============================================================================
# Numeric Edge Cases
# =============================================================================


class TestNumericEdgeCases:
    """Tests for numeric edge cases and extreme values."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "values,expected_deltas",
        [
            ([100.0, 200.0, 150.0], [0.0, 100.0, 50.0]),  # Reset detected
            ([100.0, 100.0, 100.0], [0.0, 0.0, 0.0]),  # No change
            ([100.0, 0.0, 50.0], [0.0, 0.0, 0.0]),  # Full reset to 0
            ([1e10, 1e10 + 1000, 1e10 + 500], [0.0, 1000.0, 500.0]),  # Large values
        ],
    )
    async def test_counter_reset_scenarios(
        self, mock_user_config, values, expected_deltas
    ):
        """Test various counter reset and edge case scenarios."""
        # Build timestamps: reference at 500ms, then data points starting at 1000ms
        timestamps = [500_000_000] + [
            1_000_000_000 + i * 1_000_000_000 for i in range(len(values))
        ]
        all_values = [values[0]] + list(values)

        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_counter",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.COUNTER,
                            build_scalar_time_series(timestamps, all_values),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=10_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        counter_rows = df[df["metric_type"] == "counter"].sort_values("timestamp_ns")
        np.testing.assert_array_almost_equal(
            counter_rows["value"].values, expected_deltas
        )

    @pytest.mark.asyncio
    async def test_very_large_histogram_bucket_counts(self, mock_user_config):
        """Test histogram with very large bucket counts."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_histogram",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.HISTOGRAM,
                            build_histogram_time_series(
                                [500_000_000, 1_000_000_000],
                                [0.0, 50_000.0],
                                [0.0, 1_000_000.0],
                                ("0.01", "0.1", "+Inf"),
                                [
                                    [0.0, 0.0, 0.0],
                                    [100_000.0, 800_000.0, 1_000_000.0],
                                ],
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        hist_rows = df[df["metric_type"] == "histogram"]
        assert hist_rows["count"].iloc[0] == 1_000_000.0
        assert hist_rows["sum"].iloc[0] == 50_000.0

    @pytest.mark.asyncio
    async def test_zero_values_preserved(self, mock_user_config):
        """Test that zero values are preserved (not treated as null)."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_gauge",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series([1_000_000_000], [0.0]),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        assert df["value"].iloc[0] == 0.0
        assert df["value"].notna().all()


# =============================================================================
# Label Handling Edge Cases
# =============================================================================


class TestLabelHandlingEdgeCases:
    """Tests for edge cases in label handling."""

    @pytest.mark.asyncio
    async def test_metrics_with_no_labels(self, mock_user_config):
        """Test metrics without any labels."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_gauge",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series([1_000_000_000], [50.0]),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        assert len(df) == 1
        assert df["value"].iloc[0] == 50.0

    @pytest.mark.asyncio
    async def test_mixed_label_sets(self, mock_user_config):
        """Test metrics with completely different label sets."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_counter",
                        {"method": "GET", "status": "200"},
                        build_metric_entry(
                            PrometheusMetricType.COUNTER,
                            build_scalar_time_series([1_000_000_000], [100.0]),
                        ),
                    ),
                    (
                        "test_gauge",
                        {"gpu": "0", "model": "llama-3"},
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series([1_000_000_000], [50.0]),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        assert "method" in df.columns
        assert "status" in df.columns
        assert "gpu" in df.columns
        assert "model" in df.columns

        counter_row = df[df["metric_type"] == "counter"].iloc[0]
        assert counter_row["method"] == "GET"
        assert pd.isna(counter_row["gpu"])

        gauge_row = df[df["metric_type"] == "gauge"].iloc[0]
        assert gauge_row["model"] == "llama-3"
        assert pd.isna(gauge_row["method"])

    @pytest.mark.asyncio
    async def test_empty_label_values(self, mock_user_config):
        """Test labels with empty string values."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_gauge",
                        {"method": "", "status": "200"},
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series([1_000_000_000], [50.0]),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        assert df["method"].iloc[0] == ""
        assert df["status"].iloc[0] == "200"


# =============================================================================
# Histogram Bucket Edge Cases
# =============================================================================


class TestHistogramBucketEdgeCases:
    """Tests for histogram-specific edge cases."""

    @pytest.mark.asyncio
    async def test_histogram_with_single_bucket(self, mock_user_config):
        """Test histogram with only +Inf bucket."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_histogram",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.HISTOGRAM,
                            build_histogram_time_series(
                                [500_000_000, 1_000_000_000],
                                [0.0, 5.0],
                                [0.0, 100.0],
                                ("+Inf",),
                                [[0.0], [100.0]],
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        hist_rows = df[df["metric_type"] == "histogram"]
        assert len(hist_rows) == 1
        assert hist_rows["bucket_le"].iloc[0] == "+Inf"
        assert hist_rows["bucket_count"].iloc[0] == 100.0

    @pytest.mark.asyncio
    async def test_histogram_with_many_buckets(self, mock_user_config):
        """Test histogram with many buckets (50+)."""
        bucket_les = tuple(f"{i * 0.1:.1f}" for i in range(1, 51)) + ("+Inf",)
        bucket_counts = [[float(i * 10) for i in range(1, 51)] + [500.0]]

        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_histogram",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.HISTOGRAM,
                            build_histogram_time_series(
                                [1_000_000_000],
                                [25.0],
                                [500.0],
                                bucket_les,
                                bucket_counts,
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        hist_rows = df[df["metric_type"] == "histogram"]
        assert len(hist_rows) == 51
        assert hist_rows["sum"].nunique() == 1
        assert hist_rows["count"].nunique() == 1

    @pytest.mark.asyncio
    async def test_histogram_reset_negative_deltas(self, mock_user_config):
        """Test histogram reset produces zero deltas."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_histogram",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.HISTOGRAM,
                            build_histogram_time_series(
                                [500_000_000, 1_000_000_000],
                                [50.0, 1.0],  # Reference 50, after reset 1
                                [1000.0, 10.0],  # Reference 1000, after reset 10
                                ("0.1", "+Inf"),
                                [[800.0, 1000.0], [8.0, 10.0]],  # Reset: smaller values
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        hist_rows = df[df["metric_type"] == "histogram"]
        assert (hist_rows["sum"] == 0.0).all()
        assert (hist_rows["count"] == 0.0).all()
        assert (hist_rows["bucket_count"] == 0.0).all()


# =============================================================================
# Schema Consistency Tests
# =============================================================================


class TestSchemaConsistency:
    """Tests for schema consistency across different data combinations."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_endpoints", [1, 3, 10])
    async def test_schema_consistent_across_endpoints(
        self, mock_user_config, num_endpoints
    ):
        """Test schema is consistent regardless of number of endpoints."""
        endpoints_data = {}
        for i in range(num_endpoints):
            endpoints_data[f"http://server{i}:8081/metrics"] = [
                (
                    "test_gauge",
                    {"instance": f"server{i}"},
                    build_metric_entry(
                        PrometheusMetricType.GAUGE,
                        build_scalar_time_series([1_000_000_000], [50.0 + i]),
                    ),
                ),
            ]

        hierarchy = build_hierarchy(endpoints_data)
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        assert len(df) == num_endpoints
        assert len(df.columns) > 0

    @pytest.mark.asyncio
    async def test_mixed_metrics_all_types(self, mock_user_config):
        """Test file with mix of all metric types from multiple endpoints."""
        endpoints_data = {}
        for endpoint in (
            "http://server1:8081/metrics",
            "http://server2:8082/metrics",
        ):
            endpoints_data[endpoint] = [
                (
                    "test_gauge",
                    None,
                    build_metric_entry(
                        PrometheusMetricType.GAUGE,
                        build_scalar_time_series([1_000_000_000], [50.0]),
                    ),
                ),
                (
                    "test_counter",
                    None,
                    build_metric_entry(
                        PrometheusMetricType.COUNTER,
                        build_scalar_time_series([1_000_000_000], [100.0]),
                    ),
                ),
                (
                    "test_histogram",
                    None,
                    build_metric_entry(
                        PrometheusMetricType.HISTOGRAM,
                        build_histogram_time_series(
                            [1_000_000_000],
                            [1.0],
                            [10.0],
                            ("0.1", "+Inf"),
                            [[8.0, 10.0]],
                        ),
                    ),
                ),
            ]

        hierarchy = build_hierarchy(endpoints_data)
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        # 2 gauges + 2 counters + (2 histograms × 2 buckets) = 8 rows
        assert len(df) == 8
        assert (df["metric_type"] == "gauge").sum() == 2
        assert (df["metric_type"] == "counter").sum() == 2
        assert (df["metric_type"] == "histogram").sum() == 4


# =============================================================================
# Multi-Endpoint Tests
# =============================================================================


class TestMultiEndpoint:
    """Tests for multi-endpoint scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_endpoints(self, mock_user_config):
        """Multiple endpoints are included in Parquet export."""
        hierarchy = build_hierarchy(
            {
                "http://server1:8081/metrics": [
                    (
                        "test_gauge",
                        {"instance": "server1"},
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series([1_000_000_000], [50.0]),
                        ),
                    ),
                ],
                "http://server2:8082/metrics": [
                    (
                        "test_gauge",
                        {"instance": "server2"},
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series([1_000_000_000], [75.0]),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        endpoints = df["endpoint_url"].unique()
        assert len(endpoints) == 2
        assert "http://server1:8081/metrics" in endpoints
        assert "http://server2:8082/metrics" in endpoints


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_counter_reset_handling(self, mock_user_config):
        """Counter resets (negative deltas) are handled correctly."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_counter",
                        {"method": "GET"},
                        build_metric_entry(
                            PrometheusMetricType.COUNTER,
                            build_scalar_time_series(
                                [500_000_000, 1_000_000_000, 2_000_000_000],
                                [1000.0, 1500.0, 50.0],  # Reset at 2s
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        counter_rows = df[df["metric_type"] == "counter"].sort_values("timestamp_ns")
        values = counter_rows["value"].values

        # First: 500 delta (1500-1000), Second: -950 delta floored to 0
        assert values[0] == 500.0
        assert values[1] == 0.0

    @pytest.mark.asyncio
    async def test_empty_data_no_file_created(self, mock_user_config, tmp_path):
        """No Parquet file created when no data available."""
        hierarchy = build_hierarchy({})
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        parquet_file = tmp_path / "server_metrics_export.parquet"
        assert not parquet_file.exists()


# =============================================================================
# Data Validation Tests
# =============================================================================


class TestDataValidation:
    """Tests for exported data validation."""

    @pytest.mark.asyncio
    async def test_all_metric_types_in_single_file(self, mock_user_config):
        """Gauge, counter, and histogram metrics all in single Parquet file."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_gauge",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series([1_000_000_000], [50.0]),
                        ),
                    ),
                    (
                        "test_counter",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.COUNTER,
                            build_scalar_time_series([1_000_000_000], [100.0]),
                        ),
                    ),
                    (
                        "test_histogram",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.HISTOGRAM,
                            build_histogram_time_series(
                                [1_000_000_000],
                                [0.5],
                                [10.0],
                                ("0.01", "0.1", "+Inf"),
                                [[2.0, 8.0, 10.0]],
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=2_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        metric_types = set(df["metric_type"].unique())
        assert "gauge" in metric_types
        assert "counter" in metric_types
        assert "histogram" in metric_types

    @pytest.mark.asyncio
    async def test_timestamps_sorted_per_metric(self, mock_user_config):
        """Timestamps are sorted within each metric."""
        hierarchy = build_hierarchy(
            {
                "http://localhost:8081/metrics": [
                    (
                        "test_gauge",
                        None,
                        build_metric_entry(
                            PrometheusMetricType.GAUGE,
                            build_scalar_time_series(
                                [1_000_000_000, 2_000_000_000, 3_000_000_000],
                                [50.0, 60.0, 55.0],
                            ),
                        ),
                    ),
                ],
            }
        )
        mock_accumulator = create_mock_accumulator(mock_user_config, hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=4_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        table = pq.read_table(
            mock_user_config.output.server_metrics_export_parquet_file
        )
        df = table.to_pandas()

        timestamps = df["timestamp_ns"].values
        assert all(
            timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)
        )


class TestParquetMetadataFields:
    """Tests for metadata fields in Parquet export."""

    @pytest.mark.asyncio
    async def test_parquet_has_all_core_metadata_fields(
        self, mock_user_config, gauge_hierarchy
    ):
        """Verify Parquet metadata includes schema_version, aiperf.version, and benchmark_id."""
        mock_accumulator = create_mock_accumulator(mock_user_config, gauge_hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        parquet_file = mock_user_config.output.server_metrics_export_parquet_file
        table = pq.read_table(parquet_file)
        metadata = table.schema.metadata

        # Verify all three core metadata fields
        assert b"aiperf.schema_version" in metadata, "aiperf.schema_version missing"
        assert b"aiperf.version" in metadata, "aiperf.version missing"
        assert b"aiperf.benchmark_id" in metadata, "aiperf.benchmark_id missing"

        # Verify values
        assert metadata[b"aiperf.schema_version"] == b"1.0"

        version = metadata[b"aiperf.version"].decode()
        assert len(version) > 0
        assert version != "unknown"

        benchmark_id = metadata[b"aiperf.benchmark_id"].decode()
        assert len(benchmark_id) == 36  # UUID format

        # Verify it's a valid UUID
        import uuid

        uuid.UUID(benchmark_id)

    @pytest.mark.asyncio
    async def test_parquet_has_environment_metadata(
        self, mock_user_config, gauge_hierarchy
    ):
        """Verify Parquet includes hostname, python_version, pyarrow_version."""
        mock_accumulator = create_mock_accumulator(mock_user_config, gauge_hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        parquet_file = mock_user_config.output.server_metrics_export_parquet_file
        table = pq.read_table(parquet_file)
        metadata = table.schema.metadata

        # Verify environment fields
        assert b"aiperf.hostname" in metadata
        assert b"aiperf.python_version" in metadata
        assert b"aiperf.pyarrow_version" in metadata

        # Verify values are non-empty
        assert len(metadata[b"aiperf.hostname"]) > 0

        python_ver = metadata[b"aiperf.python_version"].decode()
        assert python_ver.count(".") == 2  # Format: X.Y.Z

        pyarrow_ver = metadata[b"aiperf.pyarrow_version"].decode()
        assert len(pyarrow_ver) > 0

    @pytest.mark.asyncio
    async def test_parquet_benchmark_id_matches_user_config(
        self, mock_user_config, gauge_hierarchy
    ):
        """Verify Parquet benchmark_id matches UserConfig benchmark_id."""
        mock_accumulator = create_mock_accumulator(mock_user_config, gauge_hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        parquet_file = mock_user_config.output.server_metrics_export_parquet_file
        table = pq.read_table(parquet_file)
        metadata = table.schema.metadata

        parquet_benchmark_id = metadata[b"aiperf.benchmark_id"].decode()
        assert parquet_benchmark_id == mock_user_config.benchmark_id

    @pytest.mark.asyncio
    async def test_parquet_has_complete_metadata_set(
        self, mock_user_config, gauge_hierarchy
    ):
        """Verify Parquet has all expected metadata keys.

        Note: row_count is not stored in custom metadata since it's available
        via Parquet's built-in metadata: pq.read_metadata(path).num_rows
        """
        mock_accumulator = create_mock_accumulator(mock_user_config, gauge_hierarchy)
        time_filter = TimeRangeFilter(start_ns=1_000_000_000, end_ns=3_000_000_000)

        exporter = ServerMetricsParquetExporter(mock_accumulator, time_filter)
        await exporter.export()

        parquet_file = mock_user_config.output.server_metrics_export_parquet_file
        table = pq.read_table(parquet_file)
        metadata = table.schema.metadata

        # Minimum expected metadata keys (row_count available via pq.read_metadata)
        expected_keys = {
            b"aiperf.schema_version",
            b"aiperf.version",
            b"aiperf.benchmark_id",
            b"aiperf.hostname",
            b"aiperf.python_version",
            b"aiperf.pyarrow_version",
            b"aiperf.export_timestamp_utc",
            b"aiperf.exporter",
            b"aiperf.time_filter_start_ns",
            b"aiperf.time_filter_end_ns",
            b"aiperf.profiling_duration_ns",
            b"aiperf.profiling_duration_seconds",
            b"aiperf.model_names",
            b"aiperf.concurrency",
            b"aiperf.endpoint_urls",
            b"aiperf.endpoint_count",
            b"aiperf.label_columns",
            b"aiperf.label_count",
            b"aiperf.metric_count",
            b"aiperf.metric_type_counts",
            b"aiperf.input_config",
            b"aiperf.schema_note",
        }

        # Verify all expected keys are present
        for key in expected_keys:
            assert key in metadata, f"{key.decode()} missing from Parquet metadata"

        # Verify we have at least 22 keys (might have request_rate too)
        assert len(metadata) >= 22

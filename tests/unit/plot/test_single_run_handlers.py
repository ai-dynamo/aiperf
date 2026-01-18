# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for single-run plot handlers.

Tests for handler classes in aiperf.plot.handlers.single_run_handlers module.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import DataSource, MetricSpec, PlotSpec, PlotType
from aiperf.plot.exceptions import PlotGenerationError
from aiperf.plot.handlers.single_run_handlers import (
    AreaHandler,
    BaseSingleRunHandler,
    DualAxisHandler,
    HistogramHandler,
    RequestTimelineHandler,
    ScatterHandler,
    ScatterWithPercentilesHandler,
    TimeSliceHandler,
    _is_single_stat_metric,
)


class TestIsSingleStatMetric:
    """Tests for _is_single_stat_metric function."""

    def test_returns_true_for_metric_with_only_avg(self):
        """Test that metric with only avg stat returns True."""
        metric = MagicMock(spec=["avg", "unit"])
        metric.avg = 100.0
        metric.unit = "ms"
        assert _is_single_stat_metric(metric) is True

    def test_returns_false_for_metric_with_p50(self):
        """Test that metric with p50 stat returns False."""
        metric = MagicMock()
        metric.avg = 100.0
        metric.p50 = 95.0
        assert _is_single_stat_metric(metric) is False

    def test_returns_false_for_metric_with_std(self):
        """Test that metric with std stat returns False."""
        metric = MagicMock()
        metric.avg = 100.0
        metric.std = 10.0
        assert _is_single_stat_metric(metric) is False

    def test_returns_false_for_metric_with_min_max(self):
        """Test that metric with min/max stats returns False."""
        metric = MagicMock()
        metric.avg = 100.0
        metric.min = 50.0
        metric.max = 150.0
        assert _is_single_stat_metric(metric) is False

    def test_handles_dict_metric_with_only_avg(self):
        """Test that dict metric with only avg returns True."""
        metric = {"avg": 100.0, "unit": "ms"}
        assert _is_single_stat_metric(metric) is True

    def test_handles_dict_metric_with_distribution_stats(self):
        """Test that dict metric with distribution stats returns False."""
        metric = {"avg": 100.0, "p50": 95.0, "p99": 150.0}
        assert _is_single_stat_metric(metric) is False

    def test_returns_true_when_distribution_stats_are_none(self):
        """Test that metric with None distribution stats returns True."""
        # Use spec to limit attributes - MagicMock with spec only has specified attrs
        metric = MagicMock(spec=["avg", "unit"])
        metric.avg = 100.0
        metric.unit = "ms"
        # Verify the mock doesn't have distribution stats
        assert not hasattr(metric, "p50")
        assert _is_single_stat_metric(metric) is True


class TestBaseSingleRunHandler:
    """Tests for BaseSingleRunHandler base class."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance with mocked PlotGenerator."""
        mock_generator = MagicMock(spec=PlotGenerator)
        return BaseSingleRunHandler(mock_generator)

    @pytest.fixture
    def available_metrics(self):
        """Sample available metrics dictionary."""
        return {
            "time_to_first_token": {
                "display_name": "Time to First Token",
                "unit": "ms",
            },
            "request_latency": {"display_name": "Request Latency", "unit": "ms"},
            "output_token_throughput": {
                "display_name": "Output Token Throughput",
                "unit": "tokens/s",
            },
        }

    def test_get_axis_label_for_request_number(self, handler, available_metrics):
        """Test axis label for request_number metric."""
        metric_spec = MetricSpec(
            name="request_number", axis="x", source=DataSource.REQUESTS
        )
        result = handler._get_axis_label(metric_spec, available_metrics)
        assert result == "Request Number"

    def test_get_axis_label_for_timestamp(self, handler, available_metrics):
        """Test axis label for timestamp metric."""
        metric_spec = MetricSpec(name="timestamp", axis="x", source=DataSource.REQUESTS)
        result = handler._get_axis_label(metric_spec, available_metrics)
        assert result == "Time (seconds)"

    def test_get_axis_label_for_timestamp_s(self, handler, available_metrics):
        """Test axis label for timestamp_s metric."""
        metric_spec = MetricSpec(
            name="timestamp_s", axis="x", source=DataSource.REQUESTS
        )
        result = handler._get_axis_label(metric_spec, available_metrics)
        assert result == "Time (s)"

    def test_get_axis_label_for_timeslice(self, handler, available_metrics):
        """Test axis label for Timeslice metric."""
        metric_spec = MetricSpec(
            name="Timeslice", axis="x", source=DataSource.TIMESLICES
        )
        result = handler._get_axis_label(metric_spec, available_metrics)
        assert result == "Timeslice (s)"

    def test_get_axis_label_for_regular_metric(self, handler, available_metrics):
        """Test axis label for regular metric uses _get_metric_label."""
        metric_spec = MetricSpec(
            name="time_to_first_token", axis="y", stat="p50", source=DataSource.REQUESTS
        )
        result = handler._get_axis_label(metric_spec, available_metrics)
        assert "Time to First Token" in result
        assert "ms" in result

    def test_get_metric_label_with_unit(self, handler, available_metrics):
        """Test metric label includes unit."""
        result = handler._get_metric_label(
            "time_to_first_token", "avg", available_metrics
        )
        assert result == "Time to First Token (ms)"

    def test_get_metric_label_with_stat(self, handler, available_metrics):
        """Test metric label includes stat when not avg or value."""
        result = handler._get_metric_label(
            "time_to_first_token", "p99", available_metrics
        )
        assert "p99" in result
        assert "Time to First Token" in result

    def test_get_metric_label_without_stat_in_label_for_avg(
        self, handler, available_metrics
    ):
        """Test metric label excludes stat for avg."""
        result = handler._get_metric_label(
            "time_to_first_token", "avg", available_metrics
        )
        assert "avg" not in result.lower()

    def test_get_metric_label_for_unknown_metric(self, handler):
        """Test metric label returns title-cased metric name for unknown metric."""
        result = handler._get_metric_label("unknown_metric", "p50", {})
        assert result == "Unknown Metric (p50)"

    @patch("aiperf.plot.handlers.single_run_handlers.prepare_request_timeseries")
    def test_prepare_data_for_requests_source(self, mock_prepare, handler):
        """Test data preparation for REQUESTS source."""
        mock_run = MagicMock()
        mock_df = pd.DataFrame({"col": [1, 2, 3]})
        mock_prepare.return_value = mock_df

        result = handler._prepare_data_for_source(DataSource.REQUESTS, mock_run)

        mock_prepare.assert_called_once_with(mock_run)
        assert result is mock_df

    def test_prepare_data_for_timeslices_source(self, handler):
        """Test data preparation for TIMESLICES source."""
        mock_run = MagicMock()
        mock_df = pd.DataFrame({"timeslice": [1, 2, 3]})
        mock_run.timeslices = mock_df

        result = handler._prepare_data_for_source(DataSource.TIMESLICES, mock_run)

        assert result is mock_df

    def test_prepare_data_for_gpu_telemetry_source(self, handler):
        """Test data preparation for GPU_TELEMETRY source."""
        mock_run = MagicMock()
        mock_df = pd.DataFrame({"gpu_util": [80, 85, 90]})
        mock_run.gpu_telemetry = mock_df

        result = handler._prepare_data_for_source(DataSource.GPU_TELEMETRY, mock_run)

        assert result is mock_df

    def test_prepare_data_raises_for_unsupported_source(self, handler):
        """Test that unsupported source raises PlotGenerationError."""
        mock_run = MagicMock()

        with pytest.raises(PlotGenerationError):
            handler._prepare_data_for_source(DataSource.AGGREGATED, mock_run)


class TestScatterHandler:
    """Tests for ScatterHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a ScatterHandler instance."""
        mock_generator = MagicMock(spec=PlotGenerator)
        return ScatterHandler(mock_generator)

    @pytest.fixture
    def sample_spec(self):
        """Create a sample PlotSpec for scatter plots."""
        return PlotSpec(
            name="scatter-test",
            title="Test Scatter",
            plot_type=PlotType.SCATTER,
            metrics=[
                MetricSpec(name="request_number", axis="x", source=DataSource.REQUESTS),
                MetricSpec(name="latency", axis="y", source=DataSource.REQUESTS),
            ],
        )

    def test_can_handle_returns_true_with_valid_requests(self, handler, sample_spec):
        """Test can_handle returns True when requests data is available."""
        mock_run = MagicMock()
        mock_run.requests = pd.DataFrame(
            {"request_number": [1, 2], "latency": [100, 110]}
        )

        result = handler.can_handle(sample_spec, mock_run)
        assert result is True

    def test_can_handle_returns_false_with_none_requests(self, handler, sample_spec):
        """Test can_handle returns False when requests is None."""
        mock_run = MagicMock()
        mock_run.requests = None

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False

    def test_can_handle_returns_false_with_empty_requests(self, handler, sample_spec):
        """Test can_handle returns False when requests DataFrame is empty."""
        mock_run = MagicMock()
        mock_run.requests = pd.DataFrame()

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False


class TestAreaHandler:
    """Tests for AreaHandler class."""

    @pytest.fixture
    def handler(self):
        """Create an AreaHandler instance."""
        mock_generator = MagicMock(spec=PlotGenerator)
        return AreaHandler(mock_generator)

    @pytest.fixture
    def sample_spec(self):
        """Create a sample PlotSpec for area plots."""
        return PlotSpec(
            name="area-test",
            title="Test Area",
            plot_type=PlotType.AREA,
            metrics=[
                MetricSpec(name="timestamp_s", axis="x", source=DataSource.REQUESTS),
                MetricSpec(name="throughput", axis="y", source=DataSource.REQUESTS),
            ],
        )

    def test_can_handle_returns_true_with_valid_requests(self, handler, sample_spec):
        """Test can_handle returns True when requests data is available."""
        mock_run = MagicMock()
        mock_run.requests = pd.DataFrame(
            {"timestamp_s": [1, 2], "throughput": [100, 110]}
        )

        result = handler.can_handle(sample_spec, mock_run)
        assert result is True

    def test_can_handle_returns_false_with_none_requests(self, handler, sample_spec):
        """Test can_handle returns False when requests is None."""
        mock_run = MagicMock()
        mock_run.requests = None

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False


class TestTimeSliceHandler:
    """Tests for TimeSliceHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a TimeSliceHandler instance."""
        mock_generator = MagicMock(spec=PlotGenerator)
        return TimeSliceHandler(mock_generator)

    @pytest.fixture
    def sample_spec(self):
        """Create a sample PlotSpec for timeslice plots."""
        return PlotSpec(
            name="timeslice-test",
            title="Test Timeslice",
            plot_type=PlotType.TIMESLICE,
            metrics=[
                MetricSpec(name="Timeslice", axis="x", source=DataSource.TIMESLICES),
                MetricSpec(
                    name="latency", axis="y", stat="avg", source=DataSource.TIMESLICES
                ),
            ],
        )

    def test_can_handle_returns_true_with_valid_timeslices(self, handler, sample_spec):
        """Test can_handle returns True when timeslices data is available."""
        mock_run = MagicMock()
        mock_run.timeslices = pd.DataFrame(
            {"Timeslice": [1, 2], "latency_avg": [100, 110]}
        )

        result = handler.can_handle(sample_spec, mock_run)
        assert result is True

    def test_can_handle_returns_false_with_none_timeslices(self, handler, sample_spec):
        """Test can_handle returns False when timeslices is None."""
        mock_run = MagicMock()
        mock_run.timeslices = None

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False

    def test_can_handle_returns_false_with_empty_timeslices(self, handler, sample_spec):
        """Test can_handle returns False when timeslices DataFrame is empty."""
        mock_run = MagicMock()
        mock_run.timeslices = pd.DataFrame()

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False


class TestHistogramHandler:
    """Tests for HistogramHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a HistogramHandler instance."""
        mock_generator = MagicMock(spec=PlotGenerator)
        return HistogramHandler(mock_generator)

    @pytest.fixture
    def sample_spec(self):
        """Create a sample PlotSpec for histogram plots."""
        return PlotSpec(
            name="histogram-test",
            title="Test Histogram",
            plot_type=PlotType.HISTOGRAM,
            metrics=[
                MetricSpec(name="Timeslice", axis="x", source=DataSource.TIMESLICES),
                MetricSpec(
                    name="count", axis="y", stat="avg", source=DataSource.TIMESLICES
                ),
            ],
        )

    def test_can_handle_returns_true_with_valid_timeslices(self, handler, sample_spec):
        """Test can_handle returns True when timeslices data is available."""
        mock_run = MagicMock()
        mock_run.timeslices = pd.DataFrame({"Timeslice": [1, 2], "count": [10, 15]})

        result = handler.can_handle(sample_spec, mock_run)
        assert result is True

    def test_can_handle_returns_false_with_none_timeslices(self, handler, sample_spec):
        """Test can_handle returns False when timeslices is None."""
        mock_run = MagicMock()
        mock_run.timeslices = None

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False


class TestDualAxisHandler:
    """Tests for DualAxisHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a DualAxisHandler instance."""
        mock_generator = MagicMock(spec=PlotGenerator)
        return DualAxisHandler(mock_generator)

    @pytest.fixture
    def sample_spec(self):
        """Create a sample PlotSpec for dual-axis plots."""
        spec = MagicMock()
        spec.name = "dual-axis-test"
        spec.title = "Test Dual Axis"
        spec.plot_type = PlotType.DUAL_AXIS
        spec.metrics = [
            MetricSpec(name="timestamp_s", axis="x", source=DataSource.REQUESTS),
            MetricSpec(
                name="throughput_tokens_per_sec", axis="y", source=DataSource.REQUESTS
            ),
            MetricSpec(
                name="gpu_utilization", axis="y2", source=DataSource.GPU_TELEMETRY
            ),
        ]
        spec.primary_style = "area"
        spec.secondary_style = "line"
        spec.supplementary_col = None
        return spec

    def test_can_handle_returns_true_with_valid_gpu_telemetry(
        self, handler, sample_spec
    ):
        """Test can_handle returns True when GPU telemetry is available."""
        mock_run = MagicMock()
        mock_run.gpu_telemetry = pd.DataFrame(
            {"timestamp_s": [1, 2], "gpu_utilization": [80, 85]}
        )

        result = handler.can_handle(sample_spec, mock_run)
        assert result is True

    def test_can_handle_returns_false_with_none_gpu_telemetry(
        self, handler, sample_spec
    ):
        """Test can_handle returns False when GPU telemetry is None."""
        mock_run = MagicMock()
        mock_run.gpu_telemetry = None

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False

    def test_can_handle_returns_false_with_empty_gpu_telemetry(
        self, handler, sample_spec
    ):
        """Test can_handle returns False when GPU telemetry is empty."""
        mock_run = MagicMock()
        mock_run.gpu_telemetry = pd.DataFrame()

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False


class TestScatterWithPercentilesHandler:
    """Tests for ScatterWithPercentilesHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a ScatterWithPercentilesHandler instance."""
        mock_generator = MagicMock(spec=PlotGenerator)
        return ScatterWithPercentilesHandler(mock_generator)

    @pytest.fixture
    def sample_spec(self):
        """Create a sample PlotSpec for scatter with percentiles plots."""
        return PlotSpec(
            name="scatter-percentiles-test",
            title="Test Scatter with Percentiles",
            plot_type=PlotType.SCATTER_WITH_PERCENTILES,
            metrics=[
                MetricSpec(name="request_number", axis="x", source=DataSource.REQUESTS),
                MetricSpec(name="latency", axis="y", source=DataSource.REQUESTS),
            ],
        )

    def test_can_handle_returns_true_with_valid_requests(self, handler, sample_spec):
        """Test can_handle returns True when requests data is available."""
        mock_run = MagicMock()
        mock_run.requests = pd.DataFrame(
            {"request_number": [1, 2], "latency": [100, 110]}
        )

        result = handler.can_handle(sample_spec, mock_run)
        assert result is True

    def test_can_handle_returns_false_with_none_requests(self, handler, sample_spec):
        """Test can_handle returns False when requests is None."""
        mock_run = MagicMock()
        mock_run.requests = None

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False


class TestRequestTimelineHandler:
    """Tests for RequestTimelineHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a RequestTimelineHandler instance."""
        mock_generator = MagicMock(spec=PlotGenerator)
        return RequestTimelineHandler(mock_generator)

    @pytest.fixture
    def sample_spec(self):
        """Create a sample PlotSpec for request timeline plots."""
        return PlotSpec(
            name="timeline-test",
            title="Test Timeline",
            plot_type=PlotType.REQUEST_TIMELINE,
            metrics=[
                MetricSpec(name="latency", axis="y", source=DataSource.REQUESTS),
            ],
        )

    def test_can_handle_returns_true_with_valid_requests(self, handler, sample_spec):
        """Test can_handle returns True when required columns are available."""
        mock_run = MagicMock()
        mock_run.requests = pd.DataFrame(
            {
                "request_start_ns": [1000000000, 2000000000],
                "request_end_ns": [1500000000, 2500000000],
                "time_to_first_token": [50, 60],
                "latency": [100, 110],
            }
        )

        result = handler.can_handle(sample_spec, mock_run)
        assert result is True

    def test_can_handle_returns_false_with_none_requests(self, handler, sample_spec):
        """Test can_handle returns False when requests is None."""
        mock_run = MagicMock()
        mock_run.requests = None

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False

    def test_can_handle_returns_false_with_empty_requests(self, handler, sample_spec):
        """Test can_handle returns False when requests is empty."""
        mock_run = MagicMock()
        mock_run.requests = pd.DataFrame()

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False

    def test_can_handle_returns_false_with_missing_columns(self, handler, sample_spec):
        """Test can_handle returns False when required columns are missing."""
        mock_run = MagicMock()
        mock_run.requests = pd.DataFrame(
            {
                "request_start_ns": [1000000000],
                # Missing request_end_ns and time_to_first_token
            }
        )

        result = handler.can_handle(sample_spec, mock_run)
        assert result is False

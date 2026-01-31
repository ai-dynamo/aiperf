# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.error_models import ErrorDetailsCount
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    ServerMetricsResults,
)
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator
from aiperf.server_metrics.storage import ServerMetricsHierarchy


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
def sample_gauge_metric() -> MetricFamily:
    """Sample gauge metric family."""
    return MetricFamily(
        type=PrometheusMetricType.GAUGE,
        description="KV cache usage percentage",
        samples=[
            MetricSample(
                labels={"model_name": "test-model"},
                value=0.42,
            )
        ],
    )


@pytest.fixture
def sample_counter_metric() -> MetricFamily:
    """Sample counter metric family."""
    return MetricFamily(
        type=PrometheusMetricType.COUNTER,
        description="Total number of requests",
        samples=[
            MetricSample(
                labels={"model_name": "test-model"},
                value=150.0,
            )
        ],
    )


@pytest.fixture
def sample_server_metrics_record(
    sample_gauge_metric: MetricFamily,
    sample_counter_metric: MetricFamily,
) -> ServerMetricsRecord:
    """Create a sample ServerMetricsRecord with typical values."""
    return ServerMetricsRecord(
        endpoint_url="http://node1:8081/metrics",
        timestamp_ns=1_000_000_000,
        endpoint_latency_ns=5_000_000,
        metrics={
            "kv_cache_usage": sample_gauge_metric,
            "requests_total": sample_counter_metric,
        },
    )


@pytest.mark.asyncio
class TestServerMetricsResultsProcessor:
    """Test cases for ServerMetricsResultsProcessor."""

    async def test_initialization(self, mock_user_config: UserConfig) -> None:
        """Test processor initialization sets up hierarchy."""
        processor = ServerMetricsAccumulator(mock_user_config)

        assert isinstance(processor._server_metrics_hierarchy, ServerMetricsHierarchy)

    async def test_process_server_metrics_record(
        self,
        mock_user_config: UserConfig,
        sample_server_metrics_record: ServerMetricsRecord,
    ) -> None:
        """Test processing a server metrics record adds it to the hierarchy."""
        processor = ServerMetricsAccumulator(mock_user_config)

        await processor.process_server_metrics_record(sample_server_metrics_record)

        endpoint_url = sample_server_metrics_record.endpoint_url
        assert endpoint_url in processor._server_metrics_hierarchy.endpoints

    async def test_export_results_no_data(self, mock_user_config: UserConfig) -> None:
        """Test export_results returns None when no data collected."""
        processor = ServerMetricsAccumulator(mock_user_config)

        result = await processor.export_results(
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )

        assert result is None

    async def test_export_results_with_data(
        self,
        mock_user_config: UserConfig,
    ) -> None:
        """Test export_results returns ServerMetricsResults with collected data."""
        processor = ServerMetricsAccumulator(mock_user_config)

        # Add multiple records
        for i in range(5):
            gauge = MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="KV cache usage",
                samples=[MetricSample(labels=None, value=0.4 + i * 0.05)],
            )
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 100_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={"cache_usage": gauge},
            )
            await processor.process_server_metrics_record(record)

        start_ns = 1_000_000_000
        end_ns = 2_000_000_000
        result = await processor.export_results(start_ns=start_ns, end_ns=end_ns)

        assert result is not None
        assert isinstance(result, ServerMetricsResults)
        assert result.start_ns == start_ns
        assert result.end_ns == end_ns
        assert "http://node1:8081/metrics" in result.endpoints_configured
        assert "http://node1:8081/metrics" in result.endpoints_successful
        assert result.endpoint_summaries is not None
        assert len(result.endpoint_summaries) == 1

    async def test_export_results_with_error_summary(
        self,
        mock_user_config: UserConfig,
        sample_server_metrics_record: ServerMetricsRecord,
    ) -> None:
        """Test export_results includes error summary when provided."""
        processor = ServerMetricsAccumulator(mock_user_config)

        await processor.process_server_metrics_record(sample_server_metrics_record)

        from aiperf.common.models import ErrorDetails

        error_summary = [
            ErrorDetailsCount(
                error_details=ErrorDetails(
                    error_type="ConnectionError", message="Failed"
                ),
                count=5,
            )
        ]

        result = await processor.export_results(
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            error_summary=error_summary,
        )

        assert result is not None
        assert result.error_summary == error_summary

    async def test_export_results_with_time_filter(
        self,
        mock_user_config: UserConfig,
    ) -> None:
        """Test export_results includes the provided time filter."""
        processor = ServerMetricsAccumulator(mock_user_config)

        # Add records
        for i in range(5):
            gauge = MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="Cache usage",
                samples=[MetricSample(labels=None, value=0.5)],
            )
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 100_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={"cache_usage": gauge},
            )
            await processor.process_server_metrics_record(record)

        # export_results now constructs per-endpoint TimeFilters internally
        # start_ns and end_ns define the profiling phase bounds
        result = await processor.export_results(
            start_ns=1_000_000_000,  # Profiling start
            end_ns=2_000_000_000,  # Profiling end
        )

        assert result is not None
        # Per-endpoint filters used, not a single global filter
        assert result.aggregation_time_filter is None

    async def test_export_results_multiple_endpoints(
        self,
        mock_user_config: UserConfig,
    ) -> None:
        """Test export_results handles multiple endpoints correctly."""
        processor = ServerMetricsAccumulator(mock_user_config)

        endpoints = ["http://node1:8081/metrics", "http://node2:8081/metrics"]

        for endpoint in endpoints:
            for i in range(3):
                gauge = MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Cache usage",
                    samples=[MetricSample(labels=None, value=0.5)],
                )
                record = ServerMetricsRecord(
                    endpoint_url=endpoint,
                    timestamp_ns=1_000_000_000 + i * 100_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={"cache_usage": gauge},
                )
                await processor.process_server_metrics_record(record)

        result = await processor.export_results(
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )

        assert result is not None
        assert len(result.endpoints_configured) == 2
        assert len(result.endpoints_successful) == 2
        assert result.endpoint_summaries is not None
        assert len(result.endpoint_summaries) == 2

    async def test_export_results_with_labeled_metrics(
        self,
        mock_user_config: UserConfig,
    ) -> None:
        """Test export_results handles metrics with labels correctly."""
        processor = ServerMetricsAccumulator(mock_user_config)

        for i in range(3):
            gauge = MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="Cache usage per model",
                samples=[
                    MetricSample(labels={"model": "model-a"}, value=0.5),
                    MetricSample(labels={"model": "model-b"}, value=0.6),
                ],
            )
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 100_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={"cache_usage": gauge},
            )
            await processor.process_server_metrics_record(record)

        result = await processor.export_results(
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )

        assert result is not None
        assert result.endpoint_summaries is not None
        # Should have summaries for the endpoint
        assert len(result.endpoint_summaries) == 1

    async def test_export_results_computes_endpoint_metadata(
        self,
        mock_user_config: UserConfig,
    ) -> None:
        """Test export_results computes duration, scrape count, and latency correctly."""
        processor = ServerMetricsAccumulator(mock_user_config)

        # Add 5 records with known timing
        scrape_latency_ns = 10_000_000  # 10ms
        for i in range(5):
            gauge = MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="Cache usage",
                samples=[MetricSample(labels=None, value=0.5)],
            )
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=1_000_000_000 + i * 1_000_000_000,  # 1 second apart
                endpoint_latency_ns=scrape_latency_ns,
                metrics={"cache_usage": gauge},
            )
            await processor.process_server_metrics_record(record)

        result = await processor.export_results(
            start_ns=1_000_000_000,
            end_ns=6_000_000_000,
        )

        assert result is not None
        assert result.endpoint_summaries is not None

        # Get the endpoint summary (key is normalized display name)
        summary = list(result.endpoint_summaries.values())[0]
        assert summary.info.unique_updates == 5
        assert summary.info.avg_fetch_latency_ms == 10.0  # 10ms
        assert summary.info.duration_seconds == 4.0  # 4 seconds (5 samples, 1s apart)
        assert (
            summary.info.avg_update_interval_ms == 1000.0
        )  # 1000ms between unique updates
        # Median should also be 1000ms for uniform intervals
        assert summary.info.median_update_interval_ms == 1000.0

    async def test_export_results_median_robust_to_outliers(
        self, mock_user_config: UserConfig
    ):
        """Test that median_update_interval_ms is robust to outliers."""
        processor = ServerMetricsAccumulator(user_config=mock_user_config)

        # Create records with non-uniform intervals:
        # Intervals: 1s, 1s, 1s, 5s (outlier)
        # avg = (1+1+1+5)/4 = 2s = 2000ms
        # median = 1s = 1000ms (robust to outlier)
        timestamps_ns = [
            1_000_000_000,  # t=1s
            2_000_000_000,  # t=2s (interval: 1s)
            3_000_000_000,  # t=3s (interval: 1s)
            4_000_000_000,  # t=4s (interval: 1s)
            9_000_000_000,  # t=9s (interval: 5s - outlier)
        ]

        for ts_ns in timestamps_ns:
            gauge = MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="Cache usage",
                samples=[MetricSample(labels=None, value=0.5)],
            )
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=ts_ns,
                endpoint_latency_ns=1_000_000,
                metrics={"cache_usage": gauge},
            )
            await processor.process_server_metrics_record(record)

        result = await processor.export_results(
            start_ns=1_000_000_000,
            end_ns=10_000_000_000,
        )

        summary = list(result.endpoint_summaries.values())[0]
        # avg = 8s / 4 intervals = 2000ms
        assert summary.info.avg_update_interval_ms == 2000.0
        # median = 1000ms (robust to outlier)
        assert summary.info.median_update_interval_ms == 1000.0


@pytest.mark.asyncio
class TestSliceDurationConfig:
    """Test that slice_duration config controls windowed stats window size."""

    async def test_slice_duration_controls_window_size(self):
        """Test that slice_duration from config is used for windowed stats."""
        # Create config with custom slice_duration
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            )
        )
        # Set slice_duration to 2 seconds
        config.output.slice_duration = 2.0

        processor = ServerMetricsAccumulator(user_config=config)
        assert processor._slice_duration == 2.0

        # Add counter samples at 1 second intervals (10 samples = 9 seconds of data)
        for i in range(10):
            counter = MetricFamily(
                type=PrometheusMetricType.COUNTER,
                description="Request count",
                samples=[MetricSample(labels=None, value=float(i * 100))],
            )
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=i * 1_000_000_000,  # 1 second apart
                endpoint_latency_ns=1_000_000,
                metrics={"requests_total": counter},
            )
            await processor.process_server_metrics_record(record)

        result = await processor.export_results(
            start_ns=0,
            end_ns=9_000_000_000,
        )

        assert result is not None
        summary = list(result.endpoint_summaries.values())[0]
        counter_stats = summary.metrics["requests_total"].series[0]

        # With 2s windows and 9s of data, we get 4 complete + 1 partial window
        # Windows: [0-2), [2-4), [4-6), [6-8), [8-9) (partial)
        assert counter_stats.timeslices is not None
        assert len(counter_stats.timeslices) == 5

        # First 4 windows: complete 2s windows with rate 100/s (200 delta / 2s)
        for i in range(4):
            rate_point = counter_stats.timeslices[i]
            assert rate_point.rate == 100.0
            assert rate_point.is_complete is None
            # Window duration should be 2 seconds
            assert (rate_point.end_ns - rate_point.start_ns) == 2_000_000_000

        # Last window: partial 1s window with rate 100/s (100 delta / 1s)
        last_slice = counter_stats.timeslices[4]
        assert last_slice.rate == 100.0
        assert not last_slice.is_complete
        assert (last_slice.end_ns - last_slice.start_ns) == 1_000_000_000  # 1 second

    async def test_default_window_size_is_1_second(self):
        """Test that default window size is 1 second when slice_duration is None."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                streaming=False,
            )
        )
        # Ensure slice_duration is None (default)
        config.output.slice_duration = None

        processor = ServerMetricsAccumulator(user_config=config)
        # When None, windowed stats are not computed
        assert processor._slice_duration is None

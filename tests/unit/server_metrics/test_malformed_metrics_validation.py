# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for histogram parsing and skipping of summary metrics."""

import pytest

from aiperf.common.mixins.base_metrics_collector_mixin import (
    FetchResult,
    HttpTraceTiming,
)
from aiperf.server_metrics.data_collector import ServerMetricsDataCollector


def make_fetch_result(metrics_text: str, latency_ns: int = 1000) -> FetchResult:
    """Create a FetchResult for testing."""
    return FetchResult(
        text=metrics_text,
        trace_timing=HttpTraceTiming(
            start_ns=1_000_000_000,
            start_perf_ns=0,
            first_byte_perf_ns=latency_ns // 2,
            end_perf_ns=latency_ns,
        ),
        is_duplicate=False,
    )


class TestHistogramParsing:
    """Test histogram parsing with various field combinations."""

    @pytest.mark.asyncio
    async def test_histogram_with_no_buckets(self):
        """Test that histograms with sum/count but no buckets create sample with empty buckets."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_sum 100.0
http_request_duration_seconds_count 50
"""
        record = collector._parse_metrics_to_records(make_fetch_result(prometheus))

        assert record is not None
        assert "http_request_duration_seconds" in record.metrics
        sample = record.metrics["http_request_duration_seconds"].samples[0]
        assert sample.buckets == {}
        assert sample.sum == 100.0
        assert sample.count == 50

    @pytest.mark.asyncio
    async def test_histogram_with_only_buckets(self):
        """Test that histograms with only buckets (no sum/count) have None for sum/count."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 10
http_request_duration_seconds_bucket{le="1.0"} 25
http_request_duration_seconds_bucket{le="+Inf"} 50
"""
        record = collector._parse_metrics_to_records(make_fetch_result(prometheus))

        assert record is not None
        assert "http_request_duration_seconds" in record.metrics
        sample = record.metrics["http_request_duration_seconds"].samples[0]
        assert sample.buckets == {"0.1": 10, "1.0": 25, "+Inf": 50}
        assert sample.sum is None
        assert sample.count is None

    @pytest.mark.asyncio
    async def test_histogram_with_only_sum(self):
        """Test that histograms with only sum have None for count and empty buckets."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_sum 100.0
"""
        record = collector._parse_metrics_to_records(make_fetch_result(prometheus))

        assert record is not None
        assert "http_request_duration_seconds" in record.metrics
        sample = record.metrics["http_request_duration_seconds"].samples[0]
        assert sample.buckets == {}
        assert sample.sum == 100.0
        assert sample.count is None

    @pytest.mark.asyncio
    async def test_valid_histogram_accepted(self):
        """Test that valid histograms with all required fields are accepted."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        valid_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 10
http_request_duration_seconds_bucket{le="1.0"} 25
http_request_duration_seconds_bucket{le="+Inf"} 50
http_request_duration_seconds_sum 100.0
http_request_duration_seconds_count 50
"""
        record = collector._parse_metrics_to_records(
            make_fetch_result(valid_prometheus)
        )

        assert record is not None
        assert "http_request_duration_seconds" in record.metrics
        metric_family = record.metrics["http_request_duration_seconds"]
        assert len(metric_family.samples) == 1
        assert metric_family.samples[0].buckets is not None
        assert len(metric_family.samples[0].buckets) == 3
        assert metric_family.samples[0].sum == 100.0
        assert metric_family.samples[0].count == 50


class TestSummaryMetricsSkipped:
    """Test that summary metrics are skipped (not supported).

    Summary metrics compute quantiles cumulatively over the entire server lifetime,
    making them unsuitable for benchmark-specific analysis. No major LLM inference
    servers use Summary metrics - they all use Histograms instead.
    """

    @pytest.mark.asyncio
    async def test_summary_metrics_are_skipped(self):
        """Test that even valid summary metrics are skipped."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        valid_prometheus = """# HELP http_request_duration_seconds Request duration
# TYPE http_request_duration_seconds summary
http_request_duration_seconds{quantile="0.5"} 0.1
http_request_duration_seconds{quantile="0.9"} 0.5
http_request_duration_seconds{quantile="0.99"} 1.0
http_request_duration_seconds_sum 100.0
http_request_duration_seconds_count 50
"""
        record = collector._parse_metrics_to_records(
            make_fetch_result(valid_prometheus)
        )

        # Summary metrics are intentionally not supported
        assert record is None


class TestMixedMetrics:
    """Test behavior when parsing mixed metric types."""

    @pytest.mark.asyncio
    async def test_valid_metrics_preserved_with_incomplete_histogram(self):
        """Test that valid metrics are still processed alongside incomplete histograms."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8080/metrics",
            collection_interval=1.0,
        )

        mixed_prometheus = """# HELP requests_total Total requests
# TYPE requests_total counter
requests_total 1000

# HELP http_duration_seconds Request duration (incomplete - no buckets)
# TYPE http_duration_seconds histogram
http_duration_seconds_sum 100.0
http_duration_seconds_count 50

# HELP active_connections Active connections
# TYPE active_connections gauge
active_connections 42
"""
        record = collector._parse_metrics_to_records(
            make_fetch_result(mixed_prometheus)
        )

        assert record is not None
        # Note: prometheus_client strips _total suffix from counter names
        assert "requests" in record.metrics
        assert "active_connections" in record.metrics
        # Incomplete histogram is now included (with empty buckets)
        assert "http_duration_seconds" in record.metrics
        sample = record.metrics["http_duration_seconds"].samples[0]
        assert sample.buckets == {}
        assert sample.sum == 100.0
        assert sample.count == 50

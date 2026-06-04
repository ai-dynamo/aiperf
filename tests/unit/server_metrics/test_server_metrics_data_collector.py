# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.exceptions import IncompatibleMetricsEndpointError
from aiperf.common.mixins.base_metrics_collector_mixin import (
    FetchResult,
    HttpTraceTiming,
)
from aiperf.common.models import ErrorDetails
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.server_metrics.data_collector import ServerMetricsDataCollector


def make_fetch_result(metrics_text: str, latency_ns: int = 1_000_000) -> FetchResult:
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


class TestServerMetricsDataCollectorInitialization:
    """Test ServerMetricsDataCollector initialization."""

    def test_initialization_complete(self):
        """Test collector initialization with all parameters."""
        collector = ServerMetricsDataCollector(
            endpoint_url="http://localhost:8081/metrics",
            collection_interval=0.5,
            reachability_timeout=10.0,
            collector_id="test_collector",
        )

        assert collector._endpoint_url == "http://localhost:8081/metrics"
        assert collector._collection_interval == 0.5
        assert collector._reachability_timeout == 10.0
        assert collector.id == "test_collector"
        assert collector._session is None
        assert not collector.was_initialized

    def test_initialization_with_defaults(self):
        """Test collector uses default values when not specified."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        assert collector._endpoint_url == "http://localhost:8081/metrics"
        assert collector._collection_interval == 0.333  # SERVER_METRICS default (333ms)
        assert collector.id == "server_metrics_collector"


class TestPrometheusMetricParsing:
    """Test Prometheus metric parsing functionality."""

    def test_parse_counter_metrics(self):
        """Test parsing simple counter metrics."""
        metrics_text = """# HELP requests_total Total requests
# TYPE requests_total counter
requests_total{status="success"} 100.0
requests_total{status="error"} 5.0
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        assert record is not None
        assert "requests" in record.metrics
        assert record.metrics["requests"].type == PrometheusMetricType.COUNTER
        assert len(record.metrics["requests"].samples) == 2

    def test_parse_gauge_metrics(self):
        """Test parsing gauge metrics."""
        metrics_text = """# HELP gpu_utilization GPU utilization percentage
# TYPE gpu_utilization gauge
gpu_utilization{gpu="0"} 0.85
gpu_utilization{gpu="1"} 0.92
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        assert record is not None
        assert "gpu_utilization" in record.metrics
        assert record.metrics["gpu_utilization"].type == PrometheusMetricType.GAUGE
        assert len(record.metrics["gpu_utilization"].samples) == 2

    def test_parse_histogram_metrics(self, sample_prometheus_metrics):
        """Test parsing histogram metrics with buckets."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(
            make_fetch_result(sample_prometheus_metrics)
        )

        assert record is not None
        assert "vllm:time_to_first_token_seconds" in record.metrics

        histogram_metric = record.metrics["vllm:time_to_first_token_seconds"]
        assert histogram_metric.type == PrometheusMetricType.HISTOGRAM
        assert len(histogram_metric.samples) == 1

        sample = histogram_metric.samples[0]
        assert sample.buckets is not None
        assert len(sample.buckets) == 4
        assert sample.sum == 125.5
        assert sample.count == 150.0

    def test_summary_metrics_are_skipped(self):
        """Test that summary metrics are skipped (not supported)."""
        metrics_text = """# HELP request_duration_seconds Request duration
# TYPE request_duration_seconds summary
request_duration_seconds{quantile="0.5"} 0.1
request_duration_seconds{quantile="0.9"} 0.5
request_duration_seconds{quantile="0.99"} 1.0
request_duration_seconds_sum 50.0
request_duration_seconds_count 100.0
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        # Summary metrics are skipped, so no record should be returned
        assert record is None

    def test_parse_mixed_metric_types(self, sample_prometheus_metrics):
        """Test parsing response containing multiple metric types."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(
            make_fetch_result(sample_prometheus_metrics)
        )

        assert record is not None

        assert "vllm:request_success" in record.metrics
        assert "vllm:gpu_cache_usage_perc" in record.metrics
        assert "vllm:time_to_first_token_seconds" in record.metrics

        assert (
            record.metrics["vllm:request_success"].type == PrometheusMetricType.COUNTER
        )
        assert (
            record.metrics["vllm:gpu_cache_usage_perc"].type
            == PrometheusMetricType.GAUGE
        )
        assert (
            record.metrics["vllm:time_to_first_token_seconds"].type
            == PrometheusMetricType.HISTOGRAM
        )

    def test_skip_created_metrics(self):
        """Test that _created metrics are skipped during parsing."""
        metrics_text = """# HELP requests_total Total requests
# TYPE requests_total counter
requests_total 100.0
requests_total_created 1704067200.0

# HELP histogram_seconds Histogram metric
# TYPE histogram_seconds histogram
histogram_seconds_bucket{le="+Inf"} 50.0
histogram_seconds_sum 5.0
histogram_seconds_count 50.0
histogram_seconds_created 1704067200.0
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        assert record is not None

        assert "requests" in record.metrics
        assert "requests_created" not in record.metrics
        assert "histogram_seconds" in record.metrics
        assert "histogram_seconds_created" not in record.metrics

    def test_parse_metrics_with_labels(self):
        """Test parsing metrics with multiple label combinations."""
        metrics_text = """# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200"} 150.0
http_requests_total{method="POST",status="200"} 75.0
http_requests_total{method="GET",status="404"} 5.0
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        assert record is not None
        assert "http_requests" in record.metrics
        assert len(record.metrics["http_requests"].samples) == 3

    def test_parse_empty_response(self):
        """Test parsing empty or whitespace-only responses."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        empty_cases = ["", "   \n\n   "]

        for empty_data in empty_cases:
            record = collector._parse_metrics_to_records(make_fetch_result(empty_data))
            assert record is None

    def test_parse_invalid_format_raises_incompatible(self):
        """Invalid Prometheus exposition format is reclassified as
        IncompatibleMetricsEndpointError so the collector auto-disables
        instead of looping on parse failures every scrape interval."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        # Invalid TYPE directive without metric name
        invalid_format = "# HELP comment\n# TYPE comment"

        with pytest.raises(IncompatibleMetricsEndpointError):
            collector._parse_metrics_to_records(make_fetch_result(invalid_format))

    def test_parse_trtllm_iteration_stats_json_raises_incompatible(self):
        """The TRT-LLM iteration-stats JSON body (``[]`` or a JSON array of
        iteration objects) at /metrics is the canonical trigger for this
        bug class — must produce IncompatibleMetricsEndpointError, not a
        bare ValueError."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        for json_body in ("[]", '[{"iter": 1, "numActiveRequests": 0}]'):
            with pytest.raises(IncompatibleMetricsEndpointError):
                collector._parse_metrics_to_records(make_fetch_result(json_body))

    def test_parse_incomplete_histogram(self):
        """Test that incomplete histograms (missing sum/count) still create samples with None values."""
        metrics_text = """# HELP incomplete_histogram Incomplete histogram
# TYPE incomplete_histogram histogram
incomplete_histogram_bucket{le="0.01"} 5.0
incomplete_histogram_bucket{le="+Inf"} 10.0
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        # Incomplete histograms now create samples with None sum/count
        assert record is not None
        assert "incomplete_histogram" in record.metrics
        sample = record.metrics["incomplete_histogram"].samples[0]
        assert sample.buckets is not None
        assert len(sample.buckets) == 2
        assert sample.sum is None
        assert sample.count is None

    def test_record_metadata_populated(self):
        """Test that ServerMetricsRecord metadata is correctly populated."""
        metrics_text = """# HELP test_metric Test metric
# TYPE test_metric counter
test_metric 1.0
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(
            make_fetch_result(metrics_text, 5_000_000)
        )

        assert record is not None

        assert record.endpoint_url == "http://localhost:8081/metrics"
        assert record.endpoint_latency_ns == 5_000_000
        assert record.timestamp_ns > 0

    def test_nan_gauge_sample_is_filtered(self):
        """NaN gauge values (e.g. sglang:fwd_occupancy with no recent traffic)
        must be filtered before the sample is constructed; otherwise the value
        survives into ZMQ transport, fails to round-trip through serialization,
        and the receiver rejects the whole batch (silent metrics loss)."""
        metrics_text = """# HELP sglang:fwd_occupancy Forward pass GPU occupancy percentage.
# TYPE sglang:fwd_occupancy gauge
sglang:fwd_occupancy{engine_type="unified",model_name="m",moe_ep_rank="0",pp_rank="0",tp_rank="0"} NaN
# HELP sglang:cache_hit_rate Prefix cache hit rate.
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="m"} 0.42
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        assert record is not None
        # The healthy metric survives.
        assert "sglang:cache_hit_rate" in record.metrics
        assert len(record.metrics["sglang:cache_hit_rate"].samples) == 1
        assert record.metrics["sglang:cache_hit_rate"].samples[0].value == 0.42
        # The NaN-only metric family is dropped entirely (no valid samples → family suppressed).
        assert "sglang:fwd_occupancy" not in record.metrics

    def test_nan_histogram_bucket_drops_whole_label_set_sample(self):
        """A NaN on any bucket/sum/count line must drop the ENTIRE histogram
        sample for that label set, not just the offending line.

        Emitting a partial sample is worse than dropping it: HistogramTimeSeries
        locks its bucket schema from the first stored sample, then ignores any
        bucket missing from that schema on every later scrape. So a truncated
        first sample would permanently drop the NaN'd bucket even once the server
        reports finite values for it. The label set must be dropped wholesale so
        the first stored sample always carries a complete, finite schema.

        Per-label-set granularity: a sibling label set with all-finite lines is
        unaffected — only the tainted set is dropped."""
        metrics_text = """# HELP my_hist Latency.
# TYPE my_hist histogram
my_hist_bucket{model_name="m",le="0.1"} NaN
my_hist_bucket{model_name="m",le="+Inf"} 50.0
my_hist_sum{model_name="m"} 17.494
my_hist_count{model_name="m"} 50.0
my_hist_bucket{model_name="ok",le="0.1"} 5.0
my_hist_bucket{model_name="ok",le="+Inf"} 30.0
my_hist_sum{model_name="ok"} 9.0
my_hist_count{model_name="ok"} 30.0
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        assert record is not None
        assert "my_hist" in record.metrics
        samples = record.metrics["my_hist"].samples
        # The tainted "m" label set is gone entirely; only the finite "ok" set
        # survives — no partial sample for "m".
        assert len(samples) == 1
        survivor = samples[0]
        assert survivor.labels == {"model_name": "ok"}
        assert survivor.buckets == {"0.1": 5.0, "+Inf": 30.0}
        assert survivor.sum == 9.0
        assert survivor.count == 30.0
        assert all(math.isfinite(v) for v in survivor.buckets.values())

    def test_nan_sample_logs_warning_once_per_metric(self, caplog):
        """When a NaN sample is filtered, emit a one-time warning naming the
        metric so silent metric loss is surfaced. Same metric across multiple
        scrapes warns once (de-duped) to prevent log spam at 333ms cadence."""
        import logging

        metrics_text = """# TYPE sglang:fwd_occupancy gauge
sglang:fwd_occupancy{rank="0"} NaN
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        with caplog.at_level(logging.WARNING, logger="aiperf"):
            # Scrape twice — both produce NaN for the same metric.
            collector._parse_metrics_to_records(make_fetch_result(metrics_text))
            collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        # Exactly one warning, naming the metric.
        nan_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "sglang:fwd_occupancy" in r.message
        ]
        assert len(nan_warnings) == 1, (
            f"Expected exactly one warning for sglang:fwd_occupancy across two scrapes, "
            f"got {len(nan_warnings)}: {[r.message for r in nan_warnings]}"
        )
        assert (
            "non-finite" in nan_warnings[0].message.lower()
            or "nan" in nan_warnings[0].message.lower()
        )

    def test_metric_sample_construction_failure_in_simple_family_drops_only_offender(
        self, caplog, monkeypatch
    ):
        """If MetricSample construction raises ValidationError for any reason
        (future schema change, unanticipated input shape, etc.), the producer
        must drop only the offending sample, keep the rest of the batch, and
        log a warn-once warning naming the metric. Future-proofs against
        failure modes the proactive NaN/Inf filter doesn't anticipate."""
        import logging

        from aiperf.server_metrics import data_collector as dc_mod

        metrics_text = """# TYPE my_gauge gauge
my_gauge{which="bad"} 1.0
my_gauge{which="good"} 2.0
"""
        original_metric_sample = dc_mod.MetricSample

        def selective_metric_sample(labels=None, value=None, **kwargs):
            if labels and labels.get("which") == "bad":
                # Trigger a genuine MetricSample ValidationError to simulate an
                # unanticipated construction failure (a value/buckets/sum/count
                # combination the model rejects), independent of the NaN/Inf path.
                original_metric_sample(value=1.0, buckets={"+Inf": 1.0})
            return original_metric_sample(labels=labels, value=value, **kwargs)

        monkeypatch.setattr(dc_mod, "MetricSample", selective_metric_sample)

        collector = dc_mod.ServerMetricsDataCollector("http://localhost:8081/metrics")
        with caplog.at_level(logging.WARNING, logger="aiperf"):
            # Scrape twice to verify warn-once.
            record = collector._parse_metrics_to_records(
                make_fetch_result(metrics_text)
            )
            collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        # The bad sample is dropped, but the good one survives the batch.
        assert record is not None
        assert "my_gauge" in record.metrics
        assert len(record.metrics["my_gauge"].samples) == 1
        assert record.metrics["my_gauge"].samples[0].labels == {"which": "good"}
        assert record.metrics["my_gauge"].samples[0].value == 2.0

        # Exactly one warn-once warning across the two scrapes.
        construction_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "my_gauge" in r.message
            and "construction" in r.message.lower()
        ]
        assert len(construction_warnings) == 1, (
            f"Expected exactly one construction-failure warning for my_gauge across "
            f"two scrapes, got {len(construction_warnings)}: "
            f"{[r.message for r in construction_warnings]}"
        )

    def test_metric_sample_construction_failure_in_histogram_family_drops_only_offender(
        self, caplog, monkeypatch
    ):
        """Parallel to the simple-family test, but for histograms. If
        hist.to_metric_sample() construction raises, the producer must drop
        only that one histogram's MetricSample and keep healthy histograms."""
        import logging

        from aiperf.server_metrics import data_collector as dc_mod

        metrics_text = """# TYPE my_histogram histogram
my_histogram_bucket{which="bad",le="0.1"} 5.0
my_histogram_bucket{which="bad",le="+Inf"} 10.0
my_histogram_sum{which="bad"} 0.5
my_histogram_count{which="bad"} 10.0
my_histogram_bucket{which="good",le="0.1"} 3.0
my_histogram_bucket{which="good",le="+Inf"} 7.0
my_histogram_sum{which="good"} 0.3
my_histogram_count{which="good"} 7.0
"""
        original_metric_sample = dc_mod.MetricSample

        def selective_metric_sample(labels=None, **kwargs):
            if labels and labels.get("which") == "bad":
                # Trigger a genuine MetricSample ValidationError to simulate an
                # unanticipated construction failure on this histogram.
                original_metric_sample(value=1.0, buckets={"+Inf": 1.0})
            return original_metric_sample(labels=labels, **kwargs)

        monkeypatch.setattr(dc_mod, "MetricSample", selective_metric_sample)

        collector = dc_mod.ServerMetricsDataCollector("http://localhost:8081/metrics")
        with caplog.at_level(logging.WARNING, logger="aiperf"):
            record = collector._parse_metrics_to_records(
                make_fetch_result(metrics_text)
            )

        assert record is not None
        assert "my_histogram" in record.metrics
        samples = record.metrics["my_histogram"].samples
        assert len(samples) == 1
        assert samples[0].labels == {"which": "good"}

        failures = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "my_histogram" in r.message
            and "construction" in r.message.lower()
        ]
        assert len(failures) == 1

    def test_realistic_sglang_payload_with_nan_gauge_drops_only_offender(self):
        """Realistic sglang scrape with a NaN gauge AND a NaN histogram bucket
        alongside healthy metrics. Verifies the full _parse_metrics_to_records path:
          - NaN gauge is dropped (family suppressed since it had only one sample)
          - Healthy gauge survives
          - Histogram label set is dropped wholesale because one bucket is NaN
            (family suppressed since it had only the one tainted label set) — a
            partial sample would lock a truncated bucket schema downstream
          - The resulting record can be constructed without raising
        Regression test for the silent-loss bug observed against sglang
        --enable-metrics where sglang:fwd_occupancy emitted NaN, extended to
        cover the parallel histogram-path filter (Task 1b)."""
        metrics_text = """# HELP sglang:fwd_occupancy Forward pass GPU occupancy percentage.
# TYPE sglang:fwd_occupancy gauge
sglang:fwd_occupancy{engine_type="unified",model_name="Qwen/Qwen3-0.6B",moe_ep_rank="0",pp_rank="0",tp_rank="0"} NaN
# HELP sglang:cache_hit_rate Prefix cache hit rate.
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="Qwen/Qwen3-0.6B"} 0.873
# HELP sglang:num_running_reqs Number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="Qwen/Qwen3-0.6B"} 4
# HELP sglang:time_to_first_token_seconds TTFT histogram
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-0.6B",le="0.05"} NaN
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-0.6B",le="0.1"} 46.0
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-0.6B",le="+Inf"} 50.0
sglang:time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-0.6B"} 17.494
sglang:time_to_first_token_seconds_count{model_name="Qwen/Qwen3-0.6B"} 50.0
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        assert record is not None
        # Healthy metrics survive.
        assert "sglang:cache_hit_rate" in record.metrics
        assert record.metrics["sglang:cache_hit_rate"].samples[0].value == 0.873
        assert "sglang:num_running_reqs" in record.metrics
        assert record.metrics["sglang:num_running_reqs"].samples[0].value == 4.0
        # Histogram's only label set had a NaN bucket -> the whole label set is
        # dropped, suppressing the family. Dropping only the NaN line would emit a
        # partial sample that locks a truncated bucket schema in HistogramTimeSeries.
        assert "sglang:time_to_first_token_seconds" not in record.metrics
        # Offender gauge dropped.
        assert "sglang:fwd_occupancy" not in record.metrics


class TestMetricDeduplication:
    """Test metric sample deduplication logic."""

    def test_duplicate_counter_values_last_wins(self):
        """Test that duplicate counter samples keep last value."""
        metrics_text = """# HELP test_counter Test counter
# TYPE test_counter counter
test_counter{label="a"} 10.0
test_counter{label="a"} 20.0
test_counter{label="a"} 30.0
"""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))

        assert record is not None
        samples = record.metrics["test_counter"].samples

        assert len(samples) == 1
        assert samples[0].value == 30.0


class TestPrometheusFallbackProbe:
    """When the configured /metrics path returns non-Prometheus content,
    the collector should probe `<base>/prometheus/metrics` once, swap the
    URL there on success, and fall through to auto-disable on failure.

    This is the TRT-LLM compatibility path: ``return_perf_metrics: true``
    mounts Prometheus exposition at the non-standard /prometheus/metrics
    location while the default /metrics still serves iteration-stats JSON.
    """

    PROM_BODY = "# HELP up Whether the target is up\n# TYPE up gauge\nup 1.0\n"
    JSON_BODY = "[]"

    @pytest.mark.asyncio
    async def test_probe_swaps_url_on_successful_prometheus_fallback(self) -> None:
        collector = ServerMetricsDataCollector(
            endpoint_url="http://server:60000/metrics"
        )
        # Original /metrics raises Incompatible (JSON path), then the probe
        # at /prometheus/metrics succeeds with valid Prom exposition.
        fetch_results = [
            IncompatibleMetricsEndpointError("/metrics returned application/json"),
            FetchResult(
                text=self.PROM_BODY,
                trace_timing=HttpTraceTiming(
                    start_ns=1, start_perf_ns=0, first_byte_perf_ns=1, end_perf_ns=2
                ),
            ),
        ]

        async def fake_fetch() -> FetchResult:
            value = fetch_results.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        sent: list = []

        async def fake_send(records: list) -> None:
            sent.append(records)

        with (
            patch.object(collector, "_fetch_metrics_text", side_effect=fake_fetch),
            patch.object(
                collector, "_send_records_via_callback", side_effect=fake_send
            ),
        ):
            await collector._collect_and_process_metrics()

        assert collector._endpoint_url == "http://server:60000/prometheus/metrics"
        assert collector._prometheus_fallback_attempted is True
        # The fallback attempt produced a record from the alt endpoint.
        assert len(sent) == 1 and len(sent[0]) == 1

    @pytest.mark.asyncio
    async def test_probe_failure_restores_url_and_reraises(self) -> None:
        collector = ServerMetricsDataCollector(
            endpoint_url="http://server:60000/metrics"
        )

        # Original /metrics raises Incompatible, then the probe at
        # /prometheus/metrics ALSO raises (e.g. 404 or also JSON).
        async def fake_fetch() -> FetchResult:
            raise IncompatibleMetricsEndpointError("simulated probe failure")

        with (
            patch.object(collector, "_fetch_metrics_text", side_effect=fake_fetch),
            pytest.raises(IncompatibleMetricsEndpointError),
        ):
            await collector._collect_and_process_metrics()

        # URL must be restored so it shows up correctly in logs / status messages
        assert collector._endpoint_url == "http://server:60000/metrics"
        assert collector._prometheus_fallback_attempted is True

    @pytest.mark.asyncio
    async def test_probe_404_translates_to_incompatible_endpoint_error(self) -> None:
        """The realistic TRT-LLM-without-`return_perf_metrics` case:
        ``/metrics`` returns JSON, ``/prometheus/metrics`` returns 404
        (i.e. ``aiohttp.ClientResponseError``, NOT
        ``IncompatibleMetricsEndpointError``). The probe failure must be
        translated to ``IncompatibleMetricsEndpointError`` so the base
        mixin's auto-disable wrapper triggers — otherwise the collector
        would keep scraping the broken original URL every interval.
        """
        collector = ServerMetricsDataCollector(
            endpoint_url="http://server:60000/metrics"
        )
        call_count = {"n": 0}

        async def fake_fetch() -> FetchResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call (against /metrics): JSON body → Incompatible
                raise IncompatibleMetricsEndpointError("/metrics returned JSON")
            # Second call (against /prometheus/metrics): 404 → ClientResponseError
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Not Found",
            )

        with (
            patch.object(collector, "_fetch_metrics_text", side_effect=fake_fetch),
            pytest.raises(IncompatibleMetricsEndpointError) as exc_info,
        ):
            await collector._collect_and_process_metrics()

        # The 404 (a ClientResponseError) was translated, not bubbled up raw —
        # so the auto-disable wrapper will catch it.
        assert "Prometheus fallback" in str(exc_info.value)
        assert "return_perf_metrics" in str(exc_info.value)
        # And the chained __cause__ preserves the original 404 for diagnostics.
        assert isinstance(exc_info.value.__cause__, aiohttp.ClientResponseError)
        # URL restored to the original.
        assert collector._endpoint_url == "http://server:60000/metrics"

    @pytest.mark.asyncio
    async def test_probe_runs_at_most_once_per_collector(self) -> None:
        collector = ServerMetricsDataCollector(
            endpoint_url="http://server:60000/metrics"
        )

        async def always_incompatible() -> FetchResult:
            raise IncompatibleMetricsEndpointError("never works")

        with patch.object(
            collector, "_fetch_metrics_text", side_effect=always_incompatible
        ) as mock_fetch:
            with pytest.raises(IncompatibleMetricsEndpointError):
                await collector._collect_and_process_metrics()
            # Two fetches: the original and the one fallback attempt.
            assert mock_fetch.await_count == 2

            mock_fetch.reset_mock()
            with pytest.raises(IncompatibleMetricsEndpointError):
                await collector._collect_and_process_metrics()
            # Second cycle must not re-probe — single fetch on the original URL.
            assert mock_fetch.await_count == 1

    @pytest.mark.asyncio
    async def test_no_probe_when_url_already_targets_prometheus_path(self) -> None:
        collector = ServerMetricsDataCollector(
            endpoint_url="http://server:60000/prometheus/metrics"
        )

        async def fake_fetch() -> FetchResult:
            raise IncompatibleMetricsEndpointError("already on prometheus path")

        with patch.object(
            collector, "_fetch_metrics_text", side_effect=fake_fetch
        ) as mock_fetch:
            with pytest.raises(IncompatibleMetricsEndpointError):
                await collector._collect_and_process_metrics()
            # Only one fetch; the probe is skipped because we're already on
            # /prometheus/metrics — there's no further alt path to try.
            assert mock_fetch.await_count == 1
        assert collector._prometheus_fallback_attempted is False

    @pytest.mark.asyncio
    async def test_no_probe_when_url_does_not_end_with_metrics(self) -> None:
        collector = ServerMetricsDataCollector(
            endpoint_url="http://server:60000/custom/path"
        )

        async def fake_fetch() -> FetchResult:
            raise IncompatibleMetricsEndpointError("non-standard path")

        with patch.object(
            collector, "_fetch_metrics_text", side_effect=fake_fetch
        ) as mock_fetch:
            with pytest.raises(IncompatibleMetricsEndpointError):
                await collector._collect_and_process_metrics()
            assert mock_fetch.await_count == 1
        assert collector._prometheus_fallback_attempted is False


class TestAsyncLifecycle:
    """Test async lifecycle management."""

    @pytest.mark.asyncio
    async def test_initialization_creates_session(self):
        """Test that initialization creates aiohttp session."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        await collector.initialize()

        assert collector._session is not None
        assert isinstance(collector._session, aiohttp.ClientSession)

        await collector.stop()

    @pytest.mark.asyncio
    async def test_initialization_creates_connector(self):
        """Test that initialization creates TCP connector with proper settings."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        await collector.initialize()

        assert collector._connector is not None
        assert isinstance(collector._connector, aiohttp.TCPConnector)

        await collector.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_session(self):
        """Test that stop closes aiohttp session."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        await collector.initialize()
        session = collector._session

        await collector.stop()

        assert session.closed

    @pytest.mark.asyncio
    async def test_stop_closes_connector(self):
        """Test that stop closes TCP connector."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        await collector.initialize()
        connector = collector._connector

        await collector.stop()

        assert connector.closed
        assert collector._connector is None

    @pytest.mark.asyncio
    async def test_reachability_check_success(self):
        """Test URL reachability check with successful response."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        with patch.object(
            collector, "_check_reachability_with_session", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = True

            await collector.initialize()
            is_reachable = await collector.is_url_reachable()

            assert is_reachable
            mock_check.assert_called_once()

        await collector.stop()

    @pytest.mark.asyncio
    async def test_reachability_check_failure(self):
        """Test URL reachability check with failed response."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        with patch.object(
            collector, "_check_reachability_with_session", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = False

            await collector.initialize()
            is_reachable = await collector.is_url_reachable()

            assert not is_reachable

        await collector.stop()

    @pytest.mark.asyncio
    async def test_reachability_check_without_session_uses_connector(self):
        """Test reachability check creates temporary connector when no session exists."""
        collector = ServerMetricsDataCollector("http://localhost:8081/metrics")

        # Don't initialize - no session exists
        assert collector._session is None

        with patch(
            "aiperf.common.mixins.base_metrics_collector_mixin.create_tcp_connector"
        ) as mock_create:
            mock_connector = AsyncMock()
            mock_connector.close = AsyncMock()
            mock_create.return_value = mock_connector

            with patch.object(
                collector, "_check_reachability_with_session", new_callable=AsyncMock
            ) as mock_check:
                mock_check.return_value = True
                await collector.is_url_reachable()

            # Verify connector was created and closed
            mock_create.assert_called_once()
            mock_connector.close.assert_called_once()


class TestCollectorCallbackFunctionality:
    """Test callback mechanisms for records and errors."""

    @pytest.mark.asyncio
    async def test_record_callback_invoked(self):
        """Test that record callback is invoked with collected records."""
        record_callback = AsyncMock()
        collector = ServerMetricsDataCollector(
            "http://localhost:8081/metrics",
            record_callback=record_callback,
            collector_id="test_collector",
        )

        test_records = [
            ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={},
            )
        ]

        await collector._send_records_via_callback(test_records)

        record_callback.assert_called_once_with(test_records, "test_collector")

    @pytest.mark.asyncio
    async def test_error_callback_invoked(self):
        """Test that error callback is invoked on collection errors."""
        error_callback = AsyncMock()
        collector = ServerMetricsDataCollector(
            "http://localhost:8081/metrics",
            error_callback=error_callback,
            collector_id="test_collector",
        )

        await collector.initialize()

        with patch.object(
            collector,
            "_collect_and_process_metrics",
            side_effect=ValueError("Test error"),
        ):
            await collector.collect_and_process_metrics()

        error_callback.assert_called_once()
        args = error_callback.call_args[0]
        assert isinstance(args[0], ErrorDetails)
        assert args[1] == "test_collector"

        await collector.stop()

    @pytest.mark.asyncio
    async def test_no_callback_on_empty_records(self):
        """Test that record callback is not invoked for empty record list."""
        record_callback = AsyncMock()
        collector = ServerMetricsDataCollector(
            "http://localhost:8081/metrics",
            record_callback=record_callback,
        )

        await collector._send_records_via_callback([])

        record_callback.assert_not_called()


class TestServerMetricsDataCollectorCredentialRedaction:
    """Regression tests for #935: credentials embedded in --url must not leak
    into logs, record keys, or exported artifacts, while the raw URL is still
    used for the actual HTTP fetch (so authentication keeps working)."""

    CREDENTIALED_URL = "http://alice:s3cret@127.0.0.1:34883/metrics"

    def test_display_url_is_redacted_but_fetch_url_is_raw(self):
        """The raw URL (with userinfo) is retained for fetching; the display
        URL used for logs/records has the credentials stripped."""
        collector = ServerMetricsDataCollector(endpoint_url=self.CREDENTIALED_URL)
        # Raw form is kept for the actual HTTP fetch (auth must still work).
        assert collector._endpoint_url == self.CREDENTIALED_URL
        assert "s3cret" in collector._endpoint_url
        # Display form is redacted: no credentials anywhere.
        assert "s3cret" not in collector._display_url
        assert "alice" not in collector._display_url
        assert collector._display_url == "http://<redacted>@127.0.0.1:34883/metrics"

    def test_url_without_credentials_is_unchanged(self):
        """redact_url is a no-op on credential-free URLs, so display and fetch
        forms are identical."""
        plain_url = "http://localhost:8081/metrics"
        collector = ServerMetricsDataCollector(endpoint_url=plain_url)
        assert collector._endpoint_url == plain_url
        assert collector._display_url == plain_url

    def test_record_endpoint_url_is_redacted(self):
        """The endpoint_url stored on each ServerMetricsRecord feeds every
        export (csv/json/parquet); it must carry the redacted form."""
        collector = ServerMetricsDataCollector(endpoint_url=self.CREDENTIALED_URL)
        metrics_text = (
            "# HELP test_metric A test metric\n"
            "# TYPE test_metric counter\n"
            "test_metric 1.0\n"
        )
        record = collector._parse_metrics_to_records(make_fetch_result(metrics_text))
        assert record is not None
        assert "s3cret" not in record.endpoint_url
        assert record.endpoint_url == "http://<redacted>@127.0.0.1:34883/metrics"

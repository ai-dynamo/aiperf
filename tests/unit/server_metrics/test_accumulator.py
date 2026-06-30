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

    async def test_realtime_snapshot_suppresses_single_sample_counter_deltas(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test realtime counter deltas require two samples."""
        processor = ServerMetricsAccumulator(mock_user_config)
        await processor.process_server_metrics_record(
            ServerMetricsRecord(
                endpoint_url="http://127.0.0.1:8000/metrics",
                timestamp_ns=1_000_000_000,
                metrics={
                    "vllm:prefix_cache_hits": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Prefix cache hits.",
                        samples=[MetricSample(value=500.0)],
                    ),
                    "vllm:prefix_cache_queries": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Prefix cache queries.",
                        samples=[MetricSample(value=1000.0)],
                    ),
                    "vllm:num_preemptions": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Preemptions.",
                        samples=[MetricSample(value=7.0)],
                    ),
                },
            )
        )

        snapshot = processor.realtime_snapshot()

        assert "prefix_cache_hit_rate" not in snapshot
        assert "num_preemptions" not in snapshot

    async def test_realtime_snapshot_can_exclude_warmup_counters(
        self, mock_user_config: UserConfig
    ) -> None:
        """Test realtime counter deltas can use the profiling-start baseline."""
        processor = ServerMetricsAccumulator(mock_user_config)
        for timestamp_ns, hits, queries in (
            (1_000_000_000, 0.0, 0.0),
            (2_000_000_000, 10.0, 1000.0),
            (3_000_000_000, 9010.0, 10000.0),
        ):
            await processor.process_server_metrics_record(
                ServerMetricsRecord(
                    endpoint_url="http://127.0.0.1:8000/metrics",
                    timestamp_ns=timestamp_ns,
                    metrics={
                        "vllm:prefix_cache_hits": MetricFamily(
                            type=PrometheusMetricType.COUNTER,
                            description="Prefix cache hits.",
                            samples=[MetricSample(value=hits)],
                        ),
                        "vllm:prefix_cache_queries": MetricFamily(
                            type=PrometheusMetricType.COUNTER,
                            description="Prefix cache queries.",
                            samples=[MetricSample(value=queries)],
                        ),
                    },
                )
            )

        full_snapshot = processor.realtime_snapshot()
        profiling_snapshot = processor.realtime_snapshot(start_ns=2_500_000_000)

        assert full_snapshot["prefix_cache_hit_rate"] == pytest.approx(90.1)
        assert full_snapshot["unique_input_tokens_srv"] == pytest.approx(990.0)
        assert profiling_snapshot["prefix_cache_hit_rate"] == pytest.approx(100.0)
        assert profiling_snapshot["unique_input_tokens_srv"] == pytest.approx(0.0)

    async def test_realtime_snapshot_uses_sglang_retracted_total_counter(
        self, mock_user_config: UserConfig
    ) -> None:
        # SGLang exposes preemptions as `sglang:num_retracted_reqs_total`
        # (counter). `prometheus_client.parser.text_string_to_metric_families`
        # strips `_total` from counter family names before the data collector
        # stores them, so aiperf looks the metric up by the stripped form. The
        # COUNTER type filter in `_counter_delta` keeps it from picking up
        # SGLang's gauge of the same stripped name in clusters that emit both.
        processor = ServerMetricsAccumulator(mock_user_config)
        for timestamp_ns, counter_value in (
            (1_000_000_000, 10.0),
            (2_000_000_000, 12.0),
        ):
            await processor.process_server_metrics_record(
                ServerMetricsRecord(
                    endpoint_url="http://127.0.0.1:8000/metrics",
                    timestamp_ns=timestamp_ns,
                    metrics={
                        "sglang:num_retracted_reqs": MetricFamily(
                            type=PrometheusMetricType.COUNTER,
                            description="Total retracted requests.",
                            samples=[MetricSample(value=counter_value)],
                        ),
                    },
                )
            )

        snapshot = processor.realtime_snapshot()

        assert snapshot["num_preemptions"] == 2.0

    async def test_realtime_snapshot_uses_sglang_fallbacks_when_vllm_absent(
        self, mock_user_config: UserConfig
    ) -> None:
        # SGLang servers emit `sglang:*` metric names; the realtime snapshot
        # should populate the same fields as for vLLM via per-field fallbacks.
        # Counter pair `cached_tokens_total`/`prompt_tokens_total` drives the
        # cumulative prefix-cache hit rate (and the uncached delta), matching
        # vLLM's `prefix_cache_hits`/`prefix_cache_queries` shape — preferred
        # over the per-batch `sglang:cache_hit_rate` gauge because that gauge
        # reads 0 between requests.
        processor = ServerMetricsAccumulator(mock_user_config)
        for timestamp_ns, prompt_total, cached_total, generation_total in (
            (1_000_000_000, 0.0, 0.0, 0.0),
            (2_000_000_000, 1_000_000.0, 700_000.0, 5_000.0),
        ):
            await processor.process_server_metrics_record(
                ServerMetricsRecord(
                    endpoint_url="http://127.0.0.1:8000/metrics",
                    timestamp_ns=timestamp_ns,
                    metrics={
                        "sglang:cache_hit_rate": MetricFamily(
                            type=PrometheusMetricType.GAUGE,
                            description="Per-batch prefix cache hit rate (0-1).",
                            samples=[MetricSample(value=0.0)],
                        ),
                        "sglang:token_usage": MetricFamily(
                            type=PrometheusMetricType.GAUGE,
                            description="KV cache token usage (0-1).",
                            samples=[MetricSample(value=0.88)],
                        ),
                        "sglang:num_running_reqs": MetricFamily(
                            type=PrometheusMetricType.GAUGE,
                            description="Running requests.",
                            samples=[MetricSample(value=6.0)],
                        ),
                        "sglang:num_queue_reqs": MetricFamily(
                            type=PrometheusMetricType.GAUGE,
                            description="Queued requests.",
                            samples=[MetricSample(value=1.0)],
                        ),
                        # Counter family names are stored without `_total`
                        # because `prometheus_client.parser` strips that suffix
                        # before the data collector sees them. Look up by the
                        # parser-stripped form.
                        "sglang:prompt_tokens": MetricFamily(
                            type=PrometheusMetricType.COUNTER,
                            description="Total prefill tokens.",
                            samples=[MetricSample(value=prompt_total)],
                        ),
                        "sglang:cached_tokens": MetricFamily(
                            type=PrometheusMetricType.COUNTER,
                            description="Total prefix-cached prefill tokens.",
                            samples=[MetricSample(value=cached_total)],
                        ),
                        "sglang:generation_tokens": MetricFamily(
                            type=PrometheusMetricType.COUNTER,
                            description="Total generation tokens.",
                            samples=[MetricSample(value=generation_total)],
                        ),
                    },
                )
            )

        snapshot = processor.realtime_snapshot()

        # Counter pair wins over the gauge: 700k cached / 1M prompt = 70%.
        assert snapshot["prefix_cache_hit_rate"] == pytest.approx(70.0)
        assert snapshot["unique_input_tokens_srv"] == pytest.approx(300_000.0)
        assert snapshot["kv_cache_usage_pct"] == pytest.approx(88.0)
        assert snapshot["num_running"] == 6.0
        assert snapshot["num_waiting"] == 1.0
        # Counter rate over the 1s window between samples.
        assert snapshot["input_token_throughput_srv"] == pytest.approx(1_000_000.0)
        assert snapshot["output_token_throughput_srv"] == pytest.approx(5_000.0)

    async def test_realtime_snapshots_keep_workers_separate(
        self, mock_user_config: UserConfig
    ) -> None:
        """The iterative srv rows should retain per-worker values and roles."""
        processor = ServerMetricsAccumulator(mock_user_config)
        endpoint_samples = {
            "http://prefill-0:8081/metrics": (
                "prefill",
                (
                    (0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.65),
                    (80.0, 100.0, 100.0, 10.0, 1.0, 3.0, 2.0, 0.70),
                ),
            ),
            "http://decode-0:8083/metrics": (
                "backend",
                (
                    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.30),
                    (45.0, 50.0, 50.0, 500.0, 2.0, 4.0, 5.0, 0.40),
                ),
            ),
        }

        for endpoint_url, (component, samples) in endpoint_samples.items():
            for timestamp_ns, values in zip(
                (1_000_000_000, 2_000_000_000), samples, strict=True
            ):
                hits, queries, prompt, generation, preemptions, running, waiting, kv = (
                    values
                )
                metrics = {
                    name: MetricFamily(
                        type=metric_type,
                        description=name,
                        samples=[
                            MetricSample(
                                labels={"dynamo_component": component}, value=value
                            )
                        ],
                    )
                    for name, metric_type, value in (
                        ("vllm:prefix_cache_hits", PrometheusMetricType.COUNTER, hits),
                        (
                            "vllm:prefix_cache_queries",
                            PrometheusMetricType.COUNTER,
                            queries,
                        ),
                        ("vllm:prompt_tokens", PrometheusMetricType.COUNTER, prompt),
                        (
                            "vllm:generation_tokens",
                            PrometheusMetricType.COUNTER,
                            generation,
                        ),
                        (
                            "vllm:num_preemptions",
                            PrometheusMetricType.COUNTER,
                            preemptions,
                        ),
                        (
                            "vllm:num_requests_running",
                            PrometheusMetricType.GAUGE,
                            running,
                        ),
                        (
                            "vllm:num_requests_waiting",
                            PrometheusMetricType.GAUGE,
                            waiting,
                        ),
                        ("vllm:kv_cache_usage_perc", PrometheusMetricType.GAUGE, kv),
                    )
                }
                await processor.process_server_metrics_record(
                    ServerMetricsRecord(
                        endpoint_url=endpoint_url,
                        timestamp_ns=timestamp_ns,
                        metrics=metrics,
                    )
                )

        snapshots = processor.realtime_snapshots()

        assert list(snapshots) == ["decode 0", "prefill 0"]
        assert snapshots["prefill 0"]["prefix_cache_hit_rate"] == pytest.approx(80.0)
        assert snapshots["prefill 0"]["unique_input_tokens_srv"] == pytest.approx(20.0)
        assert snapshots["prefill 0"]["kv_cache_usage_pct"] == pytest.approx(70.0)
        assert snapshots["prefill 0"]["num_running"] == pytest.approx(3.0)
        assert snapshots["prefill 0"]["num_waiting"] == pytest.approx(2.0)
        assert snapshots["prefill 0"]["input_token_throughput_srv"] == pytest.approx(
            100.0
        )
        assert snapshots["decode 0"]["prefix_cache_hit_rate"] == pytest.approx(90.0)
        assert snapshots["decode 0"]["kv_cache_usage_pct"] == pytest.approx(40.0)
        assert snapshots["decode 0"]["num_running"] == pytest.approx(4.0)
        assert snapshots["decode 0"]["num_waiting"] == pytest.approx(5.0)
        assert snapshots["decode 0"]["num_preemptions"] == pytest.approx(2.0)
        assert snapshots["decode 0"]["output_token_throughput_srv"] == pytest.approx(
            500.0
        )

    async def test_realtime_snapshot_falls_back_to_sglang_gauge_when_counters_absent(
        self, mock_user_config: UserConfig
    ) -> None:
        # Older SGLang builds emit only the per-batch `cache_hit_rate` gauge
        # (no `cached_tokens_total` counter). The accumulator should still
        # populate `prefix_cache_hit_rate` from the gauge, but
        # `unique_input_tokens_srv` is unrecoverable in that case.
        processor = ServerMetricsAccumulator(mock_user_config)
        await processor.process_server_metrics_record(
            ServerMetricsRecord(
                endpoint_url="http://127.0.0.1:8000/metrics",
                timestamp_ns=1_000_000_000,
                metrics={
                    "sglang:cache_hit_rate": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        description="Per-batch prefix cache hit rate.",
                        samples=[MetricSample(value=0.42)],
                    ),
                },
            )
        )

        snapshot = processor.realtime_snapshot()

        assert snapshot["prefix_cache_hit_rate"] == pytest.approx(42.0)
        assert "unique_input_tokens_srv" not in snapshot

    async def test_realtime_snapshot_handles_parser_stripped_total_suffix(
        self, mock_user_config: UserConfig
    ) -> None:
        # Regression test for the `_total` parser-stripping bug. Drives the
        # accumulator with text routed through the same Prometheus parser the
        # data collector uses, so any future drift between lookup names and
        # the parser's family-name convention fails here loudly rather than
        # silently suppressing the throughput row at runtime. Without the fix,
        # `_counter_rate("vllm:prompt_tokens_total", ...)` returned None
        # because the parser had stored the family as "vllm:prompt_tokens".
        from prometheus_client.parser import text_string_to_metric_families

        text_t1 = """\
# HELP vllm:prompt_tokens_total Total prompt tokens.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="m"} 0.0
# HELP vllm:generation_tokens_total Total generation tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="m"} 0.0
"""
        text_t2 = """\
# HELP vllm:prompt_tokens_total Total prompt tokens.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="m"} 1000000.0
# HELP vllm:generation_tokens_total Total generation tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="m"} 5000.0
"""

        def parse(text: str) -> dict[str, MetricFamily]:
            out: dict[str, MetricFamily] = {}
            for family in text_string_to_metric_families(text):
                metric_type = PrometheusMetricType(family.type)
                samples = [
                    MetricSample(labels=dict(s.labels) or None, value=s.value)
                    for s in family.samples
                ]
                out[family.name] = MetricFamily(
                    type=metric_type,
                    description=family.documentation or "",
                    samples=samples,
                )
            return out

        processor = ServerMetricsAccumulator(mock_user_config)
        for ts, text in (
            (1_000_000_000, text_t1),
            (2_000_000_000, text_t2),
        ):
            await processor.process_server_metrics_record(
                ServerMetricsRecord(
                    endpoint_url="http://127.0.0.1:8000/metrics",
                    timestamp_ns=ts,
                    metrics=parse(text),
                )
            )

        snapshot = processor.realtime_snapshot()

        # Rates over a 1-second window between samples.
        assert snapshot["input_token_throughput_srv"] == pytest.approx(1_000_000.0)
        assert snapshot["output_token_throughput_srv"] == pytest.approx(5_000.0)

    async def test_realtime_snapshot_derives_cpu_kv_from_sglang_hicache(
        self, mock_user_config: UserConfig
    ) -> None:
        # SGLang HiCache exposes host-tier capacity as used/total token
        # gauges only; the ratio is computed here so the field populates
        # whenever HiCache is active.
        processor = ServerMetricsAccumulator(mock_user_config)
        await processor.process_server_metrics_record(
            ServerMetricsRecord(
                endpoint_url="http://127.0.0.1:8000/metrics",
                timestamp_ns=1_000_000_000,
                metrics={
                    "sglang:hicache_host_used_tokens": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        description="Tokens in host KV cache.",
                        samples=[MetricSample(value=300_000.0)],
                    ),
                    "sglang:hicache_host_total_tokens": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        description="Host KV cache capacity in tokens.",
                        samples=[MetricSample(value=1_000_000.0)],
                    ),
                },
            )
        )

        snapshot = processor.realtime_snapshot()

        assert snapshot["cpu_kv_cache_usage_pct"] == pytest.approx(30.0)

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

    async def test_export_results_includes_warmup_endpoint_summaries(
        self,
        mock_user_config: UserConfig,
    ) -> None:
        """Test export_results computes separate profiling and warmup summaries."""
        processor = ServerMetricsAccumulator(mock_user_config)

        for timestamp_ns, value in (
            (1_000_000_000, 0.1),
            (1_500_000_000, 0.2),
            (2_500_000_000, 0.8),
        ):
            gauge = MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="KV cache usage",
                samples=[MetricSample(labels=None, value=value)],
            )
            record = ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=timestamp_ns,
                endpoint_latency_ns=5_000_000,
                metrics={"cache_usage": gauge},
            )
            await processor.process_server_metrics_record(record)

        result = await processor.export_results(
            start_ns=2_000_000_000,
            end_ns=3_000_000_000,
            warmup_start_ns=1_000_000_000,
            warmup_end_ns=2_000_000_000,
        )

        assert result is not None
        assert result.endpoint_summaries is not None
        assert result.warmup_endpoint_summaries is not None
        assert result.warmup_start_ns == 1_000_000_000
        assert result.warmup_end_ns == 2_000_000_000

        endpoint_key = next(iter(result.endpoint_summaries))
        profiling_summary = result.endpoint_summaries[endpoint_key]
        warmup_summary = result.warmup_endpoint_summaries[endpoint_key]
        profiling_avg = profiling_summary.metrics["cache_usage"].series[0].stats.avg
        warmup_avg = warmup_summary.metrics["cache_usage"].series[0].stats.avg
        assert profiling_avg == pytest.approx(0.8)
        assert warmup_avg == pytest.approx(0.15)

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

    async def test_export_results_extends_parquet_filter_to_endpoint_last_update(
        self,
        mock_user_config: UserConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test Parquet export uses the same final-collection end window as summaries."""
        processor = ServerMetricsAccumulator(mock_user_config)
        exported_filters = []

        async def capture_export_filter(time_filter):
            exported_filters.append(time_filter)

        monkeypatch.setattr(
            processor, "_export_parquet_if_enabled", capture_export_filter
        )

        for timestamp_ns in (1_000_000_000, 3_000_000_000):
            gauge = MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="Cache usage",
                samples=[MetricSample(labels=None, value=0.5)],
            )
            await processor.process_server_metrics_record(
                ServerMetricsRecord(
                    endpoint_url="http://node1:8081/metrics",
                    timestamp_ns=timestamp_ns,
                    endpoint_latency_ns=5_000_000,
                    metrics={"cache_usage": gauge},
                )
            )

        result = await processor.export_results(
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )

        assert result is not None
        assert len(exported_filters) == 1
        assert exported_filters[0].start_ns == 1_000_000_000
        assert exported_filters[0].end_ns == 3_000_000_000

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

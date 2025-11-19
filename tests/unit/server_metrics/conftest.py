# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    SummaryData,
)


@pytest.fixture
def sample_prometheus_metrics() -> str:
    """Sample Prometheus metrics text from vLLM endpoint."""
    return """# HELP vllm:request_success_total Total number of successful requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 150.0

# HELP vllm:time_to_first_token_seconds Time to first token
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",le="0.001"} 0.0
vllm:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",le="0.005"} 5.0
vllm:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",le="0.01"} 15.0
vllm:time_to_first_token_seconds_bucket{model_name="meta-llama/Llama-3.1-8B-Instruct",le="+Inf"} 150.0
vllm:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 125.5
vllm:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 150.0

# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage percentage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.42
"""


@pytest.fixture
def sample_counter_metric() -> MetricFamily:
    """Sample counter metric family."""
    return MetricFamily(
        type=PrometheusMetricType.COUNTER,
        help="Total number of successful requests",
        samples=[
            MetricSample(
                labels={"model_name": "meta-llama/Llama-3.1-8B-Instruct"},
                value=150.0,
            )
        ],
    )


@pytest.fixture
def sample_gauge_metric() -> MetricFamily:
    """Sample gauge metric family."""
    return MetricFamily(
        type=PrometheusMetricType.GAUGE,
        help="GPU KV-cache usage percentage",
        samples=[
            MetricSample(
                labels={"model_name": "meta-llama/Llama-3.1-8B-Instruct"},
                value=0.42,
            )
        ],
    )


@pytest.fixture
def sample_histogram_metric() -> MetricFamily:
    """Sample histogram metric family."""
    return MetricFamily(
        type=PrometheusMetricType.HISTOGRAM,
        help="Time to first token",
        samples=[
            MetricSample(
                labels={"model_name": "meta-llama/Llama-3.1-8B-Instruct"},
                histogram=HistogramData(
                    buckets={"0.001": 0.0, "0.005": 5.0, "0.01": 15.0, "+Inf": 150.0},
                    sum=125.5,
                    count=150.0,
                ),
            )
        ],
    )


@pytest.fixture
def sample_summary_metric() -> MetricFamily:
    """Sample summary metric family."""
    return MetricFamily(
        type=PrometheusMetricType.SUMMARY,
        help="Request latency quantiles",
        samples=[
            MetricSample(
                labels={"model_name": "test-model"},
                summary=SummaryData(
                    quantiles={"0.5": 0.1, "0.9": 0.5, "0.99": 1.0},
                    sum=50.0,
                    count=100.0,
                ),
            )
        ],
    )


@pytest.fixture
def sample_server_metrics_record(
    sample_counter_metric: MetricFamily,
    sample_gauge_metric: MetricFamily,
    sample_histogram_metric: MetricFamily,
) -> ServerMetricsRecord:
    """Sample ServerMetricsRecord with multiple metric types."""
    return ServerMetricsRecord(
        endpoint_url="http://localhost:8081/metrics",
        timestamp_ns=1_000_000_000,
        endpoint_latency_ns=5_000_000,
        metrics={
            "vllm:request_success_total": sample_counter_metric,
            "vllm:gpu_cache_usage_perc": sample_gauge_metric,
            "vllm:time_to_first_token_seconds": sample_histogram_metric,
        },
    )

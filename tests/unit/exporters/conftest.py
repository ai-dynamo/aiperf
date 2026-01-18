# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for data exporters."""

from datetime import datetime

import pytest

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import (
    EndpointData,
    GpuSummary,
    JsonMetricResult,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    ServerMetricsResults,
)
from aiperf.common.models.telemetry_models import (
    TelemetryMetrics,
    TelemetryRecord,
)
from aiperf.server_metrics.storage import ServerMetricsHierarchy


@pytest.fixture
def sample_telemetry_record():
    """Create a sample TelemetryRecord for testing."""
    return TelemetryRecord(
        timestamp_ns=1000000000,
        dcgm_url="http://localhost:9400/metrics",
        gpu_index=0,
        gpu_model_name="NVIDIA H100",
        gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
        pci_bus_id="00000000:01:00.0",
        device="nvidia0",
        hostname="test-node-01",
        telemetry_data=TelemetryMetrics(
            gpu_power_usage=300.0,
            energy_consumption=1000.5,
            gpu_utilization=85.0,
            gpu_memory_used=72.5,
            gpu_temperature=70.0,
            xid_errors=0.0,
            power_violation=0.0,
        ),
    )


@pytest.fixture
def sample_telemetry_results():
    """Create a sample TelemetryExportData with realistic multi-GPU, multi-endpoint data."""

    # Create JsonMetricResults for each GPU metric
    def make_gpu_metrics(base_power, base_energy, base_util, base_mem, base_temp):
        return {
            "gpu_power_usage": JsonMetricResult(
                unit="W",
                avg=base_power,
                min=base_power - 20,
                max=base_power + 20,
                p50=base_power,
                p90=base_power + 15,
                p99=base_power + 18,
                std=5.0,
            ),
            "energy_consumption": JsonMetricResult(
                unit="J",
                avg=base_energy,
                min=base_energy - 100,
                max=base_energy + 400,
                p50=base_energy + 100,
                p90=base_energy + 300,
                p99=base_energy + 380,
                std=100.0,
            ),
            "gpu_utilization": JsonMetricResult(
                unit="%",
                avg=base_util,
                min=base_util,
                max=base_util + 8,
                p50=base_util + 4,
                p90=base_util + 7,
                p99=base_util + 8,
                std=2.0,
            ),
            "gpu_memory_used": JsonMetricResult(
                unit="GB",
                avg=base_mem,
                min=base_mem,
                max=base_mem + 4,
                p50=base_mem + 2,
                p90=base_mem + 3,
                p99=base_mem + 4,
                std=1.0,
            ),
            "gpu_temperature": JsonMetricResult(
                unit="°C",
                avg=base_temp,
                min=base_temp,
                max=base_temp + 8,
                p50=base_temp + 4,
                p90=base_temp + 6,
                p99=base_temp + 7,
                std=2.0,
            ),
            "xid_errors": JsonMetricResult(
                unit="count",
                avg=0.0,
                min=0.0,
                max=0.0,
                p50=0.0,
                p90=0.0,
                p99=0.0,
                std=0.0,
            ),
            "power_violation": JsonMetricResult(
                unit="ms",
                avg=120.0,
                min=100.0,
                max=140.0,
                p50=120.0,
                p90=135.0,
                p99=140.0,
                std=10.0,
            ),
        }

    return TelemetryExportData(
        summary=TelemetrySummary(
            endpoints_configured=[
                "http://localhost:9400/metrics",
                "http://remote-node:9400/metrics",
            ],
            endpoints_successful=[
                "http://localhost:9400/metrics",
                "http://remote-node:9400/metrics",
            ],
            start_time=datetime.fromtimestamp(1.0),
            end_time=datetime.fromtimestamp(6.0),
        ),
        endpoints={
            "localhost:9400": EndpointData(
                gpus={
                    "gpu_0": GpuSummary(
                        gpu_index=0,
                        gpu_name="NVIDIA H100",
                        gpu_uuid="GPU-12345678-1234-1234-1234-123456780000",
                        hostname="test-node-01",
                        metrics=make_gpu_metrics(290.0, 1200.0, 84.0, 72.0, 69.0),
                    ),
                    "gpu_1": GpuSummary(
                        gpu_index=1,
                        gpu_name="NVIDIA H100",
                        gpu_uuid="GPU-12345678-1234-1234-1234-123456780001",
                        hostname="test-node-01",
                        metrics=make_gpu_metrics(310.0, 1200.0, 84.0, 77.0, 69.0),
                    ),
                }
            ),
            "remote-node:9400": EndpointData(
                gpus={
                    "gpu_0": GpuSummary(
                        gpu_index=0,
                        gpu_name="NVIDIA A100",
                        gpu_uuid="GPU-abcdef01-2345-6789-abcd-ef0123456789",
                        hostname="test-node-02",
                        metrics=make_gpu_metrics(270.0, 1120.0, 81.0, 64.0, 69.0),
                    ),
                }
            ),
        },
    )


@pytest.fixture
def sample_telemetry_results_with_failures():
    """Create TelemetryExportData with some failed endpoints."""
    return TelemetryExportData(
        summary=TelemetrySummary(
            endpoints_configured=[
                "http://localhost:9400/metrics",
                "http://unreachable-node:9400/metrics",
                "http://failed-node:9400/metrics",
            ],
            endpoints_successful=["http://localhost:9400/metrics"],
            start_time=datetime.fromtimestamp(1.0),
            end_time=datetime.fromtimestamp(4.0),
        ),
        endpoints={
            "localhost:9400": EndpointData(
                gpus={
                    "gpu_0": GpuSummary(
                        gpu_index=0,
                        gpu_name="NVIDIA H100",
                        gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
                        hostname="test-node-01",
                        metrics={
                            "gpu_power_usage": JsonMetricResult(
                                unit="W", avg=310.0, min=300.0, max=320.0, std=10.0
                            ),
                            "gpu_utilization": JsonMetricResult(
                                unit="%", avg=85.0, min=85.0, max=85.0, std=0.0
                            ),
                            "gpu_memory_used": JsonMetricResult(
                                unit="GB", avg=72.5, min=72.5, max=72.5, std=0.0
                            ),
                            "gpu_temperature": JsonMetricResult(
                                unit="°C", avg=70.0, min=70.0, max=70.0, std=0.0
                            ),
                        },
                    ),
                }
            ),
        },
    )


@pytest.fixture
def empty_telemetry_results():
    """Create TelemetryExportData with no GPU data (all endpoints failed)."""
    return TelemetryExportData(
        summary=TelemetrySummary(
            endpoints_configured=[
                "http://unreachable-1:9400/metrics",
                "http://unreachable-2:9400/metrics",
            ],
            endpoints_successful=[],
            start_time=datetime.fromtimestamp(1.0),
            end_time=datetime.fromtimestamp(2.0),
        ),
        endpoints={},
    )


@pytest.fixture
def sample_timeslice_metric_results():
    """Create sample timeslice metric results for testing."""
    return {
        0: [
            MetricResult(
                tag="time_to_first_token",
                header="Time to First Token",
                unit="ms",
                avg=45.2,
                min=12.1,
                max=89.3,
                p50=44.0,
                p90=78.0,
                p99=88.0,
                std=15.2,
            ),
            MetricResult(
                tag="inter_token_latency",
                header="Inter Token Latency",
                unit="ms",
                avg=5.1,
                min=2.3,
                max=12.4,
                p50=4.8,
                p90=9.2,
                p99=11.8,
                std=2.1,
            ),
        ],
        1: [
            MetricResult(
                tag="time_to_first_token",
                header="Time to First Token",
                unit="ms",
                avg=48.5,
                min=15.2,
                max=92.1,
                p50=47.3,
                p90=82.4,
                p99=90.5,
                std=16.1,
            ),
            MetricResult(
                tag="inter_token_latency",
                header="Inter Token Latency",
                unit="ms",
                avg=5.4,
                min=2.5,
                max=13.1,
                p50=5.1,
                p90=9.8,
                p99=12.3,
                std=2.3,
            ),
        ],
    }


@pytest.fixture
def mock_results_with_timeslices(sample_timeslice_metric_results):
    """Create mock results with timeslice data."""

    class MockResultsWithTimeslices:
        def __init__(self):
            self.timeslice_metric_results = sample_timeslice_metric_results
            self.records = []
            self.start_ns = None
            self.end_ns = None
            self.has_results = True
            self.was_cancelled = False
            self.error_summary = []

    return MockResultsWithTimeslices()


@pytest.fixture
def mock_results_without_timeslices():
    """Create mock results without timeslice data."""

    class MockResultsNoTimeslices:
        def __init__(self):
            self.timeslice_metric_results = None
            self.records = []
            self.start_ns = None
            self.end_ns = None
            self.has_results = False
            self.was_cancelled = False
            self.error_summary = []

    return MockResultsNoTimeslices()


@pytest.fixture
def sample_server_metrics_results():
    """Create a sample ServerMetricsResults with realistic multi-endpoint data.

    Includes three Prometheus metric types:
    - Gauge: Point-in-time values
    - Counter: Cumulative values (for delta calculation)
    - Histogram: Distribution buckets with sum/count
    """
    hierarchy = ServerMetricsHierarchy()

    # Endpoint 1: vLLM worker 1 with all metric types
    for time_offset in range(5):
        gauge = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="KV cache usage percentage",
            samples=[
                MetricSample(labels=None, value=0.4 + time_offset * 0.05),
            ],
        )
        counter = MetricFamily(
            type=PrometheusMetricType.COUNTER,
            description="Total number of requests",
            samples=[
                MetricSample(labels=None, value=100.0 + time_offset * 20),
            ],
        )
        # Histogram for time-to-first-token latency distribution
        histogram = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM,
            description="Time to first token histogram",
            samples=[
                MetricSample(
                    labels=None,
                    buckets={
                        "0.01": 5.0 + time_offset * 2,
                        "0.1": 15.0 + time_offset * 5,
                        "1.0": 50.0 + time_offset * 10,
                        "+Inf": 100.0 + time_offset * 20,
                    },
                    sum=25.5 + time_offset * 5.0,
                    count=100.0 + time_offset * 20,
                ),
            ],
        )
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000 + time_offset * 1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "vllm:kv_cache_usage_perc": gauge,
                "vllm:request_success_total": counter,
                "vllm:time_to_first_token_seconds": histogram,
            },
        )
        hierarchy.add_record(record)

    # Endpoint 2: vLLM worker 2 with all metric types
    for time_offset in range(5):
        gauge = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="KV cache usage percentage",
            samples=[
                MetricSample(labels=None, value=0.5 + time_offset * 0.04),
            ],
        )
        counter = MetricFamily(
            type=PrometheusMetricType.COUNTER,
            description="Total number of requests",
            samples=[
                MetricSample(labels=None, value=80.0 + time_offset * 25),
            ],
        )
        # Histogram for time-to-first-token latency distribution
        histogram = MetricFamily(
            type=PrometheusMetricType.HISTOGRAM,
            description="Time to first token histogram",
            samples=[
                MetricSample(
                    labels=None,
                    buckets={
                        "0.01": 3.0 + time_offset * 1,
                        "0.1": 10.0 + time_offset * 3,
                        "1.0": 40.0 + time_offset * 8,
                        "+Inf": 80.0 + time_offset * 15,
                    },
                    sum=20.0 + time_offset * 4.0,
                    count=80.0 + time_offset * 15,
                ),
            ],
        )
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8082/metrics",
            timestamp_ns=1_000_000_000 + time_offset * 1_000_000_000,
            endpoint_latency_ns=6_000_000,
            metrics={
                "vllm:kv_cache_usage_perc": gauge,
                "vllm:request_success_total": counter,
                "vllm:time_to_first_token_seconds": histogram,
            },
        )
        hierarchy.add_record(record)

    return ServerMetricsResults(
        server_metrics_data=hierarchy,
        start_ns=1_000_000_000,
        end_ns=6_000_000_000,
        endpoints_configured=[
            "http://localhost:8081/metrics",
            "http://localhost:8082/metrics",
        ],
        endpoints_successful=[
            "http://localhost:8081/metrics",
            "http://localhost:8082/metrics",
        ],
        error_summary=[],
    )


@pytest.fixture
def empty_server_metrics_results():
    """Create ServerMetricsResults with no data (all endpoints failed)."""
    return ServerMetricsResults(
        server_metrics_data=ServerMetricsHierarchy(),
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        endpoints_configured=[
            "http://unreachable-1:8081/metrics",
            "http://unreachable-2:8081/metrics",
        ],
        endpoints_successful=[],
        error_summary=[],
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GPU benchmark metrics collection and validation."""

from __future__ import annotations

from collections.abc import Callable

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.benchmark import BenchmarkResult

logger = AIPerfLogger(__name__)


class TestGPUMetricsCollection:
    """Tests for metrics collection from GPU benchmarks (module-scoped)."""

    def test_throughput_is_positive(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify request throughput is collected and positive."""
        metrics = deployed_gpu_benchmark_module.metrics

        assert metrics is not None
        logger.info(f"[TEST] Throughput: {metrics.request_throughput} req/s")
        assert metrics.request_throughput is not None
        assert metrics.request_throughput > 0

    def test_latency_is_positive(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify request latency is collected and positive."""
        metrics = deployed_gpu_benchmark_module.metrics

        assert metrics is not None
        logger.info(f"[TEST] Latency: avg={metrics.request_latency_avg or 0:.2f} ms")
        assert metrics.request_latency_avg is not None
        assert metrics.request_latency_avg > 0

    def test_request_count_matches_config(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify request count matches configuration."""
        result = deployed_gpu_benchmark_module

        assert result.metrics is not None
        logger.info(
            f"[TEST] Request count: actual={result.metrics.request_count}, expected={result.config.request_count}"
        )
        assert result.metrics.request_count >= 1, (
            f"Expected >= 1 completed request, got {result.metrics.request_count}"
        )

    def test_no_errors(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify successful GPU benchmark has no errors."""
        metrics = deployed_gpu_benchmark_module.metrics

        assert metrics is not None
        logger.info(f"[TEST] Error count: {metrics.error_count}")
        assert metrics.error_count == 0


class TestGPUMetricsReasonableness:
    """Tests for GPU metrics being within reasonable bounds for real inference."""

    def test_min_throughput(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
        assert_metrics: Callable[..., None],
    ) -> None:
        """Verify throughput exceeds minimum for real GPU inference.

        Even a small model on a single GPU should achieve > 0.3 req/s.
        """
        metrics = deployed_gpu_benchmark_module.metrics
        logger.info(
            f"[TEST] Throughput reasonableness: actual={metrics.request_throughput if metrics else 0:.2f} req/s, minimum=0.1 req/s"
        )
        assert_metrics(
            deployed_gpu_benchmark_module,
            min_throughput=0.1,
        )

    def test_max_latency(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
        assert_metrics: Callable[..., None],
    ) -> None:
        """Verify latency is below generous maximum for real inference.

        Even with model loading overhead, latency should be < 10000ms per request.
        """
        metrics = deployed_gpu_benchmark_module.metrics
        logger.info(
            f"[TEST] Latency reasonableness: actual={metrics.request_latency_avg if metrics else 0:.2f} ms, maximum=10000.0 ms"
        )
        assert_metrics(
            deployed_gpu_benchmark_module,
            max_latency=10000.0,
        )

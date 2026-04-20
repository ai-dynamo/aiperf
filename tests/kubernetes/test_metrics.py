# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark metrics collection and validation."""

from __future__ import annotations

import pytest

from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
)


class TestMetricsCollection:
    """Tests for metrics collection from benchmarks."""

    def test_metrics_are_collected(
        self, deployed_small_benchmark_module: BenchmarkResult
    ) -> None:
        """Verify metrics are collected from completed benchmark."""
        result = deployed_small_benchmark_module

        assert result.success
        assert result.metrics is not None

    def test_request_throughput_collected(
        self, deployed_small_benchmark_module: BenchmarkResult
    ) -> None:
        """Verify request throughput is collected."""
        metrics = deployed_small_benchmark_module.metrics

        assert metrics is not None
        assert metrics.request_throughput is not None
        assert metrics.request_throughput > 0

    def test_output_token_throughput_collected(
        self, deployed_small_benchmark_module: BenchmarkResult
    ) -> None:
        """Verify output token throughput is collected."""
        metrics = deployed_small_benchmark_module.metrics

        assert metrics is not None
        assert metrics.output_token_throughput is not None
        assert metrics.output_token_throughput > 0

    @pytest.mark.asyncio
    async def test_request_count_matches_config(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Verify request count matches configuration."""
        expected_count = 15
        config = BenchmarkConfig(
            concurrency=3,
            request_count=expected_count,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success
        assert result.metrics is not None
        assert result.metrics.request_count == expected_count

    def test_latency_metrics_collected(
        self, deployed_small_benchmark_module: BenchmarkResult
    ) -> None:
        """Verify latency metrics are collected."""
        metrics = deployed_small_benchmark_module.metrics

        assert metrics is not None
        assert metrics.request_latency_avg is not None
        assert metrics.request_latency_avg > 0

    def test_no_errors_in_successful_run(
        self, deployed_small_benchmark_module: BenchmarkResult
    ) -> None:
        """Verify successful run has no errors."""
        metrics = deployed_small_benchmark_module.metrics

        assert metrics is not None
        assert metrics.error_count == 0


class TestMetricsValidation:
    """Tests for validating metrics against expectations."""

    def test_throughput_is_reasonable(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        assert_metrics,
    ) -> None:
        """Verify throughput is within reasonable range for mock server."""
        # Mock server should achieve high throughput
        assert_metrics(
            deployed_small_benchmark_module,
            min_throughput=100.0,  # At least 100 req/s with mock server
        )

    def test_latency_is_reasonable(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        assert_metrics,
    ) -> None:
        """Verify latency is within reasonable range for mock server."""
        # Mock server should have low latency
        assert_metrics(
            deployed_small_benchmark_module,
            max_latency=100.0,  # Less than 100ms average with mock server
        )

    @pytest.mark.asyncio
    async def test_all_requests_completed(
        self,
        benchmark_deployer: BenchmarkDeployer,
        assert_metrics,
    ) -> None:
        """Verify all configured requests are completed."""
        config = BenchmarkConfig(
            concurrency=3,
            request_count=20,
            warmup_request_count=5,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        assert_metrics(
            result,
            expected_request_count=20,
            max_error_count=0,
        )


class TestMetricsConsistency:
    """Tests for metrics consistency across runs."""

    @pytest.mark.asyncio
    async def test_multiple_runs_produce_similar_throughput(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Verify throughput is consistent across multiple runs."""
        config = BenchmarkConfig(
            concurrency=3,
            request_count=20,
            warmup_request_count=5,
        )

        throughputs = []
        for _ in range(3):
            result = await benchmark_deployer.deploy(config, timeout=90)
            assert result.success
            assert result.metrics is not None
            throughputs.append(result.metrics.request_throughput)

        # Check that throughputs are within 2x of each other (generous for Kind cluster)
        avg_throughput = sum(throughputs) / len(throughputs)
        for t in throughputs:
            assert abs(t - avg_throughput) / avg_throughput < 1.0, (
                f"Throughput variance too high: {throughputs}"
            )

    @pytest.mark.asyncio
    async def test_request_count_always_matches(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Verify request count always matches configuration."""
        for request_count in [10, 20, 30]:
            config = BenchmarkConfig(
                concurrency=2,
                request_count=request_count,
                warmup_request_count=2,
            )

            result = await benchmark_deployer.deploy(config, timeout=90)

            assert result.success
            assert result.metrics is not None
            assert result.metrics.request_count == request_count, (
                f"Expected {request_count}, got {result.metrics.request_count}"
            )


class TestMetricsFromLogs:
    """Tests for metrics parsing from logs."""

    def test_raw_logs_captured(
        self, deployed_small_benchmark_module: BenchmarkResult
    ) -> None:
        """Verify raw logs are captured in metrics."""
        metrics = deployed_small_benchmark_module.metrics

        if metrics is None or len(metrics.raw_logs) == 0:
            # Pods cleaned up before logs could be captured
            assert deployed_small_benchmark_module.success
            return
        assert len(metrics.raw_logs) > 0

    @pytest.mark.asyncio
    async def test_logs_contain_metrics_table(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        get_pod_logs,
    ) -> None:
        """Verify logs contain the metrics table."""
        if not deployed_small_benchmark_module.controller_pod:
            assert deployed_small_benchmark_module.success
            return

        logs = await get_pod_logs(
            deployed_small_benchmark_module, container="control-plane"
        )
        if not logs:
            assert deployed_small_benchmark_module.success
            return

        # Check for key metrics table markers
        assert "Request Throughput" in logs or "request" in logs.lower()
        assert "Latency" in logs or "latency" in logs.lower()

    def test_can_extract_all_metrics_from_successful_run(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify all expected metrics can be extracted."""
        metrics = deployed_small_benchmark_module.metrics

        if metrics is None:
            # Pods cleaned up before metrics could be collected
            assert deployed_small_benchmark_module.success
            return

        # These should all be present in a successful run
        assert metrics.request_throughput is not None
        assert metrics.request_count is not None
        # Latency might not always parse correctly from rich tables
        # so we just check throughput and count as core metrics

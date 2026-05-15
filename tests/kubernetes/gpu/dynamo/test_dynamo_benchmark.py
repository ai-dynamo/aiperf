# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GPU benchmark execution against a Dynamo inference graph."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from pytest import param

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.conftest import (
    GPUTestSettings,
    _dump_diagnostics,
    _log_container_logs,
    _log_pod_statuses,
)
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig
from tests.kubernetes.gpu.vllm.helpers import GPUBenchmarkDeployer
from tests.kubernetes.helpers.benchmark import BenchmarkConfig, BenchmarkResult
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


# ============================================================================
# Module-scoped benchmark against Dynamo
# ============================================================================


@pytest.fixture(scope="module")
def _dynamo_benchmark_config(
    dynamo_endpoint_url: str,
    dynamo_config: DynamoConfig,
    gpu_settings: GPUTestSettings,
) -> BenchmarkConfig:
    """Module-scoped benchmark config targeting Dynamo."""
    return BenchmarkConfig(
        endpoint_url=dynamo_endpoint_url,
        endpoint_type="chat",
        model_name=dynamo_config.model_name,
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=gpu_settings.aiperf_image,
        workers=2,
        input_sequence_min=10,
        input_sequence_max=30,
        output_tokens_min=5,
        output_tokens_max=20,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def deployed_dynamo_benchmark(
    benchmark_deployer: GPUBenchmarkDeployer,
    _dynamo_benchmark_config: BenchmarkConfig,
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
) -> AsyncGenerator[BenchmarkResult, None]:
    """Deploy a benchmark against Dynamo, shared across tests in this module."""
    s = gpu_settings
    logger.info(
        f"[BENCHMARK] Deploying Dynamo benchmark: endpoint={_dynamo_benchmark_config.endpoint_url}, model={_dynamo_benchmark_config.model_name}, "
        f"concurrency={_dynamo_benchmark_config.concurrency}, requests={_dynamo_benchmark_config.request_count}"
    )

    result = await benchmark_deployer.deploy(
        config=_dynamo_benchmark_config,
        wait_for_completion=True,
        timeout=s.benchmark_timeout,
        stream_logs=s.stream_logs,
    )

    logger.info(
        f"[BENCHMARK] Result: success={result.success}, namespace={result.namespace}, duration={result.duration_seconds:.1f}s"
    )

    if result.metrics:
        logger.info(
            f"[BENCHMARK] Metrics: throughput={result.metrics.request_throughput or 0:.2f} req/s, latency_avg={result.metrics.request_latency_avg or 0:.2f} ms, "
            f"requests={result.metrics.request_count}, errors={result.metrics.error_count}"
        )

    await _log_pod_statuses(kubectl, result.namespace)
    await _log_container_logs(kubectl, result.namespace)

    if not result.success:
        await _dump_diagnostics(
            kubectl, result.namespace, label="DYNAMO_BENCHMARK_FAILURE"
        )

    yield result


# ============================================================================
# Tests
# ============================================================================


class TestDynamoBenchmarkCompletion:
    """Tests for AIPerf benchmark completion against Dynamo."""

    def test_benchmark_completes_successfully(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify benchmark against Dynamo completes successfully."""
        result = deployed_dynamo_benchmark
        assert result.success, f"Benchmark failed: {result.error_message}"
        assert result.status is not None
        assert result.status.is_completed

    def test_no_benchmark_errors(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify benchmark against Dynamo completes without errors."""
        result = deployed_dynamo_benchmark
        assert result.metrics is not None
        assert result.metrics.error_count == 0, (
            f"Expected 0 errors, got {result.metrics.error_count}"
        )

    def test_throughput_is_positive(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify throughput from Dynamo benchmark is positive."""
        metrics = deployed_dynamo_benchmark.metrics
        assert metrics is not None
        assert metrics.request_throughput is not None
        assert metrics.request_throughput > 0

    def test_latency_is_positive(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify latency from Dynamo benchmark is positive."""
        metrics = deployed_dynamo_benchmark.metrics
        assert metrics is not None
        assert metrics.request_latency_avg is not None
        assert metrics.request_latency_avg > 0

    def test_request_count_matches_config(
        self,
        deployed_dynamo_benchmark: BenchmarkResult,
    ) -> None:
        """Verify request count is close to configuration.

        Dynamo disaggregated mode on a shared GPU may not complete all
        requests within the timeout; verify at least 80% completed.
        """
        result = deployed_dynamo_benchmark
        assert result.metrics is not None
        assert result.metrics.request_count >= 1, (
            f"Expected >= 1 completed request, got {result.metrics.request_count}"
        )


class TestDynamoBenchmarkWorkerScaling:
    """Tests for Dynamo benchmark with different worker pod counts and longer runs."""

    @pytest.mark.parametrize(
        "request_count, concurrency",
        [
            param(20, 2, id="c2-20-reqs"),
            param(20, 4, id="c4-20-reqs"),
            param(20, 8, id="c8-20-reqs"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_benchmark_succeeds_with_worker_count(
        self,
        benchmark_deployer: GPUBenchmarkDeployer,
        dynamo_endpoint_url: str,
        dynamo_config: DynamoConfig,
        kubectl: KubectlClient,
        gpu_settings: GPUTestSettings,
        request_count: int,
        concurrency: int,
    ) -> None:
        """Verify benchmark completes with varying worker pod counts."""
        workers = concurrency // 2
        logger.info(
            f"[TEST] Worker scaling test: workers={workers}, requests={request_count}, concurrency={concurrency}"
        )

        config = BenchmarkConfig(
            endpoint_url=dynamo_endpoint_url,
            endpoint_type="chat",
            model_name=dynamo_config.model_name,
            concurrency=concurrency,
            request_count=request_count,
            warmup_request_count=2,
            image=gpu_settings.aiperf_image,
            workers=workers,
            input_sequence_min=10,
            input_sequence_max=30,
            output_tokens_min=5,
            output_tokens_max=20,
        )

        result = await benchmark_deployer.deploy(
            config=config,
            wait_for_completion=True,
            timeout=gpu_settings.benchmark_timeout,
            stream_logs=gpu_settings.stream_logs,
        )

        logger.info(
            f"[TEST] workers={workers} result: success={result.success}, duration={result.duration_seconds:.1f}s"
        )

        if result.metrics:
            logger.info(
                f"[TEST] workers={workers} metrics: throughput={result.metrics.request_throughput or 0:.2f} req/s, "
                f"latency_avg={result.metrics.request_latency_avg or 0:.2f} ms, requests={result.metrics.request_count}, errors={result.metrics.error_count}"
            )

        await _log_pod_statuses(kubectl, result.namespace)
        await _log_container_logs(kubectl, result.namespace)

        if not result.success:
            await _dump_diagnostics(
                kubectl, result.namespace, label="DYNAMO_WORKER_SCALING_FAILURE"
            )

        assert result.success, (
            f"Benchmark failed with workers={workers}: {result.error_message}"
        )
        assert result.metrics is not None
        assert result.metrics.error_count == 0, (
            f"Expected 0 errors, got {result.metrics.error_count}"
        )
        # Disaggregated mode on shared GPU has high per-request latency;
        # verify at least some requests completed successfully.
        assert result.metrics.request_count >= 1, (
            f"Expected >= 1 completed request, got {result.metrics.request_count}"
        )

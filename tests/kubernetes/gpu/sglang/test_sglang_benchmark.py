# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GPU benchmark execution, lifecycle, and concurrency scaling with SGLang."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.conftest import (
    GPUTestSettings,
    _dump_diagnostics,
    _log_container_logs,
    _log_pod_statuses,
)
from tests.kubernetes.gpu.sglang.helpers import SGLangConfig
from tests.kubernetes.gpu.vllm.helpers import GPUBenchmarkDeployer
from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


class TestGPUBenchmarkCompletion:
    """Tests for GPU benchmark completion with SGLang (module-scoped for speed)."""

    def test_benchmark_completes_successfully(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify GPU benchmark completes successfully."""
        result = deployed_gpu_benchmark_module

        logger.info(
            f"[TEST] Checking completion: success={result.success}, status={result.status.terminal_state if result.status else 'None'}, duration={result.duration_seconds:.1f}s"
        )

        assert result.success, f"Benchmark failed: {result.error_message}"
        if result.status is not None:
            assert result.status.is_completed

    def test_jobset_reaches_completed_state(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify JobSet reaches Completed terminal state or operator cleaned up."""
        result = deployed_gpu_benchmark_module
        status = result.status

        logger.info(
            f"[TEST] JobSet state: terminal_state={status.terminal_state if status else 'None'}, completed={status.completed if status else 'None'}, restarts={status.restarts if status else 0}"
        )

        if status is None:
            assert result.success, "No JobSet status and benchmark did not succeed"
        else:
            assert status.terminal_state == "Completed"

    def test_all_pods_complete(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify all pods complete successfully (or operator cleaned them up)."""
        result = deployed_gpu_benchmark_module
        pods = [p for p in result.pods if result.job_id in p.name]

        logger.info(f"[TEST] Pod count: {len(pods)}")
        for pod in pods:
            logger.info(
                f"  {pod.name:<55} phase={pod.phase:<12} ready={pod.ready} restarts={pod.restarts}"
            )

        # If the operator already cleaned up, check via result success
        if not pods or result.success:
            assert result.success, "Benchmark did not succeed"
            return

        for pod in pods:
            assert pod.is_completed or pod.phase == "Succeeded", (
                f"Pod {pod.name} not completed: phase={pod.phase}"
            )

    def test_no_benchmark_errors(
        self,
        deployed_gpu_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify benchmark completes without errors."""
        result = deployed_gpu_benchmark_module

        if result.metrics:
            logger.info(
                f"[TEST] Error check: errors={result.metrics.error_count}, request_count={result.metrics.request_count}"
            )

        assert result.metrics is not None
        assert result.metrics.error_count == 0, (
            f"Expected 0 errors, got {result.metrics.error_count}"
        )


class TestGPUBenchmarkLifecycle:
    """Tests for GPU benchmark lifecycle management with SGLang (function-scoped)."""

    @pytest.mark.asyncio
    async def test_benchmark_creates_namespace(
        self,
        benchmark_deployer: GPUBenchmarkDeployer,
        small_gpu_benchmark_config: BenchmarkConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Verify benchmark creates its own namespace."""
        logger.info(
            f"[TEST] Deploying lifecycle test benchmark: concurrency={small_gpu_benchmark_config.concurrency}, requests={small_gpu_benchmark_config.request_count}"
        )

        result = await benchmark_deployer.deploy(
            config=small_gpu_benchmark_config,
            wait_for_completion=True,
            timeout=600,
        )

        logger.info(
            f"[TEST] Lifecycle result: namespace={result.namespace}, success={result.success}, duration={result.duration_seconds:.1f}s"
        )
        await _log_pod_statuses(kubectl, result.namespace)
        await _log_container_logs(kubectl, result.namespace, tail=50)

        if not result.success:
            await _dump_diagnostics(
                kubectl, result.namespace, label="LIFECYCLE_FAILURE"
            )

        assert result.namespace
        assert await kubectl.namespace_exists(result.namespace)

    @pytest.mark.asyncio
    async def test_cleanup_removes_namespace(
        self,
        benchmark_deployer: GPUBenchmarkDeployer,
        small_gpu_benchmark_config: BenchmarkConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Verify cleanup removes the benchmark namespace."""
        result = await benchmark_deployer.deploy(
            config=small_gpu_benchmark_config,
            wait_for_completion=True,
            timeout=600,
        )

        namespace = result.namespace
        logger.info(f"[TEST] Verifying cleanup for namespace: {namespace}")
        assert await kubectl.namespace_exists(namespace)

        await _log_pod_statuses(kubectl, namespace)
        await benchmark_deployer.cleanup(result)

        assert not await kubectl.namespace_exists(namespace)
        logger.info(f"[TEST] Namespace {namespace} cleaned up successfully")


class TestGPUBenchmarkConcurrency:
    """Tests for GPU benchmark with different concurrency levels against SGLang."""

    @pytest.mark.parametrize(
        "concurrency",
        [
            param(1, id="concurrency-1"),
            param(2, id="concurrency-2"),
            param(4, id="concurrency-4"),
            param(8, id="concurrency-8"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_benchmark_succeeds_at_concurrency_level(
        self,
        benchmark_deployer: GPUBenchmarkDeployer,
        sglang_endpoint_url: str,
        sglang_config: SGLangConfig,
        kubectl: KubectlClient,
        gpu_settings: GPUTestSettings,
        concurrency: int,
    ) -> None:
        """Verify benchmark completes at various concurrency levels."""
        logger.info(
            f"[TEST] Concurrency test: level={concurrency}, model={sglang_config.model_name}"
        )

        config = BenchmarkConfig(
            endpoint_url=sglang_endpoint_url,
            endpoint_type="chat",
            model_name=sglang_config.model_name,
            concurrency=concurrency,
            request_count=10,
            warmup_request_count=2,
            image=gpu_settings.aiperf_image,
            workers=2,
            input_sequence_min=10,
            input_sequence_max=30,
            output_tokens_min=5,
            output_tokens_max=20,
        )

        result = await benchmark_deployer.deploy(
            config=config,
            wait_for_completion=True,
            timeout=600,
        )

        logger.info(
            f"[TEST] Concurrency={concurrency} result: success={result.success}, duration={result.duration_seconds:.1f}s"
        )

        if result.metrics:
            logger.info(
                f"[TEST] Concurrency={concurrency} metrics: throughput={result.metrics.request_throughput or 0:.2f} req/s, "
                f"latency_avg={result.metrics.request_latency_avg or 0:.2f} ms, requests={result.metrics.request_count}, errors={result.metrics.error_count}"
            )

        await _log_pod_statuses(kubectl, result.namespace)

        if not result.success:
            await _dump_diagnostics(
                kubectl, result.namespace, label="CONCURRENCY_FAILURE"
            )

        assert result.success, (
            f"Benchmark failed at concurrency={concurrency}: {result.error_message}"
        )
        assert result.metrics is not None
        assert result.metrics.request_count >= 1, (
            f"Expected >= 1 completed request, got {result.metrics.request_count}"
        )

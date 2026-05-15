# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for TRT-LLM GPU E2E tests.

Provides TRT-LLM server deployment, endpoint URL resolution, and benchmark
configuration fixtures used by all tests in the ``gpu/trtllm/`` subtree.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.conftest import (
    GPUTestSettings,
    _dump_diagnostics,
    _log_container_logs,
    _log_pod_statuses,
    _release_gpu,
)
from tests.kubernetes.gpu.trtllm.helpers import TRTLLMConfig, TRTLLMDeployer
from tests.kubernetes.gpu.vllm.helpers import GPUBenchmarkDeployer
from tests.kubernetes.helpers.benchmark import BenchmarkConfig, BenchmarkResult
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


# ============================================================================
# TRT-LLM deployment fixtures
# ============================================================================


@pytest.fixture(scope="package")
def trtllm_config(gpu_settings: GPUTestSettings) -> TRTLLMConfig:
    """Create TRT-LLM configuration from settings."""
    s = gpu_settings
    return TRTLLMConfig(
        image=s.trtllm_image,
        model_name=s.model,
        gpu_count=s.count,
        max_model_len=s.max_model_len,
        tolerations=s.tolerations,
        node_selector=s.node_selector,
        hf_token_secret=s.hf_token_secret,
        image_pull_secrets=s.image_pull_secrets,
        tensor_parallel_size=s.count,
        runtime_class_name=s.runtime_class,
    )


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def trtllm_server(
    kubectl: KubectlClient,
    trtllm_config: TRTLLMConfig,
    gpu_settings: GPUTestSettings,
) -> AsyncGenerator[TRTLLMDeployer | str, None]:
    """Deploy TRT-LLM server or use existing endpoint.

    If --gpu-trtllm-endpoint is set, skips deployment and yields the URL.
    Otherwise deploys TRT-LLM and yields the deployer.
    """
    s = gpu_settings
    if s.trtllm_endpoint:
        logger.info(f"Using existing TRT-LLM endpoint: {s.trtllm_endpoint}")
        yield s.trtllm_endpoint
        return

    await _release_gpu(kubectl, trtllm_config.namespace)
    deployer = TRTLLMDeployer(kubectl=kubectl, config=trtllm_config)

    logger.info(
        f"Deploying TRT-LLM server: model={trtllm_config.model_name}, image={trtllm_config.image}, gpus={trtllm_config.gpu_count}, namespace={trtllm_config.namespace}"
    )
    manifest = deployer.generate_manifest()
    logger.debug(lambda manifest=manifest: f"[TRTLLM] Generated manifest:\n{manifest}")

    await deployer.deploy()

    logger.info(
        f"[TRTLLM] Waiting for TRT-LLM readiness (timeout={s.trtllm_deploy_timeout}s)..."
    )
    try:
        await deployer.wait_for_ready(
            timeout=s.trtllm_deploy_timeout,
            stream_logs=s.stream_logs,
        )
        logger.info(f"[TRTLLM] Server is ready at {deployer.get_endpoint_url()}")
        trtllm_logs = await deployer.get_logs(tail=30)
        logger.info(f"[TRTLLM] Recent server logs:\n{trtllm_logs}")
    except TimeoutError:
        logger.error("[TRTLLM] Server failed to become ready!")
        await _dump_diagnostics(
            kubectl, trtllm_config.namespace, label="TRTLLM_FAILURE"
        )
        raise

    yield deployer

    if not s.skip_cleanup:
        logger.info(f"[TRTLLM] Cleaning up TRT-LLM namespace {trtllm_config.namespace}")
        await deployer.cleanup()
    else:
        logger.info("Skipping TRT-LLM cleanup (--gpu-skip-cleanup)")


@pytest.fixture(scope="package")
def trtllm_endpoint_url(
    trtllm_server: TRTLLMDeployer | str,
) -> str:
    """Get the TRT-LLM endpoint URL."""
    if isinstance(trtllm_server, str):
        return trtllm_server
    return trtllm_server.get_endpoint_url()


# ============================================================================
# Cluster readiness aggregator
# ============================================================================


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def gpu_cluster_ready(
    gpu_cluster_base: None,
    trtllm_server: TRTLLMDeployer | str,
) -> None:
    """Ensure all GPU cluster infrastructure is ready for TRT-LLM tests.

    Aggregates: gpu_cluster_base + trtllm_server.
    """
    logger.info("GPU cluster infrastructure ready (with TRT-LLM)")


# ============================================================================
# Benchmark configuration fixtures
# ============================================================================


@pytest.fixture
def gpu_benchmark_config(
    trtllm_endpoint_url: str,
    trtllm_config: TRTLLMConfig,
    gpu_settings: GPUTestSettings,
) -> BenchmarkConfig:
    """Create a GPU benchmark configuration."""
    s = gpu_settings
    return BenchmarkConfig(
        endpoint_url=trtllm_endpoint_url,
        endpoint_type="chat",
        model_name=trtllm_config.model_name,
        concurrency=4,
        request_count=20,
        warmup_request_count=2,
        image=s.aiperf_image,
        workers=2,
        input_sequence_min=10,
        input_sequence_max=50,
        output_tokens_min=10,
        output_tokens_max=50,
    )


@pytest.fixture
def small_gpu_benchmark_config(
    trtllm_endpoint_url: str,
    trtllm_config: TRTLLMConfig,
    gpu_settings: GPUTestSettings,
) -> BenchmarkConfig:
    """Create a small/fast GPU benchmark configuration."""
    s = gpu_settings
    return BenchmarkConfig(
        endpoint_url=trtllm_endpoint_url,
        endpoint_type="chat",
        model_name=trtllm_config.model_name,
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=s.aiperf_image,
        workers=2,
        input_sequence_min=10,
        input_sequence_max=30,
        output_tokens_min=5,
        output_tokens_max=20,
    )


# ============================================================================
# Module-scoped benchmark result (shared across read-only tests)
# ============================================================================


@pytest.fixture(scope="module")
def _gpu_benchmark_config_module(
    trtllm_endpoint_url: str,
    trtllm_config: TRTLLMConfig,
    gpu_settings: GPUTestSettings,
) -> BenchmarkConfig:
    """Module-scoped GPU benchmark configuration."""
    s = gpu_settings
    return BenchmarkConfig(
        endpoint_url=trtllm_endpoint_url,
        endpoint_type="chat",
        model_name=trtllm_config.model_name,
        concurrency=2,
        request_count=10,
        warmup_request_count=2,
        image=s.aiperf_image,
        workers=2,
        input_sequence_min=10,
        input_sequence_max=30,
        output_tokens_min=5,
        output_tokens_max=20,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def deployed_gpu_benchmark_module(
    benchmark_deployer: GPUBenchmarkDeployer,
    _gpu_benchmark_config_module: BenchmarkConfig,
    gpu_settings: GPUTestSettings,
) -> AsyncGenerator[BenchmarkResult, None]:
    """Deploy a GPU benchmark shared across tests in a module.

    Use this for read-only tests that only need to inspect results.
    """
    s = gpu_settings
    logger.info(
        f"[BENCHMARK] Deploying module benchmark: endpoint={_gpu_benchmark_config_module.endpoint_url}, model={_gpu_benchmark_config_module.model_name}, "
        f"concurrency={_gpu_benchmark_config_module.concurrency}, requests={_gpu_benchmark_config_module.request_count}, timeout={s.benchmark_timeout}s"
    )

    result = await benchmark_deployer.deploy(
        config=_gpu_benchmark_config_module,
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

    await _log_pod_statuses(benchmark_deployer.kubectl, result.namespace)
    await _log_container_logs(benchmark_deployer.kubectl, result.namespace)

    if not result.success:
        logger.error("[BENCHMARK] Benchmark failed! Dumping full diagnostics...")
        await _dump_diagnostics(
            benchmark_deployer.kubectl, result.namespace, label="BENCHMARK_FAILURE"
        )

    yield result

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for LoRA adapter benchmarks.

Scaffold only -- opt-in via ``AIPERF_RUN_LORA_GPU_TESTS=1``. Collection must
succeed without a cluster so the matrix runner can always discover the suite.
"""

from __future__ import annotations

import os

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.conftest import (
    _dump_diagnostics,
    _log_container_logs,
    _log_pod_statuses,
)
from tests.kubernetes.gpu.lora.helpers import LoraConfig, LoraDeployer
from tests.kubernetes.gpu.vllm.helpers import GPUBenchmarkDeployer
from tests.kubernetes.helpers.benchmark import BenchmarkConfig
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


# Module-level opt-in gate: skips execution cleanly while leaving collection
# intact for ``pytest --collect-only``.
pytestmark = pytest.mark.skipif(
    not os.environ.get("AIPERF_RUN_LORA_GPU_TESTS"),
    reason=(
        "LoRA GPU tests require AIPERF_RUN_LORA_GPU_TESTS=1 "
        "(needs a real Dynamo base model with a LoRA-capable engine)."
    ),
)


class TestLoraBenchmark:
    """Baseline E2E coverage for LoRA adapter benchmarks."""

    @pytest.mark.asyncio
    async def test_benchmark_runs_to_completion(
        self,
        lora_deployer: LoraDeployer,
        lora_config: LoraConfig,
        lora_benchmark_deployer: GPUBenchmarkDeployer,
        lora_benchmark_config: BenchmarkConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Deploy the LoRA adapter, run a short benchmark, assert nonzero work.

        Primary assertions:
          * benchmark reports success,
          * ``request_throughput`` is strictly positive,
          * ``request_count`` is strictly positive.
        """
        logger.info(
            f"[TEST] Deploying LoRA adapter: name={lora_config.adapter_name}, "
            f"base={lora_config.base_model}, source={lora_config.source}"
        )
        await lora_deployer.deploy()

        logger.info(
            f"[TEST] Running LoRA benchmark: endpoint={lora_benchmark_config.endpoint_url}, "
            f"model={lora_benchmark_config.model_name}, "
            f"concurrency={lora_benchmark_config.concurrency}, "
            f"requests={lora_benchmark_config.request_count}"
        )
        result = await lora_benchmark_deployer.deploy(
            config=lora_benchmark_config,
            wait_for_completion=True,
            timeout=600,
        )

        await _log_pod_statuses(kubectl, result.namespace)
        await _log_container_logs(kubectl, result.namespace, tail=50)

        if not result.success:
            await _dump_diagnostics(kubectl, result.namespace, label="LORA_FAILURE")

        assert result.success, f"LoRA benchmark failed: {result.error_message}"
        assert result.metrics is not None, "No metrics collected for LoRA benchmark"
        assert result.metrics.request_count > 0, (
            f"Expected > 0 completed requests, got {result.metrics.request_count}"
        )
        assert (result.metrics.request_throughput or 0) > 0, (
            f"Expected > 0 request_throughput, got {result.metrics.request_throughput}"
        )

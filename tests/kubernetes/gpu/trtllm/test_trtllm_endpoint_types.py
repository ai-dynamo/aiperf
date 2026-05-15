# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GPU benchmark with different endpoint types against real TRT-LLM."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.conftest import (
    GPUTestSettings,
    _dump_diagnostics,
    _log_pod_statuses,
)
from tests.kubernetes.gpu.trtllm.helpers import TRTLLMConfig
from tests.kubernetes.gpu.vllm.helpers import GPUBenchmarkDeployer
from tests.kubernetes.helpers.benchmark import BenchmarkConfig
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


class TestTRTLLMEndpointTypes:
    """Tests for different endpoint types against a real TRT-LLM server."""

    @pytest.mark.parametrize(
        "endpoint_type",
        [
            param("chat", id="chat"),
            param("completions", id="completions"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_endpoint_type_completes_successfully(
        self,
        benchmark_deployer: GPUBenchmarkDeployer,
        trtllm_endpoint_url: str,
        trtllm_config: TRTLLMConfig,
        kubectl: KubectlClient,
        gpu_settings: GPUTestSettings,
        endpoint_type: str,
    ) -> None:
        """Verify benchmark succeeds with both chat and completions endpoints."""
        logger.info(
            f"[TEST] Endpoint type test: type={endpoint_type}, model={trtllm_config.model_name}, url={trtllm_endpoint_url}"
        )

        config = BenchmarkConfig(
            endpoint_url=trtllm_endpoint_url,
            endpoint_type=endpoint_type,
            model_name=trtllm_config.model_name,
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

        result = await benchmark_deployer.deploy(
            config=config,
            wait_for_completion=True,
            timeout=600,
        )

        logger.info(
            f"[TEST] Endpoint={endpoint_type} result: success={result.success}, duration={result.duration_seconds:.1f}s"
        )

        if result.metrics:
            logger.info(
                f"[TEST] Endpoint={endpoint_type} metrics: throughput={result.metrics.request_throughput or 0:.2f} req/s, "
                f"latency_avg={result.metrics.request_latency_avg or 0:.2f} ms, requests={result.metrics.request_count}, errors={result.metrics.error_count}"
            )

        await _log_pod_statuses(kubectl, result.namespace)

        if not result.success:
            await _dump_diagnostics(kubectl, result.namespace, label="ENDPOINT_FAILURE")

        assert result.success, (
            f"Benchmark failed with endpoint_type={endpoint_type}: "
            f"{result.error_message}"
        )
        assert result.metrics is not None
        assert result.metrics.request_count >= 1, (
            f"Expected >= 1 completed request, got {result.metrics.request_count}"
        )
        assert result.metrics.error_count == 0

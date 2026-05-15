# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for LoRA adapter GPU E2E tests.

Provides a session-scoped ``lora_deployer`` fixture that mirrors the vLLM
deployer pattern. Opt-in via ``AIPERF_RUN_LORA_GPU_TESTS=1`` -- the fixture
will otherwise short-circuit at resolution time so collection still works on
clusterless machines.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.conftest import GPUTestSettings
from tests.kubernetes.gpu.lora.helpers import (
    DEFAULT_LORA_ADAPTER_NAME,
    DEFAULT_LORA_SOURCE,
    LoraConfig,
    LoraDeployer,
    LoraMode,
)
from tests.kubernetes.gpu.vllm.helpers import GPUBenchmarkDeployer
from tests.kubernetes.helpers.benchmark import BenchmarkConfig
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


# ============================================================================
# CLI options (scoped to this subpackage)
# ============================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register LoRA-specific CLI options."""
    group = parser.getgroup("gpu-lora", "GPU LoRA E2E test options")
    group.addoption(
        "--gpu-lora-image",
        default=None,
        help=(
            "Image to use for the LoRA adapter base server. Defaults to "
            "--gpu-dynamo-image (which itself falls back to backend default)."
        ),
    )
    group.addoption(
        "--gpu-lora-source",
        default=None,
        help="LoRA adapter source URI, e.g. hf://org/repo.",
    )
    group.addoption(
        "--gpu-lora-name",
        default=None,
        help="LoRA adapter name (metadata.name on the DynamoModel CR).",
    )


# ============================================================================
# LoRA config / deployer fixtures
# ============================================================================


@pytest.fixture(scope="session")
def lora_image(
    request: pytest.FixtureRequest,
    gpu_settings: GPUTestSettings,
) -> str:
    """Resolve the image used for the LoRA base server.

    Precedence: ``--gpu-lora-image`` > ``--gpu-dynamo-image`` > sensible default.
    """
    cli = request.config.getoption("--gpu-lora-image")
    if cli:
        return str(cli)
    env = os.environ.get("AIPERF_LORA_IMAGE")
    if env:
        return env
    if gpu_settings.dynamo_image:
        return gpu_settings.dynamo_image
    # TODO: adjust for LoRA specifics -- replace once there is a canonical
    # LoRA-capable base image published for CI.
    return "nvcr.io/nvidia/dynamo/vllm-runtime:latest"


@pytest.fixture(scope="session")
def lora_config(
    request: pytest.FixtureRequest,
    gpu_settings: GPUTestSettings,
) -> LoraConfig:
    """LoRA adapter configuration resolved from CLI + env + defaults."""
    name = request.config.getoption("--gpu-lora-name") or DEFAULT_LORA_ADAPTER_NAME
    source = request.config.getoption("--gpu-lora-source") or os.environ.get(
        "AIPERF_LORA_SOURCE", DEFAULT_LORA_SOURCE
    )
    return LoraConfig(
        adapter_name=str(name),
        base_model=gpu_settings.model,
        source=str(source),
        mode=LoraMode.ADAPTER,
        tolerations=gpu_settings.tolerations,
        node_selector=gpu_settings.node_selector,
    )


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def lora_deployer(
    kubectl: KubectlClient,
    lora_config: LoraConfig,
) -> AsyncGenerator[LoraDeployer, None]:
    """Session-scoped LoRA adapter deployer.

    Mirrors ``vllm_deployer`` from ``tests/kubernetes/gpu/vllm/conftest.py``.
    The underlying Dynamo base model must be deployed separately (see
    ``tests/kubernetes/gpu/dynamo/conftest.py``) before the adapter CR can
    reconcile.
    """
    deployer = LoraDeployer(kubectl=kubectl, config=lora_config)
    yield deployer
    await deployer.cleanup()


# ============================================================================
# Benchmark configuration fixture
# ============================================================================


@pytest.fixture
def lora_benchmark_config(
    lora_config: LoraConfig,
    gpu_settings: GPUTestSettings,
) -> BenchmarkConfig:
    """Small LoRA benchmark config used by scaffolded tests."""
    s = gpu_settings
    # TODO: adjust for LoRA specifics -- endpoint_url should point at the
    # Dynamo frontend service once a real base-model fixture is wired in.
    endpoint_url = (
        f"http://dynamo-frontend.{lora_config.namespace}.svc.cluster.local:8000/v1"
    )
    return BenchmarkConfig(
        endpoint_url=endpoint_url,
        endpoint_type="chat",
        model_name=lora_config.adapter_name,
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


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def lora_benchmark_deployer(
    benchmark_deployer: GPUBenchmarkDeployer,
) -> AsyncGenerator[GPUBenchmarkDeployer, None]:
    """Alias the GPU benchmark deployer so LoRA tests have a stable name."""
    yield benchmark_deployer

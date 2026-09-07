# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM Kubernetes manifest contracts."""

from __future__ import annotations

import yaml

from tests.kubernetes.gpu.trtllm.helpers import TRTLLMConfig, TRTLLMDeployer


def test_generate_manifest_uses_trt_libraries_without_cuda_compat_shim() -> None:
    """The CUDA compatibility shim conflicts with the host driver."""
    deployer = TRTLLMDeployer(kubectl=None, config=TRTLLMConfig())  # type: ignore[arg-type]
    documents = list(yaml.safe_load_all(deployer.generate_manifest()))
    container = documents[2]["spec"]["template"]["spec"]["containers"][0]

    environment = {item["name"]: item["value"] for item in container["env"]}

    assert "/usr/local/tensorrt/lib" in environment["LD_LIBRARY_PATH"]
    assert "/usr/local/cuda/compat/lib.real" not in environment["LD_LIBRARY_PATH"]

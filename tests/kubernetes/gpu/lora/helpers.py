# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LoRA adapter deployment helpers for Kubernetes E2E tests.

Mirrors the structure of ``tests/kubernetes/gpu/vllm/helpers.py`` but targets
the ``DynamoModel`` CRD used by ``dev/kube.py cmd_deploy_lora``: the adapter
rides on top of a running Dynamo base model, identified by a HuggingFace-style
source URI (``hf://org/repo``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


# The namespace the dev CLI pins LoRA DynamoModel CRs to. Mirrors
# ``DYNAMO_NAMESPACE`` in ``dev/kube.py``.
DEFAULT_LORA_NAMESPACE = "dynamo-server"

# Default adapter name used by the scaffold. Kept short and DNS-safe.
DEFAULT_LORA_ADAPTER_NAME = "test-lora-adapter"

# Placeholder source URI. Real runs must override via the pytest option so the
# adapter actually exists on HuggingFace / the configured registry.
DEFAULT_LORA_SOURCE = "hf://org/repo-placeholder"


class LoraMode(str, Enum):
    """LoRA deployment shape.

    Currently the only supported mode is ``adapter`` (DynamoModel CRD served
    by a running Dynamo base model). The enum exists so future modes (e.g.
    inline LoRA weights or merged adapters) can slot in without reshaping the
    fixture API.
    """

    ADAPTER = "adapter"


@dataclass
class LoraConfig:
    """Configuration for a LoRA adapter deployment.

    Fields map 1:1 onto the ``DynamoModel`` CR produced by
    ``dev/kube.py::_generate_lora_manifest``.
    """

    adapter_name: str = DEFAULT_LORA_ADAPTER_NAME
    """LoRA adapter name (becomes metadata.name and spec.modelName)."""

    base_model: str = "Qwen/Qwen3-0.6B"
    """Base model name the adapter attaches to."""

    source: str = DEFAULT_LORA_SOURCE
    """Adapter source URI (e.g. hf://org/repo)."""

    namespace: str = DEFAULT_LORA_NAMESPACE
    """Kubernetes namespace for the DynamoModel CR."""

    mode: LoraMode = LoraMode.ADAPTER
    """Deployment shape (see ``LoraMode``)."""

    tolerations: list[dict[str, str]] = field(default_factory=list)
    """Pod tolerations (passed through when the base model schedules)."""

    node_selector: dict[str, str] = field(default_factory=dict)
    """Pod node selector (passed through when the base model schedules)."""


class LoraDeployer:
    """Deploys and manages a LoRA adapter (DynamoModel CRD) on Kubernetes.

    The adapter itself is just a CR; the Dynamo operator reconciles it onto a
    running base model. This deployer therefore only handles CR lifecycle --
    the Dynamo base model must already be deployed via ``DynamoDeployer``.
    """

    def __init__(self, kubectl: KubectlClient, config: LoraConfig) -> None:
        self.kubectl = kubectl
        self.config = config
        self._deployed = False

    def generate_manifest(self) -> str:
        """Render the DynamoModel CR for this LoRA adapter.

        Mirrors ``dev/kube.py::_generate_lora_manifest``.
        """
        c = self.config
        doc = {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoModel",
            "metadata": {"name": c.adapter_name, "namespace": c.namespace},
            "spec": {
                "modelName": c.adapter_name,
                "baseModelName": c.base_model,
                "modelType": "lora",
                "source": {"uri": c.source},
            },
        }
        return yaml.dump(doc, default_flow_style=False, sort_keys=False)

    async def deploy(self) -> None:
        """Apply the DynamoModel CR to the cluster."""
        logger.info(
            f"Deploying LoRA adapter: name={self.config.adapter_name}, "
            f"base={self.config.base_model}, source={self.config.source}"
        )
        manifest = self.generate_manifest()
        logger.debug(lambda manifest=manifest: f"[LORA] Applying manifest:\n{manifest}")
        output = await self.kubectl.apply(manifest)
        self._deployed = True
        logger.info(f"[LORA] kubectl apply output:\n{output.rstrip()}")

    async def cleanup(self) -> None:
        """Remove the DynamoModel CR."""
        if not self._deployed:
            return
        logger.info(f"Cleaning up LoRA adapter {self.config.adapter_name}")
        await self.kubectl.run(
            "delete",
            "dynamomodel",
            self.config.adapter_name,
            "-n",
            self.config.namespace,
            "--ignore-not-found",
            check=False,
        )
        self._deployed = False


def expected_lora_model_name(config: LoraConfig) -> str:
    """Return the model name AIPerf should target for a deployed LoRA adapter.

    For the ``adapter`` mode, Dynamo serves the adapter under its own
    ``modelName`` (same string passed as ``--name`` to ``deploy-lora``), so
    benchmarks should hit the adapter name, not the base model.
    """
    # TODO: adjust for LoRA specifics -- if Dynamo ends up exposing adapters
    # under ``{base_model}+{adapter_name}`` or similar, update this helper.
    return config.adapter_name

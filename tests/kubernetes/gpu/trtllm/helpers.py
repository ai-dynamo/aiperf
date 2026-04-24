# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM deployment helpers for Kubernetes E2E tests.

Uses the ``tensorrt_llm`` serve command which provides an OpenAI-compatible
API server with on-the-fly engine building.  This avoids the complexity of
pre-building TRT-LLM engines or setting up a Triton model repository.

The server exposes ``/v1/chat/completions`` and ``/v1/completions`` endpoints,
with a health check at ``/health``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient, background_status
from tests.kubernetes.helpers.log_streamer import PodLogStreamer

logger = AIPerfLogger(__name__)

_DEFAULT_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7"
_DEFAULT_PORT = 8000


@dataclass
class TRTLLMConfig:
    """Configuration for a TRT-LLM server deployment."""

    image: str = _DEFAULT_IMAGE
    """TRT-LLM container image."""

    model_name: str = "Qwen/Qwen3-0.6B"
    """Model name to serve."""

    gpu_count: int = 1
    """Number of GPUs to allocate."""

    namespace: str = "trtllm-server"
    """Kubernetes namespace for the deployment."""

    port: int = _DEFAULT_PORT
    """HTTP server port."""

    max_model_len: int = 4096
    """Maximum model context length."""

    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism."""

    tolerations: list[dict[str, str]] = field(default_factory=list)
    """Kubernetes pod tolerations for GPU nodes."""

    node_selector: dict[str, str] = field(default_factory=dict)
    """Kubernetes node selector for GPU nodes."""

    hf_token_secret: str | None = None
    """Kubernetes secret name containing HuggingFace token."""

    image_pull_secrets: list[str] = field(default_factory=list)
    """Image pull secret names for private registries."""

    extra_args: list[str] = field(default_factory=list)
    """Additional command-line arguments for the TRT-LLM server."""

    runtime_class_name: str | None = None
    """Kubernetes RuntimeClass for GPU pods."""


class TRTLLMDeployer:
    """Deploys and manages a TRT-LLM server on Kubernetes.

    Uses ``trtllm-serve`` which downloads the HuggingFace model, builds a
    TRT-LLM engine on the fly, and starts an OpenAI-compatible HTTP server.
    """

    def __init__(self, kubectl: KubectlClient, config: TRTLLMConfig) -> None:
        self.kubectl = kubectl
        self.config = config
        self._deployed = False

    def generate_manifest(self) -> str:
        """Generate Kubernetes manifest for TRT-LLM deployment.

        Returns:
            YAML manifest string with Namespace, Service, and Deployment.
        """
        c = self.config

        # trtllm-serve command with on-the-fly engine building
        args = [
            "trtllm-serve",
            "serve",
            c.model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(c.port),
            "--max_seq_len",
            str(c.max_model_len),
            "--tensor_parallel_size",
            str(c.tensor_parallel_size),
            *c.extra_args,
        ]

        _ld_path = (
            "/usr/local/tensorrt/lib:"
            "/usr/local/cuda/lib64:"
            "/usr/local/cuda/compat/lib.real:"
            "/usr/local/lib/python3.12/dist-packages/torch/lib:"
            "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
        )

        container: dict = {
            "name": "trtllm",
            "image": c.image,
            "command": [args[0]],
            "args": args[1:],
            "ports": [{"containerPort": c.port, "name": "http"}],
            "env": [
                {"name": "LD_LIBRARY_PATH", "value": _ld_path},
            ],
            "resources": {
                "limits": {"nvidia.com/gpu": c.gpu_count},
                "requests": {"nvidia.com/gpu": c.gpu_count},
            },
            "readinessProbe": {
                "httpGet": {"path": "/health", "port": c.port},
                "initialDelaySeconds": 30,
                "periodSeconds": 10,
                "timeoutSeconds": 10,
                "failureThreshold": 120,
            },
            "livenessProbe": {
                "httpGet": {"path": "/health", "port": c.port},
                "initialDelaySeconds": 60,
                "periodSeconds": 30,
                "timeoutSeconds": 10,
                "failureThreshold": 40,
            },
            "volumeMounts": [
                {"name": "shm", "mountPath": "/dev/shm"},
            ],
        }

        # HF token env if secret is specified
        if c.hf_token_secret:
            container["env"].append(
                {
                    "name": "HF_TOKEN",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": c.hf_token_secret,
                            "key": "token",
                        }
                    },
                }
            )

        pod_spec: dict = {
            "containers": [container],
            "volumes": [
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}},
            ],
        }

        if c.runtime_class_name:
            pod_spec["runtimeClassName"] = c.runtime_class_name

        if c.tolerations:
            pod_spec["tolerations"] = c.tolerations

        if c.node_selector:
            pod_spec["nodeSelector"] = c.node_selector

        if c.image_pull_secrets:
            pod_spec["imagePullSecrets"] = [{"name": s} for s in c.image_pull_secrets]

        documents = [
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": c.namespace},
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "trtllm-server", "namespace": c.namespace},
                "spec": {
                    "selector": {"app": "trtllm-server"},
                    "ports": [
                        {
                            "port": c.port,
                            "targetPort": c.port,
                            "protocol": "TCP",
                            "name": "http",
                        }
                    ],
                    "type": "ClusterIP",
                },
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "trtllm-server",
                    "namespace": c.namespace,
                },
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": "trtllm-server"}},
                    "template": {
                        "metadata": {"labels": {"app": "trtllm-server"}},
                        "spec": pod_spec,
                    },
                },
            },
        ]

        return "\n---\n".join(
            yaml.dump(doc, default_flow_style=False) for doc in documents
        )

    async def deploy(self) -> None:
        """Deploy TRT-LLM server to the cluster."""
        logger.info(
            f"Deploying TRT-LLM server: model={self.config.model_name}, "
            f"gpus={self.config.gpu_count}, namespace={self.config.namespace}"
        )

        manifest = self.generate_manifest()
        logger.debug(
            lambda manifest=manifest: f"[TRTLLM] Applying manifest:\n{manifest}"
        )
        output = await self.kubectl.apply(manifest)
        self._deployed = True

        logger.info(f"[TRTLLM] kubectl apply output:\n{output.rstrip()}")

    async def wait_for_ready(
        self,
        timeout: int = 900,
        poll_interval: int = 15,
        stream_logs: bool = False,
    ) -> None:
        """Wait for TRT-LLM server to become ready, logging progress periodically.

        TRT-LLM engine building takes significantly longer than vLLM model loading,
        so the default timeout is higher (900s vs 600s).

        Args:
            timeout: Timeout in seconds (default 900s for engine building).
            poll_interval: Seconds between status checks (default 15).
            stream_logs: If True, stream pod logs in the background while waiting.

        Raises:
            TimeoutError: If TRT-LLM doesn't become ready within timeout.
        """
        logger.info(f"Waiting for TRT-LLM server readiness (timeout={timeout}s)")
        start = time.perf_counter()
        deadline = start + timeout

        async with (
            PodLogStreamer(
                self.kubectl, self.config.namespace, prefix="TRTLLM"
            ) as streamer,
            background_status(
                self.kubectl, self.config.namespace, label="TRTLLM", interval=30
            ),
        ):
            if stream_logs:
                streamer.watch(name_filter="trtllm-server")

            while True:
                elapsed = time.perf_counter() - start
                if time.perf_counter() > deadline:
                    pods = await self.kubectl.get_pods(self.config.namespace)
                    for pod in pods:
                        logger.error(
                            f"[TRTLLM] Pod {pod.name:<50} phase={pod.phase:<12} ready={pod.ready} restarts={pod.restarts}"
                        )
                    events = await self.kubectl.get_events(
                        self.config.namespace, limit=20
                    )
                    logger.error(f"[TRTLLM] Events:\n{events}")
                    logs = await self.get_logs(tail=80)
                    raise TimeoutError(
                        f"TRT-LLM server failed to become ready within {timeout}s "
                        f"(waited {elapsed:.0f}s).\nRecent logs:\n{logs}"
                    )

                pods = await self.kubectl.get_pods(self.config.namespace)
                trtllm_pods = [p for p in pods if "trtllm-server" in p.name]

                if trtllm_pods and all(p.is_ready for p in trtllm_pods):
                    logger.info(f"[TRTLLM] Server is ready (took {elapsed:.1f}s)")
                    return

                if not stream_logs:
                    for pod in trtllm_pods:
                        logger.info(
                            f"[TRTLLM] Waiting... {pod.name:<50} phase={pod.phase:<12} ready={str(pod.ready):<5} "
                            f"restarts={pod.restarts} ({elapsed:.0f}s)"
                        )

                await asyncio.sleep(poll_interval)

    async def get_logs(self, tail: int | None = 100) -> str:
        """Get TRT-LLM server logs.

        Args:
            tail: Number of lines to tail.

        Returns:
            Log content.
        """
        pods = await self.kubectl.get_pods(self.config.namespace)
        trtllm_pods = [p for p in pods if "trtllm-server" in p.name]

        if not trtllm_pods:
            return "(no trtllm-server pods found)"

        return await self.kubectl.get_logs(
            trtllm_pods[0].name,
            container="trtllm",
            namespace=self.config.namespace,
            tail=tail,
        )

    def get_endpoint_url(self) -> str:
        """Get the in-cluster endpoint URL for TRT-LLM.

        Returns:
            URL string like http://trtllm-server.trtllm-server.svc.cluster.local:8000/v1
        """
        c = self.config
        return f"http://trtllm-server.{c.namespace}.svc.cluster.local:{c.port}/v1"

    async def cleanup(self) -> None:
        """Remove TRT-LLM server deployment and namespace."""
        if not self._deployed:
            return

        logger.info(f"Cleaning up TRT-LLM server in namespace {self.config.namespace}")
        await self.kubectl.delete_namespace(self.config.namespace, wait=True)
        self._deployed = False

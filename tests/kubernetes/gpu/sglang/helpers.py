# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang deployment and GPU benchmark helpers for Kubernetes E2E tests."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient, background_status
from tests.kubernetes.helpers.log_streamer import PodLogStreamer

logger = AIPerfLogger(__name__)


@dataclass
class SGLangConfig:
    """Configuration for an SGLang server deployment."""

    image: str = "lmsysorg/sglang:latest"
    """SGLang container image."""

    model_name: str = "Qwen/Qwen3-0.6B"
    """Model name to serve."""

    gpu_count: int = 1
    """Number of GPUs to allocate."""

    namespace: str = "sglang-server"
    """Kubernetes namespace for the deployment."""

    port: int = 8000
    """HTTP server port."""

    tolerations: list[dict[str, str]] = field(default_factory=list)
    """Kubernetes pod tolerations for GPU nodes."""

    node_selector: dict[str, str] = field(default_factory=dict)
    """Kubernetes node selector for GPU nodes."""

    hf_token_secret: str | None = None
    """Kubernetes secret name containing HuggingFace token."""

    image_pull_secrets: list[str] = field(default_factory=list)
    """Image pull secret names for private registries."""

    extra_args: list[str] = field(default_factory=list)
    """Additional command-line arguments for the SGLang server."""

    runtime_class_name: str | None = None
    """Kubernetes RuntimeClass for GPU pods."""


class SGLangDeployer:
    """Deploys and manages an SGLang server on Kubernetes."""

    def __init__(self, kubectl: KubectlClient, config: SGLangConfig) -> None:
        self.kubectl = kubectl
        self.config = config
        self._deployed = False

    def generate_manifest(self) -> str:
        """Generate Kubernetes manifest for SGLang deployment.

        Returns:
            YAML manifest string with Namespace, Service, and Deployment.
        """
        c = self.config

        # Build SGLang container command
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            c.model_name,
            "--port",
            str(c.port),
            "--host",
            "0.0.0.0",
            *c.extra_args,
        ]

        # Build container spec
        container: dict = {
            "name": "sglang",
            "image": c.image,
            "command": command,
            "ports": [{"containerPort": c.port, "name": "http"}],
            "resources": {
                "limits": {"nvidia.com/gpu": c.gpu_count},
                "requests": {"nvidia.com/gpu": c.gpu_count},
            },
            "readinessProbe": {
                "httpGet": {"path": "/health", "port": c.port},
                "initialDelaySeconds": 30,
                "periodSeconds": 10,
                "timeoutSeconds": 5,
                "failureThreshold": 30,
            },
            "livenessProbe": {
                "httpGet": {"path": "/health", "port": c.port},
                "initialDelaySeconds": 60,
                "periodSeconds": 15,
                "timeoutSeconds": 5,
                "failureThreshold": 5,
            },
        }

        if c.hf_token_secret:
            container["env"] = [
                {
                    "name": "HF_TOKEN",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": c.hf_token_secret,
                            "key": "token",
                        }
                    },
                }
            ]

        # Build pod spec
        pod_spec: dict = {"containers": [container]}

        if c.runtime_class_name:
            pod_spec["runtimeClassName"] = c.runtime_class_name

        if c.tolerations:
            pod_spec["tolerations"] = c.tolerations

        if c.node_selector:
            pod_spec["nodeSelector"] = c.node_selector

        if c.image_pull_secrets:
            pod_spec["imagePullSecrets"] = [{"name": s} for s in c.image_pull_secrets]

        # Build full manifest (Namespace + Service + Deployment)
        documents = [
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": c.namespace},
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "sglang-server", "namespace": c.namespace},
                "spec": {
                    "selector": {"app": "sglang-server"},
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
                    "name": "sglang-server",
                    "namespace": c.namespace,
                },
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": "sglang-server"}},
                    "template": {
                        "metadata": {"labels": {"app": "sglang-server"}},
                        "spec": pod_spec,
                    },
                },
            },
        ]

        return "\n---\n".join(
            yaml.dump(doc, default_flow_style=False) for doc in documents
        )

    async def deploy(self) -> None:
        """Deploy SGLang server to the cluster."""
        logger.info(
            f"Deploying SGLang server: model={self.config.model_name}, "
            f"gpus={self.config.gpu_count}, namespace={self.config.namespace}"
        )

        manifest = self.generate_manifest()
        logger.debug(
            lambda manifest=manifest: f"[SGLANG] Applying manifest:\n{manifest}"
        )
        output = await self.kubectl.apply(manifest)
        self._deployed = True

        logger.info(f"[SGLANG] kubectl apply output:\n{output.rstrip()}")

    async def wait_for_ready(
        self,
        timeout: int = 600,
        poll_interval: int = 15,
        stream_logs: bool = False,
    ) -> None:
        """Wait for SGLang server to become ready, logging progress periodically.

        Args:
            timeout: Timeout in seconds (default 600s for model loading).
            poll_interval: Seconds between status checks (default 15).
            stream_logs: If True, stream pod logs in the background while waiting.

        Raises:
            TimeoutError: If SGLang doesn't become ready within timeout.
        """
        logger.info(f"Waiting for SGLang server readiness (timeout={timeout}s)")
        start = time.perf_counter()
        deadline = start + timeout

        async with (
            PodLogStreamer(
                self.kubectl, self.config.namespace, prefix="SGLANG"
            ) as streamer,
            background_status(
                self.kubectl, self.config.namespace, label="SGLANG", interval=30
            ),
        ):
            if stream_logs:
                streamer.watch(name_filter="sglang-server")

            while True:
                elapsed = time.perf_counter() - start
                if time.perf_counter() > deadline:
                    pods = await self.kubectl.get_pods(self.config.namespace)
                    for pod in pods:
                        logger.error(
                            f"[SGLANG] Pod {pod.name:<50} phase={pod.phase:<12} ready={pod.ready} restarts={pod.restarts}"
                        )
                    events = await self.kubectl.get_events(
                        self.config.namespace, limit=20
                    )
                    logger.error(f"[SGLANG] Events:\n{events}")
                    logs = await self.get_logs(tail=80)
                    raise TimeoutError(
                        f"SGLang server failed to become ready within {timeout}s "
                        f"(waited {elapsed:.0f}s).\nRecent logs:\n{logs}"
                    )

                pods = await self.kubectl.get_pods(self.config.namespace)
                sglang_pods = [p for p in pods if "sglang-server" in p.name]

                if sglang_pods and all(p.is_ready for p in sglang_pods):
                    logger.info(f"[SGLANG] Server is ready (took {elapsed:.1f}s)")
                    return

                if not stream_logs:
                    for pod in sglang_pods:
                        logger.info(
                            f"[SGLANG] Waiting... {pod.name:<50} phase={pod.phase:<12} ready={str(pod.ready):<5} "
                            f"restarts={pod.restarts} ({elapsed:.0f}s)"
                        )

                await asyncio.sleep(poll_interval)

    async def get_logs(self, tail: int | None = 100) -> str:
        """Get SGLang server logs.

        Args:
            tail: Number of lines to tail.

        Returns:
            Log content.
        """
        pods = await self.kubectl.get_pods(self.config.namespace)
        sglang_pods = [p for p in pods if "sglang-server" in p.name]

        if not sglang_pods:
            return "(no sglang-server pods found)"

        return await self.kubectl.get_logs(
            sglang_pods[0].name,
            container="sglang",
            namespace=self.config.namespace,
            tail=tail,
        )

    def get_endpoint_url(self) -> str:
        """Get the in-cluster endpoint URL for SGLang.

        Returns:
            URL string like http://sglang-server.sglang-server.svc.cluster.local:8000/v1
        """
        c = self.config
        return f"http://sglang-server.{c.namespace}.svc.cluster.local:{c.port}/v1"

    async def cleanup(self) -> None:
        """Remove SGLang server deployment and namespace."""
        if not self._deployed:
            return

        logger.info(f"Cleaning up SGLang server in namespace {self.config.namespace}")
        await self.kubectl.delete_namespace(self.config.namespace, wait=True)
        self._deployed = False

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM deployment and GPU benchmark helpers for Kubernetes E2E tests."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.benchmark import BenchmarkDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient, background_status
from tests.kubernetes.helpers.log_streamer import PodLogStreamer

logger = AIPerfLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for a vLLM server deployment."""

    image: str = "vllm/vllm-openai:latest"
    """vLLM container image."""

    model_name: str = "facebook/opt-125m"
    """Model name to serve."""

    gpu_count: int = 1
    """Number of GPUs to allocate."""

    namespace: str = "vllm-server"
    """Kubernetes namespace for the deployment."""

    port: int = 8000
    """HTTP server port."""

    max_model_len: int = 512
    """Maximum model context length."""

    dtype: str = "auto"
    """Data type for model weights (auto, float16, bfloat16)."""

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

    enforce_eager: bool = False
    """Disable CUDA graph optimization for debugging."""

    gpu_memory_utilization: float | None = None
    """GPU memory utilization target (0.0-1.0), or None for default."""

    extra_args: list[str] = field(default_factory=list)
    """Additional command-line arguments for the vLLM server."""

    runtime_class_name: str | None = None
    """Kubernetes RuntimeClass for GPU pods."""


class VLLMDeployer:
    """Deploys and manages a vLLM server on Kubernetes."""

    def __init__(self, kubectl: KubectlClient, config: VLLMConfig) -> None:
        self.kubectl = kubectl
        self.config = config
        self._deployed = False

    def generate_manifest(self) -> str:
        """Generate Kubernetes manifest for vLLM deployment.

        Returns:
            YAML manifest string with Service and Deployment.
        """
        c = self.config

        # Build vLLM container args
        args = [
            "--model",
            c.model_name,
            "--port",
            str(c.port),
            "--max-model-len",
            str(c.max_model_len),
            "--dtype",
            c.dtype,
            "--tensor-parallel-size",
            str(c.tensor_parallel_size),
            *(["--enforce-eager"] if c.enforce_eager else []),
            *(
                ["--gpu-memory-utilization", str(c.gpu_memory_utilization)]
                if c.gpu_memory_utilization is not None
                else []
            ),
            *c.extra_args,
        ]

        # Build container spec
        container: dict = {
            "name": "vllm",
            "image": c.image,
            "args": args,
            "ports": [{"containerPort": c.port, "name": "http"}],
            "resources": {
                "limits": {"nvidia.com/gpu": c.gpu_count},
                "requests": {"nvidia.com/gpu": c.gpu_count},
            },
            "readinessProbe": {
                "httpGet": {"path": "/health", "port": c.port},
                "initialDelaySeconds": 15,
                "periodSeconds": 5,
                "timeoutSeconds": 5,
                "failureThreshold": 60,
            },
            "livenessProbe": {
                "httpGet": {"path": "/health", "port": c.port},
                "initialDelaySeconds": 60,
                "periodSeconds": 15,
                "timeoutSeconds": 5,
                "failureThreshold": 5,
            },
        }

        # Add HF token env if secret is specified
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
                "metadata": {"name": "vllm-server", "namespace": c.namespace},
                "spec": {
                    "selector": {"app": "vllm-server"},
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
                    "name": "vllm-server",
                    "namespace": c.namespace,
                },
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": "vllm-server"}},
                    "template": {
                        "metadata": {"labels": {"app": "vllm-server"}},
                        "spec": pod_spec,
                    },
                },
            },
        ]

        return "\n---\n".join(
            yaml.dump(doc, default_flow_style=False) for doc in documents
        )

    async def deploy(self) -> None:
        """Deploy vLLM server to the cluster."""
        logger.info(
            f"Deploying vLLM server: model={self.config.model_name}, "
            f"gpus={self.config.gpu_count}, namespace={self.config.namespace}"
        )

        manifest = self.generate_manifest()
        logger.debug(lambda manifest=manifest: f"[VLLM] Applying manifest:\n{manifest}")
        output = await self.kubectl.apply(manifest)
        self._deployed = True

        logger.info(f"[VLLM] kubectl apply output:\n{output.rstrip()}")

    async def wait_for_ready(
        self,
        timeout: int = 600,
        poll_interval: int = 15,
        stream_logs: bool = False,
    ) -> None:
        """Wait for vLLM server to become ready, logging progress periodically.

        Args:
            timeout: Timeout in seconds (default 600s for model loading).
            poll_interval: Seconds between status checks (default 15).
            stream_logs: If True, stream pod logs in the background while waiting.

        Raises:
            TimeoutError: If vLLM doesn't become ready within timeout.
        """
        logger.info(f"Waiting for vLLM server readiness (timeout={timeout}s)")
        start = time.perf_counter()
        deadline = start + timeout

        async with (
            PodLogStreamer(
                self.kubectl, self.config.namespace, prefix="VLLM"
            ) as streamer,
            background_status(
                self.kubectl, self.config.namespace, label="VLLM", interval=30
            ),
        ):
            if stream_logs:
                streamer.watch(name_filter="vllm-server")

            while True:
                elapsed = time.perf_counter() - start
                if time.perf_counter() > deadline:
                    pods = await self.kubectl.get_pods(self.config.namespace)
                    for pod in pods:
                        logger.error(
                            f"[VLLM] Pod {pod.name:<50} phase={pod.phase:<12} ready={pod.ready} restarts={pod.restarts}"
                        )
                    events = await self.kubectl.get_events(
                        self.config.namespace, limit=20
                    )
                    logger.error(f"[VLLM] Events:\n{events}")
                    logs = await self.get_logs(tail=80)
                    raise TimeoutError(
                        f"vLLM server failed to become ready within {timeout}s "
                        f"(waited {elapsed:.0f}s).\nRecent logs:\n{logs}"
                    )

                pods = await self.kubectl.get_pods(self.config.namespace)
                vllm_pods = [p for p in pods if "vllm-server" in p.name]

                if vllm_pods and all(p.is_ready for p in vllm_pods):
                    logger.info(f"[VLLM] Server is ready (took {elapsed:.1f}s)")
                    return

                if not stream_logs:
                    for pod in vllm_pods:
                        logger.info(
                            f"[VLLM] Waiting... {pod.name:<50} phase={pod.phase:<12} ready={str(pod.ready):<5} "
                            f"restarts={pod.restarts} ({elapsed:.0f}s)"
                        )

                await asyncio.sleep(poll_interval)

    async def get_logs(self, tail: int | None = 100) -> str:
        """Get vLLM server logs.

        Args:
            tail: Number of lines to tail.

        Returns:
            Log content.
        """
        pods = await self.kubectl.get_pods(self.config.namespace)
        vllm_pods = [p for p in pods if "vllm-server" in p.name]

        if not vllm_pods:
            return "(no vllm-server pods found)"

        return await self.kubectl.get_logs(
            vllm_pods[0].name,
            container="vllm",
            namespace=self.config.namespace,
            tail=tail,
        )

    def get_endpoint_url(self) -> str:
        """Get the in-cluster endpoint URL for vLLM.

        Returns:
            URL string like http://vllm-server.vllm-server.svc.cluster.local:8000/v1
        """
        c = self.config
        return f"http://vllm-server.{c.namespace}.svc.cluster.local:{c.port}/v1"

    async def cleanup(self) -> None:
        """Remove vLLM server deployment and namespace."""
        if not self._deployed:
            return

        logger.info(f"Cleaning up vLLM server in namespace {self.config.namespace}")
        await self.kubectl.delete_namespace(self.config.namespace, wait=True)
        self._deployed = False


class GPUBenchmarkDeployer(BenchmarkDeployer):
    """Benchmark deployer for GPU clusters (e.g. remote or pre-provisioned).

    Overrides image pull policy to use IfNotPresent instead of Never,
    since GPU clusters typically pull images from registries (not pre-loaded like minikube).
    """

    def _patch_image_pull_policy(self, manifest: str, image: str) -> str:
        """Patch manifest to use imagePullPolicy: IfNotPresent for GPU clusters.

        Handles both AIPerfJob CRs (single doc) and raw manifests (multi-doc
        with Role, RoleBinding, ConfigMap, JobSet).

        Args:
            manifest: YAML manifest string.
            image: Image name (unused, kept for API compat).

        Returns:
            Patched manifest.
        """
        import yaml

        logger.debug(
            lambda image=image: f"[GPU] Patching imagePullPolicy to IfNotPresent for {image}"
        )
        docs = list(yaml.safe_load_all(manifest))
        for doc in docs:
            if not doc:
                continue
            if doc.get("kind") == "AIPerfJob":
                doc.setdefault("spec", {})["imagePullPolicy"] = "IfNotPresent"
            elif doc.get("kind") == "JobSet":
                for rj in doc.get("spec", {}).get("replicatedJobs", []):
                    containers = (
                        rj.get("template", {})
                        .get("spec", {})
                        .get("template", {})
                        .get("spec", {})
                        .get("containers", [])
                    )
                    for container in containers:
                        container["imagePullPolicy"] = "IfNotPresent"
        return "---\n".join(
            yaml.dump(doc, default_flow_style=False, sort_keys=False)
            for doc in docs
            if doc
        )

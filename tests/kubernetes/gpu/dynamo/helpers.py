# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo deployment helpers for Kubernetes E2E tests.

Generates DynamoGraphDeployment CRDs for the Dynamo operator.
Supports agg, agg-router, and disagg modes with vLLM, TRT-LLM, or SGLang backends.

For single-GPU local testing, use ``DynamoConfig.single_gpu_disagg()`` which
skips nvidia.com/gpu resource requests and uses NVIDIA_VISIBLE_DEVICES=all
so both prefill and decode pods share the same physical GPU.

Requires the Dynamo operator to be installed in the cluster.
See: https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import DYNAMO_VERSION
from tests.kubernetes.helpers.kubectl import KubectlClient, background_status
from tests.kubernetes.helpers.log_streamer import PodLogStreamer

logger = AIPerfLogger(__name__)

_FRONTEND_PORT = 8000
_HEALTH_PORT = 9090
_NGC_IMAGE_BASE = "nvcr.io/nvidia/ai-dynamo"

# Downward API env var the Dynamo runtime expects but the operator may not inject.
_POD_UID_ENV = {
    "name": "POD_UID",
    "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
}

# Conservative default for single-GPU KVBM (pinned memory).
_KVBM_1GPU_DEFAULT_GB = 1


# ============================================================================
# Backend enum
# ============================================================================


class DynamoBackend(str, Enum):
    """Inference backend for Dynamo workers."""

    VLLM = "vllm"
    TRTLLM = "trtllm"
    SGLANG = "sglang"

    @property
    def default_image(self) -> str:
        """Default container image for this backend."""
        return _BACKEND_DEFAULTS[self]["image"]

    @property
    def worker_command(self) -> list[str]:
        """Worker entrypoint command."""
        return _BACKEND_DEFAULTS[self]["command"]

    @property
    def worker_working_dir(self) -> str:
        """Working directory inside the worker container."""
        return _BACKEND_DEFAULTS[self]["working_dir"]


_BACKEND_DEFAULTS: dict[DynamoBackend, dict] = {
    DynamoBackend.VLLM: {
        "image": f"{_NGC_IMAGE_BASE}/vllm-runtime:{DYNAMO_VERSION}",
        "command": ["python3", "-m", "dynamo.vllm"],
        "working_dir": "/workspace/examples/backends/vllm",
    },
    DynamoBackend.TRTLLM: {
        "image": f"{_NGC_IMAGE_BASE}/trtllm-runtime:{DYNAMO_VERSION}",
        "command": ["python3", "-m", "dynamo.trtllm"],
        "working_dir": "/workspace/examples/backends/trtllm",
    },
    DynamoBackend.SGLANG: {
        "image": f"{_NGC_IMAGE_BASE}/sglang-runtime:{DYNAMO_VERSION}",
        "command": ["python3", "-m", "dynamo.sglang"],
        "working_dir": "/workspace/examples/backends/sglang",
    },
}


class DynamoMode(str, Enum):
    """Dynamo deployment topology."""

    AGGREGATED = "agg"
    AGGREGATED_ROUTER = "agg-router"
    DISAGGREGATED = "disagg"
    DISAGGREGATED_1GPU = "disagg-1gpu"

    @property
    def is_disaggregated(self) -> bool:
        """Whether this mode uses separate prefill/decode workers."""
        return self in (DynamoMode.DISAGGREGATED, DynamoMode.DISAGGREGATED_1GPU)


# ============================================================================
# Config
# ============================================================================


@dataclass
class DynamoConfig:
    """Configuration for a Dynamo deployment."""

    model_name: str = "Qwen/Qwen3-0.6B"
    """Model name to serve."""

    image: str | None = None
    """Container image, or None for backend-specific default."""

    namespace: str = "dynamo-server"
    """Kubernetes namespace for the deployment."""

    backend: DynamoBackend = DynamoBackend.VLLM
    """Inference backend engine."""

    mode: DynamoMode = DynamoMode.AGGREGATED
    """Deployment topology (aggregated or disaggregated)."""

    gpu_count: int = 1
    """Number of GPUs per worker pod."""

    decode_replicas: int = 1
    """Number of decode worker replicas."""

    prefill_replicas: int = 1
    """Number of prefill worker replicas."""

    frontend_replicas: int = 1
    """Number of frontend router replicas."""

    max_model_len: int | None = None
    """Maximum model context length, or None for default."""

    enforce_eager: bool = False
    """Disable CUDA graph optimization for debugging."""

    router_mode: str | None = None
    """Dynamo router mode, or None for default."""

    hf_token_secret: str | None = None
    """Kubernetes secret name containing HuggingFace token."""

    tolerations: list[dict[str, str]] = field(default_factory=list)
    """Kubernetes pod tolerations for GPU nodes."""

    node_selector: dict[str, str] = field(default_factory=dict)
    """Kubernetes node selector for GPU nodes."""

    image_pull_secrets: list[str] = field(default_factory=list)
    """Image pull secret names for private registries."""

    extra_worker_args: list[str] = field(default_factory=list)
    """Additional command-line arguments for worker containers."""

    extra_envs: list[dict[str, str]] = field(default_factory=list)
    """Additional environment variables for the deployment."""

    pvc_name: str | None = None
    """PVC name for model storage, or None."""

    model_mount_path: str = "/models"
    """Mount path for the model PVC inside containers."""

    tensor_parallel_size: int | None = None
    """Number of GPUs for tensor parallelism, or None for default."""

    gpu_memory_utilization: float | None = None
    """GPU memory utilization target (0.0-1.0), or None for default."""

    runtime_class_name: str | None = None
    """Kubernetes RuntimeClass for GPU pods."""

    kvbm_cpu_cache_gb: int | None = None
    """KVBM CPU cache size in GB for prefill workers."""

    connectors: list[str] = field(default_factory=list)
    """KV cache transfer connectors for prefill workers (e.g. kvbm, nixl)."""

    @property
    def effective_image(self) -> str:
        """Resolve the container image (explicit or backend default)."""
        return self.image or self.backend.default_image

    @classmethod
    def single_gpu_disagg(cls, **overrides: object) -> DynamoConfig:
        """Preset for disaggregated mode on a single GPU (local testing).

        Skips nvidia.com/gpu resource requests so K8s doesn't block scheduling.
        Uses ``runtimeClassName: nvidia`` so the NVIDIA container runtime still
        mounts the GPU driver/devices into every pod.  Both workers share the
        physical GPU with low ``--gpu-memory-utilization`` so they coexist.

        When ``connectors`` includes ``kvbm`` and no explicit
        ``kvbm_cpu_cache_gb`` is given, defaults to a conservative 1 GB.
        """
        defaults: dict = {
            "mode": DynamoMode.DISAGGREGATED_1GPU,
            "model_name": "Qwen/Qwen3-0.6B",
            "gpu_count": 0,
            "max_model_len": 4096,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.12,
            "runtime_class_name": "nvidia",
        }
        defaults.update(overrides)

        # Auto-default KVBM CPU cache when kvbm connector is used
        connectors = defaults.get("connectors", [])
        if (
            isinstance(connectors, list)
            and "kvbm" in connectors
            and defaults.get("kvbm_cpu_cache_gb") is None
        ):
            defaults["kvbm_cpu_cache_gb"] = _KVBM_1GPU_DEFAULT_GB

        return cls(**defaults)


# ============================================================================
# Deployer
# ============================================================================


class DynamoDeployer:
    """Deploys and manages a Dynamo inference graph on Kubernetes.

    Uses the DynamoGraphDeployment CRD (nvidia.com/v1alpha1) to deploy
    inference graphs via the Dynamo operator.  Supports both vLLM and
    TRT-LLM backends.
    """

    def __init__(self, kubectl: KubectlClient, config: DynamoConfig) -> None:
        self.kubectl = kubectl
        self.config = config
        self._deployed = False

    def _build_worker_args(
        self, *, is_prefill: bool = False, is_decode: bool = False
    ) -> list[str]:
        """Build worker command-line args."""
        c = self.config
        args = ["--model", c.model_name]

        if c.max_model_len is not None:
            args.extend(["--max-model-len", str(c.max_model_len)])

        if c.tensor_parallel_size is not None:
            args.extend(["--tensor-parallel-size", str(c.tensor_parallel_size)])

        if c.gpu_memory_utilization is not None:
            args.extend(["--gpu-memory-utilization", str(c.gpu_memory_utilization)])

        if c.enforce_eager:
            args.append("--enforce-eager")

        if is_prefill:
            args.append("--is-prefill-worker")
            if c.connectors:
                args.extend(["--connector", *c.connectors])
        elif is_decode and c.mode.is_disaggregated:
            args.append("--is-decode-worker")

        args.extend(c.extra_worker_args)
        return args

    def _build_worker_envs(self, *, is_prefill: bool = False) -> list[dict[str, str]]:
        """Build environment variables for a worker."""
        envs: list[dict[str, str]] = []
        if is_prefill and self.config.kvbm_cpu_cache_gb is not None:
            envs.append(
                {
                    "name": "DYN_KVBM_CPU_CACHE_GB",
                    "value": str(self.config.kvbm_cpu_cache_gb),
                }
            )
            envs.append({"name": "DYN_KVBM_METRICS", "value": "true"})
        return envs

    def _build_probes(self) -> dict:
        """Build startup + liveness probes for workers."""
        return {
            "startupProbe": {
                "httpGet": {"path": "/live", "port": _HEALTH_PORT},
                "initialDelaySeconds": 60,
                "periodSeconds": 10,
                "failureThreshold": 60,
                "timeoutSeconds": 5,
            },
            "livenessProbe": {
                "httpGet": {"path": "/live", "port": _HEALTH_PORT},
                "initialDelaySeconds": 0,
                "periodSeconds": 10,
                "failureThreshold": 15,
                "timeoutSeconds": 5,
            },
        }

    def _build_worker_service(
        self,
        *,
        is_prefill: bool = False,
        is_decode: bool = False,
        replicas: int,
    ) -> dict:
        """Build a single worker service spec."""
        c = self.config
        dynamo_ns = self._deployment_name()

        main_container: dict = {
            "image": c.effective_image,
            "workingDir": c.backend.worker_working_dir,
            "command": c.backend.worker_command,
            "args": self._build_worker_args(is_prefill=is_prefill, is_decode=is_decode),
        }

        worker_envs = [_POD_UID_ENV, *self._build_worker_envs(is_prefill=is_prefill)]
        main_container["env"] = worker_envs

        main_container.update(self._build_probes())

        extra_pod_spec: dict = {"mainContainer": main_container}
        if c.runtime_class_name:
            extra_pod_spec["runtimeClassName"] = c.runtime_class_name

        service: dict = {
            "dynamoNamespace": dynamo_ns,
            "componentType": "worker",
            "replicas": replicas,
            "extraPodSpec": extra_pod_spec,
        }

        if c.gpu_count > 0:
            service["resources"] = {"limits": {"gpu": str(c.gpu_count)}}

        if is_prefill:
            service["subComponentType"] = "prefill"
        elif is_decode and c.mode.is_disaggregated:
            service["subComponentType"] = "decode"

        if c.hf_token_secret:
            service["envFromSecret"] = c.hf_token_secret

        if c.pvc_name:
            service["volumeMounts"] = [
                {"name": c.pvc_name, "mountPoint": c.model_mount_path}
            ]

        return service

    def _deployment_name(self) -> str:
        """Canonical deployment name derived from mode."""
        return f"dynamo-{self.config.mode.value}"

    def _worker_service_key(self, role: str) -> str:
        """Service key name for the given worker role (decode/prefill)."""
        prefixes = {
            DynamoBackend.VLLM: "Vllm",
            DynamoBackend.TRTLLM: "Trtllm",
            DynamoBackend.SGLANG: "Sglang",
        }
        return f"{prefixes[self.config.backend]}{role.title()}Worker"

    def generate_manifest(self) -> str:
        """Generate DynamoGraphDeployment CRD manifest.

        Returns:
            YAML manifest string.
        """
        c = self.config
        deploy_name = self._deployment_name()

        # Frontend service
        frontend_container: dict = {"image": c.effective_image, "env": [_POD_UID_ENV]}
        frontend_pod_spec: dict = {"mainContainer": frontend_container}
        if c.runtime_class_name:
            frontend_pod_spec["runtimeClassName"] = c.runtime_class_name

        frontend: dict = {
            "dynamoNamespace": deploy_name,
            "componentType": "frontend",
            "replicas": c.frontend_replicas,
            "extraPodSpec": frontend_pod_spec,
        }

        if c.router_mode:
            frontend["envs"] = [{"name": "DYN_ROUTER_MODE", "value": c.router_mode}]

        if c.pvc_name:
            frontend["volumeMounts"] = [
                {"name": c.pvc_name, "mountPoint": c.model_mount_path}
            ]

        # Build services dict
        services: dict = {"Frontend": frontend}

        decode_key = self._worker_service_key("decode")
        prefill_key = self._worker_service_key("prefill")

        if c.mode.is_disaggregated:
            services[decode_key] = self._build_worker_service(
                is_decode=True,
                replicas=c.decode_replicas,
            )
            services[prefill_key] = self._build_worker_service(
                is_prefill=True,
                replicas=c.prefill_replicas,
            )
        else:
            services[decode_key] = self._build_worker_service(
                is_decode=True,
                replicas=c.decode_replicas,
            )

        # Build CRD spec
        spec: dict = {"services": services}

        if c.pvc_name:
            spec["pvcs"] = [{"name": c.pvc_name}]

        if c.extra_envs:
            spec["envs"] = c.extra_envs

        # Namespace + CRD documents
        documents = [
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": c.namespace},
            },
            {
                "apiVersion": "nvidia.com/v1alpha1",
                "kind": "DynamoGraphDeployment",
                "metadata": {
                    "name": deploy_name,
                    "namespace": c.namespace,
                },
                "spec": spec,
            },
        ]

        return "\n---\n".join(
            yaml.dump(doc, default_flow_style=False) for doc in documents
        )

    async def deploy(self) -> None:
        """Deploy Dynamo inference graph to the cluster."""
        logger.info(
            f"Deploying Dynamo ({self.config.mode.value}, "
            f"backend={self.config.backend.value}): "
            f"model={self.config.model_name}, namespace={self.config.namespace}"
        )

        manifest = self.generate_manifest()
        logger.debug(
            lambda manifest=manifest: f"[DYNAMO] Applying manifest:\n{manifest}"
        )
        output = await self.kubectl.apply(manifest)
        self._deployed = True

        logger.info(f"[DYNAMO] kubectl apply output:\n{output.rstrip()}")

    async def wait_for_ready(
        self,
        timeout: int = 600,
        poll_interval: int = 15,
        stream_logs: bool = False,
    ) -> None:
        """Wait for all Dynamo pods to become ready, logging progress periodically.

        Args:
            timeout: Timeout in seconds (default 600s for model loading).
            poll_interval: Seconds between status checks (default 15).
            stream_logs: If True, stream pod logs in the background while waiting.

        Raises:
            TimeoutError: If Dynamo doesn't become ready within timeout.
        """
        logger.info(f"Waiting for Dynamo readiness (timeout={timeout}s)")
        start = time.perf_counter()
        deadline = start + timeout
        deploy_name = self._deployment_name()

        async with (
            PodLogStreamer(
                self.kubectl, self.config.namespace, prefix="DYNAMO"
            ) as streamer,
            background_status(
                self.kubectl, self.config.namespace, label="DYNAMO", interval=30
            ),
        ):
            if stream_logs:
                streamer.watch(name_filter=deploy_name)

            while True:
                elapsed = time.perf_counter() - start
                if time.perf_counter() > deadline:
                    pods = await self.kubectl.get_pods(self.config.namespace)
                    for pod in pods:
                        logger.error(
                            f"[DYNAMO] Pod {pod.name:<50} phase={pod.phase:<12} ready={pod.ready} restarts={pod.restarts}"
                        )
                    events = await self.kubectl.get_events(
                        self.config.namespace, limit=20
                    )
                    logger.error(f"[DYNAMO] Events:\n{events}")
                    logs = await self.get_logs(tail=80)
                    raise TimeoutError(
                        f"Dynamo failed to become ready within {timeout}s "
                        f"(waited {elapsed:.0f}s).\nRecent logs:\n{logs}"
                    )

                pods = await self.kubectl.get_pods(self.config.namespace)
                dynamo_pods = [p for p in pods if deploy_name in p.name]

                if dynamo_pods and all(p.is_ready for p in dynamo_pods):
                    logger.info(f"[DYNAMO] All pods ready (took {elapsed:.1f}s)")
                    return

                if not dynamo_pods and elapsed > 30:
                    operator_pods = await self.kubectl.run(
                        "get",
                        "pods",
                        "-n",
                        "dynamo-system",
                        "-l",
                        "app.kubernetes.io/name=dynamo-operator",
                        "--no-headers",
                        check=False,
                    )
                    if (
                        operator_pods.returncode != 0
                        or not operator_pods.stdout.strip()
                    ):
                        logger.warning(
                            f"[DYNAMO] No pods in {self.config.namespace} after "
                            f"{elapsed:.0f}s - Dynamo operator not found in dynamo-system!"
                        )
                    elif "Running" not in operator_pods.stdout:
                        logger.warning(
                            f"[DYNAMO] No pods in {self.config.namespace} after "
                            f"{elapsed:.0f}s - operator is not Running:\n{operator_pods.stdout.rstrip()}"
                        )
                    else:
                        logger.info(
                            f"[DYNAMO] No pods in {self.config.namespace} yet "
                            f"({elapsed:.0f}s) - operator is Running, waiting for reconciliation..."
                        )
                    events = await self.kubectl.get_events(
                        self.config.namespace, limit=10
                    )
                    if events.strip():
                        logger.info(f"[DYNAMO] Events:\n{events.rstrip()}")

                elif not stream_logs:
                    for pod in dynamo_pods:
                        logger.info(
                            f"[DYNAMO] Waiting... {pod.name:<50} phase={pod.phase:<12} ready={str(pod.ready):<5} "
                            f"restarts={pod.restarts} ({elapsed:.0f}s)"
                        )

                await asyncio.sleep(poll_interval)

    async def get_logs(self, tail: int | None = 100) -> str:
        """Get logs from all Dynamo pods.

        Args:
            tail: Number of lines to tail per pod.

        Returns:
            Combined log content.
        """
        pods = await self.kubectl.get_pods(self.config.namespace)
        dynamo_pods = [p for p in pods if self._deployment_name() in p.name]

        if not dynamo_pods:
            return "(no dynamo pods found)"

        logs: list[str] = []
        for pod in dynamo_pods:
            pod_logs = await self.kubectl.get_logs(
                pod.name,
                namespace=self.config.namespace,
                tail=tail,
            )
            logs.append(f"--- {pod.name} ---\n{pod_logs}")

        return "\n".join(logs)

    def get_endpoint_url(self) -> str:
        """Get the in-cluster endpoint URL for the Dynamo frontend.

        Returns:
            URL like http://dynamo-agg-frontend.dynamo-server.svc.cluster.local:8000/v1
        """
        c = self.config
        frontend_svc = f"{self._deployment_name()}-frontend"
        return (
            f"http://{frontend_svc}.{c.namespace}.svc.cluster.local:{_FRONTEND_PORT}/v1"
        )

    async def cleanup(self) -> None:
        """Remove Dynamo deployment and namespace."""
        if not self._deployed:
            return

        logger.info(f"Cleaning up Dynamo in namespace {self.config.namespace}")
        await self.kubectl.delete_namespace(self.config.namespace, wait=True)
        self._deployed = False

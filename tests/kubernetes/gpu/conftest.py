# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for GPU Kubernetes E2E tests.

By default, creates a Kind cluster with GPU passthrough (requires nvidia-smi,
nvidia-ctk, and Docker configured with nvidia as the default runtime). Set
``--gpu-context`` (or ``GPU_TEST_CONTEXT``) to use a pre-existing external
cluster instead, or ``--gpu-runtime minikube`` to use Minikube.

All settings can be configured via CLI options (``--gpu-*``) or environment
variables (``GPU_TEST_*``).  CLI takes precedence over environment.

Server-specific fixtures (vLLM, Dynamo) live in their own subpackage
conftest files (``gpu/vllm/conftest.py``, ``gpu/dynamo/conftest.py``).

Usage::

    uv run pytest tests/kubernetes/gpu/ -v -m gpu
    uv run pytest tests/kubernetes/gpu/vllm/ -v --gpu-stream-logs --gpu-model Qwen/Qwen3-0.6B
    uv run pytest tests/kubernetes/gpu/dynamo/ -v --gpu-context my-cluster
    uv run pytest tests/kubernetes/gpu/ -v --gpu-runtime minikube
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import pytest_asyncio

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import DYNAMO_VERSION, JOBSET_CRD_URL_TEMPLATE, JOBSET_VERSION
from tests.kubernetes.gpu.vllm.helpers import GPUBenchmarkDeployer
from tests.kubernetes.helpers.benchmark import BenchmarkResult
from tests.kubernetes.helpers.cluster import (
    ClusterConfig,
    ClusterRuntime,
    KindGpuSetup,
    LocalCluster,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)

# Stash key for GPUTestSettings on pytest.Config
_SETTINGS_KEY = pytest.StashKey["GPUTestSettings"]()


# ============================================================================
# GPUTestSettings - single source of truth for all GPU test configuration
# ============================================================================


@dataclass(frozen=True)
class GPUTestSettings:
    """Resolved GPU test configuration (CLI > env > default)."""

    context: str | None = None
    """External cluster kubectl context, or None for local cluster."""

    kubeconfig: str | None = None
    """Path to kubeconfig file, or None for default."""

    cluster: str = "aiperf-gpu"
    """Local cluster name."""

    runtime: str = "kind"
    """Cluster runtime backend (kind or minikube)."""

    quick: bool = False
    """Shorthand for reuse + skip build/cleanup/preflight."""

    reuse_cluster: bool = False
    """Reuse an existing cluster instead of creating a new one."""

    skip_build: bool = False
    """Skip building the aiperf Docker image."""

    skip_cleanup: bool = False
    """Keep resources and cluster after tests."""

    skip_preflight: bool = False
    """Skip preflight prerequisite checks."""

    stream_logs: bool = False
    """Stream pod logs in real time during deploys."""

    vllm_image: str = "vllm/vllm-openai:latest"
    """vLLM container image."""

    model: str = "Qwen/Qwen3-0.6B"
    """Model name to benchmark."""

    count: int = 1
    """Number of GPUs per inference server instance."""

    max_model_len: int = 4096
    """Maximum model context length."""

    mem_util: float = 0.5
    """GPU memory utilization target (0.0-1.0)."""

    vllm_endpoint: str | None = None
    """Pre-existing vLLM endpoint URL, or None to deploy."""

    aiperf_image: str = "aiperf:latest"
    """AIPerf container image."""

    benchmark_timeout: int = 600
    """Benchmark completion timeout in seconds."""

    vllm_deploy_timeout: int = 600
    """vLLM deployment readiness timeout in seconds."""

    tolerations: list[dict[str, str]] = field(default_factory=list)
    """Kubernetes pod tolerations for GPU nodes."""

    node_selector: dict[str, str] = field(default_factory=dict)
    """Kubernetes node selector for GPU nodes."""

    hf_token_secret: str | None = None
    """Kubernetes secret name containing HuggingFace token."""

    image_pull_secret: str | None = None
    """Image pull secret name for private registries."""

    runtime_class: str = "nvidia"
    """Kubernetes RuntimeClass for GPU pods."""

    trtllm_image: str = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7"
    """TRT-LLM container image."""

    trtllm_endpoint: str | None = None
    """Pre-existing TRT-LLM endpoint URL, or None to deploy."""

    trtllm_deploy_timeout: int = 900
    """TRT-LLM deployment readiness timeout in seconds."""

    sglang_image: str = "lmsysorg/sglang:latest"
    """SGLang container image."""

    sglang_endpoint: str | None = None
    """Pre-existing SGLang endpoint URL, or None to deploy."""

    sglang_deploy_timeout: int = 600
    """SGLang deployment readiness timeout in seconds."""

    dynamo_image: str | None = None
    """Dynamo runtime image, or None for backend-specific default."""

    dynamo_backend: str = "vllm"
    """Dynamo inference backend (vllm, trtllm, sglang)."""

    dynamo_mode: str = "disagg-1gpu"
    """Dynamo deployment mode (agg, disagg, disagg-1gpu)."""

    dynamo_connectors: str | None = None
    """Comma-separated connectors for Dynamo prefill workers."""

    dynamo_kvbm_cpu_cache_gb: int | None = None
    """KVBM CPU cache size in GB for prefill workers."""

    dynamo_endpoint: str | None = None
    """Pre-existing Dynamo endpoint URL, or None to deploy."""

    dynamo_deploy_timeout: int = 600
    """Dynamo deployment readiness timeout in seconds."""

    dynamo_version: str = DYNAMO_VERSION
    """Dynamo operator Helm chart version."""

    local_keygen: bool = False
    """Create MPI SSH secret locally instead of via Helm hook."""

    @property
    def use_external_cluster(self) -> bool:
        return self.context is not None

    @property
    def image_pull_secrets(self) -> list[str]:
        return [self.image_pull_secret] if self.image_pull_secret else []


def _get_settings(config: pytest.Config) -> GPUTestSettings:
    """Retrieve GPUTestSettings from pytest config stash."""
    return config.stash[_SETTINGS_KEY]


# ============================================================================
# CLI options
# ============================================================================

# Option definitions: (cli_flag, env_var, default, type, help)
# Types: str, int, float, bool, json_list, json_dict
_OPTIONS: list[tuple[str, str, str | None, str, str]] = [
    # Cluster management
    ("--gpu-quick", "GPU_TEST_QUICK", None, "bool", "Reuse cluster, skip build/cleanup/preflight"),
    ("--gpu-context", "GPU_TEST_CONTEXT", None, "str", "Use external cluster context (skips local cluster)"),
    ("--gpu-kubeconfig", "GPU_TEST_KUBECONFIG", None, "str", "Path to kubeconfig file"),
    ("--gpu-cluster", "GPU_TEST_CLUSTER", "aiperf-gpu", "str", "Cluster name"),
    ("--gpu-runtime", "GPU_TEST_RUNTIME", "kind", "str", "Cluster runtime: kind or minikube"),
    ("--gpu-reuse-cluster", "GPU_TEST_REUSE_CLUSTER", None, "bool", "Reuse existing cluster"),
    ("--gpu-skip-build", "GPU_TEST_SKIP_BUILD", None, "bool", "Skip building aiperf image"),
    ("--gpu-skip-cleanup", "GPU_TEST_SKIP_CLEANUP", None, "bool", "Keep resources/cluster after tests"),
    ("--gpu-skip-preflight", "GPU_TEST_SKIP_PREFLIGHT", None, "bool", "Skip preflight checks"),
    ("--gpu-stream-logs", "GPU_TEST_STREAM_LOGS", None, "bool", "Stream pod logs in real time during deploys"),
    # vLLM / model
    ("--gpu-vllm-image", "GPU_TEST_VLLM_IMAGE", "vllm/vllm-openai:latest", "str", "vLLM container image"),
    ("--gpu-model", "GPU_TEST_MODEL", "Qwen/Qwen3-0.6B", "str", "Model name"),
    ("--gpu-count", "GPU_TEST_GPU_COUNT", "1", "int", "GPUs per instance"),
    ("--gpu-max-model-len", "GPU_TEST_MAX_MODEL_LEN", "4096", "int", "Max model context length"),
    ("--gpu-mem-util", "GPU_TEST_GPU_MEM_UTIL", "0.5", "float", "GPU memory utilization (0.0-1.0)"),
    ("--gpu-vllm-endpoint", "GPU_TEST_VLLM_ENDPOINT", None, "str", "Skip vLLM deploy, use existing endpoint"),
    ("--gpu-aiperf-image", "GPU_TEST_AIPERF_IMAGE", "aiperf:latest", "str", "AIPerf container image"),
    ("--gpu-benchmark-timeout", "GPU_TEST_BENCHMARK_TIMEOUT", "600", "int", "Benchmark timeout (seconds)"),
    ("--gpu-vllm-deploy-timeout", "GPU_TEST_VLLM_DEPLOY_TIMEOUT", "600", "int", "vLLM deploy timeout (seconds)"),
    # K8s scheduling
    ("--gpu-tolerations", "GPU_TEST_TOLERATIONS", None, "json_list", "JSON array of K8s tolerations"),
    ("--gpu-node-selector", "GPU_TEST_NODE_SELECTOR", None, "json_dict", "JSON object of node selectors"),
    ("--gpu-hf-token-secret", "GPU_TEST_HF_TOKEN_SECRET", None, "str", "K8s secret with HuggingFace token"),
    ("--gpu-image-pull-secret", "GPU_TEST_IMAGE_PULL_SECRET", None, "str", "Image pull secret name"),
    ("--gpu-runtime-class", "GPU_TEST_RUNTIME_CLASS", "nvidia", "str", "RuntimeClass for GPU pods"),
    # TRT-LLM
    ("--gpu-trtllm-image", "GPU_TEST_TRTLLM_IMAGE", "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7", "str", "TRT-LLM container image"),
    ("--gpu-trtllm-endpoint", "GPU_TEST_TRTLLM_ENDPOINT", None, "str", "Skip TRT-LLM deploy, use existing endpoint"),
    ("--gpu-trtllm-deploy-timeout", "GPU_TEST_TRTLLM_DEPLOY_TIMEOUT", "900", "int", "TRT-LLM deploy timeout (seconds)"),
    # SGLang
    ("--gpu-sglang-image", "GPU_TEST_SGLANG_IMAGE", "lmsysorg/sglang:latest", "str", "SGLang container image"),
    ("--gpu-sglang-endpoint", "GPU_TEST_SGLANG_ENDPOINT", None, "str", "Skip SGLang deploy, use existing endpoint"),
    ("--gpu-sglang-deploy-timeout", "GPU_TEST_SGLANG_DEPLOY_TIMEOUT", "600", "int", "SGLang deploy timeout (seconds)"),
    # Dynamo
    ("--gpu-dynamo-image", "GPU_TEST_DYNAMO_IMAGE", None, "str", "Dynamo runtime image (default: backend-specific)"),
    ("--gpu-dynamo-backend", "GPU_TEST_DYNAMO_BACKEND", "vllm", "str", "Dynamo backend: vllm|trtllm|sglang"),
    ("--gpu-dynamo-mode", "GPU_TEST_DYNAMO_MODE", "disagg-1gpu", "str", "Dynamo mode: agg|disagg|disagg-1gpu"),
    ("--gpu-dynamo-connectors", "GPU_TEST_DYNAMO_CONNECTORS", None, "str", "Comma-separated connectors for prefill (e.g. kvbm,nixl)"),
    ("--gpu-dynamo-kvbm-cpu-cache-gb", "GPU_TEST_DYNAMO_KVBM_CPU_CACHE_GB", None, "int", "KVBM CPU cache GB for prefill workers"),
    ("--gpu-dynamo-endpoint", "GPU_TEST_DYNAMO_ENDPOINT", None, "str", "Skip Dynamo deploy, use existing endpoint"),
    ("--gpu-dynamo-deploy-timeout", "GPU_TEST_DYNAMO_DEPLOY_TIMEOUT", "600", "int", "Dynamo deploy timeout (seconds)"),
    ("--gpu-dynamo-version", "GPU_TEST_DYNAMO_VERSION", DYNAMO_VERSION, "str", "Dynamo operator Helm chart version"),
    ("--gpu-local-keygen", "GPU_TEST_LOCAL_KEYGEN", None, "bool", "Create MPI SSH secret locally instead of via Helm hook (use behind proxies)"),
]  # fmt: skip


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --gpu-* CLI options."""
    group = parser.getgroup("gpu", "GPU Kubernetes E2E test options")

    for cli_flag, _env_var, _default, typ, help_text in _OPTIONS:
        if typ == "bool":
            group.addoption(cli_flag, action="store_true", default=None, help=help_text)
        else:
            group.addoption(cli_flag, default=None, help=help_text)


def _resolve_settings(config: pytest.Config) -> GPUTestSettings:
    """Build GPUTestSettings from CLI options + env vars + defaults.

    Priority: CLI option > environment variable > default.
    """

    def _resolve(cli_flag: str, env_var: str, default: str | None, typ: str) -> object:
        # pytest stores --gpu-foo as config.option.gpu_foo
        attr = cli_flag.lstrip("-").replace("-", "_")
        cli_val = getattr(config.option, attr, None)

        if cli_val is not None:
            raw = cli_val
        else:
            env_val = os.environ.get(env_var, "")
            raw = env_val if env_val else default

        if raw is None:
            return None

        if typ == "bool":
            if isinstance(raw, bool):
                return raw
            return str(raw).lower() in ("1", "true", "yes")
        if typ == "int":
            return int(raw)
        if typ == "float":
            return float(raw)
        if typ == "json_list":
            return json.loads(raw) if raw else []
        if typ == "json_dict":
            return json.loads(raw) if raw else {}
        return str(raw)

    resolved = {}
    for cli_flag, env_var, default, typ, _help in _OPTIONS:
        # Map --gpu-foo-bar to foo_bar (strip --gpu- prefix)
        field_name = cli_flag.removeprefix("--gpu-").replace("-", "_")
        resolved[field_name] = _resolve(cli_flag, env_var, default, typ)

    # --gpu-quick implies reuse + skip build/cleanup/preflight
    if resolved.get("quick"):
        for key in ("reuse_cluster", "skip_build", "skip_cleanup", "skip_preflight"):
            if resolved.get(key) is None:
                resolved[key] = True

    # Handle json defaults that resolve to None
    if resolved.get("tolerations") is None:
        resolved["tolerations"] = []
    if resolved.get("node_selector") is None:
        resolved["node_selector"] = {}

    return GPUTestSettings(**resolved)


# ============================================================================
# Diagnostic helpers
# ============================================================================


async def _check_cluster_health(kubectl: KubectlClient) -> dict[str, list[str]]:
    """Check cluster health and return a report of issues.

    Returns:
        Dict with 'healthy', 'unhealthy', and 'missing' pod categories.
    """
    report: dict[str, list[str]] = {"healthy": [], "unhealthy": [], "missing": []}

    # Check critical kube-system pods
    result = await kubectl.run(
        "get", "pods", "-n", "kube-system", "--no-headers", check=False
    )
    if result.returncode != 0:
        report["missing"].append("kube-system (unreachable)")
        return report

    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name, ready, status = parts[0], parts[1], parts[2]
        if status == "Running" and ready.split("/")[0] == ready.split("/")[1]:
            report["healthy"].append(name)
        else:
            report["unhealthy"].append(f"{name} ({status}, ready={ready})")

    # Check for critical missing components
    # coredns and kube-dns are alternatives (Kind uses coredns)
    pod_names = result.stdout.lower()
    if "coredns" not in pod_names and "kube-dns" not in pod_names:
        report["missing"].append("dns (coredns or kube-dns)")
    if "kube-proxy" not in pod_names:
        report["missing"].append("kube-proxy")

    return report


async def _log_cluster_health(kubectl: KubectlClient) -> bool:
    """Log cluster health status. Returns True if healthy."""
    report = await _check_cluster_health(kubectl)

    if report["healthy"]:
        logger.info(f"[HEALTH] {len(report['healthy'])} healthy pods in kube-system")

    is_healthy = True
    if report["unhealthy"]:
        is_healthy = False
        logger.warning(f"[HEALTH] Unhealthy pods: {', '.join(report['unhealthy'])}")
    if report["missing"]:
        is_healthy = False
        logger.warning(
            f"[HEALTH] Missing critical components: {', '.join(report['missing'])}"
        )

    if is_healthy:
        logger.info("[HEALTH] Cluster is healthy")
    else:
        logger.warning("[HEALTH] Cluster has issues - tests may fail")

    return is_healthy


async def _log_pod_statuses(kubectl: KubectlClient, namespace: str) -> None:
    """Log all pod names, phases, and readiness in a namespace."""
    try:
        pods = await kubectl.get_pods(namespace)
        if not pods:
            logger.info(f"[PODS] No pods found in namespace {namespace}")
            return
        logger.info(f"[PODS] {len(pods)} pod(s) in namespace {namespace}:")
        for pod in pods:
            logger.info(
                f"  {pod.name:<60} phase={pod.phase:<12} ready={str(pod.ready):<5} restarts={pod.restarts}"
            )
    except Exception as e:
        logger.warning(f"[PODS] Failed to list pods in {namespace}: {e}")


async def _log_container_logs(
    kubectl: KubectlClient,
    namespace: str,
    tail: int = 80,
) -> None:
    """Log container output for every pod in a namespace."""
    try:
        pods = await kubectl.get_pods(namespace)
        for pod in pods:
            for container_name in pod.containers:
                logs = await kubectl.get_logs(
                    pod.name,
                    container=container_name,
                    namespace=namespace,
                    tail=tail,
                )
                if logs.strip():
                    logger.info(
                        f"[LOGS] {pod.name}/{container_name} (tail={tail}):\n{logs.rstrip()}"
                    )
                else:
                    logger.info(f"[LOGS] {pod.name}/{container_name}: (empty)")
    except Exception as e:
        logger.warning(f"[LOGS] Failed to collect logs in {namespace}: {e}")


async def _log_events(kubectl: KubectlClient, namespace: str) -> None:
    """Log recent Kubernetes events for a namespace."""
    try:
        events = await kubectl.get_events(namespace, limit=30)
        if events.strip():
            logger.info(f"[EVENTS] namespace={namespace}:\n{events.rstrip()}")
    except Exception as e:
        logger.warning(f"[EVENTS] Failed to get events for {namespace}: {e}")


async def _dump_diagnostics(
    kubectl: KubectlClient,
    namespace: str,
    label: str = "DIAGNOSTICS",
) -> None:
    """Dump full diagnostics for a namespace (pods, logs, events)."""
    logger.info("=" * 70)
    logger.info(f"[{label}] Full dump for namespace: {namespace}")
    logger.info("=" * 70)
    await _log_pod_statuses(kubectl, namespace)
    await _log_container_logs(kubectl, namespace, tail=150)
    await _log_events(kubectl, namespace)
    logger.info("=" * 70)


# All engine server namespaces (used for GPU-exclusive cleanup)
_ENGINE_NAMESPACES = ("dynamo-server", "sglang-server", "trtllm-server", "vllm-server")


async def _release_gpu(kubectl: KubectlClient, keep_namespace: str) -> None:
    """Delete other engine namespaces to free the GPU before deploying.

    With a single GPU, only one engine server can run at a time. Pytest
    package-scoped fixture teardowns may not execute before the next
    package's setup begins, leaving stale GPU-holding pods that block
    scheduling. This explicitly cleans up any leftover engine namespaces.
    """
    for ns in _ENGINE_NAMESPACES:
        if ns == keep_namespace:
            continue
        result = await kubectl.run("get", "namespace", ns, check=False)
        if result.returncode == 0:
            logger.info(f"[GPU] Releasing GPU: deleting leftover namespace {ns}")
            await kubectl.delete_namespace(ns, wait=True)


# ============================================================================
# Pytest hooks
# ============================================================================


def _check_no_subpackage_init_files() -> None:
    """Fail fast if vllm/, trtllm/, or dynamo/ have __init__.py (breaks package-scoped fixtures)."""
    gpu_dir = Path(__file__).parent
    for subdir in ("vllm", "trtllm", "sglang", "dynamo"):
        init = gpu_dir / subdir / "__init__.py"
        if init.exists():
            raise pytest.UsageError(
                f"{init.relative_to(gpu_dir.parent.parent.parent)} exists but must not. "
                f"Subdirectories of tests/kubernetes/gpu/ must NOT be packages, "
                f"otherwise scope='package' fixtures get duplicated per subdirectory. "
                f"See tests/kubernetes/gpu/__init__.py for details."
            )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest for GPU tests."""
    _check_no_subpackage_init_files()

    # Resolve settings from CLI + env and stash them.
    s = _resolve_settings(config)
    config.stash[_SETTINGS_KEY] = s

    # Print configuration banner so users can see resolved settings.
    _print_settings_banner(s)

    # Run GPU-specific preflight before disabling the parent's preflight.
    if not s.skip_preflight:
        _run_gpu_preflight_checks(s)

    # Skip the parent conftest's preflight checks - GPU tests manage their own cluster.
    os.environ.setdefault("K8S_TEST_SKIP_PREFLIGHT", "1")


def _print_settings_banner(s: GPUTestSettings) -> None:
    """Print resolved GPU test settings as a banner."""
    if s.use_external_cluster:
        mode = f"external cluster (context: {s.context})"
    else:
        mode = f"{s.runtime} with GPU (cluster: {s.cluster})"

    lines = [
        "",
        "=" * 60,
        "GPU Kubernetes Integration Tests",
        "=" * 60,
        f"  Mode:           {mode}",
        f"  vLLM image:     {s.vllm_image}",
        f"  Model:          {s.model}",
        f"  GPU count:      {s.count}",
        f"  GPU mem util:   {s.mem_util}",
        f"  vLLM endpoint:  {s.vllm_endpoint or '(will deploy)'}",
        f"  AIPerf image:   {s.aiperf_image}",
        f"  Reuse cluster:  {bool(s.reuse_cluster)}",
        f"  Skip build:     {bool(s.skip_build)}",
        f"  Skip cleanup:   {bool(s.skip_cleanup)}",
        f"  Stream logs:    {bool(s.stream_logs)}",
    ]
    lines.append(f"  TRT-LLM image:  {s.trtllm_image}")
    lines.append(f"  TRT-LLM endpoint: {s.trtllm_endpoint or '(will deploy)'}")
    lines.append(f"  SGLang image:   {s.sglang_image}")
    lines.append(f"  SGLang endpoint: {s.sglang_endpoint or '(will deploy)'}")
    if s.dynamo_endpoint:
        lines.append(f"  Dynamo endpoint: {s.dynamo_endpoint}")
    else:
        dynamo_img = s.dynamo_image or f"(default for {s.dynamo_backend})"
        lines.append(f"  Dynamo backend: {s.dynamo_backend}")
        lines.append(f"  Dynamo image:   {dynamo_img}")
        lines.append(f"  Dynamo mode:    {s.dynamo_mode}")
        if s.dynamo_connectors:
            lines.append(f"  Dynamo connectors: {s.dynamo_connectors}")
        if s.dynamo_kvbm_cpu_cache_gb is not None:
            lines.append(f"  Dynamo KVBM:    {s.dynamo_kvbm_cpu_cache_gb}GB")
    lines.append("=" * 60)
    lines.append("")

    print("\n".join(lines))


def _run_gpu_preflight_checks(s: GPUTestSettings) -> None:
    """Run preflight checks for GPU test prerequisites."""
    import subprocess

    from tests.kubernetes.helpers.preflight import PreflightChecker

    checker = PreflightChecker(
        title="GPU KUBERNETES E2E TEST",
        skip_message="Set --gpu-skip-preflight or GPU_TEST_SKIP_PREFLIGHT=1 to skip.",
    )

    if s.use_external_cluster:
        checker.set_mode("external cluster", context=s.context or "")
        checker.start(total_steps=3)

        c = checker.check("kubectl connectivity")
        cmd = ["kubectl", "--context", s.context]
        if s.kubeconfig:
            cmd.extend(["--kubeconfig", s.kubeconfig])
        cmd.append("cluster-info")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            c.pass_("cluster reachable")
        else:
            c.fail(f"cannot reach cluster: {result.stderr.strip()[:200]}")

        c = checker.check("GPU nodes")
        cmd = ["kubectl", "--context", s.context]
        if s.kubeconfig:
            cmd.extend(["--kubeconfig", s.kubeconfig])
        cmd.extend(["get", "nodes", "-o", "json"])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            nodes_data = json.loads(result.stdout)
            gpu_count = sum(
                int(
                    node.get("status", {})
                    .get("allocatable", {})
                    .get("nvidia.com/gpu", 0)
                )
                for node in nodes_data.get("items", [])
            )
            if gpu_count > 0:
                c.pass_(f"{gpu_count} nvidia.com/gpu allocatable")
            else:
                c.fail("no nvidia.com/gpu in allocatable resources")
        else:
            c.fail(f"failed to query nodes: {result.stderr.strip()[:200]}")

        c = checker.check("JobSet CRD")
        cmd = ["kubectl", "--context", s.context]
        if s.kubeconfig:
            cmd.extend(["--kubeconfig", s.kubeconfig])
        cmd.extend(["get", "crd", "jobsets.jobset.x-k8s.io"])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            c.pass_("jobsets.jobset.x-k8s.io installed")
        else:
            c.fail("not found - install JobSet controller first")

    else:
        runtime_label = s.runtime
        checker.set_mode(f"{runtime_label} with GPU passthrough", profile=s.cluster)
        total_steps = 4 if s.runtime == "kind" else 3
        checker.start(total_steps=total_steps)

        c = checker.check("nvidia-smi")
        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            detail = (
                result.stdout.strip().split("\n")[0] if result.returncode == 0 else ""
            )
            c.pass_(detail or "found")
        else:
            c.fail("not found - install NVIDIA drivers for GPU passthrough")

        c = checker.check("nvidia-container-toolkit")
        if shutil.which("nvidia-ctk"):
            c.pass_("found")
        else:
            c.fail("not found - install nvidia-container-toolkit")

        runtime_binary = "kind" if s.runtime == "kind" else "minikube"
        c = checker.check(runtime_binary)
        if shutil.which(runtime_binary):
            c.pass_("found")
        else:
            c.fail("not found in PATH")

        if s.runtime == "kind":
            c = checker.check("docker nvidia runtime")
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "Default Runtime" in line and "nvidia" in line:
                        c.pass_("docker default runtime is nvidia")
                        break
                else:
                    c.fail(
                        "docker default runtime is not nvidia "
                        "(run: sudo nvidia-ctk runtime configure "
                        "--runtime=docker --set-as-default && "
                        "sudo systemctl restart docker)"
                    )
            else:
                c.fail("docker not running")

    checker.finish(error_cls=pytest.UsageError)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-add gpu/vllm/trtllm/dynamo markers to tests under kubernetes/gpu/."""
    for item in items:
        fspath = str(item.fspath)
        if "kubernetes/gpu" in fspath:
            item.add_marker(pytest.mark.gpu)
            if "kubernetes/gpu/vllm" in fspath:
                item.add_marker(pytest.mark.vllm)
            elif "kubernetes/gpu/trtllm" in fspath:
                item.add_marker(pytest.mark.trtllm)
            elif "kubernetes/gpu/sglang" in fspath:
                item.add_marker(pytest.mark.sglang)
            elif "kubernetes/gpu/dynamo" in fspath:
                item.add_marker(pytest.mark.dynamo)


# ============================================================================
# Settings fixture
# ============================================================================


@pytest.fixture(scope="package")
def gpu_settings(request: pytest.FixtureRequest) -> GPUTestSettings:
    """Resolved GPU test settings (CLI > env > default)."""
    return _get_settings(request.config)


# ============================================================================
# Package-scoped infrastructure fixtures
# ============================================================================


@pytest.fixture(scope="package")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="package")
def gpu_cluster_config(gpu_settings: GPUTestSettings) -> ClusterConfig:
    """Cluster config with GPU passthrough (Kind by default, Minikube via --gpu-runtime)."""
    s = gpu_settings
    runtime = (
        ClusterRuntime.MINIKUBE if s.runtime == "minikube" else ClusterRuntime.KIND
    )
    return ClusterConfig(
        name=s.cluster,
        runtime=runtime,
        kubeconfig=Path(s.kubeconfig) if s.kubeconfig else None,
        wait_timeout=180,
        gpus=True,
        cache_images=[s.vllm_image, s.dynamo_image]
        if runtime is ClusterRuntime.MINIKUBE
        else [],
    )


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def gpu_cluster(
    gpu_cluster_config: ClusterConfig,
    gpu_settings: GPUTestSettings,
) -> AsyncGenerator[LocalCluster | None, None]:
    """Create a local cluster with GPU passthrough (Kind or Minikube).

    Skipped when --gpu-context is set (external cluster mode).
    """
    s = gpu_settings
    if s.use_external_cluster:
        yield None
        return

    cluster = LocalCluster(config=gpu_cluster_config)

    already_running = await cluster.exists()
    if already_running:
        logger.info(f"Reusing existing GPU cluster: {cluster.name}")
        temp_kubectl = KubectlClient(context=cluster.context)
        healthy = await _log_cluster_health(temp_kubectl)
        if not healthy and not s.reuse_cluster:
            logger.warning(
                f"[CLUSTER] Existing cluster '{cluster.name}' is unhealthy - "
                f"recreating (use --gpu-reuse-cluster to keep it)"
            )
            await cluster.delete()
            await cluster.create(force=False)
            already_running = False
        elif not healthy:
            logger.warning(
                f"[CLUSTER] Existing cluster '{cluster.name}' has health warnings "
                f"but --gpu-reuse-cluster is set - continuing anyway"
            )

        # Ensure GPU infrastructure (RuntimeClass, device plugin) is present
        # even on reused clusters - idempotent, safe to re-run
        if already_running:
            if s.runtime == "kind":
                node_container = f"{cluster.name}-control-plane"
                gpu_setup = KindGpuSetup(
                    context=cluster.context,
                    node_container=node_container,
                    device_plugin_version=gpu_cluster_config.device_plugin_version,
                )
                await gpu_setup.ensure_ready()
            else:
                # Minikube handles device plugin natively, just ensure RuntimeClass
                from tests.kubernetes.helpers.cluster import (
                    _create_nvidia_runtime_class,
                )

                await _create_nvidia_runtime_class(cluster.context)
    else:
        await cluster.create(force=False)

    yield cluster

    if not s.skip_cleanup and not already_running:
        await cluster.delete()
    else:
        logger.info(f"Keeping GPU cluster: {cluster.name}")


async def _docker_image_exists(image: str) -> bool:
    """Check if a Docker image exists locally."""
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "image",
        "inspect",
        image,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode == 0


async def _docker_build(image: str, project_root: Path) -> None:
    """Build a Docker image."""
    from tests.kubernetes.helpers.cluster import _run_streaming

    cmd = ["docker", "build", "-t", image, str(project_root)]
    await _run_streaming(cmd, "DOCKER", f"Failed to build {image}")


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def loaded_aiperf_image(
    gpu_cluster: LocalCluster | None,
    gpu_settings: GPUTestSettings,
    project_root: Path,
) -> None:
    """Build and load the aiperf image into the cluster.

    When --gpu-skip-build is set, falls back to building if the image
    doesn't exist locally, and loading if it's not in the cluster.
    """
    if gpu_cluster is None:
        return

    s = gpu_settings
    if s.skip_build:
        needs_build = not await _docker_image_exists(s.aiperf_image)
        needs_load = not await gpu_cluster.image_loaded(s.aiperf_image)

        if not needs_build and not needs_load:
            logger.info(
                "Skipping aiperf image build/load (--gpu-skip-build, image exists)"
            )
            return

        if needs_build:
            logger.info(
                f"--gpu-skip-build set but {s.aiperf_image} not found locally; building"
            )
            await _docker_build(s.aiperf_image, project_root)

        if needs_load:
            logger.info(
                f"--gpu-skip-build set but {s.aiperf_image} not in cluster; loading"
            )
            await gpu_cluster.load_image(s.aiperf_image)
        return

    await _docker_build(s.aiperf_image, project_root)
    await gpu_cluster.load_image(s.aiperf_image)


@pytest.fixture(scope="package")
def kubectl(
    gpu_cluster: LocalCluster | None,
    gpu_settings: GPUTestSettings,
) -> KubectlClient:
    """Create kubectl client for the GPU cluster."""
    s = gpu_settings
    if gpu_cluster is not None:
        return KubectlClient(context=gpu_cluster.context)
    return KubectlClient(context=s.context, kubeconfig=s.kubeconfig)


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def jobset_controller(kubectl: KubectlClient) -> None:
    """Ensure JobSet CRD is available on the cluster."""
    result = await kubectl.run("get", "crd", "jobsets.jobset.x-k8s.io", check=False)
    if result.returncode == 0:
        logger.info("JobSet CRD already installed")
        return

    jobset_version = os.environ.get("K8S_TEST_JOBSET_VERSION", JOBSET_VERSION)
    logger.info(f"Installing JobSet controller {jobset_version}")

    url = JOBSET_CRD_URL_TEMPLATE.format(version=jobset_version)
    await kubectl.apply_server_side(url)

    await kubectl.wait_for_condition(
        "deployment",
        "jobset-controller-manager",
        "available",
        namespace="jobset-system",
        timeout=60,
    )


async def _ensure_operator(
    kubectl: KubectlClient,
    project_root: Path,
    aiperf_image: str,
) -> None:
    """Install AIPerfJob CRD and deploy the operator (idempotent).

    Checks whether the CRD and operator deployment already exist before
    creating them, so it is safe to call on reused clusters.
    """
    from tests.kubernetes.helpers.operator import OperatorDeployer

    deployer = OperatorDeployer(
        kubectl=kubectl,
        project_root=project_root,
        operator_image=aiperf_image,
    )

    # CRD
    crd_result = await kubectl.run("get", "crd", deployer.CRD_NAME, check=False)
    if crd_result.returncode == 0:
        logger.info("AIPerfJob CRD already installed")
    else:
        logger.info("Installing AIPerfJob CRD")
        await deployer.install_crd()

    # Operator deployment
    op_result = await kubectl.run(
        "get",
        "deployment",
        "aiperf-operator",
        "-n",
        deployer.OPERATOR_NAMESPACE,
        check=False,
    )
    if op_result.returncode == 0:
        logger.info("AIPerf operator already deployed")
    else:
        logger.info("Deploying AIPerf operator")
        # Ensure namespace exists before applying namespaced resources
        await kubectl.run(
            "create",
            "namespace",
            deployer.OPERATOR_NAMESPACE,
            check=False,
        )
        await deployer.deploy_operator()


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def gpu_cluster_base(
    kubectl: KubectlClient,
    gpu_cluster: LocalCluster | None,
    loaded_aiperf_image: None,
    jobset_controller: None,
    project_root: Path,
    gpu_settings: GPUTestSettings,
) -> None:
    """Ensure cluster has aiperf image, JobSet CRD, AIPerfJob CRD, and operator.

    Lightweight prerequisite that does NOT deploy vLLM or Dynamo servers.
    Package-scoped so the cluster is shared across vLLM and Dynamo test suites.
    """
    await _ensure_operator(kubectl, project_root, gpu_settings.aiperf_image)

    try:
        result = await kubectl.run("get", "nodes", "-o", "wide", check=False)
        logger.info(f"[CLUSTER] Nodes:\n{result.stdout.rstrip()}")
    except Exception:
        pass

    await _log_cluster_health(kubectl)
    logger.info("GPU cluster base infrastructure ready")


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def benchmark_deployer(
    kubectl: KubectlClient,
    project_root: Path,
    gpu_cluster_base: None,
    gpu_settings: GPUTestSettings,
) -> AsyncGenerator[GPUBenchmarkDeployer, None]:
    """Create a GPU benchmark deployer."""
    s = gpu_settings
    deployer = GPUBenchmarkDeployer(
        kubectl=kubectl,
        project_root=project_root,
        default_image=s.aiperf_image,
    )

    yield deployer

    if not s.skip_cleanup:
        await deployer.cleanup_all()


# ============================================================================
# Utility fixtures
# ============================================================================


@pytest.fixture
def assert_metrics() -> Callable[..., None]:
    """Factory fixture for asserting benchmark metrics."""

    def _assert_metrics(
        result: BenchmarkResult,
        min_throughput: float | None = None,
        max_latency: float | None = None,
        expected_request_count: int | None = None,
        max_error_count: int = 0,
    ) -> None:
        """Assert benchmark metrics meet expectations.

        Args:
            result: Benchmark result to check.
            min_throughput: Minimum request throughput (req/s).
            max_latency: Maximum average latency (ms).
            expected_request_count: Expected number of completed requests.
            max_error_count: Maximum allowed error count.
        """
        assert result.success, f"Benchmark failed: {result.error_message}"
        assert result.metrics is not None, "No metrics collected"

        if min_throughput is not None:
            assert result.metrics.request_throughput is not None
            assert result.metrics.request_throughput >= min_throughput, (
                f"Throughput {result.metrics.request_throughput} < {min_throughput}"
            )

        if max_latency is not None:
            assert result.metrics.request_latency_avg is not None
            assert result.metrics.request_latency_avg <= max_latency, (
                f"Latency {result.metrics.request_latency_avg} > {max_latency}"
            )

        if expected_request_count is not None:
            assert result.metrics.request_count == expected_request_count, (
                f"Request count {result.metrics.request_count} != {expected_request_count}"
            )

        assert result.metrics.error_count <= max_error_count, (
            f"Error count {result.metrics.error_count} > {max_error_count}"
        )

    return _assert_metrics


@pytest.fixture
def get_pod_logs(kubectl: KubectlClient) -> Callable:
    """Factory fixture for getting pod logs."""

    async def _get_logs(
        result: BenchmarkResult,
        container: str = "control-plane",
        tail: int | None = 100,
    ) -> str:
        """Get logs from a benchmark pod.

        Args:
            result: Benchmark result.
            container: Container name.
            tail: Number of lines to tail.

        Returns:
            Log content.
        """
        controller_pod = result.controller_pod
        if not controller_pod:
            return ""

        return await kubectl.get_logs(
            controller_pod.name,
            container=container,
            namespace=result.namespace,
            tail=tail,
        )

    return _get_logs

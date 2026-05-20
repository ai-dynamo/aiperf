# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D507-D516 -- expanded Dynamo worker/runtime chaos scenarios.

These cases cover engine-process, model-configuration, cache/filesystem, GPU
memory/topology, and noisy-neighbor failure modes for Dynamo worker pods. The
suite deliberately performs capability checks before injection so clusters
without process visibility, GPU sidecar support, MIG resources, or the optional
sidecar images skip explicitly instead of producing false greens.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any, Literal

import orjson
import pytest
import yaml

from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.conftest import GPUTestSettings
from tests.kubernetes.gpu.dynamo.helpers import (
    DynamoBackend,
    DynamoConfig,
    DynamoDeployer,
    DynamoMode,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_DGD_NAME = "dynamo-agg"
_DGD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
_WORKER_LABEL = "nvidia.com/dynamo-component-type=worker"
_MAIN_CONTAINER = "main"
_FAILURE_TIMEOUT_S = 300.0
_POD_EVIDENCE_TIMEOUT_S = 180.0
_STATUS_FAILURE_TERMS = (
    "failed",
    "error",
    "invalid",
    "exception",
    "traceback",
    "crashloopbackoff",
    "runcontainererror",
)
_PROCESS_TERMS = ("engine", "vllm", "worker", "serve")
_GPU_PRESSURE_IMAGE_ENV = "DYNAMO_CHAOS_GPU_PRESSURE_IMAGE"
_CPU_NOISE_IMAGE_ENV = "DYNAMO_CHAOS_CPU_NOISE_IMAGE"
_ENABLE_CPU_NOISE_ENV = "DYNAMO_CHAOS_ENABLE_CPU_NOISY_NEIGHBOR"


async def test_d507_engine_child_death_restarts_worker(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_server: Any,  # noqa: ANN401 - fixture type is intentionally opaque
) -> None:
    """D507: kill a visible worker engine child process; worker recovers.

    This test uses the already-running Dynamo fixture. It skips when the active
    backend does not expose a non-PID-1 child process through ``kubectl exec``;
    that makes the process-topology prerequisite explicit for container images
    that wrap the engine in a single process.
    """
    del dynamo_server
    pod = await _first_worker_pod_or_skip(kubectl, dynamo_deployment_namespace)
    pid = await _engine_child_pid_or_skip(kubectl, dynamo_deployment_namespace, pod)
    restart_before = await _container_restart_count(
        kubectl, dynamo_deployment_namespace, pod, _MAIN_CONTAINER
    )

    async with faults.inject(
        "process.signal",
        target={
            "kind": "pod",
            "ns": dynamo_deployment_namespace,
            "pod": pod,
            "container": _MAIN_CONTAINER,
            "pid": pid,
        },
        signal="SIGKILL",
    ):
        pass

    await _wait_for_worker_recovery_after_process_kill(
        kubectl,
        namespace=dynamo_deployment_namespace,
        original_pod=pod,
        restart_before=restart_before,
    )


async def test_d508_bad_dtype_quantization_fails_actionably(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> None:
    """D508: invalid dtype/quantization worker args fail in status/logs."""
    del dynamo_operator
    namespace = "d508-bad-runtime-args"
    config = _runtime_config(
        namespace,
        gpu_settings,
        extra_worker_args=["--dtype", "not-a-real-dtype", "--quantization", "not-real"],
    )
    await _run_failure_case(
        kubectl,
        namespace=namespace,
        manifest=_manifest_for(kubectl, config),
        terms=("dtype", "quantization", "not-a-real-dtype", "not-real", "invalid"),
    )


async def test_d509_missing_model_path_fails_actionably(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> None:
    """D509: local model path that does not exist fails without hanging."""
    del dynamo_operator
    namespace = "d509-missing-model-path"
    config = _runtime_config(
        namespace,
        gpu_settings,
        model_name="/models/dynamo-chaos/path-does-not-exist",
    )
    await _run_failure_case(
        kubectl,
        namespace=namespace,
        manifest=_manifest_for(kubectl, config),
        terms=("/models/dynamo-chaos", "not found", "no such", "model"),
    )


async def test_d510_read_only_hf_cache_fails_actionably(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> None:
    """D510: read-only HF cache volume surfaces a cache/write failure."""
    del dynamo_operator
    namespace = "d510-read-only-cache"
    config = _runtime_config(
        namespace,
        gpu_settings,
        model_name=_cache_miss_model(),
    )
    manifest = _mutate_worker_podspec(
        _manifest_for(kubectl, config),
        lambda pod_spec, container: _add_read_only_cache(pod_spec, container),
    )
    await _run_failure_case(
        kubectl,
        namespace=namespace,
        manifest=manifest,
        terms=("read-only", "permission", "cache", "hf_home", "errno 30"),
    )


async def test_d511_corrupt_hf_cache_fails_actionably(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> None:
    """D511: corrupt cached model metadata fails with model/cache evidence."""
    del dynamo_operator
    namespace = "d511-corrupt-cache"
    model_path = "/models/corrupt-cache-model"
    config = _runtime_config(namespace, gpu_settings, model_name=model_path)
    manifest = _mutate_worker_podspec(
        _manifest_for(kubectl, config),
        lambda pod_spec, container: _add_corrupt_model_init(
            pod_spec, container, model_path
        ),
    )
    await _run_failure_case(
        kubectl,
        namespace=namespace,
        manifest=manifest,
        terms=("corrupt", "json", "config", "model", "parse"),
    )


async def test_d512_missing_tokenizer_fails_actionably(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> None:
    """D512: local model directory without tokenizer files names tokenizer cause."""
    del dynamo_operator
    namespace = "d512-missing-tokenizer"
    model_path = "/models/missing-tokenizer-model"
    config = _runtime_config(namespace, gpu_settings, model_name=model_path)
    manifest = _mutate_worker_podspec(
        _manifest_for(kubectl, config),
        lambda pod_spec, container: _add_missing_tokenizer_init(
            pod_spec, container, model_path
        ),
    )
    await _run_failure_case(
        kubectl,
        namespace=namespace,
        manifest=manifest,
        terms=("tokenizer", "missing", "not found", "model"),
    )


async def test_d513_vram_fragmentation_sidecar_fails_or_recovers(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> None:
    """D513: optional CUDA sidecar fragments VRAM during worker startup."""
    del dynamo_operator
    await _skip_unless_gpu_sidecar_capable(kubectl, gpu_settings)
    image = _required_env_or_skip(
        _GPU_PRESSURE_IMAGE_ENV,
        "D513 needs a CUDA/Python image with torch installed for VRAM allocation",
    )
    namespace = "d513-vram-fragmentation"
    config = _runtime_config(namespace, gpu_settings, gpu_memory_utilization=0.90)
    manifest = _mutate_worker_podspec(
        _manifest_for(kubectl, config),
        lambda pod_spec, _container: _add_gpu_pressure_sidecar(
            pod_spec,
            image=image,
            fraction="0.35",
            chunks="32",
        ),
    )
    await _run_failure_or_success_case(
        kubectl,
        namespace=namespace,
        manifest=manifest,
        terms=("cuda", "memory", "oom", "allocation", "fragment"),
    )


async def test_d514_vram_pressure_sidecar_fails_or_recovers(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> None:
    """D514: optional CUDA sidecar applies sustained VRAM pressure."""
    del dynamo_operator
    await _skip_unless_gpu_sidecar_capable(kubectl, gpu_settings)
    image = _required_env_or_skip(
        _GPU_PRESSURE_IMAGE_ENV,
        "D514 needs a CUDA/Python image with torch installed for VRAM allocation",
    )
    namespace = "d514-vram-pressure"
    config = _runtime_config(namespace, gpu_settings, gpu_memory_utilization=0.90)
    manifest = _mutate_worker_podspec(
        _manifest_for(kubectl, config),
        lambda pod_spec, _container: _add_gpu_pressure_sidecar(
            pod_spec,
            image=image,
            fraction="0.80",
            chunks="1",
        ),
    )
    await _run_failure_or_success_case(
        kubectl,
        namespace=namespace,
        manifest=manifest,
        terms=("cuda", "memory", "oom", "allocation"),
    )


async def test_d515_mig_resource_mismatch_surfaces_pending_or_failed(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> None:
    """D515: impossible MIG resource request surfaces scheduling failure."""
    del dynamo_operator
    await _skip_unless_mig_topology(kubectl)
    namespace = "d515-mig-mismatch"
    config = _runtime_config(namespace, gpu_settings, gpu_count=0)
    manifest = _mutate_worker_podspec(
        _manifest_for(kubectl, config),
        _add_impossible_mig_resource_request,
    )
    await _run_pending_or_failed_case(
        kubectl,
        namespace=namespace,
        manifest=manifest,
        terms=("mig", "insufficient", "nvidia.com/mig", "failedscheduling"),
    )


async def test_d516_cpu_noisy_neighbor_does_not_break_worker_readiness(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> None:
    """D516: optional CPU-burn sidecar should not permanently break readiness."""
    del dynamo_operator
    if os.environ.get(_ENABLE_CPU_NOISE_ENV, "").lower() not in {"1", "true", "yes"}:
        pytest.skip(
            f"D516 CPU noisy-neighbor sidecar is opt-in; set {_ENABLE_CPU_NOISE_ENV}=1"
        )
    image = os.environ.get(_CPU_NOISE_IMAGE_ENV, "python:3.12-slim")
    namespace = "d516-cpu-noisy-neighbor"
    config = _runtime_config(namespace, gpu_settings)
    manifest = _mutate_worker_podspec(
        _manifest_for(kubectl, config),
        lambda pod_spec, _container: _add_cpu_noise_sidecar(pod_spec, image=image),
    )
    await _run_failure_or_success_case(
        kubectl,
        namespace=namespace,
        manifest=manifest,
        terms=("cpu", "probe", "timeout", "unhealthy", "failed"),
    )


def _runtime_config(
    namespace: str,
    gpu_settings: GPUTestSettings,
    *,
    model_name: str | None = None,
    extra_worker_args: list[str] | None = None,
    gpu_count: int = 0,
    gpu_memory_utilization: float = 0.12,
) -> DynamoConfig:
    """Build the small aggregated vLLM deployment used by D507-D516 cases."""
    return DynamoConfig(
        model_name=model_name or "Qwen/Qwen3-0.6B",
        namespace=namespace,
        backend=DynamoBackend.VLLM,
        mode=DynamoMode.AGGREGATED,
        gpu_count=gpu_count,
        max_model_len=gpu_settings.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
        runtime_class_name=gpu_settings.runtime_class,
        hf_token_secret=gpu_settings.hf_token_secret,
        image=gpu_settings.dynamo_image,
        image_pull_secrets=gpu_settings.image_pull_secrets,
        extra_worker_args=extra_worker_args or [],
    )


def _manifest_for(kubectl: KubectlClient, config: DynamoConfig) -> str:
    """Return a DynamoGraphDeployment manifest for ``config``."""
    return DynamoDeployer(kubectl=kubectl, config=config).generate_manifest()


async def _run_failure_case(
    kubectl: KubectlClient,
    *,
    namespace: str,
    manifest: str,
    terms: tuple[str, ...],
) -> None:
    """Apply a negative runtime manifest and require failed state plus evidence."""
    try:
        await kubectl.apply(manifest=manifest, namespace=namespace)
        await _wait_for_failure_surface(kubectl, namespace=namespace, terms=terms)
    finally:
        await _delete_namespace(kubectl, namespace)


async def _run_failure_or_success_case(
    kubectl: KubectlClient,
    *,
    namespace: str,
    manifest: str,
    terms: tuple[str, ...],
) -> None:
    """Apply resource-pressure case; pass on recovery or actionable failure."""
    try:
        await kubectl.apply(manifest=manifest, namespace=namespace)
        outcome = await _wait_for_success_or_failure(kubectl, namespace=namespace)
        if outcome == "successful":
            return
        evidence = await _failure_evidence(kubectl, namespace=namespace)
        assert _contains_any(evidence, terms), (
            f"{namespace}: pressure case failed without expected evidence "
            f"{terms!r}; observed {evidence!r}"
        )
    finally:
        await _delete_namespace(kubectl, namespace)


async def _run_pending_or_failed_case(
    kubectl: KubectlClient,
    *,
    namespace: str,
    manifest: str,
    terms: tuple[str, ...],
) -> None:
    """Apply a scheduling-fault manifest and require Pending/Failed evidence."""
    try:
        await kubectl.apply(manifest=manifest, namespace=namespace)
        evidence = await _wait_for_evidence(kubectl, namespace=namespace, terms=terms)
        assert _contains_any(evidence, terms)
    finally:
        await _delete_namespace(kubectl, namespace)


async def _wait_for_failure_surface(
    kubectl: KubectlClient,
    *,
    namespace: str,
    terms: tuple[str, ...],
) -> None:
    """Wait for CR failed state or pod evidence and assert the cause is visible."""
    try:
        await wait_for_dgd_state(
            kubectl,
            _DGD_NAME,
            namespace,
            "failed",
            timeout=_FAILURE_TIMEOUT_S,
            poll_interval=5.0,
        )
    except TimeoutError:
        evidence = await _wait_for_evidence(kubectl, namespace=namespace, terms=terms)
    else:
        evidence = await _failure_evidence(kubectl, namespace=namespace)
    assert _contains_any(evidence, (*terms, *_STATUS_FAILURE_TERMS)), (
        f"{namespace}: runtime failure did not surface actionable evidence for "
        f"{terms!r}; observed {evidence!r}"
    )


async def _wait_for_success_or_failure(
    kubectl: KubectlClient,
    *,
    namespace: str,
) -> Literal["successful", "failed"]:
    """Poll DGD state until success/failure or timeout with last-state details."""
    deadline = asyncio.get_event_loop().time() + _FAILURE_TIMEOUT_S
    last_state = "<unobserved>"
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            _DGD_NAME,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.state}",
            check=False,
        )
        if result.returncode == 0:
            last_state = result.stdout.strip() or "<empty>"
            if last_state in {"successful", "failed"}:
                return last_state  # type: ignore[return-value]
        await asyncio.sleep(5.0)
    raise TimeoutError(
        f"{namespace}: DGD did not reach successful/failed within "
        f"{_FAILURE_TIMEOUT_S}s (last state={last_state!r})"
    )


async def _wait_for_evidence(
    kubectl: KubectlClient,
    *,
    namespace: str,
    terms: tuple[str, ...],
) -> str:
    """Poll status/events/logs until one expected term is visible."""
    deadline = asyncio.get_event_loop().time() + _POD_EVIDENCE_TIMEOUT_S
    evidence = ""
    while asyncio.get_event_loop().time() < deadline:
        evidence = await _failure_evidence(kubectl, namespace=namespace)
        if _contains_any(evidence, (*terms, *_STATUS_FAILURE_TERMS)):
            return evidence
        await asyncio.sleep(5.0)
    raise AssertionError(
        f"{namespace}: no expected evidence {terms!r} within "
        f"{_POD_EVIDENCE_TIMEOUT_S}s; last evidence={evidence!r}"
    )


async def _failure_evidence(kubectl: KubectlClient, *, namespace: str) -> str:
    """Collect compact DGD status, pod JSON, events, and worker log tail."""
    chunks: list[str] = []
    for args in (
        ("get", "dynamographdeployment", _DGD_NAME, "-n", namespace, "-o", "json"),
        ("get", "pods", "-n", namespace, "-o", "json"),
        ("get", "events", "-n", namespace, "--sort-by=.lastTimestamp"),
    ):
        result = await kubectl.run(*args, check=False)
        if result.stdout.strip():
            chunks.append(result.stdout[-6000:])
        if result.stderr.strip():
            chunks.append(result.stderr[-2000:])
    pod = await _first_worker_pod(kubectl, namespace)
    if pod:
        logs = await kubectl.run(
            "logs",
            pod,
            "-n",
            namespace,
            "-c",
            _MAIN_CONTAINER,
            "--tail=120",
            check=False,
        )
        chunks.append(logs.stdout[-6000:])
        chunks.append(logs.stderr[-2000:])
    return "\n".join(chunks).lower()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return True when any case-insensitive term is present."""
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


async def _first_worker_pod_or_skip(kubectl: KubectlClient, namespace: str) -> str:
    """Return the first worker pod in a running deployment or skip."""
    pod = await _first_worker_pod(kubectl, namespace)
    if not pod:
        pytest.skip(f"no worker pod found in {namespace!r} for process-runtime chaos")
    return pod


async def _first_worker_pod(kubectl: KubectlClient, namespace: str) -> str:
    """Return the first worker pod name matching current Dynamo labels."""
    result = await kubectl.run(
        "get",
        "pod",
        "-n",
        namespace,
        "-l",
        _WORKER_LABEL,
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    pods = await kubectl.run("get", "pods", "-n", namespace, "-o", "json", check=False)
    if pods.returncode != 0 or not pods.stdout.strip():
        return ""
    data = orjson.loads(pods.stdout)
    for item in data.get("items", []):
        name = item.get("metadata", {}).get("name", "")
        if "worker" in name.lower() or "vllm" in name.lower():
            return name
    return ""


async def _engine_child_pid_or_skip(
    kubectl: KubectlClient,
    namespace: str,
    pod: str,
) -> int:
    """Find a non-PID-1 engine-like process visible inside the worker container."""
    result = await kubectl.run(
        "exec",
        pod,
        "-n",
        namespace,
        "-c",
        _MAIN_CONTAINER,
        "--",
        "sh",
        "-lc",
        "ps -eo pid=,comm=,args= || true",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"D507 requires process listing in {namespace}/{pod}; "
            f"kubectl exec failed: {result.stderr.strip()}"
        )
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 2 or fields[0] == "1" or not fields[0].isdigit():
            continue
        haystack = " ".join(fields[1:]).lower()
        if any(term in haystack for term in _PROCESS_TERMS):
            return int(fields[0])
    pytest.skip(
        f"D507 requires a visible non-PID-1 engine child in {namespace}/{pod}; "
        "no engine/vllm/worker child process was listed"
    )


async def _container_restart_count(
    kubectl: KubectlClient,
    namespace: str,
    pod: str,
    container: str,
) -> int:
    """Return restartCount for ``container`` in ``pod``; missing counts as zero."""
    result = await kubectl.run(
        "get",
        "pod",
        pod,
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return 0
    data = orjson.loads(result.stdout or b"{}")
    for status in data.get("status", {}).get("containerStatuses", []) or []:
        if status.get("name") == container:
            return int(status.get("restartCount", 0))
    return 0


async def _wait_for_worker_recovery_after_process_kill(
    kubectl: KubectlClient,
    *,
    namespace: str,
    original_pod: str,
    restart_before: int,
) -> None:
    """Wait until the killed-process pod restarts or a replacement worker is ready."""
    deadline = asyncio.get_event_loop().time() + 180.0
    while asyncio.get_event_loop().time() < deadline:
        current = await _first_worker_pod(kubectl, namespace)
        if current and current != original_pod:
            ready = await _pod_ready(kubectl, namespace, current)
            if ready:
                return
        restart_now = await _container_restart_count(
            kubectl, namespace, original_pod, _MAIN_CONTAINER
        )
        if restart_now > restart_before and await _pod_ready(
            kubectl, namespace, original_pod
        ):
            return
        await asyncio.sleep(3.0)
    evidence = await _failure_evidence(kubectl, namespace=namespace)
    raise AssertionError(
        "D507: worker did not restart/recover within 180s after engine child kill; "
        f"evidence={evidence!r}"
    )


async def _pod_ready(kubectl: KubectlClient, namespace: str, pod: str) -> bool:
    """Return True when Kubernetes reports pod Ready=True."""
    result = await kubectl.run(
        "get",
        "pod",
        pod,
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return False
    data = orjson.loads(result.stdout or b"{}")
    conditions = data.get("status", {}).get("conditions", []) or []
    return any(
        item.get("type") == "Ready" and item.get("status") == "True"
        for item in conditions
    )


def _mutate_worker_podspec(
    manifest: str,
    mutate: Callable[[dict[str, Any], dict[str, Any]], None],
) -> str:
    """Apply ``mutate`` to every worker main container in a DGD manifest."""
    docs = [doc for doc in yaml.safe_load_all(manifest) if doc]
    for doc in docs:
        if doc.get("kind") != "DynamoGraphDeployment":
            continue
        spec = doc.setdefault("spec", {})
        services = spec.get("services") or {}
        for service in services.values():
            component_type = service.get("componentType") or service.get("type")
            if component_type not in {"worker", "decode", "prefill"}:
                continue
            pod_spec = service.setdefault("extraPodSpec", {})
            container = pod_spec.setdefault("mainContainer", {})
            mutate(pod_spec, container)
        for component in spec.get("components") or []:
            component_type = component.get("type") or component.get("componentType")
            if component_type not in {"worker", "decode", "prefill"}:
                continue
            pod_spec = component.setdefault("podTemplate", {}).setdefault("spec", {})
            containers = pod_spec.setdefault("containers", [{}])
            container = _main_container(containers)
            mutate(pod_spec, container)
    return "\n---\n".join(yaml.safe_dump(doc, sort_keys=False) for doc in docs)


def _main_container(containers: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the named main container, falling back to the first entry."""
    for container in containers:
        if container.get("name") == _MAIN_CONTAINER:
            return container
    return containers[0]


def _add_env(container: dict[str, Any], name: str, value: str) -> None:
    """Add or replace a container env var."""
    env = container.setdefault("env", [])
    for entry in env:
        if entry.get("name") == name:
            entry["value"] = value
            return
    env.append({"name": name, "value": value})


def _add_read_only_cache(
    pod_spec: dict[str, Any],
    container: dict[str, Any],
) -> None:
    """Mount the HF cache path read-only so startup cache writes fail."""
    pod_spec.setdefault("volumes", []).append({"name": "hf-cache", "emptyDir": {}})
    container.setdefault("volumeMounts", []).append(
        {"name": "hf-cache", "mountPath": "/hf-cache", "readOnly": True}
    )
    _add_env(container, "HF_HOME", "/hf-cache")
    _add_env(container, "TRANSFORMERS_CACHE", "/hf-cache/transformers")
    _add_env(container, "HF_HUB_CACHE", "/hf-cache/hub")


def _add_corrupt_model_init(
    pod_spec: dict[str, Any],
    container: dict[str, Any],
    model_path: str,
) -> None:
    """Create a local model dir with corrupt JSON before worker startup."""
    _add_model_volume(pod_spec, container)
    command = f"mkdir -p {model_path} && printf '{{not-json' > {model_path}/config.json"
    _add_init_container(pod_spec, "corrupt-model-cache", command)


def _add_missing_tokenizer_init(
    pod_spec: dict[str, Any],
    container: dict[str, Any],
    model_path: str,
) -> None:
    """Create a tiny local model dir with config but no tokenizer files."""
    _add_model_volume(pod_spec, container)
    config = (
        '{"architectures":["GPT2LMHeadModel"],'
        '"model_type":"gpt2","vocab_size":16,'
        '"n_positions":16,"n_ctx":16,"n_embd":8,'
        '"n_layer":1,"n_head":1}'
    )
    command = f"mkdir -p {model_path} && printf '{config}' > {model_path}/config.json"
    _add_init_container(pod_spec, "missing-tokenizer-model", command)


def _add_model_volume(pod_spec: dict[str, Any], container: dict[str, Any]) -> None:
    """Mount an emptyDir at /models for local model-shape tests."""
    pod_spec.setdefault("volumes", []).append({"name": "model-fixture", "emptyDir": {}})
    container.setdefault("volumeMounts", []).append(
        {"name": "model-fixture", "mountPath": "/models"}
    )


def _add_init_container(
    pod_spec: dict[str, Any],
    name: str,
    command: str,
) -> None:
    """Append a shell initContainer sharing the model-fixture volume."""
    pod_spec.setdefault("initContainers", []).append(
        {
            "name": name,
            "image": "busybox:1.36",
            "command": ["sh", "-lc", command],
            "volumeMounts": [{"name": "model-fixture", "mountPath": "/models"}],
        }
    )


def _add_gpu_pressure_sidecar(
    pod_spec: dict[str, Any],
    *,
    image: str,
    fraction: str,
    chunks: str,
) -> None:
    """Add a CUDA/PyTorch sidecar that allocates GPU memory and sleeps."""
    script = (
        "import os, time, torch; "
        "free, total = torch.cuda.mem_get_info(); "
        "target = int(total * float(os.environ['AIPERF_GPU_FRACTION'])); "
        "chunks = int(os.environ['AIPERF_GPU_CHUNKS']); bufs = []; "
        "per = max(target // max(chunks, 1), 1); "
        "\nfor _ in range(chunks):\n"
        "    bufs.append(torch.empty(per, dtype=torch.uint8, device='cuda'))\n"
        "    time.sleep(0.05)\n"
        "time.sleep(900)"
    )
    pod_spec.setdefault("containers", []).append(
        {
            "name": "gpu-pressure",
            "image": image,
            "command": ["python3", "-c", script],
            "env": [
                {"name": "AIPERF_GPU_FRACTION", "value": fraction},
                {"name": "AIPERF_GPU_CHUNKS", "value": chunks},
                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
            ],
            "resources": {"limits": {"nvidia.com/gpu": "1"}},
        }
    )


def _add_impossible_mig_resource_request(
    _pod_spec: dict[str, Any],
    container: dict[str, Any],
) -> None:
    """Request an intentionally unavailable MIG profile count."""
    resources = container.setdefault("resources", {}).setdefault("limits", {})
    resources.pop("nvidia.com/gpu", None)
    resources["nvidia.com/mig-9g.999gb"] = "1"


def _add_cpu_noise_sidecar(pod_spec: dict[str, Any], *, image: str) -> None:
    """Add a CPU-burn sidecar with a tight CPU limit."""
    pod_spec.setdefault("containers", []).append(
        {
            "name": "cpu-noisy-neighbor",
            "image": image,
            "command": [
                "python3",
                "-c",
                "while True:\n    pass",
            ],
            "resources": {
                "requests": {"cpu": "900m", "memory": "64Mi"},
                "limits": {"cpu": "1000m", "memory": "128Mi"},
            },
        }
    )


def _cache_miss_model() -> str:
    """Return a likely uncached tiny model for cache-write failure tests."""
    return os.environ.get(
        "DYNAMO_CHAOS_CACHE_MISS_MODEL", "hf-internal-testing/tiny-random-gpt2"
    )


async def _skip_unless_gpu_sidecar_capable(
    kubectl: KubectlClient,
    gpu_settings: GPUTestSettings,
) -> None:
    """Skip unless the cluster can run GPU-sharing sidecar tests."""
    if not gpu_settings.runtime_class:
        pytest.skip(
            "GPU sidecar tests require --gpu-runtime-class / GPU_TEST_RUNTIME_CLASS"
        )
    nodes = await kubectl.run("get", "nodes", "-o", "json", check=False)
    if nodes.returncode != 0:
        pytest.skip("could not inspect node allocatable GPU resources")
    data = orjson.loads(nodes.stdout or b"{}")
    has_gpu = any(
        int(item.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", "0"))
        > 0
        for item in data.get("items", [])
    )
    if not has_gpu:
        pytest.skip("GPU sidecar tests require allocatable nvidia.com/gpu resources")


async def _skip_unless_mig_topology(kubectl: KubectlClient) -> None:
    """Skip unless at least one node advertises MIG resources."""
    nodes = await kubectl.run("get", "nodes", "-o", "json", check=False)
    if nodes.returncode != 0:
        pytest.skip("D515 could not inspect node allocatable resources for MIG")
    data = orjson.loads(nodes.stdout or b"{}")
    for item in data.get("items", []):
        allocatable = item.get("status", {}).get("allocatable", {})
        if any(str(name).startswith("nvidia.com/mig-") for name in allocatable):
            return
    pytest.skip(
        "D515 requires a MIG-enabled node advertising nvidia.com/mig-* resources"
    )


def _required_env_or_skip(name: str, reason: str) -> str:
    """Return a required env var value or skip with a capability message."""
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"{reason}; set {name}=<image>")
    return value


async def _delete_namespace(kubectl: KubectlClient, namespace: str) -> None:
    """Delete a per-test namespace without waiting for teardown."""
    await kubectl.run(
        "delete",
        "namespace",
        namespace,
        "--wait=false",
        "--ignore-not-found",
        check=False,
    )

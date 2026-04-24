# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for Dynamo GPU E2E tests.

Provides Dynamo operator installation, server deployment, endpoint URL
resolution, and related fixtures used by all tests in the ``gpu/dynamo/``
subtree.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.conftest import (
    GPUTestSettings,
    _dump_diagnostics,
    _log_events,
    _log_pod_statuses,
    _release_gpu,
)
from tests.kubernetes.gpu.dynamo.helpers import (
    DynamoBackend,
    DynamoConfig,
    DynamoDeployer,
    DynamoMode,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


# ============================================================================
# Dynamo operator helpers
# ============================================================================


async def _run_streaming(cmd: list[str], error_msg: str, prefix: str = "HELM") -> None:
    """Run a command, streaming output line-by-line to the logger."""
    logger.info(f"Running: {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    async for raw_line in proc.stdout:
        line = raw_line.decode().rstrip()
        output_lines.append(line)
        logger.info(f"[{prefix}] {line}")
    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{error_msg}:\n" + "\n".join(output_lines))


async def _dynamo_operator_is_running(kubectl: KubectlClient) -> bool:
    """Check if the Dynamo operator pod is Running."""
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        "dynamo-system",
        "-l",
        "app.kubernetes.io/name=dynamo-operator",
        "--no-headers",
        check=False,
    )
    return (
        result.returncode == 0
        and bool(result.stdout.strip())
        and "Running" in result.stdout
    )


async def _uninstall_bad_dynamo_releases(kubectl: KubectlClient) -> None:
    """Uninstall failed or stuck Dynamo helm releases so we can reinstall cleanly."""
    helm_ctx: list[str] = []
    if kubectl.context:
        helm_ctx = ["--kube-context", kubectl.context]

    bad_statuses = {"failed", "pending-install", "pending-upgrade"}

    for release, namespace in [
        ("dynamo-platform", "dynamo-system"),
        ("dynamo-crds", "default"),
    ]:
        proc = await asyncio.create_subprocess_exec(
            "helm",
            "status",
            release,
            *helm_ctx,
            "-n",
            namespace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            continue
        status_text = stdout.decode().lower()
        if any(s in status_text for s in bad_statuses):
            logger.info(f"Uninstalling bad helm release: {release}")
            await _run_streaming(
                ["helm", "uninstall", release, *helm_ctx, "-n", namespace],
                f"Failed to uninstall {release}",
            )


async def _patch_dynamo_discovery_rbac(kubectl: KubectlClient, namespace: str) -> None:
    """Patch the Dynamo service-discovery Role to include dynamoworkermetadatas.

    When the webhook is disabled, the operator creates the k8s-service-discovery
    Role with only endpoints/endpointslices. Workers using kubernetes discovery
    also need access to the dynamoworkermetadatas CRD.
    """
    role_name = None
    delay = 0.5
    for _ in range(12):
        result = await kubectl.run(
            "get",
            "role",
            "-n",
            namespace,
            "--no-headers",
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                name = line.split()[0]
                if "service-discovery" in name:
                    role_name = name
                    break
        if role_name:
            break
        await asyncio.sleep(delay)
        delay = min(delay * 2, 5)

    if not role_name:
        logger.warning("[DYNAMO] Could not find service-discovery Role to patch")
        return

    patch = (
        '[{"op":"add","path":"/rules/-","value":'
        '{"apiGroups":["nvidia.com"],'
        '"resources":["dynamoworkermetadatas"],'
        '"verbs":["get","list","watch","create","update","patch","delete"]}}]'
    )
    result = await kubectl.run(
        "patch",
        "role",
        role_name,
        "-n",
        namespace,
        "--type=json",
        f"-p={patch}",
        check=False,
    )
    if result.returncode == 0:
        logger.info(f"[DYNAMO] Patched {role_name} with dynamoworkermetadatas access")
    else:
        logger.warning(f"[DYNAMO] Failed to patch {role_name}: {result.stderr}")


# ============================================================================
# Dynamo operator fixture
# ============================================================================


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def dynamo_operator(
    kubectl: KubectlClient, gpu_settings: GPUTestSettings
) -> None:
    """Ensure the Dynamo operator is running on the cluster.

    Checks both the CRD and the operator pod. If the CRD exists but the
    operator isn't running (e.g. failed helm install), cleans up and reinstalls.
    """
    # Fast path: CRD exists and operator pod is running
    crd_result = await kubectl.run(
        "get", "crd", "dynamographdeployments.nvidia.com", check=False
    )
    crd_exists = crd_result.returncode == 0

    if crd_exists and await _dynamo_operator_is_running(kubectl):
        logger.info("Dynamo operator already installed and running")
        return

    if crd_exists:
        logger.info("Dynamo CRD exists but operator is not running - reinstalling")
        await _uninstall_bad_dynamo_releases(kubectl)

    dynamo_version = gpu_settings.dynamo_version
    logger.info(f"Installing Dynamo operator {dynamo_version} via Helm")

    ngc_base = "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts"
    crds_tgz = f"dynamo-crds-{dynamo_version}.tgz"
    platform_tgz = f"dynamo-platform-{dynamo_version}.tgz"

    helm_ctx: list[str] = []
    if kubectl.context:
        helm_ctx = ["--kube-context", kubectl.context]

    # Fetch and install CRDs (skip if already present)
    if not crd_exists:
        await _run_streaming(
            ["helm", "fetch", f"{ngc_base}/{crds_tgz}"],
            "Failed to fetch Dynamo CRDs chart",
        )
        try:
            await _run_streaming(
                [
                    "helm",
                    "install",
                    "dynamo-crds",
                    crds_tgz,
                    *helm_ctx,
                    "--namespace",
                    "default",
                    "--timeout",
                    "120s",
                ],
                "Failed to install Dynamo CRDs",
            )
        finally:
            Path(crds_tgz).unlink(missing_ok=True)

    # --gpu-local-keygen: create the MPI SSH secret locally instead of via the
    # Helm pre-install hook (which pulls bitnamisecure/git + alpine/k8s images
    # that may be unreachable behind corporate proxies).
    local_keygen = gpu_settings.local_keygen
    if local_keygen:
        await kubectl.run(
            "create",
            "namespace",
            "dynamo-system",
            check=False,
        )
        mpi_secret_exists = (
            await kubectl.run(
                "get",
                "secret",
                "mpi-run-ssh-secret",
                "-n",
                "dynamo-system",
                check=False,
            )
        ).returncode == 0
        if not mpi_secret_exists:
            logger.info("Creating MPI SSH keypair secret locally (--gpu-local-keygen)")
            key_path = "/tmp/dynamo-mpi-key"
            await _run_streaming(
                ["ssh-keygen", "-t", "rsa", "-b", "2048", "-f", key_path, "-N", ""],
                "Failed to generate SSH keypair",
                prefix="KEYGEN",
            )
            await kubectl.run(
                "create",
                "secret",
                "generic",
                "mpi-run-ssh-secret",
                f"--from-file=private.key={key_path}",
                f"--from-file=private.key.pub={key_path}.pub",
                "-n",
                "dynamo-system",
            )
            Path(key_path).unlink(missing_ok=True)
            Path(f"{key_path}.pub").unlink(missing_ok=True)

    # Fetch and install platform
    await _run_streaming(
        ["helm", "fetch", f"{ngc_base}/{platform_tgz}"],
        "Failed to fetch Dynamo platform chart",
    )
    helm_sets = [
        "--set", "dynamo-operator.webhook.enabled=false",
        "--set", "grove.enabled=false",
        "--set", "kai-scheduler.enabled=false",
        # gcr.io/kubebuilder/kube-rbac-proxy:v0.15.0 was removed;
        # use the registry.k8s.io mirror instead.
        "--set", "dynamo-operator.controllerManager.kubeRbacProxy.image.repository=registry.k8s.io/kubebuilder/kube-rbac-proxy",
    ]  # fmt: skip
    if local_keygen:
        helm_sets += ["--set", "dynamo-operator.dynamo.mpiRun.sshKeygen.enabled=false"]
    try:
        await _run_streaming(
            [
                "helm",
                "install",
                "dynamo-platform",
                platform_tgz,
                *helm_ctx,
                "--namespace",
                "dynamo-system",
                "--create-namespace",
                *helm_sets,
                "--timeout",
                "600s",
            ],
            "Failed to install Dynamo platform",
        )
    finally:
        Path(platform_tgz).unlink(missing_ok=True)

    # Wait for operator pod to be Running (exponential backoff: 0.5s -> 10s cap)
    logger.info("Waiting for Dynamo operator to be ready...")
    elapsed = 0.0
    delay = 0.5
    iteration = 0
    while elapsed < 300:
        if await _dynamo_operator_is_running(kubectl):
            logger.info(f"Dynamo operator ready (waited ~{elapsed:.0f}s)")
            return
        iteration += 1
        if iteration % 6 == 0:
            await _log_pod_statuses(kubectl, "dynamo-system")
            await _log_events(kubectl, "dynamo-system")
        elif iteration % 10 == 0:
            logger.info(f"Still waiting for Dynamo operator... ({elapsed:.0f}s/300s)")
        await asyncio.sleep(delay)
        elapsed += delay
        delay = min(delay * 1.5, 10)

    # Final diagnostic dump before giving up
    await _dump_diagnostics(kubectl, "dynamo-system", label="DYNAMO_OPERATOR_TIMEOUT")
    raise RuntimeError(
        "Dynamo operator failed to start within 300s - "
        "check 'kubectl get pods -n dynamo-system' and 'kubectl get events -n dynamo-system'"
    )


# ============================================================================
# Dynamo deployment fixtures
# ============================================================================


@pytest.fixture(scope="package")
def dynamo_config(gpu_settings: GPUTestSettings) -> DynamoConfig:
    """Create Dynamo configuration from settings.

    When mode is ``disagg-1gpu``, uses :meth:`DynamoConfig.single_gpu_disagg`
    with settings overrides (matching ``dev/kube.py``'s single-GPU behavior).
    """
    s = gpu_settings
    mode = DynamoMode(s.dynamo_mode)
    backend = DynamoBackend(s.dynamo_backend)
    connectors = (
        [c.strip() for c in s.dynamo_connectors.split(",") if c.strip()]
        if s.dynamo_connectors
        else []
    )

    common_overrides: dict = {
        "model_name": s.model,
        "backend": backend,
        "tolerations": s.tolerations,
        "node_selector": s.node_selector,
        "hf_token_secret": s.hf_token_secret,
        "image_pull_secrets": s.image_pull_secrets,
        "connectors": connectors,
        "kvbm_cpu_cache_gb": s.dynamo_kvbm_cpu_cache_gb,
    }
    if s.dynamo_image is not None:
        common_overrides["image"] = s.dynamo_image

    if mode == DynamoMode.DISAGGREGATED_1GPU:
        return DynamoConfig.single_gpu_disagg(**common_overrides)

    return DynamoConfig(
        **common_overrides,
        mode=mode,
        gpu_count=s.count,
        max_model_len=s.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=s.mem_util,
        runtime_class_name=s.runtime_class,
    )


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def dynamo_server(
    kubectl: KubectlClient,
    dynamo_config: DynamoConfig,
    gpu_settings: GPUTestSettings,
    dynamo_operator: None,
) -> AsyncGenerator[DynamoDeployer | str, None]:
    """Deploy Dynamo or use existing endpoint.

    If --gpu-dynamo-endpoint is set, skips deployment and yields the URL.
    Otherwise deploys Dynamo and yields the deployer.
    """
    s = gpu_settings
    if s.dynamo_endpoint:
        logger.info(f"Using existing Dynamo endpoint: {s.dynamo_endpoint}")
        yield s.dynamo_endpoint
        return

    await _release_gpu(kubectl, dynamo_config.namespace)
    deployer = DynamoDeployer(kubectl=kubectl, config=dynamo_config)

    logger.info(
        f"Deploying Dynamo: model={dynamo_config.model_name}, image={dynamo_config.image}, mode={dynamo_config.mode.value}, namespace={dynamo_config.namespace}"
    )

    await deployer.deploy()

    # Patch the service-discovery Role to include dynamoworkermetadatas access.
    # The operator creates this Role with only endpoints/endpointslices, but the
    # kubernetes discovery backend also needs dynamoworkermetadatas CRD access.
    # (The webhook normally handles this, but we disable webhooks in test.)
    await _patch_dynamo_discovery_rbac(kubectl, dynamo_config.namespace)

    logger.info(
        f"[DYNAMO] Waiting for readiness (timeout={s.dynamo_deploy_timeout}s)..."
    )
    try:
        await deployer.wait_for_ready(
            timeout=s.dynamo_deploy_timeout,
            stream_logs=s.stream_logs,
        )
        logger.info(f"[DYNAMO] Server is ready at {deployer.get_endpoint_url()}")
        dynamo_logs = await deployer.get_logs(tail=30)
        logger.info(f"[DYNAMO] Recent server logs:\n{dynamo_logs}")
    except TimeoutError:
        logger.error("[DYNAMO] Server failed to become ready!")
        await _dump_diagnostics(
            kubectl, dynamo_config.namespace, label="DYNAMO_FAILURE"
        )
        raise

    yield deployer

    if not s.skip_cleanup:
        logger.info(f"[DYNAMO] Cleaning up namespace {dynamo_config.namespace}")
        await deployer.cleanup()
    else:
        logger.info("Skipping Dynamo cleanup (--k8s-skip-cleanup)")


@pytest.fixture(scope="package")
def dynamo_endpoint_url(
    dynamo_server: DynamoDeployer | str,
) -> str:
    """Get the Dynamo endpoint URL."""
    if isinstance(dynamo_server, str):
        return dynamo_server
    return dynamo_server.get_endpoint_url()

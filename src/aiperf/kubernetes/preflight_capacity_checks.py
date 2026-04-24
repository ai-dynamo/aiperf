# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster capacity preflight checks (quotas, nodes, secrets, image).

Split out of ``preflight.py`` / ``preflight_checks.py`` to keep each module
under the ergonomics file-size limit. All functions here are stateless — they
take an ``ApiClient`` plus any config they need and return a ``CheckResult``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.environment import (
    CONTROLLER_RESOURCE_KEYS,
    K8sEnvironment,
)
from aiperf.kubernetes.preflight_utils import (
    parse_image_ref as _shared_parse_image_ref,
)
from aiperf.kubernetes.utils import (
    format_cpu,
    format_memory,
    parse_cpu,
    parse_memory_gib,
)

if TYPE_CHECKING:
    from aiperf.kubernetes.preflight import CheckResult

_PUBLIC_REGISTRIES: frozenset[str] = frozenset(
    {
        "docker.io",
        "registry-1.docker.io",
        "ghcr.io",
        "quay.io",
        "nvcr.io",
        "registry.k8s.io",
    }
)

_CLUSTER_API_ERRORS: tuple[type[BaseException], ...] = (
    ApiException,
    aiohttp.ClientError,
    asyncio.TimeoutError,
    OSError,
    RuntimeError,
)


def _controller_resource_requirements() -> tuple[float, float]:
    """Return total controller-pod CPU cores and memory GiB."""
    cpu = 0.0
    memory = 0.0
    for key in CONTROLLER_RESOURCE_KEYS:
        settings = getattr(K8sEnvironment, key)
        cpu += parse_cpu(settings.CPU)
        memory += parse_memory_gib(settings.MEMORY)
    return cpu, memory


@dataclass(slots=True)
class _QuotaEvaluation:
    """Outcome of evaluating resource quotas against a deployment's needs."""

    details: list[str]
    would_exceed: bool


def _evaluate_quotas(
    quotas: list, *, required_cpu: float, required_mem: float
) -> _QuotaEvaluation:
    """Build detail lines and flag whether any quota would be exceeded."""
    details: list[str] = []
    would_exceed = False
    for quota in quotas:
        name = quota.metadata.name if quota.metadata else ""
        details.append(f"ResourceQuota '{name}':")
        hard = (quota.status.hard or {}) if quota.status else {}
        used = (quota.status.used or {}) if quota.status else {}
        for resource, limit in hard.items():
            details.append(f"    {resource}: {used.get(resource, '0')} / {limit}")

        hard_cpu = hard.get("cpu") or hard.get("requests.cpu")
        hard_mem = hard.get("memory") or hard.get("requests.memory")
        used_cpu = used.get("cpu") or used.get("requests.cpu")
        used_mem = used.get("memory") or used.get("requests.memory")

        if hard_cpu:
            total_needed = required_cpu + parse_cpu(used_cpu or "0")
            if total_needed > parse_cpu(hard_cpu):
                would_exceed = True
                details.append(
                    f"    -> CPU would exceed quota: "
                    f"{format_cpu(total_needed)} needed vs "
                    f"{hard_cpu} limit"
                )
        if hard_mem:
            total_needed = required_mem + parse_memory_gib(used_mem or "0")
            if total_needed > parse_memory_gib(hard_mem):
                would_exceed = True
                details.append(
                    f"    -> Memory would exceed quota: "
                    f"{format_memory(total_needed)} needed vs "
                    f"{hard_mem} limit"
                )
    return _QuotaEvaluation(details=details, would_exceed=would_exceed)


async def check_resource_quotas(
    api: ApiClient, *, namespace: str, workers: int
) -> CheckResult:
    """Check resource quotas in the namespace."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    try:
        quotas = (
            await client.CoreV1Api(api).list_namespaced_resource_quota(namespace)
        ).items

        if not quotas:
            return CheckResult(
                name="Resource Quotas",
                status=CheckStatus.PASS,
                message="No resource quotas configured",
            )

        ctrl_cpu, ctrl_mem = _controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)
        required_cpu = ctrl_cpu + (worker_cpu * workers)
        required_mem = ctrl_mem + (worker_mem * workers)

        evaluation = _evaluate_quotas(
            quotas, required_cpu=required_cpu, required_mem=required_mem
        )

        if evaluation.would_exceed:
            evaluation.details.append(
                f"Benchmark needs: {format_cpu(required_cpu)} CPU, "
                f"{format_memory(required_mem)} memory ({workers} workers)"
            )
            return CheckResult(
                name="Resource Quotas",
                status=CheckStatus.WARN,
                message="Benchmark may exceed resource quota(s)",
                details=evaluation.details,
                hints=[
                    "Request a quota increase or reduce worker count",
                    "Quota may not apply if benchmark creates its own namespace",
                ],
            )

        return CheckResult(
            name="Resource Quotas",
            status=CheckStatus.INFO,
            message=f"Found {len(quotas)} resource quota(s)",
            details=evaluation.details,
        )
    except ApiException as e:
        return CheckResult(
            name="Resource Quotas",
            status=CheckStatus.WARN,
            message=f"Error checking quotas: HTTP {e.status}",
        )


def _node_is_ready(node) -> bool:
    """Return True if a node's Ready condition is True."""
    conditions = (node.status.conditions or []) if node.status else []
    return any(c.type == "Ready" and c.status == "True" for c in conditions)


def _aggregate_ready_nodes(nodes: list) -> tuple[int, float, float]:
    """Return (ready_count, total_cpu_cores, total_memory_gib) across ready nodes."""
    ready_nodes = 0
    total_cpu = 0.0
    total_memory = 0.0
    for node in nodes:
        allocatable = (node.status.allocatable or {}) if node.status else {}
        if _node_is_ready(node) and allocatable:
            ready_nodes += 1
            total_cpu += parse_cpu(allocatable.get("cpu", "0"))
            total_memory += parse_memory_gib(allocatable.get("memory", "0"))
    return ready_nodes, total_cpu, total_memory


def _any_node_fits(nodes: list, *, max_pod_cpu: float, max_pod_mem: float) -> bool:
    """Return True if at least one ready node can fit a pod of the given size."""
    for node in nodes:
        if not _node_is_ready(node):
            continue
        allocatable = (node.status.allocatable or {}) if node.status else {}
        node_cpu = parse_cpu(allocatable.get("cpu", "0"))
        node_mem = parse_memory_gib(allocatable.get("memory", "0"))
        if node_cpu >= max_pod_cpu and node_mem >= max_pod_mem:
            return True
    return False


async def check_node_resources(api: ApiClient, *, workers: int) -> CheckResult:
    """Check if cluster has sufficient node resources."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    try:
        nodes = (await client.CoreV1Api(api).list_node()).items

        if not nodes:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.FAIL,
                message="No nodes found in cluster",
            )

        ready_nodes, total_cpu, total_memory = _aggregate_ready_nodes(nodes)

        ctrl_cpu, ctrl_mem = _controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)

        required_cpu = ctrl_cpu + (worker_cpu * workers)
        required_mem = ctrl_mem + (worker_mem * workers)

        details = [
            f"Cluster: {ready_nodes} ready nodes, "
            f"{format_cpu(total_cpu)} CPU, {format_memory(total_memory)} memory",
            f"Deployment estimate: {format_cpu(required_cpu)} CPU, "
            f"{format_memory(required_mem)} memory ({workers} workers)",
        ]

        if required_cpu > total_cpu or required_mem > total_memory:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message="Cluster may not have enough resources",
                details=details,
                hints=["Consider reducing worker count or adding cluster capacity"],
            )

        max_pod_cpu = max(ctrl_cpu, worker_cpu)
        max_pod_mem = max(ctrl_mem, worker_mem)
        if not _any_node_fits(nodes, max_pod_cpu=max_pod_cpu, max_pod_mem=max_pod_mem):
            details.append(
                f"Largest single-pod requirement: "
                f"{format_cpu(max_pod_cpu)} CPU, {format_memory(max_pod_mem)} memory"
            )
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.FAIL,
                message="No single node can fit even one pod",
                details=details,
                hints=[
                    "Each node must have enough allocatable resources for at least one pod",
                    f"Minimum per-node: {format_cpu(max_pod_cpu)} CPU, "
                    f"{format_memory(max_pod_mem)} memory",
                ],
            )

        return CheckResult(
            name="Node Resources",
            status=CheckStatus.PASS,
            message=f"Cluster has sufficient resources ({ready_nodes} nodes)",
            details=details,
        )
    except _CLUSTER_API_ERRORS as e:
        return CheckResult(
            name="Node Resources",
            status=CheckStatus.WARN,
            message=f"Could not check node resources: {e}",
        )


@dataclass(slots=True)
class _SecretClassification:
    """Classified secret names after attempting to read each from the API."""

    found: list[str]
    missing: list[str]
    permission_denied: list[str]


async def _classify_secrets(
    api: ApiClient, *, namespace: str, secret_names: list[str]
) -> _SecretClassification:
    """Read each secret and classify into found / missing / permission_denied."""
    found: list[str] = []
    missing: list[str] = []
    permission_denied: list[str] = []

    core = client.CoreV1Api(api)
    for secret_name in secret_names:
        try:
            await core.read_namespaced_secret(secret_name, namespace)
            found.append(secret_name)
        except ApiException as e:
            if e.status == 404:
                missing.append(secret_name)
            elif e.status == 403:
                permission_denied.append(secret_name)
            else:
                missing.append(f"{secret_name} (error: HTTP {e.status})")
    return _SecretClassification(
        found=found, missing=missing, permission_denied=permission_denied
    )


async def check_secrets(
    api: ApiClient,
    *,
    namespace: str,
    image_pull_secrets: list[str],
    secrets: list[str],
) -> CheckResult:
    """Check if required secrets exist."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    all_secrets = image_pull_secrets + secrets
    if not all_secrets:
        return CheckResult(
            name="Secrets",
            status=CheckStatus.SKIP,
            message="No secrets specified to verify",
            hints=["Use --image-pull-secret or --secret to verify specific secrets"],
        )

    classified = await _classify_secrets(
        api, namespace=namespace, secret_names=all_secrets
    )

    details: list[str] = []
    if classified.found:
        details.extend([f"  ✓ {s}" for s in classified.found])
    if classified.missing:
        details.extend([f"  ✗ {s} (not found)" for s in classified.missing])
    if classified.permission_denied:
        details.extend(
            [f"  ? {s} (permission denied)" for s in classified.permission_denied]
        )

    if classified.missing:
        return CheckResult(
            name="Secrets",
            status=CheckStatus.FAIL,
            message=f"{len(classified.missing)} secret(s) not found",
            details=details,
            hints=["Create missing secrets with 'kubectl create secret ...'"],
        )
    if classified.permission_denied:
        return CheckResult(
            name="Secrets",
            status=CheckStatus.WARN,
            message=f"Cannot verify {len(classified.permission_denied)} secret(s)",
            details=details,
        )
    return CheckResult(
        name="Secrets",
        status=CheckStatus.PASS,
        message=f"All {len(classified.found)} secret(s) verified",
        details=details,
    )


async def check_image(
    api: ApiClient,
    *,
    image: str | None,
    image_pull_secrets: list[str],
) -> CheckResult:
    """Check image availability information."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    if not image:
        return CheckResult(
            name="Image Pull",
            status=CheckStatus.SKIP,
            message="No image specified to verify",
            hints=["Use --image to check pull access"],
        )

    details = [f"Image: {image}"]
    registry, _repo, tag = _shared_parse_image_ref(image)
    if tag:
        details.append(f"Registry: {registry}, Tag: {tag}")
    else:
        details.append(f"Registry: {registry}, Tag: latest (implicit)")

    if image_pull_secrets:
        details.append(f"Pull secrets: {', '.join(image_pull_secrets)}")
        return CheckResult(
            name="Image Pull",
            status=CheckStatus.PASS,
            message="Image specified with pull secrets configured",
            details=details,
        )

    if registry in _PUBLIC_REGISTRIES:
        details.append(f"Public registry: {registry}")
        return CheckResult(
            name="Image Pull",
            status=CheckStatus.INFO,
            message=f"Image from public registry ({registry})",
            details=details,
            hints=[
                f"Verify manually: kubectl run test --image={image} "
                "--rm -it --restart=Never -- echo ok"
            ],
        )

    return CheckResult(
        name="Image Pull",
        status=CheckStatus.WARN,
        message="Image may require pull secrets",
        details=details,
        hints=[
            f"Registry '{registry}' may require authentication",
            "Use --image-pull-secret to specify credentials",
            f"Verify manually: kubectl run test --image={image} "
            "--rm -it --restart=Never -- echo ok",
        ],
    )

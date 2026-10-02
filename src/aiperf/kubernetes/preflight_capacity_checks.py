# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster capacity preflight checks (quotas, nodes, secrets, image).

Split out of ``preflight.py`` / ``preflight_checks.py`` to keep each module
under the ergonomics file-size limit. All functions here are stateless — they
take an ``ApiClient`` plus any config they need and return a ``CheckResult``.
"""

from __future__ import annotations

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
from aiperf.kubernetes.resource_quota import quota_violation
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
    TimeoutError,
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
    violations: list[str]

    @property
    def would_exceed(self) -> bool:
        """Return whether any quota rejects the planned workload."""
        return bool(self.violations)


def _evaluate_quotas(
    quotas: list,
    *,
    required_cpu: float,
    required_mem: float,
    required_pods: int = 0,
) -> _QuotaEvaluation:
    """Build quota detail lines and collect admission violations."""
    details: list[str] = []
    violations: list[str] = []
    for quota in quotas:
        name = quota.metadata.name if quota.metadata else ""
        details.append(f"ResourceQuota '{name}':")
        spec = getattr(quota, "spec", None)
        hard = (
            (spec.hard or {})
            if spec
            else (quota.status.hard or {})
            if quota.status
            else {}
        )
        used = (quota.status.used or {}) if quota.status else {}
        for resource, limit in hard.items():
            details.append(f"    {resource}: {used.get(resource, '0')} / {limit}")

        if violation := quota_violation(
            hard,
            used,
            required_cpu=required_cpu,
            required_mem=required_mem,
            required_pods=required_pods,
        ):
            violations.append(violation)
            details.append(f"    -> {violation}")
    return _QuotaEvaluation(details=details, violations=violations)


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
            quotas,
            required_cpu=required_cpu,
            required_mem=required_mem,
            required_pods=workers + 1,
        )

        if evaluation.would_exceed:
            evaluation.details.append(
                f"Benchmark needs: {format_cpu(required_cpu)} CPU, "
                f"{format_memory(required_mem)} memory ({workers + 1} pods)"
            )
            return CheckResult(
                name="Resource Quotas",
                status=CheckStatus.FAIL,
                message=f"Benchmark would exceed {evaluation.violations[0]}",
                details=evaluation.details,
                hints=["Request a quota increase or reduce worker count"],
            )

        return CheckResult(
            name="Resource Quotas",
            status=CheckStatus.INFO,
            message=f"Found {len(quotas)} resource quota(s)",
            details=evaluation.details,
        )
    except _CLUSTER_API_ERRORS as e:
        # Siblings catch the whole cluster-error tuple; this one caught only
        # ApiException, so a transport hiccup aborted preflight instead of
        # warning.
        status = getattr(e, "status", None)
        detail = f"HTTP {status}" if status is not None else f"{type(e).__name__}: {e}"
        return CheckResult(
            name="Resource Quotas",
            status=CheckStatus.WARN,
            message=f"Error checking quotas: {detail}",
        )
    except ValueError as e:
        # Quota values are user-authored, so an unparsable quantity is their
        # typo, not a cluster problem -- and never a reason to block a
        # benchmark on an otherwise healthy cluster.
        return CheckResult(
            name="Resource Quotas",
            status=CheckStatus.WARN,
            message=f"Could not interpret a quota value: {e}",
        )


def _node_is_ready(node) -> bool:
    """Return True if a node's Ready condition is True."""
    conditions = (node.status.conditions or []) if node.status else []
    return any(c.type == "Ready" and c.status == "True" for c in conditions)


def _node_is_schedulable(node) -> bool:
    """Return True if the node is Ready and not cordoned.

    A cordoned node (``spec.unschedulable``) still reports Ready and still
    advertises its allocatable capacity, so counting it makes a drained
    cluster look like it has room the scheduler will never hand out.
    """
    spec = getattr(node, "spec", None)
    if spec is not None and getattr(spec, "unschedulable", False):
        return False
    return _node_is_ready(node)


def _toleration_matches_taint(
    toleration: dict[str, object], taint_key: str, taint_effect: str
) -> bool:
    """Return whether a toleration admits a NoSchedule or NoExecute taint."""
    operator = toleration.get("operator") or "Equal"
    toleration_key = toleration.get("key") or ""
    toleration_effect = toleration.get("effect") or ""
    if operator == "Exists" and not toleration_key:
        return not toleration_effect or toleration_effect == taint_effect
    if toleration_key != taint_key:
        return False
    return not toleration_effect or toleration_effect == taint_effect


def _node_matches_placement(
    node,
    *,
    node_selector: dict[str, str],
    tolerations: list[dict[str, object]],
) -> bool:
    """Return whether a schedulable node matches the requested placement."""
    if not _node_is_schedulable(node):
        return False
    metadata = getattr(node, "metadata", None)
    labels = getattr(metadata, "labels", None) or {}
    if not all(labels.get(key) == value for key, value in node_selector.items()):
        return False
    spec = getattr(node, "spec", None)
    for taint in (getattr(spec, "taints", None) or []) if spec else []:
        effect = getattr(taint, "effect", None) or ""
        if effect not in ("NoSchedule", "NoExecute"):
            continue
        key = getattr(taint, "key", None) or ""
        if not any(
            _toleration_matches_taint(toleration, key, effect)
            for toleration in tolerations
        ):
            return False
    return True


def _pod_requests(pod) -> tuple[float, float]:
    """Return (cpu cores, memory GiB) requested by a pod's regular containers.

    Init containers are excluded: they have completed by the time the pod is
    Running, so their requests are not part of its steady-state footprint.
    """
    spec = getattr(pod, "spec", None)
    cpu = 0.0
    memory = 0.0
    for container in (getattr(spec, "containers", None) or []) if spec else []:
        resources = getattr(container, "resources", None)
        requests = getattr(resources, "requests", None) or {}
        try:
            cpu += parse_cpu(requests.get("cpu", "0"))
            memory += parse_memory_gib(requests.get("memory", "0"))
        except (ValueError, AttributeError):
            # A malformed quantity is one pod's problem, not a reason to
            # abandon the whole capacity estimate.
            continue
    return cpu, memory


async def _requested_by_node(api: ApiClient) -> dict[str, tuple[float, float]] | None:
    """Sum running-pod requests per node name.

    Returns ``None`` when the pod list is unavailable (commonly a CLI user
    without cluster-wide pod read), so the caller can fall back to raw
    allocatable and say so rather than silently claiming free capacity.
    """
    try:
        pod_list = await client.CoreV1Api(api).list_pod_for_all_namespaces(
            field_selector="status.phase=Running"
        )
        pods = getattr(pod_list, "items", None)
    except (*_CLUSTER_API_ERRORS, TypeError, AttributeError):
        # Free-capacity data is an improvement on allocatable, never a reason
        # to abort preflight: any client that cannot answer degrades to the
        # allocatable-only estimate, which the caller labels as such.
        return None
    if not isinstance(pods, list):
        return None

    requested: dict[str, tuple[float, float]] = {}
    for pod in pods:
        node_name = getattr(getattr(pod, "spec", None), "node_name", None)
        if not node_name:
            continue
        cpu, memory = _pod_requests(pod)
        prev_cpu, prev_memory = requested.get(node_name, (0.0, 0.0))
        requested[node_name] = (prev_cpu + cpu, prev_memory + memory)
    return requested


def _node_free(
    node, requested: dict[str, tuple[float, float]] | None
) -> tuple[float, float]:
    """Return the node's allocatable capacity minus what running pods requested."""
    allocatable = (node.status.allocatable or {}) if node.status else {}
    cpu = parse_cpu(allocatable.get("cpu", "0"))
    memory = parse_memory_gib(allocatable.get("memory", "0"))
    if requested is None:
        return cpu, memory
    used_cpu, used_memory = requested.get(
        getattr(getattr(node, "metadata", None), "name", "") or "", (0.0, 0.0)
    )
    return max(cpu - used_cpu, 0.0), max(memory - used_memory, 0.0)


def _aggregate_ready_nodes(
    nodes: list,
    requested: dict[str, tuple[float, float]] | None = None,
    *,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, object]] | None = None,
) -> tuple[int, float, float]:
    """Return (ready_count, free_cpu_cores, free_memory_gib) across usable nodes."""
    ready_nodes = 0
    total_cpu = 0.0
    total_memory = 0.0
    for node in nodes:
        allocatable = (node.status.allocatable or {}) if node.status else {}
        if (
            _node_matches_placement(
                node,
                node_selector=node_selector or {},
                tolerations=tolerations or [],
            )
            and allocatable
        ):
            ready_nodes += 1
            cpu, memory = _node_free(node, requested)
            total_cpu += cpu
            total_memory += memory
    return ready_nodes, total_cpu, total_memory


def _any_node_fits(
    nodes: list,
    *,
    max_pod_cpu: float,
    max_pod_mem: float,
    requested: dict[str, tuple[float, float]] | None = None,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, object]] | None = None,
) -> bool:
    """Return True if at least one usable node can fit a pod of the given size."""
    for node in nodes:
        if not _node_matches_placement(
            node,
            node_selector=node_selector or {},
            tolerations=tolerations or [],
        ):
            continue
        node_cpu, node_mem = _node_free(node, requested)
        if node_cpu >= max_pod_cpu and node_mem >= max_pod_mem:
            return True
    return False


async def check_node_resources(
    api: ApiClient,
    *,
    workers: int,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, object]] | None = None,
) -> CheckResult:
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

        # Allocatable is total capacity, not remaining: without subtracting
        # what running pods already requested, a 95%-booked cluster passes.
        requested = await _requested_by_node(api)

        ready_nodes, total_cpu, total_memory = _aggregate_ready_nodes(
            nodes,
            requested,
            node_selector=node_selector,
            tolerations=tolerations,
        )

        ctrl_cpu, ctrl_mem = _controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)

        required_cpu = ctrl_cpu + (worker_cpu * workers)
        required_mem = ctrl_mem + (worker_mem * workers)

        capacity_label = "free" if requested is not None else "allocatable"
        details = [
            f"Cluster: {ready_nodes} schedulable nodes, "
            f"{format_cpu(total_cpu)} {capacity_label} CPU, "
            f"{format_memory(total_memory)} {capacity_label} memory",
            f"Deployment estimate: {format_cpu(required_cpu)} CPU, "
            f"{format_memory(required_mem)} memory ({workers} workers)",
        ]
        if requested is None:
            details.append(
                "Could not list running pods: capacity shown is total allocatable, "
                "not what is actually free"
            )

        if required_cpu > total_cpu or required_mem > total_memory:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.FAIL,
                message="Cluster does not have enough resources",
                details=details,
                hints=["Consider reducing worker count or adding cluster capacity"],
            )

        max_pod_cpu = max(ctrl_cpu, worker_cpu)
        max_pod_mem = max(ctrl_mem, worker_mem)
        # Judged on raw allocatable: a node too small to ever hold one pod is
        # a structural misconfiguration, not a transient booking.
        if not _any_node_fits(
            nodes,
            max_pod_cpu=max_pod_cpu,
            max_pod_mem=max_pod_mem,
            node_selector=node_selector,
            tolerations=tolerations,
        ):
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

        # Fragmentation: every node is big enough in principle, but running
        # pods have booked all of them past the largest pod we need to place.
        if requested is not None and not _any_node_fits(
            nodes,
            max_pod_cpu=max_pod_cpu,
            max_pod_mem=max_pod_mem,
            requested=requested,
            node_selector=node_selector,
            tolerations=tolerations,
        ):
            details.append(
                f"Largest single-pod requirement: "
                f"{format_cpu(max_pod_cpu)} CPU, {format_memory(max_pod_mem)} memory"
            )
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message="No node currently has enough free capacity for one pod",
                details=details,
                hints=[
                    "Pods will stay Pending until running workloads release capacity",
                    "Reduce worker count, free capacity, or add nodes",
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
    transient: list[str]


def _is_transient_status(status: int | None) -> bool:
    """Classify an apiserver HTTP status as transient (retryable) or not."""
    return bool(status and (status >= 500 or status == 429))


async def _classify_secrets(
    api: ApiClient, *, namespace: str, secret_names: list[str]
) -> _SecretClassification:
    """Read each secret and classify into found / missing / permission_denied / transient."""
    found: list[str] = []
    missing: list[str] = []
    permission_denied: list[str] = []
    transient: list[str] = []

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
            elif _is_transient_status(e.status):
                # A 5xx/429 from the apiserver is not proof the Secret is
                # missing. Bucket it separately so it degrades preflight to a
                # WARN instead of a FAIL blaming a Secret that may well exist.
                transient.append(f"{secret_name} (error: HTTP {e.status})")
            else:
                missing.append(f"{secret_name} (error: HTTP {e.status})")
    return _SecretClassification(
        found=found,
        missing=missing,
        permission_denied=permission_denied,
        transient=transient,
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
            hints=[
                "Repeat --image-pull-secret or --secret to verify referenced secrets"
            ],
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
    if classified.transient:
        details.extend([f"  ? {s} (transient error)" for s in classified.transient])

    if classified.missing:
        return CheckResult(
            name="Secrets",
            status=CheckStatus.FAIL,
            message=f"{len(classified.missing)} secret(s) not found",
            details=details,
            hints=["Create missing secrets with 'kubectl create secret ...'"],
        )
    if classified.permission_denied or classified.transient:
        return CheckResult(
            name="Secrets",
            status=CheckStatus.WARN,
            message=(
                f"Cannot verify {len(classified.permission_denied)} secret(s) "
                f"(permission denied), {len(classified.transient)} secret(s) "
                "(transient apiserver error)"
                if classified.permission_denied and classified.transient
                else f"Cannot verify {len(classified.permission_denied) + len(classified.transient)} secret(s)"
            ),
            details=details,
            hints=["Re-run preflight; check apiserver health"]
            if classified.transient
            else [],
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
    registry, _repo, tag, digest = _shared_parse_image_ref(image)
    if digest:
        details.append(f"Registry: {registry}, Digest: {digest}")
    elif tag:
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
            "Use --image-pull-secret <name> to specify registry credentials",
            f"Verify manually: kubectl run test --image={image} "
            "--rm -it --restart=Never -- echo ok",
        ],
    )

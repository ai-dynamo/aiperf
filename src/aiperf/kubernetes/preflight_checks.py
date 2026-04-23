# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Free-function implementations of individual preflight checks.

Split out of ``preflight.py`` to keep that module under the ergonomics file-size
limit. These functions are stateless — they take an ``ApiClient`` and any
additional config they need and return a ``CheckResult``.

Capacity-focused checks (quotas, nodes, secrets, image) live in
``preflight_capacity_checks`` to keep each file under the file-size limit.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION
from aiperf.kubernetes.jobset import get_jobset_install_hint
from aiperf.kubernetes.preflight_capacity_checks import (
    check_image,
    check_node_resources,
    check_resource_quotas,
    check_secrets,
)
from aiperf.kubernetes.preflight_utils import (
    check_rbac_access as _shared_check_rbac_access,
)

if TYPE_CHECKING:
    from aiperf.kubernetes.preflight import CheckResult

__all__ = [
    "check_image",
    "check_node_resources",
    "check_resource_quotas",
    "check_secrets",
]

# Required RBAC permissions for AIPerf deployment: (verb, resource, api_group)
REQUIRED_RBAC_PERMISSIONS: list[tuple[str, str, str]] = [
    ("create", "configmaps", ""),
    ("get", "pods", ""),
    ("get", "pods/log", ""),
    ("create", "roles", "rbac.authorization.k8s.io"),
    ("create", "rolebindings", "rbac.authorization.k8s.io"),
    ("create", "jobsets", JOBSET_GROUP),
    ("get", "jobsets", JOBSET_GROUP),
    ("delete", "jobsets", JOBSET_GROUP),
]

_CLUSTER_API_ERRORS: tuple[type[BaseException], ...] = (
    ApiException,
    aiohttp.ClientError,
    asyncio.TimeoutError,
    OSError,
    RuntimeError,
)


async def _find_deployment(
    api: ApiClient, namespace: str, name_substring: str
) -> tuple[bool, bool]:
    """Check if a deployment matching name_substring exists and is ready.

    Returns:
        Tuple of (found, ready).
    """
    deployments = (
        await client.AppsV1Api(api).list_namespaced_deployment(namespace)
    ).items
    for deploy in deployments:
        name = deploy.metadata.name if deploy.metadata else ""
        if name_substring in (name or "").lower():
            ready_replicas = (deploy.status.ready_replicas or 0) if deploy.status else 0
            return True, ready_replicas > 0
    return False, False


async def check_cluster_connectivity(api: ApiClient) -> CheckResult:
    """Check if we can connect to the Kubernetes cluster."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    try:
        await client.VersionApi(api).get_code()
        return CheckResult(
            name="Cluster Connectivity",
            status=CheckStatus.PASS,
            message="Connected to Kubernetes cluster",
        )
    except _CLUSTER_API_ERRORS as e:
        return CheckResult(
            name="Cluster Connectivity",
            status=CheckStatus.FAIL,
            message=f"Failed to connect: {e}",
            hints=[
                "Check your kubeconfig (~/.kube/config) or KUBECONFIG env var",
                "Verify the cluster is running and accessible",
            ],
        )


async def check_kubernetes_version(api: ApiClient) -> CheckResult:
    """Check Kubernetes version compatibility."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    try:
        version = await client.VersionApi(api).get_code()
        major_str = re.sub(r"[^0-9]", "", version.major or "0")
        minor_str = re.sub(r"[^0-9]", "", version.minor or "0")
        major = int(major_str) if major_str else 0
        minor = int(minor_str) if minor_str else 0
        git_version = version.git_version or "unknown"

        if major > 1 or (major == 1 and minor >= 24):
            return CheckResult(
                name="Kubernetes Version",
                status=CheckStatus.PASS,
                message=f"Kubernetes {git_version} (1.24+ required)",
            )
        return CheckResult(
            name="Kubernetes Version",
            status=CheckStatus.FAIL,
            message=f"Kubernetes {git_version} is below minimum 1.24",
            hints=["Upgrade your Kubernetes cluster to version 1.24 or later"],
        )
    except _CLUSTER_API_ERRORS as e:
        return CheckResult(
            name="Kubernetes Version",
            status=CheckStatus.WARN,
            message=f"Could not determine version: {e}",
        )


async def check_namespace(api: ApiClient, *, namespace: str) -> CheckResult:
    """Check if namespace exists or can be created."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    try:
        await client.CoreV1Api(api).read_namespace(namespace)
        return CheckResult(
            name="Namespace",
            status=CheckStatus.PASS,
            message=f"Namespace '{namespace}' exists",
        )
    except ApiException as e:
        if e.status == 404:
            return await _namespace_missing_result(api, namespace)
        return CheckResult(
            name="Namespace",
            status=CheckStatus.FAIL,
            message=f"Error checking namespace: HTTP {e.status}",
        )


async def _namespace_missing_result(api: ApiClient, namespace: str) -> CheckResult:
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    try:
        allowed = await _shared_check_rbac_access(
            api, verb="create", resource="namespaces", group="", namespace=namespace
        )
        if allowed:
            return CheckResult(
                name="Namespace",
                status=CheckStatus.PASS,
                message=f"Namespace '{namespace}' will be created",
            )
        return CheckResult(
            name="Namespace",
            status=CheckStatus.FAIL,
            message=f"Namespace '{namespace}' does not exist",
            hints=[f"Ask an admin to create namespace '{namespace}'"],
        )
    except _CLUSTER_API_ERRORS as perm_err:
        return CheckResult(
            name="Namespace",
            status=CheckStatus.WARN,
            message=(
                f"Namespace '{namespace}' does not exist, "
                "cannot verify create permission"
            ),
            details=[str(perm_err)],
        )


async def check_rbac_permissions(api: ApiClient, *, namespace: str) -> CheckResult:
    """Check required RBAC permissions."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    missing: list[str] = []
    passed: list[str] = []

    for verb, resource, group in REQUIRED_RBAC_PERMISSIONS:
        display = f"{group}/{resource}" if group else resource
        try:
            allowed = await _shared_check_rbac_access(
                api, verb=verb, resource=resource, group=group, namespace=namespace
            )
            if allowed:
                passed.append(f"{verb} {display}")
            else:
                missing.append(f"{verb} {display}")
        except _CLUSTER_API_ERRORS as e:
            missing.append(f"{verb} {display} (check failed: {e})")

    if missing:
        return CheckResult(
            name="RBAC Permissions",
            status=CheckStatus.FAIL,
            message=f"Missing {len(missing)} required permission(s)",
            details=[f"  ✗ {p}" for p in missing],
            hints=[
                "Contact your cluster admin to grant the required permissions",
                f"Permissions needed in namespace '{namespace}'",
            ],
        )
    return CheckResult(
        name="RBAC Permissions",
        status=CheckStatus.PASS,
        message=f"All {len(passed)} required permissions granted",
        details=[f"  ✓ {p}" for p in passed],
    )


async def check_jobset_crd(api: ApiClient) -> CheckResult:
    """Check if JobSet CRD is installed."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    crd_name = f"{JOBSET_PLURAL}.{JOBSET_GROUP}"
    try:
        await client.ApiextensionsV1Api(api).read_custom_resource_definition(crd_name)
        return CheckResult(
            name="JobSet CRD",
            status=CheckStatus.PASS,
            message=f"JobSet CRD ({JOBSET_GROUP}/{JOBSET_VERSION}) installed",
        )
    except ApiException as e:
        if e.status == 404:
            return CheckResult(
                name="JobSet CRD",
                status=CheckStatus.FAIL,
                message="JobSet CRD not found",
                hints=[get_jobset_install_hint()],
            )
        return CheckResult(
            name="JobSet CRD",
            status=CheckStatus.WARN,
            message=f"Error checking JobSet CRD: HTTP {e.status}",
        )


async def check_jobset_controller(api: ApiClient) -> CheckResult:
    """Check if JobSet controller is running."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    try:
        found, ready = await _find_deployment(api, "jobset-system", "jobset")
        if ready:
            return CheckResult(
                name="JobSet Controller",
                status=CheckStatus.PASS,
                message="JobSet controller is running",
            )
        if found:
            return CheckResult(
                name="JobSet Controller",
                status=CheckStatus.WARN,
                message="JobSet controller found but not ready",
                hints=["Check 'kubectl get pods -n jobset-system' for issues"],
            )
        return CheckResult(
            name="JobSet Controller",
            status=CheckStatus.FAIL,
            message="JobSet controller not found",
            hints=[
                "Install JobSet controller or ensure it's in 'jobset-system' namespace"
            ],
        )
    except ApiException as e:
        if e.status == 403:
            return CheckResult(
                name="JobSet Controller",
                status=CheckStatus.SKIP,
                message="Cannot check jobset-system namespace (permission denied)",
            )
        return CheckResult(
            name="JobSet Controller",
            status=CheckStatus.WARN,
            message=f"Could not verify controller: HTTP {e.status}",
        )


async def check_network_policies(api: ApiClient, *, namespace: str) -> CheckResult:
    """Check for restrictive network policies."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    try:
        policies = (
            await client.NetworkingV1Api(api).list_namespaced_network_policy(namespace)
        ).items

        if not policies:
            return CheckResult(
                name="Network Policies",
                status=CheckStatus.PASS,
                message="No network policies found (unrestricted)",
            )

        policy_names = [(p.metadata.name if p.metadata else "") for p in policies]
        return CheckResult(
            name="Network Policies",
            status=CheckStatus.WARN,
            message=f"Found {len(policies)} network policy(ies)",
            details=[f"  Policies: {', '.join(policy_names)}"],
            hints=[
                "Ensure policies allow pod-to-pod communication within the namespace",
                "AIPerf pods need to communicate via TCP on multiple ports",
            ],
        )
    except ApiException as e:
        if e.status == 403:
            return CheckResult(
                name="Network Policies",
                status=CheckStatus.SKIP,
                message="Cannot check network policies (permission denied)",
            )
        return CheckResult(
            name="Network Policies",
            status=CheckStatus.WARN,
            message=f"Error checking network policies: HTTP {e.status}",
        )


async def check_dns(api: ApiClient) -> CheckResult:
    """Check DNS resolution capability."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    try:
        found, ready = await _find_deployment(api, "kube-system", "coredns")
        if ready:
            return CheckResult(
                name="DNS Resolution",
                status=CheckStatus.PASS,
                message="CoreDNS is running",
                details=[
                    "Workers will resolve controller DNS name for ZMQ connections"
                ],
            )
        if found:
            return CheckResult(
                name="DNS Resolution",
                status=CheckStatus.WARN,
                message="CoreDNS found but may not be ready",
                hints=["Check 'kubectl get pods -n kube-system -l k8s-app=kube-dns'"],
            )
        return CheckResult(
            name="DNS Resolution",
            status=CheckStatus.WARN,
            message="CoreDNS not found in kube-system",
            hints=["Verify your cluster has a working DNS service"],
        )
    except _CLUSTER_API_ERRORS as e:
        return CheckResult(
            name="DNS Resolution",
            status=CheckStatus.WARN,
            message=f"Could not verify DNS: {e}",
        )


async def check_endpoint_connectivity(
    api: ApiClient, *, endpoint_url: str | None
) -> CheckResult:
    """Check if the LLM endpoint is potentially reachable."""
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    if not endpoint_url:
        return CheckResult(
            name="Endpoint Connectivity",
            status=CheckStatus.SKIP,
            message="No endpoint URL specified",
            hints=["Use --endpoint to verify LLM endpoint connectivity"],
        )

    try:
        parsed = urlparse(endpoint_url)
        host = parsed.hostname or "unknown"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        details = [
            f"Endpoint: {endpoint_url}",
            f"Host: {host}, Port: {port}",
        ]

        if ".svc" in host or ".svc.cluster.local" in host:
            return await _check_cluster_service_endpoint(api, host, details)

        return CheckResult(
            name="Endpoint Connectivity",
            status=CheckStatus.INFO,
            message="External endpoint specified (cannot verify from CLI)",
            details=details,
            hints=[
                "Endpoint connectivity will be verified during deployment",
                "Ensure cluster egress allows connections to this endpoint",
            ],
        )
    except (ValueError, TypeError, AttributeError) as e:
        return CheckResult(
            name="Endpoint Connectivity",
            status=CheckStatus.WARN,
            message=f"Could not parse endpoint URL: {e}",
        )


async def _check_cluster_service_endpoint(
    api: ApiClient, host: str, details: list[str]
) -> CheckResult:
    from aiperf.kubernetes.preflight import CheckResult, CheckStatus

    before_svc = host.split(".svc")[0]
    if "." in before_svc:
        svc_name, svc_ns = before_svc.rsplit(".", 1)
    else:
        svc_name, svc_ns = before_svc, "default"

    try:
        await client.CoreV1Api(api).read_namespaced_service(svc_name, svc_ns)
        return CheckResult(
            name="Endpoint Connectivity",
            status=CheckStatus.PASS,
            message=f"Cluster service '{svc_name}' found in namespace '{svc_ns}'",
            details=details,
        )
    except (
        ApiException,
        aiohttp.ClientError,
        asyncio.TimeoutError,
        OSError,
    ):
        return CheckResult(
            name="Endpoint Connectivity",
            status=CheckStatus.FAIL,
            message=f"Cluster service not found: {host}",
            details=details,
            hints=[f"Verify the service exists: kubectl get svc -A | grep {svc_name}"],
        )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator pre-flight checks for AIPerfJob deployments.

Validates cluster readiness before creating any resources. Checks are organized
into tiers:
- Tier 1 (blocking): Kubernetes version, JobSet CRD
- Tier 2 (blocking): RBAC permissions
- Tier 3+ (concurrent): Everything else

On failure, the operator sets the CR to Failed with actionable error messages
and does not create any resources.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import (
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
    KUEUE_GROUP,
    KUEUE_LOCALQUEUE_PLURAL,
    KUEUE_VERSION,
)
from aiperf.kubernetes.jobset import get_jobset_install_hint
from aiperf.kubernetes.preflight import CheckResult, CheckStatus, PreflightResults
from aiperf.kubernetes.preflight_utils import check_rbac_access, parse_image_ref
from aiperf.kubernetes.resources import CONFIGMAP_MAX_SIZE_BYTES
from aiperf.kubernetes.utils import (
    format_cpu,
    format_memory,
    parse_cpu,
    parse_memory_gib,
)
from aiperf.operator.environment import OperatorEnvironment

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiperf.config import AIPerfConfig
    from aiperf.config.deployment import DeploymentConfig
    from aiperf.kubernetes.resources import KubernetesDeployment

logger = logging.getLogger(__name__)

# Minimum supported Kubernetes version
_MIN_K8S_MAJOR = 1
_MIN_K8S_MINOR = 24

# Required RBAC permissions for the operator to manage resources.
# (verb, resource, api_group)
_OPERATOR_RBAC_PERMISSIONS: list[tuple[str, str, str]] = [
    # Core resources
    ("create", "configmaps", ""),
    ("get", "configmaps", ""),
    ("delete", "configmaps", ""),
    ("create", "roles", "rbac.authorization.k8s.io"),
    ("create", "rolebindings", "rbac.authorization.k8s.io"),
    ("get", "pods", ""),
    ("list", "pods", ""),
    ("get", "pods/log", ""),
    ("create", "events", ""),
    ("patch", "events", ""),
    # JobSet resources
    ("create", "jobsets", JOBSET_GROUP),
    ("get", "jobsets", JOBSET_GROUP),
    ("delete", "jobsets", JOBSET_GROUP),
    ("watch", "jobsets", JOBSET_GROUP),
    ("get", "jobsets/status", JOBSET_GROUP),
]


def _controller_resource_requirements() -> tuple[float, float]:
    """Return total controller-pod CPU cores and memory GiB."""
    from aiperf.kubernetes.environment import (
        CONTROLLER_RESOURCE_KEYS,
        K8sEnvironment,
    )

    cpu = 0.0
    memory = 0.0
    for key in CONTROLLER_RESOURCE_KEYS:
        settings = getattr(K8sEnvironment, key)
        cpu += parse_cpu(settings.CPU)
        memory += parse_memory_gib(settings.MEMORY)
    return cpu, memory


# Known public registries that don't need pull secrets
_PUBLIC_REGISTRIES = frozenset(
    {
        "docker.io",
        "registry-1.docker.io",
        "ghcr.io",
        "quay.io",
        "nvcr.io",
        "registry.k8s.io",
    }
)


@dataclass(slots=True)
class OperatorPreflightChecker:
    """Validates cluster readiness before deploying an AIPerfJob.

    Runs 19 checks across 3 tiers. Blocking checks (FAIL) prevent resource
    creation. Warning checks (WARN) are logged but do not block.

    Sibling: ``aiperf.kubernetes.preflight.CLIPreflightChecker`` handles
    pre-deploy CLI-side preflight.
    """

    api: ApiClient
    """Kubernetes API client for cluster queries."""

    namespace: str
    """Target namespace for the AIPerfJob deployment."""

    deployment: KubernetesDeployment
    """Fully resolved Kubernetes deployment specification."""

    deploy_config: DeploymentConfig
    """Deployment configuration from the CRD spec."""

    config: AIPerfConfig
    """Benchmark configuration from the CRD spec."""

    total_workers: int
    """Total number of worker processes across all pods."""

    num_pods: int
    """Number of worker pods to deploy."""

    def _resource_mode_skip(self, check_name: str) -> CheckResult | None:
        """Skip resource-based checks when pod resources are intentionally omitted."""
        if self.deploy_config.resource_mode != "none":
            return None
        return CheckResult(
            name=check_name,
            status=CheckStatus.SKIP,
            message=(
                "Skipped because spec.resourceMode=none omits controller/worker "
                "CPU and memory requests/limits."
            ),
        )

    async def run_all(
        self, timeout: float = OperatorEnvironment.PREFLIGHT_TIMEOUT
    ) -> PreflightResults:
        """Run all pre-flight checks with tiered short-circuiting.

        Args:
            timeout: Maximum seconds for all checks combined.

        Returns:
            PreflightResults with all check outcomes.
        """
        results = PreflightResults()
        try:
            async with asyncio.timeout(timeout):
                # Tier 1: Cluster compatibility (sequential, short-circuit)
                for check in [
                    self._check_kubernetes_version,
                    self._check_jobset_crd,
                ]:
                    result = await self._run_check(check)
                    results.add(result)
                    if result.status == CheckStatus.FAIL:
                        return results

                # Tier 2: RBAC (short-circuit)
                result = await self._run_check(self._check_rbac_permissions)
                results.add(result)
                if result.status == CheckStatus.FAIL:
                    return results

                # Tier 3+: Concurrent checks
                remaining = [
                    self._check_jobset_controller,
                    self._check_service_account,
                    self._check_node_resources,
                    self._check_node_selector_match,
                    self._check_per_node_schedulability,
                    self._check_resource_quotas,
                    self._check_memory_estimation,
                    self._check_secrets,
                    self._check_image_reference,
                    self._check_dns,
                    self._check_network_policies,
                    self._check_kueue_queue,
                    self._check_configmap_size,
                    self._check_dry_run,
                    self._check_pod_security_admission,
                    self._check_tolerations,
                ]
                concurrent = await asyncio.gather(
                    *(self._run_check(c) for c in remaining),
                    return_exceptions=True,
                )
                for r in concurrent:
                    if isinstance(r, BaseException):
                        results.add(
                            CheckResult(
                                name="Unknown",
                                status=CheckStatus.FAIL,
                                message=f"Check raised exception: {r}",
                            )
                        )
                    else:
                        results.add(r)

        except TimeoutError:
            results.add(
                CheckResult(
                    name="Preflight Timeout",
                    status=CheckStatus.FAIL,
                    message=f"Pre-flight checks timed out after {timeout:.0f}s",
                    hints=[
                        "Increase AIPERF_PREFLIGHT_TIMEOUT or check cluster responsiveness"
                    ],
                )
            )

        return results

    async def _run_check(
        self,
        check_fn: Callable[[], Awaitable[CheckResult]],
    ) -> CheckResult:
        """Run a single check with timing and error handling."""
        start = time.perf_counter()
        try:
            result = await check_fn()
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            # Transient errors (empty message, connection errors) should warn,
            # not fail permanently - the operator can retry on the next reconcile.
            error_str = str(e).strip()
            is_transient = not error_str or "connect" in error_str.lower()
            result = CheckResult(
                name=check_fn.__name__.removeprefix("_check_")
                .replace("_", " ")
                .title(),
                status=CheckStatus.WARN if is_transient else CheckStatus.FAIL,
                message=f"Check failed with error: {e}"
                if error_str
                else "Transient API error (will retry)",
            )
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    # =========================================================================
    # Tier 1: Cluster Compatibility
    # =========================================================================

    async def _check_kubernetes_version(self) -> CheckResult:
        """Verify Kubernetes version >= 1.24."""
        vinfo = await client.VersionApi(self.api).get_code()
        major_str = re.sub(r"[^0-9]", "", vinfo.major or "0")
        minor_str = re.sub(r"[^0-9]", "", vinfo.minor or "0")
        major = int(major_str) if major_str else 0
        minor = int(minor_str) if minor_str else 0
        git_version = vinfo.git_version or "unknown"

        if major > _MIN_K8S_MAJOR or (
            major == _MIN_K8S_MAJOR and minor >= _MIN_K8S_MINOR
        ):
            return CheckResult(
                name="Kubernetes Version",
                status=CheckStatus.PASS,
                message=f"Kubernetes {git_version} (>= {_MIN_K8S_MAJOR}.{_MIN_K8S_MINOR} required)",
            )
        return CheckResult(
            name="Kubernetes Version",
            status=CheckStatus.FAIL,
            message=(
                f"Kubernetes {git_version} is below minimum "
                f"{_MIN_K8S_MAJOR}.{_MIN_K8S_MINOR}. "
                f"Upgrade your cluster to {_MIN_K8S_MAJOR}.{_MIN_K8S_MINOR}+."
            ),
        )

    async def _check_jobset_crd(self) -> CheckResult:
        """Verify JobSet CRD is installed."""
        try:
            await client.CustomObjectsApi(self.api).list_cluster_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                limit=1,
            )
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
                    message=f"JobSet CRD not found. Install with: {get_jobset_install_hint()}",
                )
            return CheckResult(
                name="JobSet CRD",
                status=CheckStatus.FAIL,
                message=f"Error checking JobSet CRD: HTTP {e.status or 'unknown'}",
            )

    # =========================================================================
    # Tier 2: RBAC
    # =========================================================================

    async def _check_rbac_permissions(self) -> CheckResult:
        """Verify the operator has all required RBAC permissions."""
        missing = []
        for verb, resource, group in _OPERATOR_RBAC_PERMISSIONS:
            try:
                allowed = await check_rbac_access(
                    self.api,
                    verb=verb,
                    resource=resource,
                    group=group,
                    namespace=self.namespace,
                )
                if not allowed:
                    display = f"{group}/{resource}" if group else resource
                    missing.append(f"{verb} {display}")
            except (
                ApiException,
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
            ) as e:
                display = f"{group}/{resource}" if group else resource
                missing.append(f"{verb} {display} (check failed: {e})")
            except Exception as e:  # noqa: BLE001 - defensive: any per-permission probe error falls through to 'assume missing' rather than abort the tier
                display = f"{group}/{resource}" if group else resource
                missing.append(f"{verb} {display} (check failed: {e})")

        if missing:
            return CheckResult(
                name="RBAC Permissions",
                status=CheckStatus.FAIL,
                message=(
                    f"Missing {len(missing)} RBAC permission(s): "
                    f"{', '.join(missing)}. "
                    f"Grant permissions in namespace '{self.namespace}'."
                ),
            )
        return CheckResult(
            name="RBAC Permissions",
            status=CheckStatus.PASS,
            message=f"All {len(_OPERATOR_RBAC_PERMISSIONS)} required permissions granted",
        )

    # =========================================================================
    # Tier 3+: Concurrent Checks
    # =========================================================================

    async def _check_jobset_controller(self) -> CheckResult:
        """Check if JobSet controller is running in jobset-system."""
        try:
            deploy_list = await client.AppsV1Api(self.api).list_namespaced_deployment(
                namespace="jobset-system",
            )
            for deploy in deploy_list.items:
                name = (deploy.metadata.name if deploy.metadata else "") or ""
                if "jobset" in name.lower():
                    ready = (
                        deploy.status.ready_replicas
                        if deploy.status and deploy.status.ready_replicas
                        else 0
                    )
                    if ready and ready > 0:
                        return CheckResult(
                            name="JobSet Controller",
                            status=CheckStatus.PASS,
                            message="JobSet controller is running",
                        )
                    return CheckResult(
                        name="JobSet Controller",
                        status=CheckStatus.WARN,
                        message="JobSet controller found but not ready",
                        hints=["Check: kubectl get pods -n jobset-system"],
                    )
            return CheckResult(
                name="JobSet Controller",
                status=CheckStatus.WARN,
                message="JobSet controller not found in jobset-system namespace",
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
                message=f"Could not verify JobSet controller: {e}",
            )

    async def _check_service_account(self) -> CheckResult:
        """Verify custom service account exists if specified."""
        sa_name = self.deploy_config.pod_template.service_account_name
        if not sa_name:
            return CheckResult(
                name="Service Account",
                status=CheckStatus.SKIP,
                message="No custom service account specified",
            )
        try:
            await client.CoreV1Api(self.api).read_namespaced_service_account(
                name=sa_name, namespace=self.namespace
            )
            return CheckResult(
                name="Service Account",
                status=CheckStatus.PASS,
                message=f"Service account '{sa_name}' exists",
            )
        except ApiException as e:
            if e.status == 404:
                return CheckResult(
                    name="Service Account",
                    status=CheckStatus.FAIL,
                    message=(
                        f"Service account '{sa_name}' not found in namespace "
                        f"'{self.namespace}'. Pod creation will fail."
                    ),
                    hints=[
                        f"kubectl create serviceaccount {sa_name} -n {self.namespace}"
                    ],
                )
            return CheckResult(
                name="Service Account",
                status=CheckStatus.WARN,
                message=f"Could not verify service account: {e}",
            )

    async def _check_node_resources(self) -> CheckResult:
        """Check aggregate allocatable CPU/mem across Ready nodes."""
        from aiperf.kubernetes.environment import K8sEnvironment

        if skip := self._resource_mode_skip("Node Resources"):
            return skip

        try:
            node_list = await client.CoreV1Api(self.api).list_node()
            nodes = node_list.items
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message=f"Could not check node resources: {e}",
            )

        if not nodes:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message="No nodes found in cluster",
            )

        total_cpu = 0.0
        total_memory = 0.0
        ready_nodes = 0

        for node in nodes:
            if not _is_node_ready_typed(node):
                continue
            ready_nodes += 1
            alloc = (node.status.allocatable or {}) if node.status else {}
            total_cpu += parse_cpu(alloc.get("cpu", "0"))
            total_memory += parse_memory_gib(alloc.get("memory", "0"))

        ctrl_cpu, ctrl_mem = _controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)
        required_cpu = ctrl_cpu + (worker_cpu * self.num_pods)
        required_mem = ctrl_mem + (worker_mem * self.num_pods)

        if required_cpu > total_cpu or required_mem > total_memory:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message=(
                    f"Cluster may not have enough resources. "
                    f"Need {format_cpu(required_cpu)} CPU, {format_memory(required_mem)} mem "
                    f"but only {format_cpu(total_cpu)} CPU, {format_memory(total_memory)} mem "
                    f"available across {ready_nodes} ready node(s)."
                ),
                hints=["Reduce worker count or add cluster capacity"],
            )

        return CheckResult(
            name="Node Resources",
            status=CheckStatus.PASS,
            message=(
                f"Cluster has sufficient resources "
                f"({ready_nodes} ready nodes, {format_cpu(total_cpu)} CPU, "
                f"{format_memory(total_memory)} mem)"
            ),
        )

    async def _check_node_selector_match(self) -> CheckResult:
        """Verify matching Ready nodes exist for nodeSelector."""
        node_selector = self.deploy_config.pod_template.node_selector
        if not node_selector:
            return CheckResult(
                name="Node Selector Match",
                status=CheckStatus.SKIP,
                message="No nodeSelector specified",
            )

        try:
            node_list = await client.CoreV1Api(self.api).list_node()
            nodes = node_list.items
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            return CheckResult(
                name="Node Selector Match",
                status=CheckStatus.WARN,
                message=f"Could not check node selectors: {e}",
            )

        matching = 0
        for node in nodes:
            if not _is_node_ready_typed(node):
                continue
            labels = (node.metadata.labels or {}) if node.metadata else {}
            if all(labels.get(k) == v for k, v in node_selector.items()):
                matching += 1

        if matching == 0:
            selector_str = ", ".join(f"{k}={v}" for k, v in node_selector.items())
            return CheckResult(
                name="Node Selector Match",
                status=CheckStatus.FAIL,
                message=(
                    f"No node matches nodeSelector {{{selector_str}}}. "
                    f"Label nodes with: kubectl label node <name> {selector_str}"
                ),
            )

        return CheckResult(
            name="Node Selector Match",
            status=CheckStatus.PASS,
            message=f"{matching} node(s) match nodeSelector",
        )

    async def _check_per_node_schedulability(self) -> CheckResult:
        """Check that at least one matching Ready node can fit the largest pod."""
        from aiperf.kubernetes.environment import K8sEnvironment

        if skip := self._resource_mode_skip("Per-Node Schedulability"):
            return skip

        try:
            node_list = await client.CoreV1Api(self.api).list_node()
            nodes = node_list.items
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            return CheckResult(
                name="Per-Node Schedulability",
                status=CheckStatus.WARN,
                message=f"Could not check per-node schedulability: {e}",
            )

        ctrl_cpu, ctrl_mem = _controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)
        max_pod_cpu = max(ctrl_cpu, worker_cpu)
        max_pod_mem = max(ctrl_mem, worker_mem)

        node_selector = self.deploy_config.pod_template.node_selector

        for node in nodes:
            if not _is_node_ready_typed(node):
                continue
            if node_selector:
                labels = (node.metadata.labels or {}) if node.metadata else {}
                if not all(labels.get(k) == v for k, v in node_selector.items()):
                    continue
            alloc = (node.status.allocatable or {}) if node.status else {}
            node_cpu = parse_cpu(alloc.get("cpu", "0"))
            node_mem = parse_memory_gib(alloc.get("memory", "0"))
            if node_cpu >= max_pod_cpu and node_mem >= max_pod_mem:
                return CheckResult(
                    name="Per-Node Schedulability",
                    status=CheckStatus.PASS,
                    message="At least one node can fit the largest pod",
                )

        return CheckResult(
            name="Per-Node Schedulability",
            status=CheckStatus.FAIL,
            message=(
                f"No single node can fit the largest pod "
                f"({format_cpu(max_pod_cpu)} CPU, {format_memory(max_pod_mem)} mem). "
                f"Add larger nodes or reduce pod resource requirements."
            ),
        )

    async def _check_resource_quotas(self) -> CheckResult:
        """Check if deployment would exceed namespace resource quotas."""
        from aiperf.kubernetes.environment import K8sEnvironment

        if skip := self._resource_mode_skip("Resource Quotas"):
            return skip

        try:
            quota_list = await client.CoreV1Api(
                self.api
            ).list_namespaced_resource_quota(
                namespace=self.namespace,
            )
            quotas = quota_list.items
        except ApiException:
            return CheckResult(
                name="Resource Quotas",
                status=CheckStatus.WARN,
                message="Could not check resource quotas",
            )

        if not quotas:
            return CheckResult(
                name="Resource Quotas",
                status=CheckStatus.PASS,
                message="No resource quotas configured",
            )

        ctrl_cpu, ctrl_mem = _controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)
        required_cpu = ctrl_cpu + (worker_cpu * self.num_pods)
        required_mem = ctrl_mem + (worker_mem * self.num_pods)

        for quota in quotas:
            hard = (quota.status.hard or {}) if quota.status else {}
            used = (quota.status.used or {}) if quota.status else {}

            hard_cpu = hard.get("cpu") or hard.get("requests.cpu")
            hard_mem = hard.get("memory") or hard.get("requests.memory")
            used_cpu = used.get("cpu") or used.get("requests.cpu")
            used_mem = used.get("memory") or used.get("requests.memory")

            if hard_cpu:
                total_needed = required_cpu + parse_cpu(used_cpu or "0")
                if total_needed > parse_cpu(hard_cpu):
                    return CheckResult(
                        name="Resource Quotas",
                        status=CheckStatus.WARN,
                        message=(
                            f"Benchmark may exceed CPU quota: "
                            f"{format_cpu(total_needed)} needed vs {hard_cpu} limit. "
                            f"Request a quota increase or reduce worker count."
                        ),
                    )
            if hard_mem:
                total_needed = required_mem + parse_memory_gib(used_mem or "0")
                if total_needed > parse_memory_gib(hard_mem):
                    return CheckResult(
                        name="Resource Quotas",
                        status=CheckStatus.WARN,
                        message=(
                            f"Benchmark may exceed memory quota: "
                            f"{format_memory(total_needed)} needed vs {hard_mem} limit. "
                            f"Request a quota increase or reduce worker count."
                        ),
                    )

        return CheckResult(
            name="Resource Quotas",
            status=CheckStatus.PASS,
            message=f"Within resource quota limits ({len(quotas)} quota(s) checked)",
        )

    async def _check_memory_estimation(self) -> CheckResult:
        """Use memory estimator to detect OOM risk."""
        if skip := self._resource_mode_skip("Memory Estimation"):
            return skip
        try:
            from aiperf.kubernetes.memory_estimator import estimate_memory

            estimate = estimate_memory(
                config=self.config,
                total_workers=self.total_workers,
                connections_per_worker=self.deploy_config.connections_per_worker,
            )

            if estimate.warnings:
                return CheckResult(
                    name="Memory Estimation",
                    status=CheckStatus.WARN,
                    message=f"OOM risk detected: {'; '.join(estimate.warnings)}",
                    hints=estimate.recommendations,
                )

            return CheckResult(
                name="Memory Estimation",
                status=CheckStatus.PASS,
                message="Memory estimates within limits",
            )
        except (ValueError, TypeError, OSError) as e:
            return CheckResult(
                name="Memory Estimation",
                status=CheckStatus.WARN,
                message=f"Could not estimate memory: {e}",
            )

    async def _check_secrets(self) -> CheckResult:
        """Verify all referenced secrets exist."""
        pod_template = self.deploy_config.pod_template
        needed: set[str] = set()

        # Image pull secrets
        for name in pod_template.image_pull_secrets:
            needed.add(name)

        # Secrets from volumes
        for vol in pod_template.volumes:
            secret = vol.get("secret", {})
            if secret_name := secret.get("secretName"):
                needed.add(secret_name)

        # Secrets from env secretKeyRef
        for env_var in pod_template.env:
            value_from = env_var.get("valueFrom", {})
            secret_ref = value_from.get("secretKeyRef", {})
            if secret_name := secret_ref.get("name"):
                needed.add(secret_name)

        if not needed:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.SKIP,
                message="No secrets referenced",
            )

        missing = []
        permission_denied = []
        core = client.CoreV1Api(self.api)
        for secret_name in sorted(needed):
            try:
                await core.read_namespaced_secret(
                    name=secret_name, namespace=self.namespace
                )
            except ApiException as e:
                if e.status == 404:
                    missing.append(secret_name)
                elif e.status == 403:
                    permission_denied.append(secret_name)
                else:
                    missing.append(secret_name)

        if missing:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.FAIL,
                message=(
                    f"Secret(s) not found: {', '.join(missing)}. "
                    f"Create with: kubectl create secret -n {self.namespace}"
                ),
            )
        if permission_denied:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.WARN,
                message=f"Cannot verify secret(s): {', '.join(permission_denied)} (permission denied)",
            )
        return CheckResult(
            name="Secrets",
            status=CheckStatus.PASS,
            message=f"All {len(needed)} secret(s) verified",
        )

    async def _check_image_reference(self) -> CheckResult:
        """Validate image format and warn on implicit latest or missing pull secrets."""
        image = self.deploy_config.image
        if not image:
            return CheckResult(
                name="Image Reference",
                status=CheckStatus.FAIL,
                message="No container image specified",
            )

        registry, _repo, tag = parse_image_ref(image)

        warnings = []
        if not tag:
            warnings.append(
                "Image uses implicit 'latest' tag which may cause inconsistent deployments"
            )

        has_pull_secrets = bool(self.deploy_config.pod_template.image_pull_secrets)
        if registry not in _PUBLIC_REGISTRIES and not has_pull_secrets:
            warnings.append(
                f"Registry '{registry}' may require authentication "
                f"but no imagePullSecrets configured"
            )

        if warnings:
            return CheckResult(
                name="Image Reference",
                status=CheckStatus.WARN,
                message=f"Image '{image}': {'; '.join(warnings)}",
            )
        return CheckResult(
            name="Image Reference",
            status=CheckStatus.PASS,
            message=f"Image '{image}' reference is valid",
        )

    async def _check_dns(self) -> CheckResult:
        """Verify CoreDNS is running in kube-system."""
        try:
            deploy_list = await client.AppsV1Api(self.api).list_namespaced_deployment(
                namespace="kube-system",
            )
            for deploy in deploy_list.items:
                name = (deploy.metadata.name if deploy.metadata else "") or ""
                if "coredns" in name.lower():
                    ready = (
                        deploy.status.ready_replicas
                        if deploy.status and deploy.status.ready_replicas
                        else 0
                    )
                    if ready and ready > 0:
                        return CheckResult(
                            name="DNS Resolution",
                            status=CheckStatus.PASS,
                            message="CoreDNS is running",
                        )
                    return CheckResult(
                        name="DNS Resolution",
                        status=CheckStatus.WARN,
                        message="CoreDNS found but not ready",
                        hints=[
                            "Check: kubectl get pods -n kube-system -l k8s-app=kube-dns"
                        ],
                    )
            return CheckResult(
                name="DNS Resolution",
                status=CheckStatus.WARN,
                message="CoreDNS not found in kube-system",
            )
        except ApiException as e:
            if e.status == 403:
                return CheckResult(
                    name="DNS Resolution",
                    status=CheckStatus.SKIP,
                    message="Cannot check kube-system namespace (permission denied)",
                )
            return CheckResult(
                name="DNS Resolution",
                status=CheckStatus.WARN,
                message=f"Could not verify DNS: {e}",
            )

    async def _check_network_policies(self) -> CheckResult:
        """Warn if restrictive network policies exist in namespace."""
        try:
            policy_list = await client.NetworkingV1Api(
                self.api
            ).list_namespaced_network_policy(namespace=self.namespace)
            policies = policy_list.items
            if not policies:
                return CheckResult(
                    name="Network Policies",
                    status=CheckStatus.PASS,
                    message="No network policies found (unrestricted)",
                )
            policy_names = [
                (p.metadata.name if p.metadata else "") or "" for p in policies
            ]
            return CheckResult(
                name="Network Policies",
                status=CheckStatus.WARN,
                message=(
                    f"Found {len(policies)} network policy(ies): {', '.join(policy_names)}. "
                    f"Ensure pod-to-pod communication is allowed."
                ),
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
                message=f"Could not check network policies: {e}",
            )

    async def _check_kueue_queue(self) -> CheckResult:
        """Verify Kueue queue configuration.

        If scheduling.queueName is set, verify the LocalQueue exists.
        If not set, check whether Kueue is installed and warn that the job
        will bypass gang-scheduling unless a namespace default queue is configured.
        """

        queue_name = self.deploy_config.scheduling.queue_name

        if queue_name:
            return await self._verify_kueue_local_queue(queue_name)

        # No explicit queue — check if Kueue is installed
        kueue_installed = await self._is_kueue_installed()
        if not kueue_installed:
            return CheckResult(
                name="Kueue Queue",
                status=CheckStatus.SKIP,
                message="Kueue not installed (queue check skipped)",
            )

        # Kueue is installed — check namespace default queue annotation
        has_default = await self._namespace_has_default_queue()
        if has_default:
            return CheckResult(
                name="Kueue Queue",
                status=CheckStatus.PASS,
                message=(
                    "No explicit queue specified but namespace has "
                    "kueue.x-k8s.io/default-queue-name annotation"
                ),
            )

        return CheckResult(
            name="Kueue Queue",
            status=CheckStatus.WARN,
            message=(
                "Kueue is installed but no queue configured. "
                "Job will bypass gang-scheduling and quota management."
            ),
            hints=[
                "Set scheduling.queueName in the CR spec, or",
                f"Annotate the namespace: kubectl annotate namespace {self.namespace} "
                "kueue.x-k8s.io/default-queue-name=<queue-name>",
            ],
        )

    async def _verify_kueue_local_queue(self, queue_name: str) -> CheckResult:
        """Verify a specific Kueue LocalQueue exists."""
        try:
            await client.CustomObjectsApi(self.api).get_namespaced_custom_object(
                group=KUEUE_GROUP,
                version=KUEUE_VERSION,
                plural=KUEUE_LOCALQUEUE_PLURAL,
                namespace=self.namespace,
                name=queue_name,
            )
            return CheckResult(
                name="Kueue Queue",
                status=CheckStatus.PASS,
                message=f"Kueue LocalQueue '{queue_name}' exists",
            )
        except ApiException as e:
            status_code = e.status or 0
            if status_code == 404:
                # Distinguish "CRD not installed" vs "queue not found".
                # If Kueue isn't installed at all, we get 404 on the queue
                # but also 404 on the plural list — treat a missing CRD as SKIP,
                # a missing queue as FAIL.
                kueue_installed = await self._is_kueue_installed()
                if not kueue_installed:
                    return CheckResult(
                        name="Kueue Queue",
                        status=CheckStatus.SKIP,
                        message="Kueue CRD not installed (queue check skipped)",
                    )
                return CheckResult(
                    name="Kueue Queue",
                    status=CheckStatus.FAIL,
                    message=(
                        f"Kueue LocalQueue '{queue_name}' not found. "
                        f"Create it or remove scheduling.queueName from spec."
                    ),
                )
            return CheckResult(
                name="Kueue Queue",
                status=CheckStatus.WARN,
                message=f"Could not verify Kueue queue: HTTP {status_code or 'unknown'}",
            )

    async def _is_kueue_installed(self) -> bool:
        """Check if the Kueue CRD is available on the cluster."""
        try:
            await client.CustomObjectsApi(self.api).list_namespaced_custom_object(
                group=KUEUE_GROUP,
                version=KUEUE_VERSION,
                plural=KUEUE_LOCALQUEUE_PLURAL,
                namespace=self.namespace,
                limit=1,
            )
            return True
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    async def _namespace_has_default_queue(self) -> bool:
        """Check if the namespace has a Kueue default queue annotation."""
        try:
            ns = await client.CoreV1Api(self.api).read_namespace(name=self.namespace)
            annotations = (ns.metadata.annotations or {}) if ns.metadata else {}
            return bool(annotations.get("kueue.x-k8s.io/default-queue-name"))
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    async def _check_configmap_size(self) -> CheckResult:
        """Verify generated ConfigMap data fits within 1 MiB limit."""
        try:
            cm_spec = self.deployment.get_configmap_spec()
            size_bytes = cm_spec.get_data_size_bytes()
            max_bytes = CONFIGMAP_MAX_SIZE_BYTES
            if size_bytes > max_bytes:
                size_mib = size_bytes / (1024 * 1024)
                return CheckResult(
                    name="ConfigMap Size",
                    status=CheckStatus.FAIL,
                    message=(
                        f"ConfigMap data size ({size_mib:.2f} MiB) exceeds "
                        f"1 MiB limit. Reduce config size."
                    ),
                )
            return CheckResult(
                name="ConfigMap Size",
                status=CheckStatus.PASS,
                message=f"ConfigMap size OK ({size_bytes:,} bytes)",
            )
        except (ValueError, TypeError, OSError) as e:
            return CheckResult(
                name="ConfigMap Size",
                status=CheckStatus.FAIL,
                message=f"Could not compute ConfigMap size: {e}",
            )

    async def _check_dry_run(self) -> CheckResult:
        """POST JobSet manifest with dryRun=All to catch API server rejections."""
        try:
            jobset_manifest = self.deployment.get_jobset_spec().to_k8s_manifest()
            await client.CustomObjectsApi(self.api).create_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                namespace=self.namespace,
                body=jobset_manifest,
                dry_run="All",
            )
            return CheckResult(
                name="Dry Run",
                status=CheckStatus.PASS,
                message="Server dry-run accepted the JobSet manifest",
            )
        except ApiException as e:
            msg = str(e)
            if e.body:
                try:
                    import orjson

                    body = orjson.loads(e.body)
                    msg = body.get("message", msg)
                except (ValueError, TypeError, orjson.JSONDecodeError):
                    # e.body was not well-formed JSON; fall back to str(e).
                    pass
            return CheckResult(
                name="Dry Run",
                status=CheckStatus.FAIL,
                message=(
                    f"Server dry-run rejected JobSet: {msg}. "
                    f"Fix: check OPA/Gatekeeper policies or admission webhooks."
                ),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            return CheckResult(
                name="Dry Run",
                status=CheckStatus.WARN,
                message=f"Dry run check failed: {e}",
            )

    async def _check_pod_security_admission(self) -> CheckResult:
        """Check namespace PSA labels for compatibility."""
        try:
            ns = await client.CoreV1Api(self.api).read_namespace(name=self.namespace)
            labels = (ns.metadata.labels or {}) if ns.metadata else {}

            psa_enforce = labels.get("pod-security.kubernetes.io/enforce")
            if not psa_enforce:
                return CheckResult(
                    name="Pod Security Admission",
                    status=CheckStatus.PASS,
                    message="No PSA enforcement label on namespace",
                )

            # Our pods run as non-root (UID 1000) with seccomp=RuntimeDefault
            # and drop all capabilities — compatible with "restricted"
            compatible_levels = {"privileged", "baseline", "restricted"}
            if psa_enforce in compatible_levels:
                return CheckResult(
                    name="Pod Security Admission",
                    status=CheckStatus.PASS,
                    message=f"PSA enforce level '{psa_enforce}' is compatible",
                )
            return CheckResult(
                name="Pod Security Admission",
                status=CheckStatus.WARN,
                message=f"Unknown PSA enforce level '{psa_enforce}'",
            )
        except ApiException as e:
            if e.status == 404:
                return CheckResult(
                    name="Pod Security Admission",
                    status=CheckStatus.WARN,
                    message=f"Namespace '{self.namespace}' not found",
                )
            return CheckResult(
                name="Pod Security Admission",
                status=CheckStatus.WARN,
                message=f"Could not check PSA: {e}",
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            return CheckResult(
                name="Pod Security Admission",
                status=CheckStatus.WARN,
                message=f"Could not check PSA: {e}",
            )

    async def _check_tolerations(self) -> CheckResult:
        """If tolerations specified, verify tainted nodes exist."""
        tolerations = self.deploy_config.pod_template.tolerations
        if not tolerations:
            return CheckResult(
                name="Tolerations",
                status=CheckStatus.SKIP,
                message="No tolerations specified",
            )

        try:
            node_list = await client.CoreV1Api(self.api).list_node()
            nodes = node_list.items
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            return CheckResult(
                name="Tolerations",
                status=CheckStatus.WARN,
                message=f"Could not check tolerations: {e}",
            )

        # Extract taint keys from our tolerations
        toleration_keys = {t.get("key") for t in tolerations if t.get("key")}

        # Check if any node has taints matching our tolerations
        for node in nodes:
            taints = (node.spec.taints or []) if node.spec else []
            for taint in taints:
                if taint.key in toleration_keys:
                    return CheckResult(
                        name="Tolerations",
                        status=CheckStatus.PASS,
                        message="Tainted nodes exist matching configured tolerations",
                    )

        return CheckResult(
            name="Tolerations",
            status=CheckStatus.WARN,
            message=(
                "No nodes have taints matching the specified tolerations. "
                "Tolerations may be unnecessary."
            ),
        )


def _is_node_ready_typed(node: Any) -> bool:
    """Check if a typed V1Node indicates Ready status."""
    if not node.status:
        return False
    conditions = node.status.conditions or []
    return any(c.type == "Ready" and c.status == "True" for c in conditions)

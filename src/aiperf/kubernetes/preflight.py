# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight check system for Kubernetes deployments.

This module provides comprehensive validation of Kubernetes cluster readiness
before deploying AIPerf benchmarks.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.console import logger
from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION
from aiperf.kubernetes.environment import (
    CONTROLLER_RESOURCE_KEYS,
    K8sEnvironment,
)
from aiperf.kubernetes.jobset import get_jobset_install_hint
from aiperf.kubernetes.preflight_utils import (
    check_rbac_access as _shared_check_rbac_access,
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


class CheckStatus(str, Enum):
    """Status of a pre-flight check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    INFO = "info"


_STATUS_ICONS: dict[CheckStatus, str] = {
    CheckStatus.PASS: "[green]✓[/green]",
    CheckStatus.FAIL: "[red]✗[/red]",
    CheckStatus.WARN: "[yellow]![/yellow]",
    CheckStatus.SKIP: "[dim]⊘[/dim]",
    CheckStatus.INFO: "[blue]ℹ[/blue]",
}

# Required RBAC permissions for AIPerf deployment: (verb, resource, api_group)
_REQUIRED_RBAC_PERMISSIONS: list[tuple[str, str, str]] = [
    ("create", "configmaps", ""),
    ("get", "pods", ""),
    ("get", "pods/log", ""),
    ("create", "roles", "rbac.authorization.k8s.io"),
    ("create", "rolebindings", "rbac.authorization.k8s.io"),
    ("create", "jobsets", JOBSET_GROUP),
    ("get", "jobsets", JOBSET_GROUP),
    ("delete", "jobsets", JOBSET_GROUP),
]


def _controller_resource_requirements() -> tuple[float, float]:
    """Return total controller-pod CPU cores and memory GiB."""
    cpu = 0.0
    memory = 0.0
    for key in CONTROLLER_RESOURCE_KEYS:
        settings = getattr(K8sEnvironment, key)
        cpu += parse_cpu(settings.CPU)
        memory += parse_memory_gib(settings.MEMORY)
    return cpu, memory


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""

    name: str
    """Human-readable check name."""

    status: CheckStatus
    """Pass/fail/warn/skip/info outcome."""

    message: str
    """Summary message describing the result."""

    details: list[str] = field(default_factory=list)
    """Additional detail lines for verbose output."""

    hints: list[str] = field(default_factory=list)
    """Actionable suggestions to resolve failures."""

    duration_ms: float | None = field(default=None)
    """Wall-clock time the check took, in milliseconds."""


class PreflightResultsDict(TypedDict):
    """Machine-parseable shape returned by ``PreflightResults.to_dict``."""

    passed: bool
    has_warnings: bool
    checks: list[dict[str, Any]]


@dataclass
class PreflightResults:
    """Aggregated results of all pre-flight checks."""

    checks: list[CheckResult] = field(default_factory=list)
    """Ordered list of individual check results."""

    @property
    def passed(self) -> bool:
        """Return True if no checks failed."""
        return not any(c.status == CheckStatus.FAIL for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        """Return True if any checks have warnings."""
        return any(c.status == CheckStatus.WARN for c in self.checks)

    def add(self, result: CheckResult) -> None:
        """Add a check result."""
        self.checks.append(result)

    def to_dict(self) -> PreflightResultsDict:
        """Convert results to a machine-parseable dict."""
        return {
            "passed": self.passed,
            "has_warnings": self.has_warnings,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "hints": c.hints,
                    "duration_ms": c.duration_ms,
                }
                for c in self.checks
            ],
        }

    def print_summary(self) -> None:
        """Print a summary of all check results."""
        logger.info("")
        if self.passed:
            if self.has_warnings:
                logger.info(
                    "[yellow bold]✓ Pre-flight checks passed with warnings[/yellow bold]"
                )
            else:
                logger.info("[green bold]✓ All pre-flight checks passed![/green bold]")
        else:
            logger.error("[red bold]✗ Some pre-flight checks failed[/red bold]")

        for check in self.checks:
            icon = _STATUS_ICONS[check.status]
            duration = (
                f" [dim]({check.duration_ms:.0f}ms)[/dim]"
                if check.duration_ms is not None
                else ""
            )
            logger.info(f"  {icon} {check.name}{duration}")

        logger.info("")
        if self.passed:
            logger.info("[dim]Your cluster is ready for AIPerf deployment.[/dim]")
        else:
            logger.info("[dim]Please resolve the issues above before deploying.[/dim]")


def _parse_image_ref(image: str) -> tuple[str, str, str]:
    """Parse a container image reference into (registry, repository, tag).

    Delegates to the shared utility in ``preflight_utils``.
    """
    return _shared_parse_image_ref(image)


def _format_duration(duration_ms: float | None) -> str:
    """Format check duration for display, or empty string if None."""
    return f" ({duration_ms:.0f}ms)" if duration_ms is not None else ""


def _print_check_result(result: CheckResult, check_num: int, total: int) -> None:
    """Log the result of a single check with verbose formatting."""
    icon = _STATUS_ICONS[result.status]
    duration = _format_duration(result.duration_ms)

    logger.info("")
    logger.info(f"[bold]\\[{check_num}/{total}] {result.name}{duration}[/bold]")
    logger.info(f"  {icon} {result.message}")

    for detail in result.details:
        logger.info(f"    {detail}")

    for hint in result.hints:
        logger.info(f"    [dim]Hint: {hint}[/dim]")


def _print_check_result_compact(result: CheckResult) -> None:
    """Log a single check result in compact one-line format."""
    icon = _STATUS_ICONS[result.status]
    logger.info(
        f"  {icon} {result.name}: {result.message}{_format_duration(result.duration_ms)}"
    )


class CLIPreflightChecker:
    """Runs pre-flight checks for Kubernetes deployment.

    Sibling: ``aiperf.operator.preflight.OperatorPreflightChecker`` handles
    operator-side reconcile-time preflight.
    """

    def __init__(
        self,
        namespace: str,
        *,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
        image: str | None = None,
        image_pull_secrets: list[str] | None = None,
        secrets: list[str] | None = None,
        endpoint_url: str | None = None,
        workers: int = 1,
    ):
        """Initialize the preflight checker.

        Args:
            namespace: Kubernetes namespace to check.
            kubeconfig: Path to kubeconfig file.
            kube_context: Kubernetes context to use.
            image: Container image to verify.
            image_pull_secrets: Image pull secret names to verify.
            secrets: Secret names to verify.
            endpoint_url: LLM endpoint URL to test connectivity.
            workers: Number of worker pods planned for deployment.
        """
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.kube_context = kube_context
        self.image = image
        self.image_pull_secrets = image_pull_secrets or []
        self.secrets = secrets or []
        self.endpoint_url = endpoint_url
        self.workers = workers

        self._api: ApiClient | None = None

    async def _run_check(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[CheckResult]],
        *,
        show_status: bool = False,
    ) -> CheckResult:
        """Run a single check with timing and optional status logging.

        Args:
            name: Check name (used as fallback in error message).
            check_fn: Async callable that returns a CheckResult.
            show_status: Print a status message before the check runs.

        Returns:
            CheckResult with duration_ms populated.
        """
        start = time.perf_counter()
        try:
            if show_status:
                logger.info(f"[cyan]... Checking {name}[/cyan]")
            result = await check_fn()
        except Exception as e:  # noqa: BLE001 - preflight dispatcher must never die on a single check failure
            result = CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                message=f"Check failed with error: {e}",
            )
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    async def run_quick_checks(
        self, *, show_progress: bool = False
    ) -> PreflightResults:
        """Run only critical pre-flight checks (connectivity, JobSet CRD, RBAC).

        When endpoint_url is set, also checks endpoint connectivity as a 4th check.

        Args:
            show_progress: Print compact results inline as each check completes.

        Returns:
            PreflightResults (without printing unless show_progress=True).
            Short-circuits on connectivity failure.
        """
        results = PreflightResults()

        checks: list[tuple[str, Callable[[], Awaitable[CheckResult]]]] = [
            ("Cluster Connectivity", self._check_cluster_connectivity),
            ("JobSet CRD", self._check_jobset_crd),
            ("RBAC Permissions", self._check_rbac_permissions),
        ]
        if self.endpoint_url:
            checks.append(("Endpoint Connectivity", self._check_endpoint_connectivity))

        for name, check_fn in checks:
            result = await self._run_check(name, check_fn)
            results.add(result)
            if show_progress:
                _print_check_result_compact(result)
            if name == "Cluster Connectivity" and result.status == CheckStatus.FAIL:
                return results

        return results

    async def run_all_checks(self) -> PreflightResults:
        """Run all pre-flight checks and return results."""
        results = PreflightResults()

        checks: list[tuple[str, Callable[[], Awaitable[CheckResult]]]] = [
            ("Cluster Connectivity", self._check_cluster_connectivity),
            ("Kubernetes Version", self._check_kubernetes_version),
            ("Namespace", self._check_namespace),
            ("RBAC Permissions", self._check_rbac_permissions),
            ("JobSet CRD", self._check_jobset_crd),
            ("JobSet Controller", self._check_jobset_controller),
            ("Resource Quotas", self._check_resource_quotas),
            ("Node Resources", self._check_node_resources),
            ("Secrets", self._check_secrets),
            ("Image Pull", self._check_image),
            ("Network Policies", self._check_network_policies),
            ("DNS Resolution", self._check_dns),
            ("Endpoint Connectivity", self._check_endpoint_connectivity),
        ]

        total = len(checks)
        for i, (name, check_fn) in enumerate(checks, 1):
            result = await self._run_check(name, check_fn, show_status=True)
            results.add(result)
            _print_check_result(result, i, total)

            if name == "Cluster Connectivity" and result.status == CheckStatus.FAIL:
                break

        results.print_summary()
        return results

    async def _check_cluster_connectivity(self) -> CheckResult:
        """Check if we can connect to the Kubernetes cluster."""
        from aiperf.kubernetes.client import k8s_client

        try:
            cm = k8s_client(
                kubeconfig=self.kubeconfig,
                context=self.kube_context,
            )
            self._api = await cm.__aenter__()
            # Validate connectivity by reading the cluster version.
            await client.VersionApi(self._api).get_code()

            return CheckResult(
                name="Cluster Connectivity",
                status=CheckStatus.PASS,
                message="Connected to Kubernetes cluster",
            )
        except (
            ApiException,
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
        ) as e:
            return CheckResult(
                name="Cluster Connectivity",
                status=CheckStatus.FAIL,
                message=f"Failed to connect: {e}",
                hints=[
                    "Check your kubeconfig (~/.kube/config) or KUBECONFIG env var",
                    "Verify the cluster is running and accessible",
                ],
            )

    async def _check_kubernetes_version(self) -> CheckResult:
        """Check Kubernetes version compatibility."""
        import re

        try:
            version = await client.VersionApi(self._api).get_code()
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
            else:
                return CheckResult(
                    name="Kubernetes Version",
                    status=CheckStatus.FAIL,
                    message=f"Kubernetes {git_version} is below minimum 1.24",
                    hints=["Upgrade your Kubernetes cluster to version 1.24 or later"],
                )
        except (
            ApiException,
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
        ) as e:
            return CheckResult(
                name="Kubernetes Version",
                status=CheckStatus.WARN,
                message=f"Could not determine version: {e}",
            )

    async def _check_access(self, verb: str, resource: str, group: str) -> bool:
        """Check if current user has a specific RBAC permission.

        Delegates to the shared utility in ``preflight_utils``.
        """
        return await _shared_check_rbac_access(
            self._api,
            verb=verb,
            resource=resource,
            group=group,
            namespace=self.namespace,
        )

    async def _check_namespace(self) -> CheckResult:
        """Check if namespace exists or can be created."""
        try:
            await client.CoreV1Api(self._api).read_namespace(self.namespace)
            return CheckResult(
                name="Namespace",
                status=CheckStatus.PASS,
                message=f"Namespace '{self.namespace}' exists",
            )
        except ApiException as e:
            if e.status == 404:
                try:
                    allowed = await self._check_access("create", "namespaces", "")
                    if allowed:
                        return CheckResult(
                            name="Namespace",
                            status=CheckStatus.PASS,
                            message=f"Namespace '{self.namespace}' will be created",
                        )
                    else:
                        return CheckResult(
                            name="Namespace",
                            status=CheckStatus.FAIL,
                            message=f"Namespace '{self.namespace}' does not exist",
                            hints=[
                                f"Ask an admin to create namespace '{self.namespace}'"
                            ],
                        )
                except (
                    ApiException,
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    OSError,
                    RuntimeError,
                ) as perm_err:
                    return CheckResult(
                        name="Namespace",
                        status=CheckStatus.WARN,
                        message=f"Namespace '{self.namespace}' does not exist, cannot verify create permission",
                        details=[str(perm_err)],
                    )
            return CheckResult(
                name="Namespace",
                status=CheckStatus.FAIL,
                message=f"Error checking namespace: HTTP {e.status}",
            )

    async def _check_rbac_permissions(self) -> CheckResult:
        """Check required RBAC permissions."""
        missing = []
        passed = []

        for verb, resource, group in _REQUIRED_RBAC_PERMISSIONS:
            try:
                allowed = await self._check_access(verb, resource, group)
                display = f"{group}/{resource}" if group else resource

                if allowed:
                    passed.append(f"{verb} {display}")
                else:
                    missing.append(f"{verb} {display}")
            except (
                ApiException,
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
                RuntimeError,
            ) as e:
                display = f"{group}/{resource}" if group else resource
                missing.append(f"{verb} {display} (check failed: {e})")

        if missing:
            return CheckResult(
                name="RBAC Permissions",
                status=CheckStatus.FAIL,
                message=f"Missing {len(missing)} required permission(s)",
                details=[f"  ✗ {p}" for p in missing],
                hints=[
                    "Contact your cluster admin to grant the required permissions",
                    f"Permissions needed in namespace '{self.namespace}'",
                ],
            )
        else:
            return CheckResult(
                name="RBAC Permissions",
                status=CheckStatus.PASS,
                message=f"All {len(passed)} required permissions granted",
                details=[f"  ✓ {p}" for p in passed],
            )

    async def _check_jobset_crd(self) -> CheckResult:
        """Check if JobSet CRD is installed."""
        crd_name = f"{JOBSET_PLURAL}.{JOBSET_GROUP}"
        try:
            await client.ApiextensionsV1Api(self._api).read_custom_resource_definition(
                crd_name
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
                    message="JobSet CRD not found",
                    hints=[
                        get_jobset_install_hint(),
                    ],
                )
            return CheckResult(
                name="JobSet CRD",
                status=CheckStatus.WARN,
                message=f"Error checking JobSet CRD: HTTP {e.status}",
            )

    async def _find_deployment(
        self, namespace: str, name_substring: str
    ) -> tuple[bool, bool]:
        """Check if a deployment matching name_substring exists and is ready.

        Args:
            namespace: Kubernetes namespace to search.
            name_substring: Lowercase substring to match in deployment names.

        Returns:
            Tuple of (found, ready).
        """
        deployments = (
            await client.AppsV1Api(self._api).list_namespaced_deployment(namespace)
        ).items
        for deploy in deployments:
            name = deploy.metadata.name if deploy.metadata else ""
            if name_substring in (name or "").lower():
                ready_replicas = (
                    (deploy.status.ready_replicas or 0) if deploy.status else 0
                )
                return True, ready_replicas > 0
        return False, False

    async def _check_jobset_controller(self) -> CheckResult:
        """Check if JobSet controller is running."""
        try:
            controller_found, controller_ready = await self._find_deployment(
                "jobset-system", "jobset"
            )

            if controller_ready:
                return CheckResult(
                    name="JobSet Controller",
                    status=CheckStatus.PASS,
                    message="JobSet controller is running",
                )
            elif controller_found:
                return CheckResult(
                    name="JobSet Controller",
                    status=CheckStatus.WARN,
                    message="JobSet controller found but not ready",
                    hints=["Check 'kubectl get pods -n jobset-system' for issues"],
                )
            else:
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

    async def _check_resource_quotas(self) -> CheckResult:
        """Check resource quotas in the namespace."""
        try:
            quotas = (
                await client.CoreV1Api(self._api).list_namespaced_resource_quota(
                    self.namespace
                )
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
            required_cpu = ctrl_cpu + (worker_cpu * self.workers)
            required_mem = ctrl_mem + (worker_mem * self.workers)

            details = []
            would_exceed = False
            for quota in quotas:
                name = quota.metadata.name if quota.metadata else ""
                details.append(f"ResourceQuota '{name}':")
                hard = (quota.status.hard or {}) if quota.status else {}
                used = (quota.status.used or {}) if quota.status else {}
                for resource, limit in hard.items():
                    details.append(
                        f"    {resource}: {used.get(resource, '0')} / {limit}"
                    )

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

            if would_exceed:
                details.append(
                    f"Benchmark needs: {format_cpu(required_cpu)} CPU, "
                    f"{format_memory(required_mem)} memory ({self.workers} workers)"
                )
                return CheckResult(
                    name="Resource Quotas",
                    status=CheckStatus.WARN,
                    message="Benchmark may exceed resource quota(s)",
                    details=details,
                    hints=[
                        "Request a quota increase or reduce worker count",
                        "Quota may not apply if benchmark creates its own namespace",
                    ],
                )

            return CheckResult(
                name="Resource Quotas",
                status=CheckStatus.INFO,
                message=f"Found {len(quotas)} resource quota(s)",
                details=details,
            )
        except ApiException as e:
            return CheckResult(
                name="Resource Quotas",
                status=CheckStatus.WARN,
                message=f"Error checking quotas: HTTP {e.status}",
            )

    async def _check_node_resources(self) -> CheckResult:
        """Check if cluster has sufficient node resources."""
        try:
            nodes = (await client.CoreV1Api(self._api).list_node()).items

            if not nodes:
                return CheckResult(
                    name="Node Resources",
                    status=CheckStatus.FAIL,
                    message="No nodes found in cluster",
                )

            total_cpu = 0.0
            total_memory = 0.0
            ready_nodes = 0

            for node in nodes:
                conditions = (node.status.conditions or []) if node.status else []
                is_ready = any(
                    c.type == "Ready" and c.status == "True" for c in conditions
                )

                allocatable = (node.status.allocatable or {}) if node.status else {}
                if is_ready and allocatable:
                    ready_nodes += 1
                    total_cpu += parse_cpu(allocatable.get("cpu", "0"))
                    total_memory += parse_memory_gib(allocatable.get("memory", "0"))

            ctrl_cpu, ctrl_mem = _controller_resource_requirements()
            worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
            worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)

            required_cpu = ctrl_cpu + (worker_cpu * self.workers)
            required_mem = ctrl_mem + (worker_mem * self.workers)

            details = [
                f"Cluster: {ready_nodes} ready nodes, "
                f"{format_cpu(total_cpu)} CPU, {format_memory(total_memory)} memory",
                f"Deployment estimate: {format_cpu(required_cpu)} CPU, "
                f"{format_memory(required_mem)} memory ({self.workers} workers)",
            ]

            if required_cpu > total_cpu or required_mem > total_memory:
                return CheckResult(
                    name="Node Resources",
                    status=CheckStatus.WARN,
                    message="Cluster may not have enough resources",
                    details=details,
                    hints=["Consider reducing worker count or adding cluster capacity"],
                )

            # Per-node schedulability: ensure at least one node can fit the largest pod
            max_pod_cpu = max(ctrl_cpu, worker_cpu)
            max_pod_mem = max(ctrl_mem, worker_mem)
            node_can_fit = False
            for node in nodes:
                conditions = (node.status.conditions or []) if node.status else []
                is_ready = any(
                    c.type == "Ready" and c.status == "True" for c in conditions
                )
                if not is_ready:
                    continue
                allocatable = (node.status.allocatable or {}) if node.status else {}
                node_cpu = parse_cpu(allocatable.get("cpu", "0"))
                node_mem = parse_memory_gib(allocatable.get("memory", "0"))
                if node_cpu >= max_pod_cpu and node_mem >= max_pod_mem:
                    node_can_fit = True
                    break

            if not node_can_fit:
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
        except (
            ApiException,
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
        ) as e:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message=f"Could not check node resources: {e}",
            )

    async def _check_secrets(self) -> CheckResult:
        """Check if required secrets exist."""
        all_secrets = self.image_pull_secrets + self.secrets
        if not all_secrets:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.SKIP,
                message="No secrets specified to verify",
                hints=[
                    "Use --image-pull-secret or --secret to verify specific secrets"
                ],
            )

        found = []
        missing = []
        permission_denied = []

        core = client.CoreV1Api(self._api)
        for secret_name in all_secrets:
            try:
                await core.read_namespaced_secret(secret_name, self.namespace)
                found.append(secret_name)
            except ApiException as e:
                if e.status == 404:
                    missing.append(secret_name)
                elif e.status == 403:
                    permission_denied.append(secret_name)
                else:
                    missing.append(f"{secret_name} (error: HTTP {e.status})")

        details = []
        if found:
            details.extend([f"  ✓ {s}" for s in found])
        if missing:
            details.extend([f"  ✗ {s} (not found)" for s in missing])
        if permission_denied:
            details.extend([f"  ? {s} (permission denied)" for s in permission_denied])

        if missing:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.FAIL,
                message=f"{len(missing)} secret(s) not found",
                details=details,
                hints=["Create missing secrets with 'kubectl create secret ...'"],
            )
        elif permission_denied:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.WARN,
                message=f"Cannot verify {len(permission_denied)} secret(s)",
                details=details,
            )
        else:
            return CheckResult(
                name="Secrets",
                status=CheckStatus.PASS,
                message=f"All {len(found)} secret(s) verified",
                details=details,
            )

    async def _check_image(self) -> CheckResult:
        """Check image availability information."""
        if not self.image:
            return CheckResult(
                name="Image Pull",
                status=CheckStatus.SKIP,
                message="No image specified to verify",
                hints=["Use --image to check pull access"],
            )

        details = [f"Image: {self.image}"]

        registry, _repo, tag = _parse_image_ref(self.image)
        if tag:
            details.append(f"Registry: {registry}, Tag: {tag}")
        else:
            details.append(f"Registry: {registry}, Tag: latest (implicit)")

        if self.image_pull_secrets:
            details.append(f"Pull secrets: {', '.join(self.image_pull_secrets)}")

        public_registries = {
            "docker.io",
            "registry-1.docker.io",
            "ghcr.io",
            "quay.io",
            "nvcr.io",
            "registry.k8s.io",
        }
        is_public = registry in public_registries

        # Pull secrets were specified and will be verified by the secrets check
        if self.image_pull_secrets:
            return CheckResult(
                name="Image Pull",
                status=CheckStatus.PASS,
                message="Image specified with pull secrets configured",
                details=details,
            )

        if is_public:
            details.append(f"Public registry: {registry}")
            return CheckResult(
                name="Image Pull",
                status=CheckStatus.INFO,
                message=f"Image from public registry ({registry})",
                details=details,
                hints=[
                    f"Verify manually: kubectl run test --image={self.image} "
                    "--rm -it --restart=Never -- echo ok"
                ],
            )

        # Private registry with no pull secrets
        return CheckResult(
            name="Image Pull",
            status=CheckStatus.WARN,
            message="Image may require pull secrets",
            details=details,
            hints=[
                f"Registry '{registry}' may require authentication",
                "Use --image-pull-secret to specify credentials",
                f"Verify manually: kubectl run test --image={self.image} "
                "--rm -it --restart=Never -- echo ok",
            ],
        )

    async def _check_network_policies(self) -> CheckResult:
        """Check for restrictive network policies."""
        try:
            policies = (
                await client.NetworkingV1Api(self._api).list_namespaced_network_policy(
                    self.namespace
                )
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

    async def _check_dns(self) -> CheckResult:
        """Check DNS resolution capability."""
        try:
            coredns_found, coredns_ready = await self._find_deployment(
                "kube-system", "coredns"
            )

            if coredns_ready:
                return CheckResult(
                    name="DNS Resolution",
                    status=CheckStatus.PASS,
                    message="CoreDNS is running",
                    details=[
                        "Workers will resolve controller DNS name for ZMQ connections"
                    ],
                )
            elif coredns_found:
                return CheckResult(
                    name="DNS Resolution",
                    status=CheckStatus.WARN,
                    message="CoreDNS found but may not be ready",
                    hints=[
                        "Check 'kubectl get pods -n kube-system -l k8s-app=kube-dns'"
                    ],
                )
            else:
                return CheckResult(
                    name="DNS Resolution",
                    status=CheckStatus.WARN,
                    message="CoreDNS not found in kube-system",
                    hints=["Verify your cluster has a working DNS service"],
                )
        except (
            ApiException,
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
        ) as e:
            return CheckResult(
                name="DNS Resolution",
                status=CheckStatus.WARN,
                message=f"Could not verify DNS: {e}",
            )

    async def _check_endpoint_connectivity(self) -> CheckResult:
        """Check if the LLM endpoint is potentially reachable."""
        if not self.endpoint_url:
            return CheckResult(
                name="Endpoint Connectivity",
                status=CheckStatus.SKIP,
                message="No endpoint URL specified",
                hints=["Use --endpoint to verify LLM endpoint connectivity"],
            )

        try:
            from urllib.parse import urlparse

            parsed = urlparse(self.endpoint_url)
            host = parsed.hostname or "unknown"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            details = [
                f"Endpoint: {self.endpoint_url}",
                f"Host: {host}, Port: {port}",
            ]

            if ".svc" in host or ".svc.cluster.local" in host:
                try:
                    # Parse name.namespace.svc[.cluster.local] or name.svc[.cluster.local]
                    before_svc = host.split(".svc")[0]
                    if "." in before_svc:
                        svc_name, svc_ns = before_svc.rsplit(".", 1)
                    else:
                        svc_name, svc_ns = before_svc, "default"

                    await client.CoreV1Api(self._api).read_namespaced_service(
                        svc_name, svc_ns
                    )
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
                        hints=[
                            f"Verify the service exists: kubectl get svc -A | grep {svc_name}"
                        ],
                    )

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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight check system for Kubernetes deployments.

This module provides comprehensive validation of Kubernetes cluster readiness
before deploying AIPerf benchmarks.

The individual check implementations live in
``aiperf.kubernetes.preflight_checks`` as stateless free functions; this module
hosts the public API (``CheckResult``, ``CheckStatus``, ``PreflightResults``,
``CLIPreflightChecker``) and orchestrates the checks.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

from kubernetes_asyncio import (
    client,  # noqa: F401 — re-exported so tests can patch `aiperf.kubernetes.preflight.client.*`
)
from kubernetes_asyncio.client import ApiClient

from aiperf.kubernetes import preflight_checks
from aiperf.kubernetes.console import logger
from aiperf.kubernetes.preflight_utils import (
    parse_image_ref as _shared_parse_image_ref,
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

        # Open the shared ApiClient context manager lazily so later checks can
        # reuse the connection; the context is closed by the caller via the
        # process lifecycle.
        cm = k8s_client(kubeconfig=self.kubeconfig, context=self.kube_context)
        try:
            self._api = await cm.__aenter__()
        except Exception as e:  # noqa: BLE001 - surface connection failures as a failed check
            return CheckResult(
                name="Cluster Connectivity",
                status=CheckStatus.FAIL,
                message=f"Failed to connect: {e}",
                hints=[
                    "Check your kubeconfig (~/.kube/config) or KUBECONFIG env var",
                    "Verify the cluster is running and accessible",
                ],
            )
        return await preflight_checks.check_cluster_connectivity(self._api)

    async def _check_kubernetes_version(self) -> CheckResult:
        """Check Kubernetes version compatibility."""
        return await preflight_checks.check_kubernetes_version(self._api)

    async def _check_namespace(self) -> CheckResult:
        """Check if namespace exists or can be created."""
        return await preflight_checks.check_namespace(
            self._api, namespace=self.namespace
        )

    async def _check_rbac_permissions(self) -> CheckResult:
        """Check required RBAC permissions."""
        return await preflight_checks.check_rbac_permissions(
            self._api, namespace=self.namespace
        )

    async def _check_jobset_crd(self) -> CheckResult:
        """Check if JobSet CRD is installed."""
        return await preflight_checks.check_jobset_crd(self._api)

    async def _check_jobset_controller(self) -> CheckResult:
        """Check if JobSet controller is running."""
        return await preflight_checks.check_jobset_controller(self._api)

    async def _check_resource_quotas(self) -> CheckResult:
        """Check resource quotas in the namespace."""
        return await preflight_checks.check_resource_quotas(
            self._api, namespace=self.namespace, workers=self.workers
        )

    async def _check_node_resources(self) -> CheckResult:
        """Check if cluster has sufficient node resources."""
        return await preflight_checks.check_node_resources(
            self._api, workers=self.workers
        )

    async def _check_secrets(self) -> CheckResult:
        """Check if required secrets exist."""
        return await preflight_checks.check_secrets(
            self._api,
            namespace=self.namespace,
            image_pull_secrets=self.image_pull_secrets,
            secrets=self.secrets,
        )

    async def _check_image(self) -> CheckResult:
        """Check image availability information."""
        return await preflight_checks.check_image(
            self._api,
            image=self.image,
            image_pull_secrets=self.image_pull_secrets,
        )

    async def _check_network_policies(self) -> CheckResult:
        """Check for restrictive network policies."""
        return await preflight_checks.check_network_policies(
            self._api, namespace=self.namespace
        )

    async def _check_dns(self) -> CheckResult:
        """Check DNS resolution capability."""
        return await preflight_checks.check_dns(self._api)

    async def _check_endpoint_connectivity(self) -> CheckResult:
        """Check if the LLM endpoint is potentially reachable."""
        return await preflight_checks.check_endpoint_connectivity(
            self._api, endpoint_url=self.endpoint_url
        )

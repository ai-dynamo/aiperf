# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main ``OperatorPreflightChecker`` class and its tiered orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiohttp
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.preflight import CheckResult, CheckStatus, PreflightResults
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.preflight._common import _is_node_ready_typed  # re-exported
from aiperf.operator.preflight._infra import _InfraChecksMixin
from aiperf.operator.preflight._resources import _ResourceChecksMixin
from aiperf.operator.preflight._tier1 import _Tier1ChecksMixin
from aiperf.operator.preflight._workload import _WorkloadChecksMixin

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiperf.config import AIPerfConfig
    from aiperf.config.deployment import DeploymentConfig
    from aiperf.kubernetes.resources import KubernetesDeployment

logger = logging.getLogger(__name__)

__all__ = ["OperatorPreflightChecker", "_is_node_ready_typed"]


@dataclass(slots=True)
class OperatorPreflightChecker(
    _Tier1ChecksMixin,
    _InfraChecksMixin,
    _ResourceChecksMixin,
    _WorkloadChecksMixin,
):
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

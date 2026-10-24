# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource-related pre-flight checks (nodes, quotas, memory estimation, tolerations)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.preflight import CheckResult, CheckStatus
from aiperf.kubernetes.resource_quota import quota_violation
from aiperf.kubernetes.utils import (
    format_cpu,
    format_memory,
    parse_cpu,
    parse_memory_gib,
)
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.preflight._common import (
    _is_node_ready_typed,
    controller_resource_requirements,
)


def _toleration_matches_taint(
    toleration: dict[str, object], taint_key: str, taint_effect: str
) -> bool:
    """Return True if a single toleration tolerates a (key, effect) taint.

    Implements the K8s toleration matching rules used by the scheduler:
      - operator=Exists with empty key tolerates all taints (and may further
        scope by effect).
      - operator=Exists with a key tolerates any taint with that key.
      - operator=Equal (or unset, the default) requires key match.
      - empty effect on the toleration matches any effect.

    Value matching is intentionally loose — capacity-planning preflight only
    needs to know whether the *node* is potentially usable, not whether a
    specific pod will actually land. Conservative on the FAIL side.
    """
    op = toleration.get("operator") or "Equal"
    tol_key = toleration.get("key") or ""
    tol_effect = toleration.get("effect") or ""

    if op == "Exists" and not tol_key:
        # Tolerates all taints (optionally scoped by effect).
        return not tol_effect or tol_effect == taint_effect

    if tol_key != taint_key:
        return False

    return not tol_effect or tol_effect == taint_effect


def _node_is_schedulable_for(
    node: object, tolerations: list[dict[str, object]]
) -> bool:
    """Return True if ``node``'s NoSchedule/NoExecute taints are all tolerated.

    PreferNoSchedule is treated as schedulable (it's a soft preference).
    """
    spec = getattr(node, "spec", None)
    taints = (getattr(spec, "taints", None) or []) if spec else []
    for taint in taints:
        effect = getattr(taint, "effect", None) or ""
        if effect not in ("NoSchedule", "NoExecute"):
            continue
        key = getattr(taint, "key", None) or ""
        if not any(_toleration_matches_taint(t, key, effect) for t in tolerations):
            return False
    return True


def _node_matches_selector(node: object, node_selector: dict[str, str] | None) -> bool:
    """Return True if ``node``'s labels satisfy every nodeSelector entry."""
    if not node_selector:
        return True
    metadata = getattr(node, "metadata", None)
    labels = (getattr(metadata, "labels", None) or {}) if metadata else {}
    return all(labels.get(key) == value for key, value in node_selector.items())


def _matching_schedulable_nodes(
    nodes: list,
    node_selector: dict[str, str] | None,
    tolerations: list[dict[str, object]],
) -> list:
    """Nodes that are Ready, tolerate their own taints, and match the selector."""
    return [
        node
        for node in nodes
        if _is_node_ready_typed(node)
        and _node_is_schedulable_for(node, tolerations)
        and _node_matches_selector(node, node_selector)
    ]


@dataclass(slots=True)
class _CapacityTotals:
    """Free and allocatable CPU/mem summed over Ready, placement-eligible nodes."""

    ready_nodes: int
    skipped_tainted: int
    free_cpu: float
    free_memory: float
    allocatable_cpu: float
    allocatable_memory: float
    free_known: bool
    """Whether running-pod requests were subtracted (False => free == allocatable)."""

    def allocatable_shortfall(self, required_cpu: float, required_mem: float) -> bool:
        """True when even total allocatable cannot fit the requirement."""
        return (
            required_cpu > self.allocatable_cpu
            or required_mem > self.allocatable_memory
        )

    def free_shortfall(self, required_cpu: float, required_mem: float) -> bool:
        """True when currently-free capacity cannot fit the requirement."""
        return required_cpu > self.free_cpu or required_mem > self.free_memory


class _ResourceChecksMixin:
    """Checks related to cluster/node resources, quotas, memory, and tolerations."""

    async def _capacity_totals(
        self, tolerations: list[dict[str, object]]
    ) -> _CapacityTotals | CheckResult:
        """Sum free and allocatable capacity over Ready, untainted-for-us nodes.

        Returns a WARN CheckResult (instead of totals) when the node list is
        unavailable or empty; the caller surfaces it directly.
        """
        from aiperf.kubernetes.preflight_capacity_checks import (
            _node_free,
            _requested_by_node,
        )

        try:
            node_list = await client.CoreV1Api(self.api).list_node()
            nodes = node_list.items
        except (TimeoutError, ApiException, aiohttp.ClientError, OSError) as e:
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

        requested = await _requested_by_node(self.api)

        totals = _CapacityTotals(0, 0, 0.0, 0.0, 0.0, 0.0, requested is not None)
        for node in nodes:
            if not _is_node_ready_typed(node):
                continue
            if not _node_is_schedulable_for(node, tolerations):
                totals.skipped_tainted += 1
                continue
            totals.ready_nodes += 1
            cpu, memory = _node_free(node, requested)
            totals.free_cpu += cpu
            totals.free_memory += memory
            alloc_cpu, alloc_memory = _node_free(node, None)
            totals.allocatable_cpu += alloc_cpu
            totals.allocatable_memory += alloc_memory
        return totals

    def _required_capacity(self) -> tuple[float, float]:
        """Total CPU cores / memory GiB the benchmark needs (controller + workers)."""
        from aiperf.kubernetes.environment import K8sEnvironment

        ctrl_cpu, ctrl_mem = controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)
        return (
            ctrl_cpu + (worker_cpu * self.num_pods),
            ctrl_mem + (worker_mem * self.num_pods),
        )

    @staticmethod
    def _provable_capacity_failure(
        totals: _CapacityTotals,
        *,
        required_cpu: float,
        required_mem: float,
    ) -> CheckResult:
        """Build the strict-mode FAIL for a CONFIRMED capacity shortfall."""
        tainted_suffix = (
            f" ({totals.skipped_tainted} tainted node(s) excluded)"
            if totals.skipped_tainted
            else ""
        )
        return CheckResult(
            name="Node Resources",
            status=CheckStatus.FAIL,
            message=(
                f"Cluster cannot fit this benchmark (shortfall persisted across a "
                f"{OperatorEnvironment.PREFLIGHT_STRICT_CONFIRM_SECONDS:.0f}s "
                f"confirmation window). Need {format_cpu(required_cpu)} CPU, "
                f"{format_memory(required_mem)} mem but only "
                f"{format_cpu(totals.allocatable_cpu)} CPU, "
                f"{format_memory(totals.allocatable_memory)} mem allocatable across "
                f"{totals.ready_nodes} schedulable node(s){tainted_suffix}. "
                "Failing fast because AIPERF_PREFLIGHT_STRICT_RESOURCES=true."
            ),
            hints=[
                "Reduce worker count or add cluster capacity",
                "Unset AIPERF_PREFLIGHT_STRICT_RESOURCES to only warn",
            ],
        )

    async def _confirm_capacity_shortfall(
        self,
        tolerations: list[dict[str, object]],
        required_cpu: float,
        required_mem: float,
    ) -> _CapacityTotals | None:
        """Re-evaluate capacity after a bounded delay and return fresh totals.

        Advertised allocatable is a changing snapshot, not proof a job can
        never fit: kubelet re-asserts real capacity between test-floor
        re-applications (the CI CPU overcommit reconciler re-patches every
        5s), tainted/NotReady nodes recover, and autoscalers add capacity.
        The shortfall must persist across
        ``AIPERF_PREFLIGHT_STRICT_CONFIRM_SECONDS`` before the job is
        permanently rejected, so one low snapshot cannot fail a job the
        cluster would admit seconds later. Returns None when the re-check
        itself fails (shortfall not provable; caller falls back to WARN).
        """
        await asyncio.sleep(OperatorEnvironment.PREFLIGHT_STRICT_CONFIRM_SECONDS)
        totals = await self._capacity_totals(tolerations)
        return None if isinstance(totals, CheckResult) else totals

    async def _check_node_resources(self) -> CheckResult:
        """Check aggregate free CPU/mem across schedulable Ready nodes.

        Nodes carrying NoSchedule/NoExecute taints that aren't tolerated by
        ``deploy_config.pod_template.tolerations`` are excluded from the
        capacity sum — otherwise a 50-node cluster of all-tainted GPU nodes
        would falsely report "sufficient resources" for a CPU-only workload.

        ``allocatable`` is total node capacity, not what's actually free: it
        does not account for requests already made by pods scheduled on the
        node. Without subtracting those, a cluster booked to 100% still
        reports PASS here and every AIPerf pod then sits Pending.

        Strict mode (``AIPERF_PREFLIGHT_STRICT_RESOURCES``) FAILs only on a
        CONFIRMED allocatable-level shortfall: the snapshot must still be
        insufficient after a bounded confirmation window, so transiently low
        capacity (a test floor between kubelet re-assertions, recovering
        NotReady/tainted nodes, an autoscaler mid-scale-up) cannot permanently
        reject a job the cluster would admit moments later.
        """
        if skip := self._resource_mode_skip("Node Resources"):
            return skip

        tolerations = self.deploy_config.pod_template.tolerations or []
        totals = await self._capacity_totals(tolerations)
        if isinstance(totals, CheckResult):
            return totals

        required_cpu, required_mem = self._required_capacity()

        if OperatorEnvironment.PREFLIGHT_STRICT_RESOURCES and totals.allocatable_shortfall(
            required_cpu, required_mem
        ):
            confirmed = await self._confirm_capacity_shortfall(
                tolerations, required_cpu, required_mem
            )
            if confirmed is not None:
                totals = confirmed
                if totals.allocatable_shortfall(required_cpu, required_mem):
                    return self._provable_capacity_failure(
                        totals,
                        required_cpu=required_cpu,
                        required_mem=required_mem,
                    )

        tainted_suffix = (
            f" ({totals.skipped_tainted} tainted node(s) excluded)"
            if totals.skipped_tainted
            else ""
        )
        capacity_label = "free" if totals.free_known else "allocatable"
        unknown_suffix = (
            ""
            if totals.free_known
            else " (could not list running pods; showing total allocatable, not free)"
        )

        if totals.ready_nodes == 0:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message=(f"No Ready, schedulable nodes available{tainted_suffix}"),
                hints=[
                    "Add tolerations or untaint nodes",
                    "Check node Ready condition",
                ],
            )

        if totals.free_shortfall(required_cpu, required_mem):
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message=(
                    f"Cluster may not have enough resources. "
                    f"Need {format_cpu(required_cpu)} CPU, {format_memory(required_mem)} mem "
                    f"but only {format_cpu(totals.free_cpu)} CPU, "
                    f"{format_memory(totals.free_memory)} mem "
                    f"{capacity_label} across {totals.ready_nodes} schedulable node(s)"
                    f"{tainted_suffix}{unknown_suffix}."
                ),
                hints=["Reduce worker count or add cluster capacity"],
            )

        return CheckResult(
            name="Node Resources",
            status=CheckStatus.PASS,
            message=(
                f"Cluster has sufficient resources "
                f"({totals.ready_nodes} schedulable nodes, "
                f"{format_cpu(totals.free_cpu)} CPU, "
                f"{format_memory(totals.free_memory)} {capacity_label} mem)"
                f"{tainted_suffix}{unknown_suffix}"
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
        except (TimeoutError, ApiException, aiohttp.ClientError, OSError) as e:
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
        """Check that at least one matching Ready node can fit the largest pod.

        Two-phase: first a structural check against raw allocatable (a node
        too small to ever hold one pod is a misconfiguration, not a
        transient booking), then — when the pod list is available — a
        free-capacity check that subtracts what's already requested by pods
        scheduled on the node, so a fully-booked cluster doesn't PASS here
        only to leave every AIPerf pod Pending.
        """
        from aiperf.kubernetes.environment import K8sEnvironment
        from aiperf.kubernetes.preflight_capacity_checks import (
            _node_free,
            _requested_by_node,
        )

        if skip := self._resource_mode_skip("Per-Node Schedulability"):
            return skip

        try:
            node_list = await client.CoreV1Api(self.api).list_node()
            nodes = node_list.items
        except (TimeoutError, ApiException, aiohttp.ClientError, OSError) as e:
            return CheckResult(
                name="Per-Node Schedulability",
                status=CheckStatus.WARN,
                message=f"Could not check per-node schedulability: {e}",
            )

        ctrl_cpu, ctrl_mem = controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)
        max_pod_cpu = max(ctrl_cpu, worker_cpu)
        max_pod_mem = max(ctrl_mem, worker_mem)

        node_selector = self.deploy_config.pod_template.node_selector
        tolerations = self.deploy_config.pod_template.tolerations or []

        matching_nodes = _matching_schedulable_nodes(nodes, node_selector, tolerations)

        def _fits(
            node: object, requested: dict[str, tuple[float, float]] | None
        ) -> bool:
            cpu, memory = _node_free(node, requested)
            return cpu >= max_pod_cpu and memory >= max_pod_mem

        if not any(_fits(node, None) for node in matching_nodes):
            return CheckResult(
                name="Per-Node Schedulability",
                status=CheckStatus.FAIL,
                message=(
                    f"No single node can fit the largest pod "
                    f"({format_cpu(max_pod_cpu)} CPU, {format_memory(max_pod_mem)} mem). "
                    f"Add larger nodes or reduce pod resource requirements."
                ),
            )

        requested = await _requested_by_node(self.api)
        if requested is not None and not any(
            _fits(node, requested) for node in matching_nodes
        ):
            return CheckResult(
                name="Per-Node Schedulability",
                status=CheckStatus.WARN,
                message=(
                    "No node currently has enough free capacity for the largest pod "
                    f"({format_cpu(max_pod_cpu)} CPU, {format_memory(max_pod_mem)} mem) "
                    "once already-scheduled pods are accounted for."
                ),
                hints=[
                    "Pods will stay Pending until running workloads release capacity",
                    "Reduce worker count, free capacity, or add nodes",
                ],
            )

        return CheckResult(
            name="Per-Node Schedulability",
            status=CheckStatus.PASS,
            message="At least one node can fit the largest pod",
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

        ctrl_cpu, ctrl_mem = controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)
        # +1 for the controller pod itself
        required_pods = self.num_pods + 1
        required_cpu = ctrl_cpu + (worker_cpu * self.num_pods)
        required_mem = ctrl_mem + (worker_mem * self.num_pods)

        for quota in quotas:
            # Use spec.hard (set immediately on creation) rather than status.hard
            # (which may be empty until the quota controller first reconciles).
            hard = (quota.spec.hard or {}) if quota.spec else {}
            used = (quota.status.used or {}) if quota.status else {}
            violation = quota_violation(
                hard,
                used,
                required_cpu=required_cpu,
                required_mem=required_mem,
                required_pods=required_pods,
            )
            if violation:
                return CheckResult(
                    name="Resource Quotas",
                    status=CheckStatus.FAIL,
                    message=(
                        f"Benchmark would exceed {violation}. "
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
        except (TimeoutError, ApiException, aiohttp.ClientError, OSError) as e:
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

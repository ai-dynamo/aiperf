# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource-related pre-flight checks (nodes, quotas, memory estimation, tolerations)."""

from __future__ import annotations

import aiohttp
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.preflight import CheckResult, CheckStatus
from aiperf.kubernetes.utils import (
    format_cpu,
    format_memory,
    parse_cpu,
    parse_memory_gib,
)
from aiperf.operator import preflight as _pf
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


class _ResourceChecksMixin:
    """Checks related to cluster/node resources, quotas, memory, and tolerations."""

    async def _check_node_resources(self) -> CheckResult:
        """Check aggregate allocatable CPU/mem across schedulable Ready nodes.

        Nodes carrying NoSchedule/NoExecute taints that aren't tolerated by
        ``deploy_config.pod_template.tolerations`` are excluded from the
        capacity sum — otherwise a 50-node cluster of all-tainted GPU nodes
        would falsely report "sufficient resources" for a CPU-only workload.
        """
        from aiperf.kubernetes.environment import K8sEnvironment

        if skip := self._resource_mode_skip("Node Resources"):
            return skip

        try:
            node_list = await _pf.client.CoreV1Api(self.api).list_node()
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

        tolerations = self.deploy_config.pod_template.tolerations or []
        total_cpu = 0.0
        total_memory = 0.0
        ready_nodes = 0
        skipped_tainted = 0

        for node in nodes:
            if not _is_node_ready_typed(node):
                continue
            if not _node_is_schedulable_for(node, tolerations):
                skipped_tainted += 1
                continue
            ready_nodes += 1
            alloc = (node.status.allocatable or {}) if node.status else {}
            total_cpu += parse_cpu(alloc.get("cpu", "0"))
            total_memory += parse_memory_gib(alloc.get("memory", "0"))

        ctrl_cpu, ctrl_mem = controller_resource_requirements()
        worker_cpu = parse_cpu(K8sEnvironment.WORKER_POD.CPU)
        worker_mem = parse_memory_gib(K8sEnvironment.WORKER_POD.MEMORY)
        required_cpu = ctrl_cpu + (worker_cpu * self.num_pods)
        required_mem = ctrl_mem + (worker_mem * self.num_pods)

        tainted_suffix = (
            f" ({skipped_tainted} tainted node(s) excluded)" if skipped_tainted else ""
        )

        if ready_nodes == 0:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message=(f"No Ready, schedulable nodes available{tainted_suffix}"),
                hints=[
                    "Add tolerations or untaint nodes",
                    "Check node Ready condition",
                ],
            )

        if required_cpu > total_cpu or required_mem > total_memory:
            return CheckResult(
                name="Node Resources",
                status=CheckStatus.WARN,
                message=(
                    f"Cluster may not have enough resources. "
                    f"Need {format_cpu(required_cpu)} CPU, {format_memory(required_mem)} mem "
                    f"but only {format_cpu(total_cpu)} CPU, {format_memory(total_memory)} mem "
                    f"available across {ready_nodes} schedulable node(s){tainted_suffix}."
                ),
                hints=["Reduce worker count or add cluster capacity"],
            )

        return CheckResult(
            name="Node Resources",
            status=CheckStatus.PASS,
            message=(
                f"Cluster has sufficient resources "
                f"({ready_nodes} schedulable nodes, {format_cpu(total_cpu)} CPU, "
                f"{format_memory(total_memory)} mem){tainted_suffix}"
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
            node_list = await _pf.client.CoreV1Api(self.api).list_node()
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
        """Check that at least one matching Ready node can fit the largest pod."""
        from aiperf.kubernetes.environment import K8sEnvironment

        if skip := self._resource_mode_skip("Per-Node Schedulability"):
            return skip

        try:
            node_list = await _pf.client.CoreV1Api(self.api).list_node()
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

        for node in nodes:
            if not _is_node_ready_typed(node):
                continue
            if not _node_is_schedulable_for(node, tolerations):
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
            quota_list = await _pf.client.CoreV1Api(
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
                        status=CheckStatus.FAIL,
                        message=(
                            f"Benchmark would exceed CPU quota: "
                            f"{format_cpu(total_needed)} needed vs {hard_cpu} limit. "
                            f"Request a quota increase or reduce worker count."
                        ),
                    )
            if hard_mem:
                total_needed = required_mem + parse_memory_gib(used_mem or "0")
                if total_needed > parse_memory_gib(hard_mem):
                    return CheckResult(
                        name="Resource Quotas",
                        status=CheckStatus.FAIL,
                        message=(
                            f"Benchmark would exceed memory quota: "
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
            node_list = await _pf.client.CoreV1Api(self.api).list_node()
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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D107 - revoke operator Deployment patch RBAC during DGD reconcile."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_REVOKED_STATE_WINDOW_S = 45.0


@dataclass(frozen=True)
class RBACTarget:
    """A reversible RBAC object that grants the Dynamo operator deployment patch."""

    kind: Literal["role", "clusterrole"]
    name: str
    namespace: str | None
    rules: list[dict[str, Any]]
    mutated_rules: list[dict[str, Any]]

    @property
    def display_name(self) -> str:
        """Return a kubectl-addressable target name for skip/failure messages."""
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def test_d107_operator_rbac_revoked_mid_reconcile(
    kubectl: KubectlClient,
    dynamo_operator: None,
    dynamo_deployment_namespace: str,
) -> None:
    """Revoke deployments/patch and assert DGD status does not falsely succeed."""
    target = await _find_reversible_deployment_patch_target(kubectl)
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=dynamo_deployment_namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    manifest = deployer.generate_manifest()
    name = deployer._deployment_name()
    namespace = dynamo_deployment_namespace
    rbac_revoked = False

    try:
        try:
            await _revoke_deployment_patch(kubectl, target)
            rbac_revoked = True
            logger.info(f"D107: revoked deployments/patch from {target.display_name}")

            await kubectl.apply(manifest, namespace=namespace)
            observed = await _observe_dgd_while_rbac_revoked(
                kubectl,
                name,
                namespace,
                timeout_s=_REVOKED_STATE_WINDOW_S,
            )
            assert observed != "successful", (
                f"DGD {name} reported status.state='successful' while "
                f"deployments/patch was revoked from {target.display_name}"
            )
        finally:
            if rbac_revoked:
                await _restore_deployment_patch(kubectl, target)
                logger.info(f"D107: restored original rules on {target.display_name}")

        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
    finally:
        await kubectl.run(
            "delete",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _find_reversible_deployment_patch_target(
    kubectl: KubectlClient,
) -> RBACTarget:
    service_account = await _operator_service_account(kubectl)
    candidates = await _operator_bound_targets(kubectl, service_account)
    inspected = [candidate.display_name for candidate in candidates]

    reversible = [
        target for target in candidates if _has_explicit_deployment_patch(target)
    ]
    if len(reversible) != 1:
        pytest.skip(
            "D107 requires exactly one reversible operator RBAC target granting "
            f"deployments/patch to {service_account}; inspected targets: "
            f"{', '.join(inspected) if inspected else '<none>'}"
        )

    target = reversible[0]
    mutated_rules = _without_deployment_patch(target.rules)
    if mutated_rules == target.rules:
        pytest.skip(
            f"D107 found {target.display_name} but could not build reversible "
            "rules without deployments/patch"
        )
    return RBACTarget(
        kind=target.kind,
        name=target.name,
        namespace=target.namespace,
        rules=target.rules,
        mutated_rules=mutated_rules,
    )


async def _operator_service_account(kubectl: KubectlClient) -> str:
    result = await kubectl.run(
        "get",
        "deployment",
        "-n",
        _OPERATOR_NAMESPACE,
        "-l",
        _OPERATOR_SELECTOR,
        "-o",
        "json",
        check=True,
    )
    deployments = orjson.loads(result.stdout).get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "D107 requires exactly one Dynamo operator deployment; inspected deployments: "
            f"{', '.join(names) if names else '<none>'}"
        )
    return deployments[0]["spec"]["template"]["spec"].get(
        "serviceAccountName",
        "default",
    )


async def _operator_bound_targets(
    kubectl: KubectlClient,
    service_account: str,
) -> list[RBACTarget]:
    role_bindings = await _bound_role_refs(
        kubectl,
        "rolebinding",
        service_account,
        namespaced=True,
    )
    cluster_role_bindings = await _bound_role_refs(
        kubectl,
        "clusterrolebinding",
        service_account,
        namespaced=False,
    )
    targets: list[RBACTarget] = []
    for kind, name, namespace in role_bindings + cluster_role_bindings:
        target = await _load_rbac_target(kubectl, kind, name, namespace)
        if target is not None:
            targets.append(target)
    return targets


async def _bound_role_refs(
    kubectl: KubectlClient,
    binding_kind: Literal["rolebinding", "clusterrolebinding"],
    service_account: str,
    *,
    namespaced: bool,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    args = ["get", binding_kind]
    if namespaced:
        args.extend(["-n", _OPERATOR_NAMESPACE])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=True)
    bindings = orjson.loads(result.stdout).get("items", [])

    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding in bindings:
        subjects = binding.get("subjects", [])
        if not _has_operator_subject(subjects, service_account):
            continue
        role_ref = binding.get("roleRef", {})
        ref_kind = role_ref.get("kind", "").lower()
        if ref_kind not in {"role", "clusterrole"}:
            continue
        namespace = (
            binding.get("metadata", {}).get("namespace") if ref_kind == "role" else None
        )
        refs.append((ref_kind, role_ref["name"], namespace))
    return refs


async def _load_rbac_target(
    kubectl: KubectlClient,
    kind: Literal["role", "clusterrole"],
    name: str,
    namespace: str | None,
) -> RBACTarget | None:
    args = ["get", kind, name]
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return None
    obj = orjson.loads(result.stdout)
    return RBACTarget(
        kind=kind,
        name=name,
        namespace=namespace,
        rules=obj.get("rules", []),
        mutated_rules=obj.get("rules", []),
    )


def _has_operator_subject(subjects: list[dict[str, Any]], service_account: str) -> bool:
    for subject in subjects:
        if (
            subject.get("kind") == "ServiceAccount"
            and subject.get("name") == service_account
            and subject.get("namespace") == _OPERATOR_NAMESPACE
        ):
            return True
    return False


def _has_explicit_deployment_patch(target: RBACTarget) -> bool:
    return any(_rule_grants_explicit_deployment_patch(rule) for rule in target.rules)


def _rule_grants_explicit_deployment_patch(rule: dict[str, Any]) -> bool:
    api_groups = set(rule.get("apiGroups", []))
    resources = set(rule.get("resources", []))
    verbs = set(rule.get("verbs", []))
    return (
        "apps" in api_groups
        and "deployments" in resources
        and "patch" in verbs
        and "*" not in api_groups
        and "*" not in resources
        and "*" not in verbs
    )


def _without_deployment_patch(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mutated: list[dict[str, Any]] = []
    for rule in rules:
        copied = dict(rule)
        if _rule_grants_explicit_deployment_patch(rule):
            copied["verbs"] = [
                verb for verb in rule.get("verbs", []) if verb != "patch"
            ]
        mutated.append(copied)
    return mutated


async def _revoke_deployment_patch(kubectl: KubectlClient, target: RBACTarget) -> None:
    await _patch_rules(kubectl, target, target.mutated_rules)


async def _restore_deployment_patch(kubectl: KubectlClient, target: RBACTarget) -> None:
    await _patch_rules(kubectl, target, target.rules)


async def _patch_rules(
    kubectl: KubectlClient,
    target: RBACTarget,
    rules: list[dict[str, Any]],
) -> None:
    patch = orjson.dumps({"rules": rules}).decode()
    args = ["patch", target.kind, target.name, "--type=merge", f"-p={patch}"]
    if target.namespace is not None:
        args.extend(["-n", target.namespace])
    await kubectl.run(*args, check=True)


async def _observe_dgd_while_rbac_revoked(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    timeout_s: float,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last_state: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            dgd = orjson.loads(result.stdout)
            last_state = dgd.get("status", {}).get("state")
            if last_state == "successful":
                return last_state
        await asyncio.sleep(2.0)
    return last_state

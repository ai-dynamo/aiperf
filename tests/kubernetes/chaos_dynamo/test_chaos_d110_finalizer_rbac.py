# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D110 - finalizer RBAC revoked during DGD delete."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_OBSERVE_REVOKED_S = 45.0


@dataclass(frozen=True, slots=True)
class _RbacOwner:
    scope: Literal["role", "clusterrole"]
    name: str
    namespace: str | None

    @property
    def label(self) -> str:
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


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
    deployments = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "Dynamo RBAC chaos requires exactly one operator deployment; found "
            f"{', '.join(names) if names else '<none>'}"
        )
    return str(
        deployments[0]["spec"]["template"]["spec"].get("serviceAccountName", "default")
    )


async def _find_unique_operator_rbac_owner(
    kubectl: KubectlClient,
    *,
    api_group: str,
    resource: str,
    verb: str,
    case_id: str,
) -> _RbacOwner:
    service_account = await _operator_service_account(kubectl)
    candidates: list[_RbacOwner] = []
    inspected: list[str] = []
    for scope, name, namespace in await _operator_bound_role_refs(
        kubectl, service_account
    ):
        inspected.append(f"{scope}/{namespace + '/' if namespace else ''}{name}")
        body = await _load_rbac(kubectl, scope, name, namespace)
        if body is None:
            continue
        if _has_exact_rule(body.get("rules") or [], api_group, resource, verb):
            candidates.append(_RbacOwner(scope=scope, name=name, namespace=namespace))
    if len(candidates) != 1:
        pytest.skip(
            f"{case_id} requires exactly one operator-bound RBAC rule granting "
            f"{verb!r} on {resource!r} apiGroup={api_group!r}; candidates="
            f"{', '.join(c.label for c in candidates) or '<none>'}; inspected="
            f"{', '.join(inspected) if inspected else '<none>'}"
        )
    return candidates[0]


async def _operator_bound_role_refs(
    kubectl: KubectlClient,
    service_account: str,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding_kind, namespaced in (
        ("rolebinding", True),
        ("clusterrolebinding", False),
    ):
        args = ["get", binding_kind, "-o", "json"]
        if namespaced:
            args.insert(2, "-n")
            args.insert(3, _OPERATOR_NAMESPACE)
        result = await kubectl.run(*args, check=True)
        for binding in orjson.loads(result.stdout or b"{}").get("items", []):
            if not _has_operator_subject(
                binding.get("subjects") or [], service_account
            ):
                continue
            role_ref = binding.get("roleRef") or {}
            scope = str(role_ref.get("kind", "")).lower()
            if scope not in {"role", "clusterrole"}:
                continue
            namespace = (
                binding.get("metadata", {}).get("namespace")
                if scope == "role"
                else None
            )
            refs.append((scope, str(role_ref.get("name", "")), namespace))
    return refs


def _has_operator_subject(subjects: list[dict[str, Any]], service_account: str) -> bool:
    return any(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace") == _OPERATOR_NAMESPACE
        for subject in subjects
    )


async def _load_rbac(
    kubectl: KubectlClient,
    scope: Literal["role", "clusterrole"],
    name: str,
    namespace: str | None,
) -> dict[str, Any] | None:
    args = ["get", scope, name]
    if namespace is not None:
        args.extend(["-n", namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return None
    return dict(orjson.loads(result.stdout or b"{}"))


def _has_exact_rule(
    rules: list[dict[str, Any]],
    api_group: str,
    resource: str,
    verb: str,
) -> bool:
    for rule in rules:
        if "*" in (rule.get("apiGroups") or []):
            continue
        if "*" in (rule.get("resources") or []):
            continue
        if "*" in (rule.get("verbs") or []):
            continue
        if (
            api_group in (rule.get("apiGroups") or [])
            and resource in (rule.get("resources") or [])
            and verb in (rule.get("verbs") or [])
        ):
            return True
    return False


def _rbac_target(owner: _RbacOwner) -> dict[str, str]:
    target = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace
    return target


async def _apply_fresh_dgd(kubectl: KubectlClient, namespace: str) -> tuple[str, str]:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()
    await kubectl.apply(deployer.generate_manifest(), namespace=namespace)
    return name, namespace


async def _observe_not_successful(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    *,
    case_id: str,
    timeout_s: float = _OBSERVE_REVOKED_S,
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
            body = orjson.loads(result.stdout or b"{}")
            last_state = body.get("status", {}).get("state")
            assert last_state != "successful", (
                f"{case_id}: DGD {namespace}/{name} reached successful while "
                "required operator RBAC was revoked"
            )
        await asyncio.sleep(2.0)
    return last_state


async def _delete_dgd(kubectl: KubectlClient, name: str, namespace: str) -> None:
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


async def test_d110_finalizer_rbac_revoked_during_delete(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    owner = await _find_unique_operator_rbac_owner(
        kubectl,
        api_group="nvidia.com",
        resource="dynamographdeployments/finalizers",
        verb="update",
        case_id="D110",
    )
    faults = request.getfixturevalue("faults")
    name, namespace = await _apply_fresh_dgd(kubectl, dynamo_deployment_namespace)
    try:
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
        before = await _get_dgd(kubectl, name, namespace)
        if not before.get("metadata", {}).get("finalizers"):
            pytest.skip(f"D110 requires {namespace}/{name} to carry a DGD finalizer")
        async with faults.inject(
            "cluster.rbac.revoke",
            target=_rbac_target(owner),
            api_group="nvidia.com",
            resource="dynamographdeployments/finalizers",
            verb="update",
        ):
            await _delete_dgd(kubectl, name, namespace)
            blocked = await _wait_for_deletion_timestamp(
                kubectl, name, namespace, timeout_s=30.0
            )
            assert blocked, (
                f"D110: DGD {namespace}/{name} was not observed stuck in Terminating "
                "while finalizer update RBAC was revoked"
            )
        await _wait_for_gone(kubectl, name, namespace, timeout_s=180.0)
    finally:
        await _delete_dgd(kubectl, name, namespace)


async def _get_dgd(kubectl: KubectlClient, name: str, namespace: str) -> dict[str, Any]:
    result = await kubectl.run(
        "get", "dynamographdeployment", name, "-n", namespace, "-o", "json", check=True
    )
    return dict(orjson.loads(result.stdout or b"{}"))


async def _wait_for_deletion_timestamp(
    kubectl: KubectlClient, name: str, namespace: str, *, timeout_s: float
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
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
            body = orjson.loads(result.stdout or b"{}")
            if body.get("metadata", {}).get("deletionTimestamp"):
                return True
        else:
            return False
        await asyncio.sleep(1.0)
    return False


async def _wait_for_gone(
    kubectl: KubectlClient, name: str, namespace: str, *, timeout_s: float
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        result = await kubectl.run(
            "get", "dynamographdeployment", name, "-n", namespace, check=False
        )
        if result.returncode != 0:
            return
        await asyncio.sleep(2.0)
    raise AssertionError(
        f"D110: DGD {namespace}/{name} still exists after RBAC restore"
    )

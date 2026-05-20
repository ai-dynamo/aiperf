# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared RBAC helper primitives for Dynamo chaos scenarios."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import orjson
import pytest

if TYPE_CHECKING:
    from tests.kubernetes.helpers.kubectl import KubectlClient


@dataclass(frozen=True, slots=True)
class RbacOwner:
    """RBAC role-like object that owns a reversible grant."""

    scope: Literal["role", "clusterrole"]
    name: str
    namespace: str | None

    @property
    def label(self) -> str:
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


def rbac_revoke_target(owner: RbacOwner) -> dict[str, str]:
    """Convert an owner to the ``cluster.rbac.revoke`` target shape."""
    target = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace
    return target


def rbac_rule_grants(
    rules: Iterable[dict[str, Any]],
    *,
    api_group: str,
    resource: str,
    verb: str,
    reject_wildcards: bool,
) -> bool:
    """Return whether any RBAC rule grants the requested resource verb."""
    for rule in rules:
        api_groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        verbs = rule.get("verbs") or []
        if reject_wildcards and ("*" in api_groups or "*" in resources or "*" in verbs):
            continue
        api_group_matches = api_group in api_groups or (
            not reject_wildcards and "*" in api_groups
        )
        resource_matches = resource in resources or (
            not reject_wildcards and "*" in resources
        )
        verb_matches = verb in verbs or (not reject_wildcards and "*" in verbs)
        if api_group_matches and resource_matches and verb_matches:
            return True
    return False


async def find_unique_operator_rbac_owner(
    kubectl: KubectlClient,
    *,
    api_group: str,
    resource: str,
    verb: str,
    case_id: str,
    operator_namespace: str = "dynamo-system",
    operator_selector: str = "app.kubernetes.io/name=dynamo-operator",
    reject_wildcards: bool = True,
) -> RbacOwner:
    """Find the single operator-bound role-like object granting an RBAC rule."""
    service_account = await _operator_service_account(
        kubectl,
        operator_namespace=operator_namespace,
        operator_selector=operator_selector,
    )
    candidates: list[RbacOwner] = []
    inspected: list[str] = []
    role_refs = await _operator_bound_role_refs(
        kubectl,
        service_account,
        operator_namespace=operator_namespace,
    )
    for scope, name, namespace in role_refs:
        inspected.append(f"{scope}/{namespace + '/' if namespace else ''}{name}")
        body = await _load_rbac(kubectl, scope, name, namespace)
        if body is None:
            continue
        if rbac_rule_grants(
            body.get("rules") or [],
            api_group=api_group,
            resource=resource,
            verb=verb,
            reject_wildcards=reject_wildcards,
        ):
            candidates.append(RbacOwner(scope=scope, name=name, namespace=namespace))
    if len(candidates) != 1:
        pytest.skip(
            f"{case_id} requires exactly one operator-bound RBAC rule granting "
            f"{verb!r} on {resource!r} apiGroup={api_group!r}; candidates="
            f"{', '.join(c.label for c in candidates) or '<none>'}; inspected="
            f"{', '.join(inspected) if inspected else '<none>'}"
        )
    return candidates[0]


async def _operator_service_account(
    kubectl: KubectlClient,
    *,
    operator_namespace: str,
    operator_selector: str,
) -> str:
    result = await kubectl.run(
        "get",
        "deployment",
        "-n",
        operator_namespace,
        "-l",
        operator_selector,
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


async def _operator_bound_role_refs(
    kubectl: KubectlClient,
    service_account: str,
    *,
    operator_namespace: str,
) -> list[tuple[Literal["role", "clusterrole"], str, str | None]]:
    refs: list[tuple[Literal["role", "clusterrole"], str, str | None]] = []
    for binding_kind, namespaced in (
        ("rolebinding", True),
        ("clusterrolebinding", False),
    ):
        args = ["get", binding_kind, "-o", "json"]
        if namespaced:
            args.insert(2, "-n")
            args.insert(3, operator_namespace)
        result = await kubectl.run(*args, check=True)
        for binding in orjson.loads(result.stdout or b"{}").get("items", []):
            if not _has_operator_subject(
                binding.get("subjects") or [],
                service_account,
                operator_namespace=operator_namespace,
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


def _has_operator_subject(
    subjects: list[dict[str, Any]],
    service_account: str,
    *,
    operator_namespace: str,
) -> bool:
    return any(
        subject.get("kind") == "ServiceAccount"
        and subject.get("name") == service_account
        and subject.get("namespace") == operator_namespace
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

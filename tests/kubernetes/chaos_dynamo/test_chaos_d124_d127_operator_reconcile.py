# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D124-D127 -- Dynamo operator reconcile and finalizer control-plane faults."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import orjson
import pytest

from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_CHILD_LABELS = (
    "nvidia.com/dynamo-graph-deployment-name",
    "nvidia.com/dynamographdeployment",
)
_SUCCESS_TIMEOUT_S = 600.0
_CHILD_RECREATE_TIMEOUT_S = 180.0
_D127_NAMESPACE = "d127-namespace-finalizer-delete"


@dataclass(frozen=True, slots=True)
class _DgdRef:
    """DynamoGraphDeployment identity used by reconcile assertions."""

    namespace: str
    name: str


@dataclass(frozen=True, slots=True)
class _ChildRef:
    """Owned child resource selected for a delete/recreate scenario."""

    kind: Literal["deployment", "service"]
    namespace: str
    name: str
    uid: str


async def test_d124_child_deployment_deleted_is_recreated_by_operator(
    kubectl: KubectlClient,
    dynamo_server,  # noqa: ANN001 - fixture provides a ready baseline DGD
    dynamo_deployment_namespace: str,
) -> None:
    """Delete one child Deployment and require the DGD controller to recreate it."""
    dgd = await _unique_dgd_in_namespace(
        kubectl, dynamo_deployment_namespace, case="D124"
    )
    child = await _unique_owned_child(kubectl, dgd, kind="deployment", case="D124")

    await kubectl.run(
        "delete",
        "deployment",
        child.name,
        "-n",
        child.namespace,
        "--wait=false",
        check=True,
    )
    recreated = await _wait_for_child_recreated(kubectl, child)
    await kubectl.run(
        "rollout",
        "status",
        "deployment",
        recreated.name,
        "-n",
        recreated.namespace,
        "--timeout=180s",
        check=True,
    )

    assert recreated.uid != child.uid, (
        f"D124: child Deployment {child.namespace}/{child.name} still has old "
        f"uid={child.uid!r} after delete; operator did not recreate it"
    )


async def test_d125_child_service_deleted_is_recreated_by_operator(
    kubectl: KubectlClient,
    dynamo_server,  # noqa: ANN001 - fixture provides a ready baseline DGD
    dynamo_deployment_namespace: str,
) -> None:
    """Delete one child Service and require the DGD controller to recreate it."""
    dgd = await _unique_dgd_in_namespace(
        kubectl, dynamo_deployment_namespace, case="D125"
    )
    child = await _unique_owned_child(kubectl, dgd, kind="service", case="D125")

    await kubectl.run(
        "delete",
        "service",
        child.name,
        "-n",
        child.namespace,
        "--wait=false",
        check=True,
    )
    recreated = await _wait_for_child_recreated(kubectl, child)

    assert recreated.uid != child.uid, (
        f"D125: child Service {child.namespace}/{child.name} still has old "
        f"uid={child.uid!r} after delete; operator did not recreate it"
    )


async def test_d126_operator_restart_with_stale_workqueue_converges(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_server,  # noqa: ANN001 - fixture provides a ready baseline DGD
    dynamo_deployment_namespace: str,
) -> None:
    """Queue repeated DGD metadata events, restart the operator, and assert convergence."""
    dgd = await _unique_dgd_in_namespace(
        kubectl, dynamo_deployment_namespace, case="D126"
    )
    before_children = await _owned_child_names(kubectl, dgd)
    if not any(before_children.values()):
        pytest.skip(
            f"D126 requires existing DGD-owned child resources for {dgd.namespace}/{dgd.name}"
        )

    for idx in range(8):
        await kubectl.run(
            "annotate",
            "dynamographdeployment",
            dgd.name,
            "-n",
            dgd.namespace,
            f"chaos.dynamo.nvidia.com/d126-workqueue-{idx}={idx}",
            "--overwrite",
            check=True,
        )

    async with faults.inject(
        "operator.kill",
        target={"selector": _OPERATOR_SELECTOR, "ns": _OPERATOR_NAMESPACE},
    ):
        pass

    await _wait_operator_available(kubectl, timeout=180.0)
    await wait_for_dgd_state(
        kubectl,
        dgd.name,
        dgd.namespace,
        "successful",
        timeout=_SUCCESS_TIMEOUT_S,
    )
    after_children = await _owned_child_names(kubectl, dgd)

    assert before_children == after_children, (
        "D126: operator restart after queued DGD events changed owned child set; "
        f"before={before_children!r}, after={after_children!r}"
    )


async def test_d127_namespace_delete_with_dgd_finalizer_completes(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures operator and CRD exist
) -> None:
    """Delete an isolated namespace while its DGD carries an operator finalizer."""
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=_D127_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()

    await kubectl.run(
        "delete",
        "namespace",
        _D127_NAMESPACE,
        "--wait=false",
        "--ignore-not-found",
        check=False,
    )
    await _wait_namespace_absent(kubectl, _D127_NAMESPACE, timeout=120.0)

    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=_D127_NAMESPACE)
        finalizers = await _wait_for_dgd_finalizers(
            kubectl,
            name=name,
            namespace=_D127_NAMESPACE,
            timeout=90.0,
        )
        if not finalizers:
            pytest.skip(
                "D127 requires the Dynamo operator to add a DGD finalizer before "
                "namespace deletion; none appeared within 90s"
            )

        await kubectl.run(
            "delete",
            "namespace",
            _D127_NAMESPACE,
            "--wait=false",
            check=True,
        )
        await _wait_namespace_absent(kubectl, _D127_NAMESPACE, timeout=240.0)
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _D127_NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _unique_dgd_in_namespace(
    kubectl: KubectlClient,
    namespace: str,
    *,
    case: str,
) -> _DgdRef:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"{case} requires list/get access to DynamoGraphDeployments in "
            f"namespace {namespace!r}; kubectl stderr={result.stderr.strip()!r}"
        )
    items = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(items) != 1:
        names = [item.get("metadata", {}).get("name", "<unnamed>") for item in items]
        pytest.skip(
            f"{case} requires exactly one DGD in namespace {namespace!r}; "
            f"found {names if names else '<none>'}"
        )
    return _DgdRef(namespace=namespace, name=items[0]["metadata"]["name"])


async def _unique_owned_child(
    kubectl: KubectlClient,
    dgd: _DgdRef,
    *,
    kind: Literal["deployment", "service"],
    case: str,
) -> _ChildRef:
    children = await _owned_children(kubectl, dgd, kind=kind)
    if not children:
        pytest.skip(
            f"{case} requires at least one DGD-owned child {kind} for "
            f"{dgd.namespace}/{dgd.name}; none were found by ownerReference or labels"
        )
    frontend = [child for child in children if child.name.endswith("-frontend")]
    candidates = frontend or children
    if len(candidates) != 1:
        names = [child.name for child in candidates]
        pytest.skip(
            f"{case} requires a unique child {kind} target; candidates={names!r}"
        )
    return candidates[0]


async def _owned_children(
    kubectl: KubectlClient,
    dgd: _DgdRef,
    *,
    kind: Literal["deployment", "service"],
) -> list[_ChildRef]:
    by_name: dict[str, _ChildRef] = {}
    for args in _child_list_arg_sets(kind, dgd.name):
        result = await kubectl.run(
            *args, "-n", dgd.namespace, "-o", "json", check=False
        )
        if result.returncode != 0 or not result.stdout.strip():
            continue
        for item in orjson.loads(result.stdout).get("items", []):
            metadata = item.get("metadata", {})
            name = metadata.get("name", "")
            uid = metadata.get("uid", "")
            if not name or not uid:
                continue
            if _owned_by_dgd(item, dgd.name) or _labeled_for_dgd(item, dgd.name):
                by_name[name] = _ChildRef(
                    kind=kind,
                    namespace=dgd.namespace,
                    name=name,
                    uid=uid,
                )
    return sorted(by_name.values(), key=lambda child: child.name)


def _child_list_arg_sets(kind: str, dgd_name: str) -> list[list[str]]:
    args = [["get", kind]]
    args.extend(["get", kind, "-l", f"{label}={dgd_name}"] for label in _CHILD_LABELS)
    return args


def _owned_by_dgd(item: dict[str, Any], dgd_name: str) -> bool:
    owners = item.get("metadata", {}).get("ownerReferences") or []
    return any(
        owner.get("kind") == "DynamoGraphDeployment" and owner.get("name") == dgd_name
        for owner in owners
        if isinstance(owner, dict)
    )


def _labeled_for_dgd(item: dict[str, Any], dgd_name: str) -> bool:
    labels = item.get("metadata", {}).get("labels") or {}
    return any(labels.get(label) == dgd_name for label in _CHILD_LABELS)


async def _wait_for_child_recreated(
    kubectl: KubectlClient,
    old: _ChildRef,
) -> _ChildRef:
    deadline = asyncio.get_running_loop().time() + _CHILD_RECREATE_TIMEOUT_S
    last_seen = "<not observed>"
    while True:
        result = await kubectl.run(
            "get",
            old.kind,
            old.name,
            "-n",
            old.namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            item = orjson.loads(result.stdout)
            uid = item.get("metadata", {}).get("uid", "")
            last_seen = uid or "<missing uid>"
            if uid and uid != old.uid:
                return _ChildRef(
                    kind=old.kind,
                    namespace=old.namespace,
                    name=old.name,
                    uid=uid,
                )
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"{old.kind} {old.namespace}/{old.name} was not recreated with a "
                f"new UID within {_CHILD_RECREATE_TIMEOUT_S}s after deletion; "
                f"old_uid={old.uid!r}, last_seen_uid={last_seen!r}"
            )
        await asyncio.sleep(2.0)


async def _owned_child_names(
    kubectl: KubectlClient, dgd: _DgdRef
) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for kind in ("deployment", "service"):
        children = await _owned_children(kubectl, dgd, kind=kind)
        names[kind] = [child.name for child in children]
    return names


async def _wait_operator_available(kubectl: KubectlClient, *, timeout: float) -> None:
    result = await kubectl.run(
        "wait",
        "-n",
        _OPERATOR_NAMESPACE,
        "-l",
        _OPERATOR_SELECTOR,
        "deployment",
        "--for=condition=Available",
        f"--timeout={int(timeout)}s",
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "D126: Dynamo operator deployment did not become Available after restart; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )


async def _wait_for_dgd_finalizers(
    kubectl: KubectlClient,
    *,
    name: str,
    namespace: str,
    timeout: float,
) -> list[str]:
    deadline = asyncio.get_running_loop().time() + timeout
    finalizers: list[str] = []
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
        if result.returncode == 0 and result.stdout.strip():
            body = orjson.loads(result.stdout)
            finalizers = list(body.get("metadata", {}).get("finalizers") or [])
            if finalizers:
                return finalizers
        await asyncio.sleep(1.0)
    return finalizers


async def _wait_namespace_absent(
    kubectl: KubectlClient,
    namespace: str,
    *,
    timeout: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_phase = "<unobserved>"
    while True:
        result = await kubectl.run(
            "get", "namespace", namespace, "-o", "json", check=False
        )
        if result.returncode != 0:
            return
        if result.stdout.strip():
            body = orjson.loads(result.stdout)
            last_phase = body.get("status", {}).get("phase", "<missing phase>")
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(
                f"D127: namespace {namespace!r} still exists after {timeout}s; "
                f"last_phase={last_phase!r}"
            )
        await asyncio.sleep(2.0)

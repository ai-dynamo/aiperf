# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D115 - operator kill during DGD delete cleans up orphans."""

from __future__ import annotations

import asyncio
from typing import Any

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_CHILD_KINDS = ("deployment", "service", "configmap", "role", "rolebinding")


async def test_d115_operator_kill_during_dgd_delete_cleans_orphans(
    faults: Any,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
) -> None:
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=dynamo_deployment_namespace,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()
    namespace = dynamo_deployment_namespace
    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=namespace)
        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=300.0)
        before = await _child_names(kubectl, namespace, name)
        if not any(before.values()):
            pytest.skip(
                f"D115 requires child resources labelled with DGD {namespace}/{name}; found none"
            )

        await kubectl.run(
            "delete",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "--wait=false",
            check=True,
        )
        async with faults.inject(
            "operator.kill",
            target={"selector": _OPERATOR_SELECTOR, "ns": _OPERATOR_NAMESPACE},
        ):
            pass

        await kubectl.run(
            "wait",
            "-n",
            _OPERATOR_NAMESPACE,
            "-l",
            _OPERATOR_SELECTOR,
            "deployment",
            "--for=condition=Available",
            "--timeout=90s",
            check=True,
        )
        await _wait_dgd_gone(kubectl, name, namespace, timeout_s=240.0)
        leftovers = await _child_names(kubectl, namespace, name)
        assert not any(leftovers.values()), (
            f"D115: child resources remained after operator-kill delete recovery: {leftovers!r}"
        )
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


async def _wait_dgd_gone(
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
        f"D115: DGD {namespace}/{name} still exists after delete recovery"
    )


async def _child_names(
    kubectl: KubectlClient, namespace: str, dgd_name: str
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for kind in _CHILD_KINDS:
        result = await kubectl.run(
            "get",
            kind,
            "-n",
            namespace,
            "-l",
            f"nvidia.com/dynamographdeployment={dgd_name}",
            "-o",
            "json",
            check=False,
        )
        if result.returncode != 0:
            out[kind] = []
            continue
        items = orjson.loads(result.stdout or b"{}").get("items", [])
        out[kind] = [str(item.get("metadata", {}).get("name", "")) for item in items]
    return out

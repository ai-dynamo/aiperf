# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D108 - apiserver pause during DynamoGraphDeployment reconcile."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from tests.kubernetes.chaos.toxiproxy import TOXIPROXY_APISERVER_PORT, ToxiproxyInjector
from tests.kubernetes.chaos_dynamo.conftest import (
    DYNAMO_TOXIPROXY_NAMESPACE,
    DYNAMO_TOXIPROXY_SERVICE,
    wait_for_dgd_state,
)
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_OPERATOR_NAMESPACE = "dynamo-system"
_OPERATOR_SELECTOR = "app.kubernetes.io/name=dynamo-operator"
_PROXY_NAME = "d108-apiserver"
_PROXY_HOST = (
    f"{DYNAMO_TOXIPROXY_SERVICE}.{DYNAMO_TOXIPROXY_NAMESPACE}.svc.cluster.local"
)


@dataclass(frozen=True, slots=True)
class _DeploymentEnv:
    name: str
    env: dict[str, str]


async def test_d108_apiserver_pause_during_reconcile_recovers(
    kubectl: KubectlClient,
    dynamo_toxiproxy: ToxiproxyInjector,
    dynamo_deployment_namespace: str,
) -> None:
    deployment_env = await _operator_deployment_env(kubectl)
    await dynamo_toxiproxy.reset()
    await dynamo_toxiproxy.add_proxy(
        name=_PROXY_NAME,
        listen=f"0.0.0.0:{TOXIPROXY_APISERVER_PORT}",
        upstream="kubernetes.default.svc:443",
    )

    name = ""
    namespace = dynamo_deployment_namespace
    routed = False
    try:
        await _route_operator_to_apiserver_proxy(kubectl, deployment_env.name)
        routed = await _wait_operator_available(kubectl, timeout_s=90.0)
        if not routed:
            await _restore_operator_env(kubectl, deployment_env)
            pytest.skip(
                "D108 requires the Dynamo operator to run with apiserver traffic "
                "routed through toxiproxy; patched deployment did not become "
                "Available, likely because the operator image lacks a TLS/SNI "
                "override for KUBERNETES_SERVICE_HOST=toxiproxy"
            )

        config = DynamoConfig(
            model_name="Qwen/Qwen3-0.6B",
            namespace=namespace,
            api_version="v1alpha1",
        )
        deployer = DynamoDeployer(kubectl, config)
        name = deployer._deployment_name()
        await kubectl.apply(deployer.generate_manifest(), namespace=namespace)

        await dynamo_toxiproxy.add_toxic(_PROXY_NAME, "timeout", {"timeout": 0})
        await asyncio.sleep(30.0)
        await dynamo_toxiproxy.remove_toxic(_PROXY_NAME, "timeout_downstream")

        await wait_for_dgd_state(kubectl, name, namespace, "successful", timeout=420.0)
    finally:
        if name:
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
        if routed:
            await _restore_operator_env(kubectl, deployment_env)
        await dynamo_toxiproxy.reset()


async def _operator_deployment_env(kubectl: KubectlClient) -> _DeploymentEnv:
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
    import orjson

    deployments = orjson.loads(result.stdout or b"{}").get("items", [])
    if len(deployments) != 1:
        names = [
            item.get("metadata", {}).get("name", "<unnamed>") for item in deployments
        ]
        pytest.skip(
            "D108 requires exactly one Dynamo operator deployment; found "
            f"{', '.join(names) if names else '<none>'}"
        )
    deployment = deployments[0]
    containers = (
        deployment.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    if not containers:
        pytest.skip("D108 requires the Dynamo operator deployment to have a container")
    env = {
        str(item.get("name")): str(item.get("value"))
        for item in containers[0].get("env", [])
        if item.get("name") and item.get("value") is not None
    }
    return _DeploymentEnv(name=deployment["metadata"]["name"], env=env)


async def _route_operator_to_apiserver_proxy(
    kubectl: KubectlClient, deployment_name: str
) -> None:
    await kubectl.run(
        "set",
        "env",
        f"deployment/{deployment_name}",
        f"KUBERNETES_SERVICE_HOST={_PROXY_HOST}",
        f"KUBERNETES_SERVICE_PORT={TOXIPROXY_APISERVER_PORT}",
        "-n",
        _OPERATOR_NAMESPACE,
        check=True,
    )


async def _restore_operator_env(
    kubectl: KubectlClient, deployment_env: _DeploymentEnv
) -> None:
    args = ["set", "env", f"deployment/{deployment_env.name}"]
    for key in ("KUBERNETES_SERVICE_HOST", "KUBERNETES_SERVICE_PORT"):
        if key in deployment_env.env:
            args.append(f"{key}={deployment_env.env[key]}")
        else:
            args.append(f"{key}-")
    args.extend(["-n", _OPERATOR_NAMESPACE])
    await kubectl.run(*args, check=False)
    await _wait_operator_available(kubectl, timeout_s=90.0)


async def _wait_operator_available(kubectl: KubectlClient, *, timeout_s: float) -> bool:
    result = await kubectl.run(
        "wait",
        "-n",
        _OPERATOR_NAMESPACE,
        "-l",
        _OPERATOR_SELECTOR,
        "deployment",
        "--for=condition=Available",
        f"--timeout={int(timeout_s)}s",
        check=False,
    )
    return result.returncode == 0

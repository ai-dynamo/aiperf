# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D116 -- spec patches made while the Dynamo operator is down converge."""

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
_NAMESPACE = "d116-patch-operator-down"
_SUCCESS_TIMEOUT_S = 600.0


async def test_d116_patch_while_operator_down_reconciles_latest_generation(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures operator and CRD exist
) -> None:
    """Scale the operator down, patch the DGD spec, restore, and assert catch-up."""
    deployment = await _single_operator_deployment(kubectl)
    config = DynamoConfig(
        model_name="Qwen/Qwen3-0.6B",
        namespace=_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()

    try:
        await kubectl.apply(deployer.generate_manifest(), namespace=_NAMESPACE)
        try:
            await wait_for_dgd_state(
                kubectl,
                name,
                _NAMESPACE,
                "successful",
                timeout=_SUCCESS_TIMEOUT_S,
            )
        except TimeoutError as exc:
            status = await _status_snapshot(kubectl, name)
            pytest.skip(
                "D116 requires a baseline DGD to reach state='successful' before "
                f"operator-down patching; status={status!r}; error={exc}"
            )

        try:
            await _scale_operator(kubectl, deployment, replicas=0)
            await _wait_operator_available(kubectl, available=False, timeout=90.0)

            patch = {"spec": {"services": {"Frontend": {"replicas": 0}}}}
            await kubectl.run(
                "patch",
                "dynamographdeployment",
                name,
                "-n",
                _NAMESPACE,
                "--type=merge",
                f"-p={orjson.dumps(patch).decode()}",
                check=True,
            )
            patched = await _read_dgd(kubectl, name)
            patched_generation = patched["metadata"]["generation"]
        finally:
            await _scale_operator(kubectl, deployment, replicas=deployment["replicas"])
            if deployment["replicas"] > 0:
                await _wait_operator_available(kubectl, available=True, timeout=180.0)

        await wait_for_dgd_state(
            kubectl,
            name,
            _NAMESPACE,
            "successful",
            timeout=_SUCCESS_TIMEOUT_S,
        )
        final_dgd = await _read_dgd(kubectl, name)
        assert final_dgd["metadata"]["generation"] == patched_generation
        assert final_dgd["status"].get("observedGeneration") == patched_generation, (
            "D116: operator restored after an offline patch but did not observe "
            f"latest generation {patched_generation}; status={final_dgd.get('status')!r}"
        )
        assert final_dgd["spec"]["services"]["Frontend"].get("replicas") == 0
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _single_operator_deployment(kubectl: KubectlClient) -> dict[str, Any]:
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
    items = orjson.loads(result.stdout).get("items", [])
    if len(items) != 1:
        names = [item.get("metadata", {}).get("name", "<unnamed>") for item in items]
        pytest.skip(
            "D116 requires exactly one Dynamo operator Deployment; found "
            f"{names if names else '<none>'}"
        )
    item = items[0]
    return {
        "name": item["metadata"]["name"],
        "replicas": int(item.get("spec", {}).get("replicas") or 1),
    }


async def _scale_operator(
    kubectl: KubectlClient,
    deployment: dict[str, Any],
    *,
    replicas: int,
) -> None:
    await kubectl.run(
        "scale",
        "deployment",
        deployment["name"],
        "-n",
        _OPERATOR_NAMESPACE,
        f"--replicas={replicas}",
        check=True,
    )


async def _wait_operator_available(
    kubectl: KubectlClient,
    *,
    available: bool,
    timeout: float,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await kubectl.run(
            "get",
            "deployment",
            "-n",
            _OPERATOR_NAMESPACE,
            "-l",
            _OPERATOR_SELECTOR,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            items = orjson.loads(result.stdout).get("items", [])
            ready = sum(
                int(item.get("status", {}).get("availableReplicas") or 0)
                for item in items
            )
            if (ready > 0) is available:
                return
        await asyncio.sleep(1.0)
    pytest.fail(f"D116: operator available={available} was not observed in {timeout}s")


async def _read_dgd(kubectl: KubectlClient, name: str) -> dict[str, Any]:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "-o",
        "json",
        check=True,
    )
    return orjson.loads(result.stdout)


async def _status_snapshot(kubectl: KubectlClient, name: str) -> str:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        _NAMESPACE,
        "-o",
        "jsonpath={.status}",
        check=False,
    )
    return result.stdout.strip() or result.stderr.strip()

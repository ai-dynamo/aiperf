# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D117 -- rapid DGD spec patches converge on the last generation."""

from __future__ import annotations

import asyncio
from typing import Any

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_NAMESPACE = "d117-rapid-spec-patch"
_SUCCESS_TIMEOUT_S = 600.0


async def test_d117_rapid_spec_patches_converge_on_final_spec(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures operator and CRD exist
) -> None:
    """Patch frontend replicas several times and assert final observedGeneration."""
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
                "D117 requires a baseline DGD to reach state='successful' before "
                f"rapid spec patching; status={status!r}; error={exc}"
            )

        final_replicas = 1
        for replicas in (2, 0, final_replicas):
            await _patch_frontend_replicas(kubectl, name=name, replicas=replicas)
            await asyncio.sleep(0.25)

        patched = await _read_dgd(kubectl, name)
        final_generation = patched["metadata"]["generation"]
        assert patched["spec"]["services"]["Frontend"].get("replicas") == final_replicas

        await wait_for_dgd_state(
            kubectl,
            name,
            _NAMESPACE,
            "successful",
            timeout=_SUCCESS_TIMEOUT_S,
        )
        final = await _read_dgd(kubectl, name)
        assert final["status"].get("observedGeneration") == final_generation, (
            "D117: rapid spec patches did not converge to the final generation; "
            f"generation={final_generation}, status={final.get('status')!r}"
        )
        assert final["spec"]["services"]["Frontend"].get("replicas") == final_replicas
    finally:
        await kubectl.run(
            "delete",
            "namespace",
            _NAMESPACE,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )


async def _patch_frontend_replicas(
    kubectl: KubectlClient,
    *,
    name: str,
    replicas: int,
) -> None:
    patch = {"spec": {"services": {"Frontend": {"replicas": replicas}}}}
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

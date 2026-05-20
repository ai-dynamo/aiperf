# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D123 -- DGD status conditions are fresh after a spec update."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_NAMESPACE = "d123-status-condition-freshness"
_SUCCESS_TIMEOUT_S = 600.0


async def test_d123_status_conditions_fresh_after_spec_patch(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures operator and CRD exist
) -> None:
    """Patch a successful DGD and assert status conditions match latest generation."""
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
                "D123 requires a baseline DGD to reach state='successful' before "
                f"condition freshness checks; status={status!r}; error={exc}"
            )

        await _patch_frontend_replicas(kubectl, name=name, replicas=0)
        patched = await _read_dgd(kubectl, name)
        patched_generation = patched["metadata"]["generation"]

        await wait_for_dgd_state(
            kubectl,
            name,
            _NAMESPACE,
            "successful",
            timeout=_SUCCESS_TIMEOUT_S,
        )
        final = await _read_dgd(kubectl, name)
        status = final.get("status", {})
        assert status.get("observedGeneration") == patched_generation, (
            "D123: status.observedGeneration is stale after spec patch; "
            f"generation={patched_generation}, status={status!r}"
        )

        conditions = status.get("conditions") or []
        if not conditions:
            pytest.skip(
                "D123 requires DynamoGraphDeployment status.conditions to be "
                "present; this operator exposes no conditions on a successful DGD"
            )
        stale = _stale_conditions(conditions, patched_generation)
        assert not stale, (
            "D123: status.conditions contain stale observedGeneration values after "
            f"generation {patched_generation}: {stale!r}"
        )
        future = _future_transition_conditions(conditions)
        assert not future, (
            f"D123: condition transition times are in the future: {future!r}"
        )
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


def _stale_conditions(
    conditions: list[dict[str, Any]],
    generation: int,
) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    for condition in conditions:
        observed = condition.get("observedGeneration")
        if observed is not None and observed != generation:
            stale.append(condition)
    return stale


def _future_transition_conditions(
    conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    now = datetime.now(tz=timezone.utc)
    future: list[dict[str, Any]] = []
    for condition in conditions:
        timestamp = condition.get("lastTransitionTime")
        if not timestamp:
            continue
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if parsed > now:
            future.append(condition)
    return future

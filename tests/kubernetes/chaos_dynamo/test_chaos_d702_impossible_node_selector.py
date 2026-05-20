# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D702 -- impossible nodeSelector surfaces through DGD status.

Applies a minimal DynamoGraphDeployment whose child pod can never schedule
because its pod template selects a node label that should not exist. The parent
CR must move to ``status.state=failed`` with an actionable scheduling reason.
"""

from __future__ import annotations

import asyncio

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_DGD_NAME = "d702-test"
_DGD_NAMESPACE = "d702-node-selector"
_IMPOSSIBLE_NODE_SELECTOR = {"aiperf.nvidia.com/d702-impossible-node": "true"}
_EVENT_TIMEOUT_S = 90.0
_FAILED_TIMEOUT_S = 120.0
_STATUS_TERMS = (
    "nodeselector",
    "node selector",
    "node affinity",
    "didn't match",
    "unschedulable",
    "failedscheduling",
)


async def test_d702_impossible_node_selector_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Unsatisfiable node selector -> child pod Pending -> DGD failed status."""
    await kubectl.delete_namespace(_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_DGD_NAMESPACE)
    try:
        await kubectl.apply(_manifest(), namespace=_DGD_NAMESPACE)
        scheduling_event = await _wait_for_event(kubectl, _DGD_NAMESPACE)
        assert scheduling_event, (
            f"D702: no unschedulable/nodeSelector event appeared in namespace "
            f"{_DGD_NAMESPACE!r} within {_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl, _DGD_NAME, _DGD_NAMESPACE, "failed", timeout=_FAILED_TIMEOUT_S
        )
        assert observed_state == "failed"

        status_text = await _read_dgd_status_text(kubectl, _DGD_NAMESPACE, _DGD_NAME)
        assert any(term in status_text.lower() for term in _STATUS_TERMS), (
            "D702: DGD failed status did not name the nodeSelector/scheduling cause. "
            f"status={status_text!r}; event={scheduling_event!r}"
        )
    finally:
        await kubectl.delete_namespace(_DGD_NAMESPACE, wait=False)


def _manifest() -> str:
    manifest = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _DGD_NAME, "namespace": _DGD_NAMESPACE},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "nodeSelector": _IMPOSSIBLE_NODE_SELECTOR,
                        "mainContainer": {"image": "busybox:1.36"},
                    },
                }
            }
        },
    }
    return orjson.dumps(manifest).decode()


async def _wait_for_event(kubectl: KubectlClient, namespace: str) -> str:
    deadline = asyncio.get_event_loop().time() + _EVENT_TIMEOUT_S
    while asyncio.get_event_loop().time() < deadline:
        events = await _read_events(kubectl, namespace)
        if any(term in events.lower() for term in _STATUS_TERMS):
            return events
        await asyncio.sleep(1.0)
    return ""


async def _read_events(kubectl: KubectlClient, namespace: str) -> str:
    result = await kubectl.run(
        "get", "events", "-n", namespace, "-o", "json", check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    data = orjson.loads(result.stdout)
    return "\n".join(
        f"{item.get('reason', '')}: {item.get('message', '')}"
        for item in data.get("items", [])
    )


async def _read_dgd_status_text(
    kubectl: KubectlClient, namespace: str, name: str
) -> str:
    result = await kubectl.run(
        "get",
        "dynamographdeployment",
        name,
        "-n",
        namespace,
        "-o",
        "jsonpath={.status}",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""

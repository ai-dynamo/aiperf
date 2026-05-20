# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D709 -- missing PVC reference blocks cleanly and surfaces in DGD status."""

from __future__ import annotations

import asyncio

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_DGD_NAME = "d709-test"
_DGD_NAMESPACE = "d709-missing-pvc"
_MISSING_PVC = "d709-model-cache-missing"
_EVENT_TIMEOUT_S = 90.0
_FAILED_TIMEOUT_S = 120.0
_STATUS_TERMS = (
    "persistentvolumeclaim",
    "pvc",
    _MISSING_PVC,
    "not found",
    "failedmount",
    "failedscheduling",
    "unbound",
)


async def test_d709_missing_pvc_reference_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Child pod references absent PVC and parent status names the storage cause."""
    await kubectl.delete_namespace(_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_DGD_NAMESPACE)
    try:
        await kubectl.apply(_manifest(), namespace=_DGD_NAMESPACE)
        event_text = await _wait_for_event(kubectl, _DGD_NAMESPACE)
        assert event_text, f"D709: no PVC event appeared within {_EVENT_TIMEOUT_S}s"

        observed_state = await wait_for_dgd_state(
            kubectl, _DGD_NAME, _DGD_NAMESPACE, "failed", timeout=_FAILED_TIMEOUT_S
        )
        assert observed_state == "failed"

        status_text = await _read_status(kubectl, _DGD_NAMESPACE, _DGD_NAME)
        assert any(term in status_text.lower() for term in _STATUS_TERMS), (
            "D709: DGD failed status did not name the missing PVC. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_DGD_NAMESPACE, wait=False)


def _manifest() -> str:
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": _DGD_NAME, "namespace": _DGD_NAMESPACE},
        "spec": {
            "components": [
                {
                    "name": "Frontend",
                    "type": "frontend",
                    "replicas": 1,
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": "busybox:1.36",
                                    "volumeMounts": [
                                        {"name": "model-cache", "mountPath": "/models"}
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "model-cache",
                                    "persistentVolumeClaim": {
                                        "claimName": _MISSING_PVC
                                    },
                                }
                            ],
                        }
                    },
                }
            ]
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


async def _read_status(kubectl: KubectlClient, namespace: str, name: str) -> str:
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

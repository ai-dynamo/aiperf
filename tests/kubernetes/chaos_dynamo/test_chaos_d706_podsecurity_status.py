# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D706 -- PodSecurity admission rejection surfaces in DGD status."""

from __future__ import annotations

import asyncio

import orjson
import pytest

from tests.kubernetes.chaos_dynamo.conftest import wait_for_dgd_state
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_DGD_NAME = "d706-test"
_DGD_NAMESPACE = "d706-podsecurity"
_EVENT_TIMEOUT_S = 90.0
_FAILED_TIMEOUT_S = 120.0
_STATUS_TERMS = (
    "podsecurity",
    "pod security",
    "restricted",
    "privileged",
    "hostnetwork",
    "forbidden",
    "failedcreate",
)


async def test_d706_podsecurity_rejection_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Restricted namespace rejects unsafe pod spec and DGD reports the cause."""
    await kubectl.delete_namespace(_DGD_NAMESPACE, wait=True)
    await kubectl.apply(_namespace_manifest())
    try:
        await kubectl.apply(_dgd_manifest(), namespace=_DGD_NAMESPACE)

        event_text = await _wait_for_event(kubectl, _DGD_NAMESPACE)
        assert event_text, (
            f"D706: no PodSecurity admission event appeared within {_EVENT_TIMEOUT_S}s"
        )

        observed_state = await wait_for_dgd_state(
            kubectl, _DGD_NAME, _DGD_NAMESPACE, "failed", timeout=_FAILED_TIMEOUT_S
        )
        assert observed_state == "failed"

        status_text = await _read_status(kubectl, _DGD_NAMESPACE, _DGD_NAME)
        assert any(term in status_text.lower() for term in _STATUS_TERMS), (
            "D706: DGD failed status did not name the PodSecurity cause. "
            f"status={status_text!r}; event={event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_DGD_NAMESPACE, wait=False)


def _namespace_manifest() -> str:
    manifest = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": _DGD_NAMESPACE,
            "labels": {
                "pod-security.kubernetes.io/enforce": "restricted",
                "pod-security.kubernetes.io/enforce-version": "latest",
            },
        },
    }
    return orjson.dumps(manifest).decode()


def _dgd_manifest() -> str:
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
                        "hostNetwork": True,
                        "mainContainer": {
                            "image": "busybox:1.36",
                            "securityContext": {
                                "privileged": True,
                                "runAsUser": 0,
                            },
                        },
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

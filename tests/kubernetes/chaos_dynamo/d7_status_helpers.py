# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared D7 DynamoGraphDeployment status and manifest helpers."""

from __future__ import annotations

import asyncio
from typing import Any

import orjson

from tests.kubernetes.helpers.kubectl import KubectlClient


def dgd_state_from_status_text(status_text: str) -> str:
    """Extract ``status.state`` from a DGD status JSON payload."""
    if not status_text:
        return ""
    try:
        status = orjson.loads(status_text)
    except orjson.JSONDecodeError:
        return ""
    state = status.get("state")
    return state if isinstance(state, str) else ""


def mentions_any(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether ``text`` contains any needle, case-insensitively."""
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def minimal_v1alpha1_frontend_dgd_manifest(
    name: str,
    namespace: str,
    *,
    extra_pod_spec: dict[str, Any] | None = None,
) -> str:
    """Build a minimal v1alpha1 frontend-only DynamoGraphDeployment manifest."""
    manifest = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": extra_pod_spec
                    or {"mainContainer": {"image": "busybox:1.36"}},
                }
            }
        },
    }
    return orjson.dumps(manifest).decode()


async def read_dgd_status_text(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
) -> str:
    """Return the DGD ``status`` block as JSON text, or ``""`` on read failure."""
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


async def read_namespace_events_text(kubectl: KubectlClient, *, namespace: str) -> str:
    """Return namespace event reason/message text for D7 failure assertions."""
    result = await kubectl.run(
        "get",
        "events",
        "-n",
        namespace,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    try:
        data = orjson.loads(result.stdout)
    except orjson.JSONDecodeError:
        return result.stdout.strip()
    return "\n".join(
        f"{item.get('reason', '')}: {item.get('message', '')}"
        for item in data.get("items", [])
    )


async def wait_for_dgd_failed_status(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
    timeout_s: float,
    poll_interval_s: float = 2.0,
) -> tuple[str, str]:
    """Poll DGD status until ``state=failed`` or timeout, returning last status."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    observed_state = ""
    observed_status = ""
    while True:
        observed_status = await read_dgd_status_text(
            kubectl,
            namespace=namespace,
            name=name,
        )
        observed_state = dgd_state_from_status_text(observed_status)
        if observed_state == "failed" or asyncio.get_event_loop().time() >= deadline:
            return observed_state, observed_status
        await asyncio.sleep(poll_interval_s)


async def wait_for_events_or_status(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
    needles: tuple[str, ...],
    timeout_s: float,
    poll_interval_s: float = 2.0,
) -> str:
    """Poll DGD status plus namespace events until any needle is visible."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    combined = ""
    while asyncio.get_event_loop().time() < deadline:
        status = await read_dgd_status_text(kubectl, namespace=namespace, name=name)
        events = await read_namespace_events_text(kubectl, namespace=namespace)
        combined = f"{status}\n{events}"
        if mentions_any(combined, needles):
            return combined
        await asyncio.sleep(poll_interval_s)
    return combined

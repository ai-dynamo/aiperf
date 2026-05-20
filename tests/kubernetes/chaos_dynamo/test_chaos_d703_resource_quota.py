# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D703 -- namespace ResourceQuota exhaustion surfaces in DGD status.

Validates that a quota admission failure below the worker pod needs moves from
ReplicaSet events into the parent ``DynamoGraphDeployment.status`` instead of
leaving the CR opaquely pending.
"""

from __future__ import annotations

import asyncio

import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.gpu.dynamo.helpers import DynamoConfig, DynamoDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


_DGD_NAMESPACE = "d703-resource-quota"
_QUOTA_NAME = "d703-too-small"
_DGD_FAILED_TIMEOUT_S = 120.0
_QUOTA_EVENT_TIMEOUT_S = 90.0
_STATUS_CAUSE_TERMS = (
    "quota",
    "resourcequota",
    "exceeded",
    "insufficient",
    "failedscheduling",
    "unschedulable",
    "didn't match pod anti-affinity",
)


async def test_d703_resource_quota_exhaustion_surfaces_failed_status(
    kubectl: KubectlClient,
    dynamo_operator,  # noqa: ANN001 - fixture ensures the DGD operator is installed
) -> None:
    """Tiny namespace ResourceQuota -> child pods rejected -> DGD ``state=failed``.

    The namespace quota allows zero pods and only 1Mi memory request/limit, which
    is below even the smallest worker pod requirement. The test first proves the
    apiserver or scheduler observed the quota/scheduling failure, then requires
    the parent DGD status to fail with an actionable quota/scheduling reason.
    """
    config = DynamoConfig.single_gpu_disagg(
        namespace=_DGD_NAMESPACE,
        api_version="v1alpha1",
    )
    deployer = DynamoDeployer(kubectl, config)
    name = deployer._deployment_name()

    await kubectl.delete_namespace(_DGD_NAMESPACE, wait=True)
    await kubectl.create_namespace(_DGD_NAMESPACE)
    try:
        await kubectl.apply(_resource_quota_manifest(), namespace=_DGD_NAMESPACE)
        await kubectl.apply(deployer.generate_manifest())

        quota_event_text = await _wait_for_quota_or_scheduling_event(
            kubectl,
            namespace=_DGD_NAMESPACE,
            timeout=_QUOTA_EVENT_TIMEOUT_S,
        )
        assert quota_event_text, (
            f"D703: no ResourceQuota/scheduling event appeared in namespace "
            f"{_DGD_NAMESPACE!r} within {_QUOTA_EVENT_TIMEOUT_S}s after applying "
            f"DGD {name!r}; the quota may not be blocking worker pods"
        )

        observed_state, observed_status = await _wait_for_failed_dgd_status(
            kubectl,
            namespace=_DGD_NAMESPACE,
            name=name,
            timeout=_DGD_FAILED_TIMEOUT_S,
        )
        assert observed_state == "failed", (
            f"D703: quota failure was observed in namespace events but DGD {name!r} "
            f"did not reach state='failed' within {_DGD_FAILED_TIMEOUT_S}s; "
            f"observed state={observed_state!r}, status={observed_status!r}, "
            f"quota event={quota_event_text!r}"
        )

        status_lower = observed_status.lower()
        assert any(term in status_lower for term in _STATUS_CAUSE_TERMS), (
            "D703: DGD reached state='failed' but status did not name the "
            "quota/scheduling cause. "
            f"Observed status: {observed_status!r}; quota event: {quota_event_text!r}"
        )
    finally:
        await kubectl.delete_namespace(_DGD_NAMESPACE, wait=False)


def _resource_quota_manifest() -> str:
    """Return the deliberately-too-small ResourceQuota manifest."""
    quota = {
        "apiVersion": "v1",
        "kind": "ResourceQuota",
        "metadata": {"name": _QUOTA_NAME, "namespace": _DGD_NAMESPACE},
        "spec": {
            "hard": {
                "pods": "0",
                "requests.memory": "1Mi",
                "limits.memory": "1Mi",
            }
        },
    }
    return orjson.dumps(quota).decode()


async def _wait_for_quota_or_scheduling_event(
    kubectl: KubectlClient,
    *,
    namespace: str,
    timeout: float,
) -> str:
    """Poll namespace events until quota or scheduling rejection is visible."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        events_text = await _read_namespace_events(kubectl, namespace=namespace)
        events_lower = events_text.lower()
        if any(term in events_lower for term in _STATUS_CAUSE_TERMS):
            return events_text
        await asyncio.sleep(1.0)
    return ""


async def _wait_for_failed_dgd_status(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
    timeout: float,
) -> tuple[str, str]:
    """Poll DGD status until ``state=failed`` or timeout, returning last status."""
    deadline = asyncio.get_event_loop().time() + timeout
    observed_state = "<unobserved>"
    observed_status = ""
    while True:
        observed_status = await _read_dgd_status_text(
            kubectl,
            namespace=namespace,
            name=name,
        )
        observed_state = _state_from_status_text(observed_status)
        if observed_state == "failed" or asyncio.get_event_loop().time() >= deadline:
            return observed_state, observed_status
        await asyncio.sleep(2.0)


def _state_from_status_text(status_text: str) -> str:
    """Extract ``status.state`` from a JSON status payload."""
    if not status_text:
        return "<empty>"
    try:
        status = orjson.loads(status_text)
    except orjson.JSONDecodeError as exc:
        logger.debug(lambda exc=exc: f"D703 DGD status parse failed: {exc!r}")
        return "<unparsable>"
    state = status.get("state")
    return state if isinstance(state, str) and state else "<empty>"


async def _read_dgd_status_text(
    kubectl: KubectlClient,
    *,
    namespace: str,
    name: str,
) -> str:
    """Return the DGD ``status`` block as JSON text for assertion messages."""
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
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


async def _read_namespace_events(
    kubectl: KubectlClient,
    *,
    namespace: str,
) -> str:
    """Return warning event reason/message text for the isolated namespace."""
    result = await kubectl.run(
        "get",
        "events",
        "-n",
        namespace,
        "--sort-by=.lastTimestamp",
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    try:
        data = orjson.loads(result.stdout)
    except orjson.JSONDecodeError as exc:
        logger.debug(lambda exc=exc: f"D703 events parse failed: {exc!r}")
        return result.stdout.strip()

    lines: list[str] = []
    for item in data.get("items", []):
        reason = item.get("reason", "")
        message = item.get("message", "")
        involved = item.get("involvedObject", {})
        ref = f"{involved.get('kind', '')}/{involved.get('name', '')}".strip("/")
        lines.append(f"{ref}: {reason}: {message}".strip())
    return "\n".join(lines)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D202 — scale the Dynamo Frontend Deployment 1→0→1 mid-traffic.

D-series catalog § D2xx scenario. This exercises the host-visible Frontend
recovery path: while host-side chat-completions traffic is running through the
pytest port-forward, scale the unique Frontend Deployment to zero replicas,
then back to one replica, and assert fresh requests recover within 30 seconds.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

import aiohttp
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


FRONTEND_COMPONENT_LABEL = "nvidia.com/dynamo-component-type"
"""Dynamo label key used to identify frontend pods and workloads."""

FRONTEND_COMPONENT_VALUE = "frontend"
"""Dynamo label value used to identify frontend pods and workloads."""

REQUEST_INTERVAL_SECONDS = 0.5
"""Delay between background traffic-loop requests."""

REQUEST_TIMEOUT_SECONDS = 5.0
"""Per-request timeout while the frontend is intentionally unavailable."""

RECOVERY_TIMEOUT_SECONDS = 30.0
"""Maximum time allowed for a post-scale-up request to succeed."""

SCALE_WAIT_TIMEOUT_SECONDS = 30.0
"""Maximum time allowed for Deployment replica counts to match each scale step."""


@dataclass(frozen=True, slots=True)
class FrontendDeployment:
    """Frontend Deployment selected for D202 scaling."""

    name: str
    replicas: int
    labels: dict[str, str]


async def test_d202_frontend_scale_zero_recovers_mid_traffic(
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
    dynamo_deployment_namespace: str,
) -> None:
    """D202 — scale the unique Frontend Deployment down to zero and back to one.

    The default Dynamo ``disagg-1gpu`` topology has a single Frontend
    Deployment. If the installed topology has zero or multiple frontend-like
    Deployments, the test self-skips with the observed candidates rather than
    guessing which workload to disrupt.
    """
    frontend = await _resolve_unique_frontend_deployment(
        kubectl, dynamo_deployment_namespace
    )
    if frontend.replicas != 1:
        pytest.skip(
            f"D202 requires frontend Deployment {frontend.name!r} to start at "
            f"1 replica; observed replicas={frontend.replicas}"
        )

    traffic = TrafficRecorder()
    stop_event = asyncio.Event()
    traffic_task = asyncio.create_task(
        _send_background_traffic(dynamo_endpoint_url, traffic, stop_event)
    )

    try:
        await _wait_for_successful_request(
            dynamo_endpoint_url,
            timeout=RECOVERY_TIMEOUT_SECONDS,
            reason="pre-scale warmup",
        )

        await _scale_deployment(kubectl, dynamo_deployment_namespace, frontend.name, 0)
        await _wait_for_available_replicas(
            kubectl,
            dynamo_deployment_namespace,
            frontend.name,
            expected_available=0,
            timeout=SCALE_WAIT_TIMEOUT_SECONDS,
        )

        await _scale_deployment(kubectl, dynamo_deployment_namespace, frontend.name, 1)
        await _wait_for_available_replicas(
            kubectl,
            dynamo_deployment_namespace,
            frontend.name,
            expected_available=1,
            timeout=SCALE_WAIT_TIMEOUT_SECONDS,
        )
        await _wait_for_successful_request(
            dynamo_endpoint_url,
            timeout=RECOVERY_TIMEOUT_SECONDS,
            reason="post-scale recovery",
        )

        assert traffic.total_requests > 0, (
            "D202: background traffic loop sent no requests"
        )
        assert traffic.successes > 0, (
            f"D202: background traffic loop never observed a successful request; "
            f"observed outcomes={traffic.outcomes!r}"
        )
    finally:
        stop_event.set()
        traffic_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await traffic_task
        await _scale_deployment(
            kubectl,
            dynamo_deployment_namespace,
            frontend.name,
            frontend.replicas,
        )
        await _wait_for_available_replicas(
            kubectl,
            dynamo_deployment_namespace,
            frontend.name,
            expected_available=frontend.replicas,
            timeout=SCALE_WAIT_TIMEOUT_SECONDS,
        )


class TrafficRecorder:
    """Small mutable traffic summary for the background request loop."""

    def __init__(self) -> None:
        self.total_requests = 0
        self.successes = 0
        self.outcomes: list[str] = []

    def record(self, outcome: str) -> None:
        """Record one host-side request outcome."""
        self.total_requests += 1
        if outcome == "200":
            self.successes += 1
        self.outcomes.append(outcome)


async def _resolve_unique_frontend_deployment(
    kubectl: KubectlClient,
    namespace: str,
) -> FrontendDeployment:
    """Return the unique frontend Deployment or skip with observed candidates."""
    data = await kubectl.get_json("deployments", namespace=namespace)
    items = data.get("items", []) if isinstance(data, dict) else []
    candidates = [_deployment_from_item(item) for item in items]
    label_matches = [
        candidate
        for candidate in candidates
        if candidate.labels.get(FRONTEND_COMPONENT_LABEL) == FRONTEND_COMPONENT_VALUE
    ]
    name_matches = [
        candidate for candidate in candidates if candidate.name.endswith("-frontend")
    ]
    matches = label_matches or name_matches
    if len(matches) != 1:
        observed = (
            ", ".join(
                f"{candidate.name}(replicas={candidate.replicas}, "
                f"component={candidate.labels.get(FRONTEND_COMPONENT_LABEL)!r})"
                for candidate in candidates
            )
            or "<none>"
        )
        pytest.skip(
            f"D202 requires exactly one frontend Deployment in namespace {namespace!r}; "
            f"observed candidates: {observed}"
        )
    return matches[0]


def _deployment_from_item(item: dict[str, Any]) -> FrontendDeployment:
    """Build a compact frontend candidate from a Kubernetes Deployment item."""
    metadata = item.get("metadata", {})
    spec = item.get("spec", {})
    labels = metadata.get("labels", {})
    if not isinstance(labels, dict):
        labels = {}
    return FrontendDeployment(
        name=str(metadata.get("name", "")),
        replicas=int(spec.get("replicas") or 0),
        labels={str(key): str(value) for key, value in labels.items()},
    )


async def _scale_deployment(
    kubectl: KubectlClient,
    namespace: str,
    deployment: str,
    replicas: int,
) -> None:
    """Scale a Deployment to ``replicas`` with an error that names D202 context."""
    result = await kubectl.run(
        "scale",
        "deployment",
        deployment,
        f"--replicas={replicas}",
        namespace=namespace,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"D202: failed to scale Deployment {deployment!r} in namespace "
            f"{namespace!r} to replicas={replicas}: {result.stderr.strip()}"
        )
    logger.info(
        f"D202: scaled frontend Deployment {deployment!r} in {namespace!r} "
        f"to replicas={replicas}"
    )


async def _wait_for_available_replicas(
    kubectl: KubectlClient,
    namespace: str,
    deployment: str,
    expected_available: int,
    timeout: float,
) -> None:
    """Wait until Deployment status.availableReplicas equals the expected count."""
    deadline = asyncio.get_running_loop().time() + timeout
    last_status: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        data = await kubectl.get_json("deployment", deployment, namespace=namespace)
        if not isinstance(data, dict):
            raise RuntimeError(
                f"D202: kubectl returned non-object Deployment payload for {deployment!r}: "
                f"{data!r}"
            )
        last_status = data.get("status", {})
        available = int(last_status.get("availableReplicas") or 0)
        if available == expected_available:
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(
        f"D202: Deployment {deployment!r} did not reach availableReplicas="
        f"{expected_available} within {timeout}s; last_status={last_status!r}"
    )


async def _wait_for_successful_request(
    dynamo_endpoint_url: str,
    timeout: float,
    reason: str,
) -> None:
    """Poll chat-completions until one request returns HTTP 200."""
    deadline = asyncio.get_running_loop().time() + timeout
    outcomes: list[str] = []
    async with aiohttp.ClientSession() as session:
        while asyncio.get_running_loop().time() < deadline:
            outcome = await _post_chat_completion(session, dynamo_endpoint_url)
            outcomes.append(outcome)
            if outcome == "200":
                return
            await asyncio.sleep(0.5)
    raise TimeoutError(
        f"D202: no successful chat-completions request within {timeout}s during "
        f"{reason}; outcomes={outcomes!r}"
    )


async def _send_background_traffic(
    dynamo_endpoint_url: str,
    traffic: TrafficRecorder,
    stop_event: asyncio.Event,
) -> None:
    """Send host-side chat-completions traffic until ``stop_event`` is set."""
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            outcome = await _post_chat_completion(session, dynamo_endpoint_url)
            traffic.record(outcome)
            await asyncio.sleep(REQUEST_INTERVAL_SECONDS)


async def _post_chat_completion(
    session: aiohttp.ClientSession,
    dynamo_endpoint_url: str,
) -> str:
    """POST one non-streaming chat-completions request and return an outcome token."""
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "max_tokens": 10,
    }
    try:
        async with session.post(
            dynamo_endpoint_url + "/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
        ) as response:
            await response.read()
            return str(response.status)
    except asyncio.TimeoutError:
        return "timeout"
    except aiohttp.ClientError as exc:
        return type(exc).__name__

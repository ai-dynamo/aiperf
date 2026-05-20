# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D403 -- Free-after-worker-removal drains router in-flight state.

D-series catalog reference: D4xx (worker-lifecycle faults).

Targets the stale-worker free-event path documented in
``lib/kv-router/src/sequences/multi_worker.rs:346-366``. When a decode worker
leaves the topology, a later replica-sync Free event for work that moved to a
surviving worker must not strand request state in the router. The externally
observable contract is that ``dynamo_component_inflight_requests`` returns to
baseline after traffic completes; if the current Dynamo build does not export
that metric, this test proves the endpoint still serves traffic and skips with
the missing metric names rather than pretending to validate opaque internals.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.conftest import scrape_frontend_metrics
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_DECODE_POD_SELECTOR = "nvidia.com/dynamo-sub-component-type=decode"
"""Label selector for Dynamo decode-role worker pods."""

_COMPLETED_METRIC = "dynamo_frontend_requests_total"
"""Frontend completion counter used only to prove traffic is flowing."""

_INFLIGHT_METRIC_CANDIDATES = (
    "dynamo_component_inflight_requests",
    "dynamo_kv_router_inflight_requests",
    "dynamo_router_inflight_requests",
)
"""Known names for the router/component in-flight gauge across Dynamo builds."""

_DRAIN_TIMEOUT_S = 45.0
"""Maximum wait for in-flight request state to return to baseline."""

_POLL_INTERVAL_S = 2.0
"""Metrics polling interval while waiting for router state to drain."""


async def test_d403_free_after_worker_removal_drains_inflight_state(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Delete a decode worker, complete traffic elsewhere, and assert drain.

    Steps:

    1. Prove the deployed frontend serves a baseline completion.
    2. Scrape metrics and choose the exported in-flight gauge, or skip with the
       missing metric names after step 1 has already proved traffic works.
    3. Delete one ready decode pod via ``pod.kill``.
    4. Wait for a ready decode pod different from the deleted pod, then send a
       second completion so the Free event is emitted after topology removal.
    5. Poll metrics until the chosen in-flight gauge returns to baseline.
    """
    await _post_completion(dynamo_endpoint_url, prompt="d403 baseline completion")
    metrics_before = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)
    inflight_metric = _select_inflight_metric(metrics_before)
    if inflight_metric is None:
        await _post_completion(
            dynamo_endpoint_url,
            prompt="d403 traffic proof before missing-metric skip",
        )
        pytest.skip(
            "d403: traffic still works, but no router in-flight metric is exported; "
            f"missing one of {', '.join(_INFLIGHT_METRIC_CANDIDATES)}"
        )

    deleted_pod = await _first_ready_decode_pod(kubectl, dynamo_deployment_namespace)
    if deleted_pod is None:
        pytest.skip(
            f"d403: no ready decode pod found in {dynamo_deployment_namespace!r} "
            f"matching {_DECODE_POD_SELECTOR!r}"
        )

    baseline_inflight = metrics_before.get(inflight_metric, 0.0)
    baseline_completed = metrics_before.get(_COMPLETED_METRIC, 0.0)
    logger.info(
        lambda: (
            f"d403: baseline {inflight_metric}={baseline_inflight}, "
            f"{_COMPLETED_METRIC}={baseline_completed}, deleting {deleted_pod}"
        )
    )

    async with faults.inject(
        "pod.kill",
        target={"ns": dynamo_deployment_namespace, "pod": deleted_pod},
    ):
        replacement_pod = await _wait_for_replacement_decode_pod(
            kubectl,
            dynamo_deployment_namespace,
            deleted_pod=deleted_pod,
        )
        logger.info(
            lambda: (
                f"d403: replacement decode pod {replacement_pod} ready after "
                f"removing {deleted_pod}"
            )
        )
        await _post_completion(
            dynamo_endpoint_url,
            prompt="d403 completion after decode worker removal",
        )

    metrics_after = await _wait_for_inflight_drain(
        kubectl,
        dynamo_deployment_namespace,
        metric_name=inflight_metric,
        baseline=baseline_inflight,
    )
    completed_delta = metrics_after.get(_COMPLETED_METRIC, 0.0) - baseline_completed
    assert completed_delta > 0, (
        f"d403: no completed frontend requests observed after worker removal "
        f"({baseline_completed=} after={metrics_after.get(_COMPLETED_METRIC, 0.0)})"
    )


async def _post_completion(endpoint_url: str, *, prompt: str) -> None:
    """Send one non-streaming chat completion and require a successful response."""
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 8,
        "temperature": 0.0,
    }
    timeout = aiohttp.ClientTimeout(total=60.0)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.text()
        assert resp.status < 500, (
            f"d403: completion failed with HTTP {resp.status}; body={body[:512]!r}"
        )
        assert resp.status < 400, (
            f"d403: completion rejected with HTTP {resp.status}; body={body[:512]!r}"
        )


def _select_inflight_metric(metrics: dict[str, Any]) -> str | None:
    """Return the first exported in-flight metric name known to this test."""
    for name in _INFLIGHT_METRIC_CANDIDATES:
        if name in metrics:
            return name
    return None


async def _first_ready_decode_pod(
    kubectl: KubectlClient,
    namespace: str,
    *,
    excluded_pod: str | None = None,
) -> str | None:
    """Return a ready decode pod name, excluding a just-deleted pod if provided."""
    pods = await _list_decode_pods(kubectl, namespace)
    for pod in pods:
        name = pod.get("metadata", {}).get("name")
        if not isinstance(name, str) or name == excluded_pod:
            continue
        if _pod_is_ready(pod):
            return name
    return None


async def _wait_for_replacement_decode_pod(
    kubectl: KubectlClient,
    namespace: str,
    *,
    deleted_pod: str,
    timeout_s: float = 120.0,
) -> str:
    """Wait until Kubernetes has a ready decode pod other than ``deleted_pod``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pod = await _first_ready_decode_pod(
            kubectl,
            namespace,
            excluded_pod=deleted_pod,
        )
        if pod is not None:
            return pod
        await asyncio.sleep(_POLL_INTERVAL_S)
    raise TimeoutError(
        f"d403: no replacement ready decode pod in {namespace!r} matching "
        f"{_DECODE_POD_SELECTOR!r} within {timeout_s}s after deleting {deleted_pod!r}"
    )


async def _list_decode_pods(
    kubectl: KubectlClient, namespace: str
) -> list[dict[str, Any]]:
    """List decode pods as raw Kubernetes objects."""
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        _DECODE_POD_SELECTOR,
        "-o",
        "json",
        check=True,
    )
    data = orjson.loads(result.stdout)
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _pod_is_ready(pod: dict[str, Any]) -> bool:
    """Return whether a raw Pod object is Running and all containers are ready."""
    metadata = pod.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("deletionTimestamp"):
        return False
    status = pod.get("status", {})
    if not isinstance(status, dict) or status.get("phase") != "Running":
        return False
    container_statuses = status.get("containerStatuses", [])
    if not isinstance(container_statuses, list) or not container_statuses:
        return False
    return all(
        isinstance(container, dict) and container.get("ready") is True
        for container in container_statuses
    )


async def _wait_for_inflight_drain(
    kubectl: KubectlClient,
    namespace: str,
    *,
    metric_name: str,
    baseline: float,
) -> dict[str, float]:
    """Poll frontend metrics until router in-flight state returns to baseline."""
    deadline = time.monotonic() + _DRAIN_TIMEOUT_S
    last_metrics: dict[str, float] = {}
    while time.monotonic() < deadline:
        last_metrics = await scrape_frontend_metrics(kubectl, namespace)
        current = last_metrics.get(metric_name)
        if current is None:
            pytest.skip(
                f"d403: {metric_name!r} disappeared after worker-removal traffic; "
                f"missing one of {', '.join(_INFLIGHT_METRIC_CANDIDATES)}"
            )
        if current <= baseline:
            return last_metrics
        logger.info(
            lambda current=current, baseline=baseline: (
                f"d403: waiting for {metric_name} to drain, "
                f"current={current}, baseline={baseline}"
            )
        )
        await asyncio.sleep(_POLL_INTERVAL_S)
    pytest.fail(
        f"d403: {metric_name} did not drain to baseline within {_DRAIN_TIMEOUT_S}s "
        f"after decode worker removal (baseline={baseline}, "
        f"last={last_metrics.get(metric_name)})"
    )

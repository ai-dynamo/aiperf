# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D402 -- rapid decode-worker topology churn does not wedge routing.

D-series catalog reference: D4xx (KV-router lifecycle faults).

Targets the rapid add/remove race around
``lib/kv-router/src/sequences/multi_worker.rs:307-343``: a decode worker is
force-deleted and Kubernetes recreates it while the frontend/router topology view
is changing. The frontend must serve again after the replacement is ready, and a
post-churn metrics scrape must not show permanent request failure.
"""

from __future__ import annotations

import asyncio
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
_COMPLETED_METRIC = "dynamo_frontend_requests_total"
_ERRORS_METRIC = "dynamo_frontend_requests_errors_total"
_REPLACEMENT_TIMEOUT_S = 120.0
_POST_CHURN_SERVE_TIMEOUT_S = 30.0


async def test_d402_rapid_decode_topology_churn(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Delete a decode worker, wait for replacement, and assert routing recovers."""
    decode_pod = await _select_ready_decode_pod(kubectl, dynamo_deployment_namespace)
    if decode_pod is None:
        pytest.skip(
            f"D402 requires a ready decode pod in namespace "
            f"{dynamo_deployment_namespace!r} matching selector "
            f"{_DECODE_POD_SELECTOR!r}"
        )

    metrics_before = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)
    await _send_chat_completion(dynamo_endpoint_url, prompt="D402 warmup request")

    async with faults.inject(
        "pod.kill",
        target={"ns": dynamo_deployment_namespace, "pod": decode_pod},
    ):
        logger.info(
            lambda pod=decode_pod, ns=dynamo_deployment_namespace: (
                f"D402: deleted decode pod {ns}/{pod}; waiting for replacement"
            )
        )
        replacement = await _wait_for_replacement_decode_pod(
            kubectl,
            dynamo_deployment_namespace,
            old_pod=decode_pod,
            timeout=_REPLACEMENT_TIMEOUT_S,
        )

    await _wait_until_frontend_serves(dynamo_endpoint_url, _POST_CHURN_SERVE_TIMEOUT_S)
    metrics_after = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)

    assert metrics_after, (
        "D402: frontend /metrics scrape returned no samples after churn"
    )
    completed_delta = _metric_delta(metrics_after, metrics_before, _COMPLETED_METRIC)
    errors_delta = _metric_delta(metrics_after, metrics_before, _ERRORS_METRIC)
    assert completed_delta > 0, (
        f"D402: frontend served client traffic but {_COMPLETED_METRIC!r} did not "
        f"increase after topology churn (replacement={replacement!r}, "
        f"before={metrics_before.get(_COMPLETED_METRIC, 0.0)}, "
        f"after={metrics_after.get(_COMPLETED_METRIC, 0.0)})"
    )
    assert errors_delta <= completed_delta, (
        f"D402: metrics show more request failures than completions after "
        f"topology churn (replacement={replacement!r}, "
        f"completed_delta={completed_delta}, errors_delta={errors_delta})"
    )


async def _select_ready_decode_pod(
    kubectl: KubectlClient,
    namespace: str,
) -> str | None:
    data = await _get_decode_pods_json(kubectl, namespace)
    for item in data.get("items", []):
        name = _pod_name(item)
        if name and _pod_is_ready(item):
            return name
    return None


async def _wait_for_replacement_decode_pod(
    kubectl: KubectlClient,
    namespace: str,
    *,
    old_pod: str,
    timeout: float,
) -> str:
    deadline = asyncio.get_running_loop().time() + timeout
    observed: list[str] = []
    while True:
        data = await _get_decode_pods_json(kubectl, namespace)
        observed = [
            name
            for item in data.get("items", [])
            if (name := _pod_name(item)) and name != old_pod and _pod_is_ready(item)
        ]
        if observed:
            replacement = observed[0]
            logger.info(
                lambda pod=replacement, ns=namespace: (
                    f"D402: replacement decode pod ready: {ns}/{pod}"
                )
            )
            return replacement
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"D402: no ready replacement decode pod appeared in namespace "
                f"{namespace!r} matching selector {_DECODE_POD_SELECTOR!r} within "
                f"{timeout}s after deleting {old_pod!r}; last observed "
                f"replacement candidates={observed!r}"
            )
        await asyncio.sleep(2.0)


async def _wait_until_frontend_serves(endpoint_url: str, timeout: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_error = "<not attempted>"
    while True:
        try:
            await _send_chat_completion(endpoint_url, prompt="D402 post-churn request")
            return
        except (aiohttp.ClientError, asyncio.TimeoutError, AssertionError) as exc:
            last_error = repr(exc)
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(
                f"D402: frontend did not serve a chat completion within {timeout}s "
                f"after decode topology churn; last_error={last_error}"
            )
        await asyncio.sleep(2.0)


async def _send_chat_completion(endpoint_url: str, *, prompt: str) -> None:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "stream": False,
        "temperature": 0.0,
    }
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20.0)) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.text()
        assert resp.status < 500, (
            f"D402: frontend returned {resp.status} for chat completion; "
            f"body={body[:512]!r}"
        )
        assert resp.status == 200, (
            f"D402: expected HTTP 200 from chat completion, got {resp.status}; "
            f"body={body[:512]!r}"
        )


async def _get_decode_pods_json(
    kubectl: KubectlClient,
    namespace: str,
) -> dict[str, Any]:
    result = await kubectl.run(
        "get",
        "pod",
        "-n",
        namespace,
        "-l",
        _DECODE_POD_SELECTOR,
        "-o",
        "json",
        check=True,
    )
    loaded = orjson.loads(result.stdout)
    if not isinstance(loaded, dict):
        raise TypeError(f"D402: expected pod list JSON object, got {type(loaded)!r}")
    return loaded


def _pod_name(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    name = metadata.get("name", "")
    return name if isinstance(name, str) else ""


def _pod_is_ready(item: dict[str, Any]) -> bool:
    status = item.get("status", {})
    if not isinstance(status, dict) or status.get("phase") != "Running":
        return False
    conditions = status.get("conditions", [])
    if not isinstance(conditions, list):
        return False
    return any(
        isinstance(condition, dict)
        and condition.get("type") == "Ready"
        and condition.get("status") == "True"
        for condition in conditions
    )


def _metric_delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return after.get(key, 0.0) - before.get(key, 0.0)

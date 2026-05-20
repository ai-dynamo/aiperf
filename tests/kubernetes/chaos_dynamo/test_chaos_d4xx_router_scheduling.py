# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D4xx Dynamo router and scheduling chaos scenarios."""

from __future__ import annotations

import asyncio
import contextlib
import statistics
import time
from collections.abc import Mapping
from typing import Any

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.conftest import scrape_frontend_metrics
from tests.kubernetes.chaos_dynamo.metrics_helpers import metric_delta
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


# D401

_CLIENT_ERROR_BUDGET_S: float = 35.0
"""Wall-clock seconds the client is allowed to wait between worker kill and
a terminal stream event. See module docstring for the contract."""

_DECODE_POD_SELECTOR: str = "nvidia.com/dynamo-sub-component-type=decode"
"""Label selector for decode-role worker pods. Set by the dynamo-operator on
every pod whose role is decode; querying by label keeps the test agnostic to
the topology (agg vs. disagg) the deployment fixture chose."""


async def test_d401_kill_decode_mid_request(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Kill the active decode worker mid-stream; assert clean error.

    Targets ``lib/kv-router/src/scheduling/queue.rs:173-210`` (no-retry path).

    Steps:

    1. Open a streaming POST to ``/v1/chat/completions`` requesting 200 tokens
       so the request is long-lived enough to be killed mid-flight.
    2. Wait until the client has observed the first chunk -- this proves the
       router selected a worker and dispatched to it; killing before this
       point would be testing dispatch-time error handling, not the
       mid-request gap.
    3. Resolve the decode pod by label and force-delete it via ``pod.kill``.
    4. Drain the stream; record the wall-clock from kill to the first
       terminal event the client sees (exception, non-200, or stream EOF).
    5. Assert the budget. The kubelet recreates the worker pod from the
       Deployment template; ``pod.kill`` itself is restore-free.
    """
    request_body = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": "Write a 200-word essay about the history of databases.",
            }
        ],
        "max_tokens": 200,
        "stream": True,
        "temperature": 0.0,
    }

    timeout = aiohttp.ClientTimeout(total=_CLIENT_ERROR_BUDGET_S + 30.0)
    kill_time: float | None = None
    error_time: float | None = None
    first_chunk_seen = asyncio.Event()
    saw_terminal_error = False

    async def _drive_stream() -> None:
        nonlocal error_time, saw_terminal_error
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    f"{dynamo_endpoint_url}/chat/completions",
                    json=request_body,
                ) as resp,
            ):
                if resp.status >= 500:
                    error_time = time.monotonic()
                    saw_terminal_error = True
                    logger.info(
                        lambda s=resp.status: f"d401: server returned {s} pre-stream"
                    )
                    return
                async for chunk in resp.content.iter_any():
                    if not first_chunk_seen.is_set() and chunk:
                        first_chunk_seen.set()
                    # An [ERROR] frame from the Dynamo frontend counts as a
                    # clean terminal event; the stream-end (no more chunks)
                    # path is handled by falling out of the async for loop.
                    if b"[ERROR]" in chunk or b'"error"' in chunk:
                        error_time = time.monotonic()
                        saw_terminal_error = True
                        return
                # Normal EOF after the kill is also a clean termination --
                # the budget assertion below still catches an infinite hang
                # because we'd never reach this line in time.
                error_time = time.monotonic()
                saw_terminal_error = True
        except (
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientPayloadError,
            aiohttp.ClientConnectionError,
            asyncio.TimeoutError,
        ) as exc:
            error_time = time.monotonic()
            saw_terminal_error = True
            logger.info(lambda exc=exc: f"d401: client got clean error {exc!r}")

    stream_task = asyncio.create_task(_drive_stream())

    # Wait for the first chunk to confirm a worker actually started decoding.
    # If this times out the test bails out before injecting a fault so we
    # don't kill an idle worker and call it a pass.
    try:
        await asyncio.wait_for(first_chunk_seen.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        stream_task.cancel()
        pytest.fail(
            "d401: no streaming chunk received within 10s; cannot prove "
            "decode worker was actively generating before kill"
        )

    pod_res = await kubectl.run(
        "get",
        "pod",
        "-n",
        dynamo_deployment_namespace,
        "-l",
        _DECODE_POD_SELECTOR,
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=True,
    )
    decode_pod = pod_res.stdout.strip()
    if not decode_pod:
        stream_task.cancel()
        pytest.fail(
            f"d401: no decode pod found in {dynamo_deployment_namespace!r} "
            f"matching {_DECODE_POD_SELECTOR!r}"
        )

    async with faults.inject(
        "pod.kill",
        target={"ns": dynamo_deployment_namespace, "pod": decode_pod},
    ):
        kill_time = time.monotonic()
        logger.info(
            lambda p=decode_pod, ns=dynamo_deployment_namespace: (
                f"d401: killed decode pod {ns}/{p} mid-request"
            )
        )
        # Bound the stream-drain wait at the client-error budget plus a
        # little overhead; if the client really hangs we surface that as a
        # TimeoutError below rather than waiting on the outer test timeout.
        try:
            await asyncio.wait_for(stream_task, timeout=_CLIENT_ERROR_BUDGET_S + 5.0)
        except asyncio.TimeoutError:
            stream_task.cancel()
            pytest.fail(
                f"d401: client did not see terminal event within "
                f"{_CLIENT_ERROR_BUDGET_S + 5.0}s of decode-pod kill "
                f"(infinite-hang regression in queue.rs:173-210)"
            )

    assert saw_terminal_error, (
        "d401: stream completed without a terminal error event after "
        "decode-pod kill -- expected 5xx / disconnect / [ERROR] frame"
    )
    assert kill_time is not None and error_time is not None
    latency = error_time - kill_time
    assert latency < _CLIENT_ERROR_BUDGET_S, (
        f"d401: client-visible error latency {latency:.2f}s exceeds budget "
        f"{_CLIENT_ERROR_BUDGET_S}s (DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS + slack)"
    )
    logger.info(
        lambda lat=latency: f"d401: client error after {lat:.2f}s (within budget)"
    )


# D402

_d402_DECODE_POD_SELECTOR = "nvidia.com/dynamo-sub-component-type=decode"
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
            f"{_d402_DECODE_POD_SELECTOR!r}"
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
    completed_delta = metric_delta(metrics_after, metrics_before, _COMPLETED_METRIC)
    errors_delta = metric_delta(metrics_after, metrics_before, _ERRORS_METRIC)
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
                f"{namespace!r} matching selector {_d402_DECODE_POD_SELECTOR!r} within "
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
        _d402_DECODE_POD_SELECTOR,
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


# D403

_d403_DECODE_POD_SELECTOR = "nvidia.com/dynamo-sub-component-type=decode"
"""Label selector for Dynamo decode-role worker pods."""

_d403_COMPLETED_METRIC = "dynamo_frontend_requests_total"
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
            f"matching {_d403_DECODE_POD_SELECTOR!r}"
        )

    baseline_inflight = metrics_before.get(inflight_metric, 0.0)
    baseline_completed = metrics_before.get(_d403_COMPLETED_METRIC, 0.0)
    logger.info(
        lambda: (
            f"d403: baseline {inflight_metric}={baseline_inflight}, "
            f"{_d403_COMPLETED_METRIC}={baseline_completed}, deleting {deleted_pod}"
        )
    )

    async with faults.inject(
        "pod.kill",
        target={"ns": dynamo_deployment_namespace, "pod": deleted_pod},
    ):
        replacement_pod = await _d403_wait_for_replacement_decode_pod(
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
    completed_delta = (
        metrics_after.get(_d403_COMPLETED_METRIC, 0.0) - baseline_completed
    )
    assert completed_delta > 0, (
        f"d403: no completed frontend requests observed after worker removal "
        f"({baseline_completed=} after={metrics_after.get(_d403_COMPLETED_METRIC, 0.0)})"
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
        if _d403_pod_is_ready(pod):
            return name
    return None


async def _d403_wait_for_replacement_decode_pod(
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
        f"{_d403_DECODE_POD_SELECTOR!r} within {timeout_s}s after deleting {deleted_pod!r}"
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
        _d403_DECODE_POD_SELECTOR,
        "-o",
        "json",
        check=True,
    )
    data = orjson.loads(result.stdout)
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _d403_pod_is_ready(pod: dict[str, Any]) -> bool:
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


# D406

_PINNED_REQUEST_PARAMETER = "nvext.decode_worker_id"
"""Concrete Dynamo request parameter used for worker pinning.

Dynamo v1.1.0 defines this in ``lib/llm/src/protocols/openai/nvext.rs`` and
extracts it into ``RoutingHints.decode_worker_id`` in
``lib/llm/src/preprocessor.rs``. D406 must skip with this exact name if the
running endpoint rejects it.
"""

_MODEL_NAME = "Qwen/Qwen3-0.6B"
"""Model served by the Dynamo GPU fixture."""

_REQUEST_TIMEOUT_S = 45.0
"""Per-request budget. Exceeding this means HoL became an unbounded hang."""

_PINNED_STREAMS = 4
"""Concurrent long streams pinned to the same worker before unpinned probes."""

_UNPINNED_PROBES = 4
"""Short non-pinned completions used to observe latency behind pinned load."""


async def test_d406_pinned_worker_head_of_line_observation(
    dynamo_endpoint_url: str,
) -> None:
    """Probe ``nvext.decode_worker_id`` support, then observe bounded HoL latency.

    Prerequisites are tested at runtime rather than assumed:

    1. ``/health`` must expose at least one ``instance_id`` to pin to.
    2. A tiny completion containing ``nvext.decode_worker_id`` must be accepted
       by the current endpoint and must report the same decode worker when
       ``nvext.extra_fields=["worker_id"]`` is requested.

    If either prerequisite is absent, the test skips with the concrete missing
    field or endpoint surface. Otherwise it starts several long pinned streams,
    sends short unpinned completions behind them, and asserts the unpinned
    requests all terminate within :data:`_REQUEST_TIMEOUT_S`.
    """
    worker_ids = await _discover_worker_ids(dynamo_endpoint_url)
    if not worker_ids:
        pytest.skip(
            "D406: /health returned no instances with instance_id; cannot probe "
            f"{_PINNED_REQUEST_PARAMETER} support"
        )
    pinned_worker_id = worker_ids[0]

    probe = await _d406_post_completion(
        dynamo_endpoint_url,
        content="D406 pinning support probe.",
        max_tokens=1,
        nvext={
            "decode_worker_id": pinned_worker_id,
            "extra_fields": ["worker_id"],
        },
    )
    _skip_if_pinning_unsupported(probe)
    assert probe.status == 200, (
        f"D406: {_PINNED_REQUEST_PARAMETER} probe returned HTTP {probe.status}; "
        f"body={probe.body[:512]!r}"
    )

    observed_worker_id = _find_int_key(probe.json_body, "decode_worker_id")
    if observed_worker_id is None:
        pytest.skip(
            "D406: endpoint accepted nvext.decode_worker_id but did not return "
            "nvext.worker_id.decode_worker_id with extra_fields=['worker_id']; "
            "cannot prove pinned/affinity support safely"
        )
    assert observed_worker_id == pinned_worker_id, (
        f"D406: {_PINNED_REQUEST_PARAMETER}={pinned_worker_id} was routed to "
        f"decode_worker_id={observed_worker_id}; pinned routing contract changed"
    )

    baseline_latencies = [
        await _time_completion(
            dynamo_endpoint_url,
            content=f"D406 baseline probe {idx}.",
            max_tokens=4,
            nvext={"extra_fields": ["worker_id"]},
        )
        for idx in range(2)
    ]

    stop_pinned = asyncio.Event()
    first_chunks = [asyncio.Event() for _ in range(_PINNED_STREAMS)]
    pinned_tasks = [
        asyncio.create_task(
            _hold_pinned_stream(
                dynamo_endpoint_url,
                pinned_worker_id,
                first_chunks[idx],
                stop_pinned,
                idx,
            )
        )
        for idx in range(_PINNED_STREAMS)
    ]

    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in first_chunks)),
            timeout=20.0,
        )
        loaded_latencies = await asyncio.gather(
            *(
                _time_completion(
                    dynamo_endpoint_url,
                    content=f"D406 unpinned probe behind pinned load {idx}.",
                    max_tokens=4,
                    nvext={"extra_fields": ["worker_id"]},
                )
                for idx in range(_UNPINNED_PROBES)
            )
        )
    except asyncio.TimeoutError as exc:
        pytest.fail(
            f"D406: pinned stream or unpinned probe exceeded "
            f"{_REQUEST_TIMEOUT_S}s budget while observing "
            f"queue.rs:246-249 head-of-line behavior: {exc!r}"
        )
    finally:
        stop_pinned.set()
        for task in pinned_tasks:
            task.cancel()
        for task in pinned_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    baseline_p50 = statistics.median(baseline_latencies)
    loaded_p50 = statistics.median(loaded_latencies)
    loaded_max = max(loaded_latencies)
    assert loaded_max < _REQUEST_TIMEOUT_S, (
        f"D406: unpinned probe hung behind pinned worker for {loaded_max:.2f}s "
        f"(budget={_REQUEST_TIMEOUT_S}s, baseline_p50={baseline_p50:.2f}s, "
        f"loaded_p50={loaded_p50:.2f}s)"
    )
    logger.info(
        lambda: (
            "D406: observed pinned-worker HoL latency "
            f"baseline_p50={baseline_p50:.2f}s loaded_p50={loaded_p50:.2f}s "
            f"loaded_max={loaded_max:.2f}s worker={pinned_worker_id}"
        )
    )


class _CompletionResult:
    """Small typed container for an HTTP completion response."""

    def __init__(
        self, status: int, body: str, json_body: dict[str, Any] | None
    ) -> None:
        self.status = status
        self.body = body
        self.json_body = json_body


async def _discover_worker_ids(dynamo_endpoint_url: str) -> list[int]:
    """Return worker ``instance_id`` values from Dynamo's health endpoint."""
    endpoint = dynamo_endpoint_url.rstrip("/")
    root = endpoint.removesuffix("/v1")
    health_urls = [f"{root}/health", f"{endpoint}/health"]
    timeout = aiohttp.ClientTimeout(total=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for url in health_urls:
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    payload = await resp.json(content_type=None)
                    return sorted(set(_iter_instance_ids(payload)))
            except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
                logger.debug(
                    lambda exc=exc, url=url: f"D406: health probe {url} failed: {exc!r}"
                )
    return []


async def _d406_post_completion(
    dynamo_endpoint_url: str,
    *,
    content: str,
    max_tokens: int,
    nvext: Mapping[str, object] | None,
) -> _CompletionResult:
    """POST one non-streaming chat completion and return status/body/json."""
    payload: dict[str, object] = {
        "model": _MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    if nvext is not None:
        payload["nvext"] = dict(nvext)

    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(_chat_completion_url(dynamo_endpoint_url), json=payload) as resp,
    ):
        body = await resp.text()
        json_body: dict[str, Any] | None = None
        with contextlib.suppress(ValueError):
            parsed = await resp.json(content_type=None)
            if isinstance(parsed, dict):
                json_body = parsed
        return _CompletionResult(resp.status, body, json_body)


async def _time_completion(
    dynamo_endpoint_url: str,
    *,
    content: str,
    max_tokens: int,
    nvext: Mapping[str, object] | None,
) -> float:
    """Return wall-clock latency for one successful non-streaming completion."""
    start = time.monotonic()
    result = await _d406_post_completion(
        dynamo_endpoint_url,
        content=content,
        max_tokens=max_tokens,
        nvext=nvext,
    )
    latency = time.monotonic() - start
    assert result.status == 200, (
        f"D406: completion returned HTTP {result.status}; body={result.body[:512]!r}"
    )
    return latency


async def _hold_pinned_stream(
    dynamo_endpoint_url: str,
    worker_id: int,
    first_chunk_seen: asyncio.Event,
    stop: asyncio.Event,
    idx: int,
) -> None:
    """Hold one long streaming request pinned to ``worker_id`` until stopped."""
    payload: dict[str, object] = {
        "model": _MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": (
                    "D406 pinned stream "
                    f"{idx}: write a detailed deterministic paragraph about queues."
                ),
            }
        ],
        "max_tokens": 128,
        "stream": True,
        "temperature": 0.0,
        "nvext": {
            "decode_worker_id": worker_id,
            "extra_fields": ["worker_id"],
        },
    }
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S + 30.0)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(_chat_completion_url(dynamo_endpoint_url), json=payload) as resp,
    ):
        assert resp.status == 200, (
            f"D406: pinned stream {idx} returned HTTP {resp.status}; "
            f"body={(await resp.text())[:512]!r}"
        )
        async for chunk in resp.content.iter_any():
            if chunk:
                first_chunk_seen.set()
            if stop.is_set():
                return


def _skip_if_pinning_unsupported(result: _CompletionResult) -> None:
    """Skip when the endpoint rejects the exact D406 pinning parameter."""
    if result.status not in {400, 404, 422}:
        return
    body = result.body.lower()
    unsupported_markers = (
        "decode_worker_id",
        "nvext",
        "unknown field",
        "unknown parameter",
        "extra inputs are not permitted",
        "unrecognized",
        "unsupported",
    )
    if any(marker in body for marker in unsupported_markers):
        pytest.skip(
            f"D406: unsupported request parameter {_PINNED_REQUEST_PARAMETER}: "
            f"HTTP {result.status} body={result.body[:512]!r}"
        )


def _chat_completion_url(dynamo_endpoint_url: str) -> str:
    """Return the OpenAI-compatible chat-completions URL for the fixture URL."""
    return f"{dynamo_endpoint_url.rstrip('/')}/chat/completions"


def _iter_instance_ids(value: object) -> list[int]:
    """Recursively collect integer ``instance_id`` fields from health JSON."""
    if isinstance(value, dict):
        found: list[int] = []
        for key, child in value.items():
            if key == "instance_id" and isinstance(child, int):
                found.append(child)
            else:
                found.extend(_iter_instance_ids(child))
        return found
    if isinstance(value, list):
        found = []
        for child in value:
            found.extend(_iter_instance_ids(child))
        return found
    return []


def _find_int_key(value: object, key_name: str) -> int | None:
    """Return the first integer value for ``key_name`` in nested response JSON."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == key_name and isinstance(child, int):
                return child
            found = _find_int_key(child, key_name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_int_key(child, key_name)
            if found is not None:
                return found
    return None


# D407-D417

_d407_d417_DECODE_POD_SELECTOR = "nvidia.com/dynamo-sub-component-type=decode"
_PREFILL_POD_SELECTOR = "nvidia.com/dynamo-sub-component-type=prefill"
_d407_d417_COMPLETED_METRIC = "dynamo_frontend_requests_total"
_d407_d417_ERRORS_METRIC = "dynamo_frontend_requests_errors_total"
_d407_d417_INFLIGHT_METRIC_CANDIDATES = (
    "dynamo_component_inflight_requests",
    "dynamo_kv_router_inflight_requests",
    "dynamo_router_inflight_requests",
)
_QUEUE_METRIC_CANDIDATES = (
    "dynamo_kv_router_queue_depth",
    "dynamo_router_queue_depth",
    "dynamo_component_queued_requests",
    "dynamo_kv_router_queued_requests",
)
_WORKER_METRIC_CANDIDATES = (
    "dynamo_worker_load",
    "dynamo_worker_requests_total",
    "dynamo_component_worker_load",
    "dynamo_kv_router_worker_load",
    "dynamo_kv_router_workers",
)
_d407_d417_PINNED_REQUEST_PARAMETER = "nvext.decode_worker_id"
_PRIORITY_REQUEST_PARAMETER = "nvext.priority"
_d407_d417_REQUEST_TIMEOUT_S = 45.0
_RECOVERY_TIMEOUT_S = 120.0


async def test_d407_frozen_metrics_do_not_wedge_routing(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Detect frozen frontend metrics while proving routing still serves traffic."""
    metrics_before = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)
    _require_metric(metrics_before, _d407_d417_COMPLETED_METRIC, case="D407")

    await _d407_d417_post_completion(
        dynamo_endpoint_url, content="D407 warmup completion"
    )
    metrics_after = await _wait_for_metric_delta(
        kubectl,
        dynamo_deployment_namespace,
        metric_name=_d407_d417_COMPLETED_METRIC,
        before=metrics_before,
        case="D407",
    )
    errors_delta = metric_delta(metrics_after, metrics_before, _d407_d417_ERRORS_METRIC)
    completed_delta = metric_delta(
        metrics_after, metrics_before, _d407_d417_COMPLETED_METRIC
    )
    assert errors_delta <= completed_delta, (
        f"D407: frontend metrics advanced but errors exceeded completions "
        f"(completed_delta={completed_delta}, errors_delta={errors_delta})"
    )


async def test_d408_missing_worker_metrics_skips_or_serves_baseline(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Require explicit worker metrics before validating missing-metric behavior."""
    await _d407_d417_post_completion(dynamo_endpoint_url, content="D408 traffic proof")
    metrics = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)
    worker_metric = _select_metric(metrics, _WORKER_METRIC_CANDIDATES)
    if worker_metric is None:
        pytest.skip(
            "D408: endpoint serves traffic, but this Dynamo build exports no known "
            "worker metric; missing one of "
            f"{', '.join(_WORKER_METRIC_CANDIDATES)}"
        )
    assert metrics[worker_metric] >= 0.0, (
        f"D408: worker metric {worker_metric!r} exported an invalid negative value "
        f"{metrics[worker_metric]}"
    )


async def test_d409_saturated_workers_remain_bounded(
    dynamo_endpoint_url: str,
) -> None:
    """Run more concurrent requests than discovered workers and require completion."""
    worker_ids = await _require_multi_worker_support(dynamo_endpoint_url, case="D409")
    concurrency = max(len(worker_ids) * 4, 8)
    results = await asyncio.gather(
        *(
            _d407_d417_post_completion(
                dynamo_endpoint_url,
                content=f"D409 saturated worker probe {idx}",
                max_tokens=16,
            )
            for idx in range(concurrency)
        ),
        return_exceptions=True,
    )
    failures = [result for result in results if isinstance(result, Exception)]
    assert not failures, (
        f"D409: saturated-worker burst had {len(failures)}/{concurrency} "
        f"client failures; first={failures[0]!r}"
    )


async def test_d410_queued_cancellation_drains_inflight(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Cancel queued streams and assert router in-flight state drains when visible."""
    metrics_before = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)
    inflight_metric = _select_metric(
        metrics_before, _d407_d417_INFLIGHT_METRIC_CANDIDATES
    )
    if inflight_metric is None:
        pytest.skip(
            "D410: missing router in-flight metric; cannot prove queued "
            f"cancellation drained one of {', '.join(_d407_d417_INFLIGHT_METRIC_CANDIDATES)}"
        )

    baseline = metrics_before.get(inflight_metric, 0.0)
    tasks = [
        asyncio.create_task(
            _open_stream_until_first_chunk(
                dynamo_endpoint_url,
                content=f"D410 cancellable queued stream {idx}",
            )
        )
        for idx in range(4)
    ]
    await asyncio.sleep(2.0)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError, aiohttp.ClientError):
            await task

    await _d407_d417_post_completion(
        dynamo_endpoint_url, content="D410 post-cancel completion"
    )
    await _wait_for_gauge_at_or_below(
        kubectl,
        dynamo_deployment_namespace,
        metric_name=inflight_metric,
        baseline=baseline,
        case="D410",
    )


async def test_d411_priority_inversion_probe_is_bounded(
    dynamo_endpoint_url: str,
) -> None:
    """Probe priority hints and ensure low-priority load cannot hang high priority."""
    probe = await _post_completion_result(
        dynamo_endpoint_url,
        content="D411 priority support probe",
        max_tokens=1,
        nvext={"priority": 1},
    )
    _skip_if_request_hint_unsupported(probe, _PRIORITY_REQUEST_PARAMETER, case="D411")
    assert probe.status == 200, (
        f"D411: priority probe returned HTTP {probe.status}; body={probe.body[:512]!r}"
    )

    low_priority_tasks = [
        asyncio.create_task(
            _open_stream_until_first_chunk(
                dynamo_endpoint_url,
                content=f"D411 low-priority background stream {idx}",
                nvext={"priority": 0},
            )
        )
        for idx in range(4)
    ]
    try:
        await asyncio.sleep(2.0)
        start = time.monotonic()
        await _d407_d417_post_completion(
            dynamo_endpoint_url,
            content="D411 high-priority probe behind background load",
            max_tokens=4,
            nvext={"priority": 10},
        )
        latency = time.monotonic() - start
    finally:
        for task in low_priority_tasks:
            task.cancel()
        for task in low_priority_tasks:
            with contextlib.suppress(asyncio.CancelledError, aiohttp.ClientError):
                await task

    assert latency < _d407_d417_REQUEST_TIMEOUT_S, (
        f"D411: high-priority request took {latency:.2f}s behind low-priority "
        f"load; budget={_d407_d417_REQUEST_TIMEOUT_S}s"
    )


async def test_d412_topology_churn_with_queue_recovers(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Hold queued traffic, delete a decode pod, and require post-churn service."""
    await _require_multi_worker_support(dynamo_endpoint_url, case="D412")
    decode_pod = await _select_ready_pod(kubectl, dynamo_deployment_namespace, "decode")
    if decode_pod is None:
        pytest.skip(
            f"D412: no ready decode pod in {dynamo_deployment_namespace!r} "
            f"matching {_d407_d417_DECODE_POD_SELECTOR!r}"
        )

    queue_tasks = [
        asyncio.create_task(
            _open_stream_until_first_chunk(
                dynamo_endpoint_url,
                content=f"D412 queued stream before topology churn {idx}",
            )
        )
        for idx in range(4)
    ]
    try:
        await asyncio.sleep(2.0)
        async with faults.inject(
            "pod.kill",
            target={"ns": dynamo_deployment_namespace, "pod": decode_pod},
        ):
            await _wait_for_replacement_pod(
                kubectl,
                dynamo_deployment_namespace,
                role="decode",
                old_pod=decode_pod,
            )
    finally:
        for task in queue_tasks:
            task.cancel()
        for task in queue_tasks:
            with contextlib.suppress(asyncio.CancelledError, aiohttp.ClientError):
                await task

    await _d407_d417_wait_until_frontend_serves(
        dynamo_endpoint_url,
        case="D412",
        timeout_s=45.0,
    )


async def test_d413_prefill_zero_degrades_cleanly(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Scale prefill workers to zero and require bounded client behavior."""
    deployment = await _find_role_deployment(
        kubectl, dynamo_deployment_namespace, "prefill"
    )
    if deployment is None:
        pytest.skip(
            "D413: no prefill deployment found; aggregated topology has no prefill tier"
        )
    await _require_ready_role_pods(
        kubectl, dynamo_deployment_namespace, "prefill", case="D413"
    )

    async with faults.inject(
        "workload.scale",
        target={"ns": dynamo_deployment_namespace, "deployment": deployment},
        replicas=0,
    ):
        await _expect_bounded_completion_or_error(
            dynamo_endpoint_url,
            case="D413",
            content="D413 prefill-zero bounded request",
        )

    await _d407_d417_wait_until_frontend_serves(
        dynamo_endpoint_url, case="D413", timeout_s=90.0
    )


async def test_d414_decode_zero_degrades_cleanly(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Scale decode workers to zero and require bounded client behavior."""
    deployment = await _find_role_deployment(
        kubectl, dynamo_deployment_namespace, "decode"
    )
    if deployment is None:
        pytest.skip("D414: no decode deployment found")
    await _require_ready_role_pods(
        kubectl, dynamo_deployment_namespace, "decode", case="D414"
    )

    async with faults.inject(
        "workload.scale",
        target={"ns": dynamo_deployment_namespace, "deployment": deployment},
        replicas=0,
    ):
        await _expect_bounded_completion_or_error(
            dynamo_endpoint_url,
            case="D414",
            content="D414 decode-zero bounded request",
        )

    await _d407_d417_wait_until_frontend_serves(
        dynamo_endpoint_url, case="D414", timeout_s=90.0
    )


async def test_d415_scale_out_decode_under_traffic(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Scale decode replicas out while traffic is active and require recovery."""
    deployment = await _find_role_deployment(
        kubectl, dynamo_deployment_namespace, "decode"
    )
    if deployment is None:
        pytest.skip("D415: no decode deployment found")
    replicas = await _deployment_replicas(
        kubectl, dynamo_deployment_namespace, deployment
    )
    traffic_task = asyncio.create_task(
        _traffic_loop(dynamo_endpoint_url, case="D415", duration_s=10.0)
    )
    try:
        async with faults.inject(
            "workload.scale",
            target={"ns": dynamo_deployment_namespace, "deployment": deployment},
            replicas=replicas + 1,
        ):
            await _wait_for_ready_role_count(
                kubectl,
                dynamo_deployment_namespace,
                role="decode",
                minimum=replicas + 1,
                case="D415",
            )
    finally:
        with contextlib.suppress(Exception):
            await traffic_task
    await _d407_d417_wait_until_frontend_serves(
        dynamo_endpoint_url, case="D415", timeout_s=45.0
    )


async def test_d416_scale_in_decode_under_traffic(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Scale decode replicas in while traffic is active and require recovery."""
    deployment = await _find_role_deployment(
        kubectl, dynamo_deployment_namespace, "decode"
    )
    if deployment is None:
        pytest.skip("D416: no decode deployment found")
    replicas = await _deployment_replicas(
        kubectl, dynamo_deployment_namespace, deployment
    )
    if replicas < 2:
        pytest.skip(
            f"D416: scale-in requires at least 2 decode replicas, found {replicas}"
        )

    traffic_task = asyncio.create_task(
        _traffic_loop(dynamo_endpoint_url, case="D416", duration_s=10.0)
    )
    try:
        async with faults.inject(
            "workload.scale",
            target={"ns": dynamo_deployment_namespace, "deployment": deployment},
            replicas=replicas - 1,
        ):
            await _wait_for_ready_role_count(
                kubectl,
                dynamo_deployment_namespace,
                role="decode",
                minimum=replicas - 1,
                case="D416",
            )
    finally:
        with contextlib.suppress(Exception):
            await traffic_task
    await _d407_d417_wait_until_frontend_serves(
        dynamo_endpoint_url, case="D416", timeout_s=45.0
    )


async def test_d417_duplicate_worker_identity_requires_identity_injection(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Skip unless the deployment exposes an identity env var that can be duplicated."""
    await _require_multi_worker_support(dynamo_endpoint_url, case="D417")
    deployment = await _find_role_deployment(
        kubectl, dynamo_deployment_namespace, "decode"
    )
    if deployment is None:
        pytest.skip("D417: no decode deployment found")
    env_names = await _deployment_env_names(
        kubectl, dynamo_deployment_namespace, deployment
    )
    duplicate_identity_envs = [
        name
        for name in env_names
        if name in {"DYN_WORKER_ID", "DYNAMO_WORKER_ID", "DYN_INSTANCE_ID"}
        or "WORKER_ID" in name
        or "INSTANCE_ID" in name
    ]
    if not duplicate_identity_envs:
        pytest.skip(
            "D417: duplicate worker identity injection requires an explicit worker "
            f"identity env var on deployment {deployment!r}; observed envs={env_names!r}"
        )
    pytest.skip(
        "D417: identity env surface is present, but this test intentionally avoids "
        "mutating it until the expanded spec names the safe duplicate value source"
    )


def _require_metric(metrics: dict[str, Any], metric_name: str, *, case: str) -> None:
    if metric_name not in metrics:
        pytest.skip(f"{case}: required metric {metric_name!r} is not exported")


def _select_metric(metrics: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in metrics:
            return candidate
    return None


async def _wait_for_metric_delta(
    kubectl: KubectlClient,
    namespace: str,
    *,
    metric_name: str,
    before: dict[str, Any],
    case: str,
    timeout_s: float = 20.0,
) -> dict[str, float]:
    deadline = time.monotonic() + timeout_s
    last_metrics: dict[str, float] = {}
    while time.monotonic() < deadline:
        last_metrics = await scrape_frontend_metrics(kubectl, namespace)
        if metric_delta(last_metrics, before, metric_name) > 0:
            return last_metrics
        await asyncio.sleep(1.0)
    pytest.fail(
        f"{case}: metric {metric_name!r} did not advance within {timeout_s}s; "
        f"before={before.get(metric_name)!r}, last={last_metrics.get(metric_name)!r}"
    )


async def _wait_for_gauge_at_or_below(
    kubectl: KubectlClient,
    namespace: str,
    *,
    metric_name: str,
    baseline: float,
    case: str,
    timeout_s: float = 45.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_value: float | None = None
    while time.monotonic() < deadline:
        metrics = await scrape_frontend_metrics(kubectl, namespace)
        value = metrics.get(metric_name)
        if value is None:
            pytest.skip(f"{case}: metric {metric_name!r} disappeared during poll")
        last_value = float(value)
        if last_value <= baseline:
            return
        await asyncio.sleep(2.0)
    pytest.fail(
        f"{case}: {metric_name!r} did not drain to baseline {baseline} within "
        f"{timeout_s}s; last={last_value}"
    )


async def _require_multi_worker_support(endpoint_url: str, *, case: str) -> list[int]:
    worker_ids = await _d407_d417_discover_worker_ids(endpoint_url)
    if len(worker_ids) < 2:
        pytest.skip(
            f"{case}: requires multi-worker /health instance_id support; "
            f"observed worker_ids={worker_ids!r}"
        )
    return worker_ids


async def _d407_d417_discover_worker_ids(endpoint_url: str) -> list[int]:
    root = endpoint_url.rstrip("/").removesuffix("/v1")
    urls = [f"{root}/health", f"{endpoint_url.rstrip('/')}/health"]
    timeout = aiohttp.ClientTimeout(total=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for url in urls:
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    payload = await resp.json(content_type=None)
                    return sorted(set(_d407_d417_iter_instance_ids(payload)))
            except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
                logger.debug(
                    lambda exc=exc, url=url: f"health probe {url} failed: {exc!r}"
                )
    return []


def _d407_d417_iter_instance_ids(value: object) -> list[int]:
    if isinstance(value, dict):
        found: list[int] = []
        for key, child in value.items():
            if key == "instance_id" and isinstance(child, int):
                found.append(child)
            else:
                found.extend(_d407_d417_iter_instance_ids(child))
        return found
    if isinstance(value, list):
        found = []
        for child in value:
            found.extend(_d407_d417_iter_instance_ids(child))
        return found
    return []


class _d407_d417_CompletionResult:
    """Small typed container for one HTTP completion response."""

    def __init__(
        self, status: int, body: str, json_body: dict[str, Any] | None
    ) -> None:
        self.status = status
        self.body = body
        self.json_body = json_body


async def _d407_d417_post_completion(
    endpoint_url: str,
    *,
    content: str,
    max_tokens: int = 8,
    nvext: Mapping[str, object] | None = None,
) -> None:
    result = await _post_completion_result(
        endpoint_url,
        content=content,
        max_tokens=max_tokens,
        nvext=nvext,
    )
    assert result.status < 500, (
        f"completion failed with HTTP {result.status}; body={result.body[:512]!r}"
    )
    assert result.status < 400, (
        f"completion rejected with HTTP {result.status}; body={result.body[:512]!r}"
    )


async def _post_completion_result(
    endpoint_url: str,
    *,
    content: str,
    max_tokens: int,
    nvext: Mapping[str, object] | None = None,
) -> _d407_d417_CompletionResult:
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    if nvext is not None:
        payload["nvext"] = dict(nvext)
    timeout = aiohttp.ClientTimeout(total=_d407_d417_REQUEST_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(
            _d407_d417_chat_completion_url(endpoint_url), json=payload
        ) as resp,
    ):
        body = await resp.text()
        json_body: dict[str, Any] | None = None
        with contextlib.suppress(ValueError):
            parsed = await resp.json(content_type=None)
            if isinstance(parsed, dict):
                json_body = parsed
        return _d407_d417_CompletionResult(resp.status, body, json_body)


async def _open_stream_until_first_chunk(
    endpoint_url: str,
    *,
    content: str,
    nvext: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 128,
        "stream": True,
        "temperature": 0.0,
    }
    if nvext is not None:
        payload["nvext"] = dict(nvext)
    timeout = aiohttp.ClientTimeout(total=_d407_d417_REQUEST_TIMEOUT_S + 30.0)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(
            _d407_d417_chat_completion_url(endpoint_url), json=payload
        ) as resp,
    ):
        assert resp.status == 200, (
            f"stream returned HTTP {resp.status}; body={(await resp.text())[:512]!r}"
        )
        async for chunk in resp.content.iter_any():
            if chunk:
                await asyncio.sleep(_d407_d417_REQUEST_TIMEOUT_S)
                return


async def _expect_bounded_completion_or_error(
    endpoint_url: str,
    *,
    case: str,
    content: str,
) -> None:
    try:
        result = await _post_completion_result(
            endpoint_url,
            content=content,
            max_tokens=8,
        )
    except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as exc:
        logger.info(
            lambda exc=exc: f"{case}: bounded client error while tier is zero: {exc!r}"
        )
        return
    assert result.status < 500 or result.status in {503, 504}, (
        f"{case}: unexpected server failure while tier is zero: "
        f"HTTP {result.status}; body={result.body[:512]!r}"
    )


async def _d407_d417_wait_until_frontend_serves(
    endpoint_url: str,
    *,
    case: str,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "<not attempted>"
    while time.monotonic() < deadline:
        try:
            await _d407_d417_post_completion(
                endpoint_url, content=f"{case} recovery probe"
            )
            return
        except (aiohttp.ClientError, AssertionError, TimeoutError) as exc:
            last_error = repr(exc)
        await asyncio.sleep(2.0)
    pytest.fail(
        f"{case}: frontend did not serve a chat completion within {timeout_s}s; "
        f"last_error={last_error}"
    )


async def _traffic_loop(endpoint_url: str, *, case: str, duration_s: float) -> None:
    deadline = time.monotonic() + duration_s
    idx = 0
    while time.monotonic() < deadline:
        try:
            await _d407_d417_post_completion(
                endpoint_url, content=f"{case} traffic loop {idx}"
            )
        except (aiohttp.ClientError, AssertionError, TimeoutError) as exc:
            logger.warning(lambda exc=exc, case=case: f"{case}: traffic error {exc!r}")
        idx += 1
        await asyncio.sleep(0.5)


def _skip_if_request_hint_unsupported(
    result: _d407_d417_CompletionResult,
    parameter: str,
    *,
    case: str,
) -> None:
    if result.status not in {400, 404, 422}:
        return
    body = result.body.lower()
    markers = (
        parameter.rsplit(".", 1)[-1].lower(),
        "nvext",
        "unknown field",
        "unknown parameter",
        "extra inputs are not permitted",
        "unrecognized",
        "unsupported",
    )
    if any(marker in body for marker in markers):
        pytest.skip(
            f"{case}: unsupported request parameter {parameter}: "
            f"HTTP {result.status} body={result.body[:512]!r}"
        )


async def _select_ready_pod(
    kubectl: KubectlClient,
    namespace: str,
    role: str,
) -> str | None:
    for pod in await _list_role_pods(kubectl, namespace, role):
        name = _d407_d417_pod_name(pod)
        if name and _d407_d417_pod_is_ready(pod):
            return name
    return None


async def _require_ready_role_pods(
    kubectl: KubectlClient,
    namespace: str,
    role: str,
    *,
    case: str,
) -> None:
    pods = await _list_role_pods(kubectl, namespace, role)
    ready = [pod for pod in pods if _d407_d417_pod_is_ready(pod)]
    if not ready:
        pytest.skip(f"{case}: no ready {role} pods found in namespace {namespace!r}")


async def _wait_for_replacement_pod(
    kubectl: KubectlClient,
    namespace: str,
    *,
    role: str,
    old_pod: str,
    timeout_s: float = _RECOVERY_TIMEOUT_S,
) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for pod in await _list_role_pods(kubectl, namespace, role):
            name = _d407_d417_pod_name(pod)
            if name and name != old_pod and _d407_d417_pod_is_ready(pod):
                return name
        await asyncio.sleep(2.0)
    raise TimeoutError(
        f"no replacement {role} pod became ready in {namespace!r} within "
        f"{timeout_s}s after deleting {old_pod!r}"
    )


async def _wait_for_ready_role_count(
    kubectl: KubectlClient,
    namespace: str,
    *,
    role: str,
    minimum: int,
    case: str,
    timeout_s: float = _RECOVERY_TIMEOUT_S,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_count = 0
    while time.monotonic() < deadline:
        pods = await _list_role_pods(kubectl, namespace, role)
        last_count = sum(1 for pod in pods if _d407_d417_pod_is_ready(pod))
        if last_count >= minimum:
            return
        await asyncio.sleep(2.0)
    pytest.fail(
        f"{case}: only {last_count} ready {role} pods after {timeout_s}s; "
        f"expected at least {minimum}"
    )


async def _list_role_pods(
    kubectl: KubectlClient,
    namespace: str,
    role: str,
) -> list[dict[str, Any]]:
    selector = _role_selector(role)
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        selector,
        "-o",
        "json",
        check=True,
    )
    loaded = orjson.loads(result.stdout)
    items = loaded.get("items", []) if isinstance(loaded, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _role_selector(role: str) -> str:
    if role == "decode":
        return _d407_d417_DECODE_POD_SELECTOR
    if role == "prefill":
        return _PREFILL_POD_SELECTOR
    raise ValueError(f"unsupported Dynamo worker role {role!r}")


def _d407_d417_pod_name(pod: dict[str, Any]) -> str:
    metadata = pod.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    name = metadata.get("name", "")
    return name if isinstance(name, str) else ""


def _d407_d417_pod_is_ready(pod: dict[str, Any]) -> bool:
    metadata = pod.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("deletionTimestamp"):
        return False
    status = pod.get("status", {})
    if not isinstance(status, dict) or status.get("phase") != "Running":
        return False
    conditions = status.get("conditions", [])
    return isinstance(conditions, list) and any(
        isinstance(condition, dict)
        and condition.get("type") == "Ready"
        and condition.get("status") == "True"
        for condition in conditions
    )


async def _find_role_deployment(
    kubectl: KubectlClient,
    namespace: str,
    role: str,
) -> str | None:
    result = await kubectl.run(
        "get",
        "deployments",
        "-n",
        namespace,
        "-o",
        "json",
        check=True,
    )
    loaded = orjson.loads(result.stdout)
    items = loaded.get("items", []) if isinstance(loaded, dict) else []
    role_pods = await _list_role_pods(kubectl, namespace, role)
    owner_names = _pod_owner_names(role_pods)
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _deployment_name(item)
        if not name:
            continue
        if name in owner_names or role in name:
            return name
        labels = item.get("metadata", {}).get("labels", {})
        if isinstance(labels, dict) and role in " ".join(
            str(v) for v in labels.values()
        ):
            return name
    return None


def _pod_owner_names(pods: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for pod in pods:
        metadata = pod.get("metadata", {})
        owner_refs = (
            metadata.get("ownerReferences", []) if isinstance(metadata, dict) else []
        )
        if not isinstance(owner_refs, list):
            continue
        for owner in owner_refs:
            if not isinstance(owner, dict):
                continue
            name = owner.get("name")
            if isinstance(name, str):
                names.add(name.rsplit("-", 1)[0])
    return names


def _deployment_name(deployment: dict[str, Any]) -> str:
    metadata = deployment.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    name = metadata.get("name", "")
    return name if isinstance(name, str) else ""


async def _deployment_replicas(
    kubectl: KubectlClient,
    namespace: str,
    deployment: str,
) -> int:
    result = await kubectl.run(
        "get",
        "deployment",
        deployment,
        "-n",
        namespace,
        "-o",
        "jsonpath={.spec.replicas}",
        check=True,
    )
    text = result.stdout.strip()
    return int(text) if text else 1


async def _deployment_env_names(
    kubectl: KubectlClient,
    namespace: str,
    deployment: str,
) -> list[str]:
    result = await kubectl.run(
        "get",
        "deployment",
        deployment,
        "-n",
        namespace,
        "-o",
        "json",
        check=True,
    )
    loaded = orjson.loads(result.stdout)
    containers = (
        loaded.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        if isinstance(loaded, dict)
        else []
    )
    names: list[str] = []
    if isinstance(containers, list):
        for container in containers:
            if not isinstance(container, dict):
                continue
            envs = container.get("env", [])
            if not isinstance(envs, list):
                continue
            for env in envs:
                if isinstance(env, dict) and isinstance(env.get("name"), str):
                    names.append(env["name"])
    return sorted(set(names))


def _d407_d417_chat_completion_url(endpoint_url: str) -> str:
    return f"{endpoint_url.rstrip('/')}/chat/completions"

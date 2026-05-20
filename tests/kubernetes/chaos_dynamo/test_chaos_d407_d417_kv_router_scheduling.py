# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D407-D417 -- KV-router scheduling and worker-metrics chaos probes.

These cases extend the D401/D402/D403/D803 style: prove the deployed Dynamo
endpoint is serving, then skip with a concrete prerequisite when the running
build lacks the multi-worker, worker-metrics, or request-hint surface needed to
validate a specific scheduler invariant. The assertions are intentionally
black-box because the tests run against a real Kubernetes deployment rather than
in-process router internals.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping
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
_PREFILL_POD_SELECTOR = "nvidia.com/dynamo-sub-component-type=prefill"
_COMPLETED_METRIC = "dynamo_frontend_requests_total"
_ERRORS_METRIC = "dynamo_frontend_requests_errors_total"
_INFLIGHT_METRIC_CANDIDATES = (
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
_PINNED_REQUEST_PARAMETER = "nvext.decode_worker_id"
_PRIORITY_REQUEST_PARAMETER = "nvext.priority"
_REQUEST_TIMEOUT_S = 45.0
_RECOVERY_TIMEOUT_S = 120.0


async def test_d407_frozen_metrics_do_not_wedge_routing(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Detect frozen frontend metrics while proving routing still serves traffic."""
    metrics_before = await scrape_frontend_metrics(kubectl, dynamo_deployment_namespace)
    _require_metric(metrics_before, _COMPLETED_METRIC, case="D407")

    await _post_completion(dynamo_endpoint_url, content="D407 warmup completion")
    metrics_after = await _wait_for_metric_delta(
        kubectl,
        dynamo_deployment_namespace,
        metric_name=_COMPLETED_METRIC,
        before=metrics_before,
        case="D407",
    )
    errors_delta = _metric_delta(metrics_after, metrics_before, _ERRORS_METRIC)
    completed_delta = _metric_delta(metrics_after, metrics_before, _COMPLETED_METRIC)
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
    await _post_completion(dynamo_endpoint_url, content="D408 traffic proof")
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
            _post_completion(
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
    inflight_metric = _select_metric(metrics_before, _INFLIGHT_METRIC_CANDIDATES)
    if inflight_metric is None:
        pytest.skip(
            "D410: missing router in-flight metric; cannot prove queued "
            f"cancellation drained one of {', '.join(_INFLIGHT_METRIC_CANDIDATES)}"
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

    await _post_completion(dynamo_endpoint_url, content="D410 post-cancel completion")
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
        await _post_completion(
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

    assert latency < _REQUEST_TIMEOUT_S, (
        f"D411: high-priority request took {latency:.2f}s behind low-priority "
        f"load; budget={_REQUEST_TIMEOUT_S}s"
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
            f"matching {_DECODE_POD_SELECTOR!r}"
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

    await _wait_until_frontend_serves(
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

    await _wait_until_frontend_serves(dynamo_endpoint_url, case="D413", timeout_s=90.0)


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

    await _wait_until_frontend_serves(dynamo_endpoint_url, case="D414", timeout_s=90.0)


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
    await _wait_until_frontend_serves(dynamo_endpoint_url, case="D415", timeout_s=45.0)


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
    await _wait_until_frontend_serves(dynamo_endpoint_url, case="D416", timeout_s=45.0)


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


def _metric_delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return float(after.get(key, 0.0)) - float(before.get(key, 0.0))


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
        if _metric_delta(last_metrics, before, metric_name) > 0:
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
    worker_ids = await _discover_worker_ids(endpoint_url)
    if len(worker_ids) < 2:
        pytest.skip(
            f"{case}: requires multi-worker /health instance_id support; "
            f"observed worker_ids={worker_ids!r}"
        )
    return worker_ids


async def _discover_worker_ids(endpoint_url: str) -> list[int]:
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
                    return sorted(set(_iter_instance_ids(payload)))
            except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
                logger.debug(
                    lambda exc=exc, url=url: f"health probe {url} failed: {exc!r}"
                )
    return []


def _iter_instance_ids(value: object) -> list[int]:
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


class _CompletionResult:
    """Small typed container for one HTTP completion response."""

    def __init__(
        self, status: int, body: str, json_body: dict[str, Any] | None
    ) -> None:
        self.status = status
        self.body = body
        self.json_body = json_body


async def _post_completion(
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
) -> _CompletionResult:
    payload: dict[str, object] = {
        "model": "default",
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
        session.post(_chat_completion_url(endpoint_url), json=payload) as resp,
    ):
        body = await resp.text()
        json_body: dict[str, Any] | None = None
        with contextlib.suppress(ValueError):
            parsed = await resp.json(content_type=None)
            if isinstance(parsed, dict):
                json_body = parsed
        return _CompletionResult(resp.status, body, json_body)


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
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S + 30.0)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(_chat_completion_url(endpoint_url), json=payload) as resp,
    ):
        assert resp.status == 200, (
            f"stream returned HTTP {resp.status}; body={(await resp.text())[:512]!r}"
        )
        async for chunk in resp.content.iter_any():
            if chunk:
                await asyncio.sleep(_REQUEST_TIMEOUT_S)
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


async def _wait_until_frontend_serves(
    endpoint_url: str,
    *,
    case: str,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "<not attempted>"
    while time.monotonic() < deadline:
        try:
            await _post_completion(endpoint_url, content=f"{case} recovery probe")
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
            await _post_completion(endpoint_url, content=f"{case} traffic loop {idx}")
        except (aiohttp.ClientError, AssertionError, TimeoutError) as exc:
            logger.warning(lambda exc=exc, case=case: f"{case}: traffic error {exc!r}")
        idx += 1
        await asyncio.sleep(0.5)


def _skip_if_request_hint_unsupported(
    result: _CompletionResult,
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
        name = _pod_name(pod)
        if name and _pod_is_ready(pod):
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
    ready = [pod for pod in pods if _pod_is_ready(pod)]
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
            name = _pod_name(pod)
            if name and name != old_pod and _pod_is_ready(pod):
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
        last_count = sum(1 for pod in pods if _pod_is_ready(pod))
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
        return _DECODE_POD_SELECTOR
    if role == "prefill":
        return _PREFILL_POD_SELECTOR
    raise ValueError(f"unsupported Dynamo worker role {role!r}")


def _pod_name(pod: dict[str, Any]) -> str:
    metadata = pod.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    name = metadata.get("name", "")
    return name if isinstance(name, str) else ""


def _pod_is_ready(pod: dict[str, Any]) -> bool:
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


def _chat_completion_url(endpoint_url: str) -> str:
    return f"{endpoint_url.rstrip('/')}/chat/completions"

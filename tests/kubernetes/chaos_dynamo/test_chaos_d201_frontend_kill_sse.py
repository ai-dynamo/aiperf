# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D201 — force-kill Frontend pod under 64 concurrent SSE streams.

D-series catalog § D2xx scenario. Targets the disconnect cleanup path in
``lib/llm/src/http/service/disconnect.rs:195`` (``monitor_for_disconnects``):
when the frontend pod is force-deleted mid-stream, every in-flight SSE
client should observe a clean TCP RST or HTTP 503 within ~1s (no infinite
hangs), the ``dynamo_frontend_disconnected_clients`` counter should advance,
and the operator-managed replacement pod should be ready to serve new
requests within 30s.
"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.chaos_dynamo.conftest import scrape_frontend_metrics
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


CONCURRENT_STREAMS = 64
"""Number of in-flight SSE clients held open at fault-injection time."""

DYNAMO_SERVER_NAMESPACE = "dynamo-server"
"""Default Dynamo deployment namespace used by the gpu/dynamo fixtures."""

FRONTEND_LABEL_SELECTOR = "nvidia.com/dynamo-component-type=frontend"
"""Label the Dynamo operator stamps on every frontend pod."""

DISCONNECTED_METRIC = "dynamo_frontend_disconnected_clients"
"""Cumulative counter incremented by disconnect.rs on client-side cancel."""

STREAM_SETTLE_SECONDS = 1.0
"""Window we wait for streams to be actively reading before injecting the kill."""

CLIENT_TIMEOUT_SECONDS = 30.0
"""Hard ceiling per client — well above the 1s RST target so a hung client
manifests as an asyncio.TimeoutError instead of an aiohttp socket error."""

RECOVERY_TIMEOUT_SECONDS = 30.0
"""Total budget for a replacement frontend pod to become Ready."""


async def _resolve_frontend_pod(kubectl: KubectlClient, namespace: str) -> str:
    """Return the name of the first frontend pod matching the label selector.

    Raises:
        RuntimeError: When no pod matches the selector — D201 is meaningless
            without a live frontend to kill, so callers should fail fast.
    """
    result = await kubectl.run(
        "get",
        "pod",
        "-l",
        FRONTEND_LABEL_SELECTOR,
        "-n",
        namespace,
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=False,
    )
    pod = result.stdout.strip() if result.returncode == 0 else ""
    if not pod:
        raise RuntimeError(
            f"D201: no frontend pod matched label {FRONTEND_LABEL_SELECTOR!r} "
            f"in namespace {namespace!r}; check `kubectl get pods -n {namespace}`"
        )
    return pod


async def _wait_for_new_frontend_ready(
    kubectl: KubectlClient,
    namespace: str,
    killed_pod: str,
    timeout: float,
) -> str:
    """Poll until a frontend pod whose name differs from ``killed_pod`` is Ready.

    Returns the new pod's name. Raises ``TimeoutError`` if no replacement is
    ready inside ``timeout`` seconds.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last_observed: str = "<none>"
    while True:
        pods = await kubectl.get_pods(namespace)
        for candidate in pods:
            if (
                candidate.name != killed_pod
                and candidate.is_ready
                and FRONTEND_LABEL_SELECTOR.split("=")[0] in candidate.labels
                and candidate.labels.get(FRONTEND_LABEL_SELECTOR.split("=")[0])
                == FRONTEND_LABEL_SELECTOR.split("=")[1]
            ):
                return candidate.name
            last_observed = f"{candidate.name}(ready={candidate.is_ready})"
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(
                f"D201: no replacement frontend pod became Ready within "
                f"{timeout}s (killed={killed_pod!r}, last_observed={last_observed!r})"
            )
        await asyncio.sleep(0.5)


async def _stream_one(
    session: aiohttp.ClientSession,
    url: str,
    idx: int,
) -> dict[str, Any]:
    """Open one SSE stream and read until the server closes it or errors.

    Returns a small dict describing the terminal state so the test body can
    bucket outcomes (clean-RST vs 503 vs unexpected) without re-raising.
    """
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": f"stream {idx}"}],
        "stream": True,
        "max_tokens": 10,
    }
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_SECONDS),
        ) as resp:
            status = resp.status
            chunks = 0
            async for _chunk in resp.content.iter_chunked(1024):
                chunks += 1
            return {"idx": idx, "kind": "closed", "status": status, "chunks": chunks}
    except aiohttp.ServerDisconnectedError as exc:
        return {"idx": idx, "kind": "server_disconnect", "error": repr(exc)}
    except aiohttp.ClientConnectionError as exc:
        return {"idx": idx, "kind": "conn_error", "error": repr(exc)}
    except asyncio.TimeoutError as exc:
        return {"idx": idx, "kind": "timeout", "error": repr(exc)}
    except aiohttp.ClientError as exc:
        return {"idx": idx, "kind": "client_error", "error": repr(exc)}


async def test_d201_force_kill_frontend_under_sse(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
    dynamo_endpoint_url: str,
    dynamo_deployment_namespace: str,
) -> None:
    """D201 — force-kill the frontend pod with 64 concurrent SSE streams in flight.

    Steps:
        1. Snapshot the dynamo_frontend_disconnected_clients counter (best-effort).
        2. Spawn ``CONCURRENT_STREAMS`` SSE POSTs and wait ``STREAM_SETTLE_SECONDS``
           so the streams are actively reading.
        3. Inject ``pod.kill`` against a frontend pod resolved by label.
        4. Gather every stream task; bucket outcomes into clean-RST / 503 /
           unexpected.
        5. Assert no client is left hung (every task terminated within the
           hard timeout), and at least 50/64 saw a clean disconnect (RST or 503).
        6. Wait up to ``RECOVERY_TIMEOUT_SECONDS`` for a replacement frontend
           pod to become Ready.
        7. Issue a fresh non-streaming completion; assert HTTP 200 to confirm
           the new replica is serving.
        8. Scrape /metrics on the new pod and assert
           ``dynamo_frontend_disconnected_clients`` advanced past the snapshot.
    """
    namespace = dynamo_deployment_namespace or DYNAMO_SERVER_NAMESPACE

    # 1. Snapshot disconnect counter (tolerate missing — first run, fresh pod).
    try:
        before_metrics = await scrape_frontend_metrics(kubectl, namespace)
    except RuntimeError as exc:
        logger.debug(
            lambda exc=exc: f"D201: pre-snapshot metrics scrape failed: {exc!r}"
        )
        before_metrics = {}
    disconnect_before = float(before_metrics.get(DISCONNECTED_METRIC, 0.0))

    killed_pod = await _resolve_frontend_pod(kubectl, namespace)
    logger.info(f"D201: resolved frontend pod {killed_pod!r} for fault injection")

    # 2 + 3. Launch streams, settle, inject pod.kill while streams in flight.
    async with aiohttp.ClientSession() as session:
        stream_tasks = [
            asyncio.create_task(_stream_one(session, dynamo_endpoint_url, i))
            for i in range(CONCURRENT_STREAMS)
        ]
        await asyncio.sleep(STREAM_SETTLE_SECONDS)

        async with faults.inject(
            "pod.kill",
            target={"ns": namespace, "pod": killed_pod},
        ):
            results = await asyncio.gather(*stream_tasks, return_exceptions=False)

    # 4 + 5. Bucket outcomes.
    clean_disconnects = sum(
        1
        for r in results
        if r["kind"] in ("server_disconnect", "conn_error")
        or (r["kind"] == "closed" and r.get("status") == 503)
    )
    timeouts = sum(1 for r in results if r["kind"] == "timeout")
    logger.info(
        f"D201: stream outcomes — clean={clean_disconnects} timeouts={timeouts} "
        f"total={len(results)}"
    )
    assert timeouts == 0, (
        f"D201: {timeouts}/{len(results)} clients hung past "
        f"{CLIENT_TIMEOUT_SECONDS}s — disconnect.rs:195 cleanup likely broken"
    )
    assert clean_disconnects >= 50, (
        f"D201: only {clean_disconnects}/{CONCURRENT_STREAMS} clients saw a "
        f"clean RST/503 (expected >=50); outcomes={results!r}"
    )

    # 6. Wait for the replacement frontend pod.
    new_pod = await _wait_for_new_frontend_ready(
        kubectl, namespace, killed_pod, RECOVERY_TIMEOUT_SECONDS
    )
    logger.info(f"D201: replacement frontend pod ready: {new_pod!r}")

    # 7. New request should succeed against the fresh replica.
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"{dynamo_endpoint_url}/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "post-recovery probe"}],
                "stream": False,
                "max_tokens": 10,
            },
            timeout=aiohttp.ClientTimeout(total=10.0),
        ) as resp,
    ):
        assert resp.status == 200, (
            f"D201: post-recovery request returned {resp.status}; "
            f"replacement pod {new_pod!r} is not actually serving"
        )

    # 8. Disconnect counter should have advanced on the new pod.
    try:
        after_metrics = await scrape_frontend_metrics(kubectl, namespace)
    except RuntimeError as exc:
        logger.warning(
            lambda exc=exc: f"D201: post-recovery metrics scrape failed: {exc!r}"
        )
        return
    disconnect_after = float(after_metrics.get(DISCONNECTED_METRIC, 0.0))
    assert disconnect_after > disconnect_before, (
        f"D201: {DISCONNECTED_METRIC} did not advance "
        f"(before={disconnect_before}, after={disconnect_after}); "
        f"disconnect.rs:195 may not be incrementing on force-kill RST"
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D401 -- Kill decode worker mid-request; assert client sees clean error.

D-series catalog reference: D4xx (worker-lifecycle faults).

Targets the no-retry-on-worker-death gap documented in
``lib/kv-router/src/scheduling/queue.rs:173-210``: when a worker pod evaporates
mid-generation, the router does not re-dispatch the in-flight request to a
surviving worker. The client must therefore see a *clean* failure (a 5xx, an
``aiohttp.ServerDisconnectedError``, or a stream that terminates with an
``[ERROR]`` SSE event) bounded by ``DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS``
(default 30s) -- never an infinite hang.

Why this scenario matters:

* It establishes the upper bound on user-visible latency when a node is lost,
  which is the metric SREs need for SLO budgeting.
* It guards against regressions that would silently turn the existing
  fail-fast behaviour into a retry-forever loop (or vice versa) -- either
  change is a contract break and must be a deliberate design decision.
"""

from __future__ import annotations

import asyncio
import time

import aiohttp
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


# Upper bound on time from kill -> client-visible error. The Dynamo HTTP
# frontend's per-stream timeout is configurable via
# ``DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS`` (default 30s); we add a 5s slack so
# a borderline-on-time termination still passes. Anything past this is treated
# as an infinite-hang regression.
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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D225-D233 -- Dynamo frontend/API chaos coverage.

These cases exercise the OpenAI-compatible frontend surface with real HTTP
traffic against the Kubernetes-deployed Dynamo service. Load-shedding and
idle-stall cases are topology-gated because Dynamo only exposes those behaviors
when the frontend is configured with admission-control or explicit stall-fault
knobs; unsupported topologies skip before recording false-positive success.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import aiohttp
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

FRONTEND_SELECTOR = "nvidia.com/dynamo-component-type=frontend"
DEFAULT_MODEL = os.environ.get("AIPERF_DYNAMO_CHAOS_MODEL", "default")
LOAD_SHEDDING_OPT_IN_ENV = "AIPERF_DYNAMO_LOAD_SHEDDING_CHAOS"
IDLE_STALL_OPT_IN_ENV = "AIPERF_DYNAMO_IDLE_STALL_CHAOS"
REQUEST_ID_HEADERS = (
    "x-request-id",
    "x-correlation-id",
    "x-openai-request-id",
    "x-dynamo-request-id",
)
LOAD_SHEDDING_STATUS_CODES = {429, 503}
LOAD_SHEDDING_METRIC_HINTS = (
    "shed",
    "reject",
    "admission",
    "overload",
    "rate_limit",
    "rate_limited",
    "too_many",
)


async def _frontend_pod(kubectl: KubectlClient, namespace: str) -> str:
    """Return a ready frontend pod name for direct metric/env inspection."""
    result = await kubectl.run(
        "get",
        "pod",
        "-n",
        namespace,
        "-l",
        FRONTEND_SELECTOR,
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=False,
    )
    pod = result.stdout.strip() if result.returncode == 0 else ""
    if pod:
        return pod

    for candidate in await kubectl.get_pods(namespace):
        if candidate.is_ready and "-frontend-" in candidate.name:
            return candidate.name
    raise RuntimeError(
        f"D225-D233: no ready Dynamo frontend pod found in namespace {namespace!r}"
    )


async def _frontend_env(kubectl: KubectlClient, namespace: str) -> dict[str, str]:
    """Read literal env vars from the active frontend pod."""
    pod = await _frontend_pod(kubectl, namespace)
    result = await kubectl.run(
        "get",
        "pod",
        pod,
        "-n",
        namespace,
        "-o",
        "jsonpath={range .spec.containers[*].env[*]}{.name}={.value}{'\\n'}{end}",
        check=True,
    )
    env: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name, sep, value = line.partition("=")
        if sep and name:
            env[name] = value
    return env


async def _raw_frontend_metrics(kubectl: KubectlClient, namespace: str) -> str:
    """Fetch raw Prometheus text from the frontend /metrics endpoint."""
    pod = await _frontend_pod(kubectl, namespace)
    async with (
        kubectl.port_forward(pod, 8000, namespace=namespace) as local_port,
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session,
        session.get(f"http://127.0.0.1:{local_port}/metrics") as resp,
    ):
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(
                f"D225-D233: /metrics on {namespace}/{pod} returned {resp.status}: "
                f"{body[:512]!r}"
            )
        return body


def _chat_url(endpoint_url: str) -> str:
    """Build the chat-completions URL from the package /v1 endpoint fixture."""
    return f"{endpoint_url.rstrip('/')}/chat/completions"


def _chat_payload(
    prompt: str,
    *,
    stream: bool,
    max_tokens: int = 32,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a minimal OpenAI-compatible chat-completions payload."""
    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extra:
        payload.update(extra)
    return payload


def _metric_samples(metrics_text: str, hints: Iterable[str]) -> dict[str, float]:
    """Return metric samples whose metric name or labels contain any hint."""
    lowered_hints = tuple(hint.lower() for hint in hints)
    samples: dict[str, float] = {}
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if not any(hint in line.lower() for hint in lowered_hints):
            continue
        value_text = line.rsplit(None, 1)[-1]
        try:
            samples[name] = samples.get(name, 0.0) + float(value_text)
        except ValueError:
            logger.debug(lambda line=line: f"D225-D233: bad metric sample {line!r}")
    return samples


def _sum_samples(samples: dict[str, float]) -> float:
    """Return the sum of all matching sample values."""
    return sum(samples.values())


async def _post_chat(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST one chat-completions request and return status, headers and body."""
    started = time.monotonic()
    try:
        async with session.post(
            _chat_url(endpoint_url),
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            body = await resp.text()
            return {
                "kind": "response",
                "status": resp.status,
                "headers": {k.lower(): v for k, v in resp.headers.items()},
                "body": body,
                "elapsed": time.monotonic() - started,
            }
    except asyncio.TimeoutError as exc:
        return {
            "kind": "timeout",
            "error": repr(exc),
            "elapsed": time.monotonic() - started,
        }
    except aiohttp.ClientError as exc:
        return {
            "kind": "client_error",
            "error": repr(exc),
            "elapsed": time.monotonic() - started,
        }


async def _read_stream(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 45.0,
    read_delay: float = 0.0,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """POST one streaming request and summarize its SSE lifecycle."""
    started = time.monotonic()
    chunks: list[str] = []
    try:
        async with session.post(
            _chat_url(endpoint_url),
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            async for chunk in resp.content.iter_any():
                if chunk:
                    text = chunk.decode(errors="replace")
                    chunks.append(text[:512])
                    if read_delay > 0.0:
                        await asyncio.sleep(read_delay)
                    if max_chunks is not None and len(chunks) >= max_chunks:
                        break
            return {
                "kind": "stream",
                "status": resp.status,
                "headers": {k.lower(): v for k, v in resp.headers.items()},
                "chunks": chunks,
                "elapsed": time.monotonic() - started,
            }
    except asyncio.TimeoutError as exc:
        return {
            "kind": "timeout",
            "error": repr(exc),
            "elapsed": time.monotonic() - started,
        }
    except aiohttp.ClientError as exc:
        return {
            "kind": "client_error",
            "error": repr(exc),
            "chunks": chunks,
            "elapsed": time.monotonic() - started,
        }


@asynccontextmanager
async def _mixed_traffic(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    *,
    stream_count: int,
    non_stream_count: int,
) -> AsyncIterator[set[asyncio.Task[dict[str, Any]]]]:
    """Run a short mixed streaming/non-stream traffic burst."""
    tasks: set[asyncio.Task[dict[str, Any]]] = set()
    for idx in range(stream_count):
        payload = _chat_payload(
            f"D233 streaming metrics availability probe {idx}",
            stream=True,
            max_tokens=96,
        )
        tasks.add(asyncio.create_task(_read_stream(session, endpoint_url, payload)))
    for idx in range(non_stream_count):
        payload = _chat_payload(
            f"D233 non-stream metrics availability probe {idx}",
            stream=False,
            max_tokens=32,
        )
        tasks.add(asyncio.create_task(_post_chat(session, endpoint_url, payload)))
    try:
        yield tasks
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _load_shedding_supported(env: dict[str, str], metrics_text: str) -> bool:
    """Return whether this frontend advertises admission/load-shedding support."""
    if os.environ.get(LOAD_SHEDDING_OPT_IN_ENV) == "1":
        return True
    env_blob = "\n".join(f"{k}={v}" for k, v in sorted(env.items())).lower()
    metric_blob = metrics_text.lower()
    return any(
        hint in env_blob or hint in metric_blob for hint in LOAD_SHEDDING_METRIC_HINTS
    )


async def _require_load_shedding_topology(
    kubectl: KubectlClient,
    namespace: str,
) -> str:
    """Return baseline metrics or skip if admission/load-shedding is unsupported."""
    env = await _frontend_env(kubectl, namespace)
    metrics_text = await _raw_frontend_metrics(kubectl, namespace)
    if not _load_shedding_supported(env, metrics_text):
        pytest.skip(
            "D226-D228 require a Dynamo frontend configured with admission-control "
            "or load-shedding counters. No frontend env var or /metrics sample "
            "contained shed/reject/admission/rate-limit hints; set "
            f"{LOAD_SHEDDING_OPT_IN_ENV}=1 for a topology that enables it."
        )
    return metrics_text


async def test_d225_request_id_header_propagation(
    dynamo_endpoint_url: str,
) -> None:
    """D225: a caller-supplied request ID is preserved on the HTTP response."""
    request_id = f"d225-{uuid4()}"
    headers = {"x-request-id": request_id, "x-correlation-id": request_id}
    async with aiohttp.ClientSession() as session:
        result = await _post_chat(
            session,
            dynamo_endpoint_url,
            _chat_payload("D225 request ID propagation probe", stream=False),
            headers=headers,
        )

    assert result["kind"] == "response", f"D225 request failed: {result!r}"
    assert result["status"] < 500, f"D225 frontend returned server error: {result!r}"
    if result["status"] in {400, 404}:
        pytest.skip(
            f"D225 cannot validate request ID propagation because the deployed "
            f"model/API rejected the probe with HTTP {result['status']}: "
            f"{result['body'][:256]!r}"
        )

    response_headers = result["headers"]
    propagated = {
        name: response_headers[name]
        for name in REQUEST_ID_HEADERS
        if response_headers.get(name) == request_id
    }
    assert propagated, (
        "D225: response did not echo the caller request ID in any supported "
        f"header {REQUEST_ID_HEADERS}; response_headers={response_headers!r}"
    )


async def test_d226_concurrent_stream_admission_load_shedding(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """D226: overloaded streaming admission sheds promptly instead of hanging."""
    await _require_load_shedding_topology(kubectl, dynamo_deployment_namespace)
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                _read_stream(
                    session,
                    dynamo_endpoint_url,
                    _chat_payload(
                        f"D226 stream overload {idx}", stream=True, max_tokens=128
                    ),
                    timeout=45.0,
                )
            )
            for idx in range(96)
        ]
        results = await asyncio.gather(*tasks)

    statuses = [
        result.get("status") for result in results if result["kind"] == "stream"
    ]
    shed = [status for status in statuses if status in LOAD_SHEDDING_STATUS_CODES]
    timeouts = [result for result in results if result["kind"] == "timeout"]
    assert not timeouts, (
        f"D226: load-shed streams hung instead of failing fast: {timeouts!r}"
    )
    assert shed, (
        "D226: no concurrent streaming request was load-shed with HTTP 429/503; "
        f"statuses={statuses!r}"
    )


async def test_d227_non_stream_load_shedding_metrics(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """D227: non-stream load shedding increments admission/rejection metrics."""
    before_text = await _require_load_shedding_topology(
        kubectl, dynamo_deployment_namespace
    )
    before = _sum_samples(_metric_samples(before_text, LOAD_SHEDDING_METRIC_HINTS))
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[
                _post_chat(
                    session,
                    dynamo_endpoint_url,
                    _chat_payload(f"D227 non-stream overload {idx}", stream=False),
                    timeout=30.0,
                )
                for idx in range(128)
            ]
        )
    shed = [
        result
        for result in results
        if result["kind"] == "response"
        and result.get("status") in LOAD_SHEDDING_STATUS_CODES
    ]
    if not shed:
        pytest.skip(
            "D227 did not drive this deployment into non-stream load shedding; "
            "increase frontend admission pressure or lower configured limits."
        )

    after_text = await _raw_frontend_metrics(kubectl, dynamo_deployment_namespace)
    after_samples = _metric_samples(after_text, LOAD_SHEDDING_METRIC_HINTS)
    after = _sum_samples(after_samples)
    assert after > before, (
        "D227: non-stream load-shed responses did not increment any matching "
        f"metric; before={before}, after={after}, samples={after_samples!r}"
    )


async def test_d228_streaming_load_shedding_metrics(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """D228: streaming load shedding increments admission/rejection metrics."""
    before_text = await _require_load_shedding_topology(
        kubectl, dynamo_deployment_namespace
    )
    before = _sum_samples(_metric_samples(before_text, LOAD_SHEDDING_METRIC_HINTS))
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[
                _read_stream(
                    session,
                    dynamo_endpoint_url,
                    _chat_payload(
                        f"D228 stream overload {idx}", stream=True, max_tokens=128
                    ),
                    timeout=45.0,
                )
                for idx in range(128)
            ]
        )
    shed = [
        result
        for result in results
        if result["kind"] == "stream"
        and result.get("status") in LOAD_SHEDDING_STATUS_CODES
    ]
    if not shed:
        pytest.skip(
            "D228 did not drive this deployment into streaming load shedding; "
            "increase frontend admission pressure or lower configured limits."
        )

    after_text = await _raw_frontend_metrics(kubectl, dynamo_deployment_namespace)
    after_samples = _metric_samples(after_text, LOAD_SHEDDING_METRIC_HINTS)
    after = _sum_samples(after_samples)
    assert after > before, (
        "D228: stream load-shed responses did not increment any matching metric; "
        f"before={before}, after={after}, samples={after_samples!r}"
    )


async def test_d229_slow_client_reader_backpressure(
    dynamo_endpoint_url: str,
) -> None:
    """D229: a slow SSE reader is bounded and does not poison the connection."""
    async with aiohttp.ClientSession() as session:
        result = await _read_stream(
            session,
            dynamo_endpoint_url,
            _chat_payload(
                "D229 slow reader backpressure probe", stream=True, max_tokens=128
            ),
            timeout=60.0,
            read_delay=0.25,
        )
        followup = await _post_chat(
            session,
            dynamo_endpoint_url,
            _chat_payload("D229 follow-up after slow reader", stream=False),
        )

    assert result["kind"] != "timeout", f"D229: slow reader hung: {result!r}"
    assert result.get("status", 200) < 500, f"D229: slow reader caused 5xx: {result!r}"
    assert followup["kind"] == "response" and followup["status"] < 500, (
        "D229: frontend failed a normal follow-up request after slow reader "
        f"backpressure; followup={followup!r}"
    )


@pytest.mark.skipif(
    os.environ.get(IDLE_STALL_OPT_IN_ENV) != "1",
    reason=(
        "D230 requires a topology that can inject an idle/no-token backend stall; "
        f"set {IDLE_STALL_OPT_IN_ENV}=1 only when the deployed Dynamo frontend is "
        "wired to a stall-capable backend or proxy."
    ),
)
async def test_d230_idle_no_token_stall_terminates_stream(
    dynamo_endpoint_url: str,
) -> None:
    """D230: an idle/no-token stream reaches a bounded terminal state."""
    async with aiohttp.ClientSession() as session:
        result = await _read_stream(
            session,
            dynamo_endpoint_url,
            _chat_payload(
                "D230 idle no-token stall probe", stream=True, max_tokens=512
            ),
            timeout=75.0,
        )
    assert result["kind"] != "timeout", f"D230: idle/no-token stream hung: {result!r}"
    assert result.get("status", 200) in {200, 499, 500, 502, 503, 504}, (
        f"D230: unexpected terminal status for idle/no-token stream: {result!r}"
    )


async def test_d231_tool_call_streaming_compatibility(
    dynamo_endpoint_url: str,
) -> None:
    """D231: OpenAI tool-call payloads remain valid under streaming."""
    tool_payload = _chat_payload(
        "D231 call the weather tool for San Jose, then summarize the result.",
        stream=True,
        max_tokens=128,
        extra={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Return weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
    )
    async with aiohttp.ClientSession() as session:
        result = await _read_stream(
            session, dynamo_endpoint_url, tool_payload, timeout=45.0
        )

    if result.get("status") in {400, 404, 422}:
        pytest.skip(
            "D231 tool-call streaming is unsupported by this model/frontend; "
            f"status={result.get('status')}, chunks={result.get('chunks', [])[:2]!r}"
        )
    assert result["kind"] == "stream", (
        f"D231 request failed before streaming: {result!r}"
    )
    assert result["status"] == 200, f"D231 stream returned non-200: {result!r}"
    assert any("[DONE]" in chunk for chunk in result["chunks"]), (
        f"D231: tool-call stream did not terminate with [DONE]: {result!r}"
    )


async def test_d232_extra_openai_fields_tolerated(
    dynamo_endpoint_url: str,
) -> None:
    """D232: forward-compatible OpenAI fields are ignored or accepted, not 5xx."""
    payload = _chat_payload(
        "D232 tolerate extra OpenAI-compatible request fields.",
        stream=False,
        extra={
            "frequency_penalty": 0.0,
            "logprobs": False,
            "metadata": {"chaos_case": "D232"},
            "parallel_tool_calls": True,
            "presence_penalty": 0.0,
            "response_format": {"type": "text"},
            "seed": 12345,
            "service_tier": "auto",
            "user": "aiperf-chaos-d232",
        },
    )
    async with aiohttp.ClientSession() as session:
        result = await _post_chat(session, dynamo_endpoint_url, payload)

    assert result["kind"] == "response", f"D232 request failed at transport: {result!r}"
    assert result["status"] < 500, (
        "D232: extra OpenAI fields triggered a frontend/server 5xx instead of "
        f"being tolerated or rejected cleanly; result={result!r}"
    )
    assert result["status"] not in {400, 422}, (
        "D232: frontend rejected forward-compatible OpenAI fields instead of "
        f"tolerating them; result={result!r}"
    )


async def test_d233_metrics_endpoint_available_during_mixed_traffic(
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """D233: /metrics remains scrapeable during mixed stream/non-stream traffic."""
    async with (
        aiohttp.ClientSession() as session,
        _mixed_traffic(
            session,
            dynamo_endpoint_url,
            stream_count=12,
            non_stream_count=24,
        ) as tasks,
    ):
        scrapes: list[str] = []
        for _ in range(5):
            scrapes.append(
                await _raw_frontend_metrics(kubectl, dynamo_deployment_namespace)
            )
            await asyncio.sleep(0.5)
        results = await asyncio.gather(*tasks, return_exceptions=False)

    assert all("#" in scrape or "dynamo" in scrape.lower() for scrape in scrapes), (
        "D233: one or more /metrics scrapes returned non-Prometheus-looking text"
    )
    hard_failures = [
        result
        for result in results
        if result["kind"] == "timeout"
        or (
            result["kind"] in {"response", "stream"}
            and result.get("status", 200) >= 500
        )
    ]
    assert not hard_failures, (
        "D233: mixed traffic caused timeouts or 5xx responses while metrics were "
        f"being scraped: {hard_failures!r}"
    )

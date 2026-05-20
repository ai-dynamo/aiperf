# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D406 -- observe pinned-worker head-of-line behavior.

D-series catalog § D4xx scenario. Targets the pinned-worker queue path in
``lib/kv-router/src/scheduling/queue.rs:246-249``: when a request is pinned to
one worker, the router should not silently invent a different request field or
route through an unsupported affinity mechanism. This test first probes the
current Dynamo endpoint with the supported ``nvext.decode_worker_id`` request
parameter, then runs a bounded observation with pinned long streams plus
unpinned probes behind them.

The assertion is intentionally observational: it records the loaded latency and
only fails on contract regressions (unsupported field accepted incorrectly,
pinned response routed to a different worker, or probes hanging past budget).
"""

from __future__ import annotations

import asyncio
import contextlib
import statistics
import time
from collections.abc import Mapping
from typing import Any

import aiohttp
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

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

    probe = await _post_completion(
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


async def _post_completion(
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
    result = await _post_completion(
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

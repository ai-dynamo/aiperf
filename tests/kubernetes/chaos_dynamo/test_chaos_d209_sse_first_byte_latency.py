# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D209 -- SSE first-byte latency remains bounded under frontend throttling.

The case fronts Dynamo's OpenAI-compatible frontend through the D-series
Toxiproxy ``frontend`` port and adds downstream latency. It verifies the client
observes delayed-but-bounded time to first SSE byte rather than a hang or an
incorrect non-stream response.
"""

from __future__ import annotations

import time

import aiohttp
import pytest

from tests.kubernetes.chaos.toxiproxy import ToxiproxyError
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_FRONTEND_PROXY = "d209-frontend-first-byte"
_FRONTEND_PROXY_PORT = 20011
_LATENCY_MS = 1200
_FIRST_BYTE_TIMEOUT_S = 12.0


async def _frontend_service_name(kubectl: KubectlClient, namespace: str) -> str | None:
    result = await kubectl.run(
        "get",
        "service",
        "-n",
        namespace,
        "-o",
        "jsonpath={.items[*].metadata.name}",
        check=False,
    )
    if result.returncode != 0:
        return None
    for name in result.stdout.split():
        if name.endswith("-frontend"):
            return name
    return None


async def _configure_frontend_proxy(request: pytest.FixtureRequest) -> None:
    """Create the frontend proxy and downstream latency toxic, or skip explicitly."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    service = await _frontend_service_name(kubectl, namespace)
    if service is None:
        pytest.skip(
            "D209 requires a Dynamo frontend Service ending in '-frontend'; "
            f"none was found in namespace {namespace!r}."
        )
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    try:
        await dynamo_toxiproxy.add_proxy(
            name=_FRONTEND_PROXY,
            listen=f"0.0.0.0:{_FRONTEND_PROXY_PORT}",
            upstream=f"{service}.{namespace}.svc.cluster.local:8000",
        )
    except ToxiproxyError as exc:
        pytest.skip(
            f"D209 requires the frontend Toxiproxy port; proxy setup failed: {exc}"
        )
    await dynamo_toxiproxy.add_toxic(
        _FRONTEND_PROXY,
        "latency",
        {"latency": _LATENCY_MS, "jitter": 0},
        stream="downstream",
        name="d209-first-byte-latency",
    )


async def test_d209_sse_first_byte_latency_under_frontend_throttling(
    request: pytest.FixtureRequest,
) -> None:
    """Inject frontend downstream latency and assert first SSE byte is bounded."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    try:
        await _configure_frontend_proxy(request)
        async with kubectl.port_forward(
            "service/toxiproxy",
            _FRONTEND_PROXY_PORT,
            namespace="chaos-toxiproxy",
        ) as local_port:
            url = f"http://127.0.0.1:{local_port}/v1"
            payload = {
                "model": "default",
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "max_tokens": 32,
                "stream": True,
                "temperature": 0.0,
            }
            started = time.monotonic()
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=_FIRST_BYTE_TIMEOUT_S)
                ) as session,
                session.post(f"{url}/chat/completions", json=payload) as resp,
            ):
                body = await resp.text() if resp.status != 200 else ""
                assert resp.status == 200, (
                    f"D209: expected HTTP 200, got {resp.status}: {body}"
                )
                first = await resp.content.read(1)
            elapsed = time.monotonic() - started
    finally:
        await dynamo_toxiproxy.reset()

    assert first, "D209: streaming response ended before any SSE byte was received"
    assert elapsed >= (_LATENCY_MS / 1000.0) * 0.75, (
        f"D209: first byte arrived in {elapsed:.3f}s despite {_LATENCY_MS}ms "
        "downstream latency; the request likely bypassed the frontend proxy"
    )
    assert elapsed < _FIRST_BYTE_TIMEOUT_S, (
        f"D209: first byte latency {elapsed:.3f}s exceeded bounded throttle budget "
        f"{_FIRST_BYTE_TIMEOUT_S}s"
    )

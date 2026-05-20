# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D210 -- SSE clients tolerate fragmented frames from the Dynamo frontend."""

from __future__ import annotations

import aiohttp
import orjson
import pytest

from tests.kubernetes.chaos.toxiproxy import ToxiproxyError
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_FRONTEND_PROXY = "d210-frontend-slicer"
_FRONTEND_PROXY_PORT = 20011


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


def _append_complete_sse_payloads(buffer: str, payloads: list[str]) -> str:
    """Append complete LF/CRLF-delimited SSE data payloads; return suffix."""
    while "\n\n" in buffer or "\r\n\r\n" in buffer:
        lf_pos = buffer.find("\n\n") if "\n\n" in buffer else len(buffer) + 1
        crlf_pos = buffer.find("\r\n\r\n") if "\r\n\r\n" in buffer else len(buffer) + 1
        delimiter = "\n\n" if lf_pos < crlf_pos else "\r\n\r\n"
        raw_event, buffer = buffer.split(delimiter, 1)
        for raw_line in raw_event.splitlines():
            line = raw_line.rstrip("\r")
            if line.startswith("data:"):
                payloads.append(line.removeprefix("data:").strip())
    return buffer


async def _configure_sliced_frontend_proxy(request: pytest.FixtureRequest) -> None:
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    service = await _frontend_service_name(kubectl, namespace)
    if service is None:
        pytest.skip(
            "D210 requires a Dynamo frontend Service ending in '-frontend'; "
            f"none was found in namespace {namespace!r}."
        )
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    try:
        await dynamo_toxiproxy.add_proxy(
            name=_FRONTEND_PROXY,
            listen=f"0.0.0.0:{_FRONTEND_PROXY_PORT}",
            upstream=f"{service}.{namespace}.svc.cluster.local:8000",
        )
        await dynamo_toxiproxy.add_toxic(
            _FRONTEND_PROXY,
            "slicer",
            {"average_size": 7, "size_variation": 2, "delay": 2},
            stream="downstream",
            name="d210-sse-slicer",
        )
    except ToxiproxyError as exc:
        pytest.skip(
            f"D210 requires a frontend Toxiproxy slicer toxic; setup failed: {exc}"
        )


async def test_d210_sse_fragmented_frames_parse_to_complete_chunks(
    request: pytest.FixtureRequest,
) -> None:
    """Slice frontend TCP bytes and assert complete OpenAI SSE frames are readable."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    dynamo_toxiproxy = request.getfixturevalue("dynamo_toxiproxy")
    payloads: list[str] = []
    try:
        await _configure_sliced_frontend_proxy(request)
        async with kubectl.port_forward(
            "service/toxiproxy",
            _FRONTEND_PROXY_PORT,
            namespace="chaos-toxiproxy",
        ) as local_port:
            url = f"http://127.0.0.1:{local_port}/v1/chat/completions"
            body = {
                "model": "default",
                "messages": [{"role": "user", "content": "Count from one to ten."}],
                "max_tokens": 64,
                "stream": True,
                "temperature": 0.0,
            }
            buffer = ""
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30.0)
                ) as session,
                session.post(url, json=body) as resp,
            ):
                error_body = await resp.text() if resp.status != 200 else ""
                assert resp.status == 200, (
                    f"D210: expected HTTP 200, got {resp.status}: {error_body}"
                )
                async for chunk in resp.content.iter_chunked(3):
                    buffer += chunk.decode("utf-8", errors="replace")
                    buffer = _append_complete_sse_payloads(buffer, payloads)
                    if payloads and payloads[-1] == "[DONE]":
                        break
    finally:
        await dynamo_toxiproxy.reset()

    assert payloads, (
        "D210: no complete SSE data payloads were parsed from sliced stream"
    )
    assert payloads[-1] == "[DONE]", (
        f"D210: stream did not end with [DONE]: {payloads!r}"
    )
    data_payloads = [payload for payload in payloads if payload != "[DONE]"]
    assert data_payloads, f"D210: stream contained no JSON data frames: {payloads!r}"
    for payload in data_payloads:
        decoded = orjson.loads(payload)
        assert isinstance(decoded, dict), (
            f"D210: SSE payload is not a JSON object: {payload!r}"
        )
        assert "choices" in decoded, f"D210: JSON chunk missing choices: {decoded!r}"

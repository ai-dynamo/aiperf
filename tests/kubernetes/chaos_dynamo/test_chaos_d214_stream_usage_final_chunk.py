# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D214 -- stream_options.include_usage emits a final usage-only SSE chunk."""

from __future__ import annotations

import aiohttp
import orjson
import pytest

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


def _append_sse_payloads(buffer: str, payloads: list[str]) -> str:
    """Append complete SSE ``data:`` payloads and return the incomplete suffix."""
    while "\n" in buffer:
        raw_line, buffer = buffer.split("\n", 1)
        line = raw_line.rstrip("\r")
        if line.startswith("data:"):
            payloads.append(line.removeprefix("data:").strip())
    return buffer


async def _collect_stream_payloads(endpoint_url: str) -> list[str]:
    request_body = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with a short sentence."}],
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    payloads: list[str] = []
    buffer = ""
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session,
        session.post(f"{endpoint_url}/chat/completions", json=request_body) as resp,
    ):
        body = await resp.text() if resp.status != 200 else ""
        assert resp.status == 200, f"D214: expected HTTP 200, got {resp.status}: {body}"
        async for chunk in resp.content.iter_any():
            buffer += chunk.decode("utf-8", errors="replace")
            buffer = _append_sse_payloads(buffer, payloads)
    return payloads


def _is_usage_chunk(payload: str) -> bool:
    if payload == "[DONE]":
        return False
    decoded = orjson.loads(payload)
    if not isinstance(decoded, dict):
        return False
    usage = decoded.get("usage")
    choices = decoded.get("choices")
    return isinstance(usage, dict) and choices == []


async def test_d214_stream_usage_final_chunk_precedes_done(
    dynamo_endpoint_url: str,
) -> None:
    """Request include_usage=true and assert the final usage-only chunk shape."""
    payloads = await _collect_stream_payloads(dynamo_endpoint_url)
    assert payloads, "D214: no SSE payloads received"
    assert payloads[-1] == "[DONE]", (
        f"D214: stream did not terminate with [DONE]: {payloads!r}"
    )
    assert len(payloads) >= 2, (
        f"D214: stream too short to contain usage chunk: {payloads!r}"
    )
    assert _is_usage_chunk(payloads[-2]), (
        "D214: expected usage-only chunk immediately before [DONE] when "
        f"include_usage=true; observed tail={payloads[-3:]!r}"
    )

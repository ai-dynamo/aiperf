# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D215 -- stream_options.include_usage=false suppresses usage-only chunks."""

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


def _is_usage_chunk(payload: str) -> bool:
    if payload == "[DONE]":
        return False
    decoded = orjson.loads(payload)
    return (
        isinstance(decoded, dict)
        and isinstance(decoded.get("usage"), dict)
        and decoded.get("choices") == []
    )


async def test_d215_include_usage_false_has_no_usage_only_sse_chunk(
    dynamo_endpoint_url: str,
) -> None:
    """Request include_usage=false and assert no usage-only SSE chunk appears."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with a short sentence."}],
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": False},
        "temperature": 0.0,
    }
    payloads: list[str] = []
    buffer = ""
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session,
        session.post(f"{dynamo_endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.text() if resp.status != 200 else ""
        assert resp.status == 200, f"D215: expected HTTP 200, got {resp.status}: {body}"
        async for chunk in resp.content.iter_any():
            buffer += chunk.decode("utf-8", errors="replace")
            buffer = _append_sse_payloads(buffer, payloads)

    assert payloads, "D215: no SSE payloads received"
    assert payloads[-1] == "[DONE]", (
        f"D215: stream did not terminate with [DONE]: {payloads!r}"
    )
    usage_chunks = [item for item in payloads if _is_usage_chunk(item)]
    assert not usage_chunks, (
        "D215: include_usage=false still emitted usage-only chunks: "
        f"{usage_chunks!r}; full stream={payloads!r}"
    )

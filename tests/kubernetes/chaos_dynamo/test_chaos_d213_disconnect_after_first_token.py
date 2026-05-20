# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D213 -- client disconnect after first streamed token is cleaned up."""

from __future__ import annotations

import aiohttp
import pytest

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def _read_first_data_payload(resp: aiohttp.ClientResponse) -> str | None:
    """Return the first non-DONE SSE data payload from ``resp``."""
    buffer = ""
    async for chunk in resp.content.iter_any():
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            raw_line, buffer = buffer.split("\n", 1)
            line = raw_line.rstrip("\r")
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload and payload != "[DONE]":
                return payload
    return None


async def _assert_followup_chat_succeeds(endpoint_url: str) -> None:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "max_tokens": 8,
        "stream": False,
        "temperature": 0.0,
    }
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.text()
    assert resp.status == 200, (
        f"D213: follow-up request returned HTTP {resp.status}: {body}"
    )


async def test_d213_disconnect_after_first_token_is_clean(
    dynamo_endpoint_url: str,
) -> None:
    """Read one SSE data frame, close the socket, then verify fresh traffic works."""
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": "Write three short sentences about streaming clients.",
            }
        ],
        "max_tokens": 128,
        "stream": True,
        "temperature": 0.0,
    }
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20.0))
    try:
        resp = await session.post(
            f"{dynamo_endpoint_url}/chat/completions", json=payload
        )
        try:
            assert resp.status == 200, (
                f"D213: streaming setup returned HTTP {resp.status}: {await resp.text()}"
            )
            first_payload = await _read_first_data_payload(resp)
            assert first_payload is not None, (
                "D213: stream ended before first SSE data payload"
            )
            resp.close()
        finally:
            resp.release()
    finally:
        await session.close()

    await _assert_followup_chat_succeeds(dynamo_endpoint_url)

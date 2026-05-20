# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D212 -- client disconnect before first streamed token does not poison frontend."""

from __future__ import annotations

import aiohttp
import pytest

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def _assert_followup_chat_succeeds(endpoint_url: str, case_id: str) -> None:
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
        f"{case_id}: follow-up request returned HTTP {resp.status}: {body}"
    )


async def test_d212_disconnect_before_first_token_is_clean(
    dynamo_endpoint_url: str,
) -> None:
    """Open a stream and drop the socket before reading any SSE body bytes."""
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": "Write a long paragraph about graceful streaming cancellation.",
            }
        ],
        "max_tokens": 256,
        "stream": True,
        "temperature": 0.0,
    }
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15.0))
    try:
        resp = await session.post(
            f"{dynamo_endpoint_url}/chat/completions", json=payload
        )
        try:
            assert resp.status == 200, (
                f"D212: streaming setup returned HTTP {resp.status}: {await resp.text()}"
            )
            resp.close()
        finally:
            resp.release()
    finally:
        await session.close()

    await _assert_followup_chat_succeeds(dynamo_endpoint_url, "D212")

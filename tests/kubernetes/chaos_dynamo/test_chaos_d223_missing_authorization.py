# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D223 -- missing Authorization header compatibility."""

from __future__ import annotations

from dataclasses import dataclass

import aiohttp
import orjson
import pytest

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


@dataclass(frozen=True)
class _HTTPJSON:
    status: int
    body: dict[str, object]
    text: str


async def _post_chat_without_auth(
    session: aiohttp.ClientSession,
    url: str,
) -> _HTTPJSON:
    """POST a valid chat completion without an Authorization header."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with pong."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _HTTPJSON(status=resp.status, body=body, text=text)


def _assert_chat_completion_shape(result: _HTTPJSON) -> None:
    """Assert a minimal OpenAI chat-completion response envelope."""
    assert result.status == 200, (
        f"D223: missing Authorization should remain compatible; "
        f"HTTP {result.status} body={result.text!r}"
    )
    choices = result.body.get("choices")
    assert isinstance(choices, list) and choices, (
        f"D223: response missing non-empty choices; body={result.body!r}"
    )
    first_choice = choices[0]
    assert isinstance(first_choice, dict), (
        f"D223: first choice is not an object; body={result.body!r}"
    )
    assert isinstance(first_choice.get("message"), dict), (
        f"D223: first choice missing message object; body={result.body!r}"
    )


async def test_d223_missing_authorization_compatibility(
    dynamo_endpoint_url: str,
) -> None:
    """Dynamo local/frontend compatibility should not require Authorization."""
    url = f"{dynamo_endpoint_url.rstrip('/')}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _post_chat_without_auth(session, url)

    if result.status in {401, 403}:
        pytest.skip(
            "D223: this Dynamo deployment has auth enforcement enabled "
            f"(HTTP {result.status})"
        )
    _assert_chat_completion_shape(result)

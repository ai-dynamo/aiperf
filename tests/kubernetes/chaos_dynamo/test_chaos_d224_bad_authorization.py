# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D224 -- bad Authorization header compatibility."""

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


async def _post_chat_with_bad_auth(
    session: aiohttp.ClientSession,
    url: str,
) -> _HTTPJSON:
    """POST a valid chat completion with an intentionally bad bearer token."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with pong."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    async with session.post(
        url,
        json=payload,
        headers={"authorization": "Bearer definitely-not-a-real-token"},
    ) as resp:
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
        f"D224: bad Authorization should remain compatible; "
        f"HTTP {result.status} body={result.text!r}"
    )
    choices = result.body.get("choices")
    assert isinstance(choices, list) and choices, (
        f"D224: response missing non-empty choices; body={result.body!r}"
    )
    first_choice = choices[0]
    assert isinstance(first_choice, dict), (
        f"D224: first choice is not an object; body={result.body!r}"
    )
    assert isinstance(first_choice.get("message"), dict), (
        f"D224: first choice missing message object; body={result.body!r}"
    )


async def test_d224_bad_authorization_compatibility(dynamo_endpoint_url: str) -> None:
    """Dynamo local/frontend compatibility should ignore a bad bearer token."""
    url = f"{dynamo_endpoint_url.rstrip('/')}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _post_chat_with_bad_auth(session, url)

    if result.status in {401, 403}:
        pytest.skip(
            "D224: this Dynamo deployment has auth enforcement enabled "
            f"(HTTP {result.status})"
        )
    _assert_chat_completion_shape(result)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D219 -- OpenAI legacy Completions API compatibility probe."""

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


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
) -> _HTTPJSON:
    """POST JSON and decode an object response when present."""
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _HTTPJSON(status=resp.status, body=body, text=text)


def _assert_completions_shape(result: _HTTPJSON) -> None:
    """Assert a minimal OpenAI legacy completion response envelope."""
    assert result.status == 200, (
        f"D219: /completions returned HTTP {result.status}; body={result.text!r}"
    )
    assert isinstance(result.body.get("id"), str), (
        f"D219: response missing string id; body={result.body!r}"
    )
    choices = result.body.get("choices")
    assert isinstance(choices, list) and choices, (
        f"D219: response missing non-empty choices; body={result.body!r}"
    )
    first_choice = choices[0]
    assert isinstance(first_choice, dict), (
        f"D219: first choice is not an object; body={result.body!r}"
    )
    assert isinstance(first_choice.get("text"), str), (
        f"D219: first choice missing text string; body={result.body!r}"
    )


async def test_d219_completions_api_compatibility_probe(
    dynamo_endpoint_url: str,
) -> None:
    """Probe /completions and skip explicitly when this Dynamo build lacks it."""
    payload: dict[str, object] = {
        "model": "default",
        "prompt": "Reply with the word pong.",
        "max_tokens": 8,
        "temperature": 0.0,
    }
    url = f"{dynamo_endpoint_url.rstrip('/')}/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _post_json(session, url, payload)

    if result.status in {404, 405, 501}:
        pytest.skip(
            "D219: Dynamo deployment does not expose legacy Completions API "
            f"(HTTP {result.status})"
        )
    assert result.status < 500, (
        f"D219: /completions must not fail with server error; "
        f"HTTP {result.status} body={result.text!r}"
    )
    _assert_completions_shape(result)

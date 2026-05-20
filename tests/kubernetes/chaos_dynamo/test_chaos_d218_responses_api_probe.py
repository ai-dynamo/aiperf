# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D218 -- OpenAI Responses API compatibility probe."""

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


def _assert_responses_shape(result: _HTTPJSON) -> None:
    """Assert a minimal Responses API success envelope."""
    assert result.status == 200, (
        f"D218: /responses returned HTTP {result.status}; body={result.text!r}"
    )
    assert isinstance(result.body.get("id"), str), (
        f"D218: response missing string id; body={result.body!r}"
    )
    assert result.body.get("object") == "response", (
        f"D218: response object must be 'response'; body={result.body!r}"
    )
    assert "output" in result.body or "output_text" in result.body, (
        f"D218: Responses API success missing output/output_text; body={result.body!r}"
    )


async def test_d218_responses_api_compatibility_probe(
    dynamo_endpoint_url: str,
) -> None:
    """Probe /responses and skip explicitly when this Dynamo build lacks it."""
    payload: dict[str, object] = {
        "model": "default",
        "input": "Reply with the word pong.",
        "max_output_tokens": 8,
        "temperature": 0.0,
    }
    url = f"{dynamo_endpoint_url.rstrip('/')}/responses"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _post_json(session, url, payload)

    if result.status in {404, 405, 501}:
        pytest.skip(
            "D218: Dynamo deployment does not expose Responses API "
            f"(HTTP {result.status})"
        )
    assert result.status < 500, (
        f"D218: /responses must not fail with server error; "
        f"HTTP {result.status} body={result.text!r}"
    )
    _assert_responses_shape(result)

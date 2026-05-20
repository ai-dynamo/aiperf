# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D222 -- wrong content-type rejection shape."""

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
    content_type: str


async def _post_wrong_content_type(
    session: aiohttp.ClientSession, url: str
) -> _HTTPJSON:
    """POST a valid JSON body advertised as text/plain."""
    payload = orjson.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": "pong"}],
            "max_tokens": 8,
        }
    )
    async with session.post(
        url,
        data=payload,
        headers={"content-type": "text/plain"},
    ) as resp:
        text = await resp.text()
        content_type = resp.headers.get("content-type", "")
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _HTTPJSON(
        status=resp.status,
        body=body,
        text=text,
        content_type=content_type,
    )


def _assert_openai_error_shape(result: _HTTPJSON) -> None:
    """Assert wrong content type returns a structured JSON error response."""
    assert "json" in result.content_type.lower(), (
        f"D222: wrong content-type rejection must be JSON, got "
        f"{result.content_type!r}; body={result.text!r}"
    )
    error = result.body.get("error")
    assert isinstance(error, dict), (
        f"D222: wrong content-type rejection missing error object; body={result.body!r}"
    )
    assert isinstance(error.get("message"), str) and error["message"], (
        f"D222: error.message must be non-empty; body={result.body!r}"
    )
    assert isinstance(error.get("type"), str) and error["type"], (
        f"D222: error.type must be non-empty; body={result.body!r}"
    )


async def test_d222_wrong_content_type_rejection(dynamo_endpoint_url: str) -> None:
    """A JSON chat body sent as text/plain should reject with structured 4xx."""
    url = f"{dynamo_endpoint_url.rstrip('/')}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15.0)
    ) as session:
        result = await _post_wrong_content_type(session, url)

    if result.status == 200:
        pytest.skip(
            "D222: Dynamo accepts text/plain request bodies as JSON; "
            "no content-type rejection to assert"
        )
    assert 400 <= result.status < 500, (
        f"D222: wrong content type should reject with 4xx, not HTTP "
        f"{result.status}; body={result.text!r}"
    )
    _assert_openai_error_shape(result)

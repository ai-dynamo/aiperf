# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D221 -- malformed JSON body rejection shape."""

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


async def _post_malformed_json(session: aiohttp.ClientSession, url: str) -> _HTTPJSON:
    """POST a truncated JSON body with an application/json content type."""
    async with session.post(
        url,
        data=b'{"model":"default","messages":[',
        headers={"content-type": "application/json"},
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
    """Assert malformed JSON returns a structured JSON error response."""
    assert "json" in result.content_type.lower(), (
        f"D221: malformed JSON rejection must be JSON, got "
        f"{result.content_type!r}; body={result.text!r}"
    )
    error = result.body.get("error")
    assert isinstance(error, dict), (
        f"D221: malformed JSON rejection missing error object; body={result.body!r}"
    )
    assert isinstance(error.get("message"), str) and error["message"], (
        f"D221: error.message must be non-empty; body={result.body!r}"
    )
    assert isinstance(error.get("type"), str) and error["type"], (
        f"D221: error.type must be non-empty; body={result.body!r}"
    )


async def test_d221_malformed_json_body_rejection(dynamo_endpoint_url: str) -> None:
    """Malformed JSON should fail with 4xx and an OpenAI-style error object."""
    url = f"{dynamo_endpoint_url.rstrip('/')}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15.0)
    ) as session:
        result = await _post_malformed_json(session, url)

    assert 400 <= result.status < 500, (
        f"D221: malformed JSON should reject with 4xx, not HTTP "
        f"{result.status}; body={result.text!r}"
    )
    _assert_openai_error_shape(result)

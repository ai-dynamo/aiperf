# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D220 -- embeddings endpoint rejection shape."""

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


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
) -> _HTTPJSON:
    """POST JSON and decode an object response when present."""
    async with session.post(url, json=payload) as resp:
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


def _assert_openai_error_shape(case_id: str, result: _HTTPJSON) -> None:
    """Assert a JSON OpenAI-compatible error envelope."""
    assert "json" in result.content_type.lower(), (
        f"{case_id}: rejection must be JSON, got {result.content_type!r}; "
        f"body={result.text!r}"
    )
    error = result.body.get("error")
    assert isinstance(error, dict), (
        f"{case_id}: rejection missing error object; body={result.body!r}"
    )
    assert isinstance(error.get("message"), str) and error["message"], (
        f"{case_id}: error.message must be non-empty; body={result.body!r}"
    )
    assert isinstance(error.get("type"), str) and error["type"], (
        f"{case_id}: error.type must be non-empty; body={result.body!r}"
    )


async def test_d220_embeddings_rejection_shape(dynamo_endpoint_url: str) -> None:
    """Unsupported embeddings requests should return structured 4xx JSON."""
    payload: dict[str, object] = {
        "model": "default",
        "input": "embedding probe",
    }
    url = f"{dynamo_endpoint_url.rstrip('/')}/embeddings"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        result = await _post_json(session, url, payload)

    if result.status == 200:
        pytest.skip(
            "D220: Dynamo deployment supports embeddings; no rejection to assert"
        )
    assert 400 <= result.status < 500, (
        f"D220: unsupported embeddings should reject with 4xx, not "
        f"HTTP {result.status}; body={result.text!r}"
    )
    _assert_openai_error_shape("D220", result)

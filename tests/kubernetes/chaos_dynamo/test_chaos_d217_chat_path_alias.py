# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D217 -- chat path alias compatibility probe."""

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


def _service_root(endpoint_url: str) -> str:
    """Return the Dynamo service root regardless of whether fixture includes /v1."""
    trimmed = endpoint_url.rstrip("/")
    return trimmed.removesuffix("/v1")


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, object],
) -> _HTTPJSON:
    """POST JSON and decode the response body for assertion diagnostics."""
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return _HTTPJSON(status=resp.status, body=body, text=text)


def _assert_chat_completion_shape(case_id: str, result: _HTTPJSON) -> None:
    """Assert a minimal OpenAI chat-completion response envelope."""
    assert result.status == 200, (
        f"{case_id}: chat endpoint returned HTTP {result.status}; body={result.text!r}"
    )
    assert isinstance(result.body.get("id"), str), (
        f"{case_id}: response missing string id; body={result.body!r}"
    )
    choices = result.body.get("choices")
    assert isinstance(choices, list) and choices, (
        f"{case_id}: response missing non-empty choices; body={result.body!r}"
    )
    first_choice = choices[0]
    assert isinstance(first_choice, dict), (
        f"{case_id}: first choice is not an object; body={result.body!r}"
    )
    assert isinstance(first_choice.get("message"), dict), (
        f"{case_id}: first choice missing message object; body={result.body!r}"
    )


async def test_d217_chat_path_alias_compatibility(dynamo_endpoint_url: str) -> None:
    """Both /v1/chat/completions and /chat/completions should serve chat."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with the word pong."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    root = _service_root(dynamo_endpoint_url)
    canonical_url = f"{root}/v1/chat/completions"
    alias_url = f"{root}/chat/completions"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30.0)
    ) as session:
        canonical = await _post_json(session, canonical_url, payload)
        alias = await _post_json(session, alias_url, payload)

    _assert_chat_completion_shape("D217 canonical", canonical)
    if alias.status in {404, 405, 501}:
        pytest.skip(
            "D217: Dynamo deployment does not expose /chat/completions alias "
            f"(HTTP {alias.status}); canonical /v1 path passed"
        )
    _assert_chat_completion_shape("D217 alias", alias)

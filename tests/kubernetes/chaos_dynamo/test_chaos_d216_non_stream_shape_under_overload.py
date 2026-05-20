# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D216 -- non-stream OpenAI responses keep JSON shape under overload."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp
import orjson
import pytest

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_OVERLOAD_OPT_IN_ENV = "AIPERF_DYNAMO_OVERLOAD_CHAOS"
_CONCURRENCY = 32


async def _post_non_stream(endpoint_url: str, idx: int) -> dict[str, Any]:
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": f"Reply with one concise sentence for overload request {idx}.",
            }
        ],
        "max_tokens": 32,
        "stream": False,
        "temperature": 0.0,
    }
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45.0)) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body = await resp.read()
    try:
        decoded = orjson.loads(body)
    except orjson.JSONDecodeError:
        decoded = body.decode(errors="replace")[:512]
    return {"status": resp.status, "body": decoded}


def _assert_openai_non_stream_shape(result: dict[str, Any]) -> None:
    status = result["status"]
    body = result["body"]
    assert isinstance(body, dict), (
        f"D216: non-stream overload response was not JSON object: status={status}, body={body!r}"
    )
    if status == 200:
        assert body.get("object") == "chat.completion", (
            f"D216: success response has wrong OpenAI object shape: {body!r}"
        )
        choices = body.get("choices")
        assert isinstance(choices, list) and choices, (
            f"D216: success response missing non-empty choices: {body!r}"
        )
        return
    assert status in {429, 500, 503, 504}, (
        f"D216: overload response used unexpected HTTP status {status}: {body!r}"
    )
    error = body.get("error")
    assert isinstance(error, dict), (
        f"D216: overload error response missing OpenAI error object: {body!r}"
    )
    assert isinstance(error.get("message"), str) and error["message"], (
        f"D216: overload error missing non-empty error.message: {body!r}"
    )


@pytest.mark.skipif(
    os.environ.get(_OVERLOAD_OPT_IN_ENV) != "1",
    reason=(
        "D216 requires an overload topology or externally tuned Dynamo deployment; "
        f"set {_OVERLOAD_OPT_IN_ENV}=1 only when concurrency is expected to trigger "
        "queueing/throttling without destabilizing the shared test cluster."
    ),
)
async def test_d216_non_stream_shape_under_overload(
    dynamo_endpoint_url: str,
) -> None:
    """Fan out non-stream requests and require every terminal body to be JSON-shaped."""
    results = await asyncio.gather(
        *(_post_non_stream(dynamo_endpoint_url, idx) for idx in range(_CONCURRENCY))
    )
    for result in results:
        _assert_openai_non_stream_shape(result)
    if not any(item["status"] != 200 for item in results):
        pytest.skip(
            "D216 overload prerequisite did not trigger any non-200 response; "
            f"statuses={[item['status'] for item in results]!r}. Increase load or use an "
            "overload-tuned Dynamo topology before treating this case as covered."
        )

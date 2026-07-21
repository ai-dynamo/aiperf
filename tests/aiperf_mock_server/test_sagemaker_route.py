# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire-contract tests for the mock SageMaker InvokeEndpoint routes.

Proves the non-streaming ``/endpoints/{name}/invocations`` route serves an
OpenAI-chat-shaped body, and that the streaming
``/endpoints/{name}/invocations-response-stream`` route's raw bytes round-trip
through the *production* eventstream decoder
(:class:`aiperf.transports.sagemaker_eventstream.EventStreamReader`), not just
a structurally-similar look-alike.
"""

from collections.abc import AsyncIterator

import pytest
from aiperf.transports.sagemaker_eventstream import (
    EVENTSTREAM_CONTENT_TYPE,
    EventStreamReader,
)
from aiperf_mock_server.app import asgi_app
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.asyncio

_ENDPOINT = "my-endpoint"


def _body(**overrides):
    body = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hello SageMaker"}],
    }
    body.update(overrides)
    return body


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=asgi_app), base_url="http://mock"
    ) as c:
        yield c


async def test_invocations_returns_openai_chat_shape(client):
    resp = await client.post(f"/endpoints/{_ENDPOINT}/invocations", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "mock-model"
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"]
    assert "usage" in data


async def test_response_stream_content_type_is_eventstream(client):
    resp = await client.post(
        f"/endpoints/{_ENDPOINT}/invocations-response-stream",
        json=_body(stream=True),
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith(EVENTSTREAM_CONTENT_TYPE)


async def test_response_stream_decodes_with_production_reader(client):
    resp = await client.post(
        f"/endpoints/{_ENDPOINT}/invocations-response-stream",
        json=_body(stream=True),
    )
    assert resp.status_code == 200
    raw = resp.content

    async def _one_chunk() -> AsyncIterator[bytes]:
        yield raw

    messages = [m async for m in EventStreamReader(_one_chunk()).__aiter__()]

    # Each streamed SSE chunk becomes one PayloadPart line -> one message.
    assert messages, "no eventstream messages decoded"

    content = ""
    saw_done = False
    for msg in messages:
        if msg.line == "[DONE]":
            saw_done = True
            continue
        data = msg.get_json()
        for choice in data.get("choices", []):
            content += choice.get("delta", {}).get("content", "") or ""

    assert saw_done, "stream did not terminate with [DONE]"
    assert content, "no content reassembled from decoded PayloadPart chunks"

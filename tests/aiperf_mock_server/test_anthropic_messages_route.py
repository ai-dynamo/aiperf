# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire-contract tests for the mock /v1/messages route.

Shapes are validated against real api.anthropic.com captures (raw exports and
proxy traces): full-key usage in message_start, cumulative usage in
message_delta (split shape behind config), Anthropic stop_reason vocabulary,
required max_tokens, required anthropic-version header, disjoint cache
accounting, and chunked input_json_delta fragments.
"""

import orjson
import pytest
from aiperf_mock_server import utils as mock_utils
from tests.aiperf_mock_server.app import asgi_app
from tests.aiperf_mock_server.config import server_config
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.asyncio

HEADERS = {"anthropic-version": "2023-06-01"}
USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
}


def _body(**overrides):
    body = {
        "model": "mock-model",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Hello from the contract tests"}],
    }
    body.update(overrides)
    return body


def _sse_events(text: str) -> list[dict]:
    events = []
    for chunk in text.split("\n\n"):
        for line in chunk.split("\n"):
            if line.startswith("data: "):
                events.append(orjson.loads(line[6:]))
    return events


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=asgi_app), base_url="http://mock"
    ) as c:
        yield c


@pytest.fixture(autouse=True)
def _fresh_cache_and_config():
    mock_utils.reset_anthropic_prefix_cache()
    original = server_config.anthropic_split_usage
    yield
    server_config.anthropic_split_usage = original


async def test_missing_anthropic_version_header_rejected(client):
    resp = await client.post("/v1/messages", json=_body())
    assert resp.status_code == 400
    err = resp.json()
    assert err["type"] == "error"
    assert err["error"]["type"] == "invalid_request_error"
    assert "anthropic-version" in err["error"]["message"]


async def test_missing_max_tokens_rejected(client):
    body = _body()
    del body["max_tokens"]
    resp = await client.post("/v1/messages", json=body, headers=HEADERS)
    assert resp.status_code == 422


async def test_non_streaming_usage_shape_and_stop_reason(client):
    resp = await client.post("/v1/messages", json=_body(), headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "message"
    assert set(data["usage"]) == USAGE_KEYS
    assert data["usage"]["cache_read_input_tokens"] == 0
    assert data["usage"]["cache_creation_input_tokens"] == 0
    # OpenAI-style finish reasons must not leak onto the Anthropic wire.
    assert data["stop_reason"] in {"end_turn", "max_tokens"}


async def test_max_tokens_cap_maps_to_max_tokens_stop_reason(client):
    resp = await client.post("/v1/messages", json=_body(max_tokens=1), headers=HEADERS)
    assert resp.json()["stop_reason"] == "max_tokens"


async def test_streaming_message_start_carries_full_usage_keys(client):
    resp = await client.post("/v1/messages", json=_body(stream=True), headers=HEADERS)
    events = _sse_events(resp.text)
    start = next(e for e in events if e["type"] == "message_start")
    usage = start["message"]["usage"]
    assert set(usage) == USAGE_KEYS
    assert usage["output_tokens"] == 1  # real API reports 1, not 0


async def test_streaming_message_delta_usage_cumulative_by_default(client):
    resp = await client.post("/v1/messages", json=_body(stream=True), headers=HEADERS)
    events = _sse_events(resp.text)
    start = next(e for e in events if e["type"] == "message_start")
    delta = next(e for e in events if e["type"] == "message_delta")
    # Modern shape: message_delta repeats the full cumulative key set.
    assert set(delta["usage"]) == USAGE_KEYS
    assert delta["usage"]["input_tokens"] == start["message"]["usage"]["input_tokens"]
    assert delta["usage"]["output_tokens"] >= 1
    assert delta["delta"]["stop_reason"] in {"end_turn", "max_tokens"}


async def test_streaming_split_usage_config_reverts_to_docs_shape(client):
    server_config.anthropic_split_usage = True
    resp = await client.post("/v1/messages", json=_body(stream=True), headers=HEADERS)
    events = _sse_events(resp.text)
    delta = next(e for e in events if e["type"] == "message_delta")
    assert set(delta["usage"]) == {"output_tokens"}


async def test_cache_simulation_write_then_read_ladder(client):
    long_content = "repeat this exact prefix " * 40
    body = _body(
        cache_control={"type": "ephemeral"},
        messages=[{"role": "user", "content": long_content}],
    )

    first = (await client.post("/v1/messages", json=body, headers=HEADERS)).json()
    u1 = first["usage"]
    total = (
        u1["input_tokens"]
        + u1["cache_read_input_tokens"]
        + u1["cache_creation_input_tokens"]
    )
    # Cold cache: everything is written, nothing read, disjoint sum holds.
    assert u1["cache_read_input_tokens"] == 0
    assert u1["cache_creation_input_tokens"] == total

    second = (await client.post("/v1/messages", json=body, headers=HEADERS)).json()
    u2 = second["usage"]
    # Warm cache: the identical prompt is fully served from cache.
    assert u2["cache_read_input_tokens"] == total
    assert u2["cache_creation_input_tokens"] == 0

    # Extended history: the old prefix reads, the new tail writes.
    extended = _body(
        cache_control={"type": "ephemeral"},
        messages=[
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "and now something new"},
        ],
    )
    third = (await client.post("/v1/messages", json=extended, headers=HEADERS)).json()
    u3 = third["usage"]
    assert u3["cache_read_input_tokens"] > 0
    assert u3["cache_creation_input_tokens"] > 0
    total3 = (
        u3["input_tokens"]
        + u3["cache_read_input_tokens"]
        + u3["cache_creation_input_tokens"]
    )
    assert total3 >= total


async def test_no_cache_fields_without_opt_in(client):
    resp = await client.post("/v1/messages", json=_body(), headers=HEADERS)
    usage = resp.json()["usage"]
    # No cache_control -> no-cache identity: full prompt is uncached input.
    assert usage["cache_read_input_tokens"] == 0
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["input_tokens"] > 0


async def test_streaming_tool_use_chunks_input_json_delta(client):
    body = _body(
        stream=True,
        tools=[
            {
                "name": "lookup_row",
                "description": "Look up a row",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )
    resp = await client.post("/v1/messages", json=body, headers=HEADERS)
    events = _sse_events(resp.text)
    fragments = [
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta"
        and e["delta"].get("type") == "input_json_delta"
    ]
    assert fragments, "tool_use requested but no input_json_delta emitted"
    reassembled = orjson.loads("".join(fragments))
    assert reassembled == {"arg": "value"}
    delta = next(e for e in events if e["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"


async def test_streaming_thinking_budget_emits_thinking_and_signature_deltas(client):
    body = _body(
        stream=True,
        thinking={"type": "enabled", "budget_tokens": 8},
    )
    resp = await client.post("/v1/messages", json=body, headers=HEADERS)
    events = _sse_events(resp.text)
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "thinking"
    thinking_fragments = [
        e["delta"]["thinking"]
        for e in events
        if e["type"] == "content_block_delta"
        and e["delta"].get("type") == "thinking_delta"
    ]
    assert thinking_fragments, "thinking budget requested but no thinking_delta"
    signature_fragments = [
        e["delta"]["signature"]
        for e in events
        if e["type"] == "content_block_delta"
        and e["delta"].get("type") == "signature_delta"
    ]
    assert signature_fragments, "thinking block closed without a signature_delta"
    text_fragments = [
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"].get("type") == "text_delta"
    ]
    assert text_fragments, "thinking must leave budget for the text block"


async def test_non_streaming_thinking_budget_emits_thinking_block(client):
    body = _body(thinking={"type": "enabled", "budget_tokens": 8})
    resp = await client.post("/v1/messages", json=body, headers=HEADERS)
    payload = resp.json()
    block_types = [block["type"] for block in payload["content"]]
    assert block_types[0] == "thinking"
    assert "text" in block_types
    thinking_block = payload["content"][0]
    assert thinking_block["thinking"]


async def test_thinking_disabled_or_absent_emits_no_thinking(client):
    for body in (_body(), _body(thinking={"type": "disabled"})):
        resp = await client.post("/v1/messages", json=body, headers=HEADERS)
        payload = resp.json()
        block_types = [block["type"] for block in payload["content"]]
        assert "thinking" not in block_types

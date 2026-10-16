# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.common.models import RequestRecord, Text, TextResponse, Turn
from aiperf.endpoints.anthropic_messages import MessagesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "content",
    [
        param([{"type": "redacted_thinking", "data": "opaque-data"}], id="redacted-only"),
        param([
            {"type": "thinking", "thinking": "visible thinking", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque-data"},
            {"type": "text", "text": "I will use a tool."},
            {"type": "tool_use", "id": "tool-1", "name": "search", "input": {}},
        ], id="thinking-and-tool"),
    ],
)  # fmt: skip
def test_assistant_replay_preserves_redacted_thinking(
    streaming: bool,
    content: list[dict[str, Any]],
) -> None:
    model_endpoint = create_model_endpoint(EndpointType.MESSAGES, streaming=streaming)
    endpoint = MessagesEndpoint(model_endpoint=model_endpoint)
    if streaming:
        events = []
        for index, block in enumerate(content):
            if block["type"] == "text":
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    }
                )
            else:
                events.append(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": block,
                    }
                )
    else:
        events = [{"type": "message", "content": content}]
    record = RequestRecord(
        responses=[
            TextResponse(
                perf_ns=index,
                text=orjson.dumps(event).decode(),
                content_type="application/json",
            )
            for index, event in enumerate(events, start=1)
        ],
        start_perf_ns=0,
        end_perf_ns=len(events) + 1,
    )

    assistant_turn = endpoint.build_assistant_turn(record)

    assert assistant_turn is not None
    request_info = create_request_info(
        model_endpoint=model_endpoint,
        turns=[assistant_turn, Turn(texts=[Text(contents=["Continue"])])],
    )
    payload = endpoint.format_payload(request_info)
    assert payload["messages"][0] == {"role": "assistant", "content": content}
    assert "opaque-data" not in endpoint.extract_payload_inputs(payload).texts

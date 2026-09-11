# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end HTTP integration tests for Mooncake message_mode='delta' replay.

Runs the real aiperf CLI (subprocess, real HTTP transport) against an in-test
recording server that streams SSE chat chunks with tool_calls deltas split
across chunks. The test inspects the ACTUAL second HTTP request body received
by the server to prove it contains the newly generated assistant text AND the
reassembled streamed tool_calls (id/name/arguments), followed by the dataset's
tool-result delta whose ``tool_call_id`` matches the live-generated call id.
"""

import socket
from pathlib import Path
from typing import Any

import orjson
import pytest
from aiohttp import web

from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults

LIVE_TOOL_CALL_ID = "call_live_0001"
ASSISTANT_TEXT_TURN_1 = "Checking the weather now."
ASSISTANT_TEXT_TURN_2 = "It is 21C in Paris."

SYSTEM_MSG = {"role": "system", "content": "You are a weather assistant."}
USER_MSG = {"role": "user", "content": "What's the weather in Paris?"}
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]
TOOL_RESULT_MSG = {
    "role": "tool",
    "tool_call_id": LIVE_TOOL_CALL_ID,
    "content": "21C, sunny",
}


def _sse(data: dict[str, Any]) -> bytes:
    return b"data: " + orjson.dumps(data) + b"\n\n"


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
    return _sse(
        {
            "id": "chatcmpl-recording",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "recording-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


class RecordingToolCallServer:
    """Localhost SSE chat server that records every request body it receives.

    The first request is answered with assistant text plus a tool call whose
    ``id``/``function.name``/``function.arguments`` are deliberately streamed
    across separate SSE chunks (matching OpenAI's streaming shape), so the
    client must reassemble them. Later requests get a plain text reply.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._runner: web.AppRunner | None = None
        self.port: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle_chat)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _handle_chat(self, request: web.Request) -> web.StreamResponse:
        body = orjson.loads(await request.read())
        self.requests.append(body)
        request_index = len(self.requests)

        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)

        if request_index == 1:
            await response.write(
                _chunk({"role": "assistant", "content": ASSISTANT_TEXT_TURN_1})
            )
            # Streamed tool_calls deltas: id + name first, then the JSON
            # arguments split across two fragments.
            await response.write(
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": LIVE_TOOL_CALL_ID,
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    }
                )
            )
            await response.write(
                _chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"city": '}}
                        ]
                    }
                )
            )
            await response.write(
                _chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"Paris"}'}}
                        ]
                    },
                    finish_reason="tool_calls",
                )
            )
        else:
            await response.write(
                _chunk({"role": "assistant", "content": ASSISTANT_TEXT_TURN_2})
            )
            await response.write(_chunk({}, finish_reason="stop"))

        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


def _write_trace(path: Path, records: list[dict[str, Any]]) -> Path:
    trace_file = path / "delta_trace.jsonl"
    with open(trace_file, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record))
            f.write(b"\n")
    return trace_file


def _token_cap(payload: dict[str, Any]) -> int | None:
    return payload.get("max_completion_tokens", payload.get("max_tokens"))


@pytest.mark.integration
@pytest.mark.asyncio
class TestMooncakeMessageDeltaHTTP:
    async def test_second_http_request_contains_live_assistant_and_tool_calls(
        self, cli: AIPerfCLI, tmp_path: Path
    ):
        """The actual second HTTP request must carry the live assistant turn.

        Turn 1 sends the initial history; the server streams text + a tool
        call. Turn 2's dataset delta is only the tool-result message, so
        everything between the initial history and the tool result in request
        2 must have been captured live from the streamed response - including
        the reassembled tool_call id/name/arguments. The tool-result fixture's
        ``tool_call_id`` matches the id the live server emits; no ordinal
        remapping is performed by AIPerf.
        """
        records = [
            {"session_id": "weather-1", "message_mode": "delta", "timestamp": 0, "messages": [SYSTEM_MSG, USER_MSG], "tools": TOOL_DEFS, "output_length": 64},
            {"session_id": "weather-1", "message_mode": "delta", "delay": 100, "messages": [TOOL_RESULT_MSG], "output_length": 32, "extra": {"temperature": 0.25}},
        ]  # fmt: skip
        trace_file = _write_trace(tmp_path, records)

        server = RecordingToolCallServer()
        await server.start()
        try:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {server.url} \
                    --endpoint-type chat \
                    --streaming \
                    --input-file {trace_file} \
                    --custom-dataset-type mooncake_trace \
                    --fixed-schedule \
                    --workers-max 1 \
                    --ui {defaults.ui}
                """,
                timeout=120.0,
            )
        finally:
            await server.stop()

        assert result.request_count == 2
        assert len(server.requests) == 2
        first_request, second_request = server.requests

        # Request 1: exactly the authored initial history + tool schemas.
        assert first_request["messages"] == [SYSTEM_MSG, USER_MSG]
        assert first_request["tools"] == TOOL_DEFS
        assert _token_cap(first_request) == 64

        # Request 2: initial history, then the LIVE assistant message with
        # the reassembled streamed tool_call, then the tool-result delta.
        messages = second_request["messages"]
        assert messages[:2] == [SYSTEM_MSG, USER_MSG]

        assistant_msg = messages[2]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == ASSISTANT_TEXT_TURN_1
        assert assistant_msg["tool_calls"] == [
            {
                "id": LIVE_TOOL_CALL_ID,
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
            }
        ]

        # The dataset's tool-result delta references the SAME id the live
        # server generated - proving id continuity, not positional mapping.
        assert messages[3] == TOOL_RESULT_MSG
        assert messages[3]["tool_call_id"] == assistant_msg["tool_calls"][0]["id"]
        assert messages[3:] == [TOOL_RESULT_MSG]

        # Tool schemas inherited from turn 1; per-turn cap/extra from turn 2.
        assert second_request["tools"] == TOOL_DEFS
        assert _token_cap(second_request) == 32
        assert second_request["temperature"] == 0.25

    async def test_third_turn_history_includes_post_tool_assistant_text(
        self, cli: AIPerfCLI, tmp_path: Path
    ):
        """Context keeps growing: turn 3 sees both live assistant replies."""
        follow_up = {"role": "user", "content": "Thanks! And in London?"}
        records = [
            {"session_id": "weather-1", "message_mode": "delta", "timestamp": 0, "messages": [SYSTEM_MSG, USER_MSG], "tools": TOOL_DEFS, "output_length": 64},
            {"session_id": "weather-1", "message_mode": "delta", "delay": 50, "messages": [TOOL_RESULT_MSG], "output_length": 32},
            {"session_id": "weather-1", "message_mode": "delta", "delay": 50, "messages": [follow_up], "output_length": 32},
        ]  # fmt: skip
        trace_file = _write_trace(tmp_path, records)

        server = RecordingToolCallServer()
        await server.start()
        try:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {server.url} \
                    --endpoint-type chat \
                    --streaming \
                    --input-file {trace_file} \
                    --custom-dataset-type mooncake_trace \
                    --fixed-schedule \
                    --workers-max 1 \
                    --ui {defaults.ui}
                """,
                timeout=120.0,
            )
        finally:
            await server.stop()

        assert result.request_count == 3
        assert len(server.requests) == 3
        third_request = server.requests[2]
        messages = third_request["messages"]

        # [system, user, live assistant+tool_call, tool result,
        #  live assistant text, follow-up user delta]
        assert messages[:2] == [SYSTEM_MSG, USER_MSG]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["tool_calls"][0]["id"] == LIVE_TOOL_CALL_ID
        assert messages[3] == TOOL_RESULT_MSG
        assert messages[4] == {
            "role": "assistant",
            "content": ASSISTANT_TEXT_TURN_2,
        }
        assert messages[5:] == [follow_up]

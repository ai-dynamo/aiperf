# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration coverage for Mooncake message_mode='delta' replay.

Verifies the request payloads actually dispatched to the (fake) transport:
the second request in a delta session must contain the initial history, the
LIVE assistant reply captured from the first response, and the second turn's
delta messages - plus per-turn cap/extra semantics and tool schema
inheritance/replacement.
"""

from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.common.models import RawRecordInfo, SSEMessage
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}
USER_MSG_1 = {"role": "user", "content": "What is machine learning?"}
USER_MSG_2 = {"role": "user", "content": "Give an example."}
TOOLS_A = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
TOOLS_B = [{"type": "function", "function": {"name": "get_forecast", "parameters": {}}}]


def _write_trace(path: Path, records: list[dict]) -> Path:
    trace_file = path / "trace.jsonl"
    with open(trace_file, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record))
            f.write(b"\n")
    return trace_file


def _streamed_assistant_text(record: RawRecordInfo) -> str:
    """Reassemble the assistant text the (fake) server streamed for a record.

    Mirrors the capture semantics of ``BaseEndpoint.build_assistant_turn``:
    concatenated ``delta.content``, falling back to ``delta.reasoning_content``
    when the server streamed everything as reasoning.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for response in record.responses:
        if not isinstance(response, SSEMessage):
            continue
        data = response.extract_data_content()
        if not data or data.strip() == "[DONE]":
            continue
        chunk = orjson.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
        if isinstance(delta.get("reasoning_content"), str):
            reasoning_parts.append(delta["reasoning_content"])
    return "".join(content_parts) or "".join(reasoning_parts)


def _token_cap(payload: dict[str, Any]) -> int | None:
    return payload.get("max_completion_tokens", payload.get("max_tokens"))


def _run_delta_profile(
    cli: AIPerfCLI, trace_file: Path, extra_args: str = ""
) -> list[RawRecordInfo]:
    result = cli.run_sync(
        f"""
        aiperf profile \
            --model {defaults.model} \
            --custom-dataset-type mooncake_trace \
            --input-file {trace_file} \
            --streaming \
            --fixed-schedule \
            --workers-max 1 \
            --export-level raw \
            --ui {defaults.ui} {extra_args}
        """,
        timeout=60.0,
    )
    assert result.raw_records is not None
    return sorted(result.raw_records, key=lambda r: r.metadata.turn_index)


@pytest.mark.component_integration
class TestMooncakeMessageDeltaReplay:
    def test_delta_second_request_contains_live_assistant_reply(
        self, cli: AIPerfCLI, tmp_path: Path
    ) -> None:
        records = [
            {"session_id": "s1", "message_mode": "delta", "timestamp": 0, "messages": [SYSTEM_MSG, USER_MSG_1], "tools": TOOLS_A, "output_length": 8, "extra": {"temperature": 0.1}},
            {"session_id": "s1", "message_mode": "delta", "delay": 5, "messages": [USER_MSG_2], "output_length": 6, "extra": {"temperature": 0.5}},
        ]  # fmt: skip
        trace_file = _write_trace(tmp_path, records)

        raw = _run_delta_profile(cli, trace_file)
        assert len(raw) == 2
        first, second = raw

        # First request: exactly the authored initial history.
        assert first.payload["messages"] == [SYSTEM_MSG, USER_MSG_1]
        assert first.payload["tools"] == TOOLS_A
        assert _token_cap(first.payload) == 8
        assert first.payload["temperature"] == 0.1

        # Second request: initial history + LIVE assistant reply + delta.
        live_text = _streamed_assistant_text(first)
        assert live_text, "fake server streamed no assistant text"
        messages = second.payload["messages"]
        assert messages[:2] == [SYSTEM_MSG, USER_MSG_1]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == live_text
        assert messages[3:] == [USER_MSG_2]

        # Tool schema inherited from the first turn; per-turn cap/extra
        # scoped to the dispatching turn.
        assert second.payload["tools"] == TOOLS_A
        assert _token_cap(second.payload) == 6
        assert second.payload["temperature"] == 0.5
        assert "messages" not in (records[1].get("extra") or {})

    def test_delta_turn_tools_replacement(self, cli: AIPerfCLI, tmp_path: Path) -> None:
        records = [
            {"session_id": "s1", "message_mode": "delta", "timestamp": 0, "messages": [USER_MSG_1], "tools": TOOLS_A, "output_length": 8},
            {"session_id": "s1", "message_mode": "delta", "delay": 5, "messages": [USER_MSG_2], "tools": TOOLS_B, "output_length": 6},
        ]  # fmt: skip
        trace_file = _write_trace(tmp_path, records)

        raw = _run_delta_profile(cli, trace_file)
        assert len(raw) == 2
        assert raw[0].payload["tools"] == TOOLS_A
        assert raw[1].payload["tools"] == TOOLS_B

    def test_history_mode_still_replays_verbatim_without_live_responses(
        self, cli: AIPerfCLI, tmp_path: Path
    ) -> None:
        """Backward compatibility: default history mode is untouched."""
        turn2_history = [
            USER_MSG_1,
            {"role": "assistant", "content": "canned"},
            USER_MSG_2,
        ]
        records = [
            {"session_id": "s1", "timestamp": 0, "messages": [USER_MSG_1], "output_length": 8},
            {"session_id": "s1", "delay": 5, "messages": turn2_history, "output_length": 6},
        ]  # fmt: skip
        trace_file = _write_trace(tmp_path, records)

        raw = _run_delta_profile(cli, trace_file)
        assert len(raw) == 2
        assert raw[0].payload["messages"] == [USER_MSG_1]
        # Verbatim replay: the canned assistant turn is used, not the live one.
        assert raw[1].payload["messages"] == turn2_history

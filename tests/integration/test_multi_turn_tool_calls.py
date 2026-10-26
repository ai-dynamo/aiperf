# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_multi_turn_tool_messages_reach_chat_requests(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    tmp_path: Path,
    streaming: bool,
) -> None:
    tool_calls = [
        {
            "id": "call_read",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{ "path": "main.py" }'},
        }
    ]
    turns = [
        {"role": "user", "text": "Read main.py"},
        {"role": "assistant", "text": None, "tool_calls": tool_calls},
        {"role": "tool", "text": "print('hello')", "tool_call_id": "call_read"},
        {"role": "user", "text": "Summarize the file"},
    ]
    input_file = tmp_path / "tools.jsonl"
    input_file.write_bytes(
        orjson.dumps({"session_id": "tools", "turns": turns}) + b"\n"
    )

    result = await cli.run(
        f"""
        aiperf profile \
            --model {defaults.model} \
            --tokenizer builtin \
            --url {aiperf_mock_server.url} \
            --endpoint-type chat \
            --input-file {input_file} \
            --custom-dataset-type multi_turn \
            --request-count 4 \
            --concurrency 1 \
            --workers-max 1 \
            --osl 4 \
            --export-level raw \
            --no-gpu-telemetry \
            --no-server-metrics \
            --ui none \
            {"--streaming" if streaming else ""}
        """,
        timeout=120.0,
    )

    assert result.exit_code == 0
    assert result.request_count == 4, result.stdout + (result.log or "")[-5000:]
    assert result.raw_records is not None
    payloads = sorted(
        (record.payload for record in result.raw_records),
        key=lambda payload: len(payload["messages"]),
    )
    assert len(payloads) == 4
    expected = [
        {"role": "user", "content": "Read main.py"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "content": "print('hello')", "tool_call_id": "call_read"},
        {"role": "user", "content": "Summarize the file"},
    ]
    for index, payload in enumerate(payloads):
        messages = payload["messages"]
        assert len(messages) == 2 * index + 1
        assert messages[::2] == expected[: index + 1]
        assert all(
            message["role"] == "assistant" and set(message) == {"role", "content"}
            for message in messages[1::2]
        )
        assert payload["stream"] is streaming

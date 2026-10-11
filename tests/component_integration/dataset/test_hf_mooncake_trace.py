# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the actual requests and stop limits for complete recorded trajectories."""

from typing import Any

import pytest

from aiperf.dataset.loader.hf_mooncake_trace import HFMooncakeTraceDatasetLoader
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.fixture
def trajectory_rows() -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": "Recorded system"},
        {"role": "user", "content": "List files"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Recorded reasoning",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "Recorded files"},
        {"role": "assistant", "content": "Recorded answer"},
        {"role": "user", "content": "Explain them"},
        {"role": "assistant", "content": "Recorded final answer"},
    ]
    return [
        {
            "question_id": f"trajectory-{i}",
            "model": "source/model",
            "benchmark": "test-benchmark",
            "turns": messages[:end],
            "extras": {
                "tools": [{"type": "function", "function": {"name": "bash"}}],
                "max_tokens": 8,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            },
        }
        for i, end in enumerate([7, 5])
    ]


@pytest.mark.component_integration
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("request_limit", [False, True])
def test_trajectory_dispatch_preserves_recorded_history(
    cli: AIPerfCLI,
    monkeypatch: pytest.MonkeyPatch,
    trajectory_rows: list[dict[str, Any]],
    streaming: bool,
    request_limit: bool,
) -> None:
    monkeypatch.setattr(
        HFMooncakeTraceDatasetLoader, "_load_hf_dataset", lambda self: trajectory_rows
    )
    limits = (
        "--request-count 4 --concurrency 1"
        if request_limit
        else "--num-sessions 2 --concurrency 2"
    )
    result = cli.run_sync(
        f"""aiperf profile
            --model {defaults.model} --endpoint-type chat
            --hf-mooncake-dataset example/trajectories
            --mooncake-trace-to-hf-mapping session_id=question_id
            --mooncake-trace-to-hf-mapping messages=turns
            --mooncake-trace-to-hf-mapping tools=extras.tools
            --mooncake-trace-to-hf-mapping output_length=extras.max_tokens
            --mooncake-trace-to-hf-mapping extra.temperature=extras.temperature
            --mooncake-trace-to-hf-mapping extra.top_p=extras.top_p
            --mooncake-trace-to-hf-mapping extra.top_k=extras.top_k
            --dataset-filter model=source/model
            --dataset-filter benchmark=test-benchmark
            {limits} {"--streaming" if streaming else ""}
            --workers-max 1 --export-level raw --ui {defaults.ui}
        """,
        timeout=60.0,
    )
    assert result.request_count == (4 if request_limit else 5)
    assert result.raw_records is not None
    assert len(result.raw_records) == (4 if request_limit else 5)
    session_turns: dict[str, list[int]] = {}
    for record in result.raw_records:
        idx = record.metadata.turn_index
        session_turns.setdefault(record.metadata.x_correlation_id, []).append(idx)
        payload = record.payload
        assert payload["messages"] == trajectory_rows[0]["turns"][: [2, 4, 6][idx]]
        assert payload["tools"] == trajectory_rows[0]["extras"]["tools"]
        assert payload["model"] == defaults.model
        assert payload.get("max_completion_tokens", payload.get("max_tokens")) == 8
        for key, value in {"temperature": 0.7, "top_p": 0.8, "top_k": 20}.items():
            assert payload[key] == value
    assert sorted(sorted(turns) for turns in session_turns.values()) == (
        [[0], [0, 1, 2]] if request_limit else [[0, 1], [0, 1, 2]]
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the actual requests and stop limits for complete recorded trajectories."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiperf.common.environment import Environment
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
            "extra": {"presence_penalty": 0.5}
            if i == 0
            else {"frequency_penalty": 0.25},
            "turns": [
                messages[0] | {"content": f"Recorded system {i}"},
                *messages[1:end],
            ],
            "extras": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash" if i == 0 else "read_file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command" if i == 0 else "path": {"type": "string"},
                                },
                            },
                        },
                    }
                ],
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
    tmp_path: Path,
    trajectory_rows: list[dict[str, Any]],
    streaming: bool,
    request_limit: bool,
) -> None:
    dataset_dir = tmp_path / "recorded trajectories"
    dataset_dir.mkdir()
    pq.write_table(pa.Table.from_pylist(trajectory_rows), dataset_dir / "test.parquet")
    limits = (
        "--request-count 4 --concurrency 1"
        if request_limit
        else "--num-sessions 2 --concurrency 2"
    )
    result = cli.run_sync(
        f"""aiperf profile
            --model {defaults.model} --endpoint-type chat
            --hf-mooncake-dataset "{dataset_dir}"
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
    sources = {row["question_id"]: row for row in trajectory_rows}
    assert len(result.raw_records) == (4 if request_limit else 5)
    session_turns: dict[str, list[int]] = {}
    for record in result.raw_records:
        idx = record.metadata.turn_index
        session_turns.setdefault(record.metadata.x_correlation_id, []).append(idx)
        payload = record.payload
        source = sources[record.metadata.conversation_id]
        assert payload["messages"] == source["turns"][: [2, 4, 6][idx]]
        assert payload["tools"] == source["extras"]["tools"]
        assert payload["model"] == defaults.model
        assert payload.get("max_completion_tokens", payload.get("max_tokens")) == 8
        for key, value in {"temperature": 0.7, "top_p": 0.8, "top_k": 20}.items():
            assert payload[key] == value
        assert {
            key: payload[key]
            for key in ("presence_penalty", "frequency_penalty")
            if key in payload
        } == source["extra"]
    assert sorted(sorted(turns) for turns in session_turns.values()) == (
        [[0], [0, 1, 2]] if request_limit else [[0, 1], [0, 1, 2]]
    )


@pytest.mark.component_integration
def test_protected_extras_rejected_before_and_after_cache_population(
    cli: AIPerfCLI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", True)
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_DIR", tmp_path / "cache")
    dataset_dir = tmp_path / "trajectories"
    dataset_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Recorded answer"},
                    ]
                },
            ]
        ),
        dataset_dir / "test.parquet",
    )
    command = f"""aiperf profile
        --model {defaults.model} --endpoint-type chat
        --hf-mooncake-dataset "{dataset_dir}"
        --request-count 1 --concurrency 1 --random-seed 42
        --workers-max 1 --ui {defaults.ui}
    """
    for warm_cache in (False, True):
        if warm_cache:
            cli.run_sync(command, timeout=60.0)
            cached = cli.run_sync(command, timeout=60.0)
            assert "dataset cache HIT" in cached.log
        for extra in ("model:other-model", '{"tools": []}'):
            result = cli.run_sync(
                f"{command} --extra-inputs '{extra}'",
                timeout=60.0,
                assert_success=False,
            )
            assert result.exit_code != 0
            assert "must not override" in result.stdout + result.stderr

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and dynamo-trace builders for graph component-integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.common.environment import Environment
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli

# Fixed seed so every graph lowering in this package is byte-reproducible.
DYNAMO_SEED = 1234

# The trie lowering only accepts hash counts consistent with the recorded block
# size, so every builder derives input_length as 16 * len(hashes).
DYNAMO_BLOCK_SIZE = 16


@pytest.fixture
def mmap_base_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect MMAP_BASE_PATH to tmp_path so stores land in a known dir."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)
    return tmp_path


def dynamo_request_end(
    *,
    ts: int,
    session_id: str,
    hashes: list[int] | None = None,
    parent_session_id: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int = 16,
) -> dict[str, Any]:
    """One ``dynamo.request.trace.v1`` ``request_end`` with recorded replay hashes."""
    # Block alignment gate: (n-1)*16 < input_length <= n*16 for n hash ids.
    input_length = (
        input_tokens
        if input_tokens is not None
        else (DYNAMO_BLOCK_SIZE * len(hashes) if hashes else 32)
    )
    ctx: dict[str, Any] = {"session_id": session_id}
    if parent_session_id is not None:
        ctx["parent_session_id"] = parent_session_id
    req: dict[str, Any] = {
        "request_id": f"{session_id}-{ts}",
        "model": "m",
        "input_tokens": input_length,
        "output_tokens": output_tokens,
        "cached_tokens": 0,
    }
    if hashes is not None:
        req["replay"] = {
            "trace_block_size": DYNAMO_BLOCK_SIZE,
            "input_length": input_length,
            "input_sequence_hashes": hashes,
        }
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": ctx,
        "request": req,
    }


def dynamo_tool_event(
    *, ts: int, session_id: str, event_type: str, tool_call_id: str
) -> dict[str, Any]:
    """One ``tool_start``/``tool_end`` event on a session's tool_breakdown metadata."""
    tool: dict[str, Any] = {"tool_call_id": tool_call_id, "tool_class": "search"}
    if event_type == "tool_end":
        tool["duration_ms"] = 40.0
        tool["status"] = "succeeded"
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": event_type,
        "event_time_unix_ms": ts,
        "event_source": "harness",
        "agent_context": {"session_id": session_id},
        "tool": tool,
    }


def write_dynamo_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write dynamo trace records as newline-delimited JSON and return the path."""
    with path.open("wb") as f:
        for record in records:
            f.write(orjson.dumps(record))
            f.write(b"\n")
    return path


def dynamo_run(fixture: Path, *, seed: int = DYNAMO_SEED):
    """A resolved non-streaming chat BenchmarkRun over a dynamo trace fixture."""
    return make_run_from_cli(
        CLIConfig(
            model_names=["m"],
            endpoint_type=EndpointType.CHAT,
            streaming=False,
            url="http://localhost:8000",
            input_file=str(fixture),
            random_seed=seed,
        )
    )

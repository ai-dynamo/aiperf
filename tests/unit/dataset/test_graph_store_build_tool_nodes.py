# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tool-carrying graph through the REAL production build plane.

Every other tool-execution test stands one layer up: it builds a ``ParsedGraph``
by hand or constructs a ``TraceExecutor`` directly. That skips
``resolve_graph_parse_context`` -> ``GraphStoreBuilder.build``, which is where
the whole-graph sweeps live -- so an LlmNode-only assumption in the prefix-cache
extraction or the sidecar strip crashes only in production. These tests run the
real chain with ``--graph-execute-tools`` on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aiperf.common.environment import Environment
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.codecs import decode_graph_meta_sidecar
from aiperf.dataset.graph.models import LlmNode, ToolNode
from aiperf.dataset.graph.store_build import GraphStoreBuilder
from tests.unit.conftest import make_run_from_cli

BASE_TS = 1_700_000_000.0


def _model_call(event_id: int, step: int, start: float, dur: float) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "model_call",
        "timestamp": start + dur,
        "step": step,
        "duration_ns": int(dur * 1e9),
        "provider_request": {
            "messages": [{"role": "user", "content": f"turn {step}"}],
            "model": "openai/m",
        },
        "response_message": {
            "role": "assistant",
            "extra": {
                "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            },
        },
    }


def _tool_call(
    event_id: int, step: int, start: float, dur: float, command: str
) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "tool_call",
        "timestamp": start + dur,
        "step": step,
        "action_index": 0,
        "duration_ns": int(dur * 1e9),
        "action": {"command": command},
        "output": {"output": "", "returncode": 0},
    }


@pytest.fixture
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the store root to tmp_path so build artifacts land in a known dir."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)
    return tmp_path


@pytest.fixture
def recording_path(tmp_path: Path) -> Path:
    """One Agent Trace Replay recording with a tool step between two model calls."""
    recording = {
        "format": "mini-swe-agent-recording-1.0",
        "metadata": {"instance_id": "task_demo"},
        "events": [
            {"id": 0, "type": "run_start", "timestamp": BASE_TS},
            _model_call(1, 0, BASE_TS, 2.0),
            _tool_call(2, 0, BASE_TS + 2.0, 0.1, "mkdir -p src"),
            _model_call(3, 1, BASE_TS + 2.5, 1.0),
            {"id": 9, "type": "run_end", "timestamp": BASE_TS + 10},
        ],
    }
    path = tmp_path / "demo-recording.json"
    path.write_text(json.dumps(recording))
    return path


def _run(recording_path: Path, *, execute_tools: bool):
    return make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(recording_path),
            tokenizer_name="builtin",
            graph_execute_tools=execute_tools,
            open_loop_replay=False,
        )
    )


@pytest.mark.asyncio
async def test_build_with_execute_tools_completes_and_keeps_tool_nodes(
    store_root: Path, recording_path: Path
) -> None:
    """`--graph-execute-tools` survives the real build; the ToolNode reaches the sidecar."""
    run = _run(recording_path, execute_tools=True)

    result = await GraphStoreBuilder(run).build(recording_path)

    assert result.facet.trace_ids == ["task_demo"]
    assert result.sidecar_path.exists()
    parsed, _, _ = decode_graph_meta_sidecar(result.sidecar_path.read_bytes())
    graph = parsed.graphs["task_demo"]
    tools = {nid: n for nid, n in graph.nodes.items() if isinstance(n, ToolNode)}
    assert len(tools) == 1
    assert next(iter(tools.values())).commands == ["mkdir -p src"]


@pytest.mark.asyncio
async def test_prefix_cache_facet_skips_tool_nodes(
    store_root: Path, recording_path: Path
) -> None:
    """The prefix-cache facet reads an LlmNode-only field, so a ToolNode must be skipped, not crash."""
    run = _run(recording_path, execute_tools=True)

    result = await GraphStoreBuilder(run).build(recording_path)

    stamped = result.facet.prefix_cache_by_trace.get("task_demo", {})
    parsed, _, _ = decode_graph_meta_sidecar(result.sidecar_path.read_bytes())
    llm_ids = {
        nid
        for nid, n in parsed.graphs["task_demo"].nodes.items()
        if isinstance(n, LlmNode)
    }
    assert set(stamped) <= llm_ids


@pytest.mark.asyncio
async def test_tool_nodes_do_not_change_the_llm_node_build(
    store_root: Path, recording_path: Path
) -> None:
    """Turning tool execution ON adds a ToolNode without perturbing the LLM manifest."""
    with_tools = await GraphStoreBuilder(
        _run(recording_path, execute_tools=True)
    ).build(recording_path)
    without = await GraphStoreBuilder(_run(recording_path, execute_tools=False)).build(
        recording_path
    )

    def _llm_ids(result) -> set[str]:
        parsed, _, _ = decode_graph_meta_sidecar(result.sidecar_path.read_bytes())
        return {
            nid
            for nid, n in parsed.graphs["task_demo"].nodes.items()
            if isinstance(n, LlmNode)
        }

    assert _llm_ids(with_tools) == _llm_ids(without)

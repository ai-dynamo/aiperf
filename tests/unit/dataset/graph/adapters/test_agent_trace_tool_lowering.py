# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering recorded tool_call events into executable ToolNodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.mini_swe_agent_trace.recording import (
    AgentTraceRecordingAdapter,
    from_mini_swe_agent_trace,
)
from aiperf.dataset.graph.models import LlmNode, ToolNode
from aiperf.dataset.graph.parse_context import GraphParseContext

BASE_TS = 1_700_000_000.0


def _model_call(
    event_id: int, step: int, start: float, dur: float, msgs: list[dict]
) -> dict:
    return {
        "id": event_id,
        "type": "model_call",
        "timestamp": start + dur,
        "step": step,
        "duration_ns": int(dur * 1e9),
        "provider_request": {"messages": msgs, "model": "openai/m"},
        "response_message": {
            "role": "assistant",
            "extra": {
                "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            },
        },
    }


def _tool_call(
    event_id: int,
    step: int,
    start: float,
    dur: float,
    command: str,
    error: dict | None = None,
) -> dict:
    event = {
        "id": event_id,
        "type": "tool_call",
        "timestamp": start + dur,
        "step": step,
        "action_index": 0,
        "duration_ns": int(dur * 1e9),
        "action": {"command": command},
        "output": {"output": "", "returncode": 0},
    }
    if error is not None:
        event["error"] = error
    return event


# The real corpus shape for a terminal submit command: the command RAN, and the
# agent then raised Submitted on inspecting its output.
SUBMITTED_ERROR = {
    "type": "Submitted",
    "message": "",
    "messages": [{"role": "exit", "exit_status": "Submitted"}],
}


@pytest.fixture
def recording_path(tmp_path: Path) -> Path:
    """Two model calls with TWO batched tool calls between them."""
    recording: dict[str, Any] = {
        "format": "mini-swe-agent-recording-1.0",
        "metadata": {"instance_id": "task_demo"},
        "events": [
            {"id": 0, "type": "run_start", "timestamp": BASE_TS},
            _model_call(1, 0, BASE_TS, 2.0, [{"role": "user", "content": "a"}]),
            _tool_call(2, 0, BASE_TS + 2.0, 0.1, "mkdir -p src"),
            _tool_call(3, 0, BASE_TS + 2.1, 0.1, "echo hi > src/x"),
            _model_call(4, 1, BASE_TS + 2.5, 1.0, [{"role": "user", "content": "b"}]),
            {"id": 9, "type": "run_end", "timestamp": BASE_TS + 10},
        ],
    }
    path = tmp_path / "demo-recording.json"
    path.write_text(json.dumps(recording))
    return path


def test_default_lowering_is_unchanged_and_emits_no_tool_nodes(
    recording_path: Path,
) -> None:
    """execute_tools defaults off, so the shipped edge-delay behavior is intact."""
    graph = from_mini_swe_agent_trace(recording_path).graphs["task_demo"]
    assert sorted(graph.nodes) == ["n0", "n1"]
    assert all(isinstance(n, LlmNode) for n in graph.nodes.values())
    edge = next(e for e in graph.edges if e.target == "n1")
    assert edge.delay_after_predecessor_us == pytest.approx(500_000.0)


def test_batched_tool_calls_become_one_node_in_recorded_order(
    recording_path: Path,
) -> None:
    graph = from_mini_swe_agent_trace(recording_path, execute_tools=True).graphs[
        "task_demo"
    ]
    assert sorted(graph.nodes) == ["n0", "n1", "t0"]
    tool = graph.nodes["t0"]
    assert isinstance(tool, ToolNode)
    assert tool.commands == ["mkdir -p src", "echo hi > src/x"]


def test_tool_node_is_chained_between_the_flanking_llm_nodes(
    recording_path: Path,
) -> None:
    graph = from_mini_swe_agent_trace(recording_path, execute_tools=True).graphs[
        "task_demo"
    ]
    pairs = {(e.source, e.target) for e in graph.edges}
    assert ("n0", "t0") in pairs
    assert ("t0", "n1") in pairs
    assert ("n0", "n1") not in pairs


def test_recorded_gap_is_not_also_replayed_as_a_delay(recording_path: Path) -> None:
    """The tool now costs real time; replaying the recorded gap would double-count."""
    graph = from_mini_swe_agent_trace(recording_path, execute_tools=True).graphs[
        "task_demo"
    ]
    for edge in graph.edges:
        if edge.target in ("t0", "n1"):
            assert edge.delay_after_predecessor_us is None


def test_tool_output_channel_is_declared(recording_path: Path) -> None:
    graph = from_mini_swe_agent_trace(recording_path, execute_tools=True).graphs[
        "task_demo"
    ]
    assert "t0_out" in graph.state


def test_steps_without_tool_calls_produce_no_tool_node(tmp_path: Path) -> None:
    recording = {
        "format": "mini-swe-agent-recording-1.0",
        "metadata": {"instance_id": "task_notools"},
        "events": [
            _model_call(1, 0, BASE_TS, 1.0, [{"role": "user", "content": "a"}]),
            _model_call(2, 1, BASE_TS + 1.5, 1.0, [{"role": "user", "content": "b"}]),
        ],
    }
    path = tmp_path / "notools-recording.json"
    path.write_text(json.dumps(recording))
    graph = from_mini_swe_agent_trace(path, execute_tools=True).graphs["task_notools"]
    assert sorted(graph.nodes) == ["n0", "n1"]


@pytest.fixture
def trailing_tool_recording_path(tmp_path: Path) -> Path:
    """The shipped `task_files` shape: model, tool, model, tool, model, tool.

    Agent trajectories routinely end with a submit/finalize command AFTER the
    final model call, so the terminal tool call is the common shape rather than
    an edge case. That command carries `error={"type": "Submitted"}` in the real
    corpus -- agent control flow, not a command failure.
    """
    recording: dict[str, Any] = {
        "format": "mini-swe-agent-recording-1.0",
        "metadata": {"instance_id": "task_trailing"},
        "events": [
            {"id": 0, "type": "run_start", "timestamp": BASE_TS},
            _model_call(1, 0, BASE_TS, 1.0, [{"role": "user", "content": "a"}]),
            _tool_call(2, 0, BASE_TS + 1.0, 0.1, "mkdir -p src"),
            _model_call(3, 1, BASE_TS + 1.5, 1.0, [{"role": "user", "content": "b"}]),
            _tool_call(4, 1, BASE_TS + 2.5, 0.1, "cat src/x"),
            _model_call(5, 2, BASE_TS + 3.0, 1.0, [{"role": "user", "content": "c"}]),
            _tool_call(
                6,
                2,
                BASE_TS + 4.0,
                0.0947,
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                error=SUBMITTED_ERROR,
            ),
            {"id": 9, "type": "run_end", "timestamp": BASE_TS + 10},
        ],
    }
    path = tmp_path / "trailing-recording.json"
    path.write_text(json.dumps(recording))
    return path


def test_trailing_tool_call_after_the_last_model_call_is_lowered(
    trailing_tool_recording_path: Path,
) -> None:
    """The terminal submit command is real measured work and must not be dropped."""
    graph = from_mini_swe_agent_trace(
        trailing_tool_recording_path, execute_tools=True
    ).graphs["task_trailing"]
    assert sorted(graph.nodes) == ["n0", "n1", "n2", "t0", "t1", "t2"]
    tail = graph.nodes["t2"]
    assert isinstance(tail, ToolNode)
    assert tail.commands == ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]


def test_trailing_tool_node_is_chained_between_last_llm_and_end(
    trailing_tool_recording_path: Path,
) -> None:
    graph = from_mini_swe_agent_trace(
        trailing_tool_recording_path, execute_tools=True
    ).graphs["task_trailing"]
    pairs = {(e.source, e.target) for e in graph.edges}
    assert ("n2", "t2") in pairs
    assert ("t2", "END") in pairs
    assert ("n2", "END") not in pairs


def test_trailing_tool_edges_carry_no_recorded_delay(
    trailing_tool_recording_path: Path,
) -> None:
    graph = from_mini_swe_agent_trace(
        trailing_tool_recording_path, execute_tools=True
    ).graphs["task_trailing"]
    for edge in graph.edges:
        if edge.source == "t2" or edge.target == "t2":
            assert edge.delay_after_predecessor_us is None


def test_trailing_tool_output_channel_is_declared(
    trailing_tool_recording_path: Path,
) -> None:
    graph = from_mini_swe_agent_trace(
        trailing_tool_recording_path, execute_tools=True
    ).graphs["task_trailing"]
    assert "t2_out" in graph.state


def test_trailing_tool_call_is_ignored_when_tools_do_not_execute(
    trailing_tool_recording_path: Path,
) -> None:
    """The shipped edge-delay path is unchanged: no tool node, END hangs off n2."""
    graph = from_mini_swe_agent_trace(trailing_tool_recording_path).graphs[
        "task_trailing"
    ]
    assert sorted(graph.nodes) == ["n0", "n1", "n2"]
    assert ("n2", "END") in {(e.source, e.target) for e in graph.edges}


def test_control_flow_error_does_not_exclude_a_command(tmp_path: Path) -> None:
    """Submitted means the command ran and the agent then ended the episode."""
    recording = {
        "format": "mini-swe-agent-recording-1.0",
        "metadata": {"instance_id": "task_submitted"},
        "events": [
            _model_call(1, 0, BASE_TS, 1.0, [{"role": "user", "content": "a"}]),
            _tool_call(2, 0, BASE_TS + 1.0, 0.1, "echo done", error=SUBMITTED_ERROR),
            _model_call(3, 1, BASE_TS + 1.5, 1.0, [{"role": "user", "content": "b"}]),
        ],
    }
    path = tmp_path / "submitted-recording.json"
    path.write_text(json.dumps(recording))
    graph = from_mini_swe_agent_trace(path, execute_tools=True).graphs["task_submitted"]
    assert graph.nodes["t0"].commands == ["echo done"]


def test_genuine_failure_excludes_the_command(tmp_path: Path) -> None:
    """A real exception means the command did not complete; do not replay it."""
    recording = {
        "format": "mini-swe-agent-recording-1.0",
        "metadata": {"instance_id": "task_failed"},
        "events": [
            _model_call(1, 0, BASE_TS, 1.0, [{"role": "user", "content": "a"}]),
            _tool_call(
                2,
                0,
                BASE_TS + 1.0,
                0.1,
                "sleep 999",
                error={"type": "RuntimeError", "message": "boom"},
            ),
            _model_call(3, 1, BASE_TS + 1.5, 1.0, [{"role": "user", "content": "b"}]),
        ],
    }
    path = tmp_path / "failed-recording.json"
    path.write_text(json.dumps(recording))
    graph = from_mini_swe_agent_trace(path, execute_tools=True).graphs["task_failed"]
    assert sorted(graph.nodes) == ["n0", "n1"]


def test_open_loop_with_tool_execution_is_refused(recording_path: Path) -> None:
    """Pacing against the recorded timeline would floor e2e at the capture host."""
    ctx = GraphParseContext(open_loop_replay=True)
    with pytest.raises(NotImplementedError, match="open-loop") as exc:
        AgentTraceRecordingAdapter.parse(recording_path, ctx, execute_tools=True)
    assert "--no-open-loop-replay" in str(exc.value)


def test_open_loop_strict_knob_with_tool_execution_is_refused(
    recording_path: Path,
) -> None:
    """Strict is an open-loop-only modifier, so it implies the pacing too."""
    ctx = GraphParseContext(replay_only_knobs=("--open-loop-strict",))
    with pytest.raises(NotImplementedError, match="--open-loop-strict"):
        AgentTraceRecordingAdapter.parse(recording_path, ctx, execute_tools=True)


def test_closed_loop_with_tool_execution_lowers_tool_nodes(
    recording_path: Path,
) -> None:
    """Closed-loop dispatch is the supported pairing for executed tools."""
    ctx = GraphParseContext(open_loop_replay=False)
    graph = AgentTraceRecordingAdapter.parse(
        recording_path, ctx, execute_tools=True
    ).graphs["task_demo"]
    assert isinstance(graph.nodes["t0"], ToolNode)


def test_open_loop_without_tool_execution_is_allowed(recording_path: Path) -> None:
    """The shipped replay path is the default configuration and must stay usable."""
    ctx = GraphParseContext(open_loop_replay=True)
    graph = AgentTraceRecordingAdapter.parse(recording_path, ctx).graphs["task_demo"]
    assert sorted(graph.nodes) == ["n0", "n1"]


def test_ctxless_parse_with_tool_execution_does_not_refuse(
    recording_path: Path,
) -> None:
    """Nothing told the adapter pacing was on, so there is nothing to refuse."""
    graph = AgentTraceRecordingAdapter.parse(recording_path, None, execute_tools=True)
    assert isinstance(graph.graphs["task_demo"].nodes["t0"], ToolNode)


def test_resolved_context_reports_open_loop_on_for_a_default_run() -> None:
    """The guard is only real if resolution actually reports the DEFAULT-on case.

    Open-loop replay defaults to True, so a resolver that reported it only when
    the operator named the flag would leave the refusal permanently unreachable
    -- which is exactly the defect this test exists to catch.
    """
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.dataset.graph.workload_detect import resolve_graph_parse_context
    from tests.unit.conftest import make_run_from_cli
    from tests.unit.dataset.graph.conftest import DYNAMO_NESTED_FIXTURE

    base = dict(
        model_names=["test-model"],
        input_file=str(DYNAMO_NESTED_FIXTURE),
        tokenizer_name="builtin",
    )
    assert (
        resolve_graph_parse_context(
            make_run_from_cli(CLIConfig(**base))
        ).open_loop_replay
        is True
    )
    off = make_run_from_cli(CLIConfig(**base, open_loop_replay=False))
    assert resolve_graph_parse_context(off).open_loop_replay is False


def test_resolved_context_reports_execute_tools_from_the_cli_flag() -> None:
    """`--graph-execute-tools` must survive the whole CLI -> run -> ctx chain.

    The flag is worthless if it stops at ``CLIConfig``: the only production
    caller (``GraphStoreBuilder`` -> ``parse_graph_workload(run, path)``) passes
    no adapter kwargs, so the ctx is the ONLY channel that can carry it. A test
    that faked the ctx would pass against a flag the converter silently drops --
    which is exactly the defect this asserts against.
    """
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.dataset.graph.workload_detect import resolve_graph_parse_context
    from tests.unit.conftest import make_run_from_cli
    from tests.unit.dataset.graph.conftest import DYNAMO_NESTED_FIXTURE

    base = dict(
        model_names=["test-model"],
        input_file=str(DYNAMO_NESTED_FIXTURE),
        tokenizer_name="builtin",
    )
    on = make_run_from_cli(CLIConfig(**base, graph_execute_tools=True))
    assert resolve_graph_parse_context(on).execute_tools is True
    off = make_run_from_cli(CLIConfig(**base))
    assert resolve_graph_parse_context(off).execute_tools is False


def test_ctx_execute_tools_lowers_tool_nodes_without_the_keyword(
    recording_path: Path,
) -> None:
    """The CLI-driven path: no caller passes the keyword, the ctx carries it."""
    ctx = GraphParseContext(execute_tools=True, open_loop_replay=False)
    graph = AgentTraceRecordingAdapter.parse(recording_path, ctx).graphs["task_demo"]
    assert isinstance(graph.nodes["t0"], ToolNode)


def test_explicit_execute_tools_keyword_overrides_the_ctx(
    recording_path: Path,
) -> None:
    """Programmatic callers keep the last word, in BOTH directions."""
    on_ctx = GraphParseContext(execute_tools=True, open_loop_replay=False)
    off = AgentTraceRecordingAdapter.parse(recording_path, on_ctx, execute_tools=False)
    assert sorted(off.graphs["task_demo"].nodes) == ["n0", "n1"]

    off_ctx = GraphParseContext(execute_tools=False, open_loop_replay=False)
    on = AgentTraceRecordingAdapter.parse(recording_path, off_ctx, execute_tools=True)
    assert isinstance(on.graphs["task_demo"].nodes["t0"], ToolNode)


def test_ctx_execute_tools_with_open_loop_is_refused(recording_path: Path) -> None:
    """The refusal must key on the RESOLVED value, not on the keyword.

    A CLI-driven run passes no keyword, so a guard testing the keyword alone
    would wave through the one combination that silently mis-measures: pacing
    against a timeline that already contains the recorded tool durations.
    """
    ctx = GraphParseContext(execute_tools=True, open_loop_replay=True)
    with pytest.raises(NotImplementedError, match="open-loop") as exc:
        AgentTraceRecordingAdapter.parse(recording_path, ctx)
    assert "--no-open-loop-replay" in str(exc.value)


def _family_recording(benchmark: str) -> dict[str, Any]:
    """The two-call fixture above, tagged with a task family."""
    return {
        "format": "mini-swe-agent-recording-1.0",
        "metadata": {"instance_id": "task_demo", "benchmark": benchmark},
        "events": [
            {"id": 0, "type": "run_start", "timestamp": BASE_TS},
            _model_call(1, 0, BASE_TS, 2.0, [{"role": "user", "content": "a"}]),
            _model_call(4, 1, BASE_TS + 2.5, 1.0, [{"role": "user", "content": "b"}]),
            {"id": 9, "type": "run_end", "timestamp": BASE_TS + 10},
        ],
    }


@pytest.mark.parametrize(
    "benchmark,expected",
    [
        param(
            "swebench",
            {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0,
                "parallel_tool_calls": True,
            },
            id="swebench_gets_the_wrapper_sampling",
        ),
        param("pinchbench", None, id="pinchbench_gets_none"),
    ],
)  # fmt: skip
def test_family_sampling_matches_source_runner_wire_body(
    tmp_path: Path, benchmark: str, expected: dict[str, Any] | None
) -> None:
    """Agent Trace Replay sends different sampling per task family, so the replay must too.

    Its live request is rebuilt from its own config plus the recorded messages,
    so the recording's sampling fields are provenance, not what goes on the
    wire. swebench uses run-mixed-playback.sh values (temperature=0.7, top_p=0.8,
    top_k=20, min_p=0, parallel_tool_calls=true); pinchbench contributes no
    sampling at all.
    """
    path = tmp_path / f"{benchmark}-recording.json"
    path.write_text(json.dumps(_family_recording(benchmark)))
    graph = from_mini_swe_agent_trace(path).graphs["task_demo"]
    assert graph.nodes["n0"].extra_body == expected


def test_family_sampling_can_be_disabled(tmp_path: Path) -> None:
    path = tmp_path / "swebench-recording.json"
    path.write_text(json.dumps(_family_recording("swebench")))
    graph = from_mini_swe_agent_trace(path, family_sampling=False).graphs["task_demo"]
    assert graph.nodes["n0"].extra_body is None


def test_unknown_family_sends_no_sampling(tmp_path: Path) -> None:
    """An unrecognised family must not silently inherit another family's knobs."""
    path = tmp_path / "other-recording.json"
    path.write_text(json.dumps(_family_recording("some-new-benchmark")))
    graph = from_mini_swe_agent_trace(path).graphs["task_demo"]
    assert graph.nodes["n0"].extra_body is None

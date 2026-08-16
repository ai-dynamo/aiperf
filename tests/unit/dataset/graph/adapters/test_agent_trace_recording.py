# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent Trace Replay recording adapter: detection, lowering, and replay fidelity."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.mini_swe_agent_trace.recording import (
    AgentTraceRecordingAdapter,
    from_mini_swe_agent_trace,
)
from aiperf.dataset.graph.adapters.mini_swe_agent_trace.recording_reader import (
    AgentTraceRecordingError,
    EmptyAgentTraceRecordingError,
    is_recording_file,
)
from aiperf.dataset.graph.parse_context import GraphParseContext

# Recorded events stamp `timestamp` when the event is RECORDED, i.e. at its
# END. A call starting at T with duration D is stamped T + D.
BASE_TS = 1_700_000_000.0


def _model_call(
    *,
    event_id: int,
    step: int,
    start: float,
    duration_s: float,
    messages: list[dict[str, Any]],
    completion_tokens: int = 42,
    tools: list[dict[str, Any]] | None = None,
    model: str = "openai/qwen3.6:27b",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "model_call",
        "timestamp": start + duration_s,
        "step": step,
        "duration_ns": int(duration_s * 1e9),
        "provider_request": {
            "messages": messages,
            "model": model,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 16384,
            "api_base": "http://model-endpoint:11434/v1",
            "api_key": "ollama",
            **({"tools": tools} if tools is not None else {}),
        },
        "response_message": {
            "role": "assistant",
            "content": None,
            "extra": {
                "response": {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": completion_tokens,
                    }
                }
            },
        },
    }


def _recording(
    *,
    instance_id: str | None = "task_demo",
    calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if calls is None:
        calls = [
            _model_call(
                event_id=1,
                step=0,
                start=BASE_TS,
                duration_s=2.0,
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                ],
                completion_tokens=60,
                tools=[{"type": "function", "function": {"name": "bash"}}],
            ),
            # Starts 0.5s after the first call ends: the recorded tool gap.
            _model_call(
                event_id=3,
                step=1,
                start=BASE_TS + 2.5,
                duration_s=1.0,
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c1", "type": "function"}],
                    },
                    {"role": "user", "content": "observation"},
                ],
                completion_tokens=17,
                tools=[{"type": "function", "function": {"name": "bash"}}],
            ),
        ]
    return {
        "format": "mini-swe-agent-recording-1.0",
        "mini_version": "2.2.8",
        "task": "demo task",
        "metadata": ({"instance_id": instance_id} if instance_id else {}),
        "events": [
            {"id": 0, "type": "run_start", "timestamp": BASE_TS},
            *calls,
            {"id": 99, "type": "run_end", "timestamp": BASE_TS + 10},
        ],
    }


@pytest.fixture
def recording_path(tmp_path: Path) -> Path:
    path = tmp_path / "demo-recording.json"
    path.write_text(json.dumps(_recording()))
    return path


def test_can_load_recording_file_returns_true(recording_path: Path) -> None:
    assert AgentTraceRecordingAdapter.can_load(recording_path)


def test_can_load_gzipped_recording_returns_true(tmp_path: Path) -> None:
    path = tmp_path / "demo-recording.json.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(_recording()).encode())
    assert AgentTraceRecordingAdapter.can_load(path)


@pytest.mark.parametrize(
    "name,payload",
    [
        param("manifest.json", {"name": "default", "tasks": []}, id="manifest"),
        param("notes.json", {"format": "something-else-1.0"}, id="other_format"),
        param("truncated.json", None, id="not_json"),
    ],
)  # fmt: skip
def test_can_load_non_recording_returns_false(
    tmp_path: Path, name: str, payload: dict | None
) -> None:
    path = tmp_path / name
    path.write_text(json.dumps(payload) if payload is not None else "{not json")
    assert not AgentTraceRecordingAdapter.can_load(path)


def test_can_load_directory_skips_manifest_by_content(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"name": "default"}))
    (tmp_path / "a-recording.json").write_text(json.dumps(_recording()))
    assert AgentTraceRecordingAdapter.can_load(tmp_path)
    assert not is_recording_file(tmp_path / "manifest.json")


def test_lowering_produces_one_node_per_model_call(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path)
    graph = parsed.graphs["task_demo"]
    assert sorted(graph.nodes) == ["n0", "n1"]
    assert [t.id for t in parsed.traces] == ["task_demo"]


def test_swebench_trace_uses_agent_trace_shell_settings(recording_path: Path) -> None:
    payload = _recording()
    payload["metadata"]["benchmark"] = "swebench"
    payload["metadata"]["docker_image"] = "swebench:latest"
    recording_path.write_text(json.dumps(payload))

    parsed = from_mini_swe_agent_trace(recording_path)

    sandbox = parsed.traces[0].tool_sandbox
    assert sandbox is not None
    assert sandbox.cwd == "/testbed"
    assert sandbox.interpreter == ("bash", "-c")


def test_swebench_trace_without_image_keeps_shell_settings(
    recording_path: Path,
) -> None:
    payload = _recording()
    payload["metadata"]["benchmark"] = "swebench"
    recording_path.write_text(json.dumps(payload))

    parsed = from_mini_swe_agent_trace(recording_path)

    sandbox = parsed.traces[0].tool_sandbox
    assert sandbox is not None
    assert sandbox.container is None
    assert sandbox.cwd == "/testbed"
    assert sandbox.interpreter == ("bash", "-c")


def test_prompt_round_trips_verbatim_including_tool_calls(
    recording_path: Path,
) -> None:
    """The interned prompt must materialize to the recorded array byte-for-byte.

    `tool_calls` on the assistant turn and an explicit `content: None` are the
    parts a normalizing round-trip would quietly drop, changing the replayed
    prompt's token count.
    """
    parsed = from_mini_swe_agent_trace(recording_path)
    graph = parsed.graphs["task_demo"]
    recorded = _recording()
    calls = [e for e in recorded["events"] if e["type"] == "model_call"]
    for index, call in enumerate(calls):
        node = graph.nodes[f"n{index}"]
        materialized = parsed.segment_pool.materialize(
            node.metadata["trie"]["prompt_segment_ids"]
        )
        assert materialized == call["provider_request"]["messages"]


def test_shared_prefix_interns_once(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path)
    graph = parsed.graphs["task_demo"]
    first = graph.nodes["n0"].metadata["trie"]["prompt_segment_ids"]
    second = graph.nodes["n1"].metadata["trie"]["prompt_segment_ids"]
    assert second[: len(first)] == first
    # 2 + 4 message slots collapse to 4 distinct segments.
    assert len(parsed.segment_pool.by_id) == 4


def test_output_cap_pins_generation_to_recorded_length(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path)
    graph = parsed.graphs["task_demo"]
    assert graph.nodes["n0"].max_tokens == 60
    assert graph.nodes["n1"].max_tokens == 17


def test_zero_output_call_upgrades_cap_to_one(tmp_path: Path) -> None:
    path = tmp_path / "zero-recording.json"
    path.write_text(
        json.dumps(
            _recording(
                calls=[
                    _model_call(
                        event_id=1,
                        step=0,
                        start=BASE_TS,
                        duration_s=1.0,
                        messages=[{"role": "user", "content": "hi"}],
                        completion_tokens=0,
                    )
                ]
            )
        )
    )
    parsed = from_mini_swe_agent_trace(path)
    assert parsed.graphs["task_demo"].nodes["n0"].max_tokens == 1


def test_tools_are_carried_onto_the_node(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path)
    node = parsed.graphs["task_demo"].nodes["n0"]
    assert node.raw_tools == [{"type": "function", "function": {"name": "bash"}}]


def test_recorded_model_and_sampling_are_not_carried_by_default(
    recording_path: Path,
) -> None:
    """A LiteLLM model string is not an endpoint model id, and the whole point
    is replaying one trajectory against a different model."""
    parsed = from_mini_swe_agent_trace(recording_path)
    node = parsed.graphs["task_demo"].nodes["n0"]
    assert node.model is None
    assert node.extra_body is None


def test_swebench_uses_agent_trace_replay_sampling_overrides(
    recording_path: Path,
) -> None:
    """The default payload mirrors run-mixed-playback.sh, not the base YAML."""
    payload = _recording()
    payload["metadata"]["benchmark"] = "swebench"
    recording_path.write_text(json.dumps(payload))
    parsed = from_mini_swe_agent_trace(recording_path)
    node = parsed.graphs["task_demo"].nodes["n0"]
    assert node.extra_body == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "parallel_tool_calls": True,
    }


def test_opting_into_recorded_model_and_sampling(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(
        recording_path, use_recorded_model=True, use_recorded_sampling=True
    )
    node = parsed.graphs["task_demo"].nodes["n0"]
    assert node.model == "openai/qwen3.6:27b"
    assert node.extra_body == {"temperature": 0.2, "top_p": 0.9}


def test_edge_delay_reproduces_recorded_tool_gap(recording_path: Path) -> None:
    """The gap between one call ending and the next starting is where the
    recorded agent ran its tools."""
    parsed = from_mini_swe_agent_trace(recording_path)
    edge = next(e for e in parsed.graphs["task_demo"].edges if e.target == "n1")
    assert edge.delay_after_predecessor_us == pytest.approx(500_000.0)


def test_delay_cap_compresses_recorded_gap(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, delay_cap_seconds=0.1)
    edge = next(e for e in parsed.graphs["task_demo"].edges if e.target == "n1")
    assert edge.delay_after_predecessor_us == pytest.approx(100_000.0)


def test_ignore_delays_drops_recorded_gap(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, ignore_delays=True)
    edge = next(e for e in parsed.graphs["task_demo"].edges if e.target == "n1")
    assert edge.delay_after_predecessor_us is None


def test_timestamps_use_call_start_not_recorded_end(recording_path: Path) -> None:
    """`timestamp` is stamped after the call returns; the node's start is
    `timestamp - duration`. Using the raw stamp would shift every node late by
    its own duration."""
    parsed = from_mini_swe_agent_trace(recording_path)
    graph = parsed.graphs["task_demo"]
    assert graph.nodes["n0"].arrival_offset_us == 0
    assert graph.nodes["n1"].arrival_offset_us == 2_500_000


def test_absolute_start_timestamps_are_not_stamped(recording_path: Path) -> None:
    """Each recording is an independent task run, so it carries no absolute clock.

    Open-loop replay paces every trace against one corpus-wide
    `schedule_zero = min(recorded_start_unix_ms)`. That is right for a
    co-recorded trace stream and wrong here: Agent Trace Replay runs its tasks
    sequentially and never paces one against another's wall clock. The shipped
    default set spans 95 days, so stamping absolute starts parks five of its
    eight traces 65-95 days out and they never dispatch.
    """
    parsed = from_mini_swe_agent_trace(recording_path)
    graph = parsed.graphs["task_demo"]
    assert all(node.recorded_start_unix_ms is None for node in graph.nodes.values())


def test_directory_lowers_each_recording_as_its_own_graph(tmp_path: Path) -> None:
    (tmp_path / "a-recording.json").write_text(
        json.dumps(_recording(instance_id="task_a"))
    )
    (tmp_path / "b-recording.json").write_text(
        json.dumps(_recording(instance_id="task_b"))
    )
    parsed = from_mini_swe_agent_trace(tmp_path)
    assert sorted(parsed.graphs) == ["task_a", "task_b"]
    assert sorted(t.id for t in parsed.traces) == ["task_a", "task_b"]
    # Identical content across the two traces interns once.
    assert len(parsed.segment_pool.by_id) == 4


def test_pinchbench_trace_without_explicit_image_uses_agent_trace_sandbox(
    tmp_path: Path,
) -> None:
    """PinchBench recordings execute in Agent Trace Replay's shared task image."""
    recording = _recording()
    recording["metadata"]["benchmark"] = "pinchbench"
    path = tmp_path / "pinchbench-recording.json"
    path.write_text(json.dumps(recording))

    parsed = from_mini_swe_agent_trace(path)

    assert parsed.traces[0].tool_sandbox is not None
    assert parsed.traces[0].tool_sandbox.container == "agent-trace-pinchbench:latest"


def test_trace_id_falls_back_to_file_stem_without_instance_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "swe-corpus-django__django-14500.json.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(_recording(instance_id=None)).encode())
    parsed = from_mini_swe_agent_trace(path)
    assert [t.id for t in parsed.traces] == ["swe-corpus-django__django-14500"]


def test_duplicate_trace_ids_raise(tmp_path: Path) -> None:
    (tmp_path / "a-recording.json").write_text(json.dumps(_recording()))
    (tmp_path / "b-recording.json").write_text(json.dumps(_recording()))
    with pytest.raises(AgentTraceRecordingError, match="duplicate trace id"):
        from_mini_swe_agent_trace(tmp_path)


def test_failed_model_call_refuses_to_lower(tmp_path: Path) -> None:
    recording = _recording()
    calls = [e for e in recording["events"] if e["type"] == "model_call"]
    calls[1]["error"] = {"type": "APIError", "message": "boom"}
    path = tmp_path / "broken-recording.json"
    path.write_text(json.dumps(recording))
    with pytest.raises(AgentTraceRecordingError, match="did not succeed"):
        from_mini_swe_agent_trace(path)


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(
        EmptyAgentTraceRecordingError, match="no Agent Trace Replay recordings"
    ):
        from_mini_swe_agent_trace(tmp_path)


def test_ctx_forwards_only_the_knobs_it_sets(recording_path: Path) -> None:
    ctx = GraphParseContext(delay_cap_seconds=0.25, num_dataset_entries=1)
    parsed = AgentTraceRecordingAdapter.parse(recording_path, ctx)
    edge = next(e for e in parsed.graphs["task_demo"].edges if e.target == "n1")
    assert edge.delay_after_predecessor_us == pytest.approx(250_000.0)
    assert parsed.graphs["task_demo"].nodes["n0"].streaming is True


def test_ctx_num_dataset_entries_caps_traces(tmp_path: Path) -> None:
    (tmp_path / "a-recording.json").write_text(
        json.dumps(_recording(instance_id="task_a"))
    )
    (tmp_path / "b-recording.json").write_text(
        json.dumps(_recording(instance_id="task_b"))
    )
    parsed = AgentTraceRecordingAdapter.parse(
        tmp_path, GraphParseContext(num_dataset_entries=1)
    )
    assert len(parsed.traces) == 1


@pytest.mark.asyncio
async def test_worker_materializes_the_recorded_request_body(
    recording_path: Path, tmp_path: Path
) -> None:
    """End-to-end: parse -> unified store -> worker materialization.

    The node-level assertions above check the IR. This checks the bytes the
    worker will actually put on the wire, which is the only thing that makes
    the replay faithful. `model` is absent so the run's --model applies.
    """
    from aiperf.dataset.graph.segment_trie.store_builder import (
        build_unified_trie_store_interned,
    )
    from aiperf.dataset.graph_segment_unified_store import (
        GraphSegmentUnifiedBackingStore,
        GraphSegmentUnifiedClient,
    )
    from aiperf.graph.worker_materialize import materialize_graph_request_unified

    parsed = from_mini_swe_agent_trace(recording_path)
    trace_id = parsed.traces[0].id
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    store = GraphSegmentUnifiedBackingStore(store_dir, "bench")
    catalog = await build_unified_trie_store_interned(parsed, store)

    client = GraphSegmentUnifiedClient(store_dir, "bench")
    client.open()
    calls = [e for e in _recording()["events"] if e["type"] == "model_call"]
    for _, ordinal in sorted(catalog[trace_id].items(), key=lambda kv: kv[1]):
        request = materialize_graph_request_unified(client, trace_id, ordinal)
        recorded = calls[ordinal]
        assert request["messages"] == recorded["provider_request"]["messages"]
        assert request["tools"] == recorded["provider_request"]["tools"]
        assert (
            request["max_completion_tokens"]
            == (
                recorded["response_message"]["extra"]["response"]["usage"][
                    "completion_tokens"
                ]
            )
        )
        assert "model" not in request


def test_every_node_output_channel_is_declared(recording_path: Path) -> None:
    """The runtime channel store rejects a write to an undeclared channel.

    Nothing consumes these outputs -- recorded prompts are self-contained --
    but the declaration is still mandatory, and its absence fails at dispatch
    (`UnknownChannelError`) rather than at parse, where no IR-level assertion
    would catch it.
    """
    parsed = from_mini_swe_agent_trace(recording_path)
    graph = parsed.graphs["task_demo"]
    assert set(graph.state) == {node.output for node in graph.nodes.values()}

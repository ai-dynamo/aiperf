# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent Trace Replay warmup-parity feature: emit_warmup lowering, id grammar, store round-trip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.dataset.graph.adapters.mini_swe_agent_trace.recording import (
    WARMUP_MAX_TOKENS,
    WARMUP_PROMPT,
    from_mini_swe_agent_trace,
)

BASE_TS = 1_700_000_000.0

BASH_TOOL = {"type": "function", "function": {"name": "bash"}}


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
            "max_tokens": 16384,
            **({"tools": tools} if tools is not None else {}),
        },
        "response_message": {
            "role": "assistant",
            "content": None,
            "extra": {"response": {"usage": {"completion_tokens": completion_tokens}}},
        },
    }


def _recording(
    *,
    instance_id: str = "task_demo",
    benchmark: str | None = "pinchbench",
    tools: list[dict[str, Any]] | None = None,
    model: str = "openai/qwen3.6:27b",
) -> dict[str, Any]:
    calls = [
        _model_call(
            event_id=1,
            step=0,
            start=BASE_TS,
            duration_s=2.0,
            messages=[{"role": "user", "content": "task"}],
            tools=tools or [BASH_TOOL],
            model=model,
        ),
        _model_call(
            event_id=3,
            step=1,
            start=BASE_TS + 2.5,
            duration_s=1.0,
            messages=[
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "done"},
            ],
            tools=tools or [BASH_TOOL],
            model=model,
        ),
    ]
    meta: dict[str, Any] = {"instance_id": instance_id}
    if benchmark is not None:
        meta["benchmark"] = benchmark
    return {
        "format": "mini-swe-agent-recording-1.0",
        "mini_version": "2.2.8",
        "task": "demo task",
        "metadata": meta,
        "events": [
            {"id": 0, "type": "run_start", "timestamp": BASE_TS},
            *calls,
            {"id": 99, "type": "run_end", "timestamp": BASE_TS + 10},
        ],
    }


@pytest.fixture
def recording_path(tmp_path: Path) -> Path:
    p = tmp_path / "warmup-demo.json"
    p.write_text(json.dumps(_recording()))
    return p


# ---------------------------------------------------------------------------
# warmup_traces / all_traces on ParsedGraph
# ---------------------------------------------------------------------------


def test_emit_warmup_false_produces_no_warmup_traces(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=False)
    assert parsed.warmup_traces == []
    assert len(parsed.traces) == 1


def test_emit_warmup_true_produces_one_warmup_trace_per_recording(
    recording_path: Path,
) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    assert len(parsed.warmup_traces) == 1
    assert len(parsed.traces) == 1


def test_all_traces_returns_union_of_profiling_and_warmup(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    all_ids = {t.id for t in parsed.all_traces}
    profiling_ids = {t.id for t in parsed.traces}
    warmup_ids = {t.id for t in parsed.warmup_traces}
    assert all_ids == profiling_ids | warmup_ids
    assert len(parsed.all_traces) == 2


def test_all_traces_without_warmup_equals_traces(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=False)
    assert [t.id for t in parsed.all_traces] == [t.id for t in parsed.traces]


# ---------------------------------------------------------------------------
# warmup trace id grammar — must NOT use :: as separator
# ---------------------------------------------------------------------------


def test_warmup_trace_id_does_not_contain_double_colon(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    warmup_id = parsed.warmup_traces[0].id
    assert "::" not in warmup_id, (
        f"warmup id {warmup_id!r} contains '::' which violates the graph-ID grammar "
        "(template_id = id.split('::', 1)[0] would resolve to the wrong key)"
    )


def test_warmup_trace_id_encodes_the_profiling_trace_id(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    profiling_id = parsed.traces[0].id
    warmup_id = parsed.warmup_traces[0].id
    assert profiling_id in warmup_id


def test_warmup_trace_id_starts_with_warmup_prefix(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    assert parsed.warmup_traces[0].id.startswith("warmup-")


# ---------------------------------------------------------------------------
# warmup agent graph content
# ---------------------------------------------------------------------------


def test_warmup_graph_has_single_llm_node(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    warmup_id = parsed.warmup_traces[0].id
    warmup_graph = parsed.graphs[warmup_id]
    assert len(warmup_graph.nodes) == 1
    node = next(iter(warmup_graph.nodes.values()))
    assert node.max_tokens == WARMUP_MAX_TOKENS


def test_warmup_node_carries_excluded_dispatch_metadata(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    warmup_id = parsed.warmup_traces[0].id
    warmup_graph = parsed.graphs[warmup_id]
    node = next(iter(warmup_graph.nodes.values()))
    dispatch = (node.metadata or {}).get("dispatch", {})
    assert dispatch.get("own_output_cap") is True
    assert dispatch.get("disable_cache_bust") is True


def test_warmup_node_inherits_first_call_tools(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    warmup_id = parsed.warmup_traces[0].id
    warmup_graph = parsed.graphs[warmup_id]
    node = next(iter(warmup_graph.nodes.values()))
    assert node.raw_tools == [BASH_TOOL]


def test_warmup_node_prompt_is_the_warmup_call(recording_path: Path) -> None:
    """Segment pool should contain a segment whose wire_json encodes the warmup prompt."""
    import orjson

    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    pool = parsed.segment_pool
    assert pool is not None
    warmup_segments = [
        seg
        for seg in pool.by_id.values()
        if seg.wire_json is not None
        and orjson.loads(seg.wire_json).get("content") == WARMUP_PROMPT
    ]
    assert warmup_segments, "WARMUP_PROMPT not found in segment pool"


def test_warmup_node_has_no_model_by_default(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    warmup_id = parsed.warmup_traces[0].id
    node = next(iter(parsed.graphs[warmup_id].nodes.values()))
    assert node.model is None


def test_warmup_node_uses_recorded_model_when_flag_set(recording_path: Path) -> None:
    parsed = from_mini_swe_agent_trace(
        recording_path, emit_warmup=True, use_recorded_model=True
    )
    warmup_id = parsed.warmup_traces[0].id
    node = next(iter(parsed.graphs[warmup_id].nodes.values()))
    assert node.model == "openai/qwen3.6:27b"


# ---------------------------------------------------------------------------
# own_output_cap flows through store → materialize with max_tokens=8
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_own_output_cap_survives_store_round_trip_as_warmup_max_tokens(
    recording_path: Path, tmp_path: Path
) -> None:
    """The full build path: lower -> unified store -> worker materialize.

    This is the exact path that was broken (own_output_cap dropped by
    add_node_manifest_interned). Verifies that a WARMUP-phase materialize
    uses the corpus-authored max_tokens=8, not the 1-token BOUNDARY_SNAPSHOT cap.
    """
    from aiperf.dataset.graph.segment_trie.store_builder import (
        build_unified_trie_store_interned,
    )
    from aiperf.dataset.graph_segment_unified_store import (
        GraphSegmentUnifiedBackingStore,
        GraphSegmentUnifiedClient,
    )
    from aiperf.graph.worker_materialize import materialize_graph_request_unified

    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=True)
    warmup_id = parsed.warmup_traces[0].id

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    store = GraphSegmentUnifiedBackingStore(store_dir, "bench")
    catalog = await build_unified_trie_store_interned(parsed, store)

    assert warmup_id in catalog, "warmup trace not in store catalog"

    client = GraphSegmentUnifiedClient(store_dir, "bench")
    client.open()
    ordinals = catalog[warmup_id]
    assert len(ordinals) == 1
    ordinal = next(iter(ordinals.values()))

    # WARMUP phase: own_output_cap must protect the authored 8 from being clobbered.
    warmup_request = materialize_graph_request_unified(
        client, warmup_id, ordinal, phase=CreditPhase.WARMUP
    )
    cap_field = warmup_request.get("max_completion_tokens") or warmup_request.get(
        "max_tokens"
    )
    assert cap_field == WARMUP_MAX_TOKENS, (
        f"expected max_tokens={WARMUP_MAX_TOKENS} (own_output_cap) "
        f"but got {cap_field!r} -- own_output_cap was likely dropped by the store"
    )


@pytest.mark.asyncio
async def test_warmup_node_without_own_output_cap_gets_clamped(
    recording_path: Path, tmp_path: Path
) -> None:
    """Contrasting case: a WARMUP node with no own_output_cap gets clamped to 1."""
    from aiperf.common.environment import Environment
    from aiperf.dataset.graph.segment_trie.store_builder import (
        build_unified_trie_store_interned,
    )
    from aiperf.dataset.graph_segment_unified_store import (
        GraphSegmentUnifiedBackingStore,
        GraphSegmentUnifiedClient,
    )
    from aiperf.graph.worker_materialize import materialize_graph_request_unified

    parsed = from_mini_swe_agent_trace(recording_path, emit_warmup=False)
    trace_id = parsed.traces[0].id

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    store = GraphSegmentUnifiedBackingStore(store_dir, "bench")
    catalog = await build_unified_trie_store_interned(parsed, store)

    client = GraphSegmentUnifiedClient(store_dir, "bench")
    client.open()
    ordinals = catalog[trace_id]
    ordinal = sorted(ordinals.values())[0]

    request = materialize_graph_request_unified(
        client, trace_id, ordinal, phase=CreditPhase.WARMUP
    )
    cap_field = request.get("max_completion_tokens") or request.get("max_tokens")
    assert cap_field == Environment.GRAPH.WARMUP_MAX_OUTPUT_TOKENS

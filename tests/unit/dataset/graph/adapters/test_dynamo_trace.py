# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Dynamo agent-trace -> flat segment-trie ParsedGraph adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import (
    DynamoTraceAdapter,
    DynamoTraceAdapterError,
    EmptyDynamoTraceError,
    from_dynamo_trace,
)
from aiperf.dataset.graph.models import LlmNode, ParsedGraph, StaticEdge
from tests.unit.dataset.graph.adapters.conftest import write_jsonl, write_jsonl_gz

# --- fixture builders -----------------------------------------------------


def _ctx(
    *,
    session_id: str,
    parent_session_id: str | None = None,
) -> dict:
    out: dict = {
        "session_id": session_id,
    }
    if parent_session_id is not None:
        out["parent_session_id"] = parent_session_id
    return out


def _request_end(
    *,
    ts: int,
    session_id: str,
    parent_session_id: str | None = None,
    request_id: str | None = None,
    model: str = "m",
    input_tokens: int | None = 10,
    output_tokens: int | None = 20,
    cached_tokens: int | None = 5,
    kv_hit_rate: float | None = 0.5,
    ttft_ms: float | None = 100.0,
    replay: dict | None = None,
) -> dict:
    req: dict = {
        "request_id": request_id or f"r{ts}",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "kv_hit_rate": kv_hit_rate,
        "ttft_ms": ttft_ms,
    }
    if replay is not None:
        req["replay"] = replay
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": _ctx(
            session_id=session_id,
            parent_session_id=parent_session_id,
        ),
        "request": req,
    }


def _tool_end(
    *,
    ts: int,
    session_id: str,
    parent_session_id: str | None = None,
    name: str = "search",
    duration_ms: float | None = 50.0,
    status: str = "succeeded",
    tool_call_id: str | None = None,
) -> dict:
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "tool_end",
        "event_time_unix_ms": ts,
        "event_source": "harness",
        "agent_context": _ctx(
            session_id=session_id,
            parent_session_id=parent_session_id,
        ),
        "tool": {
            "tool_call_id": tool_call_id or f"tc{ts}",
            "tool_class": name,
            "duration_ms": duration_ms,
            "status": status,
        },
    }


def _replay(hashes: list[int], *, bs: int = 16) -> dict:
    """Block-aligned replay dict (input_length = len(hashes) * bs)."""
    return {
        "trace_block_size": bs,
        "input_length": len(hashes) * bs,
        "input_sequence_hashes": hashes,
    }


def _parse(tmp_path: Path, records: list[dict], **kwargs: Any) -> ParsedGraph:
    """Write ``records`` to a trace file under ``tmp_path`` and adapt it."""
    return from_dynamo_trace(write_jsonl(tmp_path / "trace.jsonl", records), **kwargs)


def _static_edges(pb: ParsedGraph) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in pb.graph.edges if isinstance(e, StaticEdge)}


# --- 1. flat lowering: one LlmNode per request_end ------------------------


def test_single_session_three_turns_flat_nodes(tmp_path: Path) -> None:
    """Three recorded turns of one session flatten to three LlmNodes chained by finished-before edges."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="p1"),
            _request_end(ts=1100, session_id="p1"),
            _request_end(ts=1200, session_id="p1"),
        ],
    )

    assert isinstance(pb, ParsedGraph)
    aid = "p1"
    assert sorted(pb.graph.nodes) == [f"{aid}:0", f"{aid}:1", f"{aid}:2"]
    assert all(isinstance(n, LlmNode) for n in pb.graph.nodes.values())
    assert pb.segment_pool is not None
    # Sequential recorded turns chain :0 -> :1 -> :2 (finished-before edges).
    edges = _static_edges(pb)
    assert ("START", f"{aid}:0") in edges
    assert (f"{aid}:0", f"{aid}:1") in edges
    assert (f"{aid}:1", f"{aid}:2") in edges


def test_single_trace_record_with_base_tag(tmp_path: Path) -> None:
    """A single-root capture emits exactly one session-named trace tagged from-dynamo-trace and never multi-root."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="p1"),
            _request_end(ts=1100, session_id="p1"),
        ],
    )

    assert len(pb.traces) == 1
    assert pb.traces[0].id == "p1"
    assert "from-dynamo-trace" in pb.traces[0].tags
    assert "multi-root" not in pb.traces[0].tags


def test_multi_root_file_parses_one_trace_per_root_tree(tmp_path: Path) -> None:
    """Two parentless root sessions become TWO per-tree traces (multi-graph)."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="p1"),
            _request_end(ts=1010, session_id="p2"),
            _request_end(ts=1100, session_id="p1"),
            _request_end(ts=1110, session_id="p2"),
        ],
    )

    # One id-sorted trace per tree; each selects its own graph via graph_ref, and
    # the multi-root tag is dropped (each tree is its own single-root trace).
    assert [t.id for t in pb.traces] == ["p1", "p2"]
    for trace in pb.traces:
        assert trace.graph_ref == trace.id
        assert "multi-root" not in trace.tags
    # graph is the FIRST tree (p1); p2's node lives in its own graph, not here.
    assert "p1:0" in pb.graph.nodes
    assert "p2:0" not in pb.graph.nodes
    assert "p2:0" in pb.graphs["p2"].nodes


def test_every_node_output_channel_declared_in_state(tmp_path: Path) -> None:
    """Every lowered node writes a ``<node_id>_out`` channel that is declared in the graph state."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="p1"),
            _request_end(ts=1100, session_id="child", parent_session_id="p1"),
            _request_end(ts=1200, session_id="p1"),
        ],
    )

    for nid, node in pb.graph.nodes.items():
        assert node.output == f"{nid}_out"
        assert f"{nid}_out" in pb.graph.state


# --- 2. edge timing: recorded delays replay by default (idle-warped) --------


def _two_turn_edge(pb: ParsedGraph) -> StaticEdge:
    aid = "p1"
    return next(
        e
        for e in pb.graph.edges
        if isinstance(e, StaticEdge)
        and (e.source, e.target) == (f"{aid}:0", f"{aid}:1")
    )


@pytest.mark.parametrize(
    "cap_kwargs,expected_delay_us,expected_arrival_us",
    [
        # 1s recorded gap (zero api_time) sits below the shared 60s cap: survives unwarped.
        param({}, 1_000_000.0, 1_000_000, id="default_cap_keeps_recorded_gap"),
        param({"idle_gap_cap_seconds": 0.25}, 250_000.0, None, id="explicit_cap_compresses_gap"),
    ],
)  # fmt: skip
def test_recorded_idle_gap_replays_on_binding_edge(
    tmp_path: Path,
    cap_kwargs: dict[str, float],
    expected_delay_us: float,
    expected_arrival_us: int | None,
) -> None:
    """Recorded end-to-start gaps replay on the binding edge, warped down to the idle-gap cap when one applies (weka idle-warp parity)."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="p1"),
            _request_end(ts=2000, session_id="p1"),
        ],
        **cap_kwargs,
    )

    assert _two_turn_edge(pb).delay_after_predecessor_us == pytest.approx(
        expected_delay_us
    )
    if expected_arrival_us is not None:
        # Arrival stamping rides the same warped clock as the edge delay.
        assert pb.graph.nodes["p1:1"].arrival_offset_us == expected_arrival_us


# --- 3. session identity via x-dynamo-* headers -----------------------------


def test_session_headers_every_turn_final_on_last(tmp_path: Path) -> None:
    """Session identity is HEADER-borne (x-dynamo-*) on every turn with the final marker only on the last, never body nvext."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="sess-X"),
            _request_end(ts=1100, session_id="sess-X"),
            _request_end(ts=1200, session_id="sess-X"),
        ],
    )
    aid = "sess-X"

    def headers(nid: str) -> dict:
        node = pb.graph.nodes[nid]
        # Dynamo's NvExt is deny_unknown_fields and rejects body-level agent_context.
        assert "nvext" not in (node.extra_body or {}), (
            "body nvext is rejected by dynamo (deny_unknown_fields)"
        )
        return node.extra_headers

    h1, h2, h3 = (headers(f"{aid}:{k}") for k in (0, 1, 2))
    for h in (h1, h2, h3):
        assert h["x-dynamo-session-id"] == "sess-X"
    assert "x-dynamo-session-final" not in h1
    assert "x-dynamo-session-final" not in h2
    assert h3["x-dynamo-session-final"] == "true"


def test_child_session_headers_carry_parent_session_id(tmp_path: Path) -> None:
    """A child session's headers carry x-dynamo-parent-session-id while it flattens into the parent's single trace."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="parent"),
            _request_end(ts=1100, session_id="child", parent_session_id="parent"),
            _request_end(ts=1200, session_id="parent"),
        ],
    )
    child = pb.graph.nodes["child:0"]
    h = child.extra_headers
    assert h["x-dynamo-session-id"] == "child"
    assert h["x-dynamo-parent-session-id"] == "parent"
    # Single-turn child session: its one turn is also the final one.
    assert h["x-dynamo-session-final"] == "true"
    parent_a1 = pb.graph.nodes["parent:0"]
    assert "x-dynamo-parent-session-id" not in parent_a1.extra_headers
    # Only the parent emits a trace; the child flattens into the same graph.
    assert [t.id for t in pb.traces] == ["parent"]
    assert "multi-root" not in pb.traces[0].tags


def test_session_headers_reach_store_envelope(tmp_path: Path) -> None:
    """The store envelope carries extra_headers so the worker can merge them into the request headers (Turn.extra_headers -> transport merge)."""
    from aiperf.dataset.graph.segment_trie.store_builder import (
        _prompt_segment_ids,
        _trie_envelope,
    )

    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="s1"),
            _request_end(ts=1100, session_id="s1"),
        ],
    )
    aid = "s1"
    for k, final in ((0, False), (1, True)):
        node = pb.graph.nodes[f"{aid}:{k}"]
        env = _trie_envelope(node, _prompt_segment_ids(node) or [])
        h = env["extra_headers"]
        assert h["x-dynamo-session-id"] == "s1"
        assert ("x-dynamo-session-final" in h) is final
        assert "nvext" not in env["dispatch_overrides"]


# --- 4. recorded output pinning ----------------------------------------------


def test_recorded_output_always_pins_native_max_tokens(tmp_path: Path) -> None:
    """Recorded output_tokens always pin the native ``LlmNode.max_tokens`` (weka parity); a recorded 0 upgrades to 1 with a warning (wire_output_cap)."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="p1", output_tokens=256),
            _request_end(ts=1100, session_id="p1", output_tokens=300),
            _request_end(ts=1200, session_id="p1", output_tokens=0),
        ],
    )

    aid = "p1"
    assert pb.graph.nodes[f"{aid}:0"].max_tokens == 256
    assert pb.graph.nodes[f"{aid}:1"].max_tokens == 300
    assert pb.graph.nodes[f"{aid}:2"].max_tokens == 1


# --- 5. empty trace file ----------------------------------------------------


def test_empty_trace_file_raises(tmp_path: Path) -> None:
    """A zero-byte capture raises EmptyDynamoTraceError rather than yielding an empty graph."""
    p = tmp_path / "trace.jsonl"
    p.write_text("")

    with pytest.raises(EmptyDynamoTraceError):
        from_dynamo_trace(p)


def test_replay_only_trace_synthesizes_request_root_sessions(tmp_path: Path) -> None:
    """Context-free request_end records become deterministic independent roots."""
    replay_only = _request_end(ts=1000, session_id="ignored")
    del replay_only["agent_context"]
    replay_only_2 = _request_end(ts=1100, session_id="ignored", request_id="r2")
    del replay_only_2["agent_context"]

    parsed = _parse(tmp_path, [replay_only, replay_only_2])

    assert {trace.id for trace in parsed.traces} == {"request-r1000", "request-r2"}
    assert all(
        node.metadata["dynamo"]["session_id"] in {"request-r1000", "request-r2"}
        for graph in parsed.graphs.values()
        for node_id, node in graph.nodes.items()
        if node_id != "__start__"
    )


# --- 6. virtual-hash fallback tag --------------------------------------------


@pytest.mark.parametrize(
    "record_kwargs,fallback_tagged",
    [
        param({}, True, id="no_replay_metrics_tags_fallback"),
        param(
            {
                "input_tokens": 32,
                "replay": {
                    "trace_block_size": 16,
                    "input_length": 32,
                    "input_sequence_hashes": [11, 22],
                },
            },
            False,
            id="recorded_replay_metrics_no_fallback",
        ),
    ],
)  # fmt: skip
def test_virtual_hash_fallback_tag_tracks_recorded_replay(
    tmp_path: Path, record_kwargs: dict[str, Any], fallback_tagged: bool
) -> None:
    """The virtual-hash-fallback trace tag appears only when a turn has no recorded replay hashes to reuse."""
    pb = _parse(tmp_path, [_request_end(ts=1000, session_id="p1", **record_kwargs)])

    assert ("virtual-hash-fallback" in pb.traces[0].tags) is fallback_tagged


# --- 7. session_id_filter restricts sessions --------------------------------


def test_session_id_filter_restricts_sessions(tmp_path: Path) -> None:
    """``session_id_filter`` keeps only the named session's turns and trace."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="p1"),
            _request_end(ts=1010, session_id="p2"),
            _request_end(ts=1100, session_id="p1"),
        ],
        session_id_filter="p1",
    )

    assert [t.id for t in pb.traces] == ["p1"]
    assert set(pb.graph.nodes) == {"p1:0", "p1:1"}


def test_session_id_filter_no_matches_raises(tmp_path: Path) -> None:
    """A ``session_id_filter`` matching no session raises EmptyDynamoTraceError."""
    with pytest.raises(EmptyDynamoTraceError):
        _parse(
            tmp_path,
            [_request_end(ts=1000, session_id="p1")],
            session_id_filter="p-OTHER",
        )


# --- 8. expected + observed metadata ----------------------------------------


def test_expected_tokens_populated_from_request(tmp_path: Path) -> None:
    """Recorded input/output/cached token counts populate ``LlmNode.expected``, leaving cache_creation_tokens unset."""
    pb = _parse(
        tmp_path,
        [
            _request_end(
                ts=1000,
                session_id="p1",
                input_tokens=128,
                output_tokens=256,
                cached_tokens=64,
            ),
        ],
    )

    a1 = pb.graph.nodes["p1:0"]
    assert isinstance(a1, LlmNode)
    assert a1.expected is not None
    assert a1.expected.input_tokens == 128
    assert a1.expected.output_tokens == 256
    assert a1.expected.cache_read_tokens == 64
    assert a1.expected.cache_creation_tokens is None


def test_node_metadata_carries_only_identity_breadcrumbs(tmp_path: Path) -> None:
    """Node metadata carries only the dynamo identity breadcrumbs (session/turn, small_prompt) -- no recorded-scalar round-trip is stamped."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="p1", kv_hit_rate=0.42),
            _request_end(ts=1100, session_id="p1", kv_hit_rate=0.99),
        ],
    )

    # Everything else lives on native fields or in the capture file, and this
    # metadata survives the content-free sidecar broadcast.
    a1 = pb.graph.nodes["p1:0"]
    assert set(a1.metadata) == {"dynamo", "trie"}
    assert set(a1.metadata["dynamo"]) == {
        "session_id",
        "parent_session_id",
        "turn_index",
        "small_prompt",
    }


def test_tool_events_are_recognized_but_not_lowered(tmp_path: Path) -> None:
    """tool_start/tool_end/tool_error records parse cleanly and produce no per-node tool metadata."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="p1"),
            _tool_end(ts=1100, session_id="p1", name="search", duration_ms=80.0),
            _tool_end(ts=1150, session_id="p1", name="fetch", duration_ms=20.0),
            _request_end(ts=1200, session_id="p1"),
        ],
    )

    # Tool time is implicit in the recorded end-to-start gaps the replay honors,
    # and no consumer reads a per-node breakdown.
    aid = "p1"
    assert sorted(pb.graph.nodes) == [f"{aid}:0", f"{aid}:1"]
    for node in pb.graph.nodes.values():
        assert "tool_breakdown" not in node.metadata["dynamo"]


# --- 9. cycle and depth guards ----------------------------------------------


def test_cycle_in_parent_link_raises(tmp_path: Path) -> None:
    """Mutually-pointing parent_link records raise DynamoTraceAdapterError."""
    with pytest.raises(DynamoTraceAdapterError) as ei:
        _parse(
            tmp_path,
            [
                _request_end(ts=1000, session_id="A", parent_session_id="B"),
                _request_end(ts=1100, session_id="B", parent_session_id="A"),
            ],
        )
    assert "cycle" in str(ei.value).lower()


def test_depth_overflow_respects_env_setting(tmp_path: Path, monkeypatch) -> None:
    """A linear chain longer than AIPERF_DYNAMO_MAX_SUBAGENT_DEPTH raises, and passes once the cap is raised above it."""
    # Build a 5-level chain p1 -> p2 -> p3 -> p4 -> p5
    pids = [f"p{i}" for i in range(1, 6)]
    records = []
    ts = 1000
    for i, pid in enumerate(pids):
        parent = pids[i - 1] if i > 0 else None
        records.append(_request_end(ts=ts, session_id=pid, parent_session_id=parent))
        ts += 10
    p = write_jsonl(tmp_path / "trace.jsonl", records)

    # Patch the adapter module's *local* `Environment` reference instead of
    # mutating the shared `Environment.DYNAMO` Pydantic singleton. The latter
    # was xdist-flaky: concurrent workers could observe a torn singleton
    # mid-monkeypatch. Replacing the module-level name binding is fully
    # isolated to this test and still exercises the env-read code path in
    # `from_dynamo_trace`.
    from types import SimpleNamespace

    from aiperf.dataset.graph.adapters.dynamo import trace as _dt_mod

    # Cap depth at 3 -> chain depth 5 should raise.
    monkeypatch.setattr(
        _dt_mod,
        "Environment",
        SimpleNamespace(DYNAMO=SimpleNamespace(MAX_SUBAGENT_DEPTH=3)),
    )
    with pytest.raises(DynamoTraceAdapterError) as ei:
        from_dynamo_trace(p)
    assert "depth" in str(ei.value).lower()

    # Cap depth at 10 -> chain depth 5 should pass.
    monkeypatch.setattr(
        _dt_mod,
        "Environment",
        SimpleNamespace(DYNAMO=SimpleNamespace(MAX_SUBAGENT_DEPTH=10)),
    )
    pb = from_dynamo_trace(p)
    assert isinstance(pb, ParsedGraph)


# --- deliberate divergences from dynamo-source behavior ----------------------


def test_self_parent_session_is_root_not_cycle(tmp_path: Path) -> None:
    """Dynamo's generic header mapping passes parent==session through verbatim; it means "no parent", never a cycle."""
    pb = _parse(
        tmp_path,
        [_request_end(ts=1000, session_id="s1", parent_session_id="s1")],
    )
    assert pb.traces, "self-parent session must parse as a root"


def test_parent_from_first_non_self_record_not_recs0(tmp_path: Path) -> None:
    """The parent header may appear only on later calls of a session; the chain parent must still be picked up (first non-self parent wins)."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="root"),
            _request_end(ts=1100, session_id="child"),
            _request_end(ts=1200, session_id="child", parent_session_id="root"),
        ],
    )
    # One root trace: child is linked under root, not emitted as a second root.
    assert len(pb.traces) == 1
    assert "multi-root" not in pb.traces[0].tags


def test_parent_found_when_earliest_record_is_parentless_tool_event(
    tmp_path: Path,
) -> None:
    """A parentless harness tool event as the session's EARLIEST record must not hide the parent from the later request_end (parent comes from the parent_link map, not any single record)."""
    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="root"),
            _tool_end(ts=1050, session_id="child"),  # no parent header
            _request_end(ts=1200, session_id="child", parent_session_id="root"),
        ],
    )
    assert len(pb.traces) == 1
    assert pb.traces[0].id == "root"
    assert "multi-root" not in pb.traces[0].tags
    child = pb.graph.nodes["child:0"]
    h = child.extra_headers
    assert h["x-dynamo-parent-session-id"] == "root"


def test_duplicate_records_across_dir_files_dedup(tmp_path: Path) -> None:
    """Dual sinks share one output path, so aggregated dirs can duplicate records; the same request_end must still lower to ONE node."""
    rec = _request_end(ts=1000, session_id="s1", request_id="r-dup")
    d = tmp_path / "capture"
    d.mkdir()
    write_jsonl(d / "a.jsonl", [rec])
    write_jsonl(d / "b.jsonl", [rec])

    pb = from_dynamo_trace(d)

    llm_nodes = [n for n in pb.graph.nodes.values() if isinstance(n, LlmNode)]
    assert len(llm_nodes) == 1, "duplicated request_end must lower to ONE node"


def test_can_load_sniffs_sink_envelope(tmp_path: Path) -> None:
    """Real dynamo captures wrap every line in {"timestamp","event"}, and both can_load and the adapter see through the envelope."""
    wrapped = {
        "timestamp": 5,
        "event": _request_end(ts=1000, session_id="s1"),
    }
    write_jsonl_gz(tmp_path / "trace.000000.jsonl.gz", [wrapped])

    assert DynamoTraceAdapter.can_load(tmp_path) is True
    assert from_dynamo_trace(tmp_path / "trace").traces


@pytest.mark.asyncio
async def test_session_headers_survive_interned_store_round_trip(
    tmp_path: Path,
) -> None:
    """The INTERNED unified store (what the worker actually reads via read_node_envelope) must carry extra_headers end to end."""
    from aiperf.dataset.graph.segment_trie.store_builder import (
        build_unified_trie_store_interned,
        flat_trie_ordinals,
    )
    from aiperf.dataset.graph_segment_unified_store import (
        GraphSegmentUnifiedBackingStore,
        GraphSegmentUnifiedClient,
    )
    from aiperf.graph.worker_materialize import read_node_envelope

    pb = _parse(
        tmp_path,
        [
            _request_end(ts=1000, session_id="s1"),
            _request_end(ts=1100, session_id="s1"),
        ],
    )
    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="bench-hdr"
    )
    await build_unified_trie_store_interned(pb, store)

    client = GraphSegmentUnifiedClient(
        base_path=tmp_path, benchmark_id="bench-hdr"
    ).open()
    trace = pb.traces[0]
    ordinals = flat_trie_ordinals(pb, trace)
    aid = "s1"
    env1 = read_node_envelope(client, trace.id, ordinals[f"{aid}:0"], "profiling")
    env2 = read_node_envelope(client, trace.id, ordinals[f"{aid}:1"], "profiling")
    assert env1 is not None and env2 is not None
    assert env1["extra_headers"]["x-dynamo-session-id"] == "s1"
    assert "x-dynamo-session-final" not in env1["extra_headers"]
    assert env2["extra_headers"]["x-dynamo-session-final"] == "true"
    assert "nvext" not in env1["dispatch_overrides"]


# --- read-time hash-id interning propagation --------------------------------


def test_lowered_hash_ids_share_interned_objects(tmp_path: Path) -> None:
    """Interned canonical hash objects survive lowering: an equal value shared across turns AND sessions stays ONE object across all nodes, while release_replay frees every recorded replay."""
    from aiperf.common.environment import Environment
    from aiperf.dataset.graph.adapters.dynamo.trace import _collect_chains
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    a, b, c, d = 2**63 + 1, 2**63 + 2, 2**63 + 3, 2**63 + 4
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [
            _request_end(ts=1000, session_id="s1", replay=_replay([a, b])),
            _request_end(ts=2000, session_id="s1", replay=_replay([a, b, c])),
            _request_end(ts=1500, session_id="s2", replay=_replay([a, d])),
        ],
    )
    chains = _collect_chains(p, None, max_depth=Environment.DYNAMO.MAX_SUBAGENT_DEPTH)
    nodes, _bs, _tags = dynamo_trie_nodes(chains, release_replay=True)

    ids_by_value: dict[int, set[int]] = {}
    for n in nodes:
        for h in n.request.hash_ids:
            ids_by_value.setdefault(h, set()).add(id(h))

    assert set(ids_by_value) == {a, b, c, d}
    # Every distinct value maps to exactly ONE object across all nodes.
    assert all(len(ids) == 1 for ids in ids_by_value.values()), ids_by_value
    # release_replay still nulled every recorded replay after the copy.
    assert all(
        t.record.request.replay is None for ch in chains.values() for t in ch.turns
    )


def test_intern_does_not_disturb_virtual_hash_fallback(tmp_path: Path) -> None:
    """Non-interference: a mixed recorded/virtual session lowers correctly through the interning path -- the recorded turn keeps its exact values and the non-replay turn extends the prefix with fresh negative virtual ids."""
    from aiperf.common.environment import Environment
    from aiperf.dataset.graph.adapters.dynamo.trace import _collect_chains
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    a, b = 2**63 + 1, 2**63 + 2
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [
            _request_end(ts=1000, session_id="s1", replay=_replay([a, b])),
            # No replay metadata -> virtual-hash fallback at lowering.
            _request_end(ts=2000, session_id="s1", input_tokens=64),
        ],
    )
    chains = _collect_chains(p, None, max_depth=Environment.DYNAMO.MAX_SUBAGENT_DEPTH)
    nodes, bs, tags = dynamo_trie_nodes(chains)

    assert "virtual-hash-fallback" in tags
    h1, h2 = nodes[0].request.hash_ids, nodes[1].request.hash_ids
    assert h1 == [a, b]
    assert h2[:2] == [a, b] and len(h2) == 64 // bs
    assert all(h < 0 for h in h2[2:])

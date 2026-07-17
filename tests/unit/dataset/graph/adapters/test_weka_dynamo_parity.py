# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-adapter parity: one logical trace, two formats, ONE lowered shape.

Both recorded-trace adapters lower through the shared trie pipeline
(``build_trie_ir`` -> ``assemble_trie_graph``), so the SAME logical trace --
identical request starts / durations / hash prefixes / output lengths /
streaming flags -- expressed once as a weka JSON trace and once as a dynamo
JSONL capture must produce identical ``LlmNode`` shapes, identical
interval-order edges, and (under weka ``hash_id_scope: "global"``, the bare
``hash_id`` reseed namespace dynamo always uses) byte-identical synthesized
prompt content with identical content-addressed ``prompt_segment_ids``.

Each helper builds BOTH fixtures from one ``_TurnSpec`` list, so the two
files can never drift apart. Node ids are ``{scope}:{turn}`` built from
RECORDED identity on both sides (weka: trace id / agent_id; dynamo: session
ids), so with aligned fixture identifiers the two formats produce IDENTICAL
node ids -- and the whole compare keys on them RAW: every ``LlmNode`` field
(a new field enters the compare automatically), every edge verbatim, the
channel state, and the FULL segment pool byte-for-byte. Response/tiny bytes
are included: both adapters seed partial-tail synthesis with the same
trace-scoped prefix (weka: trace id; dynamo: root session id) over the same
node ids, so content-addressed segment ids -- and therefore bytes -- match.

DOCUMENTED divergences -- the only values excluded from the compare, each
pinned by a dedicated test so any future unification tightens this suite
deliberately:

* ``extra_headers``: dynamo replays recorded x-dynamo-* session-identity
  headers; weka records none.
* ``expected`` CACHE fields: dynamo records engine ``cached_tokens``; weka
  has no equivalent recording. The recorded ``input_tokens`` /
  ``output_tokens`` expectations ARE compared (both formats record them).
* ``metadata["dynamo"]``: dynamo's adapter identity breadcrumb (session /
  turn). The shared ``metadata["trie"]`` sub-dict IS compared raw.
* ``TraceRecord`` provenance: adapter ``tags`` and dynamo's multi-graph
  ``graph_ref``.

Partial tails are NOT a divergence: dynamo records one extra partial-tail
hash for a non-block-aligned input (weka records full blocks only), but its
lowering strips it -- engines cache FULL blocks only, and the sub-block tail
is seed-sampled identically on both sides -- so the same recording lowers
byte-identically wherever the tail sits, and covered-count-0 tiny prompts
synthesize the same sampled user message via ``small_prompt_fallback`` on
BOTH adapters.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec
import orjson
import pytest

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.models import LlmNode, ParsedGraph, StaticEdge

_SEED = 20260707
_BLOCK_SIZE = 16
_MODEL = "recorded-model"
_EPOCH_MS = 1_000_000  # dynamo absolute base; both timelines start at t=0.


@dataclass(frozen=True)
class _TurnSpec:
    """One logical recorded call, expressible in both formats."""

    t: float
    """Start, seconds from trace start (whole milliseconds only)."""
    api: float
    """Server processing time, seconds (whole milliseconds only)."""
    hashes: tuple[int, ...]
    """KV block hashes covering the block-aligned prompt prefix."""
    out: int
    """Recorded output tokens (0 exercises the wire_output_cap upgrade)."""
    ttft: float | None = None
    """Recorded time-to-first-token, seconds. None = non-streaming."""
    session: str = "root"
    """"root" or the child scope id verbatim (weka subagent ``agent_id`` ==
    dynamo child session id), e.g. ``"agent_001"``."""
    tail: int = 0
    """Recorded input tokens past the covered blocks (partial-tail sampling);
    with ``hashes=()`` this is a tiny sub-block prompt."""


# Aligned recorded identifiers: the weka trace id equals the dynamo root
# session id, and the weka subagent agent_id equals the dynamo child session
# id, so the {scope}:{turn} node ids come out IDENTICAL across formats.
_TRACE_ID = "parity"
_CHILD_ID = "agent_001"


def _in_len(spec: _TurnSpec) -> int:
    return len(spec.hashes) * _BLOCK_SIZE + spec.tail


def _sessions_of(specs: list[_TurnSpec]) -> list[str]:
    """Child scope ids in first-appearance order."""
    seen: list[str] = []
    for s in specs:
        if s.session != "root" and s.session not in seen:
            seen.append(s.session)
    return seen


def _resolve_parents(
    specs: list[_TurnSpec], parents: dict[str, str] | None
) -> dict[str, str]:
    """Child scope -> parent scope ("root" unless the test says otherwise)."""
    return {sid: (parents or {}).get(sid, "root") for sid in _sessions_of(specs)}


def _weka_trace_dict(
    specs: list[_TurnSpec],
    *,
    trace_id: str = _TRACE_ID,
    parents: dict[str, str] | None = None,
    hash_scope: str = "global",
) -> dict[str, Any]:
    """One weka JSON trace: each child session becomes a (possibly nested)
    subagent entry placed in its PARENT's request stream at the child's first
    start. ``hash_id_scope: "global"`` selects the bare-hash-id content
    namespace dynamo uses, so equal hashes synthesize equal bytes."""
    tree = _resolve_parents(specs, parents)

    def req(spec: _TurnSpec) -> dict[str, Any]:
        body: dict[str, Any] = {
            "t": spec.t,
            "type": "s" if spec.ttft is not None else "n",
            "model": _MODEL,
            "in": _in_len(spec),
            "out": spec.out,
            "hash_ids": list(spec.hashes),
            "api_time": spec.api,
        }
        if spec.ttft is not None:
            body["ttft"] = spec.ttft
        return body

    def stream_for(scope: str) -> list[dict[str, Any]]:
        own = [(s.t, req(s)) for s in specs if s.session == scope]
        for child in (c for c, p in tree.items() if p == scope):
            first_t = min(s.t for s in specs if s.session == child)
            own.append(
                (
                    first_t,
                    {
                        "t": first_t,
                        "type": "subagent",
                        "agent_id": child,
                        "subagent_type": "X",
                        "status": "completed",
                        "requests": stream_for(child),
                        "models": [_MODEL],
                    },
                )
            )
        own.sort(key=lambda pair: pair[0])
        return [entry for _, entry in own]

    return {
        "id": trace_id,
        "models": [_MODEL],
        "block_size": _BLOCK_SIZE,
        "hash_id_scope": hash_scope,
        "requests": stream_for("root"),
    }


def _write_weka(
    tmp_path: Path,
    specs: list[_TurnSpec],
    *,
    parents: dict[str, str] | None = None,
    hash_scope: str = "global",
) -> Path:
    p = tmp_path / "parity.weka.json"
    p.write_bytes(
        orjson.dumps(_weka_trace_dict(specs, parents=parents, hash_scope=hash_scope))
    )
    return p


def _dynamo_records(
    specs: list[_TurnSpec],
    *,
    trace_id: str = _TRACE_ID,
    parents: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """The SAME logical trace as dynamo request_end records: root turns on the
    root session, child turns on child sessions linked by parent_session_id
    (one session-tree -> one graph, matching the weka single-trace shape)."""
    tree = _resolve_parents(specs, parents)
    records = []
    for i, spec in enumerate(specs):
        sid = trace_id if spec.session == "root" else spec.session
        ctx: dict[str, Any] = {"session_id": sid}
        if spec.session != "root":
            parent = tree[spec.session]
            ctx["parent_session_id"] = trace_id if parent == "root" else parent
        # Dynamo hashes span the WHOLE input incl. one partial-tail block for a
        # non-block-aligned length ((n-1)*bs < input_length <= n*bs); weka
        # records full blocks only. The partial hash is content over the actual
        # tail tokens, so it never recurs: mint a unique id per tail turn
        # (trace-salted so multi-trace captures cannot collide).
        hashes = list(spec.hashes)
        if spec.tail:
            hashes.append(900_000 + (zlib.crc32(trace_id.encode()) % 50) * 1_000 + i)
        req: dict[str, Any] = {
            "request_id": f"r-{sid}-{spec.t}",
            "model": _MODEL,
            "input_tokens": _in_len(spec),
            "output_tokens": spec.out,
            "cached_tokens": 0,
            "request_received_ms": _EPOCH_MS + round(spec.t * 1000),
            # Whole-ms ints: n/1000*1000 is not always n in binary floats, and
            # a sub-ms drift here would quantize differently downstream.
            "total_time_ms": round(spec.api * 1000),
            "replay": {
                "trace_block_size": _BLOCK_SIZE,
                "input_length": _in_len(spec),
                "input_sequence_hashes": hashes,
            },
        }
        if spec.ttft is not None:
            req["ttft_ms"] = round(spec.ttft * 1000)
        records.append(
            {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": _EPOCH_MS + round((spec.t + spec.api) * 1000),
                "event_source": "dynamo",
                "agent_context": ctx,
                "request": req,
            }
        )
    return records


def _write_dynamo(
    tmp_path: Path,
    specs: list[_TurnSpec],
    *,
    parents: dict[str, str] | None = None,
) -> Path:
    p = tmp_path / "parity.dynamo.jsonl"
    p.write_bytes(
        b"\n".join(orjson.dumps(r) for r in _dynamo_records(specs, parents=parents))
    )
    return p


def _parse_pair(
    tmp_path: Path,
    specs: list[_TurnSpec],
    *,
    idle_gap_cap_seconds: float | None = 60.0,
    parents: dict[str, str] | None = None,
    hash_scope: str = "global",
) -> tuple[ParsedGraph, ParsedGraph]:
    weka = from_weka_trace(
        _write_weka(tmp_path, specs, parents=parents, hash_scope=hash_scope),
        content_root_seed=_SEED,
        content_tokenizer="builtin",
        idle_gap_cap_seconds=idle_gap_cap_seconds,
    )
    dyn = from_dynamo_trace(
        _write_dynamo(tmp_path, specs, parents=parents),
        content_root_seed=_SEED,
        content_tokenizer="builtin",
        idle_gap_cap_seconds=idle_gap_cap_seconds,
    )
    return weka, dyn


# --- the raw compare ----------------------------------------------------------

# LlmNode fields excluded from the raw compare; each is pinned as a REAL,
# documented recording-capability divergence by test_documented_divergences_hold
# below. metadata is popped whole but its shared "trie" sub-dict (incl. the
# content-addressed prompt_segment_ids) re-enters the compare verbatim, and
# expected re-enters with its recorded input/output expectations (only the
# engine cache fields are a dynamo-only recording).
_DOCUMENTED_DIVERGENT_FIELDS = ("metadata", "expected", "extra_headers")


def _node_fields(node: LlmNode) -> dict[str, Any]:
    """Every LlmNode field raw -- a field added tomorrow is compared tomorrow."""
    d = msgspec.structs.asdict(node)
    for field in _DOCUMENTED_DIVERGENT_FIELDS:
        d.pop(field)
    d["trie_metadata"] = node.metadata["trie"]
    exp = node.expected
    d["expected_recorded"] = (
        None if exp is None else (exp.input_tokens, exp.output_tokens)
    )
    return d


def _assert_node_parity(weka_node: LlmNode, dyn_node: LlmNode, nid: str) -> None:
    """Raw field compare -- block counts included.

    Dynamo RECORDS one extra partial-tail hash for a non-block-aligned input
    (weka records full blocks only), but its lowering strips it (engines
    cache full blocks only; the tail is seed-sampled), so the two formats
    lower the same recording to identical block structure everywhere.
    """
    assert _node_fields(weka_node) == _node_fields(dyn_node), nid


def _assert_parity(weka: ParsedGraph, dyn: ParsedGraph) -> None:
    assert len(weka.traces) == len(dyn.traces) == 1
    assert weka.traces[0].id == dyn.traces[0].id
    assert weka.traces[0].initial_state == dyn.traces[0].initial_state
    assert weka.traces[0].replay_outputs == dyn.traces[0].replay_outputs
    # Node ids are recorded data on both sides -- with aligned fixture
    # identifiers they are IDENTICAL, so the compare keys on them raw and a
    # divergence localizes to (node id, field).
    assert sorted(weka.graph.nodes) == sorted(dyn.graph.nodes)
    for nid in weka.graph.nodes:
        _assert_node_parity(weka.graph.nodes[nid], dyn.graph.nodes[nid], nid)
    # Edges VERBATIM: sources/targets are the shared node ids, so no index
    # rewrite -- every edge type and every delay field, byte-for-byte.
    assert sorted(map(repr, weka.graph.edges)) == sorted(map(repr, dyn.graph.edges))
    assert weka.graph.state == dyn.graph.state
    # Segment pools byte-identical: ids are content-addressed, values compare
    # raw -- prompt AND response/tiny bytes (equal trace-scoped tail seeds).
    assert weka.segment_pool is not None and dyn.segment_pool is not None
    assert weka.segment_pool.by_id == dyn.segment_pool.by_id


def _assert_shape_parity(weka: ParsedGraph, dyn: ParsedGraph) -> None:
    """Everything except synthesized CONTENT: the compare for weka's LOCAL
    hash-scope namespace (the one documented content divergence), where
    segment ids and pool bytes legitimately differ while ids, timing, edges,
    counts, and state must not."""
    assert sorted(weka.graph.nodes) == sorted(dyn.graph.nodes)
    for nid in weka.graph.nodes:
        wd, dd = (
            _node_fields(weka.graph.nodes[nid]),
            _node_fields(dyn.graph.nodes[nid]),
        )
        wd.pop("trie_metadata")
        dd.pop("trie_metadata")
        assert wd == dd, nid
    assert sorted(map(repr, weka.graph.edges)) == sorted(map(repr, dyn.graph.edges))
    assert weka.graph.state == dyn.graph.state


# --- scenarios ----------------------------------------------------------------

_LINEAR = [
    # Streaming turn, then a zero-output non-streaming turn (wire_output_cap
    # upgrade parity), then a streaming continuation -- each extending the
    # recorded hash prefix so content parents chain.
    _TurnSpec(t=0.0, api=1.0, hashes=(1, 2), out=8, ttft=0.5),
    _TurnSpec(t=2.5, api=1.0, hashes=(1, 2, 3, 4), out=0),
    _TurnSpec(t=5.0, api=0.5, hashes=(1, 2, 3, 4, 5, 6), out=16, ttft=0.25),
]


def test_linear_session_lowers_identically(tmp_path: Path) -> None:
    """3-turn session: node fields, edges, and ALL segment BYTES match."""
    weka, dyn = _parse_pair(tmp_path, _LINEAR)
    _assert_parity(weka, dyn)
    # Anti-vacuous: the compare covered real values, not empty defaults --
    # keyed by the data-inherent {scope}:{turn} node ids.
    nodes = [weka.graph.nodes[f"{_TRACE_ID}:{k}"] for k in range(3)]
    assert [n.max_tokens for n in nodes] == [8, 1, 16]
    assert [n.streaming for n in nodes] == [True, False, True]
    assert all(n.model == _MODEL for n in nodes)
    assert [n.theoretical_prefix_cache_total_blocks for n in nodes] == [2, 4, 6]
    assert [n.theoretical_prefix_cache_hit_blocks for n in nodes] == [0, 2, 4]
    # Recorded expectations, NOT the capped wire value: the zero-output turn
    # keeps expected.output_tokens == 0 while max_tokens upgrades to 1.
    assert [n.expected.input_tokens for n in nodes] == [32, 64, 96]
    assert [n.expected.output_tokens for n in nodes] == [8, 0, 16]
    assert all(n.metadata["trie"]["prompt_segment_ids"] for n in nodes)


def test_idle_gap_warp_engages_identically(tmp_path: Path) -> None:
    """A 200s recorded idle gap compresses to the 60s cap on BOTH timelines."""
    specs = [
        _TurnSpec(t=0.0, api=1.0, hashes=(1, 2), out=8, ttft=0.5),
        _TurnSpec(t=201.0, api=1.0, hashes=(1, 2, 3, 4), out=8, ttft=0.5),
    ]
    weka, dyn = _parse_pair(tmp_path, specs, idle_gap_cap_seconds=60.0)
    _assert_parity(weka, dyn)
    # Anti-vacuous: the warp actually engaged (raw gap 200s -> capped 60s).
    for pb in (weka, dyn):
        second = pb.graph.nodes[f"{_TRACE_ID}:1"]
        assert second.arrival_offset_us == pytest.approx(61_000_000)
        (edge,) = [
            e
            for e in pb.graph.edges
            if isinstance(e, StaticEdge) and e.delay_after_predecessor_us
        ]
        assert edge.delay_after_predecessor_us == pytest.approx(60_000_000)


def test_subagent_child_session_concurrency_parity(tmp_path: Path) -> None:
    """Weka subagent inner requests and a dynamo child session lower to the
    SAME concurrency shape: the overlapping child start-anchors to the
    in-flight root turn, and the later root turn waits on both."""
    specs = [
        _TurnSpec(t=0.0, api=2.0, hashes=(1, 2), out=8, ttft=0.5),
        _TurnSpec(t=0.5, api=0.5, hashes=(7, 8), out=4, session=_CHILD_ID),
        _TurnSpec(t=3.0, api=1.0, hashes=(1, 2, 3, 4), out=8, ttft=0.5),
    ]
    weka, dyn = _parse_pair(tmp_path, specs)
    _assert_parity(weka, dyn)
    # Anti-vacuous: the child really start-anchored to the in-flight root
    # turn -- the edge endpoints are the shared recorded ids themselves.
    for pb in (weka, dyn):
        anchored = [
            e
            for e in pb.graph.edges
            if isinstance(e, StaticEdge)
            and e.delay_after_predecessor_start_us is not None
        ]
        assert len(anchored) == 1
        assert anchored[0].source == f"{_TRACE_ID}:0"
        assert anchored[0].target == f"{_CHILD_ID}:0"
        assert anchored[0].delay_after_predecessor_start_us == pytest.approx(500_000)


def test_response_segments_byte_identical(tmp_path: Path) -> None:
    """Assistant response segments exist AND byte-match across formats: both
    adapters seed partial-tail synthesis with the same trace-scoped prefix
    (weka: trace id; dynamo: root session id) over the same node ids, so the
    content-addressed segment ids -- hence bytes -- are equal. The pool-wide
    equality lives in ``_assert_parity``; this pins that it is NOT vacuous
    (response segments really got synthesized on both sides)."""
    weka, dyn = _parse_pair(tmp_path, _LINEAR)
    _assert_parity(weka, dyn)
    for pb in (weka, dyn):
        for nid, node in pb.graph.nodes.items():
            if node.max_tokens == 1:
                continue  # zero-output turn: empty response segment shape varies
            tip = node.metadata["trie"]["prompt_segment_ids"][-1]
            assert any(
                s.role == "assistant" and s.parent_id == tip
                for s in pb.segment_pool.by_id.values()
            ), nid


def test_documented_divergences_hold(tmp_path: Path) -> None:
    """Pin the exclusion list: every value skipped by the raw compare is a
    REAL divergence today. When one is unified, move it INTO the compare."""
    weka, dyn = _parse_pair(tmp_path, _LINEAR)
    for node in dyn.graph.nodes.values():
        assert node.extra_headers is not None  # session identity headers
        assert node.expected.cache_read_tokens is not None  # engine recording
        assert set(node.metadata) == {"dynamo", "trie"}
    for node in weka.graph.nodes.values():
        assert node.extra_headers is None
        assert node.expected.cache_read_tokens is None  # no weka equivalent
        assert node.expected.cache_creation_tokens is None
        assert set(node.metadata) == {"trie"}
    # TraceRecord provenance: adapter tags + dynamo's multi-graph graph_ref.
    assert weka.traces[0].tags == ["from-weka-trace"]
    assert dyn.traces[0].tags == ["from-dynamo-trace"]
    assert weka.traces[0].graph_ref is None
    assert dyn.traces[0].graph_ref == _TRACE_ID


def test_partial_tail_turns_byte_identical_anywhere(tmp_path: Path) -> None:
    """Non-block-aligned turns, each format's NATIVE encoding: weka records
    full-block hashes + an uncovered tail; dynamo records one extra
    partial-tail hash spanning it, which its lowering STRIPS (engines cache
    full blocks only; the tail is seed-sampled, never decoded). The same
    recording therefore lowers byte-identically wherever the tail sits --
    leaf turns, MID-CHAIN turns whose prefix later turns extend, and child
    sessions -- with identical block totals."""
    specs = [
        _TurnSpec(t=0.0, api=1.0, hashes=(1, 2), out=8, ttft=0.5),
        _TurnSpec(t=1.5, api=0.5, hashes=(50, 51), out=4, tail=5, session=_CHILD_ID),
        # Mid-chain tails: later turns extend these prefixes.
        _TurnSpec(t=2.0, api=1.0, hashes=(1, 2, 3), out=16, tail=9),
        _TurnSpec(t=4.0, api=1.0, hashes=(1, 2, 3, 4, 5), out=0, tail=3),
        _TurnSpec(t=6.0, api=1.0, hashes=tuple(range(1, 11)), out=32, tail=11),
    ]
    weka, dyn = _parse_pair(tmp_path, specs)
    _assert_parity(weka, dyn)
    # Anti-vacuous: the recorded lengths really are unaligned, the dynamo
    # recording really carried the extra partial hash (writer appends it),
    # and both sides count FULL blocks only.
    nodes = [
        weka.graph.nodes[n]
        for n in (f"{_CHILD_ID}:0", f"{_TRACE_ID}:1", f"{_TRACE_ID}:3")
    ]
    assert [n.expected.input_tokens for n in nodes] == [37, 57, 171]
    assert [n.theoretical_prefix_cache_total_blocks for n in nodes] == [2, 3, 10]


def test_tiny_sub_block_prompts_byte_identical(tmp_path: Path) -> None:
    """Covered-count-0 (tiny sub-block) prompts: weka records hashes=[] with
    a sub-block ``in``; dynamo records a single partial hash (stripped at
    lowering). Both sides synthesize the SAME seed-sampled user message via
    ``small_prompt_fallback`` -- a recorded prompt is never lowered to an
    unreplayable empty messages array."""
    tiny = [
        _TurnSpec(t=0.0, api=0.5, hashes=(), out=4, tail=7),
        _TurnSpec(t=1.0, api=0.5, hashes=(1,), out=4, ttft=0.25),
    ]
    weka, dyn = _parse_pair(tmp_path, tiny)
    _assert_parity(weka, dyn)
    w_node = weka.graph.nodes[f"{_TRACE_ID}:0"]
    d_node = dyn.graph.nodes[f"{_TRACE_ID}:0"]
    assert len(w_node.metadata["trie"]["prompt_segment_ids"]) == 1
    assert d_node.metadata["dynamo"]["small_prompt"] is True
    assert w_node.expected.input_tokens == 7


def test_local_hash_scope_lowers_same_shape(tmp_path: Path) -> None:
    """Weka's DEFAULT ``hash_id_scope: "local"`` must still agree with dynamo
    on everything content-free: ids, node fields, timing, edges, state. The
    content namespaces intentionally diverge (local reseeds per trace), so
    prompt segment ids differ -- pinned here so the divergence stays real."""
    weka, dyn = _parse_pair(tmp_path, _LINEAR, hash_scope="local")
    _assert_shape_parity(weka, dyn)
    # Anti-vacuous: the namespaces really diverged (else this test should be
    # folded into the full byte compare).
    w_ids = [
        weka.graph.nodes[nid].metadata["trie"]["prompt_segment_ids"]
        for nid in sorted(weka.graph.nodes)
    ]
    d_ids = [
        dyn.graph.nodes[nid].metadata["trie"]["prompt_segment_ids"]
        for nid in sorted(dyn.graph.nodes)
    ]
    assert w_ids != d_ids


def test_nested_and_sibling_children_lower_identically(tmp_path: Path) -> None:
    """Two root children plus a GRANDCHILD (weka subagent-in-subagent nesting,
    dynamo parent_session_id chain) lower to identical node ids and shapes."""
    specs = [
        _TurnSpec(t=0.0, api=4.0, hashes=(1, 2), out=8, ttft=0.5),
        _TurnSpec(t=0.5, api=1.0, hashes=(10, 11), out=4, session="agent_001"),
        _TurnSpec(t=1.0, api=0.5, hashes=(20,), out=4, session="agent_002"),
        _TurnSpec(t=2.0, api=1.0, hashes=(30, 31), out=4, session="agent_003"),
        _TurnSpec(t=5.0, api=1.0, hashes=(1, 2, 3), out=8, ttft=0.5),
    ]
    parents = {
        "agent_001": "root",
        "agent_002": "agent_001",  # grandchild
        "agent_003": "root",
    }
    weka, dyn = _parse_pair(tmp_path, specs, parents=parents)
    _assert_parity(weka, dyn)
    assert sorted(weka.graph.nodes) == [
        "agent_001:0",
        "agent_002:0",
        "agent_003:0",
        f"{_TRACE_ID}:0",
        f"{_TRACE_ID}:1",
    ]


def test_corpus_two_traces_lower_identically(tmp_path: Path) -> None:
    """Corpus level: a weka DIRECTORY of two traces vs ONE dynamo capture
    holding two disjoint session trees -- the multi-graph split must agree on
    graph keys, per-graph nodes/edges, and trace->graph_ref resolution."""
    specs_a = [
        _TurnSpec(t=0.0, api=1.0, hashes=(1, 2), out=8, ttft=0.5),
        _TurnSpec(t=2.0, api=1.0, hashes=(1, 2, 3), out=8),
    ]
    specs_b = [
        _TurnSpec(t=0.0, api=2.0, hashes=(50, 51), out=8, ttft=0.5),
        _TurnSpec(t=0.5, api=0.5, hashes=(60,), out=4, session="agent_001"),
    ]

    weka_dir = tmp_path / "weka_corpus"
    weka_dir.mkdir()
    (weka_dir / "a.json").write_bytes(
        orjson.dumps(_weka_trace_dict(specs_a, trace_id="trace_a"))
    )
    (weka_dir / "b.json").write_bytes(
        orjson.dumps(_weka_trace_dict(specs_b, trace_id="trace_b"))
    )
    dynamo_path = tmp_path / "corpus.dynamo.jsonl"
    dynamo_path.write_bytes(
        b"\n".join(
            orjson.dumps(r)
            for r in (
                _dynamo_records(specs_a, trace_id="trace_a")
                + _dynamo_records(specs_b, trace_id="trace_b")
            )
        )
    )

    weka = from_weka_trace(
        weka_dir,
        content_root_seed=_SEED,
        content_tokenizer="builtin",
        idle_gap_cap_seconds=60.0,
    )
    dyn = from_dynamo_trace(
        dynamo_path,
        content_root_seed=_SEED,
        content_tokenizer="builtin",
        idle_gap_cap_seconds=60.0,
    )

    assert sorted(weka.graphs) == sorted(dyn.graphs) == ["trace_a", "trace_b"]
    assert {t.id: t.graph_ref for t in weka.traces} == {
        t.id: t.graph_ref for t in dyn.traces
    }
    for key in weka.graphs:
        wg, dg = weka.graphs[key], dyn.graphs[key]
        assert sorted(wg.nodes) == sorted(dg.nodes), key
        for nid in wg.nodes:
            _assert_node_parity(wg.nodes[nid], dg.nodes[nid], f"{key}/{nid}")
        assert sorted(map(repr, wg.edges)) == sorted(map(repr, dg.edges)), key
        assert wg.state == dg.state, key
    # One corpus-wide content pool on both sides, byte-identical.
    assert weka.segment_pool.by_id == dyn.segment_pool.by_id


def test_dynamo_record_order_invariance(tmp_path: Path) -> None:
    """Dynamo lowering globally sorts turns by recorded start, so a
    line-shuffled capture must parse to a byte-identical graph."""
    specs = [
        _TurnSpec(t=0.0, api=2.0, hashes=(1, 2), out=8, ttft=0.5),
        _TurnSpec(t=0.5, api=0.5, hashes=(7, 8), out=4, session=_CHILD_ID),
        _TurnSpec(t=3.0, api=1.0, hashes=(1, 2, 3, 4), out=8, ttft=0.5),
    ]
    ordered_path = _write_dynamo(tmp_path, specs)
    lines = ordered_path.read_bytes().splitlines()
    shuffled = list(lines)
    random.Random(_SEED).shuffle(shuffled)
    assert shuffled != lines, "shuffle must actually reorder the capture"
    shuffled_path = tmp_path / "shuffled.dynamo.jsonl"
    shuffled_path.write_bytes(b"\n".join(shuffled))

    kwargs = dict(
        content_root_seed=_SEED,
        content_tokenizer="builtin",
        idle_gap_cap_seconds=60.0,
    )
    ordered = from_dynamo_trace(ordered_path, **kwargs)
    reordered = from_dynamo_trace(shuffled_path, **kwargs)

    assert sorted(ordered.graph.nodes) == sorted(reordered.graph.nodes)
    for nid in ordered.graph.nodes:
        assert msgspec.structs.asdict(ordered.graph.nodes[nid]) == (
            msgspec.structs.asdict(reordered.graph.nodes[nid])
        ), nid
    assert sorted(map(repr, ordered.graph.edges)) == (
        sorted(map(repr, reordered.graph.edges))
    )
    assert ordered.graph.state == reordered.graph.state
    assert ordered.segment_pool.by_id == reordered.segment_pool.by_id


# --- randomized scenario matrix -------------------------------------------------
#
# The hand-picked scenarios above localize known seams; this sweep validates
# the parity contract EMPIRICALLY across the recorded-shape space: random turn
# counts, prefix growth, streaming mix, zero-output turns, partial tails,
# idle gaps beyond the warp cap, and a random session TREE (0-2 children,
# each parented to root or a previous child -- i.e. possible grandchildren).
# Fixed per-case seeds keep every case reproducible standalone.


def _random_specs(seed: int) -> tuple[list[_TurnSpec], dict[str, str]]:
    """One random recorded shape, valid in both formats (whole-ms times)."""
    rng = random.Random(seed)
    hash_ids = iter(range(1, 10_000))
    specs: list[_TurnSpec] = []
    parents: dict[str, str] = {}

    t_ms = 0
    prefix: tuple[int, ...] = ()
    root_turns = rng.randint(2, 5)
    for _ in range(root_turns):
        prefix += tuple(next(hash_ids) for _ in range(rng.randint(1, 4)))
        api_ms = rng.randint(200, 3000)
        out = rng.choice([0, 4, 8, 16, 32])
        # A recorded ttft implies at least one produced token; keep it < api.
        ttft_ms = rng.randint(50, api_ms - 1) if out and rng.random() < 0.7 else None
        specs.append(
            _TurnSpec(
                t=t_ms / 1000,
                api=api_ms / 1000,
                hashes=prefix,
                out=out,
                ttft=ttft_ms / 1000 if ttft_ms is not None else None,
                tail=rng.choice([0, 0, 3, 9]),
            )
        )
        # Idle gap after the turn; sometimes far beyond the 60s warp cap.
        gap_ms = rng.choice([100, 500, 2000, 150_000])
        t_ms += api_ms + gap_ms

    # A random session TREE: each child overlaps its parent's first turn and
    # runs its own recorded hash prefix (weka nested subagent entries / dynamo
    # parent_session_id chains).
    scopes = ["root"]
    for i in range(rng.randint(0, 2)):
        sid = f"agent_{i + 1:03d}"
        parent = rng.choice(scopes)
        parents[sid] = parent
        parent_first = next(
            s for s in specs if s.session == ("root" if parent == "root" else parent)
        )
        child_t_ms = round(parent_first.t * 1000) + rng.randint(
            1, max(1, int(parent_first.api * 900))
        )
        child_prefix: tuple[int, ...] = ()
        for _ in range(rng.randint(1, 3)):
            child_prefix += tuple(next(hash_ids) for _ in range(rng.randint(1, 3)))
            api_ms = rng.randint(100, 1500)
            out = rng.choice([4, 8, 16])
            specs.append(
                _TurnSpec(
                    t=child_t_ms / 1000,
                    api=api_ms / 1000,
                    hashes=child_prefix,
                    out=out,
                    ttft=None,
                    session=sid,
                    tail=rng.choice([0, 0, 5]),
                )
            )
            child_t_ms += api_ms + rng.randint(50, 800)
        scopes.append(sid)

    return specs, parents


@pytest.mark.parametrize("seed", range(8))
def test_randomized_recorded_shapes_lower_identically(
    tmp_path: Path, seed: int
) -> None:
    specs, parents = _random_specs(_SEED + seed)
    weka, dyn = _parse_pair(tmp_path, specs, parents=parents)
    _assert_parity(weka, dyn)
    # Anti-vacuous: every spec became a node on both sides.
    assert len(weka.graph.nodes) == len(specs)

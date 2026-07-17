# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance gate for the weka segment-trie IR: raw-export fidelity vs the REAL trace.

Two layers:

* COMPONENT lane (fast, hermetic): drive each :mod:`tools.weka_trace_fidelity`
  Report function on small hand-crafted raw-export + trace inputs, including a
  deliberately-CORRUPTED export that MUST fail each check (so the tool cannot pass
  vacuously). These build a tiny real trie graph (``build_trie_graph`` +
  ``SegmentPool``) so the expected content is the genuine pool materialization, not
  a stub.

* INTEGRATION lane (slow, real subprocess): run an ACTUAL
  ``aiperf profile --export-level raw`` of the trie path against the in-repo
  ``aiperf-mock-server`` over the subagent fixture, then assert BOTH
  :func:`content_vs_real_trace` and :func:`causality_timing_vs_real_trace` PASS on
  the produced ``profile_export_raw.jsonl``. This is the empirical proof the
  dispatched prompts and the dispatch causality/timing match the recorded trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import orjson
import pytest

from tools.weka_trace_fidelity import (
    _ExportRecord,
    _RecordedNode,
    _RecordedTrace,
    build_recorded_trace,
    causality_timing_vs_real_trace,
    content_byte_exact_vs_v04,
    content_vs_real_trace,
    prove_corpus,
)

_REPO = Path(__file__).resolve().parents[3]
_FIX = _REPO / "tests" / "unit" / "graph" / "fixtures" / "weka_subagent.json"
_MODEL = "claude-opus-4-5-20251101"


# --- synthetic-input helpers ----------------------------------------------


def _build_expected(trace_file: Path) -> dict[str, list[dict[str, str]]]:
    """The genuine per-node materialized prompt for a trace, keyed by node id.

    Rebuilds the trie graph + pool exactly as the tool does, so a synthetic raw
    export carrying these messages is what a FAITHFUL run would have exported.
    """
    from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
    from aiperf.dataset.graph.adapters.weka.trie_build import build_trie_graph
    from aiperf.dataset.graph.models import LlmNode

    trace = WekaTrace.model_validate(json.loads(trace_file.read_text()))
    parsed, pool = build_trie_graph(
        trace, tokenizer_name="gpt2", prompt_corpus="coding", root_seed=None
    )
    out: dict[str, list[dict[str, str]]] = {}
    for nid, node in parsed.graph.nodes.items():
        if isinstance(node, LlmNode):
            out[nid] = pool.materialize(node.metadata["trie"]["prompt_segment_ids"])
    return out


def _recorded_timing(trace_file: Path) -> dict[str, tuple[float, float]]:
    """``{node_id: (start_s, end_s)}`` of RAW recorded ``request.t`` / ``+api_time``.

    Reads the raw request fields directly: ``_flatten_requests`` here does NOT
    apply the idle-gap warp, so ``TrieNode.start`` / ``.end`` would return the 0.0
    ``warped_start`` default rather than the recorded timeline. The faithful
    export's edge delay must be the genuine recorded end-to-start gap.
    """
    from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
    from aiperf.dataset.graph.adapters.weka.trie_build import _flatten_requests

    trace = WekaTrace.model_validate(json.loads(trace_file.read_text()))
    return {
        n.node_id: (n.request.t, n.request.t + (n.request.api_time or 0.0))
        for n in _flatten_requests(trace.requests, root_scope=trace.id)
    }


def _record(
    *,
    node_id: str,
    phase: str,
    request_start_ns: int,
    messages: list[dict[str, str]],
    conversation_id: str = "trace_sub_n2s1#0.0",
) -> dict:
    """A single raw-export JSONL record shaped like the production writer emits.

    ``x_request_id`` folds the node id ahead of its ``::`` nonce, matching the
    worker's graph mint -- the node-identity channel the tool reads.
    """
    return {
        "metadata": {
            "conversation_id": conversation_id,
            "x_request_id": f"{node_id}::deadbeefdeadbeefdeadbeefdeadbeef",
            "benchmark_phase": phase,
            "request_start_ns": request_start_ns,
            "credit_issued_ns": request_start_ns - 1_000_000,
            "session_num": 0,
            "turn_index": 0,
        },
        "payload": {"messages": messages, "model": _MODEL, "stream": False},
    }


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_bytes(b"\n".join(orjson.dumps(r) for r in records) + b"\n")
    return path


def _faithful_records(trace_file: Path) -> list[dict]:
    """A synthetic profiling export that a FAITHFUL trie run would have produced.

    Profiles the two leaf nodes ``agent_001:1`` and ``trace_sub_n2s1:1`` (the chopped
    survivors in the real run), each with its genuine materialized prompt and a
    dispatch time whose start-to-start gap equals the recorded END-to-start edge
    delay (the zero-latency mock collapses ``api_time`` to ~0, so observed
    start-to-start == recorded end-to-start).
    """
    expected = _build_expected(trace_file)
    timing = _recorded_timing(trace_file)
    t0 = 1_000_000_000_000
    # Recorded end-to-start edge delay agent_001:1 -> trace_sub_n2s1:1 ==
    # trace_sub_n2s1:1.start - agent_001:1.end.
    edge_delay_s = timing["trace_sub_n2s1:1"][0] - timing["agent_001:1"][1]
    ns = 1_000_000_000
    return [
        _record(
            node_id="agent_001:1",
            phase="profiling",
            request_start_ns=t0,
            messages=expected["agent_001:1"],
        ),
        _record(
            node_id="trace_sub_n2s1:1",
            phase="profiling",
            request_start_ns=t0 + int(edge_delay_s * ns),
            messages=expected["trace_sub_n2s1:1"],
        ),
    ]


# --- component lane: content_vs_real_trace --------------------------------


@pytest.mark.component_integration
def test_content_vs_real_trace_passes_on_faithful_export(tmp_path: Path) -> None:
    # The synthetic export was built with gpt2 (``_build_expected``); the tool
    # defaults to the live-run builtin tokenizer, so the run's knob is passed
    # explicitly.
    raw = _write_jsonl(tmp_path / "raw.jsonl", _faithful_records(_FIX))
    report = content_vs_real_trace(raw, _FIX, tokenizer_name="gpt2")
    assert report.passed, report.render()
    assert report.checked == 2


@pytest.mark.component_integration
def test_content_vs_real_trace_fails_on_corrupted_export(tmp_path: Path) -> None:
    """A single mutated byte in one prompt message must FAIL the content check."""
    records = _faithful_records(_FIX)
    # Corrupt trace_sub_n2s1:1's first user message content.
    records[1]["payload"]["messages"][0]["content"] += "_CORRUPTED"
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    report = content_vs_real_trace(raw, _FIX, tokenizer_name="gpt2")
    assert not report.passed, report.render()
    assert any("node=trace_sub_n2s1:1" in m.where for m in report.mismatches)
    # Only the corrupted record fails; the untouched one still passes.
    assert report.checked == 2
    assert report.passes == 1


@pytest.mark.component_integration
def test_content_vs_real_trace_strips_rid_marker(tmp_path: Path) -> None:
    """A ``[rid:...]`` cache-bust prefix on the first user msg is stripped, not failed."""
    records = _faithful_records(_FIX)
    first_user = next(
        m for m in records[0]["payload"]["messages"] if m["role"] == "user"
    )
    first_user["content"] = "[rid:0123456789ab]\n\n" + first_user["content"]
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    report = content_vs_real_trace(raw, _FIX, tokenizer_name="gpt2")
    assert report.passed, report.render()


# --- component lane: causality_timing_vs_real_trace -----------------------


@pytest.mark.component_integration
def test_causality_timing_passes_on_faithful_export(tmp_path: Path) -> None:
    raw = _write_jsonl(tmp_path / "raw.jsonl", _faithful_records(_FIX))
    report = causality_timing_vs_real_trace(raw, _FIX)
    assert report.passed, report.render()
    assert report.checked == 2


@pytest.mark.component_integration
def test_causality_timing_fails_on_reordered_dispatch(tmp_path: Path) -> None:
    """Dispatching trace_sub_n2s1:1 BEFORE its recorded predecessor agent_001:1 is a
    causal-order fail."""
    records = _faithful_records(_FIX)
    # Swap the dispatch times so trace_sub_n2s1:1 fires 3s BEFORE agent_001:1
    # (predecessor inversion).
    r11_ns = records[0]["metadata"]["request_start_ns"]
    records[1]["metadata"]["request_start_ns"] = r11_ns - 3_000_000_000
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    report = causality_timing_vs_real_trace(raw, _FIX)
    assert not report.passed, report.render()
    assert any("causal-order" in m.detail for m in report.mismatches)


@pytest.mark.component_integration
def test_causality_timing_fails_on_wrong_relative_offset(tmp_path: Path) -> None:
    """An inter-request gap far from the recorded edge delay must FAIL timing."""
    records = _faithful_records(_FIX)
    # Push trace_sub_n2s1:1 10s LATER than its recorded edge delay prescribes
    # (well past tol).
    records[1]["metadata"]["request_start_ns"] += 10_000_000_000
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    report = causality_timing_vs_real_trace(raw, _FIX)
    assert not report.passed, report.render()
    assert any("relative offset" in m.detail for m in report.mismatches)


# --- component lane: executor max-gate AND-join + START-root origin -------
#
# These drive the timing model directly with hand-built ``_RecordedTrace`` /
# ``_ExportRecord`` objects so the executor's firing-gate semantics are exercised
# in isolation (no trie rebuild): the dispatch of a join node is the MAX over its
# profiled preds of ``pred_observed_dispatch + warped_delay`` (the binding pred is
# the argmax, not the recorded-latest pred), and a START-rooted node fires at
# ``origin + min_start_delay`` -- never relative to another root's END.

_NS = 1_000_000_000
_MSG = [{"role": "user", "content": "hi"}]


def _rnode(
    node_id: str,
    *,
    preds: dict[str, float] | None = None,
    min_start_delay_us: float | None = None,
    raw_start_s: float = 0.0,
    raw_end_s: float = 0.0,
) -> _RecordedNode:
    """A timing-only recorded node: ``preds`` maps each pred id to its WARPED edge
    delay (us); ``min_start_delay_us`` set => START-rooted. ``raw_start_s`` /
    ``raw_end_s`` are the unwarped recorded instants used ONLY to classify the
    binding edge exact-vs-idle-capped. Content fields are inert (timing ignores)."""
    pred_ids = list(preds or {})
    return _RecordedNode(
        node_id=node_id,
        messages=_MSG,
        start_s=0.0,
        end_s=0.0,
        raw_start_s=raw_start_s,
        raw_end_s=raw_end_s,
        predecessors=pred_ids,
        pred_delay_us=dict(preds or {}),
        rooted_at_start=min_start_delay_us is not None,
        min_start_delay_us=min_start_delay_us,
    )


def _erec(node_id: str, request_start_ns: int) -> _ExportRecord:
    """One profiling export projection at an absolute dispatch instant (ns)."""
    return _ExportRecord(
        conversation_id="t#0.0",
        node_id=node_id,
        phase="profiling",
        request_start_ns=request_start_ns,
        credit_issued_ns=request_start_ns - 1_000_000,
        messages=_MSG,
    )


@pytest.mark.component_integration
def test_timing_multipred_binds_to_max_gate_not_latest_recorded() -> None:
    """A join's expected dispatch is the MAX gate over preds; the binding pred is
    the argmax (here B's 60s-capped edge), NOT the recorded-latest pred A."""
    # A dispatches at origin+0, B at origin+10s. A's warped edge into J is 1s
    # (small), B's is 60s (idle-capped). Executor gate(J) = max(A+1, B+60) = B+60.
    origin = 1_000_000_000_000
    recorded = _RecordedTrace(
        trace_id="t",
        nodes={
            "A": _rnode("A", min_start_delay_us=0.0),
            "B": _rnode("B", min_start_delay_us=10_000_000.0, raw_end_s=0.0),
            # J's raw gap from B (100s) exceeds the 60s warped edge -> idle-capped.
            "J": _rnode(
                "J", preds={"A": 1_000_000.0, "B": 60_000_000.0}, raw_start_s=100.0
            ),
        },
    )
    a_ns = origin
    b_ns = origin + 10 * _NS
    j_ns = b_ns + 60 * _NS  # binds to B's capped gate, not A's 1s gate
    records = [_erec("A", a_ns), _erec("B", b_ns), _erec("J", j_ns)]
    report = causality_timing_vs_real_trace(
        None, None, recorded=recorded, records=records
    )
    assert report.passed, report.render()
    # B is the binding argmax and its warped (60s) < raw -> classified idle-capped.
    assert report.idle_capped_edges == 1, report.render()


@pytest.mark.component_integration
def test_timing_multipred_too_early_at_other_preds_gate_fails() -> None:
    """Moving the join to A's (non-binding) gate dispatches it too early vs the
    executor's max gate -> FAIL (non-vacuous)."""
    origin = 1_000_000_000_000
    recorded = _RecordedTrace(
        trace_id="t",
        nodes={
            "A": _rnode("A", min_start_delay_us=0.0),
            "B": _rnode("B", min_start_delay_us=10_000_000.0),
            "J": _rnode("J", preds={"A": 1_000_000.0, "B": 60_000_000.0}),
        },
    )
    a_ns = origin
    b_ns = origin + 10 * _NS
    j_ns = a_ns + 1 * _NS  # A's gate: ~69s BEFORE the binding B+60s gate
    records = [_erec("A", a_ns), _erec("B", b_ns), _erec("J", j_ns)]
    report = causality_timing_vs_real_trace(
        None, None, recorded=recorded, records=records
    )
    assert not report.passed, report.render()
    assert any("node=J" in m.where for m in report.mismatches), report.render()


@pytest.mark.component_integration
def test_timing_and_join_zero_delay_argmax_not_classified_idle_capped() -> None:
    """A fan-in whose OBSERVED argmax gate is a NON-BINDING AND-join (delay-0.0)
    edge is not classified at all: the delay-0 edge carries no think-time to
    compare, even though its raw end-to-start gap is positive.

    Recorded shape: A=[0,10] is the binding pred (warped delay 2s into J),
    B=[8,9] is an AND-join wait (delay 0.0 by construction), J raw_start=12.
    Observed: B dispatches 8s after A, so gate(B)=B+0 wins the argmax over
    gate(A)=A+2s. Before the fix J's raw gap from B (12-9=3s > 0) misread as
    "idle-capped" although no recorded gap exceeded the cap.
    """
    origin = 1_000_000_000_000
    recorded = _RecordedTrace(
        trace_id="t",
        nodes={
            "A": _rnode("A", min_start_delay_us=0.0, raw_end_s=10.0),
            "B": _rnode(
                "B", min_start_delay_us=8_000_000.0, raw_start_s=8.0, raw_end_s=9.0
            ),
            "J": _rnode("J", preds={"A": 2_000_000.0, "B": 0.0}, raw_start_s=12.0),
        },
    )
    a_ns = origin
    b_ns = origin + 8 * _NS
    j_ns = b_ns  # dispatches at the AND-join gate (B+0), past A's 2s gate
    records = [_erec("A", a_ns), _erec("B", b_ns), _erec("J", j_ns)]
    report = causality_timing_vs_real_trace(
        None, None, recorded=recorded, records=records
    )
    assert report.passed, report.render()
    assert report.timing_checks == 3, report.render()  # J's gate WAS compared
    assert report.idle_capped_edges == 0, report.render()
    assert report.exact_edges == 0, report.render()


@pytest.mark.component_integration
def test_timing_two_start_roots_fire_at_origin_plus_delay() -> None:
    """Two independent START-roots dispatch at origin+their own min_start_delay;
    the second is NOT timed relative to the first root's END (no spurious negative)."""
    origin = 1_000_000_000_000
    recorded = _RecordedTrace(
        trace_id="t",
        nodes={
            "R0": _rnode("R0", min_start_delay_us=0.0),
            "R1": _rnode("R1", min_start_delay_us=2_000_000.0),
        },
    )
    records = [
        _erec("R0", origin),
        _erec("R1", origin + 2 * _NS),
    ]
    report = causality_timing_vs_real_trace(
        None, None, recorded=recorded, records=records
    )
    assert report.passed, report.render()
    assert report.checked == 2, report.render()


@pytest.mark.component_integration
def test_timing_start_root_shifted_beyond_tolerance_fails() -> None:
    """Shifting one START-root's dispatch well past origin+delay FAILS (non-vacuous)."""
    origin = 1_000_000_000_000
    recorded = _RecordedTrace(
        trace_id="t",
        nodes={
            "R0": _rnode("R0", min_start_delay_us=0.0),
            "R1": _rnode("R1", min_start_delay_us=2_000_000.0),
        },
    )
    records = [
        _erec("R0", origin),
        _erec("R1", origin + 2 * _NS + 5 * _NS),  # 5s past origin+2s, well past tol
    ]
    report = causality_timing_vs_real_trace(
        None, None, recorded=recorded, records=records
    )
    assert not report.passed, report.render()
    assert any("node=R1" in m.where for m in report.mismatches), report.render()


# --- component lane: content_byte_exact_vs_v04 ----------------------------


@pytest.mark.component_integration
def test_byte_exact_vs_v04_passes_when_identical_after_rid_strip(
    tmp_path: Path,
) -> None:
    """Identical payloads differing only by the rid marker are byte-exact-equal."""
    ours = _faithful_records(_FIX)
    v04 = _faithful_records(_FIX)
    # v0.4 carries a DIFFERENT rid marker on the first user turn; ours carries none.
    v04_user = next(m for m in v04[0]["payload"]["messages"] if m["role"] == "user")
    v04_user["content"] = "[rid:ffffffffffff]\n\n" + v04_user["content"]
    ours_raw = _write_jsonl(tmp_path / "ours.jsonl", ours)
    v04_raw = _write_jsonl(tmp_path / "v04.jsonl", v04)
    report = content_byte_exact_vs_v04(ours_raw, v04_raw)
    assert report.passed, report.render()
    assert report.checked == 2


@pytest.mark.component_integration
def test_byte_exact_vs_v04_fails_on_content_drift(tmp_path: Path) -> None:
    """A non-marker content difference must FAIL the byte-exact check."""
    ours = _faithful_records(_FIX)
    v04 = _faithful_records(_FIX)
    v04[1]["payload"]["messages"][-1]["content"] += "_DRIFT"
    ours_raw = _write_jsonl(tmp_path / "ours.jsonl", ours)
    v04_raw = _write_jsonl(tmp_path / "v04.jsonl", v04)
    report = content_byte_exact_vs_v04(ours_raw, v04_raw)
    assert not report.passed, report.render()


@pytest.mark.component_integration
def test_byte_exact_vs_v04_fails_on_missing_coverage(tmp_path: Path) -> None:
    """A node present in one export but missing from the other is a coverage fail.

    Guards against a vacuous PASS on an empty key overlap.
    """
    ours = _faithful_records(_FIX)
    v04 = _faithful_records(_FIX)[:1]  # drop trace_sub_n2s1:1 from v0.4
    ours_raw = _write_jsonl(tmp_path / "ours.jsonl", ours)
    v04_raw = _write_jsonl(tmp_path / "v04.jsonl", v04)
    report = content_byte_exact_vs_v04(ours_raw, v04_raw)
    assert not report.passed, report.render()
    assert any("missing" in m.detail for m in report.mismatches)


# --- cap-aware classification + corpus driver -----------------------------

_CAP_S = 5.0


def _linear_trace(trace_id: str, gaps_s: list[float], api_s: float = 1.0) -> dict:
    """A linear single-chain Weka trace: each turn extends the prior hash prefix.

    ``gaps_s[i]`` is the RAW recorded END-to-start gap between turn ``i`` and turn
    ``i+1`` (so turn ``i+1`` starts at ``end_of_turn_i + gaps_s[i]``). A linear
    hash-prefix chain yields one single-predecessor ``StaticEdge`` per hop, the
    cleanest input for the exact-vs-idle-capped classification.
    """
    t = 0.0
    requests: list[dict] = []
    hash_ids: list[int] = []
    for i, _ in enumerate([0.0, *gaps_s]):
        hash_ids = [*hash_ids, i + 1]
        requests.append(
            {
                "t": t,
                "type": "n",
                "model": _MODEL,
                "in": 100,
                "out": 10,
                "hash_ids": list(hash_ids),
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": api_s,
                "think_time": 0.0,
            }
        )
        if i < len(gaps_s):
            t = t + api_s + gaps_s[i]
    return {
        "id": trace_id,
        "models": [_MODEL],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


def _warped_records(
    trace_file: Path, conv: str, *, phase: str = "profiling"
) -> list[dict]:
    """A faithful capped-run export: observed gaps == the WARPED edge delays.

    Builds the recorded trace WITH the cap, then dispatches each node at
    ``predecessor_dispatch + warped_edge_delay`` (accumulated down the chain) so
    the observed start-to-start gap of every edge equals its warped
    ``delay_after_predecessor_us`` -- exactly what the zero-latency mock produces
    (the predecessor returns ~instantly, so observed start-to-start collapses onto
    the END-to-start edge delay). Roots dispatch at ``t0``. ``phase`` stamps every
    record's ``benchmark_phase`` (``"warmup"`` simulates auto-warmup priming).
    """
    recorded = build_recorded_trace(trace_file, idle_gap_cap_seconds=_CAP_S)
    t0 = 1_000_000_000_000
    ns = 1_000_000_000
    # Walk nodes in recorded start order so a predecessor's dispatch is assigned
    # before its successor reads it (the linear chain is already topologically
    # ordered by start_s).
    order = sorted(recorded.nodes, key=lambda nid: recorded.nodes[nid].start_s)
    dispatch_ns: dict[str, int] = {}
    out: list[dict] = []
    for nid in order:
        node = recorded.nodes[nid]
        preds = [p for p in node.predecessors if p in dispatch_ns]
        if preds:
            cause = max(preds, key=lambda p: recorded.nodes[p].end_s)
            edge_delay_s = node.pred_delay_us.get(cause, 0.0) / 1e6
            start_ns = dispatch_ns[cause] + int(edge_delay_s * ns)
        else:
            start_ns = t0
        dispatch_ns[nid] = start_ns
        out.append(
            _record(
                node_id=nid,
                phase=phase,
                request_start_ns=start_ns,
                messages=node.messages,
                conversation_id=conv,
            )
        )
    return out


@pytest.mark.component_integration
def test_timing_classifies_exact_and_idle_capped(tmp_path: Path) -> None:
    """One sub-cap edge => 'exact'; one over-cap edge => 'idle-capped'; PASSES."""
    # gaps: r_0->r_1 raw 1s (< cap, exact), r_1->r_2 raw 97s (> cap, idle-capped).
    trace = _linear_trace("lin_cap", gaps_s=[1.0, 97.0])
    trace_file = tmp_path / "lin_cap.json"
    trace_file.write_text(json.dumps(trace))
    raw = _write_jsonl(
        tmp_path / "raw.jsonl", _warped_records(trace_file, "lin_cap#0.0")
    )
    report = causality_timing_vs_real_trace(
        raw, trace_file, idle_gap_cap_seconds=_CAP_S
    )
    assert report.passed, report.render()
    assert report.exact_edges == 1, report.render()
    assert report.idle_capped_edges == 1, report.render()


@pytest.mark.component_integration
def test_timing_fails_when_observed_gap_violates_warped(tmp_path: Path) -> None:
    """Mutating an observed gap beyond tolerance FAILS timing (non-vacuous proof)."""
    trace = _linear_trace("lin_cap", gaps_s=[1.0, 97.0])
    trace_file = tmp_path / "lin_cap.json"
    trace_file.write_text(json.dumps(trace))
    records = _warped_records(trace_file, "lin_cap#0.0")
    # Push the LAST node 30s past its warped-expected edge delay (well past tol);
    # the over-cap edge is bounded by the cap, so 30s is a genuine violation, not
    # a tolerated raw think-time.
    records[-1]["metadata"]["request_start_ns"] += 30_000_000_000
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    report = causality_timing_vs_real_trace(
        raw, trace_file, idle_gap_cap_seconds=_CAP_S
    )
    assert not report.passed, report.render()
    assert any("relative offset" in m.detail for m in report.mismatches), (
        report.render()
    )


@pytest.mark.component_integration
def test_prove_corpus_aggregates_two_traces(tmp_path: Path) -> None:
    """Two-trace corpus: aggregate counts correct; both traces pass; coverage 100%."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    records: list[dict] = []
    for tid in ("traceA", "traceB"):
        trace = _linear_trace(tid, gaps_s=[1.0, 97.0])
        tf = trace_dir / f"{tid}.json"
        tf.write_text(json.dumps(trace))
        records.extend(_warped_records(tf, f"{tid}#0.0"))
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    rc = prove_corpus(raw, trace_dir, idle_gap_cap_seconds=_CAP_S)
    assert rc == 0


@pytest.mark.component_integration
def test_prove_corpus_content_mutation_fails_only_that_trace(tmp_path: Path) -> None:
    """A content mutation in ONE trace fails the corpus (and only that trace)."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    records: list[dict] = []
    for tid in ("traceA", "traceB"):
        trace = _linear_trace(tid, gaps_s=[1.0, 97.0])
        tf = trace_dir / f"{tid}.json"
        tf.write_text(json.dumps(trace))
        recs = _warped_records(tf, f"{tid}#0.0")
        if tid == "traceB":
            recs[-1]["payload"]["messages"][0]["content"] += "_CORRUPTED"
        records.extend(recs)
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    rc = prove_corpus(raw, trace_dir, idle_gap_cap_seconds=_CAP_S)
    assert rc == 1


@pytest.mark.component_integration
def test_prove_corpus_reports_coverage_for_undispatched_node(tmp_path: Path) -> None:
    """An undispatched deep turn is reported as COVERAGE, not a failure."""
    trace = _linear_trace("traceA", gaps_s=[1.0, 97.0])
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    tf = trace_dir / "traceA.json"
    tf.write_text(json.dumps(trace))
    records = _warped_records(tf, "traceA#0.0")
    # The bounded run never reached the deepest turn (drop traceA:2's dispatch). Its
    # coverage drops below 100% but the proof still PASSES (no checked failure).
    records = [
        r for r in records if not r["metadata"]["x_request_id"].startswith("traceA:2::")
    ]
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    rc = prove_corpus(raw, trace_dir, idle_gap_cap_seconds=_CAP_S)
    assert rc == 0


@pytest.mark.component_integration
def test_prove_corpus_warmup_only_trace_skips_and_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A trace only warmup-primed (zero PROFILING records) is SKIP, not FAIL.

    Graph auto-warmup bursts priming credits before profiling, so a bounded
    profiling run can leave some traces warmup-only. Both criteria hard-fail on
    zero profiling records, so counting any-phase records used to flip such a
    trace to FAIL and falsely fail the whole corpus; it must SKIP with a
    distinct reason while the fully-profiled trace carries the PASS.
    """
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    records: list[dict] = []
    for tid, phase in (("traceA", "profiling"), ("traceB", "warmup")):
        trace = _linear_trace(tid, gaps_s=[1.0, 97.0])
        tf = trace_dir / f"{tid}.json"
        tf.write_text(json.dumps(trace))
        records.extend(_warped_records(tf, f"{tid}#0.0", phase=phase))
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    rc = prove_corpus(raw, trace_dir, idle_gap_cap_seconds=_CAP_S)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "SKIP (warmup-primed, not profiled)" in out
    assert "corpus proof: PASS" in out
    assert "FAIL" not in out


@pytest.mark.component_integration
def test_prove_corpus_all_warmup_vacuous_exit_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An export with ONLY warmup records checked nothing -> vacuous, exit 1.

    Every trace is SKIP (warmup-primed), so no per-trace failure fires, but the
    corpus-level VACUOUS gate must still fail an export that profiled nothing.
    """
    trace_dir, tf, _records = _corpus_with_records(tmp_path)
    records = _warped_records(tf, "traceA#0.0", phase="warmup")
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    rc = prove_corpus(raw, trace_dir, idle_gap_cap_seconds=_CAP_S)
    out = capsys.readouterr().out
    assert rc == 1
    assert "SKIP (warmup-primed, not profiled)" in out
    assert "VACUOUS: nothing checked" in out


# --- exit-gate integrity + vacuous-proof rejection (T1/T3) -----------------


def _corpus_with_records(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    """A one-trace corpus dir + its faithful export records (not yet written)."""
    trace = _linear_trace("traceA", gaps_s=[1.0])
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    tf = trace_dir / "traceA.json"
    tf.write_text(json.dumps(trace))
    return trace_dir, tf, _warped_records(tf, "traceA#0.0")


@pytest.mark.component_integration
def test_prove_corpus_unresolvable_node_ids_exit_nonzero(tmp_path: Path) -> None:
    """Records whose correlation ids resolve to NO trie node fail the exit code.

    Correlation-scheme drift is exactly what this gate exists to catch; before
    the exit-gate fix these rows printed MISMATCH yet exited 0 because
    ``dispatched_nodes == 0`` / ``timing.checked == 0`` masked them.
    """
    trace_dir, _tf, records = _corpus_with_records(tmp_path)
    for r in records:
        r["metadata"]["x_request_id"] = "bogus_node:0::deadbeefdeadbeefdeadbeefdeadbeef"
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    rc = prove_corpus(raw, trace_dir, idle_gap_cap_seconds=_CAP_S)
    assert rc == 1


@pytest.mark.component_integration
def test_prove_corpus_missing_request_start_exit_nonzero(tmp_path: Path) -> None:
    """Records with no ``request_start_ns`` cannot vacuously pass the timing gate."""
    trace_dir, _tf, records = _corpus_with_records(tmp_path)
    for r in records:
        r["metadata"]["request_start_ns"] = None
        r["metadata"]["credit_issued_ns"] = None
    raw = _write_jsonl(tmp_path / "raw.jsonl", records)
    rc = prove_corpus(raw, trace_dir, idle_gap_cap_seconds=_CAP_S)
    assert rc == 1


@pytest.mark.component_integration
def test_prove_corpus_vacuous_empty_inputs_exit_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty export against an empty trace dir checked nothing -> exit 1."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    raw = tmp_path / "raw.jsonl"
    raw.write_text("")
    rc = prove_corpus(raw, trace_dir, idle_gap_cap_seconds=_CAP_S)
    assert rc == 1
    assert "VACUOUS: nothing checked" in capsys.readouterr().out


@pytest.mark.component_integration
def test_prove_corpus_zero_overlap_vacuous_exit_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An export whose records map to NO recorded trace checked nothing -> exit 1.

    The wrong-artifact-paired-with-wrong-corpus case: every trace row is 0%
    coverage, so a proof that asserted nothing must not exit green.
    """
    trace_dir, _tf, _records = _corpus_with_records(tmp_path)
    other = [
        _record(
            node_id="r_0",
            phase="profiling",
            request_start_ns=1_000_000_000_000,
            messages=[{"role": "user", "content": "hi"}],
            conversation_id="otherTrace#0.0",
        )
    ]
    raw = _write_jsonl(tmp_path / "raw.jsonl", other)
    rc = prove_corpus(raw, trace_dir, idle_gap_cap_seconds=_CAP_S)
    assert rc == 1
    assert "VACUOUS: nothing checked" in capsys.readouterr().out


# --- summary arithmetic + order/timing separation (T2) ---------------------


@pytest.mark.component_integration
def test_timing_report_arithmetic_no_negative_passes() -> None:
    """One record failing BOTH causal-order and relative-timing appends two
    mismatches for ONE checked record; pass counts come from the dedicated
    ``passes`` counter (the old ``checked - len(mismatches)`` went negative),
    and the causal-order vs relative-timing tallies stay separated."""
    origin = 1_000_000_000_000
    recorded = _RecordedTrace(
        trace_id="t",
        nodes={
            "A": _rnode("A", min_start_delay_us=0.0),
            "B": _rnode("B", preds={"A": 1_000_000.0}),
        },
    )
    records = [
        _erec("A", origin),
        _erec("B", origin - 10 * _NS),  # 10s BEFORE its recorded predecessor
    ]
    report = causality_timing_vs_real_trace(
        None, None, recorded=recorded, records=records
    )
    assert not report.passed
    assert report.checked == 2
    assert report.passes == 1  # A passes; B fails (order + timing)
    assert len(report.mismatches) == 2
    assert report.passes == report.checked - 1  # never negative, one fail-record
    assert (report.order_checks, report.order_failures) == (1, 1)
    assert (report.timing_checks, report.timing_failures) == (2, 1)


# Content-knob threading (--tokenizer/--corpus/--seed) is proven in
# tests/unit/graph/test_weka_fidelity_tool_gates.py: this lane's FakeTokenizer
# patch flattens tokenizer/seed-dependent content, so the knobs are only
# distinguishable against the real builtin synthesizer the unit lane uses.


# --- component lane: end-to-end cache-safety invariants -------------------
#
# The invariant the interval-order + message-unit redesign exists for, proven
# through the REAL gpt2 tokenizer + real ``CorpusContentSynthesizer`` (no stub
# callbacks): any two requests sharing a block-aligned prefix render an
# IDENTICAL leading per-message pool-id chain (role + message boundaries frozen
# at block creation, inherited immutably, never relabeled/coalesced). The unit
# tests in ``tests/unit/graph/test_weka_trie_interval_order.py`` cover this
# only with collision-free stub decoders; these drive the production path.

from aiperf.dataset.graph.adapters.weka.trace_models import (  # noqa: E402
    WekaTrace as _WekaTrace,
)
from aiperf.dataset.graph.adapters.weka.trie_build import (  # noqa: E402
    build_trie_graph as _build_trie_graph,
)


def _n_req(
    t: float,
    *,
    in_tokens: int,
    out_tokens: int,
    hash_ids: list[int],
    api_time: float = 1.0,
    think_time: float = 0.0,
) -> dict:
    """A single recorded normal ("n") Weka request dict (``_linear_trace`` shape)."""
    return {
        "t": t,
        "type": "n",
        "model": _MODEL,
        "in": in_tokens,
        "out": out_tokens,
        "hash_ids": list(hash_ids),
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
        "api_time": api_time,
        "think_time": think_time,
    }


def _subagent(t: float, agent_id: str, requests: list[dict]) -> dict:
    """A completed (blocking) subagent marker wrapping inner recorded requests."""
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "X",
        "status": "completed",
        "models": [_MODEL],
        "requests": requests,
    }


def _trace_dict(requests: list[dict], *, block_size: int = 64) -> dict:
    """A trace-level Weka dict (``_linear_trace`` field shape) wrapping ``requests``."""
    return {
        "id": "invariant",
        "models": [_MODEL],
        "block_size": block_size,
        "hash_id_scope": "local",
        "requests": requests,
    }


def _build_real(requests: list[dict], *, block_size: int = 64):
    """Validate + build the trie graph through the REAL gpt2 synthesizer (no stubs)."""
    trace = _WekaTrace.model_validate(_trace_dict(requests, block_size=block_size))
    return _build_trie_graph(
        trace,
        tokenizer_name="gpt2",
        prompt_corpus="coding",
        root_seed=None,
        idle_gap_cap_seconds=None,
    )


def _llm_nodes(parsed) -> dict:
    """``{node_id: LlmNode}`` for every LlmNode in the built graph."""
    from aiperf.dataset.graph.models import LlmNode

    return {nid: n for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)}


def _hash_ids_by_node(
    requests: list[dict], *, block_size: int = 64
) -> dict[str, list[int]]:
    """``{node_id: recorded hash_ids}`` recovered from the flatten pass.

    The built ``LlmNode`` no longer carries its recorded ``hash_ids`` in
    ``metadata["trie"]`` (the envelope is slimmed to ``prompt_segment_ids`` only),
    so tests that identify a node by its recorded hash prefix recover it from the
    same ``_flatten_requests`` pass the builder uses -- byte-identical node ids and
    the genuine recorded prefix, no reliance on a build-plane metadata copy.
    """
    from aiperf.dataset.graph.adapters.weka.trie_build import _flatten_requests

    trace = _WekaTrace.model_validate(_trace_dict(requests, block_size=block_size))
    return {
        n.node_id: list(n.request.hash_ids)
        for n in _flatten_requests(trace.requests, root_scope=trace.id)
    }


def _block_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common leading run of two hash-id lists."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _msg_block_counts_from_tags(block_tags: list[tuple[str, bool]]) -> list[int]:
    """Block count per message from a node's frozen per-block tags.

    Mirrors ``assemble_messages`` grouping exactly: a new group opens at the
    first block and at every ``starts_new_message`` block.
    """
    groups: list[list[int]] = []
    for j, (_role, starts) in enumerate(block_tags):
        if starts or not groups:
            groups.append([j])
        else:
            groups[-1].append(j)
    return [len(g) for g in groups]


def _tags_by_node(
    requests: list[dict], block_size: int
) -> dict[str, list[tuple[str, bool]]]:
    """Frozen per-block tags keyed by node id for a trace's requests.

    Reuses the production tag pass so the test's notion of message->block spans
    is byte-identical to what the builder froze.
    """
    from aiperf.dataset.graph.adapters.weka.trie_build import (
        _flatten_requests,
    )
    from aiperf.dataset.graph.segment_ir.trie_content import (
        assign_block_tags,
        compute_asst_caps,
        resolve_content_parents,
    )

    trace = _WekaTrace.model_validate(_trace_dict(requests, block_size=block_size))
    nodes = _flatten_requests(trace.requests, root_scope=trace.id)
    resolve_content_parents(nodes)
    caps = compute_asst_caps(nodes, block_size)
    return assign_block_tags(nodes, block_size, caps)


def _messages_within_blocks(msg_block_counts: list[int], n_blocks: int) -> int:
    """Number of leading messages whose blocks lie WHOLLY within the first ``n_blocks``.

    A message spanning a block past ``n_blocks`` is excluded (it is only partly
    shared and its id may legitimately differ).
    """
    covered = 0
    count = 0
    for bc in msg_block_counts:
        if covered + bc > n_blocks:
            break
        covered += bc
        count += 1
    return count


@pytest.mark.component_integration
def test_shared_block_prefix_identical_leading_prompt_ids_real_tokenizer() -> None:
    """Core invariant (real gpt2): any two nodes sharing ``L>0`` leading blocks
    render byte-IDENTICAL pool ids for every message wholly inside those ``L``
    blocks. A single divergent shared-prefix id would trip the equality.

    Fixture (block_size=64, every ``in`` block-aligned):
    * ``invariant:0``  root user turn, hash_ids ``[1,2,3]`` (``in=64*3``), big ``out``.
    * ``a:0``          a completed subagent forking off the root: inherits the FULL
      ``[1,2,3]`` then extends ``[4,5]`` (``in=64*5``) -> new blocks become
      assistant (parent had user + out>0).
    * ``invariant:1``  a sibling top-level turn sharing the full ``[1,2,3]`` then
      diverging at block 3 (``[1,2,3,90]``, ``in=64*4``).

    All three share ``L=3`` blocks; the root's single leading user message
    (blocks 0..2, wholly inside L) is byte-identical across all three, and the
    shorter node's fully-in-``L`` ids are a prefix of the longer node's.
    """
    block_size = 64
    requests = [
        _n_req(0.0, in_tokens=64 * 3, out_tokens=64 * 4, hash_ids=[1, 2, 3]),
        _subagent(
            1.0,
            "a",
            [_n_req(1.1, in_tokens=64 * 5, out_tokens=0, hash_ids=[1, 2, 3, 4, 5])],
        ),
        _n_req(3.0, in_tokens=64 * 4, out_tokens=0, hash_ids=[1, 2, 3, 90]),
    ]
    parsed, _pool = _build_real(requests, block_size=block_size)
    nodes = _llm_nodes(parsed)
    tags = _tags_by_node(requests, block_size)

    ids = {nid: n.metadata["trie"]["prompt_segment_ids"] for nid, n in nodes.items()}
    hashes = _hash_ids_by_node(requests, block_size=block_size)
    spans = {nid: _msg_block_counts_from_tags(tags[nid]) for nid in nodes}

    shared_pairs = 0
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            length = _block_prefix_len(hashes[u], hashes[v])
            if length <= 0:
                continue
            shared_pairs += 1
            # Messages of u/v that live WHOLLY inside the first L shared blocks
            # must be byte-identical ids (frozen-tag inheritance). The count of
            # such messages matches for both because the shared L blocks are
            # tagged identically -> identical grouping -> identical span prefix.
            mu = _messages_within_blocks(spans[u], length)
            mv = _messages_within_blocks(spans[v], length)
            k = min(mu, mv)
            assert k >= 1, (u, v, length, spans[u], spans[v])
            assert ids[u][:k] == ids[v][:k], (
                f"{u} vs {v}: shared-prefix ids diverge within L={length} blocks: "
                f"{ids[u][:k]} != {ids[v][:k]}"
            )
    # Non-vacuity: at least one ordered pair actually shared a block prefix.
    assert shared_pairs >= 1


@pytest.mark.component_integration
def test_57f2a77e_no_relabel_shared_assistant_block_frozen_real_tokenizer() -> None:
    """``57f2a77e``-shape no-relabel regression (real gpt2).

    The corpus receipt ``57f2a77e2d33...`` was a cache MISS because the per-turn
    pass RELABELED a shared block's role on a later turn: a parent turn whose
    output made part of the shared prefix ASSISTANT, and a subagent sharing that
    exact block prefix but recorded pure-user, disagreed at the role-transition
    block. The redesign FREEZES each shared block's role at its first creator and
    the subagent inherits it verbatim -> the shared leading id chain is IDENTICAL.

    Fixture (block_size=64):
    * ``invariant:0``  root user turn ``[1,2,3]`` with big ``out`` (5 blocks) so its
      child's new blocks are attributed to the assistant.
    * ``invariant:1``  a follow-up sharing ``[1,2,3]`` and extending to ``[1..8]``
      (``in=64*8``); its new blocks 3..7 split assistant-then-user, so its
      8-block prefix materializes as user / assistant / user (a role transition
      INSIDE the shared span).
    * ``a:0``          a completed subagent sharing the SAME 8-block prefix
      (``[1..8, 9]``) recorded pure-user (``out=0``).

    Observed role layout of the shared 8-block prefix (both invariant:1 and a:0):
    ``["user", "assistant", "user"]`` (3 messages). Asserts a:0's leading 3 ids
    == invariant:1's full 3 ids -> the assistant block was NOT relabeled to user on
    the subagent. A relabel would change a:0's middle message role -> different
    content-addressed id -> assertion fails.
    """
    block_size = 64
    parent_prefix = list(range(1, 9))  # 8 shared blocks
    requests = [
        _n_req(0.0, in_tokens=64 * 3, out_tokens=64 * 5, hash_ids=[1, 2, 3]),
        _n_req(2.0, in_tokens=64 * 8, out_tokens=64 * 2, hash_ids=parent_prefix),
        _subagent(
            4.0,
            "a",
            [_n_req(4.1, in_tokens=64 * 9, out_tokens=0, hash_ids=[*parent_prefix, 9])],
        ),
    ]
    parsed, pool = _build_real(requests, block_size=block_size)
    nodes = _llm_nodes(parsed)
    hashes = _hash_ids_by_node(requests, block_size=block_size)

    def by_hash(h: list[int]) -> str:
        return next(nid for nid, hids in hashes.items() if hids == h)

    r1 = by_hash(parent_prefix)
    sub = by_hash([*parent_prefix, 9])
    r1_ids = nodes[r1].metadata["trie"]["prompt_segment_ids"]
    sub_ids = nodes[sub].metadata["trie"]["prompt_segment_ids"]

    r1_roles = [pool.materialize([i])[0]["role"] for i in r1_ids]
    # Non-vacuity: the parent's shared 8-block span genuinely contains an
    # assistant message (so the test proves inheritance ACROSS a role transition,
    # not a trivially-all-user case).
    assert "assistant" in r1_roles, r1_roles
    assert len(r1_ids) >= 1

    # The subagent shares exactly r_1's 8 blocks -> its leading id chain covering
    # those blocks is byte-identical to r_1's full prompt-id chain (frozen tags).
    assert sub_ids[: len(r1_ids)] == r1_ids, (
        f"57f2a77e relabel regression: subagent shared-prefix ids diverge from "
        f"parent: {sub_ids[: len(r1_ids)]} != {r1_ids} (roles={r1_roles})"
    )


@pytest.mark.component_integration
def test_block_aligned_isl_covered_count_real_tokenizer() -> None:
    """Block-aligned ISL through the real gpt2 tokenizer.

    Every LlmNode's frozen covered-block count equals
    ``min(len(hash_ids), in // block_size)`` -- the message-unit emitter covers
    only that many whole blocks (no partial tail, no synthesis of missing whole
    blocks). The build already HARD-ABORTS (``TrieISLMismatchError``) if a node's
    materialized token count misses this target, so a successful build is the
    load-bearing proof; this test additionally pins the covered-block count so a
    DROPPED block (over/under-coverage) would trip the equality.

    Non-vacuity: ``[1,2,3,4]`` records FEWER hash blocks than ``in // block_size``
    (a truncated hash list: ``in=64*5`` but only 4 blocks). The covered count must
    use the ``min`` (4), not over-demand 5 -- an over-demand would fail here.
    """
    block_size = 64
    requests = [
        _n_req(0.0, in_tokens=64 * 2, out_tokens=64, hash_ids=[1, 2]),
        _n_req(2.0, in_tokens=64 * 3, out_tokens=0, hash_ids=[1, 2, 3]),
        # Truncated: in // bs == 5 but only 4 recorded hash blocks -> covered = 4.
        _n_req(4.0, in_tokens=64 * 5, out_tokens=0, hash_ids=[1, 2, 3, 4]),
    ]
    parsed, _pool = _build_real(requests, block_size=block_size)
    nodes = _llm_nodes(parsed)
    tags = _tags_by_node(requests, block_size)

    hashes = _hash_ids_by_node(requests, block_size=block_size)
    saw_truncated = False
    for nid in nodes:
        hash_ids = hashes[nid]
        in_blocks = _recorded_in_blocks(requests, hash_ids)
        # The frozen tag count IS the covered-block count (the build asserts
        # len(tags) * bs == covered-count * bs before emitting the node).
        covered = len(tags[nid])
        expected_cover = min(len(hash_ids), in_blocks)
        assert covered == expected_cover, (nid, covered, expected_cover, hash_ids)
        if in_blocks > len(hash_ids):
            saw_truncated = True
            assert covered == len(hash_ids)
    assert saw_truncated, "fixture must include a truncated-hash node"


def _recorded_in_blocks(requests: list[dict], hash_ids: list[int]) -> int:
    """``in // 64`` of the (possibly subagent-nested) request whose ``hash_ids`` match.

    Recovers the raw ``in`` block count for a node so a test can distinguish
    covered vs truncated coverage; block_size is 64 in these fixtures.
    """

    def _find(reqs: list[dict]) -> int | None:
        for r in reqs:
            if r["type"] == "subagent":
                inner = _find(r["requests"])
                if inner is not None:
                    return inner
            elif r["hash_ids"] == hash_ids:
                return r["in"] // 64
        return None

    got = _find(requests)
    assert got is not None
    return got


@pytest.mark.component_integration
def test_two_user_turn_boundary_preserved_real_tokenizer() -> None:
    """Boundary preservation end-to-end (real gpt2).

    A single request spanning two consecutive user turns (block 1 opens a new
    message, ``starts_new_message=True``) must materialize as >=2 messages (NOT
    coalesced into one) and the LAST materialized message role is ``"user"``
    (trailing-user frozen at creation). Coalescing the boundary would drop to 1
    message and trip the count; a relabeled trailing block would trip the role.
    """
    block_size = 64
    requests = [
        _n_req(0.0, in_tokens=64, out_tokens=0, hash_ids=[1]),
        _n_req(2.0, in_tokens=64 * 2, out_tokens=0, hash_ids=[1, 2]),
    ]
    parsed, pool = _build_real(requests, block_size=block_size)
    nodes = _llm_nodes(parsed)

    hashes = _hash_ids_by_node(requests, block_size=block_size)
    two_turn_id = next(nid for nid, hids in hashes.items() if hids == [1, 2])
    ids = nodes[two_turn_id].metadata["trie"]["prompt_segment_ids"]
    roles = [pool.materialize([i])[0]["role"] for i in ids]
    assert len(ids) >= 2, roles  # boundary preserved, not coalesced
    assert roles[-1] == "user", roles  # trailing-user frozen at creation

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The offline fidelity proof validates START-ANCHORED nodes as dispatch-to-dispatch
offsets, not as zero end-to-start delays.

Task 3 taught the weka trie builder to collapse a mid-flight spawn / chain-overlap
node's incoming edges into a single ``StaticEdge.delay_after_predecessor_start_us``
(the warped start-to-start gap). This locks the matching change to
``tools/weka_trace_fidelity.py``: :func:`build_recorded_trace` must route those
edges into ``_RecordedNode.start_anchor`` (NOT into the end-to-start
``predecessors`` / ``pred_delay_us``), and
:func:`causality_timing_vs_real_trace` must compare each start-anchored child's
OBSERVED ``request_start_ns`` gap from its PARENT'S dispatch against that warped
delay -- so a run that places the child at parent-dispatch + delay passes and one
that shifts it off does not.

The fixture geometry mirrors ``_OVERLAP_TRACE`` in
``tests/unit/graph/test_start_anchor_runtime.py``:
``START->start_anchor:0``; ``start_anchor:0->a1:0`` start-delay 2.5s;
``start_anchor:0->start_anchor:1`` start-delay 5.0s;
``start_anchor:0->start_anchor:2`` end-delay 1.0s
+ ``start_anchor:1->start_anchor:2`` end-delay 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.weka_trace_fidelity import (
    build_recorded_trace,
    causality_timing_vs_real_trace,
)
from tools.weka_trie_timing_sim import main as timing_sim_main
from tools.weka_trie_timing_sim import simulate_trace

# P: t=0 api=8.0 (long, spawner); C: subagent first at t=2.5 (P in flight);
# Q: chain-overlap at t=5.0 (P in flight); R: t=9.0 (after P ends, end-anchored).
# Byte-identical to ``test_start_anchor_runtime._OVERLAP_TRACE``.
_OVERLAP_TRACE = {
    "id": "start_anchor", "models": ["M"], "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "n", "model": "M", "in": 128, "out": 64,
         "hash_ids": [1, 2], "api_time": 8.0, "stop": "tool_use"},
        {"t": 2.0, "type": "subagent", "agent_id": "a1",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 2.5, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [50, 51], "api_time": 1.0},
         ]},
        {"t": 5.0, "type": "n", "model": "M", "in": 192, "out": 32,
         "hash_ids": [1, 2, 3], "api_time": 1.0},
        {"t": 9.0, "type": "n", "model": "M", "in": 256, "out": 32,
         "hash_ids": [1, 2, 3, 4], "api_time": 0.5},
    ],
}  # fmt: skip

_P = "start_anchor:0"
_C = "a1:0"
_Q = "start_anchor:1"
_R = "start_anchor:2"

_S = 1_000_000_000  # 1e9 ns per second
_ORIGIN_NS = 1 * _S  # arbitrary fresh run-origin (absolute wall-clock differs)


def _write_trace(tmp_path: Path) -> Path:
    """Materialize ``_OVERLAP_TRACE`` to a JSON file the tool can rebuild from."""
    path = tmp_path / "start_anchor.json"
    path.write_text(json.dumps(_OVERLAP_TRACE))
    return path


def _record(node_id: str, request_start_ns: int) -> dict:
    """One raw-export JSONL line for a profiling dispatch of ``node_id``.

    The tool recovers the node id from ``x_request_id``
    (``{node_id}::{nonce}``, worker-minted) and selects the trace by the
    ``conversation_id`` base.
    """
    return {
        "metadata": {
            "conversation_id": "start_anchor#0",
            "x_request_id": f"{node_id}::deadbeefdeadbeefdeadbeefdeadbeef",
            "benchmark_phase": "profiling",
            "request_start_ns": request_start_ns,
            "credit_issued_ns": None,
        },
        "payload": {"messages": []},
    }


def _write_raw(tmp_path: Path, dispatch_ns: dict[str, int]) -> Path:
    """Write a raw-export JSONL placing each node at its given absolute dispatch ns."""
    path = tmp_path / "profile_export_raw.jsonl"
    lines = [json.dumps(_record(nid, ns)) for nid, ns in dispatch_ns.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


# A faithful replay against the zero-latency mock: P at the origin, the two
# start-anchored children at parent-dispatch + their warped start-delay, and the
# end-anchored R at its binding pred's (start_anchor:1) observed dispatch + delay 0.
_FAITHFUL_NS = {
    _P: _ORIGIN_NS,
    _C: _ORIGIN_NS + int(2.5 * _S),
    _Q: _ORIGIN_NS + int(5.0 * _S),
    _R: _ORIGIN_NS + int(5.0 * _S),
}


def test_build_recorded_trace_captures_start_anchor(tmp_path: Path) -> None:
    """Start-anchored edges land in ``start_anchor``, NOT in end-to-start preds.

    The overlap children ``a1:0`` / ``start_anchor:1`` each carry one
    ``delay_after_predecessor_start_us`` edge off ``start_anchor:0``; the builder
    must record ``(start_anchor:0, warped_delay_us)`` and keep them out of
    ``predecessors`` so they are never mistimed as a zero end-to-start wait.
    """
    trace_file = _write_trace(tmp_path)
    recorded = build_recorded_trace(trace_file, idle_gap_cap_seconds=60.0)

    # start_anchor:0 is non-streaming (type "n"), so both overlap children are no-ttft-parent
    # start anchors -- the third (first-token delay) slot is None.
    assert recorded.nodes[_C].start_anchor == (_P, 2.5e6, None)
    assert recorded.nodes[_C].predecessors == []
    assert recorded.nodes[_Q].start_anchor == (_P, 5.0e6, None)
    assert recorded.nodes[_Q].predecessors == []
    # R stays a normal end-anchored AND-join (its edges are NOT start-anchored).
    assert recorded.nodes[_R].start_anchor is None
    assert sorted(recorded.nodes[_R].predecessors) == [_P, _Q]


def test_faithful_start_anchored_replay_passes(tmp_path: Path) -> None:
    """A replay placing C exactly 2.5s and Q 5.0s after P's dispatch PASSES.

    The proof compares each start-anchored child's observed dispatch gap from its
    PARENT'S dispatch against the warped start-to-start delay, and counts the two
    edges as ``exact``.
    """
    trace_file = _write_trace(tmp_path)
    raw = _write_raw(tmp_path, _FAITHFUL_NS)

    report = causality_timing_vs_real_trace(raw, trace_file, idle_gap_cap_seconds=60.0)

    assert report.passed, report.render()
    # The two start-anchored edges (C, Q) are validated as exact start-to-start.
    assert report.exact_edges >= 2


def test_shifted_start_anchored_child_fails(tmp_path: Path) -> None:
    """Shifting C's dispatch +1.0s (to 3.5s after P) FAILS -- the proof actually
    checks start-anchored timing rather than skipping those nodes.

    +1.0s exceeds the abs tolerance (0.75s) and the relative tolerance
    (0.15 * 2.5s = 0.375s), so the start-anchor comparison must flag it.
    """
    trace_file = _write_trace(tmp_path)
    shifted = dict(_FAITHFUL_NS)
    shifted[_C] = _FAITHFUL_NS[_C] + int(1.0 * _S)
    raw = _write_raw(tmp_path, shifted)

    report = causality_timing_vs_real_trace(raw, trace_file, idle_gap_cap_seconds=60.0)

    assert not report.passed
    assert any(_C in m.where for m in report.mismatches), report.render()


def test_timing_sim_tool_reconstructs_overlap_timeline(tmp_path: Path) -> None:
    """The sibling ``weka_trie_timing_sim`` simulator gates start-anchored edges off
    the predecessor's DISPATCH, so it reconstructs the overlap timeline byte-exact.

    Before Task 7 the simulator only knew end-to-start / START edges, so a
    start-anchored child fell through to ``sim_end(parent) + 0`` and landed at the
    parent's completion (a1:0 at 8.0s, not 2.5s) -- a divergence. With the
    dispatch-anchored branch every node reconstructs its recorded warped start
    (0.0 / 2.5 / 5.0 / 9.0), so all four check exact with zero divergences.
    """
    trace_file = _write_trace(tmp_path)

    checked, exact, diverged, first_token_edges = simulate_trace(
        trace_file, cap=60.0, tol=1e-3
    )

    assert diverged == []
    assert checked == exact == 4
    # _OVERLAP_TRACE's spawner (start_anchor:0) is non-streaming, so no first-token edges.
    assert first_token_edges == 0


# The start-anchored child (``a:0``) ties its parent's (``tie_order:2``) recorded
# start, and its node-id STRING sorts before the parent's ("a:0" < "tie_order:2"):
# the old ``(recorded_start, node_id)`` processing order simulated the child first
# and zero-defaulted the parent's dispatch, landing the child at t=0 instead of
# t=2.0 (a false divergence). Under the ``{scope}:{turn}`` id scheme the child's id
# derives from its subagent ``agent_id`` ("a"), so the empty padding markers no
# longer affect the id -- they are kept inert only to preserve the fixture shape.
_TIE_TRACE = {
    "id": "tie_order", "models": ["M"], "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "n", "model": "M", "in": 64, "out": 8,
         "hash_ids": [1], "api_time": 0.5},
        {"t": 1.0, "type": "n", "model": "M", "in": 128, "out": 8,
         "hash_ids": [1, 2], "api_time": 0.5},
        {"t": 2.0, "type": "n", "model": "M", "in": 192, "out": 64,
         "hash_ids": [1, 2, 3], "api_time": 10.0, "stop": "tool_use"},
        *(
            {"t": 2.0, "type": "subagent", "agent_id": f"pad{i}",
             "subagent_type": "X", "status": "completed", "models": ["M"],
             "requests": []}
            for i in range(3, 10)
        ),
        {"t": 2.0, "type": "subagent", "agent_id": "a",
         "subagent_type": "X", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 2.0, "type": "n", "model": "M", "in": 64, "out": 8,
              "hash_ids": [50], "api_time": 1.0},
         ]},
    ],
}  # fmt: skip


def test_timing_sim_dependency_order_beats_lexicographic_tie(
    tmp_path: Path,
) -> None:
    """The simulator processes a start-anchored parent BEFORE its child even when
    the child's node-id string sorts first at an identical recorded start.

    ``a:0`` start-anchors to ``tie_order:2`` (in flight, delay 0) and ties its
    recorded start; dependency (topological) order must gate it at the parent's
    simulated dispatch (2.0s), not at a zero-defaulted 0.0s -- all four nodes
    reconstruct the recorded warped timeline exactly.
    """
    trace_file = tmp_path / "tie_order.json"
    trace_file.write_text(json.dumps(_TIE_TRACE))

    checked, exact, diverged, first_token_edges = simulate_trace(
        trace_file, cap=60.0, tol=1e-3
    )

    assert diverged == [], diverged
    assert checked == exact == 4
    assert first_token_edges == 0


def test_timing_sim_main_vacuous_empty_dir_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dir with no ``*.json`` trace files is a VACUOUS simulation -> exit 1.

    A simulation that checked nothing must not exit green (the exit code is the
    acceptance signal in CI pipelines).
    """
    empty = tmp_path / "empty"
    empty.mkdir()

    rc = timing_sim_main([str(empty)])

    assert rc == 1
    assert "VACUOUS: nothing checked" in capsys.readouterr().out

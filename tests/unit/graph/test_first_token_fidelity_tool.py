# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The offline fidelity proof validates POST-TTFT first-token-anchored nodes as
``first_token + D'`` offsets, falling back to ``dispatch + D`` (with a LOUD
report line) when the parent's observed TTFT is not recoverable from the export.

The weka trie builder stamps a post-TTFT overlap node's single
``StaticEdge`` with BOTH ``delay_after_predecessor_start_us`` (D) and
``delay_after_predecessor_first_token_us`` (D' = D - ttft*1e6). This locks the
matching behavior in ``tools/weka_trace_fidelity.py``:

* :func:`build_recorded_trace` must widen ``_RecordedNode.start_anchor`` to the
  triple ``(source, start_delay_us, first_token_delay_us | None)`` -- pre-TTFT
  and no-ttft-parent overlaps carry ``None`` in the third slot.
* :func:`causality_timing_vs_real_trace` must, for a post-TTFT node, (a) compare
  the child's observed dispatch to ``parent_first_token + D'`` when the parent's
  observed first token is recoverable from the export (``responses[0].perf_ns -
  start_perf_ns`` added to the parent's dispatch wall clock); (b) else fall back
  to ``parent_dispatch + D`` AND emit a LOUD ``FALLBACK`` note -- never silently
  skipping the edge.

Geometry is byte-identical to ``_TTFT_TRACE`` in
``tests/unit/graph/test_first_token_runtime.py``: streaming P (ttft_anchor:0) ttft
2.0 api 8.0; PRE-TTFT child (a1:0) at t=1.0 (pure dispatch anchor); POST-TTFT child
(a2:0) at t=4.0 (D=4.0, D'=2.0); end-anchored tail (ttft_anchor:1) at t=9.0.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.weka_trace_fidelity import (
    build_recorded_trace,
    causality_timing_vs_real_trace,
)
from tools.weka_trie_timing_sim import simulate_trace

# Byte-identical to ``test_first_token_runtime._TTFT_TRACE``.
_TTFT_TRACE = {
    "id": "ttft_anchor", "models": ["M"], "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "s", "ttft": 2.0, "api_time": 8.0, "in": 128, "out": 64,
         "hash_ids": [1, 2], "stop": "tool_use", "model": "M"},
        {"t": 0.5, "type": "subagent", "agent_id": "a1",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 1.0, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [50, 51], "api_time": 1.0},
         ]},
        {"t": 3.5, "type": "subagent", "agent_id": "a2",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 4.0, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [60, 61], "api_time": 1.0},
         ]},
        {"t": 9.0, "type": "n", "model": "M", "in": 256, "out": 32,
         "hash_ids": [1, 2, 3, 4], "api_time": 0.5},
    ],
}  # fmt: skip

_P = "ttft_anchor:0"
_C_PRE = "a1:0"
_C_POST = "a2:0"
_TAIL = "ttft_anchor:1"

_S = 1_000_000_000  # 1e9 ns per second
_ORIGIN_NS = 1 * _S  # arbitrary fresh run-origin (absolute wall-clock differs)


def _write_trace(tmp_path: Path) -> Path:
    """Materialize ``_TTFT_TRACE`` to a JSON file the tool can rebuild from."""
    path = tmp_path / "ttft_anchor.json"
    path.write_text(json.dumps(_TTFT_TRACE))
    return path


def _record(
    node_id: str,
    request_start_ns: int,
    *,
    start_perf_ns: int | None = None,
    responses: list[dict] | None = None,
) -> dict:
    """One raw-export JSONL line for a profiling dispatch of ``node_id``.

    ``start_perf_ns`` + ``responses`` (each with a ``perf_ns``) let the tool
    recover a streaming parent's OBSERVED first token; omit both for a
    zero-latency-style export where the observed TTFT is unrecoverable.
    """
    return {
        "metadata": {
            "conversation_id": "ttft_anchor#0",
            "x_request_id": f"{node_id}::deadbeefdeadbeefdeadbeefdeadbeef",
            "benchmark_phase": "profiling",
            "request_start_ns": request_start_ns,
            "credit_issued_ns": None,
        },
        "start_perf_ns": start_perf_ns,
        "responses": responses or [],
        "payload": {"messages": []},
    }


def _write_raw(tmp_path: Path, records: list[dict]) -> Path:
    """Write a raw-export JSONL from pre-built record dicts."""
    path = tmp_path / "profile_export_raw.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_build_recorded_trace_captures_first_token_delay(tmp_path: Path) -> None:
    """Post-TTFT edges carry D' as the third ``start_anchor`` slot; pre-TTFT None.

    The overlap children each carry one ``delay_after_predecessor_start_us`` edge
    off ``ttft_anchor:0`` (streaming, ttft 2.0). ``a1:0`` started PRE-TTFT (t=1.0 <
    2.0) so its edge has no first-token delay; ``a2:0`` started POST-TTFT (t=4.0 >= 2.0)
    so its edge carries D' = D - ttft*1e6 = 4.0e6 - 2.0e6 = 2.0e6.
    """
    trace_file = _write_trace(tmp_path)
    recorded = build_recorded_trace(trace_file, idle_gap_cap_seconds=60.0)

    assert recorded.nodes[_C_PRE].start_anchor == (_P, 1.0e6, None)
    assert recorded.nodes[_C_POST].start_anchor == (_P, 4.0e6, 2.0e6)
    assert recorded.nodes[_C_PRE].predecessors == []
    assert recorded.nodes[_C_POST].predecessors == []


def test_fallback_replay_passes_with_loud_note(tmp_path: Path) -> None:
    """A zero-latency export (no recoverable parent TTFT) times the post-TTFT child
    at ``parent_dispatch + D`` and emits a LOUD FALLBACK note; the faithful placement
    at ``parent_start + 4.0s`` PASSES.
    """
    trace_file = _write_trace(tmp_path)
    raw = _write_raw(
        tmp_path,
        [
            _record(_P, _ORIGIN_NS),
            _record(_C_PRE, _ORIGIN_NS + int(1.0 * _S)),
            _record(_C_POST, _ORIGIN_NS + int(4.0 * _S)),
        ],
    )

    report = causality_timing_vs_real_trace(raw, trace_file, idle_gap_cap_seconds=60.0)

    assert report.passed, report.render()
    # Both start-anchored children validate exactly (pre-TTFT + post-TTFT fallback).
    assert report.exact_edges >= 2
    # The post-TTFT edge fell back to dispatch + D and said so, LOUDLY.
    assert any("FALLBACK" in n and _C_POST in n for n in report.notes), report.render()


def test_shifted_first_token_child_fails(tmp_path: Path) -> None:
    """Shifting the post-TTFT child's dispatch +1.0s (to 5.0s after P) FAILS naming
    it -- the proof actually checks first-token timing rather than skipping the node.

    +1.0s exceeds the abs tolerance (0.75s) and the relative tolerance
    (0.15 * 4.0s = 0.6s), so the fallback comparison must flag it.
    """
    trace_file = _write_trace(tmp_path)
    raw = _write_raw(
        tmp_path,
        [
            _record(_P, _ORIGIN_NS),
            _record(_C_PRE, _ORIGIN_NS + int(1.0 * _S)),
            _record(_C_POST, _ORIGIN_NS + int(5.0 * _S)),
        ],
    )

    report = causality_timing_vs_real_trace(raw, trace_file, idle_gap_cap_seconds=60.0)

    assert not report.passed
    assert any(_C_POST in m.where for m in report.mismatches), report.render()


def test_recoverable_first_token_uses_first_token_anchor(tmp_path: Path) -> None:
    """When the parent's observed first token IS recoverable, the proof anchors the
    post-TTFT child on ``first_token + D'`` -- NOT the dispatch fallback.

    P streams its first token at an OBSERVED 3.0s (differs from the recorded 2.0),
    so first_token_wall = P_dispatch + 3.0 and the expected child dispatch is
    3.0 + D'(2.0) = 5.0s after P. Placing the child there PASSES; placing it at the
    dispatch-fallback position (P_dispatch + D = 4.0s) FAILS, proving the observed
    first token -- not the recorded gap -- drives the comparison.
    """
    trace_file = _write_trace(tmp_path)
    p_rec = _record(
        _P, _ORIGIN_NS, start_perf_ns=0, responses=[{"perf_ns": int(3.0 * _S)}]
    )

    faithful = _write_raw(
        tmp_path,
        [p_rec, _record(_C_POST, _ORIGIN_NS + int(5.0 * _S))],
    )
    report = causality_timing_vs_real_trace(
        faithful, trace_file, idle_gap_cap_seconds=60.0
    )
    assert report.passed, report.render()
    assert report.exact_edges >= 1
    # First-token anchor was used, so no fallback note fired for the child.
    assert not any("FALLBACK" in n and _C_POST in n for n in report.notes)

    # Twin at the dispatch-fallback position (P + D = 4.0s) must FAIL, because the
    # tool anchored on the observed first token (expected 5.0s), not dispatch + D.
    raw2 = tmp_path / "profile_export_raw.jsonl"
    raw2.write_text(
        "\n".join(
            json.dumps(r) for r in [p_rec, _record(_C_POST, _ORIGIN_NS + int(4.0 * _S))]
        )
        + "\n"
    )
    report2 = causality_timing_vs_real_trace(
        raw2, trace_file, idle_gap_cap_seconds=60.0
    )
    assert not report2.passed
    assert any(_C_POST in m.where for m in report2.mismatches), report2.render()


def test_timing_sim_tool_counts_first_token_edge(tmp_path: Path) -> None:
    """The sibling ``weka_trie_timing_sim`` simulator gates first-token edges off the
    predecessor's DISPATCH (observed ttft == recorded ttft in a pure replay, so
    ``first_token + D' == dispatch + D``) and counts them under a distinct label.

    ``_TTFT_TRACE`` has exactly one post-TTFT overlap (a2:0), so the simulator
    reconstructs the recorded warped timeline byte-exact for all four nodes and
    reports exactly one first-token edge.
    """
    trace_file = _write_trace(tmp_path)

    checked, exact, diverged, first_token_edges = simulate_trace(
        trace_file, cap=60.0, tol=1e-3
    )

    assert diverged == []
    assert checked == exact == 4
    assert first_token_edges == 1

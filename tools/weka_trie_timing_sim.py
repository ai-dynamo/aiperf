#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discrete-event timing simulator for the weka segment-trie IR.

Replays a trie ``ParsedGraph`` with each request's RECORDED ``api_time`` as its
processing duration and checks that the reconstructed per-node start times land
byte-exact on the ORIGINAL recorded timeline (idle-gap-warped). This is a pure
simulation -- no inference server, no ``aiperf`` run -- so it validates the
trie's causal + timing MODEL directly against the recorded ground truth, rather
than (as ``weka_trace_fidelity.py`` does) against a run that itself replays the
same builder.

The check is mathematically tight. The builder fires each node at
``max over incoming edges of (predecessor_completion + edge_delay)`` (START-roots
at ``min_start_delay``; a START-ANCHORED edge --
``delay_after_predecessor_start_us``, a mid-flight spawn / overlap -- fires at
``predecessor_DISPATCH + start_delay`` instead of its completion). If, for every
node, the BINDING cause is the
latest-completing one and its ``delay_after_predecessor_us`` is the warped
end-to-start gap ``warped_start(node) - warped_end(binding)``, then feeding the
recorded ``api_time`` back in reconstructs ``warped_start(node)`` EXACTLY:

    sim_end(binding) = warped_start(binding) + api(binding) = warped_end(binding)
    sim_start(node)  = warped_end(binding) + (warped_start(node) - warped_end(binding))
                     = warped_start(node)                       # exact

and every wait-only edge (delay 0) contributes ``warped_end(pred) <=
warped_end(binding) <= warped_start(node)``, so the ``max`` is the binding's
term. Any divergence therefore flags a real model error: a wrong binding, a bad
delay, a mis-attributed AND-join, or a warp that did not preserve temporal shape.

The recorded warped timeline used as the reference is computed by an INDEPENDENT
idle warp here (over the union of recorded active intervals), so the simulator
does not borrow the builder's own warp for its ground truth.

Usage:
    python tools/weka_trie_timing_sim.py <trace-dir-or-file> [--cap 60] [--tol 1e-3]
"""

from __future__ import annotations

import argparse
import heapq
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson

_DEFAULT_CAP = 60.0
_DEFAULT_TOL = 1e-3  # seconds; float round-trip slack on us<->s conversions


class _IdleWarp:
    """Independent idle warp over the UNION of active intervals (rule 1).

    A true IDLE gap is dead air -- ``next_start`` minus the running max end
    (nothing active in between). Idle > cap collapses to cap; everything after
    shifts left. Active stretches are never cut, so temporal shape is preserved.
    Deliberately re-implemented here (not imported) so the reference timeline is
    independent of the builder.
    """

    def __init__(self, intervals: list[tuple[float, float]], cap: float) -> None:
        self._cuts: list[tuple[float, float]] = []
        if not intervals:
            return
        ordered = sorted(intervals)
        running_end = ordered[0][1]
        cumulative = 0.0
        for start, end in ordered[1:]:
            if start > running_end and (start - running_end) > cap:
                cumulative += (start - running_end) - cap
                self._cuts.append((start, cumulative))
            if end > running_end:
                running_end = end

    def map(self, t: float) -> float:
        shift = 0.0
        for next_start, cumulative in self._cuts:
            if t < next_start:
                break
            shift = cumulative
        return t - shift


def _walk_ids(reqs: list, scope: str) -> Iterator[tuple[str, dict]]:
    """Yield ``(node_id, leaf_request)`` in recorded order, keyed by the same
    ``{scope}:{turn}`` id scheme the trie builder uses.

    ``scope`` is the trajectory scope -- the trace id for the top-level chain,
    the recorded ``agent_id`` for each subagent marker (nested markers use their
    own ``agent_id``). ``turn`` is the 0-based leaf index WITHIN that scope, so a
    subagent marker never consumes a top-level turn index.
    """
    turn = 0
    for req in reqs:
        if isinstance(req, dict) and "requests" in req:
            yield from _walk_ids(req["requests"], req["agent_id"])
        else:
            yield f"{scope}:{turn}", req
            turn += 1


def _flatten_api(trace_dict: dict) -> dict[str, float]:
    """Recorded ``api_time`` per node id, keyed by the same ``{scope}:{turn}`` id
    scheme the trie builder uses."""
    return {
        nid: float(req.get("api_time") or 0.0)
        for nid, req in _walk_ids(trace_dict["requests"], trace_dict["id"])
    }


def _raw_starts(trace_dict: dict) -> dict[str, float]:
    return {
        nid: float(req["t"])
        for nid, req in _walk_ids(trace_dict["requests"], trace_dict["id"])
    }


def _dependency_order(
    trace_file: Path,
    llm: dict[str, Any],
    incoming: dict[str, list[Any]],
    recorded_start: dict[str, float],
) -> list[str]:
    """Topological order over the static edges among LLM nodes.

    Guarantees every predecessor (end- OR start-anchored) is simulated before
    its dependents. A plain ``(recorded_start, node_id)`` sort breaks when a
    start-anchored child ties its parent's recorded start and its node-id string
    sorts first (e.g. ``agent:0`` < ``trace:0``), which would leave the parent
    unsimulated at the child's gate. Ties inside the ready set break by recorded
    start then node id so output stays deterministic.
    """
    indegree = {nid: 0 for nid in llm}
    dependents: dict[str, list[str]] = {nid: [] for nid in llm}
    for nid in llm:
        for e in incoming[nid]:
            if e.source in llm:
                indegree[nid] += 1
                dependents[e.source].append(nid)
    ready = [
        (recorded_start.get(nid, 0.0), nid) for nid, deg in indegree.items() if deg == 0
    ]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        _, nid = heapq.heappop(ready)
        order.append(nid)
        for succ in dependents[nid]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                heapq.heappush(ready, (recorded_start.get(succ, 0.0), succ))
    if len(order) != len(llm):
        unresolved = sorted(set(llm) - set(order))
        raise RuntimeError(
            f"{trace_file}: dependency cycle among trie LLM nodes; cannot "
            f"simulate (unresolved: {unresolved[:5]})"
        )
    return order


def simulate_trace(
    trace_file: Path, cap: float, tol: float
) -> tuple[int, int, list[str], int]:
    """Return (n_checked, n_exact, divergence_lines, n_first_token_edges) for one trace.

    ``n_first_token_edges`` counts start-anchored edges that ALSO carry
    ``delay_after_predecessor_first_token_us`` (a POST-TTFT overlap): a distinct
    kind label. Their gate is unchanged -- a pure replay observes ttft == recorded
    ttft, so ``first_token + D' == dispatch + D`` and the start-anchored branch's
    ``sim_start(source) + start_delay`` already reconstructs them exactly.
    """
    from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
    from aiperf.dataset.graph.adapters.weka.trie_build import build_trie_graph
    from aiperf.dataset.graph.models import LlmNode, StaticEdge

    raw = orjson.loads(Path(trace_file).read_bytes())
    trace = WekaTrace.model_validate(raw)
    graph = build_trie_graph(
        trace,
        tokenizer_name="builtin",
        prompt_corpus="coding",
        root_seed=None,
        idle_gap_cap_seconds=cap,
    )[0].graph

    api = _flatten_api(raw)
    starts = _raw_starts(raw)
    # Independent recorded warped timeline (ground truth) from raw starts + api.
    warp = _IdleWarp([(starts[nid], starts[nid] + api[nid]) for nid in starts], cap)
    recorded_start = {nid: warp.map(t) for nid, t in starts.items()}

    # Incoming static edges per node.
    incoming: dict[str, list[StaticEdge]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        if isinstance(e, StaticEdge) and e.target in incoming:
            incoming[e.target].append(e)

    llm = {nid: n for nid, n in graph.nodes.items() if isinstance(n, LlmNode)}
    order = _dependency_order(trace_file, llm, incoming, recorded_start)

    sim_start: dict[str, float] = {}
    sim_end: dict[str, float] = {}
    checked = exact = first_token_edges = 0
    diverged: list[str] = []
    for nid in order:
        gate = 0.0
        for e in incoming[nid]:
            if e.source == "START":
                gate = max(gate, (e.min_start_delay_us or 0.0) / 1e6)
                continue
            if e.source not in sim_start:
                # Dependency order guarantees every LLM predecessor is simulated
                # first; an unknown source here is a real model error (a dangling
                # or non-LLM endpoint) that zero-defaulting would silently absorb.
                raise RuntimeError(
                    f"{trace_file}: edge {e.source} -> {nid} references a "
                    f"predecessor that was never simulated (unknown or non-LLM "
                    f"source); refusing to zero-default its gate"
                )
            if e.delay_after_predecessor_start_us is not None:
                # Start-anchored: gate off the predecessor's DISPATCH (its own
                # sim_start), NOT its completion -- a mid-flight spawn / overlap
                # fires ``delay`` after the parent dispatched, not after it ended.
                # A POST-TTFT edge (first-token delay also present) shares this
                # gate: in a pure replay first_token + D' == dispatch + D.
                if e.delay_after_predecessor_first_token_us is not None:
                    first_token_edges += 1
                d = e.delay_after_predecessor_start_us / 1e6
                gate = max(gate, sim_start[e.source] + d)
            else:
                d = (e.delay_after_predecessor_us or 0.0) / 1e6
                gate = max(gate, sim_end[e.source] + d)
        sim_start[nid] = gate
        sim_end[nid] = gate + api.get(nid, 0.0)

        ref = recorded_start.get(nid)
        if ref is None:
            continue
        checked += 1
        err = abs(gate - ref)
        if err <= tol:
            exact += 1
        else:
            diverged.append(
                f"    {nid}: sim_start={gate:.3f}s != recorded_warped={ref:.3f}s "
                f"(|d|={err:.3f}s)"
            )
    return checked, exact, diverged, first_token_edges


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="trace dir or single .json trace file")
    ap.add_argument("--cap", type=float, default=_DEFAULT_CAP)
    ap.add_argument("--tol", type=float, default=_DEFAULT_TOL)
    ap.add_argument(
        "--show", type=int, default=5, help="max divergences to print per trace"
    )
    args = ap.parse_args(argv)

    files = sorted(args.path.glob("*.json")) if args.path.is_dir() else [args.path]
    print(f"weka trie timing simulator (idle cap = {args.cap}s, tol = {args.tol}s)")
    if not files:
        print("VACUOUS: nothing checked (no *.json trace files found)")
        return 1
    print(f"{'trace':36s} {'checked':>8s} {'exact':>8s} {'ft_edges':>9s}  status")
    grand_checked = grand_exact = grand_first_token = 0
    failed = False
    for tf in files:
        checked, exact, diverged, first_token_edges = simulate_trace(
            tf, args.cap, args.tol
        )
        ok = exact == checked
        failed = failed or not ok
        print(
            f"{tf.stem[:36]:36s} {checked:8d} {exact:8d} {first_token_edges:9d}  "
            f"{'OK' if ok else 'DIVERGE'}"
        )
        for line in diverged[: args.show]:
            print(line)
        grand_checked += checked
        grand_exact += exact
        grand_first_token += first_token_edges
    print(
        f"\nTOTAL: {grand_exact}/{grand_checked} nodes reconstruct the recorded warped timeline"
    )
    print(
        f"first-token edges (post-TTFT overlap, gated at dispatch + D): "
        f"{grand_first_token}"
    )
    if grand_checked == 0:
        print("VACUOUS: nothing checked (zero LLM nodes simulated)")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

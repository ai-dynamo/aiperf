# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the byte-exact warmup-handoff residual/rebuild golden fixture.

Drives the REAL Python oracle methods on
``aiperf.timing.strategies.agentic_replay.AgenticReplayStrategy``:

- ``_handoff_base_delay_ms`` / ``_handoff_residual_delay_ms`` (lines 958-1006)
  over fixed ``(base inputs, returned_ns, finalized_ns, cap)`` rows, exercising
  the ``delay_ms`` path, the timestamp-fallback base path, the non-finite guard,
  the elapsed-subtraction floor, and the idle-gap-cap clamp.
- ``_build_handoff_replay_boundaries`` + the ``_build_handoff_trajectories``
  state sort ``(agent_depth, x_correlation_id)`` + empty-lane recycle draw
  (lines 1008-1094), with ``uuid.uuid4`` monkeypatched to a deterministic
  sequence injected identically on the Rust side.

The bare strategy instance is built with ``object.__new__`` so only the exact
attributes the ported methods touch are populated (no full ``__init__``).

Run: ``source .venv/bin/activate && python tools/agentx_handoff_residual_golden.py``
Writes ``tests/fixtures/agentx/handoff_residual_golden.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing import strategies as _strategies_pkg  # noqa: F401
from aiperf.timing.trajectory_source import (
    ConversationState,
    ReplayResumeBoundary,
    Trajectory,
)
from aiperf.timing import trajectory_source as _traj_mod

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "agentx"
    / "handoff_residual_golden.json"
)


def _decode_f64(v):
    """Decode the JSON-safe float encoding used in this fixture."""
    if v is None:
        return None
    if isinstance(v, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[v]
    return float(v)


def _turn(**kw):
    return SimpleNamespace(
        timestamp_ms=kw.get("timestamp_ms"),
        delay_ms=kw.get("delay_ms"),
        api_time_ms=kw.get("api_time_ms"),
    )


class _FakeSource:
    """Minimal ConversationSource surface for the base-delay oracle."""

    def __init__(self, turns):
        self._turns = turns

    def get_next_turn_metadata(self, credit):
        next_index = credit.turn_index + 1
        if next_index >= len(self._turns):
            raise ValueError("no next turn")
        return self._turns[next_index]

    def get_metadata(self, conversation_id):
        return SimpleNamespace(turns=self._turns)


def _bare_strategy(*, turns, returned_at_ns, cap_ms):
    strat = object.__new__(AgenticReplayStrategy)
    strat.conversation_source = _FakeSource(turns)
    strat._handoff_returned_at_ns = dict(returned_at_ns)
    strat._phase_offset_cap_ms = cap_ms
    return strat


def _residual_rows():
    rows = []

    def add(name, *, next_delay_ms, prev_ts_ms, next_ts_ms, prev_api_ms,
            returned_ns, finalized_ns, cap_ms, turn_index=0, num_turns=3):
        # turns[0] = previous (returned) turn, turns[1] = next turn.
        turns = [
            _turn(timestamp_ms=_decode_f64(prev_ts_ms), api_time_ms=_decode_f64(prev_api_ms)),
            _turn(timestamp_ms=_decode_f64(next_ts_ms), delay_ms=_decode_f64(next_delay_ms)),
        ]
        credit = SimpleNamespace(
            conversation_id="conv",
            x_correlation_id="x-corr",
            turn_index=turn_index,
            num_turns=num_turns,
        )
        returned_at = {} if returned_ns is None else {"x-corr": returned_ns}
        strat = _bare_strategy(turns=turns, returned_at_ns=returned_at, cap_ms=_decode_f64(cap_ms))
        base = strat._handoff_base_delay_ms(credit)
        residual = strat._handoff_residual_delay_ms(credit, finalized_at_ns=finalized_ns)
        rows.append({
            "name": name,
            "next_delay_ms": next_delay_ms,
            "prev_timestamp_ms": prev_ts_ms,
            "next_timestamp_ms": next_ts_ms,
            "prev_api_time_ms": prev_api_ms,
            "returned_ns": returned_ns,
            "finalized_ns": finalized_ns,
            "cap_ms": cap_ms,
            "expected_base_ms": base,
            "expected_residual_ms": residual,
        })

    # delay_ms path, no returned wall, no cap.
    add("delay_path_no_returned", next_delay_ms=25.0, prev_ts_ms=None,
        next_ts_ms=None, prev_api_ms=None, returned_ns=None, finalized_ns=0, cap_ms=None)
    # delay_ms path, elapsed subtraction floor (elapsed < base).
    add("delay_path_elapsed_partial", next_delay_ms=25.0, prev_ts_ms=None,
        next_ts_ms=None, prev_api_ms=None, returned_ns=1_000_000, finalized_ns=6_000_000, cap_ms=None)
    # delay_ms path, elapsed exceeds base -> floored to 0.
    add("delay_path_elapsed_floor", next_delay_ms=5.0, prev_ts_ms=None,
        next_ts_ms=None, prev_api_ms=None, returned_ns=1_000_000, finalized_ns=99_000_000, cap_ms=None)
    # idle-gap cap clamp (base above cap).
    add("delay_path_cap_clamp", next_delay_ms=500.0, prev_ts_ms=None,
        next_ts_ms=None, prev_api_ms=None, returned_ns=None, finalized_ns=0, cap_ms=120.0)
    # negative delay_ms clamps to 0.
    add("delay_negative_clamp", next_delay_ms=-10.0, prev_ts_ms=None,
        next_ts_ms=None, prev_api_ms=None, returned_ns=None, finalized_ns=0, cap_ms=None)
    # non-finite delay_ms -> None -> timestamp fallback.
    add("nonfinite_delay_timestamp_fallback", next_delay_ms="nan", prev_ts_ms=100.0,
        next_ts_ms=180.0, prev_api_ms=30.0, returned_ns=None, finalized_ns=0, cap_ms=None)
    # timestamp fallback: next - prev - max(0, api).
    add("timestamp_fallback", next_delay_ms=None, prev_ts_ms=1000.0,
        next_ts_ms=1250.0, prev_api_ms=40.0, returned_ns=None, finalized_ns=0, cap_ms=None)
    # timestamp fallback with negative api treated as 0.
    add("timestamp_fallback_negative_api", next_delay_ms=None, prev_ts_ms=1000.0,
        next_ts_ms=1250.0, prev_api_ms=-40.0, returned_ns=None, finalized_ns=0, cap_ms=None)
    # timestamp fallback where gap < api -> floored to 0.
    add("timestamp_fallback_floor", next_delay_ms=None, prev_ts_ms=1000.0,
        next_ts_ms=1050.0, prev_api_ms=200.0, returned_ns=None, finalized_ns=0, cap_ms=None)
    # missing next timestamp -> base 0.
    add("timestamp_fallback_missing", next_delay_ms=None, prev_ts_ms=1000.0,
        next_ts_ms=None, prev_api_ms=10.0, returned_ns=None, finalized_ns=0, cap_ms=None)
    # final credit (no next turn) -> base 0, everything below cap.
    add("final_credit_no_next", next_delay_ms=None, prev_ts_ms=None,
        next_ts_ms=None, prev_api_ms=None, returned_ns=None, finalized_ns=0, cap_ms=None,
        turn_index=2, num_turns=3)
    # combined: timestamp fallback + returned elapsed + cap.
    add("fallback_returned_and_cap", next_delay_ms=None, prev_ts_ms=1000.0,
        next_ts_ms=2000.0, prev_api_ms=0.0, returned_ns=2_000_000, finalized_ns=5_000_000, cap_ms=400.0)
    return rows


def _trajectory_scenario():
    """Oracle for the rebuild: state sort + boundary merge + empty-lane recycle."""
    strat = object.__new__(AgenticReplayStrategy)

    # Deterministic recycle draws injected identically on both sides.
    recycle_ids = ["recycle-conv-0", "recycle-conv-1"]
    recycle_corrs = ["recycle-corr-0", "recycle-corr-1"]
    draw_iter = iter(recycle_ids)
    corr_iter = iter(recycle_corrs)

    # Completed-prefix history per tree root (merged into boundaries).
    completed = {
        "root-a": [ReplayResumeBoundary("conv-a", 2), ReplayResumeBoundary("conv-hist", 5)],
    }

    class _Gate:
        def completed_prefixes(self, root_correlation_id):
            return completed.get(root_correlation_id, [])

    class _Source:
        def __init__(self):
            self.trajectories = [
                Trajectory("prev-a", 0, None, "prev-corr-a"),
                Trajectory("prev-b", 0, None, "prev-corr-b"),
            ]

        def next_recycle_conversation_id(self):
            return next(draw_iter, None)

    strat.conversation_source = _Source()
    strat.credit_issuer = SimpleNamespace(replay_gate=_Gate())

    # Monkeypatch uuid4 in trajectory_source (where Trajectory's default lives)
    # AND in agentic_replay (the recycle correlation id) to the injected sequence.
    import uuid as _uuid

    orig = _uuid.uuid4
    _uuid.uuid4 = lambda: next(corr_iter)

    # Lane 0: two live streams (depth 1 then depth 0, out of order to prove sort);
    # lane 1: empty -> recycle draw.
    states_by_lane = {
        0: [
            ConversationState(
                conversation_id="conv-child",
                x_correlation_id="x-child",
                next_turn_index=3,
                agent_depth=1,
                root_correlation_id="root-a",
            ),
            ConversationState(
                conversation_id="conv-a",
                x_correlation_id="root-a",
                next_turn_index=2,
                agent_depth=0,
                root_correlation_id=None,
            ),
        ],
        1: [],
    }
    boundaries_by_lane = {
        lane: strat._build_handoff_replay_boundaries(states)
        for lane, states in states_by_lane.items()
    }
    rebuilt = strat._build_handoff_trajectories(states_by_lane, boundaries_by_lane)
    _uuid.uuid4 = orig

    lanes = []
    for lane, traj in enumerate(rebuilt):
        snap = traj.snapshot
        lanes.append({
            "lane": lane,
            "state_order": [
                [s.agent_depth, s.x_correlation_id] for s in snap.states
            ],
            "boundaries": [
                [b.conversation_id, b.next_turn_index]
                for b in snap.replay_resume_boundaries
            ],
        })

    return {
        "recycle_conversation_ids": recycle_ids,
        "recycle_correlation_ids": recycle_corrs,
        "completed_prefixes": {
            root: [[b.conversation_id, b.next_turn_index] for b in bs]
            for root, bs in completed.items()
        },
        # Input states expressed as (lane, conv, x_corr, next_turn_index,
        # agent_depth, root_correlation_id) so the Rust side reconstructs them.
        "input_states": [
            {
                "lane": lane,
                "conversation_id": s.conversation_id,
                "x_correlation_id": s.x_correlation_id,
                "next_turn_index": s.next_turn_index,
                "agent_depth": s.agent_depth,
                "root_correlation_id": s.root_correlation_id,
            }
            for lane, states in states_by_lane.items()
            for s in states
        ],
        "num_lanes": len(states_by_lane),
        "expected_lanes": lanes,
    }


def _pending_scenario():
    """Oracle for the PENDING builder path (Finding 2).

    Drives the REAL ``_build_handoff_states`` (returned + pending) plus
    ``_build_handoff_replay_boundaries`` so the fixture exercises: (a) a
    returned mid-flight credit at residual offset, (b) barrier-pending turns at
    offset ``0.0``, (c) returned-vs-pending dedup, and (d) two pending turns that
    share an effective-root where the SECOND resolves its lane only via the
    intra-finalize ``_root_to_lane`` mutation written by the FIRST (Finding 1).
    """
    from aiperf.timing.trajectory_source import ConversationBranchMode

    fork = ConversationBranchMode.FORK

    def _credit(**kw):
        ns = SimpleNamespace(branch_mode=fork, **kw)
        ns.effective_root_correlation_id = (
            kw.get("root_correlation_id") or kw["x_correlation_id"]
        )
        return ns

    # Returned mid-flight credit on lane 0 (turn 0 of 3 -> resumes at turn 1).
    returned = _credit(
        conversation_id="conv-a",
        x_correlation_id="x-a",
        turn_index=0,
        num_turns=3,
        agent_depth=0,
        parent_correlation_id=None,
        root_correlation_id=None,
    )
    returned_ns = 1_000_000  # 1ms
    finalized_ns = 4_000_000  # 3ms elapsed
    # base delay for the returned credit, served via the fake source below.
    returned_next_delay_ms = 30.0

    # Pending turns. turnA resolves via correlation_to_lane[parent]; it writes
    # _root_to_lane["shared-root"]=1. turnB (grouped under its own root key
    # "grp2") can ONLY resolve via that write -> exercises Finding 1.
    turn_a = _credit(
        conversation_id="conv-A",
        x_correlation_id="x-A",
        turn_index=1,
        num_turns=3,
        agent_depth=1,
        parent_correlation_id="x-A-parent",
        root_correlation_id="shared-root",
    )
    turn_b = _credit(
        conversation_id="conv-B",
        x_correlation_id="x-B",
        turn_index=0,
        num_turns=2,
        agent_depth=0,
        parent_correlation_id=None,
        root_correlation_id="shared-root",
    )
    # A pending turn that duplicates the returned state key (conv-a, x-a, 1):
    # deduped away by seen_states (returned processed first).
    turn_dup = _credit(
        conversation_id="conv-a",
        x_correlation_id="x-a",
        turn_index=1,
        num_turns=3,
        agent_depth=0,
        parent_correlation_id=None,
        root_correlation_id=None,
    )

    pending_by_root = {
        "grp1": (turn_a,),
        "grp2": (turn_b,),
        "x-a": (turn_dup,),
    }

    class _Gate:
        def pending_turns_by_root(self):
            return dict(pending_by_root)

        def completed_prefixes(self, root_correlation_id):
            if root_correlation_id == "x-a":
                return [ReplayResumeBoundary("conv-a", 2)]
            return []

    class _Source:
        def __init__(self):
            # length == num_lanes; content unused by _build_handoff_states.
            self.trajectories = [None, None]

        def get_next_turn_metadata(self, credit):
            # Only the returned credit needs a base delay; pending turns use 0.0.
            return SimpleNamespace(delay_ms=returned_next_delay_ms)

        def get_metadata(self, conversation_id):
            raise KeyError(conversation_id)

    strat = object.__new__(AgenticReplayStrategy)
    strat.conversation_source = _Source()
    strat.credit_issuer = SimpleNamespace(replay_gate=_Gate())
    strat.branch_orchestrator = None
    strat._handoff_credits = {"x-a": returned}
    strat._handoff_returned_at_ns = {"x-a": returned_ns}
    strat._phase_offset_cap_ms = None
    strat._root_to_lane = {"x-a": 0}
    strat._correlation_to_lane = {"x-A-parent": 1}

    states_by_lane = strat._build_handoff_states(finalized_at_ns=finalized_ns)
    boundaries_by_lane = {
        lane: strat._build_handoff_replay_boundaries(states)
        for lane, states in states_by_lane.items()
    }

    def _emit_state(s):
        return {
            "conversation_id": s.conversation_id,
            "x_correlation_id": s.x_correlation_id,
            "next_turn_index": s.next_turn_index,
            "next_dispatch_offset_ms": s.next_dispatch_offset_ms,
            "agent_depth": s.agent_depth,
            "parent_correlation_id": s.parent_correlation_id,
            "root_correlation_id": s.root_correlation_id,
        }

    expected_lanes = []
    for lane in sorted(states_by_lane):
        # Match the Rust finalize per-lane sort (agent_depth, x_correlation_id).
        sorted_states = sorted(
            states_by_lane[lane],
            key=lambda s: (s.agent_depth, s.x_correlation_id),
        )
        expected_lanes.append({
            "lane": lane,
            "states": [_emit_state(s) for s in sorted_states],
            "boundaries": [
                [b.conversation_id, b.next_turn_index]
                for b in boundaries_by_lane[lane]
            ],
        })

    def _emit_turn(t):
        return {
            "conversation_id": t.conversation_id,
            "x_correlation_id": t.x_correlation_id,
            "turn_index": t.turn_index,
            "num_turns": t.num_turns,
            "agent_depth": t.agent_depth,
            "parent_correlation_id": t.parent_correlation_id,
            "root_correlation_id": t.root_correlation_id,
        }

    return {
        "num_lanes": 2,
        "finalized_ns": finalized_ns,
        "cap_ms": None,
        "returned_credits": [
            {
                **_emit_turn(returned),
                "returned_ns": returned_ns,
                "base_delay_inputs": {"next_delay_ms": returned_next_delay_ms},
            }
        ],
        "root_to_lane": {"x-a": 0},
        "correlation_to_lane": {"x-A-parent": 1},
        "pending_by_root": [
            {"root": root, "turns": [_emit_turn(t) for t in turns]}
            for root, turns in pending_by_root.items()
        ],
        "completed_prefixes": {"x-a": [["conv-a", 2]]},
        "expected_lanes": expected_lanes,
    }


def main():
    fixture = {
        "_comment": "Generated by tools/agentx_handoff_residual_golden.py from the "
        "real AgenticReplayStrategy oracle. Do not edit by hand.",
        "residual_rows": _residual_rows(),
        "trajectory": _trajectory_scenario(),
        "pending": _pending_scenario(),
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix C (F2/A2) — graph credits bypass CreditCounter session arithmetic.

The proven bug: graph nodes flow through ``CreditCounter.increment_sent`` with
``agent_depth==0`` and per-node ``turn_index==0``, so each NODE bumped
``_sent_sessions`` / ``_total_session_turns`` and tripped ``is_final_credit``
after the Nth node (freezing the sent-count mid-trace -> lost records).

The fix gates on ``turn.trace_id is not None``: graph credits bump ONLY
``_requests_sent`` (the per-node record count) and NEVER flip ``is_final_credit``.
These tests pin both the graph bypass and the unchanged non-graph path.
"""

from __future__ import annotations

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter


def _graph_turn(trace_id: str = "t-1#0") -> TurnToSend:
    """A graph node dispatch: distinct nodes all fire with turn_index==0."""
    return TurnToSend(
        conversation_id=trace_id,
        x_correlation_id=f"x|{trace_id}",
        turn_index=0,
        num_turns=1,
        trace_id=trace_id,
    )


def _linear_turn(turn_index: int, num_turns: int) -> TurnToSend:
    return TurnToSend(
        conversation_id="c0",
        x_correlation_id="x0",
        turn_index=turn_index,
        num_turns=num_turns,
    )


def _counter(**caps) -> CreditCounter:
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.GRAPH_IR,
        **caps,
    )
    return CreditCounter(config)


def test_graph_credits_do_not_increment_sessions():
    """Three distinct-node graph credits bump requests_sent only, never sessions."""
    counter = _counter()
    for _ in range(3):
        _, is_final = counter.increment_sent(_graph_turn())
        assert is_final is False

    assert counter.requests_sent == 3
    assert counter.sent_sessions == 0
    assert counter.total_session_turns == 0
    assert counter.root_requests_sent == 0


def test_graph_credits_never_trip_session_cap():
    """The proven bug: with expected_num_sessions=1, is_final_credit fired on
    the FIRST node. After the fix it must NEVER fire for a graph credit."""
    counter = _counter(expected_num_sessions=1)
    for _ in range(5):
        _, is_final = counter.increment_sent(_graph_turn())
        assert is_final is False
    assert counter.sent_sessions == 0


def test_graph_credits_never_trip_request_cap_via_counter():
    """Even with a request-count cap, the COUNTER must not auto-flip
    is_final_credit for graph credits -- the strategy owns the cap (the
    RequestCountStopCondition gate enforces it at issue time)."""
    counter = _counter(total_expected_requests=2)
    for _ in range(3):
        _, is_final = counter.increment_sent(_graph_turn())
        assert is_final is False
    assert counter.requests_sent == 3


def test_non_graph_session_accounting_unchanged():
    """A 2-turn linear session counts as ONE session with 2 turns, and
    is_final_credit fires only once both clauses are satisfied -- the graph
    trace_id gate must not perturb this."""
    counter = _counter(expected_num_sessions=1)

    _, final0 = counter.increment_sent(_linear_turn(0, 2))
    assert counter.sent_sessions == 1
    assert counter.total_session_turns == 2
    assert counter.requests_sent == 1
    assert final0 is False

    _, final1 = counter.increment_sent(_linear_turn(1, 2))
    assert counter.sent_sessions == 1
    assert counter.requests_sent == 2
    assert final1 is True


def test_graph_returns_do_not_touch_session_counters():
    """TC3: graph returns bypass session counters like the sent side.

    Every graph credit is minted ``turn_index=0, num_turns=1`` so
    ``is_final_turn`` is always True at return time; without the bypass each
    NODE return bumped ``completed_sessions`` while ``sent_sessions`` stayed 0
    (negative ``in_flight_sessions``, bogus progress %). Session stats must
    stay 0 on BOTH sides for a graph phase; progress is request-count driven.
    """
    counter = _counter()
    for _ in range(4):
        counter.increment_sent(_graph_turn())
    for _ in range(2):
        counter.increment_returned(is_final_turn=True, cancelled=False)
    counter.increment_returned(is_final_turn=True, cancelled=False, errored=True)
    counter.increment_returned(is_final_turn=True, cancelled=True)

    assert counter.completed_sessions == 0
    assert counter.cancelled_sessions == 0
    assert counter.sent_sessions == 0
    assert counter.in_flight_sessions == 0  # never negative
    # Request-level counters still drive progress + the all-returned invariant.
    assert counter.requests_completed == 3
    assert counter.requests_cancelled == 1
    assert counter.request_errors == 1
    assert counter.in_flight == 0


def test_graph_returns_still_drive_all_returned_invariant():
    """The completion barrier stays request-count driven for graph phases."""
    counter = _counter()
    for _ in range(2):
        counter.increment_sent(_graph_turn())
    counter.freeze_sent_counts()

    assert counter.increment_returned(is_final_turn=True, cancelled=False) is False
    assert counter.increment_returned(is_final_turn=True, cancelled=False) is True
    assert counter.check_all_returned_or_cancelled() is True


def test_non_graph_phase_returned_session_accounting_unchanged():
    """A linear (REQUEST_RATE) phase still counts completed/cancelled sessions."""
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
    )
    counter = CreditCounter(config)
    counter.increment_sent(_linear_turn(0, 1))
    counter.increment_sent(_linear_turn(0, 1))

    counter.increment_returned(is_final_turn=True, cancelled=False)
    counter.increment_returned(is_final_turn=True, cancelled=True)

    assert counter.completed_sessions == 1
    assert counter.cancelled_sessions == 1


def test_non_graph_dag_child_accounting_unchanged():
    """A DAG child (agent_depth > 0, trace_id None) still bumps requests_sent
    only and honors the request cap -- unchanged by the graph gate."""
    counter = _counter(total_expected_requests=2)
    child = TurnToSend(
        conversation_id="c0",
        x_correlation_id="x0::child",
        turn_index=0,
        num_turns=1,
        agent_depth=1,
    )
    _, final0 = counter.increment_sent(child)
    assert counter.requests_sent == 1
    assert counter.sent_sessions == 0
    assert final0 is False

    _, final1 = counter.increment_sent(child)
    assert counter.requests_sent == 2
    assert final1 is True  # request cap crossed on a child

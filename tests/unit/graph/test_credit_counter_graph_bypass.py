# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix C (F2/A2) — graph credits bypass CreditCounter session arithmetic."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

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
    """A classic session turn: no trace_id, so session arithmetic applies."""
    return TurnToSend(
        conversation_id="c0",
        x_correlation_id="x0",
        turn_index=turn_index,
        num_turns=num_turns,
    )


def _counter(
    timing_mode: TimingMode = TimingMode.AGENT_GRAPH, **caps: Any
) -> CreditCounter:
    """A PROFILING-phase counter under the given timing mode and stop caps."""
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=timing_mode,
        **caps,
    )
    return CreditCounter(config)


@pytest.mark.parametrize(
    ("caps", "num_credits"),
    [
        param({}, 3, id="no-caps"),
        # The proven bug: is_final_credit fired on the FIRST node here.
        param({"expected_num_sessions": 1}, 5, id="session-cap-never-trips"),
        # The strategy (RequestCountStopCondition) owns the request cap, not the
        # counter -- it must not auto-flip is_final_credit for graph credits.
        param({"total_expected_requests": 2}, 3, id="request-cap-never-trips"),
    ],
)  # fmt: skip
def test_graph_credits_bump_requests_only_and_never_flip_is_final(
    caps: dict[str, Any], num_credits: int
) -> None:
    """Graph credits bump requests_sent only, leave session counters at 0, never finalize."""
    counter = _counter(**caps)

    for _ in range(num_credits):
        _, is_final = counter.increment_sent(_graph_turn())
        assert is_final is False

    assert counter.requests_sent == num_credits
    assert counter.sent_sessions == 0
    assert counter.total_session_turns == 0
    assert counter.root_requests_sent == 0


def test_graph_returns_do_not_touch_session_counters() -> None:
    """TC3: graph returns bypass session counters exactly like the sent side."""
    # Every graph credit is minted turn_index=0/num_turns=1, so is_final_turn is
    # always True at return time; without the bypass each NODE return bumped
    # completed_sessions while sent_sessions stayed 0 (negative in_flight_sessions).
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


def test_graph_returns_still_drive_all_returned_invariant() -> None:
    """The completion barrier stays request-count driven for graph phases."""
    counter = _counter()
    for _ in range(2):
        counter.increment_sent(_graph_turn())
    counter.freeze_sent_counts()

    assert counter.increment_returned(is_final_turn=True, cancelled=False) is False
    assert counter.increment_returned(is_final_turn=True, cancelled=False) is True
    assert counter.check_all_returned_or_cancelled() is True


def test_non_graph_session_accounting_unchanged() -> None:
    """A 2-turn linear session stays ONE session of 2 turns, finalizing on the last turn."""
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


def test_non_graph_phase_returned_session_accounting_unchanged() -> None:
    """A linear (REQUEST_RATE) phase still counts completed/cancelled sessions."""
    counter = _counter(timing_mode=TimingMode.REQUEST_RATE)
    counter.increment_sent(_linear_turn(0, 1))
    counter.increment_sent(_linear_turn(0, 1))

    counter.increment_returned(is_final_turn=True, cancelled=False)
    counter.increment_returned(is_final_turn=True, cancelled=True)

    assert counter.completed_sessions == 1
    assert counter.cancelled_sessions == 1


def test_non_graph_dag_child_accounting_unchanged() -> None:
    """A DAG child (agent_depth>0, trace_id None) bumps requests only but still honors the cap."""
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

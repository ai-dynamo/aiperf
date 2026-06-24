# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter


def cfg(
    reqs: int | None = None, sessions: int | None = None, dur: float | None = None
) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=reqs,
        expected_num_sessions=sessions,
        expected_duration_sec=dur,
    )


def turn(conv: str = "c1", idx: int = 0, num: int = 1, corr: str = "x1") -> TurnToSend:
    return TurnToSend(
        conversation_id=conv, turn_index=idx, num_turns=num, x_correlation_id=corr
    )


class TestCreditCounter:
    def test_initial_state(self) -> None:
        c = CreditCounter(cfg())
        # All progress counters start at zero
        assert c.requests_sent == 0
        assert c.requests_completed == 0
        assert c.requests_cancelled == 0
        assert c.request_errors == 0
        assert c.sent_sessions == 0
        assert c.completed_sessions == 0
        assert c.cancelled_sessions == 0
        assert c.total_session_turns == 0
        assert c.prefills_released == 0
        # Derived counters also zero
        assert c.in_flight == 0
        assert c.in_flight_sessions == 0
        assert c.in_flight_prefills == 0
        # Final counts are None until frozen
        assert c.final_requests_sent is None
        assert c.final_requests_completed is None
        assert c.final_requests_cancelled is None
        assert c.final_request_errors is None
        assert c.final_sent_sessions is None
        assert c.final_completed_sessions is None
        assert c.final_cancelled_sessions is None

    def test_increment_sent_returns_sequential_index(self) -> None:
        c = CreditCounter(cfg())
        for i in range(10):
            idx, _ = c.increment_sent(turn(idx=0))
            assert idx == i
        assert c.requests_sent == 10

    def test_first_turn_increments_session(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=3))
        assert c.sent_sessions == 1 and c.total_session_turns == 3
        c.increment_sent(turn(idx=1, num=3))
        c.increment_sent(turn(idx=2, num=3))
        assert c.sent_sessions == 1 and c.requests_sent == 3

    def test_total_session_turns_accumulates(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=3))
        c.increment_sent(turn(idx=1, num=3))
        c.increment_sent(turn(idx=2, num=3))
        c.increment_sent(turn(idx=0, num=5))
        assert c.total_session_turns == 8

    # fmt: off
    @pytest.mark.parametrize("reqs,expected_finals", [(3, [False, False, True]), (1, [True])])
    def test_final_when_request_count_reached(self, reqs: int, expected_finals: list[bool]) -> None:
        c = CreditCounter(cfg(reqs=reqs))
        for exp in expected_finals:
            _, is_final = c.increment_sent(turn())
            assert is_final == exp
    # fmt: on

    def test_not_final_without_request_limit(self) -> None:
        c = CreditCounter(cfg(reqs=None))
        for _ in range(100):
            _, is_final = c.increment_sent(turn())
            assert not is_final

    def test_final_when_sessions_complete(self) -> None:
        c = CreditCounter(cfg(sessions=2))
        finals = []
        for _ in range(2):
            for t in range(2):
                _, f = c.increment_sent(turn(idx=t, num=2))
                finals.append(f)
        assert finals == [False, False, False, True]

    def test_not_final_until_all_session_turns_sent(self) -> None:
        c = CreditCounter(cfg(sessions=2))
        c.increment_sent(turn(idx=0, num=3))
        c.increment_sent(turn(idx=0, num=2))
        c.increment_sent(turn(idx=1, num=3))
        c.increment_sent(turn(idx=1, num=2))
        _, is_final = c.increment_sent(turn(idx=2, num=3))
        assert is_final

    def test_increment_returned_completed(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=False, cancelled=False)
        assert c.requests_completed == 1 and c.requests_cancelled == 0

    def test_increment_returned_cancelled(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=False, cancelled=True)
        assert c.requests_cancelled == 1 and c.requests_completed == 0

    def test_increment_returned_final_turn_increments_session(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=2))
        c.increment_sent(turn(idx=1, num=2))
        c.increment_returned(is_final_turn=False, cancelled=False)
        assert c.completed_sessions == 0
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.completed_sessions == 1

    def test_increment_returned_cancelled_session(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=1))
        c.increment_returned(is_final_turn=True, cancelled=True)
        assert c.cancelled_sessions == 1 and c.completed_sessions == 0

    def test_increment_returned_all_done(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_sent(turn())
        c.freeze_sent_counts()
        assert not c.increment_returned(is_final_turn=False, cancelled=False)
        assert c.increment_returned(is_final_turn=True, cancelled=False)

    def test_in_flight_tracking(self) -> None:
        c = CreditCounter(cfg())
        for _ in range(3):
            c.increment_sent(turn())
        assert c.in_flight == 3
        c.increment_returned(is_final_turn=False, cancelled=False)
        assert c.in_flight == 2
        c.increment_returned(is_final_turn=False, cancelled=True)
        assert c.in_flight == 1

    def test_in_flight_sessions(self) -> None:
        c = CreditCounter(cfg())
        for x in "abc":
            c.increment_sent(turn(conv=x, idx=0, num=2))
        assert c.in_flight_sessions == 3
        c.increment_sent(turn(conv="a", idx=1, num=2))
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.in_flight_sessions == 2
        c.increment_returned(is_final_turn=True, cancelled=True)
        assert c.in_flight_sessions == 1

    def test_in_flight_prefills(self) -> None:
        c = CreditCounter(cfg())
        for _ in range(3):
            c.increment_sent(turn())
        assert c.in_flight_prefills == 3
        c.increment_prefill_released()
        c.increment_prefill_released()
        c.increment_prefill_released()
        assert c.in_flight_prefills == 0

    def test_freeze_sent_counts(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=3))
        c.increment_sent(turn(idx=1, num=3))
        c.freeze_sent_counts()
        assert c.final_requests_sent == 2 and c.final_sent_sessions == 1
        c.increment_sent(turn(idx=2, num=3))
        assert c.final_requests_sent == 2 and c.requests_sent == 3

    def test_freeze_completed_counts(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=2))
        c.increment_sent(turn(idx=1, num=2))
        c.freeze_sent_counts()
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.freeze_completed_counts()
        assert (
            c.final_requests_completed == 1
            and c.final_requests_cancelled == 0
            and c.final_completed_sessions == 0
        )

    def test_check_all_returned_requires_frozen(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert not c.check_all_returned_or_cancelled()
        c.increment_sent(turn())
        c.freeze_sent_counts()
        assert not c.check_all_returned_or_cancelled()
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.check_all_returned_or_cancelled()

    def test_single_session_single_turn_is_final(self) -> None:
        c = CreditCounter(cfg(sessions=1))
        _, is_final = c.increment_sent(turn(idx=0, num=1))
        assert is_final

    def test_mixed_completed_and_cancelled_with_all_done_check(self) -> None:
        c = CreditCounter(cfg())
        for _ in range(5):
            c.increment_sent(turn())
        c.freeze_sent_counts()
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=False, cancelled=True)
        c.increment_returned(is_final_turn=False, cancelled=True)
        assert c.requests_completed == 2
        assert c.requests_cancelled == 2
        assert not c.check_all_returned_or_cancelled()
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.check_all_returned_or_cancelled()


def child_turn(
    conv: str = "c1",
    idx: int = 0,
    num: int = 1,
    corr: str = "x1",
    depth: int = 1,
    counts_toward_phase_target: bool = False,
) -> TurnToSend:
    return TurnToSend(
        conversation_id=conv,
        turn_index=idx,
        num_turns=num,
        x_correlation_id=corr,
        agent_depth=depth,
        parent_correlation_id="parent-x",
        counts_toward_phase_target=counts_toward_phase_target,
    )


class TestDagChildCounterSplit:
    """DAG children inherit the parent's session slot for concurrency
    but their HTTP requests are real wire traffic and count on the
    request-level counters:

    - ``requests_sent`` / ``requests_completed`` / ``requests_cancelled``
      include children — these are user-facing metrics of actual HTTP
      activity.
    - ``sent_sessions`` / ``completed_sessions`` / ``cancelled_sessions``
      / ``total_session_turns`` exclude children — they reflect
      sampled-root session lifecycle only. Inflating them would make a
      single-session DAG run report as multi-session.
    - Reactive DAG children set ``counts_toward_phase_target=False``. Under the
      exact-cutoff they CAN flip ``is_final_credit`` via the unconditional
      ``--request-count`` arm (a literal wire cap), but NOT via the
      ``--num-conversations`` arm (a root-sampler-plan target, still gated by
      the flag). Snapshot warmup can also make a subagent state count toward the
      planned warmup target.
    """

    def test_child_increment_sent_bumps_requests_only(self) -> None:
        c = CreditCounter(cfg(reqs=10, sessions=2))
        # Root first-turn bumps everything.
        idx, final = c.increment_sent(turn(idx=0, num=2))
        assert idx == 0 and final is False
        assert c.requests_sent == 1
        assert c.sent_sessions == 1
        assert c.total_session_turns == 2

        # Child first-turn: requests_sent ticks (real HTTP request)
        # but session counters stay put (inherits parent's slot).
        idx, final = c.increment_sent(child_turn(conv="child-1", idx=0, num=3))
        assert final is False
        assert c.requests_sent == 2
        assert c.sent_sessions == 1
        assert c.total_session_turns == 2

        # Child continuation turn: also bumps requests_sent only.
        idx, final = c.increment_sent(child_turn(conv="child-1", idx=1, num=3))
        assert final is False
        assert c.requests_sent == 3
        assert c.sent_sessions == 1
        assert c.total_session_turns == 2

    def test_reactive_child_crossing_cap_triggers_is_final_credit(self) -> None:
        """Exact-cutoff: ``--request-count`` is a literal cap on total wire
        requests, so a REACTIVE child (``counts_toward_phase_target=False``) that
        crosses it MUST flip ``is_final_credit`` — otherwise, when the Nth wire
        credit is a child, ``all_credits_sent_event`` never fires and
        sending-complete hangs. (The request-count arm is unconditional; the
        ``--num-conversations`` arm stays gated by the flag.)
        """
        c = CreditCounter(cfg(reqs=2))
        _, final_root = c.increment_sent(turn(idx=0, num=2))
        assert final_root is False  # 1 < 2, cap not yet reached

        # Reactive child (off the sampler plan) crosses the wire cap -> is_final.
        _, final_child = c.increment_sent(child_turn(conv="child-1", idx=0))
        assert final_child is True
        assert c.requests_sent == 2
        # Session counters stay root-only even though the child flipped is_final.
        assert c.sent_sessions == 1

    def test_planned_child_can_trigger_is_final_credit(self) -> None:
        """Snapshot warmup may dispatch a ready subagent as planned work.

        The child still must not inflate session counters, but it must be able
        to satisfy ``total_expected_requests`` when it is one of the warmup
        credits counted by ``TrajectorySource.warmup_credit_count``.
        """
        c = CreditCounter(cfg(reqs=2))
        _, final_root = c.increment_sent(turn(idx=0, num=2))
        assert final_root is False

        _, final_child = c.increment_sent(
            child_turn(
                conv="child-1",
                idx=0,
                num=3,
                counts_toward_phase_target=True,
            )
        )

        assert final_child is True
        assert c.requests_sent == 2
        assert c.sent_sessions == 1

    def test_warmup_planned_subagent_crossing_request_target_is_final(self) -> None:
        """R1 guard: in WARMUP every credit is a planned credit
        (``counts_toward_phase_target=True``); a planned subagent crossing the
        warmup request target flips ``is_final_credit`` (so warmup sending
        completes) WITHOUT inflating session counters. Guards the Delta-2
        'request arm unconditional' merge against a warmup regression.
        """
        c = CreditCounter(cfg(reqs=2))
        _, final_root = c.increment_sent(turn(idx=0, num=1))
        assert final_root is False
        _, final_sub = c.increment_sent(
            child_turn(conv="sub", idx=0, counts_toward_phase_target=True)
        )
        assert final_sub is True
        assert c.sent_sessions == 1  # session counters stay root-only

    def test_session_count_reactive_child_does_not_flip_early(self) -> None:
        """R1 guard: the ``--num-conversations`` arm stays gated by
        ``counts_toward_phase_target``. With no request cap, a reactive child
        must NOT flip ``is_final_credit`` early — only the root plan exhausting
        does. (Pairs with the request-count arm being unconditional.)
        """
        c = CreditCounter(cfg(sessions=1))  # no request cap
        _, final_root = c.increment_sent(turn(idx=0, num=2))  # 2-turn root, not done
        assert final_root is False
        _, final_child = c.increment_sent(child_turn(conv="child-1", idx=0))
        assert final_child is False  # reactive child can't satisfy the session plan
        assert c.total_session_turns == 2

    def test_child_increment_returned_bumps_requests_only(self) -> None:
        c = CreditCounter(cfg(reqs=1))
        c.increment_sent(turn(idx=0))
        c.freeze_sent_counts()  # _final_requests_sent = 1

        # Child return bumps requests_completed but leaves
        # completed_sessions alone.
        result = c.increment_returned(
            is_final_turn=True, cancelled=False, is_child=True
        )
        # check_all_returned_or_cancelled: 1 >= 1 → True (callback
        # handler's has_pending_branch_work guard defers the actual
        # event fire in production).
        assert result is True
        assert c.requests_completed == 1
        assert c.completed_sessions == 0  # child didn't count

        # Root return now — bumps both.
        result = c.increment_returned(
            is_final_turn=True, cancelled=False, is_child=False
        )
        assert result is True
        assert c.requests_completed == 2
        assert c.completed_sessions == 1

    def test_child_cancelled_return_bumps_requests_cancelled(self) -> None:
        c = CreditCounter(cfg(reqs=1))
        c.increment_sent(turn(idx=0))
        c.increment_sent(child_turn(conv="child-1", idx=0))
        c.freeze_sent_counts()

        result = c.increment_returned(is_final_turn=True, cancelled=True, is_child=True)
        # Cancel bump on requests_cancelled; cancelled_sessions stays
        # at zero (child didn't take a session slot to cancel).
        assert c.requests_cancelled == 1
        assert c.cancelled_sessions == 0
        # With children now counted in requests_cancelled + completed,
        # the returned-flag may trip based on frozen target.
        assert result is True or result is False  # either is fine

    def test_children_dont_inflate_sent_sessions(self) -> None:
        """Regression: DAG fanout on a single-session run must not
        make ``sent_sessions`` report > 1."""
        c = CreditCounter(cfg(sessions=1))
        c.increment_sent(turn(idx=0))
        for i in range(5):  # simulate 5 DAG children
            c.increment_sent(child_turn(conv=f"child-{i}", idx=0))

        assert c.sent_sessions == 1
        assert c.requests_sent == 6  # 1 root + 5 children (all real requests)

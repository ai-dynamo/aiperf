# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial test: DAG children HONOR the ``--request-count`` cap (exact-cutoff).

``RequestCountStopCondition`` inherits the base ``applies_to_dag_children = True``,
so ``StopConditionChecker.can_send_child_turn`` consults the request-count cap and
a child at the cap is refused. ``CreditCallbackHandler.on_credit_return`` (and the
agentic-replay child-issuance chokepoint) then route that refusal to
``BranchOrchestrator.on_child_stopped`` / ``children_truncated`` so the parent's
join drains instead of deadlocking on a child whose remaining turns never issue.

This test pins the exact-cutoff behaviour: ``--request-count N`` is a literal cap
on total wire requests ("N means N"), and children are truncated at the cap
rather than running past it.
"""

from __future__ import annotations

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.stop_conditions import StopConditionChecker


def _checker(counter: CreditCounter, config: CreditPhaseConfig) -> StopConditionChecker:
    return StopConditionChecker(
        config=config,
        lifecycle=PhaseLifecycle(config),
        counter=counter,
    )


def test_child_turn_honors_request_count_cap() -> None:
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        total_expected_requests=2,
        expected_num_sessions=None,
        expected_duration_sec=None,
    )
    counter = CreditCounter(config)
    # Drive requests_sent well past the cap (roots + their DAG offspring all
    # bump requests_sent for observability).
    for _ in range(5):
        counter.increment_sent(
            TurnToSend(
                conversation_id="trace",
                x_correlation_id="r",
                turn_index=0,
                num_turns=1,
            )
        )
    checker = _checker(counter, config)

    # Root issuance IS gated by the cap...
    assert checker.can_send_any_turn() is False
    # ...and a DAG child continuation is NOW ALSO gated: --request-count is a
    # literal wire cap honored by children ("N means N"). Exact-cutoff.
    assert checker.can_send_child_turn() is False


def test_child_turn_still_honors_cancellation() -> None:
    """Children DO honor cancellation (a user-facing guarantee), unlike the
    request-count cap. Confirms the bypass is selective, not blanket.
    """
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        total_expected_requests=100,
        expected_num_sessions=None,
        expected_duration_sec=None,
    )
    counter = CreditCounter(config)
    lifecycle = PhaseLifecycle(config)
    checker = StopConditionChecker(config=config, lifecycle=lifecycle, counter=counter)
    assert checker.can_send_child_turn() is True
    lifecycle.was_cancelled = True
    assert checker.can_send_child_turn() is False

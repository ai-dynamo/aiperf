# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial test: DAG children HONOR the ``--request-count`` cap (v2)."""

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
    # ...and in v2 a DAG child continuation is gated too -- --request-count is
    # a literal wire cap that children honor (RequestCountStopCondition
    # applies_to_dag_children is True). Only SessionCountStopCondition opts out.
    assert checker.can_send_child_turn() is False


def test_child_turn_still_honors_cancellation() -> None:
    """Children honor cancellation (a user-facing guarantee) even though they bypass the request-count cap, confirming the bypass is selective."""
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

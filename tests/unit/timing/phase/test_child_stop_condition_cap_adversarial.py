# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial test: DAG children ignore the ``--request-count`` cap.

``RequestCountStopCondition`` sets ``applies_to_dag_children = False``
(stop_conditions.py:143), so ``StopConditionChecker.can_send_child_turn`` never
consults the request-count cap. That is the *intended* behaviour per the
stop-condition docstrings -- but ``CreditCallbackHandler.on_credit_return``
documents the opposite: "the global ``--request-count`` cap still applies"
(callback_handler.py:325-328) and routes a cap-blocked child continuation to
``on_child_stopped`` / ``children_truncated``. Because the cap can never block
a child, that ``on_child_stopped`` path is unreachable for the cap and
``children_truncated`` will never reflect it.

This test pins the ACTUAL behaviour (children run past the cap) so the stale
callback-handler comment is caught.
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


def test_child_turn_bypasses_request_count_cap() -> None:
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
    # ...but a DAG child continuation is NOT -- the cap is bypassed.
    assert checker.can_send_child_turn() is True


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

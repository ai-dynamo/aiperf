# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for ``CreditIssuer`` DAG-child target membership.

``counts_toward_phase_target`` decides whether a credit can flip
``is_final_credit`` in ``CreditCounter.increment_sent`` (and therefore freeze
sent counts + fire ``all_credits_sent_event``). Reactive DAG work is spawned
*after* the root plan is sampled, so it must NOT count toward the phase target.

``CreditIssuer.dispatch_child_turn`` correctly strips the flag
(issuer.py:313-314). But ``dispatch_join_turn`` builds its ``TurnToSend`` with
no ``counts_toward_phase_target`` argument (issuer.py:344-355) -> it defaults
to True -> and routes through ``try_issue_credit``, which never strips it. For
a *nested* parent (``parent_agent_depth > 0``) the join turn is itself reactive
DAG work, yet it reaches the counter as a target-eligible credit.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend
from aiperf.timing.branch_orchestrator import PendingBranchJoin


@pytest.fixture
def captured_router() -> MagicMock:
    mock = MagicMock()
    mock.send_credit = AsyncMock()
    return mock


@pytest.fixture
def issuer(captured_router: MagicMock) -> CreditIssuer:
    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)
    stop_checker.can_start_new_session = MagicMock(return_value=True)
    stop_checker.can_send_child_turn = MagicMock(return_value=True)

    progress = MagicMock()
    progress.increment_sent = MagicMock(return_value=(0, False))
    progress.freeze_sent_counts = MagicMock()
    progress.all_credits_sent_event = asyncio.Event()

    concurrency = MagicMock()
    concurrency.acquire_session_slot = AsyncMock(return_value=True)
    concurrency.acquire_prefill_slot = AsyncMock(return_value=True)
    concurrency.try_acquire_session_slot = MagicMock(return_value=True)
    concurrency.try_acquire_prefill_slot = MagicMock(return_value=True)
    concurrency.release_session_slot = MagicMock()
    concurrency.release_prefill_slot = MagicMock()

    cancellation = MagicMock()
    cancellation.next_cancellation_delay_ns = MagicMock(return_value=None)

    lifecycle = MagicMock()
    lifecycle.started_at_ns = time.time_ns()
    lifecycle.started_at_perf_ns = time.perf_counter_ns()

    return CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=stop_checker,
        progress=progress,
        concurrency_manager=concurrency,
        credit_router=captured_router,
        cancellation_policy=cancellation,
        lifecycle=lifecycle,
    )


def _nested_pending_join() -> PendingBranchJoin:
    """A join for a *nested* parent: the parent is itself a DAG child."""
    return PendingBranchJoin(
        parent_x_correlation_id="child-parent",
        parent_conversation_id="trace::sa:agent_0",
        parent_num_turns=6,
        parent_agent_depth=1,  # <-- nested: this parent is a DAG child
        parent_parent_correlation_id="root",
        gated_turn_index=3,
    )


def test_dispatch_child_turn_strips_target_membership(issuer: CreditIssuer) -> None:
    """Baseline: dispatch_child_turn forces counts_toward_phase_target=False."""
    turn = TurnToSend(
        conversation_id="trace::sa:agent_0",
        x_correlation_id="child-1",
        turn_index=1,
        num_turns=4,
        agent_depth=1,
        parent_correlation_id="root",
        counts_toward_phase_target=True,  # caller passed True...
    )
    asyncio.run(issuer.dispatch_child_turn(turn))
    credit = issuer._credit_router.send_credit.call_args.kwargs["credit"]
    assert credit.counts_toward_phase_target is False  # ...issuer strips it


def test_nested_join_turn_does_not_count_toward_target(issuer: CreditIssuer) -> None:
    asyncio.run(issuer.dispatch_join_turn(_nested_pending_join()))
    credit = issuer._credit_router.send_credit.call_args.kwargs["credit"]
    assert credit.agent_depth == 1
    assert credit.counts_toward_phase_target is False

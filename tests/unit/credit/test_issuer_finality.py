# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards the credit issuer's lineage-finality stamp (three-touch touch #2).

Touch #1 (the ``Credit`` struct fields) and touch #3 (worker -> ``RequestInfo``,
``test_create_request_info_plumbs_finality_from_credit``) are already covered.
This file covers touch #2: ``CreditIssuer`` reading per-tree state from a REAL
``SessionTreeRegistry`` (``_finality_for_issue``) AND stamping the result onto
the emitted ``Credit`` at its sole construction site.

Both the registry and the emitted ``Credit`` are real objects here on purpose --
a MagicMock would auto-create ``is_parent_final`` / ``is_tree_final`` and pass
even if the ``Credit(...)`` kwargs were deleted, defeating the guard.
"""

import time
from unittest.mock import MagicMock

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.session_tree import SessionTreeRegistry


class _FakeConcurrency:
    """Slots always granted; releases are no-ops.

    The real ``SessionTreeRegistry`` only needs ``release_session_slot``; the
    issuer's acquire path needs the two coroutines. Neither the registry nor
    the emitted ``Credit`` is a mock -- that is the point of this file.
    """

    async def acquire_session_slot(self, phase: CreditPhase, can_proceed) -> bool:
        return True

    async def acquire_prefill_slot(self, phase: CreditPhase, can_proceed) -> bool:
        return True

    def release_session_slot(self, phase: CreditPhase) -> None:
        pass


class _CapturingRouter:
    """Captures the emitted ``Credit`` so tests can assert its stamped finality."""

    def __init__(self) -> None:
        self.sent: list[Credit] = []

    async def send_credit(self, *, credit: Credit) -> None:
        self.sent.append(credit)


def _make_registry() -> SessionTreeRegistry:
    return SessionTreeRegistry(_FakeConcurrency())


def _make_issuer(
    registry: SessionTreeRegistry | None,
) -> tuple[CreditIssuer, _CapturingRouter]:
    """Minimal real issuer: mocked scalars/lifecycle, REAL registry + router.

    ``session_tree_registry_enabled=True`` engages ``registry`` regardless of
    phase; passing ``None`` for the registry yields the non-agentic path.
    """
    progress = MagicMock()
    progress.increment_sent = MagicMock(return_value=(1, False))
    progress.freeze_sent_counts = MagicMock()
    progress.all_credits_sent_event = MagicMock()

    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)
    stop_checker.can_start_new_session = MagicMock(return_value=True)
    stop_checker.can_send_child_turn = MagicMock(return_value=True)

    cancellation = MagicMock()
    cancellation.next_cancellation_delay_ns = MagicMock(return_value=None)

    lifecycle = MagicMock()
    lifecycle.started_at_ns = time.time_ns()
    lifecycle.started_at_perf_ns = time.perf_counter_ns()

    router = _CapturingRouter()
    issuer = CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=stop_checker,
        progress=progress,
        concurrency_manager=_FakeConcurrency(),
        credit_router=router,
        cancellation_policy=cancellation,
        lifecycle=lifecycle,
        session_tree_registry=registry,
        session_tree_registry_enabled=True,
    )
    return issuer, router


def _root_turn(root_id: str = "root-1") -> TurnToSend:
    """Depth-0 root, single-turn (final), no forks."""
    return TurnToSend(
        conversation_id="conv-1",
        x_correlation_id=root_id,
        turn_index=0,
        num_turns=1,
    )


def _child_turn(root_id: str = "root-1", child_id: str = "child-1") -> TurnToSend:
    """Child whose parent IS the root, single-turn (final), no forks."""
    return TurnToSend(
        conversation_id="conv-1",
        x_correlation_id=child_id,
        turn_index=0,
        num_turns=1,
        agent_depth=1,
        parent_correlation_id=root_id,
        root_correlation_id=root_id,
    )


# =============================================================================
# _finality_for_issue: reads REAL SessionTreeRegistry state
# =============================================================================


def test_finality_root_final_turn_no_descendants_is_tree_final():
    """Scenario 1: root, final turn, no descendants, no forks."""
    registry = _make_registry()
    registry.open_tree("root-1", CreditPhase.PROFILING, root_pending=True)
    issuer, _ = _make_issuer(registry)

    is_parent_final, is_tree_final = issuer._finality_for_issue(_root_turn())

    assert is_parent_final is None
    assert is_tree_final is True


def test_finality_root_with_outstanding_descendant_not_tree_final():
    """Scenario 2: root, final turn, one outstanding descendant -> not last."""
    registry = _make_registry()
    registry.open_tree("root-1", CreditPhase.PROFILING, root_pending=True)
    registry.register_descendants("root-1", n=1)
    issuer, _ = _make_issuer(registry)

    is_parent_final, is_tree_final = issuer._finality_for_issue(_root_turn())

    assert is_parent_final is None
    assert is_tree_final is False


def test_finality_sole_child_after_root_terminal_is_both_final():
    """Scenario 3: child whose parent is the root; root terminal; sole
    outstanding child on its final turn -> both facts True."""
    registry = _make_registry()
    registry.open_tree("root-1", CreditPhase.PROFILING, root_pending=True)
    registry.register_descendants("root-1", n=1)
    registry.on_root_terminal("root-1")  # root_pending cleared; tree still live
    issuer, _ = _make_issuer(registry)

    is_parent_final, is_tree_final = issuer._finality_for_issue(_child_turn())

    assert is_parent_final is True
    assert is_tree_final is True


def test_finality_no_registry_is_conservative_none_false():
    """Scenario 4: no registry engaged -> conservative ``(None, False)``."""
    issuer, _ = _make_issuer(None)

    assert issuer._finality_for_issue(_root_turn()) == (None, False)


# =============================================================================
# GUARD: the Credit(...) construction site must pass the helper's results through
# =============================================================================


async def test_issue_credit_stamps_finality_onto_emitted_credit():
    """RED if either ``is_parent_final=`` / ``is_tree_final=`` kwarg is removed
    from the ``Credit(...)`` construction in ``_issue_credit_internal``.

    Uses scenario 3 (both facts True) so the stamped values differ from the
    struct defaults (``None`` / ``False``): a dropped kwarg reverts the emitted
    ``Credit`` to the default and this assertion fails.
    """
    registry = _make_registry()
    registry.open_tree("root-1", CreditPhase.PROFILING, root_pending=True)
    registry.register_descendants("root-1", n=1)
    registry.on_root_terminal("root-1")
    issuer, router = _make_issuer(registry)

    await issuer.issue_credit(_child_turn())

    assert len(router.sent) == 1
    credit = router.sent[0]
    assert credit.is_parent_final is True
    assert credit.is_tree_final is True

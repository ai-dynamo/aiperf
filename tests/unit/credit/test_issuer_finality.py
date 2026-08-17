# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards the credit issuer's lineage-finality stamp."""

from collections.abc import Callable

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend
from aiperf.timing.session_tree import SessionTreeRegistry
from tests.unit.timing.conftest import (
    CapturingRouter,
    make_registry,
    make_tree_issuer,
)


def _make_issuer(
    registry: SessionTreeRegistry | None,
) -> tuple[CreditIssuer, CapturingRouter]:
    """Minimal real issuer: mocked scalars/lifecycle, REAL registry + router."""
    return make_tree_issuer(registry, registry_enabled=True)


def _root_turn(root_id: str = "root-1") -> TurnToSend:
    """Depth-0 root, single-turn (final), no forks."""
    return TurnToSend(
        conversation_id="conv-1",
        x_correlation_id=root_id,
        turn_index=0,
        num_turns=1,
    )


def _spawning_root_turn() -> TurnToSend:
    """Depth-0 root final turn that declares a SPAWN-shaped branch (no forks)."""
    return TurnToSend(
        conversation_id="conv-1",
        x_correlation_id="root-1",
        turn_index=0,
        num_turns=1,
        has_forks=False,
        has_branches=True,
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


def _open_root() -> SessionTreeRegistry:
    """Registry with one open, root-pending tree and no descendants."""
    registry = make_registry()
    registry.open_tree("root-1", CreditPhase.PROFILING, root_pending=True)
    return registry


def _open_root_with_descendant() -> SessionTreeRegistry:
    """Open root-pending tree with one outstanding descendant."""
    registry = _open_root()
    registry.register_descendants("root-1", n=1)
    return registry


def _root_terminal_with_descendant() -> SessionTreeRegistry:
    """Root already terminal with its sole descendant still outstanding."""
    registry = _open_root_with_descendant()
    registry.on_root_terminal("root-1")  # root_pending cleared; tree still live
    return registry


# =============================================================================
# _finality_for_issue: reads REAL SessionTreeRegistry state
# =============================================================================


@pytest.mark.parametrize(
    "build_registry,build_turn,expected",
    [
        param(
            _open_root, _root_turn, (None, True),
            id="root_final_turn_no_descendants_is_tree_final",
        ),
        param(
            _open_root_with_descendant, _root_turn, (None, False),
            id="outstanding_descendant_blocks_tree_final",
        ),
        param(
            _root_terminal_with_descendant, _child_turn, (True, True),
            id="sole_child_after_root_terminal_is_both_final",
        ),
        param(
            lambda: None, _root_turn, (None, False),
            id="no_registry_is_conservative",
        ),
        param(
            _open_root, _spawning_root_turn, (None, False),
            id="spawning_final_turn_never_tree_final",
        ),
    ],
)  # fmt: skip
def test_finality_for_issue_reads_registry_state(
    build_registry: Callable[[], SessionTreeRegistry | None],
    build_turn: Callable[[], TurnToSend],
    expected: tuple[bool | None, bool],
) -> None:
    """``_finality_for_issue`` derives ``(is_parent_final, is_tree_final)`` from the tree."""
    # The spawning-root case shares scenario-1 registry state (which yields
    # True) and differs only by declaring a branch, so it pins that
    # has_branches -- not just has_forks -- gates the stamp.
    issuer, _ = _make_issuer(build_registry())
    assert issuer._finality_for_issue(build_turn()) == expected


# =============================================================================
# GUARD: the Credit(...) construction site must pass the helper's results through
# =============================================================================


@pytest.mark.asyncio
async def test_issue_credit_stamps_finality_onto_emitted_credit() -> None:
    """The emitted Credit carries the finality the helper computed, not defaults."""
    # RED if either is_parent_final= / is_tree_final= kwarg is dropped from the
    # Credit(...) construction in _issue_credit_internal.
    issuer, router = _make_issuer(_root_terminal_with_descendant())

    # Child inherits the root's slot; dispatch_child_turn is the wire path that
    # runs _issue_credit_internal for a DAG child (issue_credit's session-slot
    # path is for roots).
    await issuer.dispatch_child_turn(_child_turn())

    assert len(router.sent) == 1
    credit = router.sent[0]
    assert credit.is_parent_final is True
    assert credit.is_tree_final is True

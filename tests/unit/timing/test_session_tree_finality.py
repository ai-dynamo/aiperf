# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finality-query unit tests for the trimmed main ``SessionTreeRegistry``."""

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models.record_models import RequestInfo
from aiperf.credit.structs import Credit
from aiperf.timing.session_tree import SessionTreeRegistry
from tests.unit.timing.conftest import make_registry, registry_with_tree


def test_root_terminal_unknown_tree_is_none() -> None:
    """A tree the registry never opened reports root terminality as unknown."""
    assert make_registry().root_terminal("nope") is None


def test_root_terminal_flips_false_to_true_on_root_terminal() -> None:
    """``on_root_terminal`` is what promotes a pending root to terminal."""
    # A live descendant keeps the tree open past on_root_terminal so the
    # post-terminal state is observable; a drained tree would be retired
    # (popped) and root_terminal would then read as unknown/None.
    registry = registry_with_tree(descendants=1)
    assert registry.root_terminal("root-1") is False
    registry.on_root_terminal("root-1")
    assert registry.root_terminal("root-1") is True


@pytest.mark.parametrize(
    "build_registry,root,is_final_turn,is_root_credit,has_branches,expected",
    [
        param(
            lambda: registry_with_tree(),
            "root-1", True, True, False, True,
            id="root_final_turn_no_descendants_is_last",
        ),
        param(
            lambda: registry_with_tree(descendants=1),
            "root-1", True, True, False, False,
            id="descendant_outstanding_blocks_last",
        ),
        param(
            lambda: registry_with_tree(root="root-2"),
            "root-2", True, True, True, False,
            id="pending_branches_block_last",
        ),
        param(
            lambda: registry_with_tree(root="root-2"),
            "root-2", False, True, False, False,
            id="non_final_turn_is_not_last",
        ),
        param(
            lambda: _root_terminal_with_one_child(),
            "root-1", True, False, False, True,
            id="final_child_after_root_done_is_last",
        ),
        param(
            make_registry,
            "nope", True, True, False, False,
            id="unknown_tree_is_conservative_false",
        ),
    ],
)  # fmt: skip
def test_is_last_tree_request_scenarios(
    build_registry,
    root: str,
    is_final_turn: bool,
    is_root_credit: bool,
    has_branches: bool,
    expected: bool,
) -> None:
    """``is_last_tree_request`` is true only for the genuinely final credit in a tree."""
    registry = build_registry()
    assert (
        bool(
            registry.is_last_tree_request(
                root,
                is_final_turn=is_final_turn,
                is_root_credit=is_root_credit,
                has_branches=has_branches,
            )
        )
        is expected
    )


def _root_terminal_with_one_child() -> SessionTreeRegistry:
    """Tree whose root has gone terminal with a single live descendant left."""
    registry = registry_with_tree(descendants=1)
    registry.on_root_terminal("root-1")
    return registry


def test_descendant_done_retires_drained_tree() -> None:
    """Draining the last descendant of a terminal root retires the tree outright."""
    registry = _root_terminal_with_one_child()
    assert registry.has_tree("root-1")
    assert registry.on_descendant_done("root-1") is True
    assert not registry.has_tree("root-1")
    assert registry.root_terminal("root-1") is None


def test_release_all_retires_open_trees_without_slot_release() -> None:
    """``release_all`` retires every open tree and returns how many it closed."""
    registry = make_registry()
    registry.open_tree("a", CreditPhase.PROFILING, root_pending=True)
    registry.open_tree("b", CreditPhase.PROFILING, root_pending=True)
    assert registry.open_count() == 2
    assert registry.release_all() == 2
    assert registry.open_count() == 0


def test_final_turn_spawn_resurrects_retired_tree_for_grandchild_finality() -> None:
    """Regression: a retired tree resurrects when a child's final turn spawns a grandchild."""
    registry = registry_with_tree(descendants=1)  # root + one live child C
    registry.on_root_terminal("root-1")  # root's terminal turn returns; C still live

    # C is the last outstanding descendant; its final-turn return decrements it,
    # draining and retiring the tree (step 4b, before C's spawn intercept).
    assert registry.on_descendant_done("root-1") is True
    assert not registry.has_tree("root-1")

    # Step 5: C's final-turn SPAWN registers the grandchild AFTER that retire.
    # Old behavior buffered it into a retired root nothing drains; now it
    # resurrects the tree root-terminal with the grandchild outstanding.
    registry.register_descendants("root-1", n=1)
    assert registry.has_tree("root-1")
    assert registry.root_terminal("root-1") is True

    # The grandchild's genuinely-last (non-branching final) credit CAN now be
    # stamped tree-final -- root terminal, sole outstanding descendant.
    assert registry.is_last_tree_request(
        "root-1", is_final_turn=True, is_root_credit=False, has_branches=False
    )

    # The grandchild finishing re-drains and re-retires the tree coherently.
    assert registry.on_descendant_done("root-1") is True
    assert not registry.has_tree("root-1")
    assert registry.late_events == 0


def test_release_all_drains_pending_descendants() -> None:
    """``release_all`` also clears the pre-open descendant buffer, not just open trees."""
    registry = make_registry()
    # No open_tree first: register_descendants buffers into _pending_descendants
    # (the defensive pre-open path).
    registry.register_descendants("orphan-root", n=2)
    assert registry._pending_descendants  # buffered, not yet folded into a tree

    registry.release_all()

    assert registry._pending_descendants == {}
    assert registry._retired_roots == {}


def test_lineage_finality_fields_exist_on_both_credit_and_request_info() -> None:
    """Schema guard: the worker has matching lineage-finality field names to copy between."""
    # Name presence only; value plumbing is covered by
    # test_create_request_info_plumbs_finality_from_credit.
    credit_fields = {"is_parent_final", "is_tree_final", "root_correlation_id"}
    assert credit_fields <= set(Credit.__struct_fields__)
    assert credit_fields <= set(RequestInfo.model_fields)


def test_retired_roots_window_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retired-root ledger evicts oldest-first instead of growing per session."""
    # Unbounded, this held one entry per retired tree for the entire PROFILING
    # phase -- hundreds of MB on a long durability ramp.
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.AGENTX, "RECYCLE_GUARD_MAX_WINDOW", 3)
    registry = make_registry()
    for i in range(10):
        root = f"root-{i}"
        registry.open_tree(root, phase=CreditPhase.PROFILING, root_pending=True)
        registry.register_descendants(root, n=1)
        registry.on_root_terminal(root)
        registry.on_descendant_done(root)
    assert list(registry._retired_roots) == ["root-7", "root-8", "root-9"]

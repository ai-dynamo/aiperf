# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finality-query unit tests for the trimmed main ``SessionTreeRegistry``.

The main registry is finality bookkeeping only: it holds no concurrency
manager and releases no slot (unlike the agentic-replay variant it was ported
from). These tests exercise the state map + the two finality queries directly.
"""

from aiperf.common.enums import CreditPhase
from aiperf.timing.session_tree import SessionTreeRegistry


def _make_registry() -> SessionTreeRegistry:
    return SessionTreeRegistry()


def _registry_with_tree(
    root: str = "root-1", descendants: int = 0
) -> SessionTreeRegistry:
    registry = _make_registry()
    registry.open_tree(root, phase=CreditPhase.PROFILING, root_pending=True)
    if descendants:
        registry.register_descendants(root, n=descendants)
    return registry


def test_root_terminal_unknown_tree_is_none():
    assert _make_registry().root_terminal("nope") is None


def test_root_terminal_false_while_pending_true_after():
    # A live descendant keeps the tree open past on_root_terminal so the
    # post-terminal state is observable; a drained tree would be retired
    # (popped) and root_terminal would then read as unknown/None.
    registry = _registry_with_tree(descendants=1)
    assert registry.root_terminal("root-1") is False
    registry.on_root_terminal("root-1")
    assert registry.root_terminal("root-1") is True


def test_last_tree_request_root_with_no_descendants():
    registry = _registry_with_tree()
    assert registry.is_last_tree_request(
        "root-1", is_final_turn=True, is_root_credit=True, has_branches=False
    )


def test_not_last_when_descendants_outstanding_or_branches_pending():
    registry = _registry_with_tree(descendants=1)
    assert not registry.is_last_tree_request(
        "root-1", is_final_turn=True, is_root_credit=True, has_branches=False
    )
    registry_no_desc = _registry_with_tree(root="root-2")
    assert not registry_no_desc.is_last_tree_request(
        "root-2", is_final_turn=True, is_root_credit=True, has_branches=True
    )
    assert not registry_no_desc.is_last_tree_request(
        "root-2", is_final_turn=False, is_root_credit=True, has_branches=False
    )


def test_last_tree_request_final_child_after_root_done():
    registry = _registry_with_tree(descendants=1)
    registry.on_root_terminal("root-1")
    assert registry.is_last_tree_request(
        "root-1", is_final_turn=True, is_root_credit=False, has_branches=False
    )


def test_unknown_tree_is_conservative_false():
    assert not _make_registry().is_last_tree_request(
        "nope", is_final_turn=True, is_root_credit=True, has_branches=False
    )


def test_descendant_done_retires_drained_tree():
    # Sole child completes after root terminal -> tree drains and is retired.
    registry = _registry_with_tree(descendants=1)
    registry.on_root_terminal("root-1")
    assert registry.has_tree("root-1")
    retired = registry.on_descendant_done("root-1")
    assert retired is True
    assert not registry.has_tree("root-1")
    assert registry.root_terminal("root-1") is None


def test_release_all_retires_open_trees_without_slot_release():
    registry = _make_registry()
    registry.open_tree("a", CreditPhase.PROFILING, root_pending=True)
    registry.open_tree("b", CreditPhase.PROFILING, root_pending=True)
    assert registry.open_count() == 2
    assert registry.release_all() == 2
    assert registry.open_count() == 0


def test_final_turn_spawn_resurrects_retired_tree_for_grandchild_finality():
    """Regression: root done -> last-outstanding child C's final turn declares a
    SPAWN grandchild. The callback order retires the tree on C's own
    on_descendant_done (step 4b) BEFORE the return-intercept registers C's
    grandchildren (step 5); register_descendants must RESURRECT the retired tree
    so the grandchild's genuinely-last credit can still stamp is_tree_final=True.
    """
    registry = _registry_with_tree(descendants=1)  # root + one live child C
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


def test_release_all_drains_pending_descendants():
    """release_all must clear the pre-open descendant buffer, not just _trees."""
    registry = _make_registry()
    # No open_tree first: register_descendants buffers into _pending_descendants
    # (the defensive pre-open path).
    registry.register_descendants("orphan-root", n=2)
    assert registry._pending_descendants  # buffered, not yet folded into a tree

    registry.release_all()

    assert registry._pending_descendants == {}
    assert registry._retired_roots == {}


def test_finality_flows_credit_to_request_info():
    """Schema guard: the three lineage-finality fields exist on BOTH the Credit
    struct and the RequestInfo model, so the worker has fields to copy between.

    This asserts field-NAME presence only -- it does NOT verify a value is
    actually copied (deleting the plumb kwargs in ``worker._create_request_info``
    keeps this green). The value-level plumb guard is
    ``test_worker.py::test_create_request_info_plumbs_finality_from_credit``,
    which stamps a REAL Credit and asserts the values surface on the RequestInfo.
    """
    from aiperf.common.models.record_models import RequestInfo
    from aiperf.credit.structs import Credit

    credit_fields = {"is_parent_final", "is_tree_final", "root_correlation_id"}
    assert credit_fields <= set(Credit.__struct_fields__)
    assert credit_fields <= set(RequestInfo.model_fields)

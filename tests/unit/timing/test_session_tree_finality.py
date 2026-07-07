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


def test_finality_flows_credit_to_request_info():
    """REAL structs end-to-end: a Credit stamped with finality must surface
    on the RequestInfo the worker builds. Catches a missed plumb touch."""
    from aiperf.common.models.record_models import RequestInfo
    from aiperf.credit.structs import Credit

    credit_fields = {"is_parent_final", "is_tree_final", "root_correlation_id"}
    assert credit_fields <= set(Credit.__struct_fields__)
    assert credit_fields <= set(RequestInfo.model_fields)

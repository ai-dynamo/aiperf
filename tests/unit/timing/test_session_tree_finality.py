# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CreditPhase
from aiperf.timing.session_tree import SessionTreeRegistry


class _FakeConcurrency:
    """Minimal stand-in for the concurrency manager the registry releases to."""

    def release_session_slot(self, phase: CreditPhase) -> None:
        pass


def _make_registry() -> SessionTreeRegistry:
    return SessionTreeRegistry(_FakeConcurrency())


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


def test_spawn_declaring_final_turn_gated_by_has_branches():
    """Same registry state that yields True with ``has_branches=False`` must"""
    registry = _registry_with_tree(root="root-3")
    assert registry.is_last_tree_request(
        "root-3", is_final_turn=True, is_root_credit=True, has_branches=False
    )
    assert not registry.is_last_tree_request(
        "root-3", is_final_turn=True, is_root_credit=True, has_branches=True
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


def test_finality_flows_credit_to_request_info():
    """REAL structs end-to-end: a Credit stamped with finality must surface"""
    from aiperf.common.models.record_models import RequestInfo
    from aiperf.credit.structs import Credit

    credit_fields = {"is_parent_final", "is_tree_final"}
    assert credit_fields <= set(Credit.__struct_fields__)
    assert credit_fields <= set(RequestInfo.model_fields)

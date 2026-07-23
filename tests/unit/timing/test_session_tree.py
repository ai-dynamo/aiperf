# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SessionTreeRegistry: per-session-tree session-slot accounting."""

from __future__ import annotations

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.timing.session_tree import SessionTreeRegistry


class FakeConcurrencyManager:
    """Records release_session_slot calls per phase (acquire stays elsewhere)."""

    def __init__(self) -> None:
        self.released: list[CreditPhase] = []

    def release_session_slot(self, phase: CreditPhase) -> None:
        self.released.append(phase)


@pytest.fixture
def cm() -> FakeConcurrencyManager:
    return FakeConcurrencyManager()


@pytest.fixture
def registry(cm: FakeConcurrencyManager) -> SessionTreeRegistry:
    return SessionTreeRegistry(cm)


PROFILING = CreditPhase.PROFILING
WARMUP = CreditPhase.WARMUP


def test_root_only_tree_releases_on_root_terminal(cm, registry):
    """A root with no descendants drains and releases the slot at root terminal."""
    registry.open_tree("root-a", PROFILING, root_pending=True)
    assert registry.open_count() == 1
    released = registry.on_root_terminal("root-a")
    assert released is True
    assert cm.released == [PROFILING]
    assert registry.open_count() == 0


def test_descendant_holds_slot_until_it_drains(cm, registry):
    """The slot is held while a descendant is in flight, then released when it"""
    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.register_descendants("root-a", 1)

    released_now = registry.on_root_terminal("root-a")
    assert released_now is False
    assert cm.released == []
    assert registry.open_count() == 1

    registry.on_descendant_done("root-a")
    assert cm.released == [PROFILING]
    assert registry.open_count() == 0


def test_descendant_completing_before_root_does_not_release(cm, registry):
    """If every descendant finishes before the root's terminal turn, the slot is"""
    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.register_descendants("root-a", 2)

    registry.on_descendant_done("root-a")
    registry.on_descendant_done("root-a")
    assert cm.released == []
    assert registry.open_count() == 1

    registry.on_root_terminal("root-a")
    assert cm.released == [PROFILING]


def test_rootless_tree_releases_when_last_descendant_drains(cm, registry):
    """A rootless lane (no root credit ever) drains purely on descendant"""
    registry.open_tree("lane-root", PROFILING, root_pending=False)
    registry.register_descendants("lane-root", 3)

    registry.on_descendant_done("lane-root")
    registry.on_descendant_done("lane-root")
    assert cm.released == []
    registry.on_descendant_done("lane-root")
    assert cm.released == [PROFILING]
    assert registry.open_count() == 0


def test_descendants_registered_before_open_are_counted(cm, registry):
    """Snapshot regression: a lane's subagents register BEFORE the lane slot is"""
    registry.register_descendants("lane-root", 3)
    assert registry.open_count() == 0
    registry.open_tree("lane-root", PROFILING, root_pending=False)

    registry.on_descendant_done("lane-root")
    registry.on_descendant_done("lane-root")
    assert cm.released == []
    registry.on_descendant_done("lane-root")
    assert cm.released == [PROFILING]


def test_descendant_completing_before_open_does_not_leak_slot(cm, registry):
    """H3 regression: a subagent at offset 0 can complete BEFORE the lane's root"""
    registry.register_descendants("lane-root", 2)
    registry.on_descendant_done("lane-root")
    registry.open_tree("lane-root", PROFILING, root_pending=False)

    assert cm.released == []
    registry.on_descendant_done("lane-root")
    assert cm.released == [PROFILING]


def test_buffered_descendants_combine_with_post_open_registrations(cm, registry):
    """Pre-open (buffered) and post-open (live-spawned) descendants both count."""
    registry.register_descendants("root-a", 2)
    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.register_descendants("root-a", 1)
    registry.on_root_terminal("root-a")
    for _ in range(3):
        assert cm.released == []
        registry.on_descendant_done("root-a")
    assert cm.released == [PROFILING]


def test_release_is_exactly_once(cm, registry):
    """Double root-terminal / extra descendant-done never double-release."""
    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.on_root_terminal("root-a")
    assert registry.on_root_terminal("root-a") is False
    registry.on_descendant_done("root-a")
    registry.register_descendants("root-a", 5)
    assert cm.released == [PROFILING]


def test_unknown_tree_is_tolerated(cm, registry):
    """Events for never-opened trees are no-ops (engagement-gate / pre-session)."""
    assert registry.on_root_terminal("ghost") is False
    registry.on_descendant_done("ghost")
    registry.register_descendants("ghost", 2)
    assert cm.released == []
    assert registry.open_count() == 0


def test_descendant_done_clamps_at_zero(cm, registry):
    """Spurious extra descendant-done cannot drive outstanding negative."""
    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.register_descendants("root-a", 1)
    registry.on_descendant_done("root-a")
    registry.on_descendant_done("root-a")
    assert cm.released == []
    registry.on_root_terminal("root-a")
    assert cm.released == [PROFILING]


def test_drain_callback_fires_on_release(cm, registry):
    """The drain callback fires exactly once per tree, on release, with the"""
    drained: list[tuple[str, CreditPhase]] = []
    registry.set_drain_callback(lambda root, phase: drained.append((root, phase)))

    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.register_descendants("root-a", 1)
    registry.on_root_terminal("root-a")
    assert drained == []
    registry.on_descendant_done("root-a")
    assert drained == [("root-a", PROFILING)]


def test_release_all_releases_open_trees_for_phase_without_drain_callback(cm, registry):
    """Teardown releases every still-open tree's slot for the phase and does NOT"""
    drained: list[str] = []
    registry.set_drain_callback(lambda root, phase: drained.append(root))

    registry.open_tree("r1", PROFILING, root_pending=True)
    registry.open_tree("r2", PROFILING, root_pending=True)
    registry.register_descendants("r2", 2)
    registry.open_tree("w1", WARMUP, root_pending=True)

    released = registry.release_all(PROFILING)
    assert released == 2
    assert cm.released == [PROFILING, PROFILING]
    assert drained == []
    assert registry.open_count(PROFILING) == 0
    assert registry.open_count(WARMUP) == 1


def test_open_count_by_phase(registry):
    registry.open_tree("r1", PROFILING, root_pending=True)
    registry.open_tree("w1", WARMUP, root_pending=True)
    assert registry.open_count() == 2
    assert registry.open_count(PROFILING) == 1
    assert registry.open_count(WARMUP) == 1


def test_release_uses_the_trees_own_phase(cm, registry):
    """A tree releases against the phase it was opened with, even if drained"""
    registry.open_tree("w-root", WARMUP, root_pending=False)
    registry.register_descendants("w-root", 1)
    registry.on_descendant_done("w-root")
    assert cm.released == [WARMUP]

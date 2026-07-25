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
    """The slot is held while a descendant is in flight, then released when it completes after the root has gone terminal."""
    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.register_descendants("root-a", 1)

    # Root finishes first; the subagent is still running -> slot must be held.
    released_now = registry.on_root_terminal("root-a")
    assert released_now is False
    assert cm.released == []
    assert registry.open_count() == 1

    # Subagent finishes -> tree drains -> slot released exactly once.
    registry.on_descendant_done("root-a")
    assert cm.released == [PROFILING]
    assert registry.open_count() == 0


def test_descendant_completing_before_root_does_not_release(cm, registry):
    """If every descendant finishes before the root's terminal turn, the slot is held until the root itself goes terminal."""
    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.register_descendants("root-a", 2)

    registry.on_descendant_done("root-a")
    registry.on_descendant_done("root-a")
    assert cm.released == []  # root still pending
    assert registry.open_count() == 1

    registry.on_root_terminal("root-a")
    assert cm.released == [PROFILING]


def test_rootless_tree_releases_when_last_descendant_drains(cm, registry):
    """A rootless lane (no root credit ever) drains purely on descendant completion, with root_pending=False from the start."""
    registry.open_tree("lane-root", PROFILING, root_pending=False)
    registry.register_descendants("lane-root", 3)

    registry.on_descendant_done("lane-root")
    registry.on_descendant_done("lane-root")
    assert cm.released == []
    registry.on_descendant_done("lane-root")
    assert cm.released == [PROFILING]
    assert registry.open_count() == 0


def test_descendants_registered_before_open_are_counted(cm, registry):
    """Snapshot regression: subagents registered before the lane slot is acquired must be buffered and folded into the tree at open_tree, else the tree opens at outstanding=0 and drains prematurely on the first child completion."""
    registry.register_descendants("lane-root", 3)  # tree not open yet
    assert registry.open_count() == 0
    registry.open_tree("lane-root", PROFILING, root_pending=False)  # rootless lane

    # The tree opened aware of all 3 subagents: draining the first two does NOT
    # release; only the third (the true last) drains the tree.
    registry.on_descendant_done("lane-root")
    registry.on_descendant_done("lane-root")
    assert cm.released == []
    registry.on_descendant_done("lane-root")
    assert cm.released == [PROFILING]


def test_descendant_completing_before_open_does_not_leak_slot(cm, registry):
    """H3 regression: a subagent completing before the lane's root opens the tree must decrement the pending buffer, else open_tree folds in a stale full count and the tree never drains, leaking the session slot until teardown."""
    registry.register_descendants("lane-root", 2)  # 2 subagents seeded pre-open
    registry.on_descendant_done("lane-root")  # one completes BEFORE open_tree
    registry.open_tree("lane-root", PROFILING, root_pending=False)  # rootless lane

    # Only the 1 remaining subagent is outstanding; its completion must drain.
    assert cm.released == []
    registry.on_descendant_done("lane-root")
    assert cm.released == [PROFILING]


def test_buffered_descendants_combine_with_post_open_registrations(cm, registry):
    """Pre-open (buffered) and post-open (live-spawned) descendants both count."""
    registry.register_descendants("root-a", 2)  # buffered (pre-open)
    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.register_descendants("root-a", 1)  # live spawn (post-open)
    registry.on_root_terminal("root-a")
    for _ in range(3):
        assert cm.released == []  # 3 descendants outstanding
        registry.on_descendant_done("root-a")
    assert cm.released == [PROFILING]


def test_release_is_exactly_once(cm, registry):
    """Double root-terminal / extra descendant-done never double-release."""
    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.on_root_terminal("root-a")
    # Tree already gone; further events are no-ops.
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
    registry.on_descendant_done("root-a")  # extra
    assert cm.released == []  # still root_pending, outstanding clamped at 0
    registry.on_root_terminal("root-a")
    assert cm.released == [PROFILING]


def test_drain_callback_fires_on_release(cm, registry):
    """The drain callback fires exactly once per tree, on release, with the root id and phase, so the strategy can recycle the freed lane."""
    drained: list[tuple[str, CreditPhase]] = []
    registry.set_drain_callback(lambda root, phase: drained.append((root, phase)))

    registry.open_tree("root-a", PROFILING, root_pending=True)
    registry.register_descendants("root-a", 1)
    registry.on_root_terminal("root-a")
    assert drained == []  # held
    registry.on_descendant_done("root-a")
    assert drained == [("root-a", PROFILING)]


def test_release_all_releases_open_trees_for_phase_without_drain_callback(cm, registry):
    """Teardown releases every still-open tree's slot for the phase and does NOT fire the drain callback (no recycle at teardown)."""
    drained: list[str] = []
    registry.set_drain_callback(lambda root, phase: drained.append(root))

    registry.open_tree("r1", PROFILING, root_pending=True)
    registry.open_tree("r2", PROFILING, root_pending=True)
    registry.register_descendants("r2", 2)  # still has work
    registry.open_tree("w1", WARMUP, root_pending=True)

    released = registry.release_all(PROFILING)
    assert released == 2
    assert cm.released == [PROFILING, PROFILING]
    assert drained == []  # teardown does not recycle
    assert registry.open_count(PROFILING) == 0
    assert registry.open_count(WARMUP) == 1  # other phase untouched


def test_open_count_by_phase(registry):
    registry.open_tree("r1", PROFILING, root_pending=True)
    registry.open_tree("w1", WARMUP, root_pending=True)
    assert registry.open_count() == 2
    assert registry.open_count(PROFILING) == 1
    assert registry.open_count(WARMUP) == 1


def test_release_uses_the_trees_own_phase(cm, registry):
    """A tree releases against the phase it was opened with, even if drained via a descendant-done call that carries no phase."""
    registry.open_tree("w-root", WARMUP, root_pending=False)
    registry.register_descendants("w-root", 1)
    registry.on_descendant_done("w-root")
    assert cm.released == [WARMUP]

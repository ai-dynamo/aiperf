# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deferred FORK-parent eviction must be order-independent and bounded.

``fork_refcount`` alone is not a safe eviction trigger: FORK children arrive
one credit at a time, so the refcount returns to 0 between siblings. Evicting
the parent on the first join stranded every later sibling with an empty
history (``seed_from_parent`` silently no-ops on a missing parent), which
reads as a model regression rather than a cache bug.

The complementary hazard is the opposite one: a parent left in
``pending_fork_eviction`` forever when its children never reach this worker
(failed spawn, or a sticky miss routing them elsewhere). Nothing collects
those, and the LRU overflow pass skips them, so they defeat the cache cap.
"""

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.workers.session_manager import UserSession, UserSessionManager


@pytest.fixture
def session_manager() -> UserSessionManager:
    return UserSessionManager()


def _fork_conv(num_children: int, session_id: str = "root") -> Conversation:
    branch = ConversationBranchInfo(
        branch_id=f"{session_id}:0",
        mode=ConversationBranchMode.FORK,
        child_conversation_ids=[f"child-{i}" for i in range(num_children)],
    )
    return Conversation(
        session_id=session_id,
        turns=[Turn(branch_ids=[branch.branch_id])],
        branches=[branch],
    )


def _plain_session(x_correlation_id: str) -> UserSession:
    return UserSession(
        x_correlation_id=x_correlation_id,
        num_turns=1,
        conversation=Conversation(session_id=x_correlation_id, turns=[Turn()]),
    )


class TestExpectedForkChildrenStamping:
    @pytest.mark.parametrize(
        "num_children",
        [
            pytest.param(0, id="no_children"),
            pytest.param(1, id="one_child"),
            pytest.param(4, id="four_children"),
        ],
    )  # fmt: skip
    def test_expected_count_stamped_at_creation(
        self, session_manager: UserSessionManager, num_children: int
    ) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr",
            conversation=_fork_conv(num_children),
            num_turns=1,
        )
        assert session.expected_fork_children == num_children

    def test_spawn_children_are_not_counted(
        self, session_manager: UserSessionManager
    ) -> None:
        branch = ConversationBranchInfo(
            branch_id="s:0",
            mode=ConversationBranchMode.SPAWN,
            child_conversation_ids=["c0", "c1"],
        )
        conv = Conversation(
            session_id="s",
            turns=[Turn(branch_ids=["s:0"])],
            branches=[branch],
        )
        session = session_manager.create_and_store(
            x_correlation_id="corr-spawn", conversation=conv, num_turns=1
        )
        assert session.expected_fork_children == 0

    def test_count_survives_payload_bytes_round_trip(
        self, session_manager: UserSessionManager
    ) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-pb", conversation=_fork_conv(3), num_turns=1
        )
        session.conversation.branches = []
        assert session.expected_fork_children == 3


class TestDeferredEvictionSurvivesSerialChildArrival:
    def test_first_child_join_does_not_strand_later_siblings(
        self, session_manager: UserSessionManager
    ) -> None:
        """The regression: children pin serially, not all at once."""
        session = session_manager.create_and_store(
            x_correlation_id="parent", conversation=_fork_conv(2), num_turns=1
        )
        # Parent's terminal turn declared forks -> defer.
        session.pending_fork_eviction = True
        session_manager.evict_if_unpinned("parent")
        assert session_manager.get("parent") is not None

        # Child 0 arrives, pins, then joins. Refcount is back to 0 here.
        session_manager.pin_for_fork_child("parent")
        session_manager.release_fork_child("parent")
        assert session_manager.get("parent") is not None, (
            "parent evicted before sibling #2 could seed from it"
        )

        # Child 1 arrives, pins, seeds, joins -> now collectable.
        session_manager.pin_for_fork_child("parent")
        session_manager.release_fork_child("parent")
        assert session_manager.get("parent") is None

    def test_seed_still_works_for_the_second_child(
        self, session_manager: UserSessionManager
    ) -> None:
        parent = session_manager.create_and_store(
            x_correlation_id="parent", conversation=_fork_conv(2), num_turns=1
        )
        parent.turn_list = [Turn(), Turn()]
        parent.pending_fork_eviction = True
        session_manager.evict_if_unpinned("parent")

        session_manager.pin_for_fork_child("parent")
        session_manager.release_fork_child("parent")

        child = session_manager.create_and_store(
            x_correlation_id="child-1",
            conversation=Conversation(session_id="child-1", turns=[Turn()]),
            num_turns=1,
            parent_correlation_id="parent",
            branch_mode=ConversationBranchMode.FORK,
        )
        session_manager.seed_from_parent("child-1", "parent")
        assert len(child.turn_list) == 2

    def test_children_joining_before_parent_terminal_is_not_stranded(
        self, session_manager: UserSessionManager
    ) -> None:
        """All children join first; the parent's own terminal turn collects it."""
        session = session_manager.create_and_store(
            x_correlation_id="parent", conversation=_fork_conv(2), num_turns=1
        )
        for _ in range(2):
            session_manager.pin_for_fork_child("parent")
            session_manager.release_fork_child("parent")
        assert session_manager.get("parent") is not None

        session.pending_fork_eviction = True
        session_manager.evict_if_unpinned("parent")
        assert session_manager.get("parent") is None

    def test_active_pin_still_blocks_eviction(
        self, session_manager: UserSessionManager
    ) -> None:
        session_manager.create_and_store(
            x_correlation_id="parent", conversation=_fork_conv(1), num_turns=1
        )
        session_manager.pin_for_fork_child("parent")
        session_manager.evict_if_unpinned("parent")
        assert session_manager.get("parent") is not None


def _stale_pending_session(x_correlation_id: str) -> UserSession:
    """A parent whose deferral is already satisfied but that nothing collected.

    Real shape of the leak: the terminal turn set ``pending_fork_eviction``,
    the declared children joined (or never reached this worker at all), and
    ``release_fork_child`` had already run its last decrement elsewhere.
    """
    return UserSession(
        x_correlation_id=x_correlation_id,
        num_turns=1,
        conversation=Conversation(session_id=x_correlation_id, turns=[Turn()]),
        is_fork_parent=True,
        pending_fork_eviction=True,
        expected_fork_children=1,
        joined_fork_children=1,
    )


class TestStaleDeferredParentsAreReclaimed:
    """Children that never reach this worker must not make the parent immortal."""

    def test_unpinned_sessions_are_still_evicted_first(self) -> None:
        mgr = UserSessionManager(max_sessions=2)
        mgr.store("pending", _stale_pending_session("pending"))
        mgr.store("plain", _plain_session("plain"))

        mgr.store("newcomer", _plain_session("newcomer"))

        assert "pending" in mgr._cache, "a deferred parent outranked a plain session"
        assert "plain" not in mgr._cache
        assert mgr.stale_fork_evictions == 0

    def test_outstanding_children_keep_the_parent_over_cap(self) -> None:
        """Reclaim is a last resort, never at the cost of a live child."""
        mgr = UserSessionManager(max_sessions=1)
        parent = mgr.create_and_store(
            x_correlation_id="parent", conversation=_fork_conv(2), num_turns=1
        )
        parent.pending_fork_eviction = True

        mgr.store("plain", _plain_session("plain"))

        assert "parent" in mgr._cache
        assert mgr.stale_fork_evictions == 0

    def test_all_pinned_by_refcount_still_stays_over_cap(self) -> None:
        """An actively-pinned parent is never reclaimed, cap pressure or not."""
        mgr = UserSessionManager(max_sessions=2)
        for key in ("a", "b"):
            mgr.store(key, _plain_session(key))
        mgr.pin_for_fork_child("a")
        mgr.pin_for_fork_child("b")

        mgr.store("c", _plain_session("c"))

        assert set(mgr._cache) == {"a", "b"}
        assert mgr.stale_fork_evictions == 0

    def test_stale_pending_parents_do_not_defeat_the_cap(self) -> None:
        mgr = UserSessionManager(max_sessions=4)
        for i in range(50):
            mgr.store(f"parent-{i}", _stale_pending_session(f"parent-{i}"))

        assert len(mgr._cache) == 4, "deferred parents grew the cache without bound"
        assert mgr.stale_fork_evictions == 46
        # Least-recently-used stale parents went first.
        assert set(mgr._cache) == {f"parent-{i}" for i in range(46, 50)}

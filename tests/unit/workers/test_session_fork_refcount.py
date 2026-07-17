# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for FORK-pin refcounting on ``UserSession`` / ``UserSessionManager``.

A DAG-FORK parent session must remain resident in the worker's session
cache while any of its FORK children still need to dispatch credits
against the parent's history. Refcount semantics:

- ``fork_refcount`` defaults to 0 on a freshly-created session.
- ``pin_for_fork_child`` increments the count (one bump per child seed).
- ``release_fork_child`` decrements, floored at 0 (no negatives).
- ``evict_if_unpinned`` is a no-op while the count is > 0; it removes
  the session from the cache only when the count has reached 0.

The worker call sites that drive this refcount (bump on child seed,
decrement on child join) land in the orchestrator wiring task (P2T26);
this test exercises the storage half only.
"""

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.workers.session_manager import UserSessionManager


@pytest.fixture
def session_manager() -> UserSessionManager:
    return UserSessionManager()


def _conv_with_n_forks(n: int, session_id: str = "root") -> Conversation:
    branches = [
        ConversationBranchInfo(
            branch_id=f"{session_id}:{i}",
            mode=ConversationBranchMode.FORK,
            child_conversation_ids=[f"child-{i}"],
        )
        for i in range(n)
    ]
    return Conversation(
        session_id=session_id,
        turns=[Turn(branch_ids=[f"{session_id}:{i}" for i in range(n)])],
        branches=branches,
    )


def _conv_with_packed_forks(n: int, session_id: str = "root") -> Conversation:
    """One FORK branch entry carrying ``n`` children -- the dag_jsonl loader shape."""
    branch = ConversationBranchInfo(
        branch_id=f"{session_id}:0",
        mode=ConversationBranchMode.FORK,
        child_conversation_ids=[f"child-{i}" for i in range(n)],
    )
    return Conversation(
        session_id=session_id,
        turns=[Turn(branch_ids=[branch.branch_id])],
        branches=[branch],
    )


class TestForkRefcount:
    def test_default_refcount_zero(self, session_manager: UserSessionManager) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-zero",
            conversation=_conv_with_n_forks(2),
            num_turns=1,
        )
        assert session.fork_refcount == 0

    def test_increment_per_child_seed(
        self, session_manager: UserSessionManager
    ) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-inc",
            conversation=_conv_with_n_forks(3),
            num_turns=1,
        )
        for _ in range(3):
            session_manager.pin_for_fork_child("corr-inc")
        assert session.fork_refcount == 3

    def test_decrement_on_child_join(self, session_manager: UserSessionManager) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-dec",
            conversation=_conv_with_n_forks(2),
            num_turns=1,
        )
        session_manager.pin_for_fork_child("corr-dec")
        session_manager.pin_for_fork_child("corr-dec")
        session_manager.release_fork_child("corr-dec")
        assert session.fork_refcount == 1

    def test_evict_only_when_refcount_zero(
        self, session_manager: UserSessionManager
    ) -> None:
        session_manager.create_and_store(
            x_correlation_id="corr-evict",
            conversation=_conv_with_n_forks(1),
            num_turns=1,
        )
        session_manager.pin_for_fork_child("corr-evict")

        # Pinned: evict_if_unpinned should be a no-op.
        session_manager.evict_if_unpinned("corr-evict")
        assert session_manager.get("corr-evict") is not None

        # Released: evict_if_unpinned should remove it.
        session_manager.release_fork_child("corr-evict")
        session_manager.evict_if_unpinned("corr-evict")
        assert session_manager.get("corr-evict") is None

    def test_release_floor_zero(self, session_manager: UserSessionManager) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-floor",
            conversation=_conv_with_n_forks(1),
            num_turns=1,
        )
        # Already at 0 — release must not go negative.
        session_manager.release_fork_child("corr-floor")
        session_manager.release_fork_child("corr-floor")
        assert session.fork_refcount == 0

    def test_pin_unknown_session_raises(
        self, session_manager: UserSessionManager
    ) -> None:
        with pytest.raises(KeyError):
            session_manager.pin_for_fork_child("does-not-exist")

    def test_release_unknown_session_is_noop(
        self, session_manager: UserSessionManager
    ) -> None:
        # Should not raise — release is best-effort.
        session_manager.release_fork_child("does-not-exist")

    def test_evict_if_unpinned_unknown_session_is_noop(
        self, session_manager: UserSessionManager
    ) -> None:
        session_manager.evict_if_unpinned("does-not-exist")


class TestPendingForkEvictionLateSibling:
    """A pending-eviction FORK parent must survive until every declared child pins.

    Regression for the fork-minimal byte-parity race: with two children
    forking off the parent's terminal turn, an early child can complete
    its WHOLE conversation (pin -> seed -> run -> release) before the
    sibling's credit reaches the worker. The release drops the refcount
    back to 0 with ``pending_fork_eviction`` set; evicting there yanks
    the parent out from under the late sibling, which then dispatches
    with NO inherited history (bare user message on the wire).
    """

    @pytest.mark.parametrize(
        "conversation_builder",
        [
            pytest.param(lambda: _conv_with_n_forks(2), id="one-entry-per-child"),
            # The REAL dag_jsonl loader shape: _apply_forks packs every fork
            # declared on one turn into a SINGLE ConversationBranchInfo whose
            # child_conversation_ids lists all of them. Counting branch
            # ENTRIES instead of CHILDREN here read expected=1 and evicted
            # the parent after the first child's release.
            pytest.param(
                lambda: _conv_with_packed_forks(2), id="single-entry-two-children"
            ),
        ],
    )  # fmt: skip
    def test_parent_survives_release_until_all_expected_children_pin(
        self, session_manager: UserSessionManager, conversation_builder
    ) -> None:
        session = session_manager.create_and_store(
            x_correlation_id="corr-late-sibling",
            conversation=conversation_builder(),
            num_turns=1,
        )
        assert session.fork_children_expected == 2

        # Parent's terminal turn fired before any child arrived: eviction
        # deferred, session stays resident.
        session.pending_fork_eviction = True
        session_manager.evict_if_unpinned("corr-late-sibling")
        assert session_manager.get("corr-late-sibling") is not None

        # Child A pins, seeds, runs its whole conversation, and releases --
        # all before child B's credit reaches the worker.
        session_manager.pin_for_fork_child("corr-late-sibling")
        session_manager.release_fork_child("corr-late-sibling")
        assert session_manager.get("corr-late-sibling") is not None, (
            "parent evicted after the FIRST child's release while the "
            "second declared child had not pinned yet"
        )

        # Child B arrives late, pins + seeds, then releases: NOW every
        # declared child has joined and the parent is collected.
        session_manager.pin_for_fork_child("corr-late-sibling")
        session_manager.release_fork_child("corr-late-sibling")
        assert session_manager.get("corr-late-sibling") is None

    def test_parent_without_pending_eviction_survives_all_releases(
        self, session_manager: UserSessionManager
    ) -> None:
        """Mid-conversation forks: children join while the parent is still
        running (no pending eviction) -- releases never collect the session."""
        session_manager.create_and_store(
            x_correlation_id="corr-mid-conv",
            conversation=_conv_with_n_forks(2),
            num_turns=1,
        )
        for _ in range(2):
            session_manager.pin_for_fork_child("corr-mid-conv")
            session_manager.release_fork_child("corr-mid-conv")
        assert session_manager.get("corr-mid-conv") is not None

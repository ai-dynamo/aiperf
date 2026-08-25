# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-worker session cache must be bounded.

Sessions are normally evicted on the final turn or on cancellation. Abandoned
ones never are: a non-final CreditReturn reclaimed sticky-router side on worker
reconnect or detach leaves an entry that no final-turn credit will ever reach.
Unbounded, those accrue for the process lifetime and the container is
eventually OOMKilled on a long multi-turn run.
"""

import pytest

from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.workers.session_manager import (
    DEFAULT_MAX_SESSIONS,
    UserSession,
    UserSessionManager,
)


def _session(x_correlation_id: str) -> UserSession:
    """A real session, not a MagicMock.

    The eviction path reads ``fork_refcount`` / ``pending_fork_eviction`` to
    decide what it is allowed to drop, and every attribute of a MagicMock is
    truthy -- against mocks this file passed while claiming every session was
    pinned.
    """
    return UserSession(
        x_correlation_id=x_correlation_id,
        num_turns=1,
        conversation=Conversation(session_id=x_correlation_id, turns=[Turn()]),
    )


def _add(mgr: UserSessionManager, n: int) -> None:
    for i in range(n):
        mgr.store(f"sess-{i}", _session(f"sess-{i}"))


class TestSessionCacheIsBounded:
    def test_default_cap_is_exposed(self):
        assert DEFAULT_MAX_SESSIONS >= 1

    def test_cache_never_exceeds_the_cap(self):
        mgr = UserSessionManager(max_sessions=10)
        _add(mgr, 25)
        assert len(mgr._cache) == 10

    def test_oldest_untouched_session_is_evicted_first(self):
        mgr = UserSessionManager(max_sessions=3)
        _add(mgr, 3)
        # touch the oldest so it is no longer the least-recently-used
        mgr.get("sess-0")

        mgr.store("sess-3", _session("sess-3"))

        assert "sess-0" in mgr._cache, "a recently used session was evicted"
        assert "sess-1" not in mgr._cache
        assert set(mgr._cache) == {"sess-0", "sess-2", "sess-3"}

    def test_rejects_a_nonsensical_cap(self):
        with pytest.raises(ValueError):
            UserSessionManager(max_sessions=0)

    def test_explicit_removal_still_works(self):
        mgr = UserSessionManager(max_sessions=5)
        _add(mgr, 2)
        mgr.evict("sess-0")
        assert "sess-0" not in mgr._cache


class TestOverflowNeverDropsAPinnedParent:
    """LRU order must not outrank a FORK pin.

    A pinned parent is the context its not-yet-dispatched children seed from.
    Evicting it makes ``seed_from_parent`` a silent no-op and every child goes
    out with an empty history, which looks like a model regression rather than
    a cache bug.
    """

    def test_pinned_parent_survives_while_younger_sessions_are_evicted(self):
        mgr = UserSessionManager(max_sessions=3)
        _add(mgr, 3)
        mgr.pin_for_fork_child("sess-0")  # oldest, would be the first LRU victim

        _add_more = [f"extra-{i}" for i in range(3)]
        for key in _add_more:
            mgr.store(key, _session(key))

        assert "sess-0" in mgr._cache, "a FORK-pinned parent was evicted"
        assert len(mgr._cache) == 3

    def test_pending_fork_eviction_parent_survives(self):
        mgr = UserSessionManager(max_sessions=2)
        _add(mgr, 2)
        mgr._cache["sess-0"].pending_fork_eviction = True

        mgr.store("sess-2", _session("sess-2"))

        assert "sess-0" in mgr._cache
        assert "sess-1" not in mgr._cache

    def test_all_pinned_keeps_the_cache_over_cap_rather_than_evicting(self):
        mgr = UserSessionManager(max_sessions=2)
        _add(mgr, 2)
        mgr.pin_for_fork_child("sess-0")
        mgr.pin_for_fork_child("sess-1")

        mgr.store("sess-2", _session("sess-2"))

        # Newest is unpinned, so it goes first; the pinned pair stays resident
        # and the cache is knowingly left over cap.
        assert set(mgr._cache) == {"sess-0", "sess-1"}

    def test_unpinning_lets_the_parent_be_evicted_again(self):
        mgr = UserSessionManager(max_sessions=2)
        _add(mgr, 2)
        mgr.pin_for_fork_child("sess-0")
        mgr.release_fork_child("sess-0")

        mgr.store("sess-2", _session("sess-2"))

        assert "sess-0" not in mgr._cache
        assert set(mgr._cache) == {"sess-1", "sess-2"}

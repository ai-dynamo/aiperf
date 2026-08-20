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

from aiperf.workers.session_manager import DEFAULT_MAX_SESSIONS, UserSessionManager


def _add(mgr: UserSessionManager, n: int) -> None:
    from unittest.mock import MagicMock

    for i in range(n):
        mgr.store(f"sess-{i}", MagicMock())


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
        from unittest.mock import MagicMock

        mgr.store("sess-3", MagicMock())

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

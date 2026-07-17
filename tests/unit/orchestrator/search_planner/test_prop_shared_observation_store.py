# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Shared Observation Store Invariant.

Feature: multi-tier-slo-search, Property 4: Shared Observation Store Invariant

Validates: Requirements 2.2, 2.4

For any sequence of probes, the observation store SHALL contain exactly one
entry per distinct concurrency level probed, regardless of how many tiers
requested that concurrency.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.multi_tier_store import SharedObservationStore

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _run_result(label: str = "run") -> RunResult:
    """Create a minimal valid RunResult."""
    return RunResult(
        label=label,
        success=True,
        summary_metrics={},
        artifacts_path=Path("/tmp/fake"),
    )


def _probe_sequence() -> st.SearchStrategy[list[tuple[int, list[RunResult]]]]:
    """Generate a random sequence of (concurrency_level, results_list) pairs.

    Concurrency levels are drawn from a small range to encourage duplicates.
    """
    return st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=50),
            st.lists(
                st.builds(
                    lambda label, success: RunResult(
                        label=label,
                        success=success,
                        summary_metrics={},
                        artifacts_path=Path("/tmp/fake"),
                    ),
                    label=st.text(
                        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
                        min_size=1,
                        max_size=8,
                    ),
                    success=st.booleans(),
                ),
                min_size=1,
                max_size=5,
            ),
        ),
        min_size=1,
        max_size=30,
    )


# ---------------------------------------------------------------------------
# Property 4: Shared Observation Store Invariant
# ---------------------------------------------------------------------------


class TestProperty4SharedObservationStoreInvariant:
    """Property 4: Shared Observation Store Invariant.

    **Validates: Requirements 2.2, 2.4**
    """

    @given(probes=_probe_sequence())
    @settings(max_examples=100, deadline=None)
    def test_concurrency_levels_match_distinct_inputs(
        self,
        probes: list[tuple[int, list[RunResult]]],
    ) -> None:
        """Store contains exactly one entry key per distinct concurrency level.

        **Validates: Requirements 2.2, 2.4**
        """
        store = SharedObservationStore()

        for concurrency, results in probes:
            store.store(concurrency, results)

        expected_levels = sorted({c for c, _ in probes})
        assert store.concurrency_levels() == expected_levels

    @given(probes=_probe_sequence())
    @settings(max_examples=100, deadline=None)
    def test_get_returns_all_probes_at_concurrency(
        self,
        probes: list[tuple[int, list[RunResult]]],
    ) -> None:
        """For each concurrency, get() returns exactly the number of times it was stored.

        **Validates: Requirements 2.2, 2.4**
        """
        store = SharedObservationStore()

        for concurrency, results in probes:
            store.store(concurrency, results)

        # Count how many times each concurrency was stored
        expected_counts: dict[int, int] = {}
        for concurrency, _ in probes:
            expected_counts[concurrency] = expected_counts.get(concurrency, 0) + 1

        for concurrency, expected_count in expected_counts.items():
            stored = store.get(concurrency)
            assert len(stored) == expected_count

    @given(probes=_probe_sequence())
    @settings(max_examples=100, deadline=None)
    def test_store_is_shared_single_index(
        self,
        probes: list[tuple[int, list[RunResult]]],
    ) -> None:
        """Multiple stores from different 'tiers' writing to same store share one index.

        Simulates multiple tiers requesting the same concurrency level and
        verifies the store maintains a single entry key regardless of requester.

        **Validates: Requirements 2.2, 2.4**
        """
        store = SharedObservationStore()

        # Simulate tier A and tier B both storing at the same concurrency levels
        for concurrency, results in probes:
            store.store(concurrency, results)
        # Store again as if from a different tier
        for concurrency, results in probes:
            store.store(concurrency, results)

        # Still exactly one key per distinct concurrency
        expected_levels = sorted({c for c, _ in probes})
        assert store.concurrency_levels() == expected_levels

        # But each concurrency now has double the probe count
        expected_counts: dict[int, int] = {}
        for concurrency, _ in probes:
            expected_counts[concurrency] = expected_counts.get(concurrency, 0) + 1

        for concurrency, expected_count in expected_counts.items():
            stored = store.get(concurrency)
            assert len(stored) == expected_count * 2

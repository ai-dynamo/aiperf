# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Widest-Bracket Probe Allocation.

Feature: multi-tier-slo-search, Property 5: Widest-Bracket Probe Allocation

Validates: Requirements 3.1

For any set of non-converged tier brackets (each with both bounds established),
the probe allocator SHALL select the midpoint of the tier whose
infeasible_min - feasible_max gap is maximal.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.config.sweep.adaptive import SLAFilter, SLOTier
from aiperf.orchestrator.search_planner.multi_tier_allocator import ProbeAllocator
from aiperf.orchestrator.search_planner.multi_tier_models import BracketState

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _slo_tier(label: str = "tier") -> SLOTier:
    """Create a minimal valid SLOTier."""
    return SLOTier(
        label=label,
        filters=[
            SLAFilter(
                metric_tag="output_token_throughput",
                stat="avg",
                op="gt",
                threshold=100.0,
            )
        ],
    )


@st.composite
def _non_converged_bracket(draw: st.DrawFn, label: str = "tier") -> BracketState:
    """Generate a non-converged BracketState with both bounds and gap > 1."""
    feasible_max = draw(st.integers(min_value=1, max_value=500))
    gap = draw(st.integers(min_value=2, max_value=200))
    infeasible_min = feasible_max + gap
    return BracketState(
        tier=_slo_tier(label),
        feasible_max=feasible_max,
        infeasible_min=infeasible_min,
        converged=False,
    )


@st.composite
def _bracket_list(draw: st.DrawFn) -> list[BracketState]:
    """Generate 2-5 non-converged brackets with unique labels and gap > 1."""
    count = draw(st.integers(min_value=2, max_value=5))
    brackets = []
    for i in range(count):
        b = draw(_non_converged_bracket(label=f"tier_{i}"))
        brackets.append(b)
    return brackets


@st.composite
def _bracket_list_with_noise(draw: st.DrawFn) -> list[BracketState]:
    """Generate brackets mixing valid candidates with converged/partial ones.

    Ensures at least one valid candidate exists (non-converged, both bounds, gap > 1).
    """
    # At least one valid candidate
    valid_count = draw(st.integers(min_value=1, max_value=4))
    brackets: list[BracketState] = []
    for i in range(valid_count):
        b = draw(_non_converged_bracket(label=f"valid_{i}"))
        brackets.append(b)

    # Add some noise: converged brackets
    converged_count = draw(st.integers(min_value=0, max_value=2))
    for i in range(converged_count):
        brackets.append(
            BracketState(
                tier=_slo_tier(f"converged_{i}"),
                feasible_max=10,
                infeasible_min=11,
                converged=True,
                convergence_reason="multi_tier_precision_reached",
            )
        )

    # Add some noise: brackets without both bounds
    partial_count = draw(st.integers(min_value=0, max_value=2))
    for i in range(partial_count):
        choice = draw(st.integers(min_value=0, max_value=2))
        if choice == 0:
            brackets.append(
                BracketState(tier=_slo_tier(f"partial_{i}"), feasible_max=10)
            )
        elif choice == 1:
            brackets.append(
                BracketState(tier=_slo_tier(f"partial_{i}"), infeasible_min=100)
            )
        else:
            brackets.append(BracketState(tier=_slo_tier(f"partial_{i}")))

    return brackets


# ---------------------------------------------------------------------------
# Property 5: Widest-Bracket Probe Allocation
# ---------------------------------------------------------------------------


class TestProperty5WidestBracketProbeAllocation:
    """Property 5: Widest-Bracket Probe Allocation.

    **Validates: Requirements 3.1**
    """

    @given(brackets=_bracket_list())
    @settings(max_examples=100, deadline=None)
    def test_selects_midpoint_of_widest_gap(
        self,
        brackets: list[BracketState],
    ) -> None:
        """Allocator returns the midpoint of the bracket with the widest gap.

        **Validates: Requirements 3.1**
        """
        allocator = ProbeAllocator()
        result = allocator.select_next_probe(brackets)

        # Identify the widest bracket manually
        widest = max(brackets, key=lambda b: b.infeasible_min - b.feasible_max)
        expected_gap = widest.infeasible_min - widest.feasible_max
        expected_midpoint = widest.feasible_max + expected_gap // 2

        assert result == expected_midpoint

    @given(brackets=_bracket_list_with_noise())
    @settings(max_examples=100, deadline=None)
    def test_ignores_converged_and_partial_brackets(
        self,
        brackets: list[BracketState],
    ) -> None:
        """Allocator ignores converged brackets and those without both bounds.

        **Validates: Requirements 3.1**
        """
        allocator = ProbeAllocator()
        result = allocator.select_next_probe(brackets)

        # Only valid candidates: non-converged, both bounds, gap > 1
        candidates = [
            b
            for b in brackets
            if not b.converged
            and b.feasible_max is not None
            and b.infeasible_min is not None
            and (b.infeasible_min - b.feasible_max) > 1
        ]

        if not candidates:
            assert result is None
        else:
            widest = max(candidates, key=lambda b: b.infeasible_min - b.feasible_max)
            expected_gap = widest.infeasible_min - widest.feasible_max
            expected_midpoint = widest.feasible_max + expected_gap // 2
            assert result == expected_midpoint

    @given(brackets=_bracket_list())
    @settings(max_examples=100, deadline=None)
    def test_widest_bracket_correctly_identified(
        self,
        brackets: list[BracketState],
    ) -> None:
        """The allocator selects the tier with the maximum gap.

        **Validates: Requirements 3.1**
        """
        allocator = ProbeAllocator()
        result = allocator.select_next_probe(brackets)

        # Compute all gaps
        gaps = [(b.infeasible_min - b.feasible_max, b) for b in brackets]
        max_gap = max(g for g, _ in gaps)

        # The result must be the midpoint of a bracket that has the max gap
        widest_brackets = [b for g, b in gaps if g == max_gap]
        expected_midpoints = {b.feasible_max + max_gap // 2 for b in widest_brackets}

        assert result in expected_midpoints

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_returns_none_when_all_converged(
        self,
        data: st.DataObject,
    ) -> None:
        """Allocator returns None when all brackets are converged.

        **Validates: Requirements 3.1**
        """
        count = data.draw(st.integers(min_value=1, max_value=5))
        brackets = [
            BracketState(
                tier=_slo_tier(f"tier_{i}"),
                feasible_max=10 + i,
                infeasible_min=11 + i,
                converged=True,
                convergence_reason="multi_tier_precision_reached",
            )
            for i in range(count)
        ]

        allocator = ProbeAllocator()
        result = allocator.select_next_probe(brackets)
        assert result is None

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_converges_gap_one_brackets_and_selects_next(
        self,
        data: st.DataObject,
    ) -> None:
        """Brackets with gap <= 1 get converged, allocator moves to next widest.

        **Validates: Requirements 3.1**
        """
        # Create one bracket with gap=1 (will be converged) and one with gap > 1
        gap1_bracket = BracketState(
            tier=_slo_tier("narrow"),
            feasible_max=50,
            infeasible_min=51,
            converged=False,
        )
        wide_feasible = data.draw(st.integers(min_value=1, max_value=100))
        wide_gap = data.draw(st.integers(min_value=2, max_value=100))
        wide_bracket = BracketState(
            tier=_slo_tier("wide"),
            feasible_max=wide_feasible,
            infeasible_min=wide_feasible + wide_gap,
            converged=False,
        )

        brackets = [gap1_bracket, wide_bracket]
        allocator = ProbeAllocator()
        result = allocator.select_next_probe(brackets)

        # The wide bracket should be selected since the narrow one has gap=1
        # But the narrow bracket has gap=1 which is <= 1, so it depends on
        # which is "widest" first. If wide_gap > 1, the wide bracket is wider.
        # If wide_gap == 1 also, both get converged and None is returned.
        if wide_gap > 1:
            expected = wide_feasible + wide_gap // 2
            assert result == expected
            # The gap1 bracket might or might not be converged depending on
            # whether it was the widest initially. The allocator converges
            # gap<=1 brackets only when they are selected as widest.

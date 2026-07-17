# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Ordering Inference Correctness.

Feature: multi-tier-slo-search, Property 8: Ordering Inference Correctness

Validates: Requirements 4.1, 4.2, 4.3

For any set of tiers with detected monotonic ordering (strict tier A, lenient
tier B), if a probe at concurrency X fails tier B, then tier A's
infeasible_min SHALL be updated to min(A.infeasible_min, X). Symmetrically,
if a probe passes tier A, then tier B's feasible_max SHALL be updated to
max(B.feasible_max, X).
"""

from __future__ import annotations

from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.config.sweep.adaptive import SLAFilter, SLOTier
from aiperf.orchestrator.search_planner.multi_tier_models import BracketState
from aiperf.orchestrator.search_planner.multi_tier_ordering import TierOrderingDetector

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_OPS_HIGHER_HARDER: list[Literal["gt", "ge"]] = ["gt", "ge"]
_OPS_LOWER_HARDER: list[Literal["lt", "le"]] = ["lt", "le"]
_STATS: list[Literal["avg", "p50", "p90", "p95", "p99"]] = [
    "avg",
    "p50",
    "p90",
    "p95",
    "p99",
]
_METRICS = ["throughput", "latency", "ttft", "tpot"]


@st.composite
def _ordered_tier_pair(draw: st.DrawFn) -> tuple[SLOTier, SLOTier, str]:
    """Generate a pair of tiers with known monotonic ordering.

    Returns (strict_tier, lenient_tier, direction) where direction is
    'higher_harder' or 'lower_harder'.
    """
    num_filters = draw(st.integers(min_value=1, max_value=3))
    direction = draw(st.sampled_from(["higher_harder", "lower_harder"]))

    strict_filters: list[SLAFilter] = []
    lenient_filters: list[SLAFilter] = []

    for i in range(num_filters):
        metric = draw(st.sampled_from(_METRICS))
        # Make each filter unique by appending index to metric
        metric_tag = f"{metric}_{i}"
        stat = draw(st.sampled_from(_STATS))

        if direction == "higher_harder":
            op = draw(st.sampled_from(_OPS_HIGHER_HARDER))
            # Strict tier has higher threshold (harder)
            lenient_threshold = draw(st.floats(min_value=1.0, max_value=500.0))
            strict_threshold = draw(
                st.floats(
                    min_value=lenient_threshold + 0.01,
                    max_value=lenient_threshold + 500.0,
                )
            )
        else:
            op = draw(st.sampled_from(_OPS_LOWER_HARDER))
            # Strict tier has lower threshold (harder)
            strict_threshold = draw(st.floats(min_value=1.0, max_value=500.0))
            lenient_threshold = draw(
                st.floats(
                    min_value=strict_threshold + 0.01,
                    max_value=strict_threshold + 500.0,
                )
            )

        strict_filters.append(
            SLAFilter(
                metric_tag=metric_tag, stat=stat, op=op, threshold=strict_threshold
            )
        )
        lenient_filters.append(
            SLAFilter(
                metric_tag=metric_tag, stat=stat, op=op, threshold=lenient_threshold
            )
        )

    strict_tier = SLOTier(label="strict", filters=strict_filters)
    lenient_tier = SLOTier(label="lenient", filters=lenient_filters)
    return strict_tier, lenient_tier, direction


@st.composite
def _optional_bracket_bound(draw: st.DrawFn) -> int | None:
    """Generate an optional pre-existing bracket bound or None."""
    if draw(st.booleans()):
        return draw(st.integers(min_value=1, max_value=500))
    return None


@st.composite
def _concurrency_value(draw: st.DrawFn) -> int:
    """Generate a valid concurrency value for probing."""
    return draw(st.integers(min_value=1, max_value=500))


# ---------------------------------------------------------------------------
# Property 8: Ordering Inference Correctness
# ---------------------------------------------------------------------------


class TestProperty8OrderingInferenceCorrectness:
    """Property 8: Ordering Inference Correctness.

    **Validates: Requirements 4.1, 4.2, 4.3**
    """

    @given(
        tier_pair=_ordered_tier_pair(),
        concurrency=_concurrency_value(),
        existing_infeasible_min=_optional_bracket_bound(),
    )
    @settings(max_examples=100, deadline=None)
    def test_fail_down_updates_strict_infeasible_min(
        self,
        tier_pair: tuple[SLOTier, SLOTier, str],
        concurrency: int,
        existing_infeasible_min: int | None,
    ) -> None:
        """When lenient tier fails at X, strict tier's infeasible_min = min(existing, X).

        **Validates: Requirements 4.1, 4.2**
        """
        strict_tier, lenient_tier, _ = tier_pair
        tiers = [strict_tier, lenient_tier]
        detector = TierOrderingDetector(tiers)
        pairs = detector.detect_ordering()

        # Verify ordering was detected
        assert (0, 1) in pairs

        brackets = [BracketState(tier=t) for t in tiers]
        brackets[0].infeasible_min = existing_infeasible_min

        detector.propagate_fail(
            failed_tier_idx=1, concurrency=concurrency, brackets=brackets
        )

        # Check if the pair was disabled due to contradiction
        if (0, 1) in detector.disabled_pairs:
            # Contradiction detected - the update still happened before disable
            # but we only verify the min semantics were applied correctly
            if existing_infeasible_min is None or concurrency < existing_infeasible_min:
                assert brackets[0].infeasible_min == concurrency
            else:
                assert brackets[0].infeasible_min == existing_infeasible_min
        else:
            # Normal propagation: infeasible_min = min(existing, X)
            if existing_infeasible_min is None:
                assert brackets[0].infeasible_min == concurrency
            else:
                assert brackets[0].infeasible_min == min(
                    existing_infeasible_min, concurrency
                )

    @given(
        tier_pair=_ordered_tier_pair(),
        concurrency=_concurrency_value(),
        existing_feasible_max=_optional_bracket_bound(),
    )
    @settings(max_examples=100, deadline=None)
    def test_pass_up_updates_lenient_feasible_max(
        self,
        tier_pair: tuple[SLOTier, SLOTier, str],
        concurrency: int,
        existing_feasible_max: int | None,
    ) -> None:
        """When strict tier passes at X, lenient tier's feasible_max = max(existing, X).

        **Validates: Requirements 4.1, 4.3**
        """
        strict_tier, lenient_tier, _ = tier_pair
        tiers = [strict_tier, lenient_tier]
        detector = TierOrderingDetector(tiers)
        pairs = detector.detect_ordering()

        # Verify ordering was detected
        assert (0, 1) in pairs

        brackets = [BracketState(tier=t) for t in tiers]
        brackets[1].feasible_max = existing_feasible_max

        detector.propagate_pass(
            passed_tier_idx=0, concurrency=concurrency, brackets=brackets
        )

        # Check if the pair was disabled due to contradiction
        if (0, 1) in detector.disabled_pairs:
            # Contradiction detected - the update still happened before disable
            if existing_feasible_max is None or concurrency > existing_feasible_max:
                assert brackets[1].feasible_max == concurrency
            else:
                assert brackets[1].feasible_max == existing_feasible_max
        else:
            # Normal propagation: feasible_max = max(existing, X)
            if existing_feasible_max is None:
                assert brackets[1].feasible_max == concurrency
            else:
                assert brackets[1].feasible_max == max(
                    existing_feasible_max, concurrency
                )

    @given(
        tier_pair=_ordered_tier_pair(),
        concurrency=_concurrency_value(),
        existing_infeasible_min=_optional_bracket_bound(),
    )
    @settings(max_examples=100, deadline=None)
    def test_fail_down_respects_min_semantics(
        self,
        tier_pair: tuple[SLOTier, SLOTier, str],
        concurrency: int,
        existing_infeasible_min: int | None,
    ) -> None:
        """Fail-down never increases infeasible_min — only decreases or sets it.

        **Validates: Requirements 4.2**
        """
        strict_tier, lenient_tier, _ = tier_pair
        tiers = [strict_tier, lenient_tier]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()

        brackets = [BracketState(tier=t) for t in tiers]
        brackets[0].infeasible_min = existing_infeasible_min

        detector.propagate_fail(
            failed_tier_idx=1, concurrency=concurrency, brackets=brackets
        )

        new_min = brackets[0].infeasible_min
        if existing_infeasible_min is not None:
            assert new_min is not None
            assert new_min <= existing_infeasible_min

    @given(
        tier_pair=_ordered_tier_pair(),
        concurrency=_concurrency_value(),
        existing_feasible_max=_optional_bracket_bound(),
    )
    @settings(max_examples=100, deadline=None)
    def test_pass_up_respects_max_semantics(
        self,
        tier_pair: tuple[SLOTier, SLOTier, str],
        concurrency: int,
        existing_feasible_max: int | None,
    ) -> None:
        """Pass-up never decreases feasible_max — only increases or sets it.

        **Validates: Requirements 4.3**
        """
        strict_tier, lenient_tier, _ = tier_pair
        tiers = [strict_tier, lenient_tier]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()

        brackets = [BracketState(tier=t) for t in tiers]
        brackets[1].feasible_max = existing_feasible_max

        detector.propagate_pass(
            passed_tier_idx=0, concurrency=concurrency, brackets=brackets
        )

        new_max = brackets[1].feasible_max
        if existing_feasible_max is not None:
            assert new_max is not None
            assert new_max >= existing_feasible_max

    @given(
        tier_pair=_ordered_tier_pair(),
        concurrency=_concurrency_value(),
    )
    @settings(max_examples=100, deadline=None)
    def test_fail_strict_does_not_propagate_to_lenient(
        self,
        tier_pair: tuple[SLOTier, SLOTier, str],
        concurrency: int,
    ) -> None:
        """Failing the strict tier does NOT propagate infeasible_min to lenient tier.

        The ordering only implies fail-down (lenient fails -> strict fails).
        A strict tier failing says nothing about the lenient tier.

        **Validates: Requirements 4.1, 4.2**
        """
        strict_tier, lenient_tier, _ = tier_pair
        tiers = [strict_tier, lenient_tier]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()

        brackets = [BracketState(tier=t) for t in tiers]

        detector.propagate_fail(
            failed_tier_idx=0, concurrency=concurrency, brackets=brackets
        )

        # Lenient tier should NOT be updated by strict tier failing
        assert brackets[1].infeasible_min is None

    @given(
        tier_pair=_ordered_tier_pair(),
        concurrency=_concurrency_value(),
    )
    @settings(max_examples=100, deadline=None)
    def test_pass_lenient_does_not_propagate_to_strict(
        self,
        tier_pair: tuple[SLOTier, SLOTier, str],
        concurrency: int,
    ) -> None:
        """Passing the lenient tier does NOT propagate feasible_max to strict tier.

        The ordering only implies pass-up (strict passes -> lenient passes).
        A lenient tier passing says nothing about the strict tier.

        **Validates: Requirements 4.1, 4.3**
        """
        strict_tier, lenient_tier, _ = tier_pair
        tiers = [strict_tier, lenient_tier]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()

        brackets = [BracketState(tier=t) for t in tiers]

        detector.propagate_pass(
            passed_tier_idx=1, concurrency=concurrency, brackets=brackets
        )

        # Strict tier should NOT be updated by lenient tier passing
        assert brackets[0].feasible_max is None

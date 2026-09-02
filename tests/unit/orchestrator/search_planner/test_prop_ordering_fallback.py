# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Ordering Fallback on Contradiction.

Feature: multi-tier-slo-search, Property 9: Ordering Fallback on Contradiction

Validates: Requirements 4.4, 10.4

For any tier pair where ordering inference produces a contradictory result
(strict tier's observed boundary > lenient tier's observed boundary), the
planner SHALL disable ordering inference for that pair and continue with
independent bracket tracking.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.config.sweep.adaptive import SLAFilter, SLOTier
from aiperf.orchestrator.search_planner.multi_tier_models import BracketState
from aiperf.orchestrator.search_planner.multi_tier_ordering import TierOrderingDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter(metric: str, stat: str, op: str, threshold: float) -> SLAFilter:
    return SLAFilter(metric_tag=metric, stat=stat, op=op, threshold=threshold)


def _tier(label: str, filters: list[SLAFilter]) -> SLOTier:
    return SLOTier(label=label, filters=filters)


def _brackets(tiers: list[SLOTier]) -> list[BracketState]:
    return [BracketState(tier=t) for t in tiers]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Threshold pairs where strict > lenient (for gt/ge operators)
_gt_threshold_pair = st.tuples(
    st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
).filter(lambda t: t[0] > t[1])

# Threshold pairs where strict < lenient (for lt/le operators)
_lt_threshold_pair = st.tuples(
    st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
).filter(lambda t: t[0] < t[1])

_op_and_thresholds = st.one_of(
    st.tuples(st.sampled_from(["gt", "ge"]), _gt_threshold_pair),
    st.tuples(st.sampled_from(["lt", "le"]), _lt_threshold_pair),
)

# Concurrency values in a realistic range
_concurrency = st.integers(min_value=1, max_value=512)


# ---------------------------------------------------------------------------
# Property 9: Ordering Fallback on Contradiction
# ---------------------------------------------------------------------------


class TestProperty9OrderingFallbackOnContradiction:
    """Property 9: Ordering Fallback on Contradiction.

    **Validates: Requirements 4.4, 10.4**
    """

    @given(
        op_thresholds=_op_and_thresholds,
        strict_feasible_max=_concurrency,
        fail_concurrency=_concurrency,
    )
    @settings(max_examples=100, deadline=None)
    def test_fail_down_contradiction_disables_pair(
        self,
        op_thresholds: tuple[str, tuple[float, float]],
        strict_feasible_max: int,
        fail_concurrency: int,
    ) -> None:
        """Fail-down contradiction: strict.feasible_max >= new infeasible_min disables pair.

        When strict tier already has feasible_max >= the concurrency where
        lenient fails, propagating that failure creates a contradiction
        (strict.feasible_max >= strict.infeasible_min). The pair is disabled.

        **Validates: Requirements 4.4, 10.4**
        """
        op, (strict_thresh, lenient_thresh) = op_thresholds

        # Ensure the fail-down produces a contradiction:
        # strict.feasible_max >= fail_concurrency (new infeasible_min for strict)
        if strict_feasible_max < fail_concurrency:
            strict_feasible_max, fail_concurrency = (
                fail_concurrency,
                strict_feasible_max,
            )
        # Ensure they actually produce the contradiction condition
        if strict_feasible_max < fail_concurrency:
            return  # Skip degenerate case where swap didn't help

        strict = _tier("strict", [_filter("metric", "avg", op, strict_thresh)])
        lenient = _tier("lenient", [_filter("metric", "avg", op, lenient_thresh)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()

        assert (0, 1) in detector.ordered_pairs

        brackets = _brackets(tiers)
        brackets[0].feasible_max = strict_feasible_max

        detector.propagate_fail(
            failed_tier_idx=1, concurrency=fail_concurrency, brackets=brackets
        )

        # Contradiction detected: pair is disabled
        assert (0, 1) in detector.disabled_pairs

    @given(
        op_thresholds=_op_and_thresholds,
        lenient_infeasible_min=_concurrency,
        pass_concurrency=_concurrency,
    )
    @settings(max_examples=100, deadline=None)
    def test_pass_up_contradiction_disables_pair(
        self,
        op_thresholds: tuple[str, tuple[float, float]],
        lenient_infeasible_min: int,
        pass_concurrency: int,
    ) -> None:
        """Pass-up contradiction: new feasible_max >= lenient.infeasible_min disables pair.

        When lenient tier already has infeasible_min <= the concurrency where
        strict passes, propagating that pass creates a contradiction
        (lenient.feasible_max >= lenient.infeasible_min). The pair is disabled.

        **Validates: Requirements 4.4, 10.4**
        """
        op, (strict_thresh, lenient_thresh) = op_thresholds

        # Ensure the pass-up produces a contradiction:
        # pass_concurrency >= lenient_infeasible_min (so new feasible_max >= infeasible_min)
        if pass_concurrency < lenient_infeasible_min:
            pass_concurrency, lenient_infeasible_min = (
                lenient_infeasible_min,
                pass_concurrency,
            )
        if pass_concurrency < lenient_infeasible_min:
            return  # Skip degenerate case

        strict = _tier("strict", [_filter("metric", "avg", op, strict_thresh)])
        lenient = _tier("lenient", [_filter("metric", "avg", op, lenient_thresh)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()

        assert (0, 1) in detector.ordered_pairs

        brackets = _brackets(tiers)
        brackets[1].infeasible_min = lenient_infeasible_min

        detector.propagate_pass(
            passed_tier_idx=0, concurrency=pass_concurrency, brackets=brackets
        )

        # Contradiction detected: pair is disabled
        assert (0, 1) in detector.disabled_pairs

    @given(
        op_thresholds=_op_and_thresholds,
        strict_feasible_max=_concurrency,
        fail_concurrency=_concurrency,
        subsequent_fail=_concurrency,
        subsequent_pass=_concurrency,
    )
    @settings(max_examples=100, deadline=None)
    def test_disabled_pair_propagation_is_noop(
        self,
        op_thresholds: tuple[str, tuple[float, float]],
        strict_feasible_max: int,
        fail_concurrency: int,
        subsequent_fail: int,
        subsequent_pass: int,
    ) -> None:
        """After disabling, subsequent propagations for the pair are no-ops.

        Once a contradiction disables ordering inference for a pair,
        further propagate_fail and propagate_pass calls must not modify
        brackets for the affected pair.

        **Validates: Requirements 4.4, 10.4**
        """
        op, (strict_thresh, lenient_thresh) = op_thresholds

        # Set up contradiction via fail-down
        if strict_feasible_max < fail_concurrency:
            strict_feasible_max, fail_concurrency = (
                fail_concurrency,
                strict_feasible_max,
            )
        if strict_feasible_max < fail_concurrency:
            return

        strict = _tier("strict", [_filter("metric", "avg", op, strict_thresh)])
        lenient = _tier("lenient", [_filter("metric", "avg", op, lenient_thresh)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)
        brackets[0].feasible_max = strict_feasible_max

        # Trigger contradiction
        detector.propagate_fail(
            failed_tier_idx=1, concurrency=fail_concurrency, brackets=brackets
        )
        assert (0, 1) in detector.disabled_pairs

        # Snapshot bracket state after contradiction
        strict_infeasible_after = brackets[0].infeasible_min
        lenient_feasible_after = brackets[1].feasible_max

        # Subsequent fail-down: strict tier's infeasible_min should not change
        detector.propagate_fail(
            failed_tier_idx=1, concurrency=subsequent_fail, brackets=brackets
        )
        assert brackets[0].infeasible_min == strict_infeasible_after

        # Subsequent pass-up: lenient tier's feasible_max should not change
        detector.propagate_pass(
            passed_tier_idx=0, concurrency=subsequent_pass, brackets=brackets
        )
        assert brackets[1].feasible_max == lenient_feasible_after

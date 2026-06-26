# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TierOrderingDetector."""

from __future__ import annotations

import logging

from aiperf.config.sweep.adaptive import SLAFilter, SLOTier
from aiperf.orchestrator.search_planner.multi_tier_models import BracketState
from aiperf.orchestrator.search_planner.multi_tier_ordering import TierOrderingDetector


def _tier(label: str, filters: list[SLAFilter]) -> SLOTier:
    return SLOTier(label=label, filters=filters)


def _filter(metric: str, stat: str, op: str, threshold: float) -> SLAFilter:
    return SLAFilter(metric_tag=metric, stat=stat, op=op, threshold=threshold)


def _brackets(tiers: list[SLOTier]) -> list[BracketState]:
    return [BracketState(tier=t) for t in tiers]


class TestDetectOrdering:
    """Tests for detect_ordering() logic."""

    def test_detect_ordering_gt_two_tiers(self):
        """Higher gt threshold is harder, so tier with gt:300 is stricter than gt:100."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        detector = TierOrderingDetector([strict, lenient])

        pairs = detector.detect_ordering()

        assert (0, 1) in pairs
        assert (1, 0) not in pairs

    def test_detect_ordering_lt_two_tiers(self):
        """Lower lt threshold is harder, so tier with lt:100 is stricter than lt:500."""
        strict = _tier("fast", [_filter("latency", "p95", "lt", 100.0)])
        lenient = _tier("standard", [_filter("latency", "p95", "lt", 500.0)])
        detector = TierOrderingDetector([strict, lenient])

        pairs = detector.detect_ordering()

        assert (0, 1) in pairs
        assert (1, 0) not in pairs

    def test_detect_ordering_multiple_filters(self):
        """Ordering detected when ALL filters in a tier are stricter."""
        strict = _tier(
            "fast",
            [
                _filter("throughput", "avg", "gt", 300.0),
                _filter("latency", "p95", "lt", 100.0),
            ],
        )
        lenient = _tier(
            "standard",
            [
                _filter("throughput", "avg", "gt", 100.0),
                _filter("latency", "p95", "lt", 500.0),
            ],
        )
        detector = TierOrderingDetector([strict, lenient])

        pairs = detector.detect_ordering()

        assert (0, 1) in pairs

    def test_no_ordering_when_different_metrics(self):
        """No ordering when tiers have different (metric_tag, stat) sets."""
        tier_a = _tier("a", [_filter("throughput", "avg", "gt", 300.0)])
        tier_b = _tier("b", [_filter("latency", "p95", "lt", 500.0)])
        detector = TierOrderingDetector([tier_a, tier_b])

        pairs = detector.detect_ordering()

        assert pairs == []

    def test_no_ordering_when_different_operators(self):
        """No ordering when filters have different operators on same metric."""
        tier_a = _tier("a", [_filter("throughput", "avg", "gt", 300.0)])
        tier_b = _tier("b", [_filter("throughput", "avg", "lt", 100.0)])
        detector = TierOrderingDetector([tier_a, tier_b])

        pairs = detector.detect_ordering()

        assert pairs == []

    def test_no_ordering_when_equal_thresholds(self):
        """No ordering when thresholds are equal (not strictly harder)."""
        tier_a = _tier("a", [_filter("throughput", "avg", "gt", 100.0)])
        tier_b = _tier("b", [_filter("throughput", "avg", "gt", 100.0)])
        detector = TierOrderingDetector([tier_a, tier_b])

        pairs = detector.detect_ordering()

        assert pairs == []

    def test_no_ordering_mixed_strictness(self):
        """No ordering when one filter is harder but another is easier."""
        tier_a = _tier(
            "a",
            [
                _filter("throughput", "avg", "gt", 300.0),
                _filter("latency", "p95", "lt", 500.0),
            ],
        )
        tier_b = _tier(
            "b",
            [
                _filter("throughput", "avg", "gt", 100.0),
                _filter("latency", "p95", "lt", 100.0),
            ],
        )
        detector = TierOrderingDetector([tier_a, tier_b])

        pairs = detector.detect_ordering()

        assert pairs == []

    def test_detect_ordering_three_tiers_chain(self):
        """Three-tier chain: fast > standard > economy."""
        fast = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        standard = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        economy = _tier("economy", [_filter("throughput", "avg", "gt", 30.0)])
        detector = TierOrderingDetector([fast, standard, economy])

        pairs = detector.detect_ordering()

        assert (0, 1) in pairs
        assert (0, 2) in pairs
        assert (1, 2) in pairs
        assert len(pairs) == 3

    def test_detect_ordering_ge_operator(self):
        """ge operator: higher threshold is harder."""
        strict = _tier("strict", [_filter("throughput", "avg", "ge", 300.0)])
        lenient = _tier("lenient", [_filter("throughput", "avg", "ge", 100.0)])
        detector = TierOrderingDetector([strict, lenient])

        pairs = detector.detect_ordering()

        assert (0, 1) in pairs

    def test_detect_ordering_le_operator(self):
        """le operator: lower threshold is harder."""
        strict = _tier("strict", [_filter("latency", "p95", "le", 100.0)])
        lenient = _tier("lenient", [_filter("latency", "p95", "le", 500.0)])
        detector = TierOrderingDetector([strict, lenient])

        pairs = detector.detect_ordering()

        assert (0, 1) in pairs


class TestPropagateFailDown:
    """Tests for propagate_fail() — fail-down inference."""

    def test_fail_down_updates_infeasible_min(self):
        """When lenient tier fails, strict tier's infeasible_min is updated."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)

        detector.propagate_fail(failed_tier_idx=1, concurrency=64, brackets=brackets)

        assert brackets[0].infeasible_min == 64

    def test_fail_down_uses_min_of_existing(self):
        """fail-down uses min(existing, new) for infeasible_min."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)
        brackets[0].infeasible_min = 32

        detector.propagate_fail(failed_tier_idx=1, concurrency=64, brackets=brackets)

        assert brackets[0].infeasible_min == 32

    def test_fail_down_updates_when_new_is_lower(self):
        """fail-down overwrites when new concurrency is lower."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)
        brackets[0].infeasible_min = 128

        detector.propagate_fail(failed_tier_idx=1, concurrency=64, brackets=brackets)

        assert brackets[0].infeasible_min == 64

    def test_fail_down_skips_converged_tier(self):
        """fail-down does not update a converged tier."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)
        brackets[0].converged = True

        detector.propagate_fail(failed_tier_idx=1, concurrency=64, brackets=brackets)

        assert brackets[0].infeasible_min is None

    def test_fail_down_no_effect_when_strict_fails(self):
        """Failing the strict tier does not propagate to the lenient tier."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)

        detector.propagate_fail(failed_tier_idx=0, concurrency=64, brackets=brackets)

        assert brackets[1].infeasible_min is None


class TestPropagatePassUp:
    """Tests for propagate_pass() — pass-up inference."""

    def test_pass_up_updates_feasible_max(self):
        """When strict tier passes, lenient tier's feasible_max is updated."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)

        detector.propagate_pass(passed_tier_idx=0, concurrency=32, brackets=brackets)

        assert brackets[1].feasible_max == 32

    def test_pass_up_uses_max_of_existing(self):
        """pass-up uses max(existing, new) for feasible_max."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)
        brackets[1].feasible_max = 64

        detector.propagate_pass(passed_tier_idx=0, concurrency=32, brackets=brackets)

        assert brackets[1].feasible_max == 64

    def test_pass_up_updates_when_new_is_higher(self):
        """pass-up overwrites when new concurrency is higher."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)
        brackets[1].feasible_max = 16

        detector.propagate_pass(passed_tier_idx=0, concurrency=32, brackets=brackets)

        assert brackets[1].feasible_max == 32

    def test_pass_up_skips_converged_tier(self):
        """pass-up does not update a converged tier."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)
        brackets[1].converged = True

        detector.propagate_pass(passed_tier_idx=0, concurrency=32, brackets=brackets)

        assert brackets[1].feasible_max is None

    def test_pass_up_no_effect_when_lenient_passes(self):
        """Passing the lenient tier does not propagate to the strict tier."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)

        detector.propagate_pass(passed_tier_idx=1, concurrency=128, brackets=brackets)

        assert brackets[0].feasible_max is None


class TestDisablePair:
    """Tests for disable_pair() and contradiction handling."""

    def test_disable_pair_stops_propagation(self):
        """After disabling a pair, propagation no longer applies."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)

        detector.disable_pair(0, 1)
        detector.propagate_fail(failed_tier_idx=1, concurrency=64, brackets=brackets)
        detector.propagate_pass(passed_tier_idx=0, concurrency=32, brackets=brackets)

        assert brackets[0].infeasible_min is None
        assert brackets[1].feasible_max is None

    def test_disable_pair_logs_warning(self, caplog):
        """disable_pair logs a warning about the contradiction."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        detector = TierOrderingDetector([strict, lenient])
        detector.detect_ordering()

        with caplog.at_level(logging.WARNING):
            detector.disable_pair(0, 1)

        assert "contradiction" in caplog.text.lower()

    def test_contradiction_auto_detected_on_fail_down(self):
        """Contradiction is detected and pair disabled when fail-down causes it."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)

        # Strict tier already has a high feasible_max
        brackets[0].feasible_max = 100
        # Lenient tier failing below the strict tier's feasible_max is a contradiction
        brackets[1].infeasible_min = 80

        detector.propagate_fail(failed_tier_idx=1, concurrency=50, brackets=brackets)

        # infeasible_min was set but the contradiction was also detected
        assert brackets[0].infeasible_min == 50
        assert (0, 1) in detector.disabled_pairs

    def test_contradiction_auto_detected_on_pass_up(self):
        """Contradiction is detected and pair disabled when pass-up causes it."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()
        brackets = _brackets(tiers)

        # Lenient tier has a low infeasible_min
        brackets[1].infeasible_min = 30

        # Strict tier passes at a concurrency higher than lenient's infeasible_min
        detector.propagate_pass(passed_tier_idx=0, concurrency=50, brackets=brackets)

        assert brackets[1].feasible_max == 50
        assert (0, 1) in detector.disabled_pairs

    def test_ordered_pairs_excludes_disabled(self):
        """The ordered_pairs property excludes disabled pairs."""
        strict = _tier("fast", [_filter("throughput", "avg", "gt", 300.0)])
        lenient = _tier("standard", [_filter("throughput", "avg", "gt", 100.0)])
        tiers = [strict, lenient]
        detector = TierOrderingDetector(tiers)
        detector.detect_ordering()

        assert len(detector.ordered_pairs) == 1

        detector.disable_pair(0, 1)

        assert len(detector.ordered_pairs) == 0

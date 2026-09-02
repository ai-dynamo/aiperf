# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier ordering detection and inference propagation for multi-tier SLO search.

Detects monotonic relationships between SLO tiers (stricter tiers have lower
boundaries) and propagates bracket bound inferences to eliminate redundant probes.
"""

from __future__ import annotations

import logging

from aiperf.config.sweep.adaptive import SLOTier
from aiperf.orchestrator.search_planner.multi_tier_models import BracketState

logger = logging.getLogger(__name__)

# Operators where a higher threshold is "harder" (stricter).
_HIGHER_IS_HARDER: frozenset[str] = frozenset({"gt", "ge"})
# Operators where a lower threshold is "harder" (stricter).
_LOWER_IS_HARDER: frozenset[str] = frozenset({"lt", "le"})


class TierOrderingDetector:
    """Detects and exploits monotonic ordering between SLO tiers."""

    def __init__(self, tiers: list[SLOTier]) -> None:
        self._tiers = tiers
        self._ordered_pairs: list[tuple[int, int]] = []
        self._disabled_pairs: set[tuple[int, int]] = set()

    def detect_ordering(self) -> list[tuple[int, int]]:
        """Return pairs (strict_idx, lenient_idx) for detected orderings.

        A tier A is strictly harder than tier B if for every filter in A
        there exists a corresponding filter in B on the same metric_tag:stat
        and A's threshold is harder (for gt/ge: higher threshold is harder;
        for lt/le: lower threshold is harder).

        Both tiers must share the exact same set of (metric_tag, stat) pairs.
        """
        self._ordered_pairs = []
        n = len(self._tiers)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._is_strictly_harder(i, j):
                    self._ordered_pairs.append((i, j))

        return list(self._ordered_pairs)

    def propagate_fail(
        self,
        failed_tier_idx: int,
        concurrency: int,
        brackets: list[BracketState],
    ) -> None:
        """Propagate failure inference downward.

        If lenient tier B fails at X, strict tier A also fails at X.
        Update: A.infeasible_min = min(A.infeasible_min, X).
        """
        for strict_idx, lenient_idx in self._ordered_pairs:
            if (strict_idx, lenient_idx) in self._disabled_pairs:
                continue
            if lenient_idx != failed_tier_idx:
                continue

            bracket = brackets[strict_idx]
            if bracket.converged:
                continue

            new_min = concurrency
            old_min = bracket.infeasible_min

            if old_min is None or new_min < old_min:
                bracket.infeasible_min = new_min
                logger.debug(
                    "Ordering fail-down: tier %d failed at %d, updated tier %d "
                    "infeasible_min from %s to %d",
                    failed_tier_idx,
                    concurrency,
                    strict_idx,
                    old_min,
                    new_min,
                )
                self._check_contradiction(strict_idx, lenient_idx, brackets)

    def propagate_pass(
        self,
        passed_tier_idx: int,
        concurrency: int,
        brackets: list[BracketState],
    ) -> None:
        """Propagate pass inference upward.

        If strict tier A passes at X, lenient tier B also passes at X.
        Update: B.feasible_max = max(B.feasible_max, X).
        """
        for strict_idx, lenient_idx in self._ordered_pairs:
            if (strict_idx, lenient_idx) in self._disabled_pairs:
                continue
            if strict_idx != passed_tier_idx:
                continue

            bracket = brackets[lenient_idx]
            if bracket.converged:
                continue

            new_max = concurrency
            old_max = bracket.feasible_max

            if old_max is None or new_max > old_max:
                bracket.feasible_max = new_max
                logger.debug(
                    "Ordering pass-up: tier %d passed at %d, updated tier %d "
                    "feasible_max from %s to %d",
                    passed_tier_idx,
                    concurrency,
                    lenient_idx,
                    old_max,
                    new_max,
                )
                self._check_contradiction(strict_idx, lenient_idx, brackets)

    def disable_pair(self, strict_idx: int, lenient_idx: int) -> None:
        """Disable ordering inference for a specific tier pair."""
        self._disabled_pairs.add((strict_idx, lenient_idx))
        logger.warning(
            "Ordering contradiction: disabling inference between tier %d (strict) "
            "and tier %d (lenient)",
            strict_idx,
            lenient_idx,
        )

    @property
    def ordered_pairs(self) -> list[tuple[int, int]]:
        """Active ordered pairs (excluding disabled ones)."""
        return [p for p in self._ordered_pairs if p not in self._disabled_pairs]

    @property
    def disabled_pairs(self) -> set[tuple[int, int]]:
        """Set of disabled tier pairs."""
        return set(self._disabled_pairs)

    def _is_strictly_harder(self, tier_a_idx: int, tier_b_idx: int) -> bool:
        """Check if tier A is strictly harder than tier B on every dimension."""
        tier_a = self._tiers[tier_a_idx]
        tier_b = self._tiers[tier_b_idx]

        keys_a = {(f.metric_tag, f.stat) for f in tier_a.filters}
        keys_b = {(f.metric_tag, f.stat) for f in tier_b.filters}

        if keys_a != keys_b:
            return False

        filters_b_map = {(f.metric_tag, f.stat): f for f in tier_b.filters}

        for fa in tier_a.filters:
            fb = filters_b_map[(fa.metric_tag, fa.stat)]

            if fa.op != fb.op:
                return False

            if fa.op in _HIGHER_IS_HARDER:
                if fa.threshold <= fb.threshold:
                    return False
            elif fa.op in _LOWER_IS_HARDER:
                if fa.threshold >= fb.threshold:
                    return False
            else:
                return False

        return True

    def _check_contradiction(
        self,
        strict_idx: int,
        lenient_idx: int,
        brackets: list[BracketState],
    ) -> None:
        """Check if the bracket bounds contradict the ordering assumption.

        A contradiction occurs when evidence suggests the strict tier's
        boundary is higher than the lenient tier's boundary. Two signals:
        1. strict.feasible_max >= lenient.infeasible_min
        2. lenient.feasible_max >= lenient.infeasible_min (inferred pass-up
           created an impossible bracket for the lenient tier)
        3. strict.feasible_max >= strict.infeasible_min (inferred fail-down
           created an impossible bracket for the strict tier)
        """
        strict_bracket = brackets[strict_idx]
        lenient_bracket = brackets[lenient_idx]

        s_max = strict_bracket.feasible_max
        s_min = strict_bracket.infeasible_min
        l_max = lenient_bracket.feasible_max
        l_min = lenient_bracket.infeasible_min

        contradiction = (
            (s_max is not None and l_min is not None and s_max >= l_min)
            or (l_max is not None and l_min is not None and l_max >= l_min)
            or (s_max is not None and s_min is not None and s_max >= s_min)
        )

        if contradiction:
            self.disable_pair(strict_idx, lenient_idx)

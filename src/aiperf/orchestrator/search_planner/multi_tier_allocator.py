# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe allocation for multi-tier SLO boundary search."""

from __future__ import annotations

from aiperf.orchestrator.search_planner.multi_tier_models import BracketState


class ProbeAllocator:
    """Selects the next probe concurrency targeting the widest bracket."""

    def select_next_probe(self, brackets: list[BracketState]) -> int | None:
        """Select the concurrency to probe next.

        Algorithm:
        1. Filter to non-converged tiers with both bounds established.
        2. For each, compute bracket gap = infeasible_min - feasible_max.
        3. Select the tier with the widest gap.
        4. If gap <= 1, mark converged and recurse to next widest.
        5. Return the midpoint of that tier's bracket.

        Returns None when all tiers are converged or no candidates have
        both bounds established.
        """
        candidates = [
            b
            for b in brackets
            if not b.converged
            and b.feasible_max is not None
            and b.infeasible_min is not None
            and b.infeasible_min > b.feasible_max  # exclude inverted brackets
        ]
        if not candidates:
            return None

        widest = max(candidates, key=lambda b: b.infeasible_min - b.feasible_max)
        gap = widest.infeasible_min - widest.feasible_max
        if gap <= 1:
            widest.converged = True
            widest.convergence_reason = "multi_tier_precision_reached"
            return self.select_next_probe(brackets)

        midpoint = widest.feasible_max + gap // 2
        return midpoint

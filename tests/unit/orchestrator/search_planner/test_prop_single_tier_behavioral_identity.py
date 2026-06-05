# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Single-Tier Behavioral Identity.

Feature: multi-tier-slo-search, Property 13: Single-Tier Behavioral Identity

Validates: Requirements 7.1, 1.2

For any configuration with a single set of --search-sla filters and no
--search-sla-tier grouping, the planner SHALL produce the same probe sequence,
verdicts, and boundary result as the existing single-tier planner
(SmoothIsotonicSLAPlanner or MonotonicSLASearchPlanner).

The design decision: MultiTierPlanner is NOT instantiated when only a single
tier is configured. The existing planners run unmodified. This property test
verifies:
1. The dispatch logic correctly prevents MultiTierPlanner activation for
   single-tier configs (0 or 1 tier entries).
2. The existing single-tier planner produces deterministic, consistent probe
   sequences for the same configuration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner
from aiperf.plugin.enums import SearchPlannerType

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _sla_filter_st(draw: st.DrawFn) -> SLAFilter:
    """Generate a random valid SLAFilter."""
    metric = draw(
        st.sampled_from(
            ["time_to_first_token", "output_token_throughput", "e2e_latency"]
        )
    )
    stat = draw(st.sampled_from(["avg", "p95", "p99"]))
    op = draw(st.sampled_from(["lt", "gt"]))
    threshold = draw(st.floats(min_value=1.0, max_value=10000.0, allow_nan=False))
    return SLAFilter(metric_tag=metric, stat=stat, op=op, threshold=threshold)


@st.composite
def _single_tier_config_st(draw: st.DrawFn) -> dict:
    """Generate a valid single-tier search configuration.

    Produces configs with 0 or 1 tiers in sla_tiers, which must NOT
    activate the MultiTierPlanner.
    """
    lo = draw(st.integers(min_value=1, max_value=10))
    hi = draw(st.integers(min_value=lo + 10, max_value=500))
    planner_type = draw(
        st.sampled_from(
            [SearchPlannerType.SMOOTH_ISOTONIC, SearchPlannerType.MONOTONIC_SLA]
        )
    )
    n_filters = draw(st.integers(min_value=1, max_value=3))
    filters = [draw(_sla_filter_st()) for _ in range(n_filters)]
    tier_count = draw(st.sampled_from([0, 1]))
    sla_tiers: list[SLOTier] = []
    if tier_count == 1:
        sla_tiers = [SLOTier(label="only", filters=filters)]

    return {
        "lo": lo,
        "hi": hi,
        "planner_type": planner_type,
        "filters": filters,
        "sla_tiers": sla_tiers,
    }


def _base_config() -> BenchmarkConfig:
    """Create a real BenchmarkConfig suitable for planner ask() calls."""
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "profiling", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 10,
                }
            ],
        }
    )


def _make_plan(cfg_dict: dict) -> MagicMock:
    """Create a mock BenchmarkPlan from the generated config."""
    plan = MagicMock()
    plan.sweep = AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency",
                lo=cfg_dict["lo"],
                hi=cfg_dict["hi"],
                kind="int",
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        planner=cfg_dict["planner_type"],
        max_iterations=20,
        n_initial_points=2,
        sla_filters=cfg_dict["filters"],
        sla_tiers=cfg_dict["sla_tiers"],
    )
    plan.configs = [_base_config()]
    return plan


# ---------------------------------------------------------------------------
# Property 13: Single-Tier Behavioral Identity
# ---------------------------------------------------------------------------


class TestProperty13SingleTierBehavioralIdentity:
    """Property 13: Single-Tier Behavioral Identity.

    **Validates: Requirements 7.1, 1.2**
    """

    @given(cfg=_single_tier_config_st())
    @settings(max_examples=100, deadline=None)
    def test_single_tier_dispatch_never_returns_multi_tier_planner(
        self,
        cfg: dict,
    ) -> None:
        """Single-tier configs (0 or 1 tier entries) never activate MultiTierPlanner.

        For any configuration with no --search-sla-tier grouping or a single
        tier entry, _build_search_planner SHALL return a single-tier planner
        instance and never a MultiTierPlanner.

        **Validates: Requirements 7.1, 1.2**
        """
        from aiperf.cli_runner._strategy import _build_search_planner

        plan = _make_plan(cfg)
        planner = _build_search_planner(plan)

        assert planner is not None
        assert not isinstance(planner, MultiTierPlanner)

    @given(cfg=_single_tier_config_st())
    @settings(max_examples=100, deadline=None)
    def test_single_tier_dispatch_returns_correct_planner_type(
        self,
        cfg: dict,
    ) -> None:
        """Dispatch returns the planner type matching the config's planner field.

        For smooth_isotonic configs, returns SmoothIsotonicSLAPlanner.
        For monotonic_sla configs, returns MonotonicSLASearchPlanner.

        **Validates: Requirements 7.1, 1.2**
        """
        from aiperf.cli_runner._strategy import _build_search_planner
        from aiperf.orchestrator.search_planner.monotonic import (
            MonotonicSLASearchPlanner,
        )
        from aiperf.orchestrator.search_planner.smooth_isotonic import (
            SmoothIsotonicSLAPlanner,
        )

        plan = _make_plan(cfg)
        planner = _build_search_planner(plan)

        if cfg["planner_type"] == SearchPlannerType.SMOOTH_ISOTONIC:
            assert isinstance(planner, SmoothIsotonicSLAPlanner)
        else:
            assert isinstance(planner, MonotonicSLASearchPlanner)

    @given(cfg=_single_tier_config_st())
    @settings(max_examples=100, deadline=None)
    def test_single_tier_planner_produces_deterministic_first_probe(
        self,
        cfg: dict,
    ) -> None:
        """The single-tier planner produces a deterministic first probe value.

        Two instances created from the same config SHALL produce the same
        first ask() result, confirming behavioral identity across
        instantiation.

        **Validates: Requirements 7.1, 1.2**
        """
        from aiperf.cli_runner._strategy import _build_search_planner

        plan1 = _make_plan(cfg)
        plan2 = _make_plan(cfg)
        planner1 = _build_search_planner(plan1)
        planner2 = _build_search_planner(plan2)

        result1 = planner1.ask()
        result2 = planner2.ask()

        assert result1 is not None
        assert result2 is not None

        _, variation1 = result1
        _, variation2 = result2

        # Same probe values for same config
        assert variation1.values == variation2.values

    @given(cfg=_single_tier_config_st())
    @settings(max_examples=100, deadline=None)
    def test_single_tier_planner_first_probe_starts_at_lo(
        self,
        cfg: dict,
    ) -> None:
        """The single-tier planner always starts probing at the lo bound.

        Both smooth_isotonic and monotonic planners use exponential ramp
        starting from the configured lo value. This is the fundamental
        behavioral identity that remains unchanged by multi-tier code.

        **Validates: Requirements 7.1, 1.2**
        """
        from aiperf.cli_runner._strategy import _build_search_planner

        plan = _make_plan(cfg)
        planner = _build_search_planner(plan)

        result = planner.ask()
        assert result is not None
        _, variation = result

        swept_values = list(variation.values.values())
        assert len(swept_values) == 1
        assert swept_values[0] == cfg["lo"]

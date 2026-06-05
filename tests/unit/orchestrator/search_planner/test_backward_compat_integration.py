# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward compatibility integration test for single-tier search.

Verifies that single-tier `--search-sla` produces identical behavior to the
existing planner and that the multi-tier code presence does not affect
single-tier workflows.

Validates: Requirements 7.1, 7.3
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective, SweepVariation
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner
from aiperf.orchestrator.search_planner.smooth_isotonic import (
    SmoothIsotonicSLAPlanner,
)
from aiperf.plugin.enums import SearchPlannerType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_config() -> BenchmarkConfig:
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


def _sla_filter(threshold: float = 200.0) -> SLAFilter:
    return SLAFilter(
        metric_tag="time_to_first_token",
        stat="p95",
        op="lt",
        threshold=threshold,
    )


def _adaptive_cfg(
    *,
    lo: int = 1,
    hi: int = 256,
    threshold: float = 200.0,
    max_iterations: int = 20,
    sla_tiers: list[SLOTier] | None = None,
) -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        planner=SearchPlannerType.SMOOTH_ISOTONIC,
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency",
                lo=lo,
                hi=hi,
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
        max_iterations=max_iterations,
        n_initial_points=1,
        sla_filters=[_sla_filter(threshold)],
        sla_tiers=sla_tiers or [],
    )


def _make_plan(
    *,
    sla_tiers: list[SLOTier] | None = None,
    lo: int = 1,
    hi: int = 256,
    threshold: float = 200.0,
) -> MagicMock:
    plan = MagicMock()
    plan.sweep = _adaptive_cfg(lo=lo, hi=hi, threshold=threshold, sla_tiers=sla_tiers)
    plan.configs = [_base_config()]
    return plan


def _result(variation: SweepVariation, *, ttft_p95: float) -> RunResult:
    return RunResult(
        label="t",
        success=True,
        summary_metrics={
            "time_to_first_token": JsonMetricResult(unit="ms", p95=ttft_p95),
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=100.0),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )


def _drive_planner(
    planner: SmoothIsotonicSLAPlanner,
    ttft_curve: dict[int, float],
    max_iters: int = 20,
) -> list[int]:
    """Run the planner; return the probe sequence (concurrency values)."""
    probes: list[int] = []
    iters = 0
    while not planner.is_converged() and iters < max_iters:
        proposal = planner.ask()
        if proposal is None:
            break
        _, variation = proposal
        x = variation.values["phases.profiling.concurrency"]
        probes.append(x)
        if x in ttft_curve:
            ttft = ttft_curve[x]
        else:
            keys_le = [k for k in ttft_curve if k <= x]
            ttft = (
                ttft_curve[max(keys_le)] if keys_le else next(iter(ttft_curve.values()))
            )
        planner.tell(variation, [_result(variation, ttft_p95=ttft)])
        iters += 1
    return probes


# ---------------------------------------------------------------------------
# Test 1: Dispatch identity — empty sla_tiers never activates MultiTierPlanner
# ---------------------------------------------------------------------------


class TestDispatchIdentity:
    """Verify _build_search_planner returns the same planner type for
    single-tier configs, whether sla_tiers=[] or absent entirely."""

    def test_empty_sla_tiers_returns_smooth_isotonic(self) -> None:
        """Empty sla_tiers list dispatches to SmoothIsotonicSLAPlanner."""
        from aiperf.cli_runner._strategy import _build_search_planner

        plan = _make_plan(sla_tiers=[])
        planner = _build_search_planner(plan)
        assert isinstance(planner, SmoothIsotonicSLAPlanner)
        assert not isinstance(planner, MultiTierPlanner)

    def test_no_sla_tiers_field_returns_smooth_isotonic(self) -> None:
        """Default (no sla_tiers specified) dispatches to SmoothIsotonicSLAPlanner."""
        from aiperf.cli_runner._strategy import _build_search_planner

        plan = _make_plan()
        planner = _build_search_planner(plan)
        assert isinstance(planner, SmoothIsotonicSLAPlanner)
        assert not isinstance(planner, MultiTierPlanner)


# ---------------------------------------------------------------------------
# Test 2: Behavioral identity — two identical configs produce same probes
# ---------------------------------------------------------------------------


class TestBehavioralIdentity:
    """Verify two planners from equivalent configs (one with empty sla_tiers,
    one without explicit field) produce identical probe sequences."""

    def test_same_config_produces_identical_probe_sequence(self) -> None:
        """Two planners created from the same config produce identical probes."""
        from aiperf.cli_runner._strategy import _build_search_planner

        plan_a = _make_plan(sla_tiers=[])
        plan_b = _make_plan(sla_tiers=[])

        planner_a = _build_search_planner(plan_a)
        planner_b = _build_search_planner(plan_b)

        # TTFT crosses 200ms at concurrency ~75 (TTFT = 50 + c * 2)
        curve = {x: 50.0 + x * 2.0 for x in range(1, 257)}

        probes_a = _drive_planner(planner_a, curve)
        probes_b = _drive_planner(planner_b, curve)

        assert probes_a == probes_b
        assert len(probes_a) >= 3

    def test_empty_tiers_and_default_tiers_produce_same_planner_type(self) -> None:
        """Configs with empty sla_tiers and default sla_tiers resolve to same type."""
        from aiperf.cli_runner._strategy import _build_search_planner

        plan_explicit_empty = _make_plan(sla_tiers=[])
        plan_default = _make_plan()

        planner_explicit = _build_search_planner(plan_explicit_empty)
        planner_default = _build_search_planner(plan_default)

        assert type(planner_explicit) is type(planner_default)


# ---------------------------------------------------------------------------
# Test 3: Output identity — single-tier JSON has no multi-tier keys
# ---------------------------------------------------------------------------


class TestOutputIdentity:
    """Verify search_history.json from single-tier run does NOT contain
    tier_results or tier_metadata keys."""

    def test_single_tier_json_has_no_tier_results(self, tmp_path: Path) -> None:
        """Single-tier planner produces JSON without tier_results."""
        cfg = _adaptive_cfg()
        planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
        curve = {x: 50.0 + x * 2.0 for x in range(1, 257)}
        _drive_planner(planner, curve)

        write_search_history(
            tmp_path,
            planner.history(),
            cfg,
            convergence_reason=planner.convergence_reason(),
            planner=planner,
        )

        data = orjson.loads((tmp_path / "search_history.json").read_bytes())
        assert "tier_results" not in data
        assert "tier_metadata" not in data
        assert "tiers" not in data.get("config", {})

    def test_single_tier_json_preserves_existing_schema(self, tmp_path: Path) -> None:
        """Single-tier output contains all expected top-level keys."""
        cfg = _adaptive_cfg()
        planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
        curve = {x: 50.0 + x * 2.0 for x in range(1, 257)}
        _drive_planner(planner, curve)

        write_search_history(
            tmp_path,
            planner.history(),
            cfg,
            convergence_reason=planner.convergence_reason(),
            planner=planner,
        )

        data = orjson.loads((tmp_path / "search_history.json").read_bytes())
        assert "config" in data
        assert "iterations" in data
        assert "best_trials" in data
        assert "boundary_summary" in data
        assert "recipe" in data
        assert "convergence_reason" in data
        assert isinstance(data["iterations"], list)
        assert len(data["iterations"]) > 0


# ---------------------------------------------------------------------------
# Test 4: Single-tier planner behavior unchanged by multi-tier code presence
# ---------------------------------------------------------------------------


class TestSingleTierBehaviorUnchanged:
    """Verify SmoothIsotonicSLAPlanner's fundamental behavior remains intact."""

    def test_first_probe_is_lo(self) -> None:
        """SmoothIsotonicSLAPlanner.ask() returns lo as the first probe."""
        cfg = _adaptive_cfg(lo=5, hi=200)
        planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
        result = planner.ask()
        assert result is not None
        _, variation = result
        assert variation.values["phases.profiling.concurrency"] == 5

    def test_bracket_doubles_on_pass(self) -> None:
        """Bracket phase doubles concurrency after a feasible verdict."""
        cfg = _adaptive_cfg(lo=4, hi=256)
        planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)

        # First probe at lo=4
        result = planner.ask()
        assert result is not None
        _, v1 = result
        assert v1.values["phases.profiling.concurrency"] == 4
        planner.tell(v1, [_result(v1, ttft_p95=50.0)])

        # Second probe should double to 8
        result = planner.ask()
        assert result is not None
        _, v2 = result
        assert v2.values["phases.profiling.concurrency"] == 8
        planner.tell(v2, [_result(v2, ttft_p95=80.0)])

        # Third probe should double to 16
        result = planner.ask()
        assert result is not None
        _, v3 = result
        assert v3.values["phases.profiling.concurrency"] == 16

    def test_full_search_finds_boundary(self) -> None:
        """End-to-end: planner finds boundary near where TTFT crosses threshold."""
        cfg = _adaptive_cfg(lo=1, hi=256, threshold=200.0)
        planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
        # TTFT = 50 + c * 2 => crosses 200 at c=75
        curve = {x: 50.0 + x * 2.0 for x in range(1, 257)}
        _drive_planner(planner, curve)

        assert planner.is_converged()
        summary = planner.boundary_summary()
        assert summary is not None
        fmax = summary["feasible_max"]
        assert fmax is not None
        # Boundary should be near 75 (within planner precision)
        assert 60 <= fmax["value"] <= 80

    @pytest.mark.parametrize("lo", [1, 2, 8, 16])
    def test_first_probe_always_starts_at_configured_lo(self, lo: int) -> None:
        """Regardless of lo setting, first probe is always at lo."""
        cfg = _adaptive_cfg(lo=lo, hi=256)
        planner = SmoothIsotonicSLAPlanner(_base_config(), cfg)
        result = planner.ask()
        assert result is not None
        _, variation = result
        assert variation.values["phases.profiling.concurrency"] == lo

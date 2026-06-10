# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-tier search_history.json output schema extension."""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.search_planner.base import SearchIteration
from aiperf.orchestrator.search_planner.multi_tier_models import TierResult


def _sla_filter(threshold: float = 200.0) -> SLAFilter:
    return SLAFilter(
        metric_tag="output_token_throughput",
        stat="avg",
        op="gt",
        threshold=threshold,
    )


def _cfg_with_tiers() -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=128, kind="int"
            )
        ],
        objectives=[Objective(metric="output_token_throughput", direction="maximize")],
        max_iterations=20,
        sla_filters=[_sla_filter(100.0)],
        sla_tiers=[
            SLOTier(label="fast", filters=[_sla_filter(300.0)]),
            SLOTier(label="standard", filters=[_sla_filter(100.0)]),
        ],
    )


def _cfg_without_tiers() -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=128, kind="int"
            )
        ],
        objectives=[Objective(metric="output_token_throughput", direction="maximize")],
        max_iterations=20,
        sla_filters=[_sla_filter(100.0)],
    )


class _MultiTierPlannerStub:
    """Stub that mimics MultiTierPlanner's interface for the exporter."""

    def __init__(self) -> None:
        self._boundary = {
            "swept_dim_path": "phases.profiling.concurrency",
            "feasible_max": {"value": 64, "iteration_idx": 5, "objective_value": None},
            "infeasible_min": {
                "value": 65,
                "iteration_idx": 6,
                "first_breach": {
                    "metric_tag": "output_token_throughput",
                    "stat": "avg",
                    "op": "gt",
                    "threshold": 100.0,
                    "observed": 95.0,
                },
            },
            "non_monotonic_warning": False,
            "convergence_reason": "multi_tier_all_converged",
        }

    def boundary_summary(self) -> dict | None:
        return self._boundary

    def tier_results(self) -> list[TierResult]:
        return [
            TierResult(
                label="fast",
                boundary_concurrency=32,
                convergence_status="converged",
                convergence_reason="multi_tier_precision_reached",
                binding_constraint={
                    "metric_tag": "output_token_throughput",
                    "stat": "avg",
                    "op": "gt",
                    "threshold": 300.0,
                    "observed": 298.5,
                },
                bracket_lower=32,
                bracket_upper=33,
                confidence_interval=None,
                probe_count=4,
                filters=[
                    {
                        "metric_tag": "output_token_throughput",
                        "stat": "avg",
                        "op": "gt",
                        "threshold": 300.0,
                    }
                ],
            ),
            TierResult(
                label="standard",
                boundary_concurrency=64,
                convergence_status="converged",
                convergence_reason="multi_tier_precision_reached",
                binding_constraint={
                    "metric_tag": "output_token_throughput",
                    "stat": "avg",
                    "op": "gt",
                    "threshold": 100.0,
                    "observed": 95.0,
                },
                bracket_lower=64,
                bracket_upper=65,
                confidence_interval=None,
                probe_count=6,
                filters=[
                    {
                        "metric_tag": "output_token_throughput",
                        "stat": "avg",
                        "op": "gt",
                        "threshold": 100.0,
                    }
                ],
            ),
        ]

    def tier_metadata(self) -> dict:
        return {
            "actual_probe_count": 8,
            "tier_evaluation_count": 10,
            "ordering_detected": True,
            "ordering_pairs": [{"strict": "fast", "lenient": "standard"}],
        }


class _SingleTierPlannerStub:
    """Stub mimicking a single-tier planner (no multi-tier methods)."""

    def boundary_summary(self) -> dict | None:
        return {
            "swept_dim_path": "phases.profiling.concurrency",
            "feasible_max": {"value": 64, "iteration_idx": 3, "objective_value": None},
            "infeasible_min": None,
        }


def _sample_history() -> list[SearchIteration]:
    return [
        SearchIteration(
            iteration_idx=0,
            variation_values={"phases.profiling.concurrency": 16},
            objective_value=350.0,
            objective_values=[350.0],
            feasible=True,
        ),
        SearchIteration(
            iteration_idx=1,
            variation_values={"phases.profiling.concurrency": 64},
            objective_value=150.0,
            objective_values=[150.0],
            feasible=True,
        ),
    ]


def test_multi_tier_adds_tier_results(tmp_path: Path):
    """Multi-tier planner adds tier_results to the output."""
    write_search_history(
        tmp_path,
        _sample_history(),
        _cfg_with_tiers(),
        convergence_reason="multi_tier_all_converged",
        planner=_MultiTierPlannerStub(),
    )
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert "tier_results" in data
    assert len(data["tier_results"]) == 2
    assert data["tier_results"][0]["label"] == "fast"
    assert data["tier_results"][0]["boundary_concurrency"] == 32
    assert data["tier_results"][0]["probe_count"] == 4
    assert data["tier_results"][1]["label"] == "standard"
    assert data["tier_results"][1]["boundary_concurrency"] == 64


def test_multi_tier_adds_tier_metadata(tmp_path: Path):
    """Multi-tier planner adds tier_metadata to the output."""
    write_search_history(
        tmp_path,
        _sample_history(),
        _cfg_with_tiers(),
        convergence_reason="multi_tier_all_converged",
        planner=_MultiTierPlannerStub(),
    )
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert "tier_metadata" in data
    assert data["tier_metadata"]["actual_probe_count"] == 8
    assert data["tier_metadata"]["tier_evaluation_count"] == 10
    assert data["tier_metadata"]["ordering_detected"] is True
    assert data["tier_metadata"]["ordering_pairs"] == [
        {"strict": "fast", "lenient": "standard"}
    ]


def test_multi_tier_adds_config_tiers(tmp_path: Path):
    """Multi-tier planner adds config.tiers to the config block."""
    write_search_history(
        tmp_path,
        _sample_history(),
        _cfg_with_tiers(),
        convergence_reason="multi_tier_all_converged",
        planner=_MultiTierPlannerStub(),
    )
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert "tiers" in data["config"]
    assert len(data["config"]["tiers"]) == 2
    assert data["config"]["tiers"][0]["label"] == "fast"
    assert data["config"]["tiers"][1]["label"] == "standard"
    assert len(data["config"]["tiers"][0]["filters"]) == 1


def test_multi_tier_preserves_boundary_summary(tmp_path: Path):
    """Multi-tier planner still populates boundary_summary for backward compat."""
    write_search_history(
        tmp_path,
        _sample_history(),
        _cfg_with_tiers(),
        convergence_reason="multi_tier_all_converged",
        planner=_MultiTierPlannerStub(),
    )
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert data["boundary_summary"] is not None
    assert data["boundary_summary"]["swept_dim_path"] == "phases.profiling.concurrency"
    assert data["boundary_summary"]["feasible_max"]["value"] == 64


def test_multi_tier_preserves_existing_fields(tmp_path: Path):
    """All existing top-level fields retain their types and semantics."""
    write_search_history(
        tmp_path,
        _sample_history(),
        _cfg_with_tiers(),
        convergence_reason="multi_tier_all_converged",
        planner=_MultiTierPlannerStub(),
    )
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert "config" in data and isinstance(data["config"], dict)
    assert "iterations" in data and isinstance(data["iterations"], list)
    assert "best_trials" in data
    assert "boundary_summary" in data
    assert "recipe" in data
    assert "convergence_reason" in data
    assert data["convergence_reason"] == "multi_tier_all_converged"


def test_single_tier_planner_does_not_add_tier_results(tmp_path: Path):
    """Single-tier planner output has no tier_results or tier_metadata."""
    write_search_history(
        tmp_path,
        _sample_history(),
        _cfg_without_tiers(),
        convergence_reason="smooth_isotonic_precision_reached",
        planner=_SingleTierPlannerStub(),
    )
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert "tier_results" not in data
    assert "tier_metadata" not in data
    assert "tiers" not in data["config"]


def test_no_planner_does_not_add_tier_results(tmp_path: Path):
    """When no planner is provided, tier_results are not added."""
    write_search_history(
        tmp_path,
        _sample_history(),
        _cfg_without_tiers(),
    )
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert "tier_results" not in data
    assert "tier_metadata" not in data

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Schema Backward Compatibility.

Feature: multi-tier-slo-search, Property 12: Schema Backward Compatibility

Validates: Requirements 6.3, 7.2

For any multi-tier search_history.json output, all fields from the existing
single-tier schema (config, iterations, best_trials, boundary_summary, recipe,
convergence_reason) SHALL be present with their existing types, so existing
consumers can deserialize without modification.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import orjson
from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.search_planner.base import SearchIteration
from aiperf.orchestrator.search_planner.multi_tier_models import TierResult

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_METRIC_TAGS = ["output_token_throughput", "time_to_first_token", "inter_token_latency"]
_STATS = ["avg", "p50", "p90", "p95", "p99"]
_OPS = ["lt", "le", "gt", "ge"]
_CONVERGENCE_STATUSES = [
    "converged",
    "partial",
    "no_pass_in_range",
    "no_failure_in_range",
]
_CONVERGENCE_REASONS = [
    "multi_tier_all_converged",
    "multi_tier_precision_reached",
    "max_iterations",
    None,
]


@st.composite
def _sla_filter_strategy(draw: st.DrawFn) -> SLAFilter:
    """Generate a valid SLAFilter."""
    return SLAFilter(
        metric_tag=draw(st.sampled_from(_METRIC_TAGS)),
        stat=draw(st.sampled_from(_STATS)),
        op=draw(st.sampled_from(_OPS)),
        threshold=draw(
            st.floats(
                min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False
            )
        ),
    )


@st.composite
def _slo_tier_strategy(draw: st.DrawFn, label: str | None = None) -> SLOTier:
    """Generate a valid SLOTier with 1-3 filters."""
    filters = draw(st.lists(_sla_filter_strategy(), min_size=1, max_size=3))
    tier_label = label or draw(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz",
            min_size=3,
            max_size=10,
        )
    )
    return SLOTier(label=tier_label, filters=filters)


@st.composite
def _tier_count_strategy(draw: st.DrawFn) -> int:
    """Generate tier count between 2 and 5."""
    return draw(st.integers(min_value=2, max_value=5))


@st.composite
def _multi_tier_config(draw: st.DrawFn) -> tuple[AdaptiveSearchSweep, list[SLOTier]]:
    """Generate a valid AdaptiveSearchSweep with multiple tiers."""
    n_tiers = draw(_tier_count_strategy())
    tiers = [draw(_slo_tier_strategy(label=f"tier_{i}")) for i in range(n_tiers)]
    cfg = AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=256, kind="int"
            )
        ],
        objectives=[Objective(metric="output_token_throughput", direction="maximize")],
        max_iterations=draw(st.integers(min_value=10, max_value=50)),
        sla_filters=[tiers[0].filters[0]],
        sla_tiers=tiers,
    )
    return cfg, tiers


@st.composite
def _search_iteration_strategy(draw: st.DrawFn, iteration_idx: int) -> SearchIteration:
    """Generate a valid SearchIteration."""
    concurrency = draw(st.integers(min_value=1, max_value=256))
    objective_val = draw(
        st.floats(
            min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False
        )
    )
    return SearchIteration(
        iteration_idx=iteration_idx,
        variation_values={"phases.profiling.concurrency": concurrency},
        objective_value=objective_val,
        objective_values=[objective_val],
        feasible=draw(st.booleans()),
    )


@st.composite
def _history_strategy(draw: st.DrawFn) -> list[SearchIteration]:
    """Generate a list of 1-10 SearchIterations."""
    n = draw(st.integers(min_value=1, max_value=10))
    return [draw(_search_iteration_strategy(iteration_idx=i)) for i in range(n)]


@st.composite
def _tier_result_strategy(draw: st.DrawFn, tier: SLOTier) -> TierResult:
    """Generate a TierResult for a given tier."""
    bracket_lower = draw(st.integers(min_value=1, max_value=128))
    bracket_upper = draw(st.integers(min_value=bracket_lower + 1, max_value=256))
    boundary = draw(st.one_of(st.just(bracket_lower), st.none()))
    return TierResult(
        label=tier.label,
        boundary_concurrency=boundary,
        convergence_status=draw(st.sampled_from(_CONVERGENCE_STATUSES)),
        convergence_reason=draw(st.sampled_from(_CONVERGENCE_REASONS)),
        binding_constraint=draw(
            st.one_of(
                st.none(),
                st.just(
                    {
                        "metric_tag": tier.filters[0].metric_tag,
                        "stat": tier.filters[0].stat,
                        "op": tier.filters[0].op,
                        "threshold": tier.filters[0].threshold,
                        "observed": 42.0,
                    }
                ),
            )
        ),
        bracket_lower=bracket_lower,
        bracket_upper=bracket_upper,
        confidence_interval=None,
        probe_count=draw(st.integers(min_value=1, max_value=20)),
        filters=[f.model_dump() for f in tier.filters],
    )


class _MultiTierPlannerStub:
    """Stub planner parameterized by generated data for property testing."""

    def __init__(
        self,
        tiers: list[SLOTier],
        tier_results_list: list[TierResult],
        ordering_detected: bool,
    ) -> None:
        self._tiers = tiers
        self._tier_results = tier_results_list
        self._ordering_detected = ordering_detected

    def boundary_summary(self) -> dict[str, Any] | None:
        # Use the most-lenient tier's bracket as the boundary_summary
        # for backward compatibility.
        if not self._tier_results:
            return None
        lenient = self._tier_results[-1]
        return {
            "swept_dim_path": "phases.profiling.concurrency",
            "feasible_max": {
                "value": lenient.bracket_lower,
                "iteration_idx": 0,
                "objective_value": None,
            },
            "infeasible_min": {
                "value": lenient.bracket_upper,
                "iteration_idx": 1,
                "first_breach": lenient.binding_constraint,
            },
        }

    def tier_results(self) -> list[TierResult]:
        return self._tier_results

    def tier_metadata(self) -> dict[str, Any]:
        total = sum(tr.probe_count for tr in self._tier_results)
        return {
            "actual_probe_count": total,
            "tier_evaluation_count": total,
            "ordering_detected": self._ordering_detected,
            "ordering_pairs": (
                [{"strict": self._tiers[0].label, "lenient": self._tiers[-1].label}]
                if self._ordering_detected and len(self._tiers) >= 2
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Property 12: Schema Backward Compatibility
# ---------------------------------------------------------------------------


class TestProperty12SchemaBackwardCompatibility:
    """Property 12: Schema Backward Compatibility.

    **Validates: Requirements 6.3, 7.2**
    """

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_all_single_tier_fields_present_with_correct_types(
        self, data: st.DataObject
    ) -> None:
        """Multi-tier output contains all existing single-tier fields with correct types.

        Verifies: config (dict), iterations (list), best_trials (any),
        boundary_summary (dict or null), recipe (str or null),
        convergence_reason (str or null).

        **Validates: Requirements 6.3, 7.2**
        """
        cfg, tiers = data.draw(_multi_tier_config())
        history = data.draw(_history_strategy())
        tier_results_list = [data.draw(_tier_result_strategy(tier=t)) for t in tiers]
        ordering_detected = data.draw(st.booleans())
        convergence_reason = data.draw(st.sampled_from(_CONVERGENCE_REASONS))

        planner = _MultiTierPlannerStub(
            tiers=tiers,
            tier_results_list=tier_results_list,
            ordering_detected=ordering_detected,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_search_history(
                out_dir,
                history,
                cfg,
                convergence_reason=convergence_reason,
                planner=planner,
            )

            output = orjson.loads((out_dir / "search_history.json").read_bytes())

        # All existing single-tier fields must be present
        assert "config" in output, "config field missing"
        assert "iterations" in output, "iterations field missing"
        assert "best_trials" in output, "best_trials field missing"
        assert "boundary_summary" in output, "boundary_summary field missing"
        assert "recipe" in output, "recipe field missing"
        assert "convergence_reason" in output, "convergence_reason field missing"

        # Type checks for existing fields
        assert isinstance(output["config"], dict), "config must be a dict"
        assert isinstance(output["iterations"], list), "iterations must be a list"
        # best_trials can be list or None
        assert output["best_trials"] is None or isinstance(output["best_trials"], list)
        # boundary_summary can be dict or None
        assert output["boundary_summary"] is None or isinstance(
            output["boundary_summary"], dict
        )
        # recipe can be str or None
        assert output["recipe"] is None or isinstance(output["recipe"], str)
        # convergence_reason can be str or None
        assert output["convergence_reason"] is None or isinstance(
            output["convergence_reason"], str
        )

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_config_block_contains_standard_fields(self, data: st.DataObject) -> None:
        """config block contains all standard single-tier fields.

        **Validates: Requirements 6.3, 7.2**
        """
        cfg, tiers = data.draw(_multi_tier_config())
        history = data.draw(_history_strategy())
        tier_results_list = [data.draw(_tier_result_strategy(tier=t)) for t in tiers]

        planner = _MultiTierPlannerStub(
            tiers=tiers,
            tier_results_list=tier_results_list,
            ordering_detected=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_search_history(
                out_dir,
                history,
                cfg,
                convergence_reason="multi_tier_all_converged",
                planner=planner,
            )

            output = orjson.loads((out_dir / "search_history.json").read_bytes())

        config = output["config"]

        # Standard config fields that must always be present
        assert "planner" in config, "config.planner missing"
        assert "objectives" in config, "config.objectives missing"
        assert "outcome_constraints" in config, "config.outcome_constraints missing"
        assert "max_iterations" in config, "config.max_iterations missing"
        assert "n_initial_points" in config, "config.n_initial_points missing"
        assert "random_seed" in config, "config.random_seed missing"
        assert "improvement_patience" in config, "config.improvement_patience missing"
        assert "plateau_window" in config, "config.plateau_window missing"
        assert "plateau_threshold" in config, "config.plateau_threshold missing"
        assert "search_space" in config, "config.search_space missing"
        assert "sla_filters" in config, "config.sla_filters missing"

        # Type checks for config fields
        assert isinstance(config["objectives"], list)
        assert isinstance(config["outcome_constraints"], list)
        assert isinstance(config["search_space"], list)
        assert isinstance(config["sla_filters"], list)

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_new_fields_are_additive(self, data: st.DataObject) -> None:
        """tier_results and tier_metadata are additive and don't break existing schema.

        **Validates: Requirements 6.3, 7.2**
        """
        cfg, tiers = data.draw(_multi_tier_config())
        history = data.draw(_history_strategy())
        tier_results_list = [data.draw(_tier_result_strategy(tier=t)) for t in tiers]
        ordering_detected = data.draw(st.booleans())

        planner = _MultiTierPlannerStub(
            tiers=tiers,
            tier_results_list=tier_results_list,
            ordering_detected=ordering_detected,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_search_history(
                out_dir,
                history,
                cfg,
                convergence_reason="multi_tier_all_converged",
                planner=planner,
            )

            output = orjson.loads((out_dir / "search_history.json").read_bytes())

        # New multi-tier fields are present (additive)
        assert "tier_results" in output, "tier_results should be present for multi-tier"
        assert "tier_metadata" in output, (
            "tier_metadata should be present for multi-tier"
        )
        assert isinstance(output["tier_results"], list)
        assert isinstance(output["tier_metadata"], dict)

        # Existing fields still present alongside new fields
        assert "config" in output
        assert "iterations" in output
        assert "best_trials" in output
        assert "boundary_summary" in output
        assert "recipe" in output
        assert "convergence_reason" in output

        # New fields don't overwrite existing fields
        assert output["config"] is not output.get("tier_results")
        assert output["iterations"] is not output.get("tier_metadata")

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_boundary_summary_populated_from_most_lenient_tier(
        self, data: st.DataObject
    ) -> None:
        """boundary_summary is populated from most-lenient tier for backward compat.

        **Validates: Requirements 6.3, 7.2**
        """
        cfg, tiers = data.draw(_multi_tier_config())
        history = data.draw(_history_strategy())
        tier_results_list = [data.draw(_tier_result_strategy(tier=t)) for t in tiers]

        planner = _MultiTierPlannerStub(
            tiers=tiers,
            tier_results_list=tier_results_list,
            ordering_detected=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_search_history(
                out_dir,
                history,
                cfg,
                convergence_reason="multi_tier_all_converged",
                planner=planner,
            )

            output = orjson.loads((out_dir / "search_history.json").read_bytes())

        # boundary_summary must be populated (not None) for backward compat
        assert output["boundary_summary"] is not None, (
            "boundary_summary must be populated for multi-tier output"
        )
        assert isinstance(output["boundary_summary"], dict)
        assert "swept_dim_path" in output["boundary_summary"]
        assert (
            output["boundary_summary"]["swept_dim_path"]
            == "phases.profiling.concurrency"
        )

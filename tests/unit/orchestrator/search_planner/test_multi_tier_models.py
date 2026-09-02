# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multi-tier SLO search data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.orchestrator.search_planner.multi_tier_models import (
    BracketState,
    SLOTier,
    TierResult,
)


class TestSLOTier:
    """Tests for SLOTier validation rules."""

    def _make_filter(self, threshold: float = 100.0) -> SLAFilter:
        return SLAFilter(
            metric_tag="output_token_throughput",
            stat="avg",
            op="gt",
            threshold=threshold,
        )

    def test_slo_tier_rejects_empty_filters_raises_validation_error(self):
        """Empty filters list must be rejected due to min_length=1."""
        with pytest.raises(ValidationError, match="filters"):
            SLOTier(label="empty", filters=[])

    def test_slo_tier_accepts_single_filter(self):
        """A tier with exactly one filter is valid."""
        tier = SLOTier(label="single", filters=[self._make_filter()])
        assert tier.label == "single"
        assert len(tier.filters) == 1

    def test_slo_tier_accepts_multiple_filters(self):
        """A tier with multiple filters is valid."""
        filters = [self._make_filter(30.0), self._make_filter(100.0)]
        tier = SLOTier(label="multi", filters=filters)
        assert len(tier.filters) == 2

    def test_slo_tier_ordering_rank_defaults_to_none(self):
        """ordering_rank defaults to None when not specified."""
        tier = SLOTier(label="unordered", filters=[self._make_filter()])
        assert tier.ordering_rank is None

    def test_slo_tier_ordering_rank_accepts_integer(self):
        """ordering_rank can be set to an integer value."""
        tier = SLOTier(label="ordered", filters=[self._make_filter()], ordering_rank=0)
        assert tier.ordering_rank == 0


class TestBracketState:
    """Tests for BracketState default initialization."""

    def _make_tier(self) -> SLOTier:
        return SLOTier(
            label="test",
            filters=[
                SLAFilter(
                    metric_tag="time_to_first_token",
                    stat="p95",
                    op="lt",
                    threshold=5000.0,
                )
            ],
        )

    def test_bracket_state_defaults_feasible_max_none(self):
        """feasible_max defaults to None."""
        state = BracketState(tier=self._make_tier())
        assert state.feasible_max is None

    def test_bracket_state_defaults_infeasible_min_none(self):
        """infeasible_min defaults to None."""
        state = BracketState(tier=self._make_tier())
        assert state.infeasible_min is None

    def test_bracket_state_defaults_converged_false(self):
        """converged defaults to False."""
        state = BracketState(tier=self._make_tier())
        assert state.converged is False

    def test_bracket_state_defaults_convergence_reason_none(self):
        """convergence_reason defaults to None."""
        state = BracketState(tier=self._make_tier())
        assert state.convergence_reason is None

    def test_bracket_state_defaults_binding_constraint_none(self):
        """binding_constraint defaults to None."""
        state = BracketState(tier=self._make_tier())
        assert state.binding_constraint is None

    def test_bracket_state_defaults_probe_count_zero(self):
        """probe_count defaults to 0."""
        state = BracketState(tier=self._make_tier())
        assert state.probe_count == 0

    def test_bracket_state_stores_tier_reference(self):
        """The tier field stores the provided SLOTier instance."""
        tier = self._make_tier()
        state = BracketState(tier=tier)
        assert state.tier is tier

    def test_bracket_state_mutable_fields(self):
        """BracketState fields can be mutated after creation."""
        state = BracketState(tier=self._make_tier())
        state.feasible_max = 64
        state.infeasible_min = 128
        state.converged = True
        state.convergence_reason = "multi_tier_precision_reached"
        state.binding_constraint = {"metric_tag": "time_to_first_token", "stat": "p95"}
        state.probe_count = 5

        assert state.feasible_max == 64
        assert state.infeasible_min == 128
        assert state.converged is True
        assert state.convergence_reason == "multi_tier_precision_reached"
        assert state.binding_constraint == {
            "metric_tag": "time_to_first_token",
            "stat": "p95",
        }
        assert state.probe_count == 5


class TestTierResult:
    """Tests for TierResult serialization round-trip."""

    def _make_tier_result(self) -> TierResult:
        return TierResult(
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
                },
                {
                    "metric_tag": "time_to_first_token",
                    "stat": "p95",
                    "op": "lt",
                    "threshold": 5000.0,
                },
            ],
        )

    def test_tier_result_serialization_round_trip(self):
        """TierResult serializes to dict and deserializes back identically."""
        original = self._make_tier_result()
        serialized = original.model_dump()
        restored = TierResult.model_validate(serialized)

        assert restored.label == original.label
        assert restored.boundary_concurrency == original.boundary_concurrency
        assert restored.convergence_status == original.convergence_status
        assert restored.convergence_reason == original.convergence_reason
        assert restored.binding_constraint == original.binding_constraint
        assert restored.bracket_lower == original.bracket_lower
        assert restored.bracket_upper == original.bracket_upper
        assert restored.confidence_interval == original.confidence_interval
        assert restored.probe_count == original.probe_count
        assert restored.filters == original.filters

    def test_tier_result_json_round_trip(self):
        """TierResult survives JSON serialization and deserialization."""
        original = self._make_tier_result()
        json_str = original.model_dump_json()
        restored = TierResult.model_validate_json(json_str)

        assert restored == original

    def test_tier_result_partial_convergence(self):
        """TierResult with partial convergence status serializes correctly."""
        result = TierResult(
            label="standard",
            boundary_concurrency=None,
            convergence_status="partial",
            convergence_reason=None,
            binding_constraint=None,
            bracket_lower=64,
            bracket_upper=256,
            confidence_interval=None,
            probe_count=3,
            filters=[
                {
                    "metric_tag": "output_token_throughput",
                    "stat": "avg",
                    "op": "gt",
                    "threshold": 100.0,
                }
            ],
        )
        serialized = result.model_dump()
        restored = TierResult.model_validate(serialized)

        assert restored.boundary_concurrency is None
        assert restored.convergence_status == "partial"
        assert restored.convergence_reason is None
        assert restored.binding_constraint is None

    def test_tier_result_with_confidence_interval(self):
        """TierResult with confidence interval serializes correctly."""
        result = TierResult(
            label="economy",
            boundary_concurrency=256,
            convergence_status="converged",
            bracket_lower=256,
            bracket_upper=257,
            confidence_interval={"low": 250.0, "high": 260.0},
            probe_count=6,
            filters=[
                {
                    "metric_tag": "output_token_throughput",
                    "stat": "avg",
                    "op": "gt",
                    "threshold": 30.0,
                }
            ],
        )
        serialized = result.model_dump()
        restored = TierResult.model_validate(serialized)

        assert restored.confidence_interval == {"low": 250.0, "high": 260.0}

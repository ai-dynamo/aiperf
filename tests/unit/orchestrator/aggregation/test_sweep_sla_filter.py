# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.orchestrator.aggregation.sweep_sla_filter``.

The module is the SLA-filter helper for ``SweepAnalyzer``. Tests exercise:
- both filter input shapes (typed ``SLAFilter`` and dict),
- both stats-dict layouts (multi-trial flattened ``"<tag>_<stat>"`` and
  single-trial nested ``{tag: {stat: value}}``),
- every operator,
- error/edge cases (unknown op, missing metric, type-coercion failure of
  ``sla_filter_to_dict`` input).
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.orchestrator.aggregation.sweep_sla_filter import (
    filter_feasible,
    sla_filter_to_dict,
)


def _flat_stats(metric_tag: str, stat: str, mean: float) -> dict[str, Any]:
    """Build the multi-trial layout: ``{<tag>_<stat>: {"mean": value}}``."""
    return {f"{metric_tag}_{stat}": {"mean": mean, "std": 0.0}}


def _nested_stats(metric_tag: str, stat: str, value: float) -> dict[str, Any]:
    """Build the single-trial layout: ``{tag: {stat: value}}``."""
    return {metric_tag: {stat: value, "avg": value, "p50": value}}


class TestSlaFilterToDict:
    def test_pydantic_model_dumps(self) -> None:
        f = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
        assert sla_filter_to_dict(f) == {
            "metric_tag": "ttft",
            "stat": "p95",
            "op": "lt",
            "threshold": 200.0,
        }

    def test_dict_passthrough_returns_copy(self) -> None:
        original = {"metric_tag": "ttft", "stat": "p95", "op": "lt", "threshold": 200.0}
        result = sla_filter_to_dict(original)
        assert result == original
        assert result is not original  # defensive copy

    def test_invalid_type_raises_typeerror_with_message(self) -> None:
        with pytest.raises(TypeError, match="SLAFilter or dict"):
            sla_filter_to_dict("not a filter")  # type: ignore[arg-type]

    def test_invalid_type_includes_offending_type_name(self) -> None:
        with pytest.raises(TypeError, match="int"):
            sla_filter_to_dict(42)  # type: ignore[arg-type]


class TestFilterFeasibleOperators:
    @pytest.mark.parametrize(
        "op, threshold, value, should_pass",
        [
            param("lt", 200.0, 150.0, True, id="lt-pass"),
            param("lt", 200.0, 200.0, False, id="lt-fail-on-boundary"),
            param("lt", 200.0, 250.0, False, id="lt-fail"),
            param("le", 200.0, 200.0, True, id="le-pass-on-boundary"),
            param("le", 200.0, 201.0, False, id="le-fail"),
            param("gt", 100.0, 150.0, True, id="gt-pass"),
            param("gt", 100.0, 100.0, False, id="gt-fail-on-boundary"),
            param("ge", 100.0, 100.0, True, id="ge-pass-on-boundary"),
            param("ge", 100.0, 99.0, False, id="ge-fail"),
        ],
    )  # fmt: skip
    def test_each_operator(
        self, op: str, threshold: float, value: float, should_pass: bool
    ) -> None:
        f = SLAFilter(metric_tag="ttft", stat="p95", op=op, threshold=threshold)  # type: ignore[arg-type]
        combo = ("c=10",)
        per_combo = {combo: _flat_stats("ttft", "p95", value)}
        result = filter_feasible(per_combo, [f])
        assert (combo in result) is should_pass


class TestFilterFeasibleLayouts:
    def test_multi_trial_flat_layout(self) -> None:
        f = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
        combo = ("c=10",)
        per_combo = {combo: _flat_stats("ttft", "p95", 150.0)}
        assert combo in filter_feasible(per_combo, [f])

    def test_single_trial_nested_layout(self) -> None:
        f = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
        combo = ("c=10",)
        per_combo = {combo: _nested_stats("ttft", "p95", 150.0)}
        assert combo in filter_feasible(per_combo, [f])

    def test_flat_layout_takes_precedence_over_nested(self) -> None:
        """Flat key wins when both are present."""
        f = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
        combo = ("c=10",)
        stats = {
            "ttft_p95": {"mean": 150.0},  # flat: feasible
            "ttft": {"p95": 999.0},  # nested: infeasible — should be ignored
        }
        per_combo = {combo: stats}
        assert combo in filter_feasible(per_combo, [f])


class TestFilterFeasibleInputShapes:
    def test_filter_as_dict(self) -> None:
        f = {"metric_tag": "ttft", "stat": "p95", "op": "lt", "threshold": 200.0}
        combo = ("c=10",)
        per_combo = {combo: _flat_stats("ttft", "p95", 150.0)}
        assert combo in filter_feasible(per_combo, [f])

    def test_filter_as_pydantic(self) -> None:
        f = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
        combo = ("c=10",)
        per_combo = {combo: _flat_stats("ttft", "p95", 150.0)}
        assert combo in filter_feasible(per_combo, [f])


class TestFilterFeasibleMissingMetric:
    def test_missing_metric_treated_as_infeasible(self) -> None:
        f = SLAFilter(metric_tag="not_present", stat="p95", op="lt", threshold=200.0)
        combo = ("c=10",)
        per_combo = {combo: _flat_stats("ttft", "p95", 150.0)}
        assert filter_feasible(per_combo, [f]) == {}

    def test_metric_present_but_stat_missing_is_infeasible(self) -> None:
        f = SLAFilter(metric_tag="ttft", stat="p99", op="lt", threshold=200.0)
        combo = ("c=10",)
        # nested layout has p95 but neither p99 nor mean — no fallback available.
        per_combo = {combo: {"ttft": {"p95": 150.0}}}
        assert filter_feasible(per_combo, [f]) == {}

    def test_single_trial_falls_back_to_mean_when_stat_missing(self) -> None:
        """Realistic single-trial blocks carry only mean/std/min/max/cv/ci_*/unit
        (see ``_json_metric_to_stats``). Requesting p95/p99 in this layout must
        fall back to ``mean`` instead of silently returning infeasible — the
        recipe convention matches ``post_process._extract_points``.
        """
        f = SLAFilter(metric_tag="itl", stat="p95", op="lt", threshold=10.0)
        combo = ("c=8",)
        per_combo = {
            combo: {
                "itl": {
                    "mean": 5.2,
                    "std": 0.0,
                    "min": 4.1,
                    "max": 6.5,
                    "cv": 0.0,
                    "ci_low": 5.2,
                    "ci_high": 5.2,
                    "unit": "ms",
                }
            }
        }
        assert combo in filter_feasible(per_combo, [f])

    def test_single_trial_mean_fallback_marks_breach_when_mean_exceeds(
        self,
    ) -> None:
        """Same fallback path, but the mean exceeds the threshold."""
        f = SLAFilter(metric_tag="itl", stat="p95", op="lt", threshold=10.0)
        combo = ("c=64",)
        per_combo = {
            combo: {
                "itl": {
                    "mean": 48.1,  # well past the 10ms p95 threshold
                    "std": 0.0,
                    "min": 45.0,
                    "max": 50.4,
                    "cv": 0.0,
                    "ci_low": 48.1,
                    "ci_high": 48.1,
                    "unit": "ms",
                }
            }
        }
        assert filter_feasible(per_combo, [f]) == {}

    def test_explicit_stat_attr_wins_over_mean_fallback(self) -> None:
        """If the legacy explicit-stat attribute is present, it takes precedence."""
        f = SLAFilter(metric_tag="ttft", stat="p99", op="lt", threshold=200.0)
        combo = ("c=10",)
        per_combo = {
            combo: {
                "ttft": {
                    "p99": 150.0,
                    "mean": 999.0,
                },  # p99 feasible, mean would breach
            }
        }
        assert combo in filter_feasible(per_combo, [f])

    def test_non_dict_block_is_infeasible(self) -> None:
        f = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
        combo = ("c=10",)
        per_combo = {combo: {"ttft": "not_a_dict"}}
        assert filter_feasible(per_combo, [f]) == {}


class TestFilterFeasibleMultipleFiltersAndCombos:
    def test_all_filters_must_pass(self) -> None:
        f1 = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
        f2 = SLAFilter(metric_tag="throughput", stat="avg", op="gt", threshold=100.0)
        combo_pass = ("c=10",)
        combo_fail_on_f2 = ("c=20",)
        per_combo = {
            combo_pass: {
                **_flat_stats("ttft", "p95", 150.0),
                **_flat_stats("throughput", "avg", 200.0),
            },
            combo_fail_on_f2: {
                **_flat_stats("ttft", "p95", 150.0),
                **_flat_stats("throughput", "avg", 50.0),
            },
        }
        result = filter_feasible(per_combo, [f1, f2])
        assert combo_pass in result
        assert combo_fail_on_f2 not in result

    def test_empty_filter_list_keeps_everything(self) -> None:
        combo = ("c=10",)
        per_combo = {combo: _flat_stats("ttft", "p95", 999.0)}
        assert filter_feasible(per_combo, []) == per_combo

    def test_empty_combinations_returns_empty(self) -> None:
        f = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
        assert filter_feasible({}, [f]) == {}


class TestFilterFeasibleErrors:
    def test_unknown_op_raises_valueerror(self) -> None:
        bad = {"metric_tag": "ttft", "stat": "p95", "op": "eq", "threshold": 200.0}
        combo = ("c=10",)
        per_combo = {combo: _flat_stats("ttft", "p95", 150.0)}
        with pytest.raises(ValueError, match="unknown SLA filter operator 'eq'"):
            filter_feasible(per_combo, [bad])

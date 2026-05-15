# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep aggregation components.

Covers ``SweepAnalyzer`` / ``identify_pareto_optimal`` and friends, plus
focused cases that name the realistic throughput-vs-TTFT shape and the
multi-axis sweep schema.
"""

import pytest

from aiperf.orchestrator.aggregation import (
    DEFAULT_PARETO_OBJECTIVES,
    OptimizationDirection,
    ParameterCombination,
    ParetoObjective,
    SweepAnalyzer,
    identify_pareto_optimal,
)


def _stat(mean: float, p99: float | None = None) -> dict[str, float]:
    """Build a metric stats dict with realistic keys."""
    out: dict[str, float] = {"mean": mean}
    if p99 is not None:
        out["p99"] = p99
    return out


class TestOptimizationDirection:
    """Tests for OptimizationDirection enum."""

    def test_maximize_value(self):
        """Test MAXIMIZE enum value."""
        assert OptimizationDirection.MAXIMIZE.value == "maximize"

    def test_minimize_value(self):
        """Test MINIMIZE enum value."""
        assert OptimizationDirection.MINIMIZE.value == "minimize"

    def test_enum_members(self):
        """Test that enum has exactly two members."""
        assert len(OptimizationDirection) == 2
        assert set(OptimizationDirection) == {
            OptimizationDirection.MAXIMIZE,
            OptimizationDirection.MINIMIZE,
        }

    def test_case_insensitive_lookup(self):
        """HEAD models OptimizationDirection as CaseInsensitiveStrEnum."""
        # CaseInsensitiveStrEnum allows lookup by any-case string.
        assert OptimizationDirection("MAXIMIZE") == OptimizationDirection.MAXIMIZE
        assert OptimizationDirection("Minimize") == OptimizationDirection.MINIMIZE
        assert OptimizationDirection("maximize") == OptimizationDirection.MAXIMIZE


class TestObjective:
    """Tests for ParetoObjective named tuple."""

    def test_create_maximize_objective(self):
        """Test creating an objective with MAXIMIZE direction."""
        obj = ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        assert obj.metric_key == "request_throughput_avg"
        assert obj.direction == OptimizationDirection.MAXIMIZE

    def test_create_minimize_objective(self):
        """Test creating an objective with MINIMIZE direction."""
        obj = ParetoObjective("time_to_first_token_p99", OptimizationDirection.MINIMIZE)
        assert obj.metric_key == "time_to_first_token_p99"
        assert obj.direction == OptimizationDirection.MINIMIZE

    def test_objective_is_immutable(self):
        """Test that ParetoObjective is immutable (NamedTuple property)."""
        obj = ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        with pytest.raises(AttributeError):
            obj.metric_key = "new_metric"

    def test_objective_equality(self):
        """Test that objectives with same values are equal."""
        obj1 = ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        obj2 = ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        assert obj1 == obj2

    def test_objective_inequality(self):
        """Test that objectives with different values are not equal."""
        obj1 = ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        obj2 = ParetoObjective(
            "time_to_first_token_p99", OptimizationDirection.MINIMIZE
        )
        assert obj1 != obj2

    def test_objective_tuple_unpacking(self):
        """Test that ParetoObjective can be unpacked like a tuple."""
        obj = ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        metric_key, direction = obj
        assert metric_key == "request_throughput_avg"
        assert direction == OptimizationDirection.MAXIMIZE

    def test_objective_indexing(self):
        """Test that ParetoObjective supports indexing."""
        obj = ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        assert obj[0] == "request_throughput_avg"
        assert obj[1] == OptimizationDirection.MAXIMIZE

    def test_objective_repr(self):
        """Test that ParetoObjective has a useful repr."""
        obj = ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        repr_str = repr(obj)
        assert "ParetoObjective" in repr_str
        assert "request_throughput_avg" in repr_str
        assert "MAXIMIZE" in repr_str


class TestDefaultParetoObjectives:
    """Tests for DEFAULT_PARETO_OBJECTIVES constant."""

    def test_default_objectives_is_list(self):
        """Test that DEFAULT_PARETO_OBJECTIVES is a list."""
        assert isinstance(DEFAULT_PARETO_OBJECTIVES, list)

    def test_default_objectives_has_two_objectives(self):
        """Test that DEFAULT_PARETO_OBJECTIVES contains exactly two objectives."""
        assert len(DEFAULT_PARETO_OBJECTIVES) == 2

    def test_default_objectives_contains_objective_instances(self):
        """Test that all items in DEFAULT_PARETO_OBJECTIVES are ParetoObjective instances."""
        for obj in DEFAULT_PARETO_OBJECTIVES:
            assert isinstance(obj, ParetoObjective)

    def test_default_objectives_first_is_throughput_maximize(self):
        """Test that first objective is to maximize request_throughput_avg."""
        obj = DEFAULT_PARETO_OBJECTIVES[0]
        assert obj.metric_key == "request_throughput_avg"
        assert obj.direction == OptimizationDirection.MAXIMIZE

    def test_default_objectives_second_is_latency_minimize(self):
        """Test that second objective is to minimize time_to_first_token_p99."""
        obj = DEFAULT_PARETO_OBJECTIVES[1]
        assert obj.metric_key == "time_to_first_token_p99"
        assert obj.direction == OptimizationDirection.MINIMIZE

    def test_default_objectives_immutable(self):
        """Test that DEFAULT_PARETO_OBJECTIVES objectives are immutable."""
        obj = DEFAULT_PARETO_OBJECTIVES[0]
        with pytest.raises(AttributeError):
            obj.metric_key = "new_metric"

    def test_default_pareto_objectives_are_throughput_max_then_ttft_min(self):
        """Compact form of the two element-by-element checks above."""
        assert len(DEFAULT_PARETO_OBJECTIVES) == 2
        assert DEFAULT_PARETO_OBJECTIVES[0] == ParetoObjective(
            "request_throughput_avg", OptimizationDirection.MAXIMIZE
        )
        assert DEFAULT_PARETO_OBJECTIVES[1] == ParetoObjective(
            "time_to_first_token_p99", OptimizationDirection.MINIMIZE
        )


class TestParameterCombination:
    """Tests for ParameterCombination named tuple."""

    def test_create_single_parameter(self):
        """Test creating a ParameterCombination with single parameter."""
        combo = ParameterCombination({"concurrency": 10})
        assert combo.parameters == {"concurrency": 10}

    def test_create_multiple_parameters(self):
        """Test creating a ParameterCombination with multiple parameters."""
        combo = ParameterCombination({"concurrency": 10, "request_rate": 20})
        assert combo.parameters == {"concurrency": 10, "request_rate": 20}

    def test_to_dict(self):
        """Test to_dict() method."""
        combo = ParameterCombination({"concurrency": 10, "request_rate": 20})
        result = combo.to_dict()
        assert result == {"concurrency": 10, "request_rate": 20}
        # Verify it's a copy
        result["concurrency"] = 999
        assert combo.parameters["concurrency"] == 10

    def test_parameter_combination_to_dict_round_trip_returns_independent_copy(self):
        """HEAD-style round-trip: mutating to_dict() result doesn't poison source."""
        combo = ParameterCombination({"concurrency": 10, "request_rate": 20})
        result = combo.to_dict()
        assert result == {"concurrency": 10, "request_rate": 20}
        result["concurrency"] = 999
        assert combo.parameters["concurrency"] == 10

    def test_str_representation(self):
        """Test string representation."""
        combo = ParameterCombination({"concurrency": 10, "request_rate": 20})
        result = str(combo)
        # Should be sorted by key
        assert result == "concurrency=10, request_rate=20"

    def test_hashable(self):
        """Test that ParameterCombination is hashable."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 10})
        combo3 = ParameterCombination({"concurrency": 20})

        # Can be used in sets
        combo_set = {combo1, combo2, combo3}
        assert len(combo_set) == 2  # combo1 and combo2 are equal

        # Can be used as dict keys
        combo_dict = {combo1: "value1", combo3: "value3"}
        assert len(combo_dict) == 2

    def test_equality(self):
        """Test equality comparison."""
        combo1 = ParameterCombination({"concurrency": 10, "request_rate": 20})
        combo2 = ParameterCombination({"concurrency": 10, "request_rate": 20})
        combo3 = ParameterCombination({"concurrency": 20, "request_rate": 20})

        assert combo1 == combo2
        assert combo1 != combo3


class TestIdentifyParetoOptimal:
    """Tests for identify_pareto_optimal() function."""

    def test_single_configuration_is_pareto_optimal(self):
        """Test that a single configuration is always Pareto optimal."""
        combo = ParameterCombination({"concurrency": 10})
        per_combination_stats = {
            combo: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            }
        }

        result = identify_pareto_optimal(per_combination_stats)
        assert result == [combo]

    def test_all_configurations_pareto_optimal_when_none_dominates(self):
        """Test that all configurations are Pareto optimal when none dominates another."""
        # Config 1: high throughput, high latency
        # Config 2: low throughput, low latency
        # Neither dominates the other
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 100.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 50.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats)
        assert set(result) == {combo1, combo2}

    def test_dominated_configuration_excluded(self):
        """Test that a dominated configuration is excluded from Pareto optimal set."""
        # Config 1: 100 throughput, 50ms latency
        # Config 2: 150 throughput, 40ms latency (dominates config 1)
        # Config 3: 80 throughput, 60ms latency (dominated by config 1)
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 150.0},
                "time_to_first_token_p99": {"mean": 40.0},
            },
            combo3: {
                "request_throughput_avg": {"mean": 80.0},
                "time_to_first_token_p99": {"mean": 60.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats)
        assert result == [combo2]  # Only config 2 is Pareto optimal

    def test_pareto_frontier_with_tradeoffs(self):
        """Test Pareto frontier with multiple optimal points showing tradeoffs."""
        # Classic Pareto frontier: as throughput increases, latency increases
        # All three are Pareto optimal (different tradeoff points)
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 50.0},
                "time_to_first_token_p99": {"mean": 30.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
            combo3: {
                "request_throughput_avg": {"mean": 150.0},
                "time_to_first_token_p99": {"mean": 80.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats)
        assert set(result) == {combo1, combo2, combo3}

    def test_uses_default_objectives_when_none_provided(self):
        """Test that function uses DEFAULT_PARETO_OBJECTIVES when objectives=None."""
        combo = ParameterCombination({"concurrency": 10})
        per_combination_stats = {
            combo: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            }
        }

        # Should not raise error and should use default objectives
        result = identify_pareto_optimal(per_combination_stats, objectives=None)
        assert result == [combo]

    def test_custom_objectives_single_maximize(self):
        """Test with custom objective that only maximizes one metric."""
        objectives = [
            ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE)
        ]

        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {"request_throughput_avg": {"mean": 100.0}},
            combo2: {"request_throughput_avg": {"mean": 150.0}},
            combo3: {"request_throughput_avg": {"mean": 120.0}},
        }

        result = identify_pareto_optimal(per_combination_stats, objectives)
        assert result == [combo2]  # Highest throughput

    def test_custom_objectives_single_minimize(self):
        """Test with custom objective that only minimizes one metric."""
        objectives = [
            ParetoObjective("time_to_first_token_p99", OptimizationDirection.MINIMIZE)
        ]

        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {"time_to_first_token_p99": {"mean": 50.0}},
            combo2: {"time_to_first_token_p99": {"mean": 30.0}},
            combo3: {"time_to_first_token_p99": {"mean": 40.0}},
        }

        result = identify_pareto_optimal(per_combination_stats, objectives)
        assert result == [combo2]  # Lowest latency

    def test_custom_objectives_three_dimensions(self):
        """Test with three objectives (N-dimensional Pareto analysis)."""
        objectives = [
            ParetoObjective("throughput", OptimizationDirection.MAXIMIZE),
            ParetoObjective("latency", OptimizationDirection.MINIMIZE),
            ParetoObjective("cost", OptimizationDirection.MINIMIZE),
        ]

        # Config 1: high throughput, high latency, low cost
        # Config 2: medium throughput, low latency, medium cost
        # Config 3: low throughput, medium latency, high cost (dominated)
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {
                "throughput": {"mean": 150.0},
                "latency": {"mean": 80.0},
                "cost": {"mean": 10.0},
            },
            combo2: {
                "throughput": {"mean": 100.0},
                "latency": {"mean": 40.0},
                "cost": {"mean": 20.0},
            },
            combo3: {
                "throughput": {"mean": 50.0},
                "latency": {"mean": 60.0},
                "cost": {"mean": 30.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats, objectives)
        # Config 3 is dominated by both 1 and 2
        assert set(result) == {combo1, combo2}

    def test_identify_pareto_optimal_3d_drops_dominated_combo(self):
        """3-D objective space using realistic flattened metric keys.

        Throughput (max), TTFT (min), TTFO (min). c_mid is dominated by
        c_best on all three axes.
        """
        c_low = ParameterCombination({"concurrency": 5})
        c_mid = ParameterCombination({"concurrency": 10})
        c_best = ParameterCombination({"concurrency": 20})
        stats = {
            c_low: {
                "request_throughput_avg": _stat(80.0),
                "time_to_first_token_p99": _stat(40.0),
                "time_to_first_output_p99": _stat(45.0),
            },
            c_mid: {
                "request_throughput_avg": _stat(120.0),
                "time_to_first_token_p99": _stat(60.0),
                "time_to_first_output_p99": _stat(70.0),
            },
            c_best: {
                "request_throughput_avg": _stat(200.0),
                "time_to_first_token_p99": _stat(50.0),
                "time_to_first_output_p99": _stat(55.0),
            },
        }
        objectives = [
            ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
            ParetoObjective("time_to_first_token_p99", OptimizationDirection.MINIMIZE),
            ParetoObjective("time_to_first_output_p99", OptimizationDirection.MINIMIZE),
        ]

        pareto = identify_pareto_optimal(stats, objectives)

        assert c_mid not in pareto
        assert c_low in pareto and c_best in pareto

    def test_equal_values_not_dominated(self):
        """Test that configurations with equal objective values are not dominated."""
        # Config 1 and 2 have identical metrics
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats)
        # Both should be Pareto optimal (neither strictly dominates)
        assert set(result) == {combo1, combo2}

    def test_strictly_better_on_all_objectives_required(self):
        """Test that domination requires being better or equal on all, strictly better on at least one."""
        # Config 2 is better on throughput and equal on latency
        # Config 2 DOES dominate config 1 (better or equal on all, strictly better on one)
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 150.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats)
        # Only config 2 is Pareto optimal (dominates config 1)
        assert result == [combo2]

    def test_result_is_sorted(self):
        """Test that result is sorted by parameter combination."""
        combo1 = ParameterCombination({"concurrency": 30})
        combo2 = ParameterCombination({"concurrency": 10})
        combo3 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 150.0},
                "time_to_first_token_p99": {"mean": 80.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 50.0},
                "time_to_first_token_p99": {"mean": 30.0},
            },
            combo3: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats)
        # Should be sorted by parameters
        assert result == sorted(
            [combo1, combo2, combo3], key=lambda c: tuple(sorted(c.parameters.items()))
        )

    def test_empty_stats_returns_empty_list(self):
        """Test that empty per_combination_stats returns empty list."""
        result = identify_pareto_optimal({})
        assert result == []

    def test_complex_pareto_frontier(self):
        """Test complex scenario with multiple dominated and non-dominated configs."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        combo4 = ParameterCombination({"concurrency": 40})
        combo5 = ParameterCombination({"concurrency": 50})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 50.0},
                "time_to_first_token_p99": {"mean": 100.0},
            },  # Dominated by 2, 3, 5
            combo2: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 80.0},
            },  # Dominated by 3, 5
            combo3: {
                "request_throughput_avg": {"mean": 150.0},
                "time_to_first_token_p99": {"mean": 60.0},
            },  # Dominated by 5
            combo4: {
                "request_throughput_avg": {"mean": 120.0},
                "time_to_first_token_p99": {"mean": 70.0},
            },  # Dominated by 3, 5
            combo5: {
                "request_throughput_avg": {"mean": 200.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },  # Pareto optimal (best on both)
        }

        result = identify_pareto_optimal(per_combination_stats)
        # Only combo5 is Pareto optimal (dominates all others)
        assert result == [combo5]

    def test_true_pareto_frontier_with_multiple_optimal(self):
        """Test a true Pareto frontier where multiple configs are optimal."""
        # Create a realistic Pareto frontier where there are tradeoffs
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        combo4 = ParameterCombination({"concurrency": 40})
        combo5 = ParameterCombination({"concurrency": 50})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 50.0},
                "time_to_first_token_p99": {"mean": 30.0},
            },  # Low throughput, low latency - Pareto optimal
            combo2: {
                "request_throughput_avg": {"mean": 80.0},
                "time_to_first_token_p99": {"mean": 45.0},
            },  # Medium throughput, medium latency - Pareto optimal
            combo3: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },  # Good throughput, medium latency - Pareto optimal
            combo4: {
                "request_throughput_avg": {"mean": 120.0},
                "time_to_first_token_p99": {"mean": 70.0},
            },  # Higher throughput, higher latency - Pareto optimal
            combo5: {
                "request_throughput_avg": {"mean": 110.0},
                "time_to_first_token_p99": {"mean": 80.0},
            },  # Dominated by 4 (4 has higher throughput and lower latency)
        }

        result = identify_pareto_optimal(per_combination_stats)
        # Configs 1, 2, 3, 4 form the Pareto frontier
        assert set(result) == {combo1, combo2, combo3, combo4}

    def test_missing_metric_key_raises_error(self):
        """Test that missing metric key in objectives raises KeyError."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {"request_throughput_avg": {"mean": 100.0}},
            combo2: {"request_throughput_avg": {"mean": 180.0}},
        }

        # ParetoObjective references metric that doesn't exist
        objectives = [
            ParetoObjective("nonexistent_metric", OptimizationDirection.MAXIMIZE)
        ]

        with pytest.raises(KeyError):
            identify_pareto_optimal(per_combination_stats, objectives)

    def test_very_close_floating_point_values(self):
        """Test Pareto identification with very close floating point values."""
        # Values differ by tiny amounts (floating point precision edge case)
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0000001},
                "time_to_first_token_p99": {"mean": 50.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 100.0000002},  # Slightly higher
                "time_to_first_token_p99": {"mean": 50.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats)
        # Config 2 dominates 1 (strictly better throughput, equal latency)
        assert result == [combo2]

    def test_large_number_of_configurations(self):
        """Test Pareto identification with many configurations (performance test)."""
        # Create 100 configurations where throughput increases and latency decreases
        # This means higher configs dominate lower ones
        per_combination_stats = {}
        combos = []
        for i in range(1, 101):
            combo = ParameterCombination({"concurrency": i})
            combos.append(combo)
            per_combination_stats[combo] = {
                "request_throughput_avg": {"mean": float(i * 10)},  # Increases
                "time_to_first_token_p99": {"mean": float(101 - i)},  # Decreases
            }

        result = identify_pareto_optimal(per_combination_stats)

        # Only the last config (100) is Pareto optimal - it dominates all others
        # (highest throughput AND lowest latency)
        assert result == [combos[-1]]

    def test_all_dominated_by_one_configuration(self):
        """Test when one configuration dominates all others."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        combo4 = ParameterCombination({"concurrency": 40})
        combo5 = ParameterCombination({"concurrency": 50})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 100.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 150.0},
                "time_to_first_token_p99": {"mean": 80.0},
            },
            combo3: {
                "request_throughput_avg": {"mean": 200.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },  # Dominates all
            combo4: {
                "request_throughput_avg": {"mean": 120.0},
                "time_to_first_token_p99": {"mean": 90.0},
            },
            combo5: {
                "request_throughput_avg": {"mean": 180.0},
                "time_to_first_token_p99": {"mean": 70.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats)
        # Only config 3 is Pareto optimal
        assert result == [combo3]

    def test_four_dimensional_pareto_analysis(self):
        """Test Pareto analysis with 4 objectives (high-dimensional)."""
        objectives = [
            ParetoObjective("throughput", OptimizationDirection.MAXIMIZE),
            ParetoObjective("latency", OptimizationDirection.MINIMIZE),
            ParetoObjective("cost", OptimizationDirection.MINIMIZE),
            ParetoObjective("memory", OptimizationDirection.MINIMIZE),
        ]

        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {
                "throughput": {"mean": 100.0},
                "latency": {"mean": 50.0},
                "cost": {"mean": 10.0},
                "memory": {"mean": 1000.0},
            },  # Pareto optimal (best cost and memory)
            combo2: {
                "throughput": {"mean": 150.0},
                "latency": {"mean": 60.0},
                "cost": {"mean": 15.0},
                "memory": {"mean": 1200.0},
            },  # Pareto optimal (best throughput)
            combo3: {
                "throughput": {"mean": 120.0},
                "latency": {"mean": 55.0},
                "cost": {"mean": 12.0},
                "memory": {"mean": 1100.0},
            },  # Pareto optimal (balanced tradeoff)
        }

        result = identify_pareto_optimal(per_combination_stats, objectives)
        # All three are Pareto optimal - none dominates another on all 4 dimensions
        assert set(result) == {combo1, combo2, combo3}

    def test_negative_metric_values_in_pareto_analysis(self):
        """Test Pareto analysis works correctly with negative metric values."""
        objectives = [
            ParetoObjective(
                "profit", OptimizationDirection.MAXIMIZE
            ),  # Can be negative
            ParetoObjective("cost", OptimizationDirection.MINIMIZE),  # Can be negative
        ]

        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {
                "profit": {"mean": -50.0},  # Worst profit
                "cost": {"mean": 10.0},  # Best cost
            },  # Pareto optimal (best cost)
            combo2: {
                "profit": {"mean": -30.0},  # Best profit (least negative)
                "cost": {"mean": 20.0},  # Worst cost
            },  # Pareto optimal (best profit)
            combo3: {
                "profit": {"mean": -40.0},  # Middle profit
                "cost": {"mean": 15.0},  # Middle cost
            },  # Dominated by neither 1 nor 2 - Pareto optimal
        }

        result = identify_pareto_optimal(per_combination_stats, objectives)
        # All three form a Pareto frontier with different tradeoffs
        assert set(result) == {combo1, combo2, combo3}

    def test_zero_values_in_metrics(self):
        """Test Pareto analysis with zero values in metrics."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 0.0},  # Zero throughput
                "time_to_first_token_p99": {"mean": 50.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 0.0},  # Zero latency
            },
            combo3: {
                "request_throughput_avg": {"mean": 50.0},
                "time_to_first_token_p99": {"mean": 25.0},
            },
        }

        result = identify_pareto_optimal(per_combination_stats)
        # Config 2 dominates 1 (higher throughput, lower latency)
        # Config 2 dominates 3 (higher throughput, lower latency)
        assert result == [combo2]

    def test_mixed_domination_patterns(self):
        """Test complex domination patterns with multiple Pareto optimal points."""
        # Create a scenario with multiple clusters of Pareto optimal points
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        combo4 = ParameterCombination({"concurrency": 40})
        combo5 = ParameterCombination({"concurrency": 50})
        combo6 = ParameterCombination({"concurrency": 60})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 50.0},
                "time_to_first_token_p99": {"mean": 20.0},
            },  # Pareto optimal (best latency)
            combo2: {
                "request_throughput_avg": {"mean": 45.0},
                "time_to_first_token_p99": {"mean": 25.0},
            },  # Dominated by 1
            combo3: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 40.0},
            },  # Pareto optimal
            combo4: {
                "request_throughput_avg": {"mean": 95.0},
                "time_to_first_token_p99": {"mean": 45.0},
            },  # Dominated by 3
            combo5: {
                "request_throughput_avg": {"mean": 150.0},
                "time_to_first_token_p99": {"mean": 60.0},
            },  # Pareto optimal
            combo6: {
                "request_throughput_avg": {"mean": 200.0},
                "time_to_first_token_p99": {"mean": 80.0},
            },  # Pareto optimal (best throughput)
        }

        result = identify_pareto_optimal(per_combination_stats)
        # Configs 1, 3, 5, 6 form the Pareto frontier
        assert set(result) == {combo1, combo3, combo5, combo6}

    def test_identify_pareto_optimal_2d_drops_dominated_combo(self):
        """HEAD-style 2-D realistic frontier: c20 dominates c30."""
        c10 = ParameterCombination({"concurrency": 10})
        c20 = ParameterCombination({"concurrency": 20})
        c30 = ParameterCombination({"concurrency": 30})
        stats = {
            c10: {
                "request_throughput_avg": _stat(100.0),
                "time_to_first_token_p99": _stat(50.0),
            },
            c20: {
                "request_throughput_avg": _stat(180.0),
                "time_to_first_token_p99": _stat(75.0),
            },
            c30: {
                "request_throughput_avg": _stat(170.0),
                "time_to_first_token_p99": _stat(90.0),
            },
        }

        pareto = identify_pareto_optimal(stats)

        assert {c.parameters["concurrency"] for c in pareto} == {10, 20}
        assert c30 not in pareto


class TestSweepAnalyzer:
    """Tests for SweepAnalyzer class."""

    def test_compute_returns_dict_with_required_keys(self):
        """Test that compute() returns a dict with all required keys."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 180.0},
                "time_to_first_token_p99": {"mean": 60.0},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Verify all required keys are present
        assert "metadata" in result
        assert "per_combination_metrics" in result
        assert "best_configurations" in result
        assert "pareto_optimal" in result

    def test_metadata_section_structure(self):
        """Test that metadata section has correct structure."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {"request_throughput_avg": {"mean": 100.0}},
            combo2: {"request_throughput_avg": {"mean": 180.0}},
            combo3: {"request_throughput_avg": {"mean": 260.0}},
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20, 30]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        metadata = result["metadata"]
        assert metadata["sweep_parameters"] == sweep_parameters
        assert metadata["num_combinations"] == 3

    def test_per_combination_metrics_structure(self):
        """Test that per_combination_metrics has correct structure."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {"request_throughput_avg": {"mean": 100.0}},
            combo2: {"request_throughput_avg": {"mean": 180.0}},
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        per_combination_metrics = result["per_combination_metrics"]
        assert len(per_combination_metrics) == 2
        assert all("parameters" in item for item in per_combination_metrics)
        assert all("metrics" in item for item in per_combination_metrics)

    def test_per_combination_metrics_preserves_stats_structure(self):
        """Test that per_combination_metrics preserves the original stats structure."""
        combo = ParameterCombination({"concurrency": 10})
        per_combination_stats = {
            combo: {
                "request_throughput_avg": {"mean": 100.0, "std": 5.0, "min": 95.0},
                "time_to_first_token_p99": {"mean": 50.0, "std": 2.0},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        metrics = result["per_combination_metrics"][0]["metrics"]
        assert metrics["request_throughput_avg"]["mean"] == 100.0
        assert metrics["request_throughput_avg"]["std"] == 5.0
        assert metrics["request_throughput_avg"]["min"] == 95.0
        assert metrics["time_to_first_token_p99"]["mean"] == 50.0
        assert metrics["time_to_first_token_p99"]["std"] == 2.0

    def test_pareto_optimal_uses_identify_pareto_optimal_function(self):
        """Test that pareto_optimal section uses identify_pareto_optimal()."""
        # Config 1: high throughput, high latency
        # Config 2: low throughput, low latency
        # Both should be Pareto optimal
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 100.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 50.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Both should be in pareto_optimal (as dicts)
        pareto_params = [item["concurrency"] for item in result["pareto_optimal"]]
        assert sorted(pareto_params) == [10, 20]

    def test_single_value_sweep(self):
        """Test compute() with single sweep value."""
        combo = ParameterCombination({"concurrency": 10})
        per_combination_stats = {
            combo: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        assert result["metadata"]["num_combinations"] == 1
        assert len(result["per_combination_metrics"]) == 1
        assert result["pareto_optimal"] == [{"concurrency": 10}]

    def test_empty_sweep_values(self):
        """Test compute() with empty sweep values."""
        result = SweepAnalyzer.compute({}, [])

        assert result["metadata"]["num_combinations"] == 1  # Empty product is 1
        assert result["per_combination_metrics"] == []
        assert result["pareto_optimal"] == []

    def test_best_configurations_is_dict(self):
        """Test that best_configurations is a dict."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {"request_throughput_avg": {"mean": 100.0}},
            combo2: {"request_throughput_avg": {"mean": 180.0}},
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        assert isinstance(result["best_configurations"], dict)

    def test_compute_is_static_method(self):
        """Test that compute() is a static method (can be called without instance)."""
        combo = ParameterCombination({"concurrency": 10})
        per_combination_stats = {
            combo: {"request_throughput_avg": {"mean": 100.0}},
        }
        sweep_parameters = [{"name": "concurrency", "values": [10]}]

        # Should be callable without creating an instance
        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)
        assert result is not None

    def test_multiple_sweep_parameters(self):
        """Test with multiple sweep parameters."""
        combo1 = ParameterCombination({"concurrency": 10, "request_rate": 100})
        combo2 = ParameterCombination({"concurrency": 10, "request_rate": 200})
        combo3 = ParameterCombination({"concurrency": 20, "request_rate": 100})
        combo4 = ParameterCombination({"concurrency": 20, "request_rate": 200})
        per_combination_stats = {
            combo1: {"request_throughput_avg": {"mean": 100.0}},
            combo2: {"request_throughput_avg": {"mean": 180.0}},
            combo3: {"request_throughput_avg": {"mean": 150.0}},
            combo4: {"request_throughput_avg": {"mean": 200.0}},
        }
        sweep_parameters = [
            {"name": "concurrency", "values": [10, 20]},
            {"name": "request_rate", "values": [100, 200]},
        ]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        assert result["metadata"]["num_combinations"] == 4
        assert len(result["per_combination_metrics"]) == 4

    def test_compute_with_complex_stats_structure(self):
        """Test compute() preserves complex nested stats structure."""
        combo = ParameterCombination({"concurrency": 10})
        per_combination_stats = {
            combo: {
                "request_throughput_avg": {
                    "mean": 100.0,
                    "std": 5.0,
                    "min": 95.0,
                    "max": 108.0,
                    "cv": 0.05,
                    "ci_low": 94.3,
                    "ci_high": 106.7,
                    "unit": "requests/sec",
                },
                "time_to_first_token_p99": {
                    "mean": 50.0,
                    "std": 2.0,
                    "unit": "ms",
                },
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        metrics = result["per_combination_metrics"][0]["metrics"]
        # Verify all nested fields are preserved
        assert metrics["request_throughput_avg"]["mean"] == 100.0
        assert metrics["request_throughput_avg"]["std"] == 5.0
        assert metrics["request_throughput_avg"]["min"] == 95.0
        assert metrics["request_throughput_avg"]["max"] == 108.0
        assert metrics["request_throughput_avg"]["cv"] == 0.05
        assert metrics["request_throughput_avg"]["ci_low"] == 94.3
        assert metrics["request_throughput_avg"]["ci_high"] == 106.7
        assert metrics["request_throughput_avg"]["unit"] == "requests/sec"

    def test_sweep_analyzer_compute_1d_concurrency_sweep_returns_full_schema(self):
        """HEAD-style: 1-D sweep over realistic throughput-vs-TTFT shape."""
        c10 = ParameterCombination({"concurrency": 10})
        c20 = ParameterCombination({"concurrency": 20})
        c30 = ParameterCombination({"concurrency": 30})
        stats = {
            c10: {
                "request_throughput_avg": _stat(100.0),
                "time_to_first_token_p99": _stat(50.0),
            },
            c20: {
                "request_throughput_avg": _stat(180.0),
                "time_to_first_token_p99": _stat(80.0),
            },
            c30: {
                "request_throughput_avg": _stat(170.0),
                "time_to_first_token_p99": _stat(95.0),
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20, 30]}]

        result = SweepAnalyzer.compute(stats, sweep_parameters)

        assert result["metadata"]["num_combinations"] == 3
        assert result["metadata"]["sweep_parameters"] == sweep_parameters

        per_combo = result["per_combination_metrics"]
        assert len(per_combo) == 3
        assert [entry["parameters"]["concurrency"] for entry in per_combo] == [
            10,
            20,
            30,
        ]

        best = result["best_configurations"]
        assert best["best_throughput"]["parameters"] == {"concurrency": 20}
        assert best["best_throughput"]["metric"] == 180.0
        assert best["best_latency_p99"]["parameters"] == {"concurrency": 10}

        pareto = result["pareto_optimal"]
        assert {tuple(p.items()) for p in pareto} == {
            (("concurrency", 10),),
            (("concurrency", 20),),
        }

    def test_sweep_analyzer_compute_empty_stats_returns_empty_blocks(self):
        """HEAD-style: empty per-combination stats with metadata still computed."""
        result = SweepAnalyzer.compute({}, [{"name": "concurrency", "values": [10]}])

        assert result["metadata"]["num_combinations"] == 1
        assert result["per_combination_metrics"] == []
        assert result["best_configurations"] == {}
        assert result["pareto_optimal"] == []


class TestBestConfigurations:
    """Tests for best_configurations section in SweepAnalyzer.compute()."""

    def test_best_throughput_identified(self):
        """Test that best throughput configuration is correctly identified."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        combo4 = ParameterCombination({"concurrency": 40})
        per_combination_stats = {
            combo1: {"request_throughput_avg": {"mean": 100.0}},
            combo2: {"request_throughput_avg": {"mean": 180.0}},
            combo3: {"request_throughput_avg": {"mean": 260.0}},
            combo4: {"request_throughput_avg": {"mean": 350.2}},  # Best
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20, 30, 40]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        best_throughput = result["best_configurations"]["best_throughput"]
        assert best_throughput["parameters"] == {"concurrency": 40}
        assert best_throughput["metric"] == 350.2

    def test_best_latency_identified(self):
        """Test that best latency configuration is correctly identified."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        combo4 = ParameterCombination({"concurrency": 40})
        per_combination_stats = {
            combo1: {"time_to_first_token_p99": {"mean": 120.5}},  # Best
            combo2: {"time_to_first_token_p99": {"mean": 150.0}},
            combo3: {"time_to_first_token_p99": {"mean": 180.0}},
            combo4: {"time_to_first_token_p99": {"mean": 200.0}},
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20, 30, 40]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        best_latency = result["best_configurations"]["best_latency_p99"]
        assert best_latency["parameters"] == {"concurrency": 10}
        assert best_latency["metric"] == 120.5

    def test_best_configurations_with_both_metrics(self):
        """Test best configurations when both throughput and latency are present."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},  # Best latency
            },
            combo2: {
                "request_throughput_avg": {"mean": 180.0},
                "time_to_first_token_p99": {"mean": 60.0},
            },
            combo3: {
                "request_throughput_avg": {"mean": 350.2},  # Best throughput
                "time_to_first_token_p99": {"mean": 80.0},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20, 30]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Best throughput at concurrency 30
        assert result["best_configurations"]["best_throughput"]["parameters"] == {
            "concurrency": 30
        }
        assert result["best_configurations"]["best_throughput"]["metric"] == 350.2

        # Best latency at concurrency 10
        assert result["best_configurations"]["best_latency_p99"]["parameters"] == {
            "concurrency": 10
        }
        assert result["best_configurations"]["best_latency_p99"]["metric"] == 50.0

    def test_best_configurations_includes_units(self):
        """Test that best configurations include unit fields."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 50.0, "unit": "ms"},
            },
            combo2: {
                "request_throughput_avg": {"mean": 180.0, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 60.0, "unit": "ms"},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Check units are included
        assert (
            result["best_configurations"]["best_throughput"]["unit"] == "requests/sec"
        )
        assert result["best_configurations"]["best_latency_p99"]["unit"] == "ms"

    def test_best_configurations_default_units_when_missing(self):
        """Test that default units are used when not present in stats."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},  # No unit
                "time_to_first_token_p99": {"mean": 50.0},  # No unit
            },
            combo2: {
                "request_throughput_avg": {"mean": 180.0},
                "time_to_first_token_p99": {"mean": 60.0},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Check default units are used
        assert (
            result["best_configurations"]["best_throughput"]["unit"] == "requests/sec"
        )
        assert result["best_configurations"]["best_latency_p99"]["unit"] == "ms"

    def test_best_configurations_empty_when_no_stats(self):
        """Test that best_configurations is empty dict when no stats provided."""
        result = SweepAnalyzer.compute({}, [])

        assert result["best_configurations"] == {}

    def test_best_configurations_only_throughput_when_latency_missing(self):
        """Test that only best_throughput is included when latency metric is missing."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {"request_throughput_avg": {"mean": 100.0}},
            combo2: {"request_throughput_avg": {"mean": 180.0}},
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Should have best_throughput
        assert "best_throughput" in result["best_configurations"]
        assert result["best_configurations"]["best_throughput"]["parameters"] == {
            "concurrency": 20
        }

        # Should NOT have best_latency_p99
        assert "best_latency_p99" not in result["best_configurations"]

    def test_best_configurations_only_latency_when_throughput_missing(self):
        """Test that only best_latency_p99 is included when throughput metric is missing."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {"time_to_first_token_p99": {"mean": 50.0}},
            combo2: {"time_to_first_token_p99": {"mean": 60.0}},
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Should have best_latency_p99
        assert "best_latency_p99" in result["best_configurations"]
        assert result["best_configurations"]["best_latency_p99"]["parameters"] == {
            "concurrency": 10
        }

        # Should NOT have best_throughput
        assert "best_throughput" not in result["best_configurations"]

    def test_best_configurations_handles_partial_metric_presence(self):
        """Test that best configurations handles case where metric is missing in some values."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
            combo2: {
                "request_throughput_avg": {"mean": 180.0},
                # Missing time_to_first_token_p99
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Should have best_throughput (present in all)
        assert "best_throughput" in result["best_configurations"]

        # Should NOT have best_latency_p99 (not present in all)
        assert "best_latency_p99" not in result["best_configurations"]

    def test_best_configurations_single_value(self):
        """Test best configurations with single sweep value."""
        combo = ParameterCombination({"concurrency": 10})
        per_combination_stats = {
            combo: {
                "request_throughput_avg": {"mean": 100.0},
                "time_to_first_token_p99": {"mean": 50.0},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Single value is best for both
        assert result["best_configurations"]["best_throughput"]["parameters"] == {
            "concurrency": 10
        }
        assert result["best_configurations"]["best_latency_p99"]["parameters"] == {
            "concurrency": 10
        }

    def test_best_configurations_structure(self):
        """Test that best configurations have correct structure with parameters, metric, and unit."""
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100.0, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 50.0, "unit": "ms"},
            },
            combo2: {
                "request_throughput_avg": {"mean": 180.0, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 60.0, "unit": "ms"},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Check structure of best_throughput
        best_throughput = result["best_configurations"]["best_throughput"]
        assert "parameters" in best_throughput
        assert "metric" in best_throughput
        assert "unit" in best_throughput
        assert isinstance(best_throughput["parameters"], dict)
        assert isinstance(best_throughput["metric"], float)
        assert isinstance(best_throughput["unit"], str)

        # Check structure of best_latency_p99
        best_latency = result["best_configurations"]["best_latency_p99"]
        assert "parameters" in best_latency
        assert "metric" in best_latency
        assert "unit" in best_latency
        assert isinstance(best_latency["parameters"], dict)
        assert isinstance(best_latency["metric"], float)
        assert isinstance(best_latency["unit"], str)

    def test_best_configurations_realistic_scenario(self):
        """Test best configurations with realistic sweep data."""
        # Realistic scenario: throughput increases with concurrency, latency also increases
        combo1 = ParameterCombination({"concurrency": 10})
        combo2 = ParameterCombination({"concurrency": 20})
        combo3 = ParameterCombination({"concurrency": 30})
        combo4 = ParameterCombination({"concurrency": 40})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 95.5, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 45.2, "unit": "ms"},  # Best latency
            },
            combo2: {
                "request_throughput_avg": {"mean": 175.3, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 52.8, "unit": "ms"},
            },
            combo3: {
                "request_throughput_avg": {"mean": 245.7, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 68.5, "unit": "ms"},
            },
            combo4: {
                "request_throughput_avg": {
                    "mean": 298.2,
                    "unit": "requests/sec",
                },  # Best throughput
                "time_to_first_token_p99": {"mean": 95.3, "unit": "ms"},
            },
        }
        sweep_parameters = [{"name": "concurrency", "values": [10, 20, 30, 40]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        # Best throughput at highest concurrency
        best_throughput = result["best_configurations"]["best_throughput"]
        assert best_throughput["parameters"] == {"concurrency": 40}
        assert best_throughput["metric"] == 298.2
        assert best_throughput["unit"] == "requests/sec"

        # Best latency at lowest concurrency
        best_latency = result["best_configurations"]["best_latency_p99"]
        assert best_latency["parameters"] == {"concurrency": 10}
        assert best_latency["metric"] == 45.2
        assert best_latency["unit"] == "ms"


class TestSweepAnalyzerLatencyResolution:
    """Tests for latency metric resolution in sweep aggregation."""

    def test_time_to_first_token_p99_recognized_for_best_latency(self):
        """time_to_first_token_p99 is the canonical TTFT key; sweep must recognize it."""
        combo1 = ParameterCombination({"concurrency": 2})
        combo2 = ParameterCombination({"concurrency": 4})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 5.0, "unit": "ms"},
            },
            combo2: {
                "request_throughput_avg": {"mean": 180, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 8.0, "unit": "ms"},
            },
        }
        sweep_params = [{"name": "concurrency", "values": [2, 4]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_params)

        assert "best_latency_p99" in result["best_configurations"]
        assert result["best_configurations"]["best_latency_p99"]["metric"] == 5.0

    def test_time_to_first_token_p99_used_for_pareto(self):
        """Pareto analysis should work with time_to_first_token_p99."""
        combo1 = ParameterCombination({"concurrency": 2})
        combo2 = ParameterCombination({"concurrency": 4})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 5.0, "unit": "ms"},
            },
            combo2: {
                "request_throughput_avg": {"mean": 180, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 8.0, "unit": "ms"},
            },
        }
        sweep_params = [{"name": "concurrency", "values": [2, 4]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_params)

        assert len(result["pareto_optimal"]) == 2

    def test_ttft_preferred_over_request_latency_when_both_present(self):
        """When both TTFT and request_latency exist, TTFT should be used."""
        combo1 = ParameterCombination({"concurrency": 2})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100, "unit": "requests/sec"},
                "time_to_first_token_p99": {"mean": 3.0, "unit": "ms"},
                "request_latency_p99": {"mean": 10.0, "unit": "ms"},
            },
        }
        sweep_params = [{"name": "concurrency", "values": [2]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_params)

        assert result["best_configurations"]["best_latency_p99"]["metric"] == 3.0

    def test_request_latency_p99_fallback_for_non_streaming(self):
        """Non-streaming endpoints without TTFT fall back to request_latency_p99."""
        combo1 = ParameterCombination({"concurrency": 2})
        per_combination_stats = {
            combo1: {
                "request_throughput_avg": {"mean": 100, "unit": "requests/sec"},
                "request_latency_p99": {"mean": 10.0, "unit": "ms"},
            },
        }
        sweep_params = [{"name": "concurrency", "values": [2]}]

        result = SweepAnalyzer.compute(per_combination_stats, sweep_params)

        assert result["best_configurations"]["best_latency_p99"]["metric"] == 10.0

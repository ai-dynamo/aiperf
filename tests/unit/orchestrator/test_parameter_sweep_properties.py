# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for parameter sweeping correctness properties.

This module tests the correctness properties defined in the parameter-sweeping design document.
Each test validates a specific property to ensure the implementation meets requirements.
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.aggregation.sweep import (
    identify_pareto_optimal,
)
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import (
    FixedTrialsStrategy,
    ParameterSweepStrategy,
)

# =============================================================================
# Test Helpers
# =============================================================================


def make_endpoint(**kwargs) -> EndpointConfig:
    """Create an EndpointConfig with sensible defaults for testing."""
    if "url" in kwargs:
        kwargs["urls"] = [kwargs.pop("url")]
    return EndpointConfig(
        model_names=kwargs.pop("model_names", ["test-model"]),
        custom_endpoint=kwargs.pop("custom_endpoint", "test"),
        **kwargs,
    )


def make_config(endpoint=None, **kwargs) -> UserConfig:
    """Create a UserConfig with sensible defaults for testing."""
    return UserConfig(endpoint=endpoint or make_endpoint(), **kwargs)


def make_run_result(label: str, concurrency: int | None = None) -> RunResult:
    """Create a RunResult for testing."""
    metadata = {}
    if concurrency is not None:
        metadata["concurrency"] = concurrency

    # Use portable temp directory instead of hardcoded /tmp
    temp_dir = Path(tempfile.gettempdir())

    return RunResult(
        label=label,
        success=True,
        summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
        artifacts_path=temp_dir / label,
        metadata=metadata,
    )


# =============================================================================
# Property 1: Concurrency List Parsing
# =============================================================================


class TestProperty1ConcurrencyListParsing:
    """Test Property 1: Concurrency List Parsing.

    **Validates: Requirements 1.1**

    For any comma-separated string of valid integers, parsing should produce
    a list containing those integers in order.
    """

    @pytest.mark.parametrize(
        "concurrency_list,expected",
        [
            ([10, 20, 30], [10, 20, 30]),
            ([5], [5]),
            ([100, 50, 25, 10], [100, 50, 25, 10]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ],
    )
    def test_concurrency_list_parsing_preserves_order(self, concurrency_list, expected):
        """Test that concurrency list parsing preserves order."""
        config = make_config()
        config.loadgen.concurrency = concurrency_list

        assert config.loadgen.concurrency == expected
        assert isinstance(config.loadgen.concurrency, list)

    def test_loadgen_concurrency_single_int_assignment_preserves_int_type(self):
        """Test that single concurrency value remains as integer (backward compatible)."""
        config = make_config()
        config.loadgen.concurrency = 10

        assert config.loadgen.concurrency == 10
        assert isinstance(config.loadgen.concurrency, int)


# =============================================================================
# Property 2: Invalid Input Rejection
# =============================================================================


class TestProperty2InvalidInputRejection:
    """Test Property 2: Invalid Input Rejection.

    **Validates: Requirements 1.3, 1.4**

    For any concurrency list containing non-integer values or values less than 1,
    the validation should reject the input with a clear error message.
    """

    @pytest.mark.parametrize(
        "invalid_values",
        [
            [0, 10, 20],
            [-1, 10],
            [10, -5, 20],
            [0],
        ],
    )
    def test_rejects_values_less_than_one(self, invalid_values):
        """Test that concurrency values less than 1 are rejected."""
        config = make_config()
        config.loadgen.concurrency = invalid_values

        with pytest.raises(
            ValueError,
            match="Invalid concurrency (list|values)|Parameter sweep requires at least 2 values",
        ):
            config.model_validate(config.model_dump())

    def test_rejects_zero_concurrency(self):
        """Test that zero concurrency is rejected."""
        config = make_config()
        config.loadgen.concurrency = 0

        with pytest.raises(
            ValueError, match="Invalid concurrency value|Concurrency must be at least 1"
        ):
            config.model_validate(config.model_dump())


# =============================================================================
# Property 3: Duplicate Values Allowed
# =============================================================================


class TestProperty3DuplicateValuesHandling:
    """Test Property 3: Duplicate Values Allowed.

    **Validates: Requirements 1.5**

    For any concurrency list with duplicate values, the system should execute
    benchmarks for each occurrence in the list.
    """

    def test_duplicate_values_create_multiple_runs(self):
        """Test that duplicate concurrency values result in multiple runs."""
        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 10, 30, 10],
        )

        # Should have 5 runs (including duplicates)
        assert len(strategy.parameter_values) == 5

        # Verify each value is used
        config = make_config()
        results = []

        for i in range(5):
            _new_config = strategy.get_next_config(config, results)
            results.append(make_run_result(f"run_{i}"))

        # Should have executed all 5 values
        assert len(results) == 5
        assert not strategy.should_continue(results)

    def test_duplicate_values_preserve_order(self):
        """Test that duplicate values preserve their order in execution."""
        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 10, 30],
        )

        config = make_config()
        results = []
        executed_values = []

        for i in range(4):
            new_config = strategy.get_next_config(config, results)
            executed_values.append(new_config.loadgen.concurrency)
            results.append(make_run_result(f"run_{i}"))

        assert executed_values == [10, 20, 10, 30]


# =============================================================================
# Property 8: Repeated Mode Execution Pattern
# NOTE: Tests moved to integration tests (test_parameter_sweep.py)
# =============================================================================


# =============================================================================
# Property 9: Independent Mode Execution Pattern
# NOTE: Tests moved to integration tests (test_parameter_sweep.py)
# =============================================================================


# =============================================================================
# Property 11: Seed Derivation Consistency
# =============================================================================


class TestProperty11SeedDerivationConsistency:
    """Test Property 11: Seed Derivation Consistency.

    **Validates: Requirements 7.1, 7.7**

    For any base seed and sweep configuration, the same base seed should
    always produce the same per-value seeds (reproducibility).
    """

    def test_same_base_seed_produces_same_derived_seeds(self):
        """Test that same base seed always produces same per-value seeds."""
        sweep_values = [10, 20, 30]
        base_seed = 42

        # First run
        strategy1 = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
            same_seed=False,
            auto_set_seed=True,
        )

        config1 = make_config()
        config1.input.random_seed = base_seed

        seeds1 = []
        results = []
        for i in range(len(sweep_values)):
            new_config = strategy1.get_next_config(config1, results)
            seeds1.append(new_config.input.random_seed)
            results.append(make_run_result(f"run_{i}"))

        # Second run with same base seed
        strategy2 = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
            same_seed=False,
            auto_set_seed=True,
        )

        config2 = make_config()
        config2.input.random_seed = base_seed

        seeds2 = []
        results = []
        for i in range(len(sweep_values)):
            new_config = strategy2.get_next_config(config2, results)
            seeds2.append(new_config.input.random_seed)
            results.append(make_run_result(f"run_{i}"))

        # Should produce identical seeds
        assert seeds1 == seeds2
        assert seeds1 == [42, 43, 44]

    def test_seed_derivation_formula(self):
        """Test that seed derivation follows base_seed + sweep_index formula."""
        sweep_values = [10, 20, 30, 40]
        base_seed = 100

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
            same_seed=False,
            auto_set_seed=True,
        )

        config = make_config()
        config.input.random_seed = base_seed

        results = []
        for i in range(len(sweep_values)):
            new_config = strategy.get_next_config(config, results)
            expected_seed = base_seed + i
            assert new_config.input.random_seed == expected_seed
            results.append(make_run_result(f"run_{i}"))


# =============================================================================
# Property 12: Same-Seed Mode
# =============================================================================


class TestProperty12SameSeedMode:
    """Test Property 12: Same-Seed Mode.

    **Validates: Requirements 7.4**

    For any sweep with --profile-run-sweep-same-seed, all sweep values
    should use the identical random seed.
    """

    def test_same_seed_mode_uses_identical_seed(self):
        """Test that same_seed=True uses identical seed for all values."""
        sweep_values = [10, 20, 30, 40]
        base_seed = 42

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
            same_seed=True,
            auto_set_seed=True,
        )

        config = make_config()
        config.input.random_seed = base_seed

        seeds = []
        results = []
        for i in range(len(sweep_values)):
            new_config = strategy.get_next_config(config, results)
            seeds.append(new_config.input.random_seed)
            results.append(make_run_result(f"run_{i}"))

        # All seeds should be identical to base seed
        assert all(seed == base_seed for seed in seeds)
        assert seeds == [42, 42, 42, 42]

    def test_same_seed_with_auto_set(self):
        """Test that same_seed works with auto-set base seed."""
        sweep_values = [10, 20, 30]

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
            same_seed=True,
            auto_set_seed=True,
        )

        config = make_config()
        config.input.random_seed = None  # Will be auto-set to 42

        seeds = []
        results = []
        for i in range(len(sweep_values)):
            new_config = strategy.get_next_config(config, results)
            seeds.append(new_config.input.random_seed)
            results.append(make_run_result(f"run_{i}"))

        # All seeds should be 42 (default)
        assert seeds == [42, 42, 42]


# =============================================================================
# Property 15: Cooldown Application
# =============================================================================


class TestProperty15CooldownApplication:
    """Test Property 15: Cooldown Application.

    **Validates: Requirements 10.1, 10.2, 10.3**

    For any sweep configuration with trial and sweep cooldowns, the correct
    cooldown duration should be applied based on position in the execution sequence.
    """

    def test_sweep_cooldown_applied_between_values(self):
        """Test that sweep cooldown is applied between sweep values."""
        sweep_strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 30],
            cooldown_seconds=5.0,
        )

        assert sweep_strategy.get_cooldown_seconds() == 5.0

    def test_trial_cooldown_applied_between_trials(self):
        """Test that trial cooldown is applied between confidence trials."""
        trial_strategy = FixedTrialsStrategy(
            num_trials=3,
            cooldown_seconds=10.0,
        )

        assert trial_strategy.get_cooldown_seconds() == 10.0

    def test_get_cooldown_seconds_independent_strategies_return_respective_values(self):
        """Test that trial and sweep cooldowns are independent."""
        sweep_strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20],
            cooldown_seconds=5.0,
        )

        trial_strategy = FixedTrialsStrategy(
            num_trials=2,
            cooldown_seconds=10.0,
        )

        assert sweep_strategy.get_cooldown_seconds() == 5.0
        assert trial_strategy.get_cooldown_seconds() == 10.0

    def test_get_cooldown_seconds_no_cooldown_specified_returns_zero(self):
        """Test that cooldown defaults to zero."""
        sweep_strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20],
        )

        assert sweep_strategy.get_cooldown_seconds() == 0.0


# =============================================================================
# Property 16: Backward Compatibility
# NOTE: Tests moved to integration tests (test_parameter_sweep.py)
# =============================================================================


# =============================================================================
# Property 17: UI Mode Validation
# NOTE: Feature not yet implemented - tests deferred
# =============================================================================


# =============================================================================
# Property-Based Tests (Using Hypothesis)
# =============================================================================


class TestProperty1PBT:
    """Property-Based Tests for Property 1: Concurrency List Parsing.

    **Validates: Requirements 1.1**

    For any comma-separated string of valid integers, parsing should produce
    a list containing those integers in order.
    """

    @given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_pbt_concurrency_list_preserves_order_and_values(self, concurrency_values):
        """Property 1: For any list of valid integers, config should preserve order and values.

        Feature: parameter-sweeping, Property 1: For any comma-separated string
        of valid integers, parsing should produce a list containing those integers in order.
        """
        config = make_config()
        config.loadgen.concurrency = concurrency_values

        # Verify list is preserved exactly
        assert config.loadgen.concurrency == concurrency_values
        assert isinstance(config.loadgen.concurrency, list)
        assert len(config.loadgen.concurrency) == len(concurrency_values)

        # Verify order is preserved
        for i, expected_value in enumerate(concurrency_values):
            assert config.loadgen.concurrency[i] == expected_value


class TestProperty2PBT:
    """Property-Based Tests for Property 2: Invalid Input Rejection.

    **Validates: Requirements 1.3, 1.4**

    For any concurrency list containing non-integer values or values less than 1,
    the validation should reject the input with a clear error message.
    """

    @given(
        st.lists(st.integers(min_value=1, max_value=100), min_size=0, max_size=10),
        st.integers(max_value=0),
        st.data(),
    )
    @settings(max_examples=100)
    def test_pbt_rejects_invalid_values(self, valid_values, invalid_value, data):
        """Property 2: For any list containing values < 1, validation should reject.

        Feature: parameter-sweeping, Property 2: For any concurrency list containing
        non-integer values or values less than 1, the validation should reject the
        input with a clear error message.
        """
        # Insert invalid value at random position using Hypothesis
        if valid_values:
            position = data.draw(st.integers(min_value=0, max_value=len(valid_values)))
            test_values = (
                valid_values[:position] + [invalid_value] + valid_values[position:]
            )
        else:
            test_values = [invalid_value]

        config = make_config()
        config.loadgen.concurrency = test_values

        # Should raise validation error
        with pytest.raises(
            ValueError,
            match="Invalid concurrency (list|values)|Parameter sweep requires at least 2 values",
        ):
            config.model_validate(config.model_dump())


class TestProperty8PBT:
    """Property-Based Tests for Property 8: Repeated Mode Execution Pattern.

    **Validates: Requirements 4.1, 4.2**

    For any sweep with repeated mode and N trials, the execution sequence
    should be [sweep], [sweep], ..., [sweep] (N times).
    """

    @given(
        st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=5),
        st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50)
    def test_pbt_repeated_mode_execution_order(self, sweep_values, num_trials):
        """Property 8: For any sweep with repeated mode, execution follows [sweep] × N pattern.

        Feature: parameter-sweeping, Property 8: For any sweep with repeated mode
        and N trials, the execution sequence should be [sweep], [sweep], ..., [sweep] (N times).
        """
        # Create strategies
        sweep_strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
        )

        trial_strategy = FixedTrialsStrategy(
            num_trials=num_trials,
        )

        # Simulate repeated mode: for trial in trials: for value in values
        config = make_config()
        executed_values = []

        trial_results = []
        for trial_idx in range(num_trials):
            trial_config = trial_strategy.get_next_config(config, trial_results)

            sweep_results = []
            for value_idx in range(len(sweep_values)):
                run_config = sweep_strategy.get_next_config(trial_config, sweep_results)
                executed_values.append(run_config.loadgen.concurrency)
                sweep_results.append(make_run_result(f"run_{trial_idx}_{value_idx}"))

            trial_results.append(sweep_results[-1])

        # Verify pattern: [sweep], [sweep], ..., [sweep]
        for trial in range(num_trials):
            start_idx = trial * len(sweep_values)
            end_idx = start_idx + len(sweep_values)
            assert executed_values[start_idx:end_idx] == sweep_values


class TestProperty9PBT:
    """Property-Based Tests for Property 9: Independent Mode Execution Pattern.

    **Validates: Requirements 4.3, 4.4**

    For any sweep with independent mode and N trials, the execution sequence
    should be N×[value1], N×[value2], ..., N×[valueK].
    """

    @given(
        st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=5),
        st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50)
    def test_pbt_independent_mode_execution_order(self, sweep_values, num_trials):
        """Property 9: For any sweep with independent mode, execution follows N×[value] pattern.

        Feature: parameter-sweeping, Property 9: For any sweep with independent mode
        and N trials, the execution sequence should be N×[value1], N×[value2], ..., N×[valueK].
        """
        # Create strategies
        sweep_strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
        )

        trial_strategy = FixedTrialsStrategy(
            num_trials=num_trials,
        )

        # Simulate independent mode: for value in values: for trial in trials
        config = make_config()
        executed_values = []

        sweep_results = []
        for value_idx in range(len(sweep_values)):
            value_config = sweep_strategy.get_next_config(config, sweep_results)

            trial_results = []
            for trial_idx in range(num_trials):
                run_config = trial_strategy.get_next_config(value_config, trial_results)
                executed_values.append(run_config.loadgen.concurrency)
                trial_results.append(make_run_result(f"run_{value_idx}_{trial_idx}"))

            sweep_results.append(trial_results[-1])

        # Verify pattern: N×[value1], N×[value2], ..., N×[valueK]
        for value_idx, value in enumerate(sweep_values):
            start_idx = value_idx * num_trials
            end_idx = start_idx + num_trials
            # All executions for this value should have the same concurrency
            assert all(v == value for v in executed_values[start_idx:end_idx])


class TestProperty11PBT:
    """Property-Based Tests for Property 11: Seed Derivation Consistency.

    **Validates: Requirements 7.1, 7.7**

    For any base seed and sweep configuration, the same base seed should
    always produce the same per-value seeds (reproducibility).
    """

    @given(
        st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=10),
        st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_pbt_seed_derivation_reproducibility(self, sweep_values, base_seed):
        """Property 11: For any base seed, same seed always produces same derived seeds.

        Feature: parameter-sweeping, Property 11: For any base seed and sweep configuration,
        the same base seed should always produce the same per-value seeds (reproducibility).
        """
        # First run
        strategy1 = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
            same_seed=False,
            auto_set_seed=True,
        )

        config1 = make_config()
        config1.input.random_seed = base_seed

        seeds1 = []
        results = []
        for i in range(len(sweep_values)):
            new_config = strategy1.get_next_config(config1, results)
            seeds1.append(new_config.input.random_seed)
            results.append(make_run_result(f"run_{i}"))

        # Second run with same base seed
        strategy2 = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
            same_seed=False,
            auto_set_seed=True,
        )

        config2 = make_config()
        config2.input.random_seed = base_seed

        seeds2 = []
        results = []
        for i in range(len(sweep_values)):
            new_config = strategy2.get_next_config(config2, results)
            seeds2.append(new_config.input.random_seed)
            results.append(make_run_result(f"run_{i}"))

        # Should produce identical seeds
        assert seeds1 == seeds2

        # Verify derivation formula: base_seed + index
        expected_seeds = [base_seed + i for i in range(len(sweep_values))]
        assert seeds1 == expected_seeds

    @given(
        st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=10),
        st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_pbt_same_seed_mode_consistency(self, sweep_values, base_seed):
        """Property 11 (variant): With same_seed=True, all values use identical seed.

        Feature: parameter-sweeping, Property 11: Seed derivation should be consistent
        and reproducible across all sweep modes.
        """
        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=sweep_values,
            same_seed=True,
            auto_set_seed=True,
        )

        config = make_config()
        config.input.random_seed = base_seed

        seeds = []
        results = []
        for i in range(len(sweep_values)):
            new_config = strategy.get_next_config(config, results)
            seeds.append(new_config.input.random_seed)
            results.append(make_run_result(f"run_{i}"))

        # All seeds should be identical to base seed
        assert all(seed == base_seed for seed in seeds)


class TestProperty13PBT:
    """Property-Based Tests for Property 13: Pareto Optimal Identification.

    **Validates: Requirements 5.6, 5.7**

    For any set of sweep results, a configuration should be identified as Pareto optimal
    if and only if no other configuration has both higher throughput AND lower latency.
    """

    @given(
        st.lists(
            st.tuples(
                st.floats(
                    min_value=1.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                st.floats(
                    min_value=1.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=100)
    def test_pbt_pareto_optimal_correctness(self, metrics):
        """Property 13: Pareto optimal points are not dominated by any other point.

        Feature: parameter-sweeping, Property 13: For any set of sweep results,
        a configuration should be identified as Pareto optimal if and only if no other
        configuration has both higher throughput AND lower latency.
        """
        from aiperf.orchestrator.aggregation.sweep import ParameterCombination

        # Create per-combination stats from generated metrics
        per_combination_stats = {}
        combos = []
        for i, (throughput, latency) in enumerate(metrics):
            combo = ParameterCombination({"config_id": i})
            combos.append(combo)
            per_combination_stats[combo] = {
                "request_throughput_avg": {"mean": throughput},
                "time_to_first_token_p99": {"mean": latency},
            }

        # Identify Pareto optimal points
        pareto = identify_pareto_optimal(per_combination_stats)

        # Verify: each Pareto optimal point is not dominated
        for p in pareto:
            p_throughput = per_combination_stats[p]["request_throughput_avg"]["mean"]
            p_latency = per_combination_stats[p]["time_to_first_token_p99"]["mean"]

            for other in per_combination_stats:
                if other == p:
                    continue

                other_throughput = per_combination_stats[other][
                    "request_throughput_avg"
                ]["mean"]
                other_latency = per_combination_stats[other]["time_to_first_token_p99"][
                    "mean"
                ]

                # Should not be strictly better on both (domination)
                # Domination: better or equal on all, strictly better on at least one
                better_throughput = other_throughput > p_throughput
                better_latency = other_latency < p_latency
                equal_throughput = other_throughput == p_throughput
                equal_latency = other_latency == p_latency

                # Other dominates p if:
                # (better_throughput OR equal_throughput) AND (better_latency OR equal_latency)
                # AND at least one is strictly better
                dominates = (
                    (better_throughput or equal_throughput)
                    and (better_latency or equal_latency)
                    and (better_throughput or better_latency)
                )

                assert not dominates, (
                    f"Pareto point {p} is dominated by {other}: "
                    f"p=({p_throughput}, {p_latency}), "
                    f"other=({other_throughput}, {other_latency})"
                )

        # Verify: each non-Pareto point is dominated by at least one Pareto point
        non_pareto = [combo for combo in per_combination_stats if combo not in pareto]
        for np in non_pareto:
            np_throughput = per_combination_stats[np]["request_throughput_avg"]["mean"]
            np_latency = per_combination_stats[np]["time_to_first_token_p99"]["mean"]

            # Should be dominated by at least one point
            is_dominated = False
            for other in per_combination_stats:
                if other == np:
                    continue

                other_throughput = per_combination_stats[other][
                    "request_throughput_avg"
                ]["mean"]
                other_latency = per_combination_stats[other]["time_to_first_token_p99"][
                    "mean"
                ]

                better_throughput = other_throughput > np_throughput
                better_latency = other_latency < np_latency
                equal_throughput = other_throughput == np_throughput
                equal_latency = other_latency == np_latency

                dominates = (
                    (better_throughput or equal_throughput)
                    and (better_latency or equal_latency)
                    and (better_throughput or better_latency)
                )

                if dominates:
                    is_dominated = True
                    break

            assert is_dominated, (
                f"Non-Pareto point {np} is not dominated by any point: "
                f"({np_throughput}, {np_latency})"
            )

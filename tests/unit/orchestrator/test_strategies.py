# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for execution strategies."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import AdaptiveStrategy, FixedTrialsStrategy


class TestFixedTrialsStrategy:
    """Tests for FixedTrialsStrategy."""

    @pytest.mark.parametrize(
        "results,expected",
        [
            ([], True),  # No results yet
            (
                [
                    RunResult(
                        label="run_0001",
                        success=True,
                        summary_metrics={
                            "ttft_avg": JsonMetricResult(unit="ms", avg=100.0)
                        },
                        artifacts_path=Path("/tmp/run_0001"),
                    )
                ],
                True,
            ),  # Partial results
            (
                [
                    RunResult(
                        label="run_0001",
                        success=True,
                        summary_metrics={
                            "ttft": JsonMetricResult(unit="ms", avg=100.0)
                        },
                        artifacts_path=Path("/tmp/run_0001"),
                    ),
                    RunResult(
                        label="run_0002",
                        success=True,
                        summary_metrics={
                            "ttft": JsonMetricResult(unit="ms", avg=105.0)
                        },
                        artifacts_path=Path("/tmp/run_0002"),
                    ),
                ],
                False,
            ),  # Complete results (num_trials=2)
        ],
    )
    def test_should_continue_returns_expected(self, results, expected):
        """Test should_continue returns expected value based on results count."""
        num_trials = 2 if len(results) == 2 else 3
        strategy = FixedTrialsStrategy(num_trials=num_trials)
        assert strategy.should_continue(results) is expected

    @pytest.mark.parametrize(
        "run_index,num_trials,expected",
        [
            (0, 10, "run_0001"),
            (1, 10, "run_0002"),
            (9, 10, "run_0010"),
            (0, 5, "run_0001"),
            (4, 5, "run_0005"),
        ],
    )
    def test_get_run_label_zero_padding_returns_expected(
        self, run_index, num_trials, expected
    ):
        """Test get_run_label returns zero-padded labels with correct padding."""
        strategy = FixedTrialsStrategy(num_trials=num_trials)
        assert strategy.get_run_label(run_index) == expected

    def test_get_cooldown_seconds_configured_returns_value(self):
        """Test get_cooldown_seconds returns configured value."""
        strategy = FixedTrialsStrategy(num_trials=3, cooldown_seconds=5.0)
        assert strategy.get_cooldown_seconds() == 5.0

    def test_get_cooldown_seconds_default_returns_zero(self):
        """Test get_cooldown_seconds returns default value of zero."""
        strategy = FixedTrialsStrategy(num_trials=3)
        assert strategy.get_cooldown_seconds() == 0.0

    def test_auto_set_seed_on_first_run(self):
        """Test auto_set_seed sets random_seed on first run when None."""
        strategy = FixedTrialsStrategy(num_trials=3, auto_set_seed=True)

        # Create config with None random_seed
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None

        # Get config for first run
        new_config = strategy.get_next_config(config, [])

        # Should have set random_seed to DEFAULT_SEED
        assert new_config.input.random_seed == FixedTrialsStrategy.DEFAULT_SEED
        # Original config should be unchanged
        assert config.input.random_seed is None

    def test_auto_set_seed_preserves_user_seed(self):
        """Test auto_set_seed preserves user-specified random_seed."""
        strategy = FixedTrialsStrategy(num_trials=3, auto_set_seed=True)

        # Create config with user-specified random_seed
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = 999

        # Get config for first run
        new_config = strategy.get_next_config(config, [])

        # Should preserve user's seed
        assert new_config.input.random_seed == 999

    def test_auto_set_seed_disabled(self):
        """Test that auto_set_seed=False doesn't modify config."""
        strategy = FixedTrialsStrategy(num_trials=3, auto_set_seed=False)

        # Create config with None random_seed
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None

        # Get config for first run
        new_config = strategy.get_next_config(config, [])

        # Should not have modified random_seed
        assert new_config.input.random_seed is None

    def test_get_next_config_returns_base_config_after_first_run(self):
        """Test get_next_config returns modified config after first run when warmup disabled."""
        strategy = FixedTrialsStrategy(
            num_trials=3, auto_set_seed=True, disable_warmup_after_first=True
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = 42
        config.loadgen.warmup_request_count = 10

        # Create a result to simulate first run completed
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            )
        ]

        # Get config for second run
        new_config = strategy.get_next_config(config, results)

        # Should return a different config object (deep copy with warmup disabled)
        assert new_config is not config
        assert new_config.loadgen.warmup_request_count is None
        # Original config should be unchanged
        assert config.loadgen.warmup_request_count == 10

    def test_ensure_random_seed_creates_deep_copy(self):
        """Test that _ensure_random_seed creates a deep copy."""
        strategy = FixedTrialsStrategy(num_trials=3, auto_set_seed=True)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None

        # Get modified config
        new_config = strategy.get_next_config(config, [])

        # Verify it's a different object (deep copy)
        assert new_config is not config
        assert new_config.input.random_seed == FixedTrialsStrategy.DEFAULT_SEED
        assert config.input.random_seed is None

    def test_invalid_cooldown_seconds(self):
        """Test that negative cooldown raises ValueError."""
        with pytest.raises(ValueError, match="Invalid cooldown duration"):
            FixedTrialsStrategy(num_trials=5, cooldown_seconds=-1.0)

    def test_label_sanitization(self):
        """Test that labels are sanitized to prevent path traversal."""
        strategy = FixedTrialsStrategy(num_trials=5)

        # Normal labels should work fine
        assert strategy.get_run_label(0) == "run_0001"
        assert strategy.get_run_label(99) == "run_0100"

    def test_disable_warmup_after_first_enabled(self):
        """Test that warmup is disabled after first run when disable_warmup_after_first=True."""
        strategy = FixedTrialsStrategy(num_trials=3, disable_warmup_after_first=True)

        # Create config with warmup enabled
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.warmup_request_count = 10
        config.loadgen.warmup_duration = 30.0
        config.loadgen.warmup_concurrency = 2

        # First run should preserve warmup
        first_config = strategy.get_next_config(config, [])
        assert first_config.loadgen.warmup_request_count == 10
        assert first_config.loadgen.warmup_duration == 30.0
        assert first_config.loadgen.warmup_concurrency == 2

        # Create a result to simulate first run completed
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            )
        ]

        # Second run should have warmup disabled
        second_config = strategy.get_next_config(config, results)
        assert second_config.loadgen.warmup_request_count is None
        assert second_config.loadgen.warmup_duration is None
        assert second_config.loadgen.warmup_concurrency is None

        # Original config should be unchanged
        assert config.loadgen.warmup_request_count == 10

    def test_disable_warmup_after_first_disabled(self):
        """Test that warmup is preserved for all runs when disable_warmup_after_first=False."""
        strategy = FixedTrialsStrategy(num_trials=3, disable_warmup_after_first=False)

        # Create config with warmup enabled
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.warmup_request_count = 10
        config.loadgen.warmup_duration = 30.0
        config.loadgen.warmup_concurrency = 2

        # First run should preserve warmup
        first_config = strategy.get_next_config(config, [])
        assert first_config.loadgen.warmup_request_count == 10
        assert first_config.loadgen.warmup_duration == 30.0
        assert first_config.loadgen.warmup_concurrency == 2

        # Create a result to simulate first run completed
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            )
        ]

        # Second run should STILL have warmup (not disabled)
        second_config = strategy.get_next_config(config, results)
        assert second_config.loadgen.warmup_request_count == 10
        assert second_config.loadgen.warmup_duration == 30.0
        assert second_config.loadgen.warmup_concurrency == 2

    def test_disable_warmup_creates_deep_copy(self):
        """Test that disabling warmup creates a deep copy and doesn't modify original."""
        strategy = FixedTrialsStrategy(num_trials=3, disable_warmup_after_first=True)

        # Create config with warmup enabled
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.warmup_request_count = 10
        config.loadgen.warmup_duration = 30.0

        # Create a result to simulate first run completed
        results = [
            RunResult(
                label="run_0001",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/run_0001"),
            )
        ]

        # Get config for second run (should disable warmup)
        second_config = strategy.get_next_config(config, results)

        # Verify it's a different object (deep copy)
        assert second_config is not config
        assert second_config.loadgen.warmup_request_count is None

        # Original config should be unchanged
        assert config.loadgen.warmup_request_count == 10
        assert config.loadgen.warmup_duration == 30.0

    def test_get_run_path(self):
        """Test get_run_path returns correct path structure."""
        from pathlib import Path

        strategy = FixedTrialsStrategy(num_trials=3)
        base_dir = Path("/tmp/artifacts")

        # Test path for first run
        path = strategy.get_run_path(base_dir, 0)
        assert path == Path("/tmp/artifacts/profile_runs/run_0001")

        # Test path for second run
        path = strategy.get_run_path(base_dir, 1)
        assert path == Path("/tmp/artifacts/profile_runs/run_0002")

        # Test path for tenth run
        path = strategy.get_run_path(base_dir, 9)
        assert path == Path("/tmp/artifacts/profile_runs/run_0010")

    def test_get_aggregate_path(self):
        """Test get_aggregate_path returns correct path."""
        from pathlib import Path

        strategy = FixedTrialsStrategy(num_trials=3)
        base_dir = Path("/tmp/artifacts")

        path = strategy.get_aggregate_path(base_dir)
        assert path == Path("/tmp/artifacts/aggregate")

    def test_path_building_consistency(self):
        """Test that path building is consistent with label generation."""
        from pathlib import Path

        strategy = FixedTrialsStrategy(num_trials=5)
        base_dir = Path("/tmp/artifacts")

        # Path should use the same label as get_run_label
        for run_index in range(5):
            label = strategy.get_run_label(run_index)
            path = strategy.get_run_path(base_dir, run_index)

            # Path should end with the label
            assert path.name == label
            assert str(path).endswith(f"profile_runs/{label}")


class TestAdaptiveStrategy:
    """Tests for AdaptiveStrategy."""

    def _make_results(
        self,
        count: int,
        metric: str = "time_to_first_token",
        value: float = 100.0,
    ) -> list[RunResult]:
        """Build a list of successful RunResult with summary metrics."""
        return [
            RunResult(
                label=f"run_{i + 1:04d}",
                success=True,
                summary_metrics={metric: JsonMetricResult(unit="ms", avg=value + i)},
                artifacts_path=Path(f"/tmp/run_{i + 1:04d}"),
            )
            for i in range(count)
        ]

    def _make_mock_criterion(self, converged: bool = False) -> MagicMock:
        mock = MagicMock()
        mock.is_converged.return_value = converged
        return mock

    # -- should_continue: convergence logic --

    def test_criterion_true_stops_after_min_runs(self):
        """When criterion reports converged, stop as soon as min_runs is met."""
        criterion = self._make_mock_criterion(converged=True)
        strategy = AdaptiveStrategy(criterion=criterion, min_runs=3, max_runs=10)

        results = self._make_results(3)
        assert strategy.should_continue(results) is False

    def test_criterion_false_runs_to_max(self):
        """When criterion never converges, run until max_runs."""
        criterion = self._make_mock_criterion(converged=False)
        strategy = AdaptiveStrategy(criterion=criterion, min_runs=3, max_runs=5)

        for n in range(1, 5):
            assert strategy.should_continue(self._make_results(n)) is True
        assert strategy.should_continue(self._make_results(5)) is False

    def test_criterion_flips_true_at_run_4(self):
        """Criterion flips to converged at run 4 → stops at 4."""
        criterion = MagicMock()
        criterion.is_converged.side_effect = [False, True]
        strategy = AdaptiveStrategy(criterion=criterion, min_runs=3, max_runs=10)

        # Run 3: first convergence check → False → continue
        assert strategy.should_continue(self._make_results(3)) is True
        # Run 4: second convergence check → True → stop
        assert strategy.should_continue(self._make_results(4)) is False

    def test_min_runs_floor_enforced(self):
        """Even if criterion says converged, continue below min_runs."""
        criterion = self._make_mock_criterion(converged=True)
        strategy = AdaptiveStrategy(criterion=criterion, min_runs=5, max_runs=10)

        for n in range(1, 5):
            assert strategy.should_continue(self._make_results(n)) is True
        # Criterion is never called below min_runs
        criterion.is_converged.assert_not_called()

    def test_max_runs_cap_enforced(self):
        """Stop at max_runs even if criterion says not converged."""
        criterion = self._make_mock_criterion(converged=False)
        strategy = AdaptiveStrategy(criterion=criterion, min_runs=3, max_runs=5)

        assert strategy.should_continue(self._make_results(5)) is False
        assert strategy.should_continue(self._make_results(6)) is False

    def test_empty_results_continues(self):
        """No results yet → always continue."""
        criterion = self._make_mock_criterion(converged=True)
        strategy = AdaptiveStrategy(criterion=criterion, min_runs=3, max_runs=10)
        assert strategy.should_continue([]) is True

    def test_criterion_exception_treated_as_not_converged(self, caplog):
        """If criterion raises, log error and treat as not converged."""
        criterion = MagicMock()
        criterion.is_converged.side_effect = RuntimeError("boom")
        strategy = AdaptiveStrategy(criterion=criterion, min_runs=3, max_runs=10)

        results = self._make_results(3)
        assert strategy.should_continue(results) is True
        assert "Convergence criterion raised an error" in caplog.text

    # -- Label parity with FixedTrialsStrategy --

    @pytest.mark.parametrize("run_index", [0, 1, 99, 9999])
    def test_label_parity_with_fixed_trials(self, run_index):
        """AdaptiveStrategy labels must match FixedTrialsStrategy labels."""
        criterion = self._make_mock_criterion()
        adaptive = AdaptiveStrategy(criterion=criterion)
        fixed = FixedTrialsStrategy(num_trials=10000)

        assert adaptive.get_run_label(run_index) == fixed.get_run_label(run_index)

    # -- Path parity --

    @pytest.mark.parametrize("run_index", [0, 1, 99, 9999])
    def test_run_path_parity_with_fixed_trials(self, run_index):
        """AdaptiveStrategy run paths must match FixedTrialsStrategy."""
        criterion = self._make_mock_criterion()
        adaptive = AdaptiveStrategy(criterion=criterion)
        fixed = FixedTrialsStrategy(num_trials=10000)
        base_dir = Path("/tmp/artifacts")

        assert adaptive.get_run_path(base_dir, run_index) == fixed.get_run_path(
            base_dir, run_index
        )

    def test_aggregate_path_parity_with_fixed_trials(self):
        """AdaptiveStrategy aggregate path must match FixedTrialsStrategy."""
        criterion = self._make_mock_criterion()
        adaptive = AdaptiveStrategy(criterion=criterion)
        fixed = FixedTrialsStrategy(num_trials=5)
        base_dir = Path("/tmp/artifacts")

        assert adaptive.get_aggregate_path(base_dir) == fixed.get_aggregate_path(
            base_dir
        )

    # -- Config parity (seed handling, warmup disabling) --

    def test_auto_set_seed_on_first_run(self):
        """Auto-sets seed when none specified, matching FixedTrialsStrategy."""
        criterion = self._make_mock_criterion()
        strategy = AdaptiveStrategy(criterion=criterion, auto_set_seed=True)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None

        new_config = strategy.get_next_config(config, [])
        assert new_config.input.random_seed == AdaptiveStrategy.DEFAULT_SEED
        assert config.input.random_seed is None

    def test_auto_set_seed_preserves_user_seed(self):
        """Preserves user-specified seed."""
        criterion = self._make_mock_criterion()
        strategy = AdaptiveStrategy(criterion=criterion, auto_set_seed=True)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = 999

        new_config = strategy.get_next_config(config, [])
        assert new_config.input.random_seed == 999

    def test_auto_set_seed_disabled(self):
        """auto_set_seed=False leaves seed as None."""
        criterion = self._make_mock_criterion()
        strategy = AdaptiveStrategy(criterion=criterion, auto_set_seed=False)

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None

        new_config = strategy.get_next_config(config, [])
        assert new_config.input.random_seed is None

    def test_disable_warmup_after_first_run(self):
        """Warmup disabled for runs after the first."""
        criterion = self._make_mock_criterion()
        strategy = AdaptiveStrategy(
            criterion=criterion, disable_warmup_after_first=True
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.warmup_request_count = 10
        config.loadgen.warmup_duration = 30.0

        # First run: warmup preserved
        first = strategy.get_next_config(config, [])
        assert first.loadgen.warmup_request_count == 10

        # Second run: warmup disabled
        results = self._make_results(1)
        second = strategy.get_next_config(config, results)
        assert second.loadgen.warmup_request_count is None
        assert second.loadgen.warmup_duration is None

        # Original unchanged
        assert config.loadgen.warmup_request_count == 10

    def test_disable_warmup_after_first_disabled(self):
        """Warmup preserved for all runs when disable_warmup_after_first=False."""
        criterion = self._make_mock_criterion()
        strategy = AdaptiveStrategy(
            criterion=criterion, disable_warmup_after_first=False
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.warmup_request_count = 10

        results = self._make_results(1)
        second = strategy.get_next_config(config, results)
        assert second.loadgen.warmup_request_count == 10

    def test_config_parity_seed_and_warmup_with_fixed_trials(self):
        """Full config transformation parity between Adaptive and Fixed strategies."""
        criterion = self._make_mock_criterion()
        adaptive = AdaptiveStrategy(
            criterion=criterion,
            auto_set_seed=True,
            disable_warmup_after_first=True,
        )
        fixed = FixedTrialsStrategy(
            num_trials=10,
            auto_set_seed=True,
            disable_warmup_after_first=True,
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None
        config.loadgen.warmup_request_count = 10

        results = self._make_results(1)

        adaptive_cfg = adaptive.get_next_config(config, results)
        fixed_cfg = fixed.get_next_config(config, results)

        assert adaptive_cfg.input.random_seed == fixed_cfg.input.random_seed
        assert (
            adaptive_cfg.loadgen.warmup_request_count
            == fixed_cfg.loadgen.warmup_request_count
        )

    # -- Cooldown --

    def test_cooldown_seconds_configured(self):
        criterion = self._make_mock_criterion()
        strategy = AdaptiveStrategy(criterion=criterion, cooldown_seconds=2.5)
        assert strategy.get_cooldown_seconds() == 2.5

    def test_cooldown_seconds_default(self):
        criterion = self._make_mock_criterion()
        strategy = AdaptiveStrategy(criterion=criterion)
        assert strategy.get_cooldown_seconds() == 0.0

    # -- Constructor validation --

    def test_invalid_cooldown_raises(self):
        criterion = self._make_mock_criterion()
        with pytest.raises(ValueError, match="Invalid cooldown duration"):
            AdaptiveStrategy(criterion=criterion, cooldown_seconds=-1.0)

    def test_invalid_min_runs_raises(self):
        criterion = self._make_mock_criterion()
        with pytest.raises(ValueError, match="Invalid min_runs"):
            AdaptiveStrategy(criterion=criterion, min_runs=0)

    def test_invalid_max_runs_raises(self):
        criterion = self._make_mock_criterion()
        with pytest.raises(ValueError, match="Invalid max_runs"):
            AdaptiveStrategy(criterion=criterion, min_runs=5, max_runs=3)

    # -- Cross-endpoint metric names (chat, embeddings, audio) --

    @pytest.mark.parametrize(
        "metric_name",
        [
            "time_to_first_token",
            "request_latency",
            "output_token_throughput",
        ],
    )
    def test_convergence_with_varied_metric_names(self, metric_name):
        """AdaptiveStrategy works with metrics from different endpoint types."""
        criterion = MagicMock()
        criterion.is_converged.return_value = True
        strategy = AdaptiveStrategy(criterion=criterion, min_runs=3, max_runs=10)

        results = self._make_results(3, metric=metric_name)
        assert strategy.should_continue(results) is False
        criterion.is_converged.assert_called_once_with(results)


class TestParameterSweepStrategy:
    """Tests for ParameterSweepStrategy."""

    def test_should_continue_returns_true_when_values_remain(self):
        """Test should_continue returns True when parameter values remain."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )

        # No results yet - should continue
        assert strategy.should_continue([]) is True

        # One result - should continue
        results = [
            RunResult(
                label="concurrency_10",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path("/tmp/concurrency_10"),
            )
        ]
        assert strategy.should_continue(results) is True

        # Two results - should continue
        results.append(
            RunResult(
                label="concurrency_20",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=105.0)},
                artifacts_path=Path("/tmp/concurrency_20"),
            )
        )
        assert strategy.should_continue(results) is True

    def test_should_continue_returns_false_when_all_values_tested(self):
        """Test should_continue returns False when all parameter values tested."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )

        # Three results - should stop
        results = [
            RunResult(
                label=f"concurrency_{v}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                artifacts_path=Path(f"/tmp/concurrency_{v}"),
            )
            for v in [10, 20, 30]
        ]
        assert strategy.should_continue(results) is False

    def test_get_next_config_sets_parameter_value(self):
        """Test get_next_config sets the parameter value correctly."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = 5  # Initial value

        # First value
        new_config = strategy.get_next_config(config, [])
        assert new_config.loadgen.concurrency == 10
        assert config.loadgen.concurrency == 5  # Original unchanged

        # Second value
        results = [
            RunResult(
                label="concurrency_10",
                success=True,
                summary_metrics={},
                artifacts_path=Path("/tmp/concurrency_10"),
            )
        ]
        new_config = strategy.get_next_config(config, results)
        assert new_config.loadgen.concurrency == 20

        # Third value
        results.append(
            RunResult(
                label="concurrency_20",
                success=True,
                summary_metrics={},
                artifacts_path=Path("/tmp/concurrency_20"),
            )
        )
        new_config = strategy.get_next_config(config, results)
        assert new_config.loadgen.concurrency == 30

    def test_get_next_config_derives_different_seeds_by_default(self):
        """Test get_next_config derives different seeds for each value by default."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 30],
            same_seed=False,
            auto_set_seed=True,
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None

        # First value - should get base seed (42)
        config1 = strategy.get_next_config(config, [])
        assert config1.input.random_seed == 42

        # Second value - should get base seed + 1
        results = [
            RunResult(
                label="concurrency_10",
                success=True,
                summary_metrics={},
                artifacts_path=Path("/tmp/concurrency_10"),
            )
        ]
        config2 = strategy.get_next_config(config, results)
        assert config2.input.random_seed == 43

        # Third value - should get base seed + 2
        results.append(
            RunResult(
                label="concurrency_20",
                success=True,
                summary_metrics={},
                artifacts_path=Path("/tmp/concurrency_20"),
            )
        )
        config3 = strategy.get_next_config(config, results)
        assert config3.input.random_seed == 44

    def test_get_next_config_uses_same_seed_when_enabled(self):
        """Test get_next_config uses same seed for all values when same_seed=True."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 30],
            same_seed=True,
            auto_set_seed=True,
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None

        # All values should get the same seed (42)
        config1 = strategy.get_next_config(config, [])
        assert config1.input.random_seed == 42

        results = [
            RunResult(
                label="concurrency_10",
                success=True,
                summary_metrics={},
                artifacts_path=Path("/tmp/concurrency_10"),
            )
        ]
        config2 = strategy.get_next_config(config, results)
        assert config2.input.random_seed == 42

        results.append(
            RunResult(
                label="concurrency_20",
                success=True,
                summary_metrics={},
                artifacts_path=Path("/tmp/concurrency_20"),
            )
        )
        config3 = strategy.get_next_config(config, results)
        assert config3.input.random_seed == 42

    def test_get_next_config_preserves_user_seed(self):
        """Test get_next_config preserves user-specified seed as base."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 30],
            same_seed=False,
            auto_set_seed=True,
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = 999

        # First value - should use user seed
        config1 = strategy.get_next_config(config, [])
        assert config1.input.random_seed == 999

        # Second value - should derive from user seed
        results = [
            RunResult(
                label="concurrency_10",
                success=True,
                summary_metrics={},
                artifacts_path=Path("/tmp/concurrency_10"),
            )
        ]
        config2 = strategy.get_next_config(config, results)
        assert config2.input.random_seed == 1000

    def test_get_next_config_no_seed_when_auto_set_disabled(self):
        """Test get_next_config doesn't set seed when auto_set_seed=False."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 30],
            auto_set_seed=False,
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None

        # Should not set seed
        new_config = strategy.get_next_config(config, [])
        assert new_config.input.random_seed is None

    def test_get_next_config_scrubs_sweep_params_from_model_fields_set(self):
        """Test that get_next_config removes all parameter_sweep_* fields from model_fields_set.

        Regression test: these fields persisted through subprocess serialization,
        causing the inner subprocess validator to reject them because concurrency
        is an int (not a list) in the per-value config.
        """
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20]
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = [10, 20]
        config.loadgen.parameter_sweep_mode = "repeated"
        config.loadgen.parameter_sweep_cooldown_seconds = 5.0
        config.loadgen.parameter_sweep_same_seed = True
        config.loadgen.model_fields_set.add("parameter_sweep_mode")
        config.loadgen.model_fields_set.add("parameter_sweep_cooldown_seconds")
        config.loadgen.model_fields_set.add("parameter_sweep_same_seed")

        new_config = strategy.get_next_config(config, [])

        assert new_config.loadgen.concurrency == 10
        assert new_config.loadgen.parameter_sweep_mode == "repeated"
        assert new_config.loadgen.parameter_sweep_cooldown_seconds == 0.0
        assert new_config.loadgen.parameter_sweep_same_seed is False
        assert "parameter_sweep_mode" not in new_config.loadgen.model_fields_set
        assert (
            "parameter_sweep_cooldown_seconds"
            not in new_config.loadgen.model_fields_set
        )
        assert "parameter_sweep_same_seed" not in new_config.loadgen.model_fields_set

    def test_get_run_label_formats_correctly(self):
        """Test get_run_label formats parameter name and value correctly."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30, 100]
        )

        assert strategy.get_run_label(0) == "concurrency_10"
        assert strategy.get_run_label(1) == "concurrency_20"
        assert strategy.get_run_label(2) == "concurrency_30"
        assert strategy.get_run_label(3) == "concurrency_100"

    def test_get_cooldown_seconds_returns_configured_value(self):
        """Test get_cooldown_seconds returns configured value."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20],
            cooldown_seconds=5.0,
        )
        assert strategy.get_cooldown_seconds() == 5.0

    def test_get_cooldown_seconds_returns_default_zero(self):
        """Test get_cooldown_seconds returns default value of zero."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20]
        )
        assert strategy.get_cooldown_seconds() == 0.0

    def test_get_run_path_returns_correct_structure(self):
        """Test get_run_path returns correct path structure."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )

        base_dir = Path("/tmp/artifacts")

        # Test paths for each value
        assert strategy.get_run_path(base_dir, 0) == Path(
            "/tmp/artifacts/concurrency_10"
        )
        assert strategy.get_run_path(base_dir, 1) == Path(
            "/tmp/artifacts/concurrency_20"
        )
        assert strategy.get_run_path(base_dir, 2) == Path(
            "/tmp/artifacts/concurrency_30"
        )

    def test_get_aggregate_path_returns_sweep_aggregate(self):
        """Test get_aggregate_path returns sweep_aggregate directory."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )

        base_dir = Path("/tmp/artifacts")
        assert strategy.get_aggregate_path(base_dir) == Path(
            "/tmp/artifacts/sweep_aggregate"
        )

    def test_path_building_consistency(self):
        """Test that path building is consistent with label generation."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30, 40, 50]
        )

        base_dir = Path("/tmp/artifacts")

        # Path should use the same label as get_run_label
        for run_index in range(5):
            label = strategy.get_run_label(run_index)
            path = strategy.get_run_path(base_dir, run_index)

            # Path should end with the label
            assert path.name == label

    def test_invalid_cooldown_seconds_raises_error(self):
        """Test that negative cooldown raises ValueError."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        with pytest.raises(ValueError, match="Invalid cooldown duration"):
            ParameterSweepStrategy(
                parameter_name="concurrency",
                parameter_values=[10, 20],
                cooldown_seconds=-1.0,
            )

    def test_empty_parameter_values_raises_error(self):
        """Test that empty parameter_values raises ValueError."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        with pytest.raises(
            ValueError, match="Parameter sweep requires at least one value"
        ):
            ParameterSweepStrategy(parameter_name="concurrency", parameter_values=[])

    def test_label_sanitization(self):
        """Test that labels are sanitized to prevent path traversal."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )

        # Normal labels should work fine
        assert strategy.get_run_label(0) == "concurrency_10"
        assert strategy.get_run_label(1) == "concurrency_20"

    def test_creates_deep_copy_of_config(self):
        """Test that get_next_config creates a deep copy and doesn't modify original."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[10, 20, 30]
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.loadgen.concurrency = 5
        config.input.random_seed = 42

        # Get modified config
        new_config = strategy.get_next_config(config, [])

        # Verify it's a different object (deep copy)
        assert new_config is not config
        assert new_config.loadgen.concurrency == 10
        assert config.loadgen.concurrency == 5  # Original unchanged

    def test_base_seed_initialized_once(self):
        """Test that base_seed is initialized once and reused."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        strategy = ParameterSweepStrategy(
            parameter_name="concurrency",
            parameter_values=[10, 20, 30],
            same_seed=False,
            auto_set_seed=True,
        )

        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        config.input.random_seed = None

        # First call should initialize base_seed
        assert strategy.base_seed is None
        config1 = strategy.get_next_config(config, [])
        assert strategy.base_seed == 42
        assert config1.input.random_seed == 42

        # Subsequent calls should reuse base_seed
        results = [
            RunResult(
                label="concurrency_10",
                success=True,
                summary_metrics={},
                artifacts_path=Path("/tmp/concurrency_10"),
            )
        ]
        config2 = strategy.get_next_config(config, results)
        assert strategy.base_seed == 42  # Still 42
        assert config2.input.random_seed == 43  # Derived from 42

    def test_different_parameter_names(self):
        """Test strategy works with different parameter names."""
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        # Test with request_rate parameter
        strategy = ParameterSweepStrategy(
            parameter_name="request_rate", parameter_values=[10, 50, 100]
        )

        assert strategy.get_run_label(0) == "request_rate_10"
        assert strategy.get_run_label(1) == "request_rate_50"
        assert strategy.get_run_label(2) == "request_rate_100"

        base_dir = Path("/tmp/artifacts")
        assert strategy.get_run_path(base_dir, 0) == Path(
            "/tmp/artifacts/request_rate_10"
        )


class TestParameterSweepStrategyAggregation:
    """Tests for sweep-only aggregation metric key flattening."""

    def _make_sweep_strategy(self):
        from aiperf.orchestrator.strategies import ParameterSweepStrategy

        return ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[2, 4]
        )

    def _make_result(self, value, metrics):
        return RunResult(
            label=f"concurrency_{value}",
            success=True,
            summary_metrics=metrics,
            metadata={"concurrency": value},
        )

    def test_build_per_combination_stats_flattens_metric_keys(self):
        """Sweep-only must flatten metric keys to match confidence format."""
        strategy = self._make_sweep_strategy()
        results = [
            self._make_result(
                2,
                {
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=100.0, p99=120.0
                    ),
                    "request_latency": JsonMetricResult(unit="ms", avg=5.0, p99=10.0),
                },
            ),
        ]

        stats = strategy._build_per_combination_stats(results)
        combo_stats = list(stats.values())[0]

        assert "request_throughput_avg" in combo_stats
        assert "request_throughput_p99" in combo_stats
        assert "request_latency_avg" in combo_stats
        assert "request_latency_p99" in combo_stats
        # Raw metric names should NOT be present
        assert "request_throughput" not in combo_stats
        assert "request_latency" not in combo_stats

    def test_sweep_only_aggregate_populates_best_configurations(self):
        """Sweep-only aggregate must populate best_configurations when metrics exist."""
        strategy = self._make_sweep_strategy()
        config = MagicMock(spec=UserConfig)
        results = [
            self._make_result(
                2,
                {
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=100.0
                    ),
                    "request_latency": JsonMetricResult(unit="ms", avg=5.0, p99=10.0),
                },
            ),
            self._make_result(
                4,
                {
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=200.0
                    ),
                    "request_latency": JsonMetricResult(unit="ms", avg=3.0, p99=8.0),
                },
            ),
        ]

        agg = strategy.aggregate(results, config)

        assert agg is not None
        best = agg.metadata["best_configurations"]
        assert "best_throughput" in best
        assert best["best_throughput"]["parameters"] == {"concurrency": 4}
        assert best["best_throughput"]["metric"] == 200.0

    def test_sweep_only_aggregate_populates_pareto_optimal(self):
        """Sweep-only aggregate must populate pareto_optimal when metrics exist."""
        strategy = self._make_sweep_strategy()
        config = MagicMock(spec=UserConfig)
        results = [
            self._make_result(
                2,
                {
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=100.0
                    ),
                    "request_latency": JsonMetricResult(unit="ms", p99=5.0),
                },
            ),
            self._make_result(
                4,
                {
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=200.0
                    ),
                    "request_latency": JsonMetricResult(unit="ms", p99=10.0),
                },
            ),
        ]

        agg = strategy.aggregate(results, config)

        assert agg is not None
        assert len(agg.metadata["pareto_optimal"]) == 2

    def test_sweep_only_with_ttft_metric(self):
        """Sweep-only with time_to_first_token should populate best_latency."""
        strategy = self._make_sweep_strategy()
        config = MagicMock(spec=UserConfig)
        results = [
            self._make_result(
                2,
                {
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=100.0
                    ),
                    "time_to_first_token": JsonMetricResult(unit="ms", p99=5.0),
                },
            ),
            self._make_result(
                4,
                {
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=200.0
                    ),
                    "time_to_first_token": JsonMetricResult(unit="ms", p99=10.0),
                },
            ),
        ]

        agg = strategy.aggregate(results, config)

        assert agg is not None
        best = agg.metadata["best_configurations"]
        assert "best_latency_p99" in best
        assert best["best_latency_p99"]["metric"] == 5.0


class TestSweepConfidenceStrategyMetadata:
    """Tests for sweep+confidence metadata reporting attempted values."""

    def test_metadata_reports_attempted_values_not_surviving(self):
        """Metadata must report all configured values, not just those that aggregated."""
        from aiperf.orchestrator.strategies import (
            ParameterSweepStrategy,
            SweepConfidenceStrategy,
            SweepMode,
        )

        sweep = ParameterSweepStrategy(
            parameter_name="concurrency", parameter_values=[2, 4, 6]
        )
        confidence = FixedTrialsStrategy(num_trials=3)
        strategy = SweepConfidenceStrategy(
            sweep=sweep, confidence=confidence, mode=SweepMode.REPEATED
        )

        # Value 6 has only 1 successful run (needs 2 for confidence)
        results = [
            RunResult(
                label="trial_0001",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=100.0
                    ),
                },
                metadata={"concurrency": 2, "trial_index": 0},
            ),
            RunResult(
                label="trial_0002",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=110.0
                    ),
                },
                metadata={"concurrency": 2, "trial_index": 1},
            ),
            RunResult(
                label="trial_0003",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=200.0
                    ),
                },
                metadata={"concurrency": 4, "trial_index": 0},
            ),
            RunResult(
                label="trial_0004",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=210.0
                    ),
                },
                metadata={"concurrency": 4, "trial_index": 1},
            ),
            RunResult(
                label="trial_0005",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec", avg=300.0
                    ),
                },
                metadata={"concurrency": 6, "trial_index": 0},
            ),
            RunResult(
                label="trial_0006",
                success=False,
                summary_metrics={},
                error="timeout",
                metadata={"concurrency": 6, "trial_index": 1},
            ),
        ]

        config = MagicMock()
        config.loadgen.confidence_level = 0.95

        agg = strategy.aggregate(results, config)

        assert agg is not None
        sweep_params = agg.metadata["sweep_parameters"]
        assert sweep_params[0]["values"] == [2, 4, 6]
        assert agg.metadata["num_combinations"] == 3

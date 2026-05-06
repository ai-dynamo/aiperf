# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LoadGeneratorConfig validators."""

import pytest
from pydantic import ValidationError

from aiperf.common.config.loadgen_config import LoadGeneratorConfig


class TestMultiRunParamsValidation:
    """Test suite for multi-run parameter validation."""

    def test_confidence_level_with_single_run_raises_error(self):
        """Test that setting confidence_level with num_profile_runs=1 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(num_profile_runs=1, confidence_level=0.99)

        # Check that the error message is helpful
        error_msg = str(exc_info.value)
        assert (
            "--confidence-level only applies when running multiple trials" in error_msg
        )

    def test_profile_run_disable_warmup_after_first_with_single_run_raises_error(self):
        """Test that setting profile_run_disable_warmup_after_first with num_profile_runs=1 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(
                num_profile_runs=1, profile_run_disable_warmup_after_first=False
            )

        # Check that the error message is helpful
        error_msg = str(exc_info.value)
        assert (
            "--profile-run-disable-warmup-after-first only applies when running multiple trials"
            in error_msg
        )

    def test_both_params_with_single_run_raises_error(self):
        """Test that setting both params with num_profile_runs=1 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(
                num_profile_runs=1,
                confidence_level=0.99,
                profile_run_disable_warmup_after_first=False,
            )

        # Should raise error about at least one of them
        error_msg = str(exc_info.value)
        assert (
            "--confidence-level only applies when running multiple trials" in error_msg
            or "--profile-run-disable-warmup-after-first only applies when running multiple trials"
            in error_msg
        )

    def test_confidence_level_with_multiple_runs_succeeds(self):
        """Test that setting confidence_level with num_profile_runs>1 succeeds."""
        config = LoadGeneratorConfig(num_profile_runs=5, confidence_level=0.99)
        assert config.num_profile_runs == 5
        assert config.confidence_level == 0.99

    def test_profile_run_disable_warmup_after_first_with_multiple_runs_succeeds(self):
        """Test that setting profile_run_disable_warmup_after_first with num_profile_runs>1 succeeds."""
        config = LoadGeneratorConfig(
            num_profile_runs=5, profile_run_disable_warmup_after_first=False
        )
        assert config.num_profile_runs == 5
        assert config.profile_run_disable_warmup_after_first is False

    def test_both_params_with_multiple_runs_succeeds(self):
        """Test that setting both params with num_profile_runs>1 succeeds."""
        config = LoadGeneratorConfig(
            num_profile_runs=5,
            confidence_level=0.99,
            profile_run_disable_warmup_after_first=False,
        )
        assert config.num_profile_runs == 5
        assert config.confidence_level == 0.99
        assert config.profile_run_disable_warmup_after_first is False

    def test_default_values_with_single_run_succeeds(self):
        """Test that using default values with num_profile_runs=1 succeeds."""
        config = LoadGeneratorConfig(num_profile_runs=1)
        assert config.num_profile_runs == 1
        assert config.confidence_level == 0.95  # default
        assert config.profile_run_disable_warmup_after_first is True  # default

    def test_default_num_profile_runs_succeeds(self):
        """Test that using default num_profile_runs (1) succeeds."""
        config = LoadGeneratorConfig()
        assert config.num_profile_runs == 1
        assert config.confidence_level == 0.95  # default
        assert config.profile_run_disable_warmup_after_first is True  # default
        assert config.set_consistent_seed is True  # default

    def test_set_consistent_seed_with_single_run_raises_error(self):
        """Test that setting set_consistent_seed with num_profile_runs=1 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(num_profile_runs=1, set_consistent_seed=False)

        # Check that the error message is helpful
        error_msg = str(exc_info.value)
        assert (
            "--set-consistent-seed only applies when running multiple trials"
            in error_msg
        )

    def test_set_consistent_seed_with_multiple_runs_succeeds(self):
        """Test that setting set_consistent_seed with num_profile_runs>1 succeeds."""
        config = LoadGeneratorConfig(num_profile_runs=5, set_consistent_seed=False)
        assert config.num_profile_runs == 5
        assert config.set_consistent_seed is False

    def test_all_multi_run_params_with_multiple_runs_succeeds(self):
        """Test that setting all multi-run params with num_profile_runs>1 succeeds."""
        config = LoadGeneratorConfig(
            num_profile_runs=5,
            confidence_level=0.99,
            profile_run_disable_warmup_after_first=False,
            set_consistent_seed=False,
        )
        assert config.num_profile_runs == 5
        assert config.confidence_level == 0.99
        assert config.profile_run_disable_warmup_after_first is False
        assert config.set_consistent_seed is False


class TestConcurrencyListValidation:
    """Test suite for concurrency list validation."""

    def test_single_concurrency_value_succeeds(self):
        """Test that single integer concurrency value works (backward compatibility)."""
        config = LoadGeneratorConfig(concurrency=10)
        assert config.concurrency == 10

    def test_concurrency_list_valid_values_succeeds(self):
        """Test that concurrency list with valid values succeeds."""
        config = LoadGeneratorConfig(concurrency=[10, 20, 30, 40])
        assert config.concurrency == [10, 20, 30, 40]

    def test_concurrency_list_with_duplicates_raises_error(self):
        """Test that concurrency list with duplicate values raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=[10, 20, 10, 30])
        assert "Duplicate values" in str(exc_info.value)

    def test_concurrency_list_single_value_raises_error(self):
        """Test that concurrency list with single value raises ValueError.

        Parameter sweep requires at least 2 values to compare. Single values
        should be specified without a list (e.g., concurrency=10 not [10]).
        """
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=[10])

        error_msg = str(exc_info.value)
        assert "Parameter sweep requires at least 2 values" in error_msg
        assert "For a single concurrency value, use --concurrency 10" in error_msg

    def test_concurrency_list_with_zero_raises_error(self):
        """Test that concurrency list with zero value raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=[10, 0, 30])

        error_msg = str(exc_info.value)
        assert "Invalid concurrency" in error_msg
        assert "'0'" in error_msg
        assert "All values must be positive integers (>= 1)" in error_msg

    def test_concurrency_list_with_negative_raises_error(self):
        """Test that concurrency list with negative value raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=[10, -5, 30])

        error_msg = str(exc_info.value)
        assert "Invalid concurrency" in error_msg
        assert "'-5'" in error_msg
        assert "All values must be positive integers (>= 1)" in error_msg

    def test_concurrency_list_with_multiple_invalid_raises_error(self):
        """Test that concurrency list with multiple invalid values raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=[10, 0, -5, 30])

        error_msg = str(exc_info.value)
        assert "Invalid concurrency" in error_msg
        # Should fail on first invalid value (0)
        assert "'0'" in error_msg
        assert "All values must be positive integers (>= 1)" in error_msg

    def test_concurrency_none_succeeds(self):
        """Test that concurrency=None succeeds (default)."""
        config = LoadGeneratorConfig(concurrency=None)
        assert config.concurrency is None

    def test_concurrency_default_is_none(self):
        """Test that default concurrency is None."""
        config = LoadGeneratorConfig()
        assert config.concurrency is None

    def test_single_concurrency_zero_raises_error(self):
        """Test that single concurrency value of zero raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=0)

        error_msg = str(exc_info.value)
        # Should get detailed error message from validator
        assert (
            "Invalid concurrency value: 0" in error_msg
            or "greater than or equal to 1" in error_msg.lower()
        )

    def test_single_concurrency_negative_raises_error(self):
        """Test that single concurrency value that is negative raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=-5)

        error_msg = str(exc_info.value)
        # Should get detailed error message from validator
        assert (
            "Invalid concurrency value: -5" in error_msg
            or "greater than or equal to 1" in error_msg.lower()
        )


class TestParameterSweepModeValidation:
    """Test suite for parameter_sweep_mode validation."""

    def test_default_parameter_sweep_mode_is_repeated(self):
        """Test that default parameter_sweep_mode is 'repeated'."""
        config = LoadGeneratorConfig()
        assert config.parameter_sweep_mode == "repeated"

    def test_parameter_sweep_mode_repeated_succeeds(self):
        """Test that setting parameter_sweep_mode to 'repeated' succeeds."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_mode="repeated"
        )
        assert config.parameter_sweep_mode == "repeated"

    def test_parameter_sweep_mode_independent_succeeds(self):
        """Test that setting parameter_sweep_mode to 'independent' succeeds."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_mode="independent"
        )
        assert config.parameter_sweep_mode == "independent"

    def test_parameter_sweep_mode_invalid_value_raises_error(self):
        """Test that setting parameter_sweep_mode to invalid value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(parameter_sweep_mode="invalid")

        error_msg = str(exc_info.value)
        assert "parameter_sweep_mode" in error_msg.lower()

    def test_parameter_sweep_mode_with_concurrency_list_succeeds(self):
        """Test that parameter_sweep_mode works with concurrency list."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_mode="repeated"
        )
        assert config.concurrency == [10, 20, 30]
        assert config.parameter_sweep_mode == "repeated"

    def test_parameter_sweep_mode_with_single_concurrency_raises_error(self):
        """Test that parameter_sweep_mode with single concurrency raises error (requires list)."""
        # After validation improvements, sweep parameters require a concurrency list
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=10, parameter_sweep_mode="independent")

        error_msg = str(exc_info.value)
        assert (
            "--parameter-sweep-mode only applies when sweeping parameters" in error_msg
        )

    def test_parameter_sweep_mode_with_multi_run_succeeds(self):
        """Test that parameter_sweep_mode works with multi-run configuration."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30],
            num_profile_runs=5,
            parameter_sweep_mode="repeated",
        )
        assert config.concurrency == [10, 20, 30]
        assert config.num_profile_runs == 5
        assert config.parameter_sweep_mode == "repeated"


class TestParameterSweepCooldownValidation:
    """Test suite for parameter_sweep_cooldown_seconds validation."""

    def test_default_parameter_sweep_cooldown_is_zero(self):
        """Test that default parameter_sweep_cooldown_seconds is 0.0."""
        config = LoadGeneratorConfig()
        assert config.parameter_sweep_cooldown_seconds == 0.0

    def test_parameter_sweep_cooldown_positive_value_succeeds(self):
        """Test that setting parameter_sweep_cooldown_seconds to positive value succeeds."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_cooldown_seconds=10.5
        )
        assert config.parameter_sweep_cooldown_seconds == 10.5

    def test_parameter_sweep_cooldown_zero_succeeds(self):
        """Test that setting parameter_sweep_cooldown_seconds to 0 succeeds."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_cooldown_seconds=0.0
        )
        assert config.parameter_sweep_cooldown_seconds == 0.0

    def test_parameter_sweep_cooldown_negative_raises_error(self):
        """Test that setting parameter_sweep_cooldown_seconds to negative value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(parameter_sweep_cooldown_seconds=-1.0)

        error_msg = str(exc_info.value)
        assert "parameter_sweep_cooldown_seconds" in error_msg.lower()
        assert "greater than or equal to 0" in error_msg.lower()

    def test_parameter_sweep_cooldown_with_concurrency_list_succeeds(self):
        """Test that parameter_sweep_cooldown_seconds works with concurrency list."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_cooldown_seconds=5.0
        )
        assert config.concurrency == [10, 20, 30]
        assert config.parameter_sweep_cooldown_seconds == 5.0

    def test_parameter_sweep_cooldown_with_single_concurrency_raises_error(self):
        """Test that parameter_sweep_cooldown_seconds with single concurrency raises error (requires list)."""
        # After validation improvements, sweep parameters require a concurrency list
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=10, parameter_sweep_cooldown_seconds=3.0)

        error_msg = str(exc_info.value)
        assert (
            "--parameter-sweep-cooldown-seconds only applies when sweeping parameters"
            in error_msg
        )

    def test_parameter_sweep_cooldown_with_multi_run_succeeds(self):
        """Test that parameter_sweep_cooldown_seconds works with multi-run configuration."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30],
            num_profile_runs=5,
            parameter_sweep_cooldown_seconds=2.5,
            profile_run_cooldown_seconds=1.0,
        )
        assert config.concurrency == [10, 20, 30]
        assert config.num_profile_runs == 5
        assert config.parameter_sweep_cooldown_seconds == 2.5
        assert config.profile_run_cooldown_seconds == 1.0

    def test_parameter_sweep_cooldown_large_value_succeeds(self):
        """Test that parameter_sweep_cooldown_seconds accepts large values."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_cooldown_seconds=3600.0
        )
        assert config.parameter_sweep_cooldown_seconds == 3600.0


class TestParameterSweepSameSeedValidation:
    """Test suite for parameter_sweep_same_seed validation."""

    def test_default_parameter_sweep_same_seed_is_false(self):
        """Test that default parameter_sweep_same_seed is False."""
        config = LoadGeneratorConfig()
        assert config.parameter_sweep_same_seed is False

    def test_parameter_sweep_same_seed_true_succeeds(self):
        """Test that setting parameter_sweep_same_seed to True succeeds."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_same_seed=True
        )
        assert config.parameter_sweep_same_seed is True

    def test_parameter_sweep_same_seed_false_succeeds(self):
        """Test that setting parameter_sweep_same_seed to False succeeds."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_same_seed=False
        )
        assert config.parameter_sweep_same_seed is False

    def test_parameter_sweep_same_seed_with_concurrency_list_succeeds(self):
        """Test that parameter_sweep_same_seed works with concurrency list."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], parameter_sweep_same_seed=True
        )
        assert config.concurrency == [10, 20, 30]
        assert config.parameter_sweep_same_seed is True

    def test_parameter_sweep_same_seed_with_single_concurrency_raises_error(self):
        """Test that parameter_sweep_same_seed with single concurrency raises error (requires list)."""
        # After validation improvements, sweep parameters require a concurrency list
        with pytest.raises(ValidationError) as exc_info:
            LoadGeneratorConfig(concurrency=10, parameter_sweep_same_seed=True)

        error_msg = str(exc_info.value)
        assert (
            "--parameter-sweep-same-seed only applies when sweeping parameters"
            in error_msg
        )

    def test_parameter_sweep_same_seed_with_multi_run_succeeds(self):
        """Test that parameter_sweep_same_seed works with multi-run configuration."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30], num_profile_runs=5, parameter_sweep_same_seed=True
        )
        assert config.concurrency == [10, 20, 30]
        assert config.num_profile_runs == 5
        assert config.parameter_sweep_same_seed is True

    def test_parameter_sweep_same_seed_with_all_sweep_params_succeeds(self):
        """Test that parameter_sweep_same_seed works with all sweep parameters."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30],
            parameter_sweep_mode="independent",
            parameter_sweep_cooldown_seconds=5.0,
            parameter_sweep_same_seed=True,
        )
        assert config.concurrency == [10, 20, 30]
        assert config.parameter_sweep_mode == "independent"
        assert config.parameter_sweep_cooldown_seconds == 5.0
        assert config.parameter_sweep_same_seed is True


class TestSweepParamsValidation:
    """Test suite for parameter sweep parameter validation when not sweeping."""

    def test_parameter_sweep_mode_with_single_concurrency_defaults_to_repeated(self):
        """Test that parameter_sweep_mode defaults work with single concurrency.

        When passing fields to the LoadGeneratorConfig constructor, Pydantic
        populates model_fields_set automatically. This test verifies that the
        default value doesn't trigger validation errors.
        """
        config = LoadGeneratorConfig(concurrency=10)
        assert config.parameter_sweep_mode == "repeated"  # default value works

    def test_parameter_sweep_cooldown_with_single_concurrency_defaults_to_zero(self):
        """Test that parameter_sweep_cooldown_seconds defaults work with single concurrency.

        When passing fields to the LoadGeneratorConfig constructor, Pydantic
        populates model_fields_set automatically. This test verifies that the
        default value doesn't trigger validation errors.
        """
        config = LoadGeneratorConfig(concurrency=10)
        assert config.parameter_sweep_cooldown_seconds == 0.0  # default value works

    def test_parameter_sweep_same_seed_with_single_concurrency_defaults_to_false(self):
        """Test that parameter_sweep_same_seed defaults work with single concurrency.

        When passing fields to the LoadGeneratorConfig constructor, Pydantic
        populates model_fields_set automatically. This test verifies that the
        default value doesn't trigger validation errors.
        """
        config = LoadGeneratorConfig(concurrency=10)
        assert config.parameter_sweep_same_seed is False  # default value works

    def test_all_sweep_params_with_concurrency_list_succeeds(self):
        """Test that all sweep parameters work correctly with concurrency list."""
        config = LoadGeneratorConfig(
            concurrency=[10, 20, 30],
            parameter_sweep_mode="independent",
            parameter_sweep_cooldown_seconds=5.0,
            parameter_sweep_same_seed=True,
        )
        assert config.concurrency == [10, 20, 30]
        assert config.parameter_sweep_mode == "independent"
        assert config.parameter_sweep_cooldown_seconds == 5.0
        assert config.parameter_sweep_same_seed is True

    def test_sweep_params_with_none_concurrency_uses_defaults(self):
        """Test that default sweep parameters work with None concurrency (default)."""
        # When using defaults (not explicitly setting), validation should pass
        config = LoadGeneratorConfig(concurrency=None)
        assert config.concurrency is None
        assert config.parameter_sweep_mode == "repeated"  # default
        assert config.parameter_sweep_cooldown_seconds == 0.0  # default
        assert config.parameter_sweep_same_seed is False  # default

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
            "--confidence-level only applies when --num-profile-runs > 1" in error_msg
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
            "--profile-run-disable-warmup-after-first only applies when --num-profile-runs > 1"
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
            "--confidence-level only applies when --num-profile-runs > 1" in error_msg
            or "--profile-run-disable-warmup-after-first only applies when --num-profile-runs > 1"
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
            "--set-consistent-seed only applies when --num-profile-runs > 1"
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

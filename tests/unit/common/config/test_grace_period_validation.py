# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for grace period configuration validation.
"""

import pytest
from pydantic import ValidationError

from aiperf.config.flags.cli_config import CLIConfig


class TestGracePeriodValidation:
    """Test validation of grace period configuration."""

    def test_grace_period_with_benchmark_duration_valid(self):
        """Test that grace period is valid when used with benchmark duration."""
        loadgen_config = CLIConfig(benchmark_duration=10.0, benchmark_grace_period=30.0)

        # Create a minimal CLIConfig to test validation
        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        cli_config = CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

        # Should not raise any validation errors
        assert cli_config.benchmark_duration == 10.0
        assert cli_config.benchmark_grace_period == 30.0

    def test_grace_period_without_benchmark_duration_no_longer_raises(self):
        """v1 LoadGeneratorConfig is validator-free; the
        --benchmark-grace-period vs --benchmark-duration cross-field check
        moved to AIPerfConfig in v2."""
        loadgen_config = CLIConfig(
            benchmark_grace_period=30.0,
            request_count=10,
        )
        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )
        # Should NOT raise on v1 DTO construction.
        CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

    def test_default_grace_period_without_duration_valid(self):
        """Test that default grace period without explicit duration is valid."""
        # When grace period is not explicitly set, it should use default
        # and not trigger validation error
        loadgen_config = CLIConfig(request_count=10)

        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        cli_config = CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

        # Should not raise validation error since grace period wasn't explicitly set
        assert cli_config.request_count == 10
        assert cli_config.benchmark_grace_period == 30.0  # Default value

    def test_zero_grace_period_with_duration_valid(self):
        """Test that zero grace period with duration is valid."""
        loadgen_config = CLIConfig(benchmark_duration=5.0, benchmark_grace_period=0.0)

        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        cli_config = CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

        assert cli_config.benchmark_duration == 5.0
        assert cli_config.benchmark_grace_period == 0.0

    def test_negative_grace_period_invalid(self):
        """Test that negative grace period raises validation error."""
        with pytest.raises(ValidationError):
            CLIConfig(benchmark_duration=5.0, benchmark_grace_period=-1.0)

    @pytest.mark.parametrize("grace_period", [0.0, 10.0, 30.0, 60.0, 120.0])
    def test_valid_grace_period_values(self, grace_period: float):
        """Test various valid grace period values."""
        loadgen_config = CLIConfig(
            benchmark_duration=10.0, benchmark_grace_period=grace_period
        )

        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        cli_config = CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

        assert cli_config.benchmark_grace_period == grace_period


class TestWarmupGracePeriodValidation:
    """Test validation of warmup grace period configuration."""

    def test_warmup_grace_period_with_warmup_duration_valid(self):
        """Test that warmup grace period is valid when used with warmup duration."""
        loadgen_config = CLIConfig(
            warmup_duration=5.0,
            warmup_grace_period=10.0,
            benchmark_duration=30.0,
        )

        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        cli_config = CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

        assert cli_config.warmup_duration == 5.0
        assert cli_config.warmup_grace_period == 10.0

    def test_warmup_grace_period_with_warmup_request_count_no_longer_raises(self):
        """v1 DTO is validator-free; warmup-grace-period vs --warmup-duration
        rule moved to AIPerfConfig in v2."""
        loadgen_config = CLIConfig(
            warmup_request_count=10,
            warmup_grace_period=15.0,
            benchmark_duration=30.0,
        )
        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )
        CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

    def test_warmup_grace_period_with_warmup_num_sessions_no_longer_raises(self):
        """See test_warmup_grace_period_with_warmup_request_count_no_longer_raises."""
        loadgen_config = CLIConfig(
            warmup_num_sessions=5,
            warmup_grace_period=20.0,
            benchmark_duration=30.0,
        )
        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )
        CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

    def test_warmup_grace_period_without_warmup_duration_no_longer_raises(self):
        """See test_warmup_grace_period_with_warmup_request_count_no_longer_raises."""
        loadgen_config = CLIConfig(
            warmup_grace_period=30.0,
            benchmark_duration=60.0,
        )
        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )
        CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

    def test_default_warmup_grace_period_without_warmup_valid(self):
        """Test that default warmup grace period (None) without warmup is valid."""
        loadgen_config = CLIConfig(benchmark_duration=10.0)

        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        cli_config = CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

        assert cli_config.warmup_grace_period is None

    def test_negative_warmup_grace_period_invalid(self):
        """Test that negative warmup grace period raises validation error."""
        with pytest.raises(ValidationError):
            CLIConfig(warmup_duration=5.0, warmup_grace_period=-1.0)

    def test_zero_warmup_grace_period_valid(self):
        """Test that zero warmup grace period is valid (no wait for responses)."""
        loadgen_config = CLIConfig(
            warmup_duration=5.0,
            warmup_grace_period=0.0,
            benchmark_duration=30.0,
        )

        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        cli_config = CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

        assert cli_config.warmup_grace_period == 0.0

    @pytest.mark.parametrize("grace_period", [0.0, 5.0, 10.0, 30.0, 60.0])
    def test_valid_warmup_grace_period_values(self, grace_period: float):
        """Test various valid warmup grace period values."""
        loadgen_config = CLIConfig(
            warmup_duration=5.0,
            warmup_grace_period=grace_period,
            benchmark_duration=30.0,
        )

        endpoint_config = CLIConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        cli_config = CLIConfig(
            **endpoint_config.model_dump(exclude_unset=True),
            **loadgen_config.model_dump(exclude_unset=True),
        )

        assert cli_config.warmup_grace_period == grace_period

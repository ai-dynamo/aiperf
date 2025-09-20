# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for grace period configuration validation.
"""

import pytest
from pydantic import ValidationError

from aiperf.common.config import UserConfig
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig


class TestGracePeriodValidation:
    """Test validation of grace period configuration."""

    def test_grace_period_with_benchmark_duration_valid(self):
        """Test that grace period is valid when used with benchmark duration."""
        loadgen_config = LoadGeneratorConfig(
            benchmark_duration=10.0, benchmark_grace_period=30.0
        )

        # Create a minimal UserConfig to test validation
        endpoint_config = EndpointConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        user_config = UserConfig(endpoint=endpoint_config, loadgen=loadgen_config)

        # Should not raise any validation errors
        assert user_config.loadgen.benchmark_duration == 10.0
        assert user_config.loadgen.benchmark_grace_period == 30.0

    def test_grace_period_without_benchmark_duration_invalid(self):
        """Test that grace period without benchmark duration raises validation error."""
        with pytest.raises(
            ValidationError,
            match=".*--benchmark-grace-period can only be used with.*duration-based benchmarking.*",
        ):
            loadgen_config = LoadGeneratorConfig(
                benchmark_grace_period=30.0,
                request_count=10,  # Using request count instead of duration
            )

            endpoint_config = EndpointConfig(
                url="http://localhost:8000/test", model_names=["test-model"]
            )

            UserConfig(endpoint=endpoint_config, loadgen=loadgen_config)

    def test_default_grace_period_without_duration_valid(self):
        """Test that default grace period without explicit duration is valid."""
        # When grace period is not explicitly set, it should use default
        # and not trigger validation error
        loadgen_config = LoadGeneratorConfig(request_count=10)

        endpoint_config = EndpointConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        user_config = UserConfig(endpoint=endpoint_config, loadgen=loadgen_config)

        # Should not raise validation error since grace period wasn't explicitly set
        assert user_config.loadgen.request_count == 10
        assert user_config.loadgen.benchmark_grace_period == 30.0  # Default value

    def test_zero_grace_period_with_duration_valid(self):
        """Test that zero grace period with duration is valid."""
        loadgen_config = LoadGeneratorConfig(
            benchmark_duration=5.0, benchmark_grace_period=0.0
        )

        endpoint_config = EndpointConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        user_config = UserConfig(endpoint=endpoint_config, loadgen=loadgen_config)

        assert user_config.loadgen.benchmark_duration == 5.0
        assert user_config.loadgen.benchmark_grace_period == 0.0

    def test_negative_grace_period_invalid(self):
        """Test that negative grace period raises validation error."""
        with pytest.raises(ValidationError):
            LoadGeneratorConfig(benchmark_duration=5.0, benchmark_grace_period=-1.0)

    @pytest.mark.parametrize("grace_period", [0.0, 10.0, 30.0, 60.0, 120.0])
    def test_valid_grace_period_values(self, grace_period: float):
        """Test various valid grace period values."""
        loadgen_config = LoadGeneratorConfig(
            benchmark_duration=10.0, benchmark_grace_period=grace_period
        )

        endpoint_config = EndpointConfig(
            url="http://localhost:8000/test", model_names=["test-model"]
        )

        user_config = UserConfig(endpoint=endpoint_config, loadgen=loadgen_config)

        assert user_config.loadgen.benchmark_grace_period == grace_period

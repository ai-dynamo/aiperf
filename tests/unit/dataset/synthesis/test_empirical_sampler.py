# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EmpiricalSampler."""

import numpy as np
import pytest

from aiperf.dataset.synthesis import EmpiricalSampler


class TestEmpiricalSampler:
    """Tests for EmpiricalSampler class."""

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_with_data(self) -> None:
        """Test EmpiricalSampler initialization with data."""
        data = [1, 2, 3, 4, 5]
        sampler = EmpiricalSampler(data)
        assert sampler is not None

    def test_initialization_with_empty_data(self) -> None:
        """Test EmpiricalSampler initialization with empty data."""
        sampler = EmpiricalSampler([])
        assert sampler is not None

    def test_initialization_with_floats(self) -> None:
        """Test EmpiricalSampler initialization with float data."""
        data = [1.5, 2.3, 3.1, 4.8]
        sampler = EmpiricalSampler(data)
        assert sampler is not None

    # ============================================================================
    # Sampling Tests
    # ============================================================================

    def test_sample_returns_value(self) -> None:
        """Test that sample returns a value."""
        data = [1, 2, 3, 4, 5]
        sampler = EmpiricalSampler(data)
        sample = sampler.sample()
        assert sample is not None

    def test_sample_in_range(self) -> None:
        """Test that sampled values are in the original data range."""
        data = [10, 20, 30, 40, 50]
        sampler = EmpiricalSampler(data)
        rng = np.random.default_rng(42)

        samples = [sampler.sample(rng) for _ in range(100)]
        assert all(s in data for s in samples)

    def test_sample_batch(self) -> None:
        """Test batch sampling."""
        data = [1, 2, 3, 4, 5]
        sampler = EmpiricalSampler(data)
        rng = np.random.default_rng(42)

        samples = sampler.sample_batch(10, rng)
        assert len(samples) == 10
        assert all(isinstance(s, int | np.integer) for s in samples)

    def test_sample_respects_distribution(self) -> None:
        """Test that sampling respects empirical distribution."""
        # Create skewed data
        data = [1] * 90 + [2] * 10  # 90% are 1, 10% are 2
        sampler = EmpiricalSampler(data)
        rng = np.random.default_rng(42)

        samples = [sampler.sample(rng) for _ in range(1000)]
        # Most samples should be 1
        count_1 = sum(1 for s in samples if s == 1)
        assert count_1 > 800  # Should be around 90%

    @pytest.mark.parametrize("size", [1, 10, 100, 1000])
    def test_sample_batch_sizes(self, size: int) -> None:
        """Test batch sampling with various sizes."""
        data = list(range(1, 11))
        sampler = EmpiricalSampler(data)
        rng = np.random.default_rng(42)

        samples = sampler.sample_batch(size, rng)
        assert len(samples) == size

    # ============================================================================
    # Statistics Tests
    # ============================================================================

    def test_get_stats(self) -> None:
        """Test getting distribution statistics."""
        data = [1, 2, 3, 4, 5]
        sampler = EmpiricalSampler(data)
        stats = sampler.get_stats()

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "median" in stats
        assert "num_unique" in stats

    def test_get_stats_values(self) -> None:
        """Test that statistics are correct."""
        data = [1, 2, 3, 4, 5]
        sampler = EmpiricalSampler(data)
        stats = sampler.get_stats()

        assert stats["min"] == 1
        assert stats["max"] == 5
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["num_unique"] == 5

    def test_get_stats_with_duplicates(self) -> None:
        """Test statistics with duplicate values."""
        data = [1, 1, 2, 2, 2, 3]
        sampler = EmpiricalSampler(data)
        stats = sampler.get_stats()

        assert stats["num_unique"] == 3

    # ============================================================================
    # Edge Cases
    # ============================================================================

    def test_sample_single_value(self) -> None:
        """Test sampling when only one value exists."""
        sampler = EmpiricalSampler([42])
        samples = [sampler.sample() for _ in range(10)]
        assert all(s == 42 for s in samples)

    def test_sample_two_values(self) -> None:
        """Test sampling with two distinct values."""
        sampler = EmpiricalSampler([1, 2])
        rng = np.random.default_rng(42)
        samples = [sampler.sample(rng) for _ in range(100)]
        assert all(s in [1, 2] for s in samples)

    def test_sample_large_data(self) -> None:
        """Test with large dataset."""
        data = list(range(1, 1001))
        sampler = EmpiricalSampler(data)
        stats = sampler.get_stats()

        assert stats["min"] == 1
        assert stats["max"] == 1000

    def test_reproducibility_with_seed(self) -> None:
        """Test that sampling is reproducible with same RNG seed."""
        data = [1, 2, 3, 4, 5]
        sampler1 = EmpiricalSampler(data)
        sampler2 = EmpiricalSampler(data)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        samples1 = [sampler1.sample(rng1) for _ in range(100)]
        samples2 = [sampler2.sample(rng2) for _ in range(100)]

        assert samples1 == samples2

    def test_float_data_handling(self) -> None:
        """Test that float data is handled correctly."""
        data = [1.5, 2.3, 3.1, 4.8, 5.2]
        sampler = EmpiricalSampler(data)
        stats = sampler.get_stats()

        # Values should be close to originals
        assert abs(stats["min"] - 1.5) < 0.1
        assert abs(stats["max"] - 5.2) < 0.1

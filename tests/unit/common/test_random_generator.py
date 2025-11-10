# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for the RandomGenerator module."""

import pytest

from aiperf.common import random_generator as rng
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.random_generator import RandomGenerator, _RNGManager


class TestRandomGeneratorBasics:
    """Test basic RandomGenerator functionality."""

    def test_repr(self):
        """Test __repr__ for debugging."""
        rng = RandomGenerator(seed=42, _internal=True)
        assert repr(rng) == "RandomGenerator(seed=42)"

        rng_no_seed = RandomGenerator(seed=None, _internal=True)
        assert repr(rng_no_seed) == "RandomGenerator(seed=None)"

    def test_seed_property(self):
        """Test seed property returns correct value."""
        rng = RandomGenerator(seed=12345, _internal=True)
        assert rng.seed == 12345

        rng_no_seed = RandomGenerator(seed=None, _internal=True)
        assert rng_no_seed.seed is None

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical sequences."""
        rng1 = RandomGenerator(seed=42, _internal=True)
        rng2 = RandomGenerator(seed=42, _internal=True)

        assert rng1.random() == rng2.random()
        assert rng1.randint(1, 100) == rng2.randint(1, 100)
        assert rng1.uniform(0.0, 10.0) == rng2.uniform(0.0, 10.0)
        assert rng1.choice([1, 2, 3, 4, 5]) == rng2.choice([1, 2, 3, 4, 5])

    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different sequences."""
        rng1 = RandomGenerator(seed=42, _internal=True)
        rng2 = RandomGenerator(seed=43, _internal=True)

        assert rng1.random() != rng2.random()


class TestChildRNGIsolation:
    """Test child RNG isolation and reproducibility."""

    def test_child_rng_isolation_order_independence(self):
        """Test that child RNG creation order doesn't affect their sequences."""
        manager1 = _RNGManager(root_seed=42)
        rng_a1 = manager1.derive("component_a")
        rng_b1 = manager1.derive("component_b")

        a1_vals = [rng_a1.random() for _ in range(5)]
        b1_vals = [rng_b1.random() for _ in range(5)]

        manager2 = _RNGManager(root_seed=42)
        rng_b2 = manager2.derive("component_b")
        rng_a2 = manager2.derive("component_a")

        a2_vals = [rng_a2.random() for _ in range(5)]
        b2_vals = [rng_b2.random() for _ in range(5)]

        assert a1_vals == a2_vals
        assert b1_vals == b2_vals

    def test_child_rng_isolation_interleaved_operations(self):
        """Test that interleaved operations on children maintain independence."""
        manager = _RNGManager(root_seed=42)
        rng_a = manager.derive("component_a")
        rng_b = manager.derive("component_b")

        a1 = rng_a.random()
        b1 = rng_b.random()
        a2 = rng_a.random()
        b2 = rng_b.random()

        manager = _RNGManager(root_seed=42)
        rng_a = manager.derive("component_a")
        rng_b = manager.derive("component_b")

        b1_new = rng_b.random()
        a1_new = rng_a.random()
        b2_new = rng_b.random()
        a2_new = rng_a.random()

        assert a1 == a1_new
        assert a2 == a2_new
        assert b1 == b1_new
        assert b2 == b2_new

    def test_child_rng_same_identifier_produces_same_seed(self):
        """Test that same identifier produces children with the same seed."""
        manager = _RNGManager(root_seed=42)

        child1 = manager.derive("my_component")
        child2 = manager.derive("my_component")

        # Different instances but same seed, so they produce same sequences
        assert child1 is not child2
        assert child1.seed == child2.seed
        assert child1.random() == child2.random()

    def test_child_seed_stability_across_runs(self):
        """Test that child seeds are stable (SHA-256 based, not Python hash)."""
        manager1 = _RNGManager(root_seed=42)
        child1 = manager1.derive("component_a")
        val1 = child1.random()

        manager2 = _RNGManager(root_seed=42)
        child2 = manager2.derive("component_a")
        val2 = child2.random()

        assert val1 == val2

    def test_child_with_none_seed_parent(self):
        """Test child generation when parent has None seed."""
        manager = _RNGManager(root_seed=None)
        child1 = manager.derive("component_a")
        child2 = manager.derive("component_b")

        assert child1 is not child2
        assert child1.seed is None
        assert child2.seed is None


class TestGlobalRNGErrorStates:
    """Test global RNG initialization and error states."""

    def test_direct_construction_raises_error(self):
        """Test that constructing RandomGenerator directly raises error."""
        with pytest.raises(
            RuntimeError,
            match="RandomGenerator should not be constructed directly",
        ):
            RandomGenerator(seed=42)

    def test_derive_before_initialization_raises_error(self):
        """Test that rng.derive() raises error before initialization."""
        rng.reset()

        with pytest.raises(
            InvalidStateError,
            match="Global RNG manager has not been initialized",
        ):
            rng.derive("my_component")

    def test_init_double_initialization_raises_error(self):
        """Test that calling rng.init() twice raises error."""
        rng.reset()
        rng.init(42)

        with pytest.raises(
            InvalidStateError,
            match="Global RNG manager has already been initialized",
        ):
            rng.init(43)

    def test_derive_returns_child_with_derived_seed(self):
        """Test that rng.derive(identifier) returns a child RNG with derived seed."""
        rng.reset()
        rng.init(42)

        child = rng.derive("my_component")

        # Child should have a deterministic seed (not None)
        assert child.seed is not None
        # Seed should be different from root seed (42)
        assert child.seed != 42

    def test_derive_requires_identifier(self):
        """Test that rng.derive() requires an identifier parameter."""
        rng.reset()
        rng.init(42)

        with pytest.raises(TypeError):
            rng.derive()  # type: ignore

    def test_reset_and_reinitialize(self):
        """Test that reset allows re-initialization."""
        rng.reset()
        rng.init(42)

        child1 = rng.derive("my_component")
        val1 = child1.random()

        rng.reset()
        rng.init(99)

        child2 = rng.derive("my_component")
        val2 = child2.random()

        assert val1 != val2


class TestSampleMethodEdgeCases:
    """Test edge cases for sample_normal and sample_positive_normal_integer."""

    def test_sample_normal_with_invalid_bounds_raises_error(self):
        """Test that sample_normal raises error when lower > upper."""
        rng = RandomGenerator(seed=42, _internal=True)

        with pytest.raises(ValueError, match="Invalid bounds"):
            rng.sample_normal(mean=100, stddev=10, lower=200, upper=50)

    def test_sample_normal_with_tight_bounds(self):
        """Test sample_normal with very tight bounds (close to mean)."""
        rng = RandomGenerator(seed=42, _internal=True)

        sample = rng.sample_normal(mean=100, stddev=10, lower=99, upper=101)
        assert 99 <= sample <= 101

    def test_sample_normal_with_unreachable_bounds_returns_fallback(self):
        """Test that sample_normal falls back to clamped mean for unreachable bounds."""
        rng = RandomGenerator(seed=42, _internal=True)

        sample = rng.sample_normal(mean=100, stddev=1, lower=200, upper=300)
        assert sample == 200

    def test_sample_positive_normal_with_negative_mean_raises_error(self):
        """Test that sample_positive_normal raises error for negative mean."""
        rng = RandomGenerator(seed=42, _internal=True)

        with pytest.raises(ValueError, match="should be greater than 0"):
            rng.sample_positive_normal(mean=-5, stddev=2)

    def test_sample_positive_normal_integer_always_returns_at_least_one(self):
        """Test that sample_positive_normal_integer always returns >= 1."""
        rng = RandomGenerator(seed=42, _internal=True)

        for mean in [0.1, 0.5, 0.9, 1.0]:
            result = rng.sample_positive_normal_integer(mean=mean, stddev=0.1)
            assert result >= 1, f"Failed for mean={mean}, got {result}"

    def test_sample_positive_normal_integer_with_zero_stddev(self):
        """Test sample_positive_normal_integer with stddev=0."""
        rng = RandomGenerator(seed=42, _internal=True)

        assert rng.sample_positive_normal_integer(mean=5.4, stddev=0) == 5
        assert rng.sample_positive_normal_integer(mean=5.6, stddev=0) == 6
        assert rng.sample_positive_normal_integer(mean=0.3, stddev=0) == 1

    def test_sample_positive_normal_integer_with_negative_stddev(self):
        """Test sample_positive_normal_integer with negative stddev (treated as 0)."""
        rng = RandomGenerator(seed=42, _internal=True)

        result = rng.sample_positive_normal_integer(mean=10, stddev=-1)
        assert result == 10

    def test_sample_positive_normal_allows_mean_zero(self):
        """Test that sample_positive_normal allows mean=0 (boundary case)."""
        rng = RandomGenerator(seed=42, _internal=True)

        sample = rng.sample_positive_normal(mean=0, stddev=1)
        assert sample >= 0


class TestNewAPIMethods:
    """Test newly added API methods (uniform, randint, sample)."""

    def test_uniform_method(self):
        """Test uniform random float generation."""
        rng = RandomGenerator(seed=42, _internal=True)

        val = rng.uniform(10.0, 20.0)
        assert 10.0 <= val <= 20.0

        rng2 = RandomGenerator(seed=42, _internal=True)
        val2 = rng2.uniform(10.0, 20.0)
        assert val == val2

    def test_randint_method_inclusive_bounds(self):
        """Test randint with inclusive upper bound."""
        rng = RandomGenerator(seed=42, _internal=True)

        samples = [rng.randint(1, 5) for _ in range(100)]

        assert all(1 <= s <= 5 for s in samples)
        assert 5 in samples

    def test_sample_method_without_replacement(self):
        """Test sample method (without replacement)."""
        rng = RandomGenerator(seed=42, _internal=True)

        population = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sample = rng.sample(population, k=5)

        assert len(sample) == 5
        assert len(set(sample)) == 5
        assert all(s in population for s in sample)

    def test_sample_method_raises_error_when_k_too_large(self):
        """Test that sample raises error when k > population size."""
        rng = RandomGenerator(seed=42, _internal=True)

        with pytest.raises(ValueError):
            rng.sample([1, 2, 3], k=5)


class TestDocumentationAndEdgeCases:
    """Test various edge cases and documented behaviors."""

    def test_choice_with_empty_sequence_raises_error(self):
        """Test that choice raises error for empty sequence."""
        rng = RandomGenerator(seed=42, _internal=True)

        with pytest.raises(IndexError):
            rng.choice([])

    def test_shuffle_modifies_in_place(self):
        """Test that shuffle modifies list in-place."""
        rng = RandomGenerator(seed=42, _internal=True)

        original = [1, 2, 3, 4, 5]
        shuffled = original.copy()
        result = rng.shuffle(shuffled)

        assert result is None
        assert shuffled != original
        assert sorted(shuffled) == sorted(original)

    def test_expovariate_mean_relationship(self):
        """Test expovariate lambda parameter relationship to mean."""
        rng = RandomGenerator(seed=42, _internal=True)

        desired_mean = 10.0
        samples = [rng.expovariate(1.0 / desired_mean) for _ in range(10000)]

        actual_mean = sum(samples) / len(samples)
        assert abs(actual_mean - desired_mean) < 1.0

    def test_random_batch_returns_numpy_array(self):
        """Test that random_batch returns numpy array."""
        rng = RandomGenerator(seed=42, _internal=True)

        batch = rng.random_batch(100)

        assert batch.shape == (100,)
        assert all(0.0 <= val < 1.0 for val in batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

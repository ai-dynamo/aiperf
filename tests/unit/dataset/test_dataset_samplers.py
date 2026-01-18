# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dataset sampling strategies."""

import pytest

from aiperf.dataset.dataset_samplers import (
    RandomSampler,
    SequentialSampler,
    ShuffleSampler,
)


class TestSequentialSampler:
    """Tests for SequentialSampler."""

    def test_sequential_iteration(self, conversation_ids: list[str]) -> None:
        """Test that sampler returns IDs in sequential order."""
        sampler = SequentialSampler(conversation_ids)
        expected = conversation_ids.copy()

        for expected_id in expected:
            assert sampler.next_conversation_id() == expected_id

    def test_single_conversation_id(self) -> None:
        """Test sequential sampler with single ID."""
        single_conversation_id = ["conv_only"]
        sampler = SequentialSampler(single_conversation_id)
        assert sampler.next_conversation_id() == single_conversation_id[0]
        assert sampler.next_conversation_id() == single_conversation_id[0]
        assert sampler.next_conversation_id() == single_conversation_id[0]

    def test_does_not_modify_original_list(self, conversation_ids: list[str]) -> None:
        """Test that original conversation_ids list is not modified."""
        original = conversation_ids.copy()
        sampler = SequentialSampler(conversation_ids)

        for _ in range(len(conversation_ids)):
            sampler.next_conversation_id()
        assert conversation_ids == original


class TestRandomSampler:
    """Tests for RandomSampler."""

    def test_returns_valid_ids(self, conversation_ids: list[str]) -> None:
        """Test that sampler only returns IDs from the original list."""
        sampler = RandomSampler(conversation_ids)

        for _ in range(len(conversation_ids) * 3):
            assert sampler.next_conversation_id() in conversation_ids

    def test_sampling_with_replacement(self, conversation_ids: list[str]) -> None:
        """Test that sampler can return the same ID multiple times (with replacement)."""
        sampler = RandomSampler(conversation_ids)
        samples = [sampler.next_conversation_id() for _ in range(100)]

        assert len(set(samples)) < len(samples)

    def test_seed_reproducibility(self, conversation_ids: list[str]) -> None:
        """Test that using the same seed produces identical sequences."""
        from aiperf.common import random_generator as rng

        rng.reset()
        rng.init(42)
        sampler1 = RandomSampler(conversation_ids)
        samples1 = [sampler1.next_conversation_id() for _ in range(20)]

        rng.reset()
        rng.init(42)
        sampler2 = RandomSampler(conversation_ids)
        samples2 = [sampler2.next_conversation_id() for _ in range(20)]

        assert samples1 == samples2

    def test_different_seeds_produce_different_sequences(
        self, conversation_ids: list[str]
    ) -> None:
        """Test that different global seeds produce different sequences."""
        from aiperf.common import random_generator as rng

        rng.reset()
        rng.init(42)
        sampler1 = RandomSampler(conversation_ids)
        samples1 = [sampler1.next_conversation_id() for _ in range(50)]

        rng.reset()
        rng.init(123)
        sampler2 = RandomSampler(conversation_ids)
        samples2 = [sampler2.next_conversation_id() for _ in range(50)]

        assert samples1 != samples2

    def test_single_conversation_id(self) -> None:
        """Test random sampler always returns the only available ID."""
        single_conversation_id = ["conv_only"]
        sampler = RandomSampler(single_conversation_id)

        for _ in range(10):
            assert sampler.next_conversation_id() == single_conversation_id[0]


class TestShuffleSampler:
    """Tests for ShuffleSampler."""

    def test_returns_all_ids_before_repetition(
        self, conversation_ids: list[str]
    ) -> None:
        """Test that all IDs are returned once before any repetition."""
        sampler = ShuffleSampler(conversation_ids)
        first_batch = [
            sampler.next_conversation_id() for _ in range(len(conversation_ids))
        ]

        assert set(first_batch) == set(conversation_ids)
        assert len(first_batch) == len(conversation_ids)

    def test_reshuffles_after_exhaustion(self, conversation_ids: list[str]) -> None:
        """Test that sampler reshuffles and continues after exhausting all IDs."""
        sampler = ShuffleSampler(conversation_ids)

        first_batch = [
            sampler.next_conversation_id() for _ in range(len(conversation_ids))
        ]
        second_batch = [
            sampler.next_conversation_id() for _ in range(len(conversation_ids))
        ]

        assert set(first_batch) == set(conversation_ids)
        assert set(second_batch) == set(conversation_ids)
        assert first_batch != second_batch

    def test_seed_reproducibility(self, conversation_ids: list[str]) -> None:
        """Test that using the same seed produces identical sequences."""
        from aiperf.common import random_generator as rng

        rng.reset()
        rng.init(42)
        sampler1 = ShuffleSampler(conversation_ids)
        samples1 = [
            sampler1.next_conversation_id() for _ in range(len(conversation_ids) * 2)
        ]

        rng.reset()
        rng.init(42)
        sampler2 = ShuffleSampler(conversation_ids)
        samples2 = [
            sampler2.next_conversation_id() for _ in range(len(conversation_ids) * 2)
        ]

        assert samples1 == samples2

    def test_different_shuffles_across_batches(
        self, conversation_ids: list[str]
    ) -> None:
        """Test that subsequent shuffles produce different orderings."""
        sampler = ShuffleSampler(conversation_ids)

        batches = []
        for _ in range(3):
            batch = [
                sampler.next_conversation_id() for _ in range(len(conversation_ids))
            ]
            batches.append(batch)

        assert batches[0] != batches[1]
        assert batches[1] != batches[2]
        assert batches[0] != batches[2]

    def test_single_conversation_id(self) -> None:
        """Test shuffle sampler with single ID."""
        single_conversation_id = ["conv_only"]
        sampler = ShuffleSampler(single_conversation_id)

        for _ in range(10):
            assert sampler.next_conversation_id() == single_conversation_id[0]


class TestSamplerEdgeCases:
    """Parameterized tests for edge cases common to all samplers."""

    @pytest.mark.parametrize(
        "sampler_class",
        [SequentialSampler, RandomSampler, ShuffleSampler],
    )
    def test_empty_conversation_ids_raises_error(self, sampler_class) -> None:
        """Test that all samplers raise ValueError for empty conversation IDs."""
        with pytest.raises(ValueError, match="conversation_ids cannot be empty"):
            sampler_class([])

    @pytest.mark.parametrize(
        "sampler_class",
        [SequentialSampler, RandomSampler, ShuffleSampler],
    )
    def test_accepts_seed_parameter(
        self, sampler_class, conversation_ids: list[str]
    ) -> None:
        """Test that all samplers accept seed parameter without error."""
        sampler = sampler_class(conversation_ids, seed=42)
        assert sampler.next_conversation_id() in conversation_ids

    @pytest.mark.parametrize(
        "sampler_class",
        [SequentialSampler, RandomSampler, ShuffleSampler],
    )
    def test_accepts_none_seed(
        self, sampler_class, conversation_ids: list[str]
    ) -> None:
        """Test that all samplers accept None as seed parameter."""
        sampler = sampler_class(conversation_ids, seed=None)
        assert sampler.next_conversation_id() in conversation_ids


class TestSamplerStatistics:
    """Tests for statistical properties of samplers."""

    def test_random_sampler_distribution(self, conversation_ids: list[str]) -> None:
        """Test that random sampler produces exact expected distribution with seed 42."""
        sampler = RandomSampler(conversation_ids)
        samples = [sampler.next_conversation_id() for _ in range(1000)]

        expected_first_20 = [
            'conv_1', 'conv_5', 'conv_3', 'conv_1', 'conv_1',
            'conv_1', 'conv_3', 'conv_1', 'conv_3', 'conv_4',
            'conv_5', 'conv_5', 'conv_5', 'conv_2', 'conv_5',
            'conv_4', 'conv_1', 'conv_3', 'conv_1', 'conv_2'
        ]  # fmt: skip
        assert samples[:20] == expected_first_20

        counts = {conv_id: samples.count(conv_id) for conv_id in conversation_ids}
        assert all(count > 0 for count in counts.values())
        assert sum(counts.values()) == 1000

    def test_shuffle_sampler_equal_frequency_per_batch(
        self, conversation_ids: list[str]
    ) -> None:
        """Test that shuffle sampler returns each ID exactly once per batch."""
        sampler = ShuffleSampler(conversation_ids)
        num_batches = 10
        total_samples = len(conversation_ids) * num_batches

        samples = [sampler.next_conversation_id() for _ in range(total_samples)]
        counts = {conv_id: samples.count(conv_id) for conv_id in conversation_ids}

        for count in counts.values():
            assert count == num_batches

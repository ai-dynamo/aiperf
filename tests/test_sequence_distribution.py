# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive tests for sequence length distribution functionality.

This test suite covers all aspects of the sequence distribution feature including:
- Basic sequence length pair validation and behavior
- Distribution sampling with probabilistic validation
- Multi-format parsing (semicolon, bracket, JSON) with and without standard deviations
- Statistical validation of sampling behavior including variance testing
- Integration with PromptConfig and CLI parameter validation
- Edge cases, error handling, and boundary conditions
"""

import numpy as np
import pytest

from aiperf.common.models.sequence_distribution import (
    DistributionParser,
    SequenceLengthDistribution,
    SequenceLengthPair,
    _sample_positive_normal_integer,
    create_balanced_distribution,
    create_uniform_distribution,
)


class TestSequenceLengthPair:
    """Test SequenceLengthPair validation and behavior."""

    def test_valid_pair_creation(self):
        """Test creating valid sequence length pairs."""
        pair = SequenceLengthPair(256, 128, 50.0)
        assert pair.input_seq_len == 256
        assert pair.output_seq_len == 128
        assert pair.probability == 50.0

    def test_valid_pair_with_stddev(self):
        """Test creating valid sequence length pairs with standard deviations."""
        pair = SequenceLengthPair(256, 128, 50.0, 10.0, 5.0)
        assert pair.input_seq_len == 256
        assert pair.output_seq_len == 128
        assert pair.probability == 50.0
        assert pair.input_seq_len_stddev == 10.0
        assert pair.output_seq_len_stddev == 5.0

    def test_invalid_input_length(self):
        """Test validation of input sequence length."""
        with pytest.raises(ValueError, match="Input sequence length must be positive"):
            SequenceLengthPair(0, 128, 50.0)

        with pytest.raises(ValueError):
            SequenceLengthPair(-1, 128, 50.0)

    def test_invalid_output_length(self):
        """Test validation of output sequence length."""
        with pytest.raises(ValueError, match="Output sequence length must be positive"):
            SequenceLengthPair(256, 0, 50.0)

        with pytest.raises(ValueError):
            SequenceLengthPair(256, -1, 50.0)

    def test_invalid_probability(self):
        """Test validation of probability values."""
        with pytest.raises(ValueError, match="Probability must be in \\[0,100\\]"):
            SequenceLengthPair(256, 128, -10.0)

        with pytest.raises(ValueError):
            SequenceLengthPair(256, 128, 110.0)

    def test_invalid_input_stddev(self):
        """Test validation of negative input sequence length standard deviation."""
        with pytest.raises(
            ValueError, match="Input sequence length stddev must be non-negative"
        ):
            SequenceLengthPair(256, 128, 50.0, input_seq_len_stddev=-1.0)

    def test_invalid_output_stddev(self):
        """Test validation of negative output sequence length standard deviation."""
        with pytest.raises(
            ValueError, match="Output sequence length stddev must be non-negative"
        ):
            SequenceLengthPair(256, 128, 50.0, output_seq_len_stddev=-2.0)

    def test_boundary_probabilities(self):
        """Test boundary probability values."""
        # Should work
        SequenceLengthPair(256, 128, 0.0)
        SequenceLengthPair(256, 128, 100.0)

    def test_immutability(self):
        """Test that pairs are immutable."""
        pair = SequenceLengthPair(256, 128, 50.0)
        with pytest.raises(AttributeError):
            pair.input_seq_len = 512

    def test_string_representation_no_stddev(self):
        """Test string representation without standard deviations."""
        pair = SequenceLengthPair(256, 128, 40.0)
        assert str(pair) == "(256,128):40.0%"

    def test_string_representation_with_stddev(self):
        """Test string representation with standard deviations."""
        pair = SequenceLengthPair(256, 128, 40.0, 10.0, 5.0)
        assert str(pair) == "(256|10.0,128|5.0):40.0%"

    def test_string_representation_partial_stddev(self):
        """Test string representation with only input stddev."""
        pair = SequenceLengthPair(256, 128, 40.0, 10.0, 0.0)
        assert str(pair) == "(256|10.0,128|0.0):40.0%"


class TestSequenceLengthDistribution:
    """Test SequenceLengthDistribution functionality."""

    @pytest.fixture(autouse=True)
    def setup_distributions(self):
        """Set up test distributions."""
        self.single_pair = [SequenceLengthPair(256, 128, 100.0)]
        self.multi_pair = [
            SequenceLengthPair(256, 128, 60.0),
            SequenceLengthPair(512, 256, 40.0),
        ]
        self.stddev_pairs = [
            SequenceLengthPair(256, 128, 60.0, 20.0, 10.0),
            SequenceLengthPair(512, 256, 40.0, 30.0, 15.0),
        ]

    def test_single_pair_distribution(self):
        """Test distribution with single pair."""
        dist = SequenceLengthDistribution(self.single_pair)

        # Should always return the same pair
        for _ in range(100):
            isl, osl = dist.sample()
            assert isl == 256
            assert osl == 128

    def test_multi_pair_distribution_sampling(self):
        """Test sampling from multi-pair distribution."""
        dist = SequenceLengthDistribution(self.multi_pair)

        # Sample many times and verify approximate distribution
        rng = np.random.default_rng(42)
        samples = [dist.sample(random_state=rng) for _ in range(10000)]

        count_256_128 = sum(1 for s in samples if s == (256, 128))
        count_512_256 = sum(1 for s in samples if s == (512, 256))

        # Should be approximately 60/40 split (±5%)
        assert abs(count_256_128 / len(samples) - 0.6) < 0.05
        assert abs(count_512_256 / len(samples) - 0.4) < 0.05

    def test_stddev_distribution_sampling(self):
        """Test sampling from distribution with standard deviations."""
        dist = SequenceLengthDistribution(self.stddev_pairs)

        # Sample many times to test variance
        rng = np.random.default_rng(42)
        samples = [dist.sample(random_state=rng) for _ in range(1000)]

        isl_values = [s[0] for s in samples]
        osl_values = [s[1] for s in samples]

        # Should have variance > 0 due to stddev
        assert np.std(isl_values) > 5, "ISL should vary due to standard deviation"
        assert np.std(osl_values) > 3, "OSL should vary due to standard deviation"

        # All values should be positive (clamped)
        assert all(isl > 0 for isl in isl_values)
        assert all(osl > 0 for osl in osl_values)

    def test_batch_sampling(self):
        """Test efficient batch sampling."""
        dist = SequenceLengthDistribution(self.multi_pair)

        batch = dist.sample_batch(1000, random_state=42)
        assert len(batch) == 1000

        # Verify all samples are valid
        for isl, osl in batch:
            assert (isl, osl) in [(256, 128), (512, 256)]

    def test_reproducible_sampling(self):
        """Test that sampling is reproducible with same seed."""
        dist = SequenceLengthDistribution(self.multi_pair)

        samples1 = [dist.sample(random_state=123) for _ in range(100)]
        samples2 = [dist.sample(random_state=123) for _ in range(100)]

        assert samples1 == samples2

    def test_empty_pairs_validation(self):
        """Test validation of empty pairs list."""
        with pytest.raises(ValueError, match="at least one sequence length pair"):
            SequenceLengthDistribution([])

    def test_probability_sum_validation(self):
        """Test validation of probability sum."""
        # Probabilities don't sum to 100.0
        invalid_pairs = [
            SequenceLengthPair(256, 128, 30.0),
            SequenceLengthPair(512, 256, 40.0),  # Sum = 70.0
        ]

        with pytest.raises(ValueError, match="must sum to 100.0"):
            SequenceLengthDistribution(invalid_pairs)

    def test_probability_sum_tolerance(self):
        """Test that small floating-point errors are tolerated."""
        # Slightly off due to floating point precision
        pairs = [
            SequenceLengthPair(256, 128, 60.0),
            SequenceLengthPair(512, 256, 40.0000000001),  # Sum ≈ 100.0
        ]

        # Should not raise exception
        dist = SequenceLengthDistribution(pairs)
        assert dist is not None

    def test_statistics_calculation(self):
        """Test distribution statistics calculation."""
        dist = SequenceLengthDistribution(self.multi_pair)
        stats = dist.get_statistics()

        # Expected ISL: 256*0.6 + 512*0.4 = 358.4
        assert abs(stats["expected_isl"] - 358.4) < 0.1

        # Expected OSL: 128*0.6 + 256*0.4 = 179.2
        assert abs(stats["expected_osl"] - 179.2) < 0.1

        assert stats["num_pairs"] == 2
        assert abs(stats["total_probability"] - 100.0) < 1e-10

    def test_string_representation(self):
        """Test string representation."""
        dist = SequenceLengthDistribution(self.multi_pair)
        str_repr = str(dist)

        assert "(256,128):60.0%" in str_repr
        assert "(512,256):40.0%" in str_repr


class TestSamplePositiveNormalInteger:
    """Test the internal _sample_positive_normal_integer function."""

    def test_zero_stddev_returns_mean(self):
        """Test that zero stddev returns the mean value."""
        result = _sample_positive_normal_integer(100.0, 0.0)
        assert result == 100

    def test_negative_stddev_returns_mean(self):
        """Test that negative stddev returns the mean value."""
        result = _sample_positive_normal_integer(100.0, -5.0)
        assert result == 100

    def test_positive_stddev_varies(self):
        """Test that positive stddev produces variance around mean."""
        np.random.seed(42)
        samples = [_sample_positive_normal_integer(100.0, 20.0) for _ in range(1000)]

        # Should have variance > 0
        assert np.std(samples) > 5

        # All values should be positive
        assert all(s > 0 for s in samples)

        # Mean should be approximately 100
        assert abs(np.mean(samples) - 100.0) < 5.0

    def test_clamping_to_positive(self):
        """Test that negative samples are clamped to 1."""
        np.random.seed(123)  # Seed that might produce negative values
        samples = [_sample_positive_normal_integer(2.0, 10.0) for _ in range(1000)]

        # All values must be at least 1
        assert all(s >= 1 for s in samples)


class TestDistributionParser:
    """Test distribution string parsing."""

    def test_semicolon_format_parsing(self):
        """Test parsing semicolon-separated format with percentages."""
        dist_str = "256,128:60;512,256:40"
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0)
        assert dist.pairs[1] == SequenceLengthPair(512, 256, 40.0)

    def test_semicolon_format_with_stddev(self):
        """Test parsing semicolon-separated format with standard deviations."""
        dist_str = "256|10,128|5:60;512|20,256|15:40"
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0, 10.0, 5.0)
        assert dist.pairs[1] == SequenceLengthPair(512, 256, 40.0, 20.0, 15.0)

    def test_semicolon_format_mixed_stddev(self):
        """Test parsing with some pairs having stddev and others not."""
        dist_str = "256|10,128:60;512,256|15:40"
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0, 10.0, 0.0)
        assert dist.pairs[1] == SequenceLengthPair(512, 256, 40.0, 0.0, 15.0)

    def test_semicolon_format_invalid_fractions(self):
        """Test that fractions are properly rejected (percentage-only enforcement)."""
        dist_str = "256,128:0.6;512,256:0.4"
        with pytest.raises(ValueError, match="Probabilities must sum to 100.0"):
            DistributionParser.parse(dist_str)

    def test_bracket_format_parsing(self):
        """Test parsing bracket format with percentages."""
        dist_str = "[(256,128):60,(512,256):40]"
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0)
        assert dist.pairs[1] == SequenceLengthPair(512, 256, 40.0)

    def test_bracket_format_with_stddev(self):
        """Test parsing bracket format with standard deviations."""
        dist_str = "[(256|10,128|5):60,(512|20,256|15):40]"
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0, 10.0, 5.0)
        assert dist.pairs[1] == SequenceLengthPair(512, 256, 40.0, 20.0, 15.0)

    def test_json_format_parsing(self):
        """Test parsing JSON format with percentages."""
        dist_str = '{"pairs": [{"isl": 256, "osl": 128, "prob": 60}, {"isl": 512, "osl": 256, "prob": 40}]}'
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0)
        assert dist.pairs[1] == SequenceLengthPair(512, 256, 40.0)

    def test_json_format_with_stddev(self):
        """Test parsing JSON format with standard deviations."""
        dist_str = '{"pairs": [{"isl": 256, "isl_stddev": 10, "osl": 128, "osl_stddev": 5, "prob": 60}, {"isl": 512, "isl_stddev": 20, "osl": 256, "osl_stddev": 15, "prob": 40}]}'
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0, 10.0, 5.0)
        assert dist.pairs[1] == SequenceLengthPair(512, 256, 40.0, 20.0, 15.0)

    def test_single_pair_parsing(self):
        """Test parsing single pair."""
        dist_str = "1024,512:100"
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 1
        assert dist.pairs[0] == SequenceLengthPair(1024, 512, 100.0)

    def test_single_pair_with_stddev(self):
        """Test parsing single pair with standard deviations."""
        dist_str = "1024|50,512|25:100"
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 1
        assert dist.pairs[0] == SequenceLengthPair(1024, 512, 100.0, 50.0, 25.0)

    def test_invalid_format_parsing(self):
        """Test parsing invalid formats."""
        invalid_formats = [
            "",
            "256,128",  # Missing probability
            "256:50",  # Missing OSL
            "invalid",
            "256,128:110",  # Invalid probability (>100)
            "256,128:-10",  # Invalid probability (<0)
            "256,128:0.6",  # Fraction not allowed (percentage-only)
            '{"invalid": "json"}',  # Invalid JSON structure
            "256|,128:100",  # Empty stddev
            "256|-5,128:100",  # Negative stddev
        ]

        for invalid_str in invalid_formats:
            with pytest.raises(ValueError):
                DistributionParser.parse(invalid_str)

    def test_decimal_probabilities(self):
        """Test parsing with decimal percentages."""
        dist_str = "256,128:33.3;512,256:66.7"
        dist = DistributionParser.parse(dist_str)

        assert abs(dist.pairs[0].probability - 33.3) < 1e-10
        assert abs(dist.pairs[1].probability - 66.7) < 1e-10

    def test_whitespace_handling(self):
        """Test parsing with various whitespace."""
        dist_str = "  256 , 128 : 60 ; 512 , 256 : 40  "
        dist = DistributionParser.parse(dist_str)

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_uniform_distribution(self):
        """Test creating uniform single-pair distribution."""
        dist = create_uniform_distribution(512, 256)

        assert len(dist.pairs) == 1
        assert dist.pairs[0] == SequenceLengthPair(512, 256, 100.0)

        # Should always return the same values
        for _ in range(10):
            isl, osl = dist.sample()
            assert isl == 512
            assert osl == 256

    def test_create_balanced_distribution(self):
        """Test creating balanced distribution."""
        pairs = [(256, 128), (512, 256), (1024, 512)]
        dist = create_balanced_distribution(pairs)

        assert len(dist.pairs) == 3

        # All probabilities should be 100/3 ≈ 33.33%
        for pair in dist.pairs:
            assert abs(pair.probability - 100.0 / 3.0) < 1e-10

    def test_create_balanced_empty_pairs(self):
        """Test creating balanced distribution with empty pairs."""
        with pytest.raises(ValueError):
            create_balanced_distribution([])


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from string to sampling."""
        # Parse distribution
        dist_str = "128,64:30;256,128:50;512,256:20"
        dist = DistributionParser.parse(dist_str)

        # Sample many times
        samples = dist.sample_batch(10000, random_state=42)

        # Verify distribution
        counts = {}
        for sample in samples:
            counts[sample] = counts.get(sample, 0) + 1

        total = len(samples)
        assert abs(counts[(128, 64)] / total - 0.3) < 0.05
        assert abs(counts[(256, 128)] / total - 0.5) < 0.05
        assert abs(counts[(512, 256)] / total - 0.2) < 0.05

    def test_end_to_end_workflow_with_stddev(self):
        """Test complete workflow with standard deviations from string to sampling."""
        # Parse distribution with stddev
        dist_str = "128|20,64|10:30;256|30,128|15:70"
        dist = DistributionParser.parse(dist_str)

        # Sample many times
        samples = dist.sample_batch(5000, random_state=42)

        # Check that we have variance (due to stddev)
        isl_values = [s[0] for s in samples]
        osl_values = [s[1] for s in samples]

        assert np.std(isl_values) > 10, "Should have ISL variance due to stddev"
        assert np.std(osl_values) > 5, "Should have OSL variance due to stddev"

        # All values should still be positive
        assert all(isl > 0 for isl in isl_values)
        assert all(osl > 0 for osl in osl_values)

    def test_statistics_accuracy(self):
        """Test that calculated statistics match empirical results."""
        pairs = [
            SequenceLengthPair(100, 50, 20.0),
            SequenceLengthPair(200, 100, 30.0),
            SequenceLengthPair(300, 150, 50.0),
        ]
        dist = SequenceLengthDistribution(pairs)

        # Get theoretical statistics
        stats = dist.get_statistics()

        # Sample empirically
        samples = dist.sample_batch(50000, random_state=123)
        empirical_isl = np.mean([s[0] for s in samples])
        empirical_osl = np.mean([s[1] for s in samples])

        # Should match within 1%
        assert abs(empirical_isl - stats["expected_isl"]) < stats["expected_isl"] * 0.01
        assert abs(empirical_osl - stats["expected_osl"]) < stats["expected_osl"] * 0.01


class TestPromptConfigIntegration:
    """Test integration with PromptConfig."""

    def test_get_sequence_distribution_with_explicit_dist(self):
        """Test getting distribution when sequence_distribution is set."""
        from aiperf.common.config.prompt_config import PromptConfig

        config = PromptConfig()
        config.sequence_distribution = "256,128:60;512,256:40"

        dist = config.get_sequence_distribution()

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0)
        assert dist.pairs[1] == SequenceLengthPair(512, 256, 40.0)

    def test_get_sequence_distribution_with_stddev(self):
        """Test getting distribution with standard deviations."""
        from aiperf.common.config.prompt_config import PromptConfig

        config = PromptConfig()
        config.sequence_distribution = "256|20,128|10:60;512|30,256|15:40"

        dist = config.get_sequence_distribution()

        assert len(dist.pairs) == 2
        assert dist.pairs[0] == SequenceLengthPair(256, 128, 60.0, 20.0, 10.0)
        assert dist.pairs[1] == SequenceLengthPair(512, 256, 40.0, 30.0, 15.0)

    def test_get_sequence_distribution_fallback_to_isl_osl(self):
        """Test that None is returned when no distribution is set."""
        from aiperf.common.config.prompt_config import PromptConfig

        config = PromptConfig()
        config.sequence_distribution = None
        config.input_tokens.mean = 512
        config.output_tokens.mean = 256

        dist = config.get_sequence_distribution()

        assert dist is None

    def test_get_sequence_distribution_default_osl(self):
        """Test that None is returned when no distribution is specified."""
        from aiperf.common.config.prompt_config import PromptConfig

        config = PromptConfig()
        config.sequence_distribution = None
        config.input_tokens.mean = 512
        config.output_tokens.mean = None  # Not specified

        dist = config.get_sequence_distribution()

        assert dist is None


class TestSequenceCaching:
    """Test turn-level sequence length caching for ISL/OSL consistency."""

    def test_turn_sequence_caching(self):
        """Test that sequence lengths are cached per turn for consistency."""
        import numpy as np

        from aiperf.common.models import Turn
        from aiperf.dataset.composer.base import BaseDatasetComposer

        # Create mock composer
        class MockComposer(BaseDatasetComposer):
            def create_dataset(self):
                return []

        # Set up composer with distribution
        composer = MockComposer.__new__(MockComposer)
        dist = DistributionParser.parse("128,64:50;256,128:50")
        composer._seq_distribution = dist
        composer._turn_sequence_cache = {}
        composer._seq_rng = np.random.default_rng(42)

        # Create a turn and get its ID
        turn = Turn()
        turn_id = id(turn)

        # Sample multiple times - should get same result due to caching
        isl1, osl1 = composer._get_turn_sequence_lengths(turn_id)
        isl2, osl2 = composer._get_turn_sequence_lengths(turn_id)
        isl3, osl3 = composer._get_turn_sequence_lengths(turn_id)

        # All calls should return the same cached result
        assert isl1 == isl2 == isl3
        assert osl1 == osl2 == osl3

        # Result should be a valid pair from the distribution
        valid_pairs = [(128, 64), (256, 128)]
        assert (isl1, osl1) in valid_pairs

        # Test cache clearing
        composer._clear_turn_cache(turn_id)

        # After clearing, can sample again (may get different result)
        isl4, osl4 = composer._get_turn_sequence_lengths(turn_id)
        assert (isl4, osl4) in valid_pairs

    def test_different_turns_get_different_cache_entries(self):
        """Test that different turns can have different cached sequence lengths."""
        import numpy as np

        from aiperf.common.models import Turn
        from aiperf.dataset.composer.base import BaseDatasetComposer

        # Create mock composer
        class MockComposer(BaseDatasetComposer):
            def create_dataset(self):
                return []

        # Set up composer with distribution
        composer = MockComposer.__new__(MockComposer)
        dist = DistributionParser.parse(
            "100,50:100"
        )  # Single pair for predictable results
        composer._seq_distribution = dist
        composer._turn_sequence_cache = {}
        composer._seq_rng = np.random.default_rng(42)

        # Create two different turns
        turn1 = Turn()
        turn2 = Turn()
        turn1_id = id(turn1)
        turn2_id = id(turn2)

        # Get sequence lengths for both turns
        isl1, osl1 = composer._get_turn_sequence_lengths(turn1_id)
        isl2, osl2 = composer._get_turn_sequence_lengths(turn2_id)

        # Both should be the same since there's only one pair in the distribution
        assert (isl1, osl1) == (100, 50)
        assert (isl2, osl2) == (100, 50)

        # But they should be cached separately
        assert turn1_id in composer._turn_sequence_cache
        assert turn2_id in composer._turn_sequence_cache

        # Clearing one shouldn't affect the other
        composer._clear_turn_cache(turn1_id)
        assert turn1_id not in composer._turn_sequence_cache
        assert turn2_id in composer._turn_sequence_cache

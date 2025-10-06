# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for sequence length distribution functionality."""

import unittest

import numpy as np

from aiperf.common.models.sequence_distribution import (
    DistributionParser,
    SequenceLengthDistribution,
    SequenceLengthPair,
    create_balanced_distribution,
    create_uniform_distribution,
)


class TestSequenceLengthPair(unittest.TestCase):
    """Test SequenceLengthPair validation and behavior."""

    def test_valid_pair_creation(self):
        """Test creating valid sequence length pairs."""
        pair = SequenceLengthPair(256, 128, 50.0)
        self.assertEqual(pair.input_seq_len, 256)
        self.assertEqual(pair.output_seq_len, 128)
        self.assertEqual(pair.probability, 50.0)

    def test_invalid_input_length(self):
        """Test validation of input sequence length."""
        with self.assertRaises(ValueError) as cm:
            SequenceLengthPair(0, 128, 50.0)
        self.assertIn("Input sequence length must be positive", str(cm.exception))

        with self.assertRaises(ValueError):
            SequenceLengthPair(-1, 128, 50.0)

    def test_invalid_output_length(self):
        """Test validation of output sequence length."""
        with self.assertRaises(ValueError) as cm:
            SequenceLengthPair(256, 0, 50.0)
        self.assertIn("Output sequence length must be positive", str(cm.exception))

        with self.assertRaises(ValueError):
            SequenceLengthPair(256, -1, 50.0)

    def test_invalid_probability(self):
        """Test validation of probability values."""
        with self.assertRaises(ValueError) as cm:
            SequenceLengthPair(256, 128, -10.0)
        self.assertIn("Probability must be in [0,100]", str(cm.exception))

        with self.assertRaises(ValueError):
            SequenceLengthPair(256, 128, 110.0)

    def test_boundary_probabilities(self):
        """Test boundary probability values."""
        # Should work
        SequenceLengthPair(256, 128, 0.0)
        SequenceLengthPair(256, 128, 100.0)

    def test_immutability(self):
        """Test that pairs are immutable."""
        pair = SequenceLengthPair(256, 128, 50.0)
        with self.assertRaises(AttributeError):
            pair.input_seq_len = 512

    def test_string_representation(self):
        """Test string representation."""
        pair = SequenceLengthPair(256, 128, 40.0)
        self.assertEqual(str(pair), "(256,128):40.0%")


class TestSequenceLengthDistribution(unittest.TestCase):
    """Test SequenceLengthDistribution functionality."""

    def setUp(self):
        """Set up test distributions."""
        self.single_pair = [SequenceLengthPair(256, 128, 100.0)]
        self.multi_pair = [
            SequenceLengthPair(256, 128, 60.0),
            SequenceLengthPair(512, 256, 40.0),
        ]

    def test_single_pair_distribution(self):
        """Test distribution with single pair."""
        dist = SequenceLengthDistribution(self.single_pair)

        # Should always return the same pair
        for _ in range(100):
            isl, osl = dist.sample()
            self.assertEqual(isl, 256)
            self.assertEqual(osl, 128)

    def test_multi_pair_distribution_sampling(self):
        """Test sampling from multi-pair distribution."""
        dist = SequenceLengthDistribution(self.multi_pair)

        # Sample many times and verify approximate distribution
        rng = np.random.default_rng(42)
        samples = [dist.sample(random_state=rng) for _ in range(10000)]

        count_256_128 = sum(1 for s in samples if s == (256, 128))
        count_512_256 = sum(1 for s in samples if s == (512, 256))

        # Should be approximately 60/40 split (±5%)
        self.assertAlmostEqual(count_256_128 / len(samples), 0.6, delta=0.05)
        self.assertAlmostEqual(count_512_256 / len(samples), 0.4, delta=0.05)

    def test_batch_sampling(self):
        """Test efficient batch sampling."""
        dist = SequenceLengthDistribution(self.multi_pair)

        batch = dist.sample_batch(1000, random_state=42)
        self.assertEqual(len(batch), 1000)

        # Verify all samples are valid
        for isl, osl in batch:
            self.assertIn((isl, osl), [(256, 128), (512, 256)])

    def test_reproducible_sampling(self):
        """Test that sampling is reproducible with same seed."""
        dist = SequenceLengthDistribution(self.multi_pair)

        samples1 = [dist.sample(random_state=123) for _ in range(100)]
        samples2 = [dist.sample(random_state=123) for _ in range(100)]

        self.assertEqual(samples1, samples2)

    def test_empty_pairs_validation(self):
        """Test validation of empty pairs list."""
        with self.assertRaises(ValueError) as cm:
            SequenceLengthDistribution([])
        self.assertIn("at least one sequence length pair", str(cm.exception))

    def test_probability_sum_validation(self):
        """Test validation of probability sum."""
        # Probabilities don't sum to 100.0
        invalid_pairs = [
            SequenceLengthPair(256, 128, 30.0),
            SequenceLengthPair(512, 256, 40.0),  # Sum = 70.0
        ]

        with self.assertRaises(ValueError) as cm:
            SequenceLengthDistribution(invalid_pairs)
        self.assertIn("must sum to 100.0", str(cm.exception))

    def test_probability_sum_tolerance(self):
        """Test that small floating-point errors are tolerated."""
        # Slightly off due to floating point precision
        pairs = [
            SequenceLengthPair(256, 128, 60.0),
            SequenceLengthPair(512, 256, 40.0000000001),  # Sum ≈ 100.0
        ]

        # Should not raise exception
        dist = SequenceLengthDistribution(pairs)
        self.assertIsNotNone(dist)

    def test_statistics_calculation(self):
        """Test distribution statistics calculation."""
        dist = SequenceLengthDistribution(self.multi_pair)
        stats = dist.get_statistics()

        # Expected ISL: 256*0.6 + 512*0.4 = 358.4
        self.assertAlmostEqual(stats["expected_isl"], 358.4, places=1)

        # Expected OSL: 128*0.6 + 256*0.4 = 179.2
        self.assertAlmostEqual(stats["expected_osl"], 179.2, places=1)

        self.assertEqual(stats["num_pairs"], 2)
        self.assertAlmostEqual(stats["total_probability"], 100.0)

    def test_string_representation(self):
        """Test string representation."""
        dist = SequenceLengthDistribution(self.multi_pair)
        str_repr = str(dist)

        self.assertIn("(256,128):60.0%", str_repr)
        self.assertIn("(512,256):40.0%", str_repr)


class TestDistributionParser(unittest.TestCase):
    """Test distribution string parsing."""

    def test_semicolon_format_parsing(self):
        """Test parsing semicolon-separated format with percentages."""
        dist_str = "256,128:60;512,256:40"
        dist = DistributionParser.parse(dist_str)

        self.assertEqual(len(dist.pairs), 2)
        self.assertEqual(dist.pairs[0], SequenceLengthPair(256, 128, 60.0))
        self.assertEqual(dist.pairs[1], SequenceLengthPair(512, 256, 40.0))

    def test_semicolon_format_backward_compatibility(self):
        """Test parsing semicolon-separated format with fractions (backward compatibility)."""
        dist_str = "256,128:0.6;512,256:0.4"
        dist = DistributionParser.parse(dist_str)

        self.assertEqual(len(dist.pairs), 2)
        self.assertEqual(dist.pairs[0], SequenceLengthPair(256, 128, 60.0))
        self.assertEqual(dist.pairs[1], SequenceLengthPair(512, 256, 40.0))

    def test_bracket_format_parsing(self):
        """Test parsing bracket format with percentages."""
        dist_str = "[(256,128):60,(512,256):40]"
        dist = DistributionParser.parse(dist_str)

        self.assertEqual(len(dist.pairs), 2)
        self.assertEqual(dist.pairs[0], SequenceLengthPair(256, 128, 60.0))
        self.assertEqual(dist.pairs[1], SequenceLengthPair(512, 256, 40.0))

    def test_json_format_parsing(self):
        """Test parsing JSON format with percentages."""
        dist_str = '{"pairs": [{"isl": 256, "osl": 128, "prob": 60}, {"isl": 512, "osl": 256, "prob": 40}]}'
        dist = DistributionParser.parse(dist_str)

        self.assertEqual(len(dist.pairs), 2)
        self.assertEqual(dist.pairs[0], SequenceLengthPair(256, 128, 60.0))
        self.assertEqual(dist.pairs[1], SequenceLengthPair(512, 256, 40.0))

    def test_single_pair_parsing(self):
        """Test parsing single pair."""
        dist_str = "1024,512:100"
        dist = DistributionParser.parse(dist_str)

        self.assertEqual(len(dist.pairs), 1)
        self.assertEqual(dist.pairs[0], SequenceLengthPair(1024, 512, 100.0))

    def test_invalid_format_parsing(self):
        """Test parsing invalid formats."""
        invalid_formats = [
            "",
            "256,128",  # Missing probability
            "256:0.5",  # Missing OSL
            "invalid",
            "256,128:110",  # Invalid probability
            '{"invalid": "json"}',  # Invalid JSON structure
        ]

        for invalid_str in invalid_formats:
            with self.assertRaises(ValueError):
                DistributionParser.parse(invalid_str)

    def test_decimal_probabilities(self):
        """Test parsing with decimal percentages."""
        dist_str = "256,128:33.3;512,256:66.7"
        dist = DistributionParser.parse(dist_str)

        self.assertAlmostEqual(dist.pairs[0].probability, 33.3)
        self.assertAlmostEqual(dist.pairs[1].probability, 66.7)

    def test_whitespace_handling(self):
        """Test parsing with various whitespace."""
        dist_str = "  256 , 128 : 60 ; 512 , 256 : 40  "
        dist = DistributionParser.parse(dist_str)

        self.assertEqual(len(dist.pairs), 2)
        self.assertEqual(dist.pairs[0], SequenceLengthPair(256, 128, 60.0))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_create_uniform_distribution(self):
        """Test creating uniform single-pair distribution."""
        dist = create_uniform_distribution(512, 256)

        self.assertEqual(len(dist.pairs), 1)
        self.assertEqual(dist.pairs[0], SequenceLengthPair(512, 256, 100.0))

        # Should always return the same values
        for _ in range(10):
            isl, osl = dist.sample()
            self.assertEqual(isl, 512)
            self.assertEqual(osl, 256)

    def test_create_balanced_distribution(self):
        """Test creating balanced distribution."""
        pairs = [(256, 128), (512, 256), (1024, 512)]
        dist = create_balanced_distribution(pairs)

        self.assertEqual(len(dist.pairs), 3)

        # All probabilities should be 100/3 ≈ 33.33%
        for pair in dist.pairs:
            self.assertAlmostEqual(pair.probability, 100.0 / 3.0, places=10)

    def test_create_balanced_empty_pairs(self):
        """Test creating balanced distribution with empty pairs."""
        with self.assertRaises(ValueError):
            create_balanced_distribution([])


class TestIntegration(unittest.TestCase):
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
        self.assertAlmostEqual(counts[(128, 64)] / total, 0.3, delta=0.05)
        self.assertAlmostEqual(counts[(256, 128)] / total, 0.5, delta=0.05)
        self.assertAlmostEqual(counts[(512, 256)] / total, 0.2, delta=0.05)

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
        self.assertAlmostEqual(
            empirical_isl, stats["expected_isl"], delta=stats["expected_isl"] * 0.01
        )
        self.assertAlmostEqual(
            empirical_osl, stats["expected_osl"], delta=stats["expected_osl"] * 0.01
        )


class TestPromptConfigIntegration(unittest.TestCase):
    """Test integration with PromptConfig."""

    def test_get_sequence_distribution_with_explicit_dist(self):
        """Test getting distribution when sequence_distribution is set."""
        from aiperf.common.config.prompt_config import PromptConfig

        config = PromptConfig()
        config.sequence_distribution = "256,128:60;512,256:40"

        dist = config.get_sequence_distribution()

        self.assertEqual(len(dist.pairs), 2)
        self.assertEqual(dist.pairs[0], SequenceLengthPair(256, 128, 60.0))
        self.assertEqual(dist.pairs[1], SequenceLengthPair(512, 256, 40.0))

    def test_get_sequence_distribution_fallback_to_isl_osl(self):
        """Test that None is returned when no distribution is set."""
        from aiperf.common.config.prompt_config import PromptConfig

        config = PromptConfig()
        config.sequence_distribution = None
        config.input_tokens.mean = 512
        config.output_tokens.mean = 256

        dist = config.get_sequence_distribution()

        self.assertIsNone(dist)

    def test_get_sequence_distribution_default_osl(self):
        """Test that None is returned when no distribution is specified."""
        from aiperf.common.config.prompt_config import PromptConfig

        config = PromptConfig()
        config.sequence_distribution = None
        config.input_tokens.mean = 512
        config.output_tokens.mean = None  # Not specified

        dist = config.get_sequence_distribution()

        self.assertIsNone(dist)


if __name__ == "__main__":
    unittest.main()

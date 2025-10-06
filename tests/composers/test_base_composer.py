# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aiperf.common.config import UserConfig
from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import Turn
from aiperf.common.models.sequence_distribution import (
    SequenceLengthDistribution,
)
from aiperf.dataset.composer.base import BaseDatasetComposer


class ConcreteBaseComposer(BaseDatasetComposer):
    """Concrete test implementation of BaseDatasetComposer."""

    def create_dataset(self):
        """Required abstract method implementation."""
        return []


class TestBaseDatasetComposer:
    """Test class for BaseDatasetComposer functionality."""

    @pytest.fixture
    def base_config(self):
        """Create a basic configuration for testing."""
        config_dict = {
            "endpoint": {
                "model_names": ["test-model-1", "test-model-2"],
                "model_selection_strategy": ModelSelectionStrategy.ROUND_ROBIN,
            },
            "input": {
                "conversation": {"num": 1, "turn": {"mean": 1}},
                "prompt": {
                    "input_tokens": {"mean": 100, "stddev": 10},
                    "output_tokens": {"mean": 50, "stddev": 5},
                    "batch_size": 1,
                    "prefix_prompt": {"length": 10},
                },
            },
        }
        return UserConfig(**config_dict)

    @pytest.fixture
    def sequence_dist_config(self):
        """Create configuration with sequence distribution."""
        config_dict = {
            "endpoint": {
                "model_names": ["test-model"],
                "model_selection_strategy": ModelSelectionStrategy.ROUND_ROBIN,
            },
            "input": {
                "conversation": {"num": 1, "turn": {"mean": 1}},
                "prompt": {
                    "input_tokens": {"mean": 100, "stddev": 10},
                    "output_tokens": {"mean": 50, "stddev": 5},
                    "batch_size": 1,
                    "sequence_distribution": "100,25:50;200,50:50",
                },
            },
        }
        return UserConfig(**config_dict)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        return MagicMock()

    def test_initialization_with_sequence_distribution(
        self, sequence_dist_config, mock_tokenizer
    ):
        """Test initialization with sequence distribution."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)

        assert composer._seq_distribution is not None
        assert isinstance(composer._seq_distribution, SequenceLengthDistribution)
        assert len(composer._seq_distribution.pairs) == 2
        assert composer._seq_rng is None  # No seed provided
        assert len(composer._turn_sequence_cache) == 0

    def test_initialization_with_random_seed(
        self, sequence_dist_config, mock_tokenizer
    ):
        """Test initialization with random seed."""
        sequence_dist_config.input.random_seed = 42
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)

        assert composer._seq_rng is not None
        assert isinstance(composer._seq_rng, np.random.Generator)

    def test_initialization_without_sequence_distribution(
        self, base_config, mock_tokenizer
    ):
        """Test initialization without sequence distribution."""
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        assert composer._seq_distribution is None
        assert composer._seq_rng is None

    def test_model_selection_round_robin(self, base_config, mock_tokenizer):
        """Test round robin model selection."""
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        # Test round robin selection
        assert composer._select_model_name() == "test-model-1"
        assert composer._select_model_name() == "test-model-2"
        assert composer._select_model_name() == "test-model-1"  # Wraps around

    def test_model_selection_random(self, base_config, mock_tokenizer):
        """Test random model selection."""
        base_config.endpoint.model_selection_strategy = ModelSelectionStrategy.RANDOM
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        # Test random selection returns a valid model
        with patch("random.choice") as mock_choice:
            mock_choice.return_value = "test-model-1"
            result = composer._select_model_name()
            assert result == "test-model-1"
            mock_choice.assert_called_once_with(["test-model-1", "test-model-2"])

    def test_model_selection_invalid_strategy(self, base_config, mock_tokenizer):
        """Test invalid model selection strategy raises error."""
        base_config.endpoint.model_selection_strategy = "INVALID"
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        with pytest.raises(ValueError, match="Invalid model selection strategy"):
            composer._select_model_name()

    def test_get_turn_sequence_lengths_with_distribution(
        self, sequence_dist_config, mock_tokenizer
    ):
        """Test getting sequence lengths with distribution."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)

        turn_id = 12345

        # Mock the sample method
        with patch.object(composer._seq_distribution, "sample") as mock_sample:
            mock_sample.return_value = (150, 75)

            # First call should sample and cache
            result = composer._get_turn_sequence_lengths(turn_id)
            assert result == (150, 75)
            mock_sample.assert_called_once()

            # Second call should return cached value
            result2 = composer._get_turn_sequence_lengths(turn_id)
            assert result2 == (150, 75)
            mock_sample.assert_called_once()  # No additional call

        # Check cache
        assert turn_id in composer._turn_sequence_cache
        assert composer._turn_sequence_cache[turn_id] == (150, 75)

    def test_get_turn_sequence_lengths_without_distribution(
        self, base_config, mock_tokenizer
    ):
        """Test getting sequence lengths without distribution (fallback)."""
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        turn_id = 12345
        result = composer._get_turn_sequence_lengths(turn_id)

        # Should use fallback values from config
        expected = (
            base_config.input.prompt.input_tokens.mean,
            base_config.input.prompt.output_tokens.mean,
        )
        assert result == expected

        # Should be cached
        assert turn_id in composer._turn_sequence_cache
        assert composer._turn_sequence_cache[turn_id] == expected

    def test_clear_turn_cache(self, sequence_dist_config, mock_tokenizer):
        """Test clearing turn cache."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)

        # Add some cache entries
        composer._turn_sequence_cache[123] = (100, 50)
        composer._turn_sequence_cache[456] = (200, 100)

        # Clear one entry
        composer._clear_turn_cache(123)
        assert 123 not in composer._turn_sequence_cache
        assert 456 in composer._turn_sequence_cache

        # Clear non-existent entry (should not error)
        composer._clear_turn_cache(999)

    def test_set_max_tokens_with_distribution(
        self, sequence_dist_config, mock_tokenizer
    ):
        """Test setting max_tokens using sequence distribution."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)
        turn = Turn()

        # Pre-cache sequence lengths for this turn
        turn_id = id(turn)
        composer._turn_sequence_cache[turn_id] = (150, 75)

        composer._set_max_tokens(turn)
        assert turn.max_tokens == 75

    @patch("aiperf.dataset.utils.sample_positive_normal_integer")
    def test_set_max_tokens_without_distribution(
        self, mock_sample, base_config, mock_tokenizer
    ):
        """Test setting max_tokens using legacy behavior."""
        mock_sample.return_value = 45

        composer = ConcreteBaseComposer(base_config, mock_tokenizer)
        turn = Turn()

        composer._set_max_tokens(turn)

        assert turn.max_tokens == 45
        mock_sample.assert_called_once_with(
            base_config.input.prompt.output_tokens.mean,
            base_config.input.prompt.output_tokens.stddev,
        )

    def test_set_max_tokens_without_distribution_none_mean(
        self, base_config, mock_tokenizer
    ):
        """Test setting max_tokens when output_tokens.mean is None."""
        base_config.input.prompt.output_tokens.mean = None

        composer = ConcreteBaseComposer(base_config, mock_tokenizer)
        turn = Turn()

        composer._set_max_tokens(turn)

        # max_tokens should remain None when no distribution and no output_tokens.mean
        assert turn.max_tokens is None

    def test_finalize_turn(self, sequence_dist_config, mock_tokenizer):
        """Test turn finalization."""
        composer = ConcreteBaseComposer(sequence_dist_config, mock_tokenizer)
        turn = Turn()
        turn_id = id(turn)

        # Pre-cache sequence lengths
        composer._turn_sequence_cache[turn_id] = (150, 75)

        composer._finalize_turn(turn)

        # Check model is set
        assert turn.model == "test-model"

        # Check max_tokens is set from cached sequence lengths
        assert turn.max_tokens == 75

        # Check cache is cleared
        assert turn_id not in composer._turn_sequence_cache

    def test_prefix_prompt_enabled_property(self, base_config, mock_tokenizer):
        """Test prefix_prompt_enabled property."""
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        # Should be enabled when length > 0
        assert composer.prefix_prompt_enabled is True

        # Should be disabled when length = 0
        base_config.input.prompt.prefix_prompt.length = 0
        composer2 = ConcreteBaseComposer(base_config, mock_tokenizer)
        assert composer2.prefix_prompt_enabled is False

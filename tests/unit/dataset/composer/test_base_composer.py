# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

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
        assert len(composer._turn_sequence_cache) == 0

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
        # With the global RNG system, we just verify it returns one of the valid models
        result = composer._select_model_name()
        assert result in ["test-model-1", "test-model-2"]

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

    def test_set_max_tokens_without_distribution(self, base_config, mock_tokenizer):
        """Test setting max_tokens using legacy behavior."""
        composer = ConcreteBaseComposer(base_config, mock_tokenizer)
        turn = Turn()

        composer._set_max_tokens(turn)

        # With global RNG seed 42, verify max_tokens is set to a positive integer
        # based on the configured mean (50) and stddev (5)
        assert turn.max_tokens is not None
        assert turn.max_tokens > 0
        assert isinstance(turn.max_tokens, int)
        # Should be roughly around the mean of 50
        assert 30 < turn.max_tokens < 70

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

    def test_inject_context_prompts_with_shared_system_prompt(
        self, base_config, mock_tokenizer
    ):
        """Test _inject_context_prompts with shared system prompt."""
        base_config.input.prompt.prefix_prompt.shared_system_prompt_length = 50
        base_config.input.prompt.prefix_prompt.length = 0
        base_config.input.conversation.num = 3

        # Patch _generate_shared_system_prompt to avoid corpus initialization
        with patch(
            "aiperf.dataset.generator.prompt.PromptGenerator._generate_shared_system_prompt"
        ):
            composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        # Create mock conversations
        from aiperf.common.models import Conversation

        conversations = [
            Conversation(session_id="conv_0"),
            Conversation(session_id="conv_1"),
            Conversation(session_id="conv_2"),
        ]

        # Mock the prompt generator method
        with patch.object(
            composer.prompt_generator,
            "get_shared_system_prompt",
            return_value="shared system prompt text",
        ):
            composer._inject_context_prompts(conversations)

        # All conversations should have the same system message
        assert conversations[0].system_message == "shared system prompt text"
        assert conversations[1].system_message == "shared system prompt text"
        assert conversations[2].system_message == "shared system prompt text"
        # No user context messages
        assert conversations[0].user_context_message is None
        assert conversations[1].user_context_message is None
        assert conversations[2].user_context_message is None

    def test_inject_context_prompts_with_user_context_prompt(
        self, base_config, mock_tokenizer
    ):
        """Test _inject_context_prompts with user context prompts."""
        base_config.input.prompt.prefix_prompt.user_context_prompt_length = 30
        base_config.input.prompt.prefix_prompt.length = 0
        base_config.input.conversation.num = 3

        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        # Create mock conversations
        from aiperf.common.models import Conversation

        conversations = [
            Conversation(session_id="conv_0"),
            Conversation(session_id="conv_1"),
            Conversation(session_id="conv_2"),
        ]

        # Mock the prompt generator method
        def mock_generate_user_context(index):
            return f"user context {index}"

        with patch.object(
            composer.prompt_generator,
            "generate_user_context_prompt",
            side_effect=mock_generate_user_context,
        ):
            composer._inject_context_prompts(conversations)

        # Each conversation should have unique user context
        assert conversations[0].user_context_message == "user context 0"
        assert conversations[1].user_context_message == "user context 1"
        assert conversations[2].user_context_message == "user context 2"
        # No system messages
        assert conversations[0].system_message is None
        assert conversations[1].system_message is None
        assert conversations[2].system_message is None

    def test_inject_context_prompts_with_both_prompts(
        self, base_config, mock_tokenizer
    ):
        """Test _inject_context_prompts with both shared system and user context prompts."""
        base_config.input.prompt.prefix_prompt.shared_system_prompt_length = 50
        base_config.input.prompt.prefix_prompt.user_context_prompt_length = 30
        base_config.input.prompt.prefix_prompt.length = 0
        base_config.input.conversation.num = 2

        # Patch _generate_shared_system_prompt to avoid corpus initialization
        with patch(
            "aiperf.dataset.generator.prompt.PromptGenerator._generate_shared_system_prompt"
        ):
            composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        # Create mock conversations
        from aiperf.common.models import Conversation

        conversations = [
            Conversation(session_id="conv_0"),
            Conversation(session_id="conv_1"),
        ]

        # Mock both prompt generator methods
        def mock_generate_user_context(index):
            return f"user context {index}"

        with (
            patch.object(
                composer.prompt_generator,
                "get_shared_system_prompt",
                return_value="shared system prompt",
            ),
            patch.object(
                composer.prompt_generator,
                "generate_user_context_prompt",
                side_effect=mock_generate_user_context,
            ),
        ):
            composer._inject_context_prompts(conversations)

        # Both conversations should have system message
        assert conversations[0].system_message == "shared system prompt"
        assert conversations[1].system_message == "shared system prompt"
        # Each should have unique user context
        assert conversations[0].user_context_message == "user context 0"
        assert conversations[1].user_context_message == "user context 1"

    def test_inject_context_prompts_with_no_prompts(self, base_config, mock_tokenizer):
        """Test _inject_context_prompts when no context prompts are configured."""
        base_config.input.prompt.prefix_prompt.length = 0
        base_config.input.prompt.prefix_prompt.shared_system_prompt_length = None
        base_config.input.prompt.prefix_prompt.user_context_prompt_length = None
        base_config.input.conversation.num = 2

        composer = ConcreteBaseComposer(base_config, mock_tokenizer)

        # Create mock conversations
        from aiperf.common.models import Conversation

        conversations = [
            Conversation(session_id="conv_0"),
            Conversation(session_id="conv_1"),
        ]

        # Should not call any prompt generator methods
        composer._inject_context_prompts(conversations)

        # No messages should be set
        assert conversations[0].system_message is None
        assert conversations[0].user_context_message is None
        assert conversations[1].system_message is None
        assert conversations[1].user_context_message is None

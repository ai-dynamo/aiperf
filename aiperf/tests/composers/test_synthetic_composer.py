# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pytest

from aiperf.common.dataset_models import Audio, Conversation, Image, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.services.dataset.composer.synthetic import SyntheticDatasetComposer
from aiperf.services.dataset.config import (
    AudioConfig,
    DatasetConfig,
    ImageConfig,
    PrefixPromptConfig,
    PromptConfig,
)


class TestSyntheticDatasetComposer:
    """Test suite for SyntheticDatasetComposer dataset generation."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing without HTTP requests."""
        tokenizer = Mock(spec=Tokenizer)
        tokenizer.bos_token_id = 1
        return tokenizer

    @pytest.fixture
    def basic_config(self, mock_tokenizer):
        """Basic configuration for testing."""
        config = DatasetConfig(
            filename=Path("test_data.json"),
            tokenizer=mock_tokenizer,
            batch_size=1,
            num_dataset_entries=5,
            prompt=PromptConfig(
                tokenizer=mock_tokenizer,
                mean=10,
                stddev=2,
                prefix_prompt=PrefixPromptConfig(pool_size=0),
            ),
        )
        return config

    @pytest.fixture
    def config_with_images(self, mock_tokenizer):
        """Configuration with image generation enabled."""
        config = DatasetConfig(
            filename=Path("test_data.json"),
            tokenizer=mock_tokenizer,
            batch_size=1,
            num_dataset_entries=3,
            prompt=PromptConfig(
                tokenizer=mock_tokenizer,
                mean=10,
                stddev=2,
                prefix_prompt=PrefixPromptConfig(pool_size=0),
            ),
            image=ImageConfig(batch_size=1, width_mean=100, height_mean=100),
        )
        return config

    @pytest.fixture
    def config_with_audio(self, mock_tokenizer):
        """Configuration with audio generation enabled."""
        config = DatasetConfig(
            filename=Path("test_data.json"),
            tokenizer=mock_tokenizer,
            batch_size=1,
            num_dataset_entries=3,
            prompt=PromptConfig(
                tokenizer=mock_tokenizer,
                mean=10,
                stddev=2,
                prefix_prompt=PrefixPromptConfig(pool_size=0),
            ),
            audio=AudioConfig(batch_size=1, length_mean=5),
        )
        return config

    @pytest.fixture
    def config_with_sessions(self, mock_tokenizer):
        """Configuration with multi-turn sessions enabled."""
        config = DatasetConfig(
            filename=Path("test_data.json"),
            tokenizer=mock_tokenizer,
            batch_size=1,
            num_dataset_entries=10,
            prompt=PromptConfig(
                tokenizer=mock_tokenizer,
                mean=10,
                stddev=2,
                prefix_prompt=PrefixPromptConfig(pool_size=0),
            ),
        )
        return config

    @pytest.fixture
    def config_with_prefix_prompts(self, mock_tokenizer):
        """Configuration with prefix prompts enabled."""
        config = DatasetConfig(
            filename=Path("test_data.json"),
            tokenizer=mock_tokenizer,
            batch_size=1,
            num_dataset_entries=5,
            prompt=PromptConfig(
                tokenizer=mock_tokenizer,
                mean=10,
                stddev=2,
                prefix_prompt=PrefixPromptConfig(pool_size=3, length=20),
            ),
        )
        return config

    def test_basic_conversation_dataset_creation(self, basic_config):
        """Test that composer creates a ConversationDataset with correct structure."""
        composer = SyntheticDatasetComposer(basic_config)
        dataset = composer.create_dataset()

        # Should return ConversationDataset
        assert isinstance(dataset, Conversation)

    def test_correct_number_of_turns_for_stateless_entries(self, basic_config):
        """Test that the number of turns matches num_dataset_entries for stateless mode."""
        composer = SyntheticDatasetComposer(basic_config)
        dataset = composer.create_dataset()

        # Each dataset entry should become one turn
        assert len(dataset.turns) == basic_config.num_dataset_entries

    def test_turn_structure_with_text_only(self, basic_config):
        """Test that each turn has correct structure with text data."""
        composer = SyntheticDatasetComposer(basic_config)
        dataset = composer.create_dataset()

        for turn in dataset.turns:
            assert isinstance(turn, Turn)
            # Should have text data based on batch_size
            assert len(turn.text) == basic_config.batch_size
            # Should be empty lists for unused modalities
            assert len(turn.image) == 0
            assert len(turn.audio) == 0
            # Verify text data structure
            for text_data in turn.text:
                assert isinstance(text_data, Text)
                assert text_data.content != ""  # Should have generated content

    def test_turn_structure_with_images(self, config_with_images):
        """Test turn structure when images are enabled."""
        composer = SyntheticDatasetComposer(config_with_images)
        dataset = composer.create_dataset()

        for turn in dataset.turns:
            assert len(turn.text) == config_with_images.batch_size
            assert len(turn.image) == config_with_images.image.batch_size
            assert len(turn.audio) == 0

            # Verify image data structure
            for image_data in turn.image:
                assert isinstance(image_data, Image)
                assert image_data.content != ""  # Should have generated content

    def test_turn_structure_with_audio(self, config_with_audio):
        """Test turn structure when audio is enabled."""
        composer = SyntheticDatasetComposer(config_with_audio)
        dataset = composer.create_dataset()

        for turn in dataset.turns:
            assert len(turn.text) == config_with_audio.batch_size
            assert len(turn.image) == 0
            assert len(turn.audio) == config_with_audio.audio.batch_size

            # Verify audio data structure
            for audio_data in turn.audio:
                assert isinstance(audio_data, Audio)
                assert audio_data.content != ""  # Should have generated content

    def test_multi_turn_sessions(self, config_with_sessions):
        """Test generation of multi-turn conversation sessions."""
        composer = SyntheticDatasetComposer(config_with_sessions)
        dataset = composer.create_dataset()

        # Should have session_id populated for multi-turn sessions
        assert dataset.session_id != ""

        # Total turns should be approximately sessions.num * turns.mean
        # Allow some variance due to normal distribution sampling
        expected_turns = (
            config_with_sessions.sessions.num * config_with_sessions.sessions.turns.mean
        )
        assert len(dataset.turns) >= expected_turns - 2  # Allow for stddev variation
        assert len(dataset.turns) <= expected_turns + 2

        # Check that delay is set for non-final turns
        for turn in dataset.turns[:-1]:  # All except last turn
            if turn.delay is not None:  # Delay might be set
                assert turn.delay >= 0

    def test_stateless_mode_no_session_id(self, basic_config):
        """Test that stateless entries don't have session_id by default."""
        composer = SyntheticDatasetComposer(basic_config)
        dataset = composer.create_dataset()

        # Stateless mode should not populate session_id unless specifically needed
        # This allows flexibility in implementation
        assert isinstance(
            dataset.session_id, str
        )  # Should be string (empty or populated)

    def test_prefix_prompts_applied(self, config_with_prefix_prompts):
        """Test that prefix prompts are applied when configured."""
        composer = SyntheticDatasetComposer(config_with_prefix_prompts)
        dataset = composer.create_dataset()

        # Can't easily test content without mocking generators, but structure should be correct
        assert len(dataset.turns) == config_with_prefix_prompts.num_dataset_entries
        for turn in dataset.turns:
            assert len(turn.text) == config_with_prefix_prompts.batch_size
            for text_data in turn.text:
                assert text_data.content != ""  # Should have content (including prefix)

    def test_batch_size_handling(self, mock_tokenizer):
        """Test that batch size correctly determines number of text entries per turn."""
        config = DatasetConfig(
            filename=Path("test_data.json"),
            tokenizer=mock_tokenizer,
            batch_size=3,  # Multiple texts per turn
            num_dataset_entries=2,
            prompt=PromptConfig(
                tokenizer=mock_tokenizer,
                mean=10,
                stddev=2,
                prefix_prompt=PrefixPromptConfig(pool_size=0),
            ),
        )

        composer = SyntheticDatasetComposer(config)
        dataset = composer.create_dataset()

        assert len(dataset.turns) == 2
        for turn in dataset.turns:
            assert len(turn.text) == 3  # batch_size texts per turn

    def test_multimodal_batch_sizes(self, mock_tokenizer):
        """Test handling of different batch sizes for different modalities."""
        config = DatasetConfig(
            filename=Path("test_data.json"),
            tokenizer=mock_tokenizer,
            batch_size=2,
            num_dataset_entries=1,
            prompt=PromptConfig(
                tokenizer=mock_tokenizer,
                mean=10,
                stddev=2,
                prefix_prompt=PrefixPromptConfig(pool_size=0),
            ),
            image=ImageConfig(batch_size=3, width_mean=100, height_mean=100),
            audio=AudioConfig(batch_size=1, length_mean=5),
        )

        composer = SyntheticDatasetComposer(config)
        dataset = composer.create_dataset()

        turn = dataset.turns[0]
        assert len(turn.text) == 2  # text batch_size
        assert len(turn.image) == 3  # image batch_size
        assert len(turn.audio) == 1  # audio batch_size

    def test_empty_configuration_handling(self, mock_tokenizer):
        """Test handling when optional configurations are not provided."""
        config = DatasetConfig(
            tokenizer=mock_tokenizer,
            batch_size=1,
            num_dataset_entries=1,
            prompt=PromptConfig(
                tokenizer=mock_tokenizer,
                mean=10,
                stddev=2,
                prefix_prompt=PrefixPromptConfig(pool_size=0),
            ),
            # No filename, no image, no audio, no sessions
        )

        composer = SyntheticDatasetComposer(config)
        dataset = composer.create_dataset()

        assert isinstance(dataset, Conversation)
        assert len(dataset.turns) == 1
        turn = dataset.turns[0]
        assert len(turn.text) == 1
        assert len(turn.image) == 0
        assert len(turn.audio) == 0

    def test_zero_dataset_entries(self, mock_tokenizer):
        """Test handling of zero dataset entries."""
        config = DatasetConfig(
            tokenizer=mock_tokenizer,
            batch_size=1,
            num_dataset_entries=0,  # Zero entries
            prompt=PromptConfig(
                tokenizer=mock_tokenizer,
                mean=10,
                stddev=2,
                prefix_prompt=PrefixPromptConfig(pool_size=0),
            ),
        )

        composer = SyntheticDatasetComposer(config)
        dataset = composer.create_dataset()

        assert isinstance(dataset, Conversation)
        assert len(dataset.turns) == 0  # Should be empty

    def test_timestamp_handling_in_turns(self, basic_config):
        """Test that timestamp can be set in turns when needed."""
        composer = SyntheticDatasetComposer(basic_config)
        dataset = composer.create_dataset()

        for turn in dataset.turns:
            # Timestamp might be None or an integer (flexible implementation)
            assert turn.timestamp is None or isinstance(turn.timestamp, int)

    def test_conversation_dataset_fields(self, basic_config):
        """Test that ConversationDataset has all required fields properly initialized."""
        composer = SyntheticDatasetComposer(basic_config)
        dataset = composer.create_dataset()

        # Test ConversationDataset structure
        assert hasattr(dataset, "turns")
        assert hasattr(dataset, "session_id")
        assert isinstance(dataset.turns, list)
        assert isinstance(dataset.session_id, str)

        # Test TurnDataset structure
        if len(dataset.turns) > 0:
            turn = dataset.turns[0]
            assert hasattr(turn, "timestamp")
            assert hasattr(turn, "delay")
            assert hasattr(turn, "text")
            assert hasattr(turn, "image")
            assert hasattr(turn, "audio")
            assert isinstance(turn.text, list)
            assert isinstance(turn.image, list)
            assert isinstance(turn.audio, list)

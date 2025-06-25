# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import patch

import numpy as np
import pytest

from aiperf.common.dataset_models import Audio, Conversation, Image, Text, Turn
from aiperf.services.dataset.composer.synthetic import SyntheticDatasetComposer
from aiperf.services.dataset.config import (
    AudioConfig,
    DatasetConfig,
    ImageConfig,
    PrefixPromptConfig,
    PromptConfig,
    TurnConfig,
    TurnDelayConfig,
)


@pytest.fixture
def basic_config(mock_tokenizer_cls):
    """Basic configuration for testing."""
    config = DatasetConfig(
        tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
        num_conversations=5,
        prompt=PromptConfig(
            mean=10,
            stddev=2,
            prefix_prompt=PrefixPromptConfig(pool_size=0),
        ),
    )
    return config


@pytest.fixture
def config_with_images(mock_tokenizer_cls):
    """Configuration with image generation enabled."""
    config = DatasetConfig(
        tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
        num_conversations=3,
        prompt=PromptConfig(
            mean=10,
            stddev=2,
            prefix_prompt=PrefixPromptConfig(pool_size=0),
        ),
        image=ImageConfig(batch_size=1, width_mean=10, height_mean=10),
    )
    return config


@pytest.fixture
def config_with_audio(mock_tokenizer_cls):
    """Configuration with audio generation enabled."""
    config = DatasetConfig(
        tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
        num_conversations=3,
        prompt=PromptConfig(
            mean=10,
            stddev=2,
            prefix_prompt=PrefixPromptConfig(pool_size=0),
        ),
        audio=AudioConfig(batch_size=1, length_mean=2),
    )
    return config


@pytest.fixture
def config_with_sessions(mock_tokenizer_cls):
    """Configuration with multi-turn sessions enabled."""
    config = DatasetConfig(
        tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
        num_conversations=10,
        prompt=PromptConfig(
            mean=10,
            stddev=2,
            prefix_prompt=PrefixPromptConfig(pool_size=0),
        ),
    )
    return config


@pytest.fixture
def config_with_prefix_prompts(mock_tokenizer_cls):
    """Configuration with prefix prompts enabled."""
    config = DatasetConfig(
        tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
        num_conversations=5,
        prompt=PromptConfig(
            mean=10,
            stddev=2,
            prefix_prompt=PrefixPromptConfig(pool_size=3, length=20),
        ),
    )
    return config


@pytest.fixture
def config_with_multimodal(mock_tokenizer_cls):
    """Configuration with both image and audio generation enabled."""
    config = DatasetConfig(
        tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
        num_conversations=2,
        prompt=PromptConfig(
            mean=10,
            stddev=2,
            batch_size=2,
            prefix_prompt=PrefixPromptConfig(pool_size=2, length=15),
        ),
        image=ImageConfig(batch_size=2, width_mean=10, height_mean=10),
        audio=AudioConfig(batch_size=2, length_mean=2),
    )
    return config


@pytest.fixture
def config_with_turns_and_delays(mock_tokenizer_cls):
    """Configuration with turn and delay settings."""
    config = DatasetConfig(
        tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
        num_conversations=3,
        prompt=PromptConfig(
            mean=10,
            stddev=2,
            prefix_prompt=PrefixPromptConfig(pool_size=0),
        ),
        turn=TurnConfig(mean=2, stddev=0, delay=TurnDelayConfig(mean=1500, stddev=0)),
    )
    return config


class TestSyntheticDatasetComposer:
    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_basic_config(self, basic_config):
        """Test that SyntheticDatasetComposer can be instantiated with basic config."""
        composer = SyntheticDatasetComposer(basic_config)

        assert composer.config == basic_config
        assert composer.config.num_conversations == 5
        assert composer.prompt_generator is not None
        assert composer.include_image is False
        assert composer.include_audio is False

    def test_initialization_with_images(self, config_with_images):
        """Test initialization with image generation enabled."""
        composer = SyntheticDatasetComposer(config_with_images)

        assert composer.config.image.width_mean == 10
        assert composer.config.image.height_mean == 10
        assert composer.include_image is True
        assert composer.include_audio is False

    def test_initialization_with_audio(self, config_with_audio):
        """Test initialization with audio generation enabled."""
        composer = SyntheticDatasetComposer(config_with_audio)

        assert composer.config.audio.length_mean == 2
        assert composer.include_image is False
        assert composer.include_audio is True

    def test_initialization_with_multimodal(self, config_with_multimodal):
        """Test initialization with both image and audio enabled."""
        composer = SyntheticDatasetComposer(config_with_multimodal)

        assert composer.include_image is True
        assert composer.include_audio is True
        assert composer.config.image.batch_size == 2
        assert composer.config.audio.batch_size == 2
        assert composer.config.image.width_mean == 10
        assert composer.config.image.height_mean == 10
        assert composer.config.audio.length_mean == 2

    def test_initialization_with_all_zero_mean(self, mock_tokenizer_cls):
        """Test initialization with no generators enabled."""
        config = DatasetConfig(
            tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
            num_conversations=5,
            prompt=PromptConfig(mean=0, stddev=0),
            image=ImageConfig(width_mean=0, height_mean=0),
            audio=AudioConfig(length_mean=0),
        )

        with pytest.raises(ValueError):
            SyntheticDatasetComposer(config)

    # ============================================================================
    # Create Dataset Method Tests
    # ============================================================================

    @patch(
        "aiperf.services.dataset.composer.synthetic.utils.sample_positive_normal_integer"
    )
    def test_create_dataset_basic(self, mock_sample, basic_config):
        """Test basic dataset creation with text-only conversations."""
        # Mock the number of turns per conversation
        mock_sample.return_value = 2

        composer = SyntheticDatasetComposer(basic_config)
        conversations = composer.create_dataset()

        # Test create_dataset returns correct number of conversations
        assert len(conversations) == 5  # num_conversations

        # Test each conversation has correct structure (session_id, turns)
        for conversation in conversations:
            assert isinstance(conversation, Conversation)
            assert conversation.session_id is not None
            assert len(conversation.turns) == 2  # mocked value

            for turn in conversation.turns:
                assert isinstance(turn, Turn)
                assert len(turn.text) == 1  # single text field per turn
                assert len(turn.text[0].content) == 1  # batch_size = 1
                assert len(turn.image) == 0  # no images
                assert len(turn.audio) == 0  # no audio

    @patch(
        "aiperf.services.dataset.composer.synthetic.utils.sample_positive_normal_integer"
    )
    def test_create_dataset_with_images(self, mock_sample, config_with_images):
        """Test dataset creation with image generation enabled."""
        mock_sample.return_value = 1

        composer = SyntheticDatasetComposer(config_with_images)
        conversations = composer.create_dataset()

        # Test conversations include image payloads
        assert len(conversations) == 3
        for conversation in conversations:
            for turn in conversation.turns:
                assert len(turn.text) == 1  # single text field per turn
                assert len(turn.text[0].content) == 1  # batch_size = 1
                assert len(turn.image) == 1  # single image field per turn
                assert len(turn.image[0].content) == 1  # batch_size = 1
                assert len(turn.audio) == 0  # no audio

                # Check image properties
                image = turn.image[0]
                assert isinstance(image, Image)
                assert image.name == "image_url"

    @patch(
        "aiperf.services.dataset.composer.synthetic.utils.sample_positive_normal_integer"
    )
    def test_create_dataset_with_audio(self, mock_sample, config_with_audio):
        """Test dataset creation with audio generation enabled."""
        mock_sample.return_value = 1

        composer = SyntheticDatasetComposer(config_with_audio)
        conversations = composer.create_dataset()

        # Test conversations include audio payloads
        assert len(conversations) == 3
        for conversation in conversations:
            for turn in conversation.turns:
                assert len(turn.text) == 1  # single text field per turn
                assert len(turn.text[0].content) == 1  # batch_size = 1
                assert len(turn.image) == 0  # no images
                assert len(turn.audio) == 1  # single audio field per turn
                assert len(turn.audio[0].content) == 1  # batch_size = 1

                # Check audio properties
                audio = turn.audio[0]
                assert isinstance(audio, Audio)

    @patch(
        "aiperf.services.dataset.composer.synthetic.utils.sample_positive_normal_integer"
    )
    def test_create_dataset_multimodal(self, mock_sample, config_with_multimodal):
        """Test dataset creation with both image and audio enabled."""
        mock_sample.return_value = 1

        composer = SyntheticDatasetComposer(config_with_multimodal)
        conversations = composer.create_dataset()

        # Test conversations include both image and audio payloads
        assert len(conversations) == config_with_multimodal.num_conversations
        for conversation in conversations:
            for turn in conversation.turns:
                # Test correct batch sizes for all modalities
                assert len(turn.text) == 1  # single text field per turn
                assert len(turn.text[0].content) == 2  # batch_size = 2
                assert len(turn.image) == 1  # single image field per turn
                assert len(turn.image[0].content) == 2  # batch_size = 2
                assert len(turn.audio) == 1  # single audio field per turn
                assert len(turn.audio[0].content) == 2  # batch_size = 2

    @patch(
        "aiperf.services.dataset.composer.synthetic.utils.sample_positive_normal_integer"
    )
    @patch(
        "aiperf.services.dataset.generator.prompt.PromptGenerator.get_random_prefix_prompt"
    )
    def test_create_dataset_with_prefix_prompts(
        self, mock_prefix, mock_sample, config_with_prefix_prompts
    ):
        """Test dataset creation with prefix prompts enabled."""
        mock_sample.return_value = 2  # 2 turns per conversation
        mock_prefix.return_value = "Prefix prompt:"

        composer = SyntheticDatasetComposer(config_with_prefix_prompts)
        conversations = composer.create_dataset()

        assert len(conversations) == 5
        for conversation in conversations:
            # Test first turns include prefix prompts
            first_turn = conversation.turns[0]
            first_text_content = first_turn.text[0].content[0]
            assert "Prefix prompt:" in first_text_content

            # Test subsequent turns don't include prefix prompts (if they exist)
            if len(conversation.turns) > 1:
                subsequent_turn = conversation.turns[1]
                subsequent_text_content = subsequent_turn.text[0].content[0]
                assert "Prefix prompt:" not in subsequent_text_content

    def test_create_dataset_multiple_turns(self, config_with_turns_and_delays):
        """Test dataset creation with multiple turns and delays."""
        composer = SyntheticDatasetComposer(config_with_turns_and_delays)
        conversations = composer.create_dataset()

        # Test conversations have multiple turns
        assert len(conversations) == 3

        for conversation in conversations:
            assert len(conversation.turns) == 2
            assert conversation.turns[0].delay is None  # first turn has no delay
            assert conversation.turns[1].delay == 1500  # subsequent turns have delays

    # ============================================================================
    # Create Turn Method Tests
    # ============================================================================

    def test_create_first_turn(self, basic_config):
        """Test _create_turn method for first turn in conversation."""
        basic_config.turn.delay.mean = 1500
        basic_config.turn.delay.stddev = 0

        composer = SyntheticDatasetComposer(basic_config)

        # Test first turn creation
        turn = composer._create_turn(is_first=True)

        assert isinstance(turn, Turn)
        assert len(turn.text) == 1  # single text field per turn
        assert len(turn.image) == 0  # no images
        assert len(turn.audio) == 0  # no audio
        assert turn.delay is None  # first turn has no delay

    def test_create_turn_subsequent_turn(self, config_with_turns_and_delays):
        """Test _create_turn method for subsequent turns in conversation."""
        composer = SyntheticDatasetComposer(config_with_turns_and_delays)

        # Test subsequent turn creation
        turn = composer._create_turn(is_first=False)

        assert isinstance(turn, Turn)
        assert len(turn.text) == 1
        # Test subsequent turns have delays
        assert turn.delay == 1500

    def test_create_turn_with_all_modalities(self, config_with_multimodal):
        """Test _create_turn method with text, image, and audio."""
        config_with_multimodal.turn.delay.mean = 1500
        config_with_multimodal.turn.delay.stddev = 0

        composer = SyntheticDatasetComposer(config_with_multimodal)

        turn = composer._create_turn(is_first=True)

        assert isinstance(turn, Turn)
        assert len(turn.text) == 1  # single text field per turn
        assert len(turn.text[0].content) == 2  # batch_size = 2
        assert len(turn.image) == 1  # single image field per turn
        assert len(turn.image[0].content) == 2  # batch_size = 2
        assert len(turn.audio) == 1  # single audio field per turn
        assert len(turn.audio[0].content) == 2  # batch_size = 2
        assert turn.delay is None  # first turn has no delay

        # Test subsequent turn creation
        turn = composer._create_turn(is_first=False)

        assert isinstance(turn, Turn)
        assert turn.delay == 1500

    # ============================================================================
    # Generate Payload Methods Tests
    # ============================================================================

    @patch("aiperf.services.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_basic(self, mock_generate, basic_config):
        """Test _generate_text_payloads method with basic configuration."""
        mock_generate.return_value = "Generated text content"

        composer = SyntheticDatasetComposer(basic_config)
        turn = Turn()

        # Test text payload generation
        composer._generate_text_payloads(turn, is_first=True)

        # Test correct number of text payloads based on batch_size
        assert len(turn.text) == 1  # batch_size = 1

        # Test text content is generated using prompt generator
        text_payload = turn.text[0]
        assert isinstance(text_payload, Text)
        assert text_payload.name == "text"
        assert text_payload.content == ["Generated text content"]

    @patch(
        "aiperf.services.dataset.generator.prompt.PromptGenerator.get_random_prefix_prompt"
    )
    @patch("aiperf.services.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_first_turn_with_prefix(
        self, mock_generate, mock_prefix, config_with_prefix_prompts
    ):
        """Test _generate_text_payloads for first turn with prefix prompts."""
        mock_generate.return_value = "User message"
        mock_prefix.return_value = "Prefix prompt:"

        composer = SyntheticDatasetComposer(config_with_prefix_prompts)
        turn = Turn()

        # Test prefix prompt is added to first turn
        composer._generate_text_payloads(turn, is_first=True)

        text_payload = turn.text[0]
        # Test prefix prompt format ("prefix prompt")
        assert text_payload.content == ["Prefix prompt: User message"]

    @patch("aiperf.services.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_subsequent_turn_no_prefix(
        self, mock_generate, config_with_prefix_prompts
    ):
        """Test _generate_text_payloads for subsequent turns without prefix prompts."""
        mock_generate.return_value = "User message"

        composer = SyntheticDatasetComposer(config_with_prefix_prompts)
        turn = Turn()

        # Test no prefix prompt is added to subsequent turns
        composer._generate_text_payloads(turn, is_first=False)

        text_payload = turn.text[0]
        assert text_payload.content == ["User message"]  # No prefix

    @patch("aiperf.services.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_multiple_batch_size(
        self, mock_generate, basic_config
    ):
        """Test _generate_text_payloads with batch_size > 1."""
        mock_generate.return_value = "Generated text"
        basic_config.prompt.batch_size = 3

        composer = SyntheticDatasetComposer(basic_config)
        turn = Turn()

        # Test multiple text payloads are generated per turn
        composer._generate_text_payloads(turn, is_first=True)

        assert len(turn.text) == 1  # single text field per turn
        assert len(turn.text[0].content) == 3  # batch_size = 3

        # Batched text payloads
        text_payload = turn.text[0]
        assert text_payload.content == [
            "Generated text",
            "Generated text",
            "Generated text",
        ]

    @patch("aiperf.services.dataset.generator.image.ImageGenerator.generate")
    def test_generate_image_payloads(self, mock_generate, config_with_images):
        """Test _generate_image_payloads method."""
        mock_generate.return_value = "fake_image_data"

        composer = SyntheticDatasetComposer(config_with_images)
        turn = Turn()

        # Test image payload generation
        composer._generate_image_payloads(turn)

        # Test correct number of image payloads based on batch_size
        assert len(turn.image) == 1  # batch_size = 1

        # Test image content is generated using image generator
        image_payload = turn.image[0]
        assert isinstance(image_payload, Image)
        assert image_payload.name == "image_url"
        assert image_payload.content == ["fake_image_data"]

    @patch("aiperf.services.dataset.generator.audio.AudioGenerator.generate")
    def test_generate_audio_payloads(self, mock_generate, config_with_audio):
        """Test _generate_audio_payloads method."""
        mock_generate.return_value = "fake_audio_data"

        composer = SyntheticDatasetComposer(config_with_audio)
        turn = Turn()

        # Test audio payload generation
        composer._generate_audio_payloads(turn)

        # Test correct number of audio payloads based on batch_size
        assert len(turn.audio) == 1  # batch_size = 1

        audio_payload = turn.audio[0]
        assert audio_payload.name == "input_audio"
        assert audio_payload.content == ["fake_audio_data"]

    # ============================================================================
    # Configuration Variations Tests
    # ============================================================================

    def test_zero_conversations(self, basic_config):
        """Test behavior with zero conversations requested."""
        basic_config.num_conversations = 0

        composer = SyntheticDatasetComposer(basic_config)
        conversations = composer.create_dataset()

        assert len(conversations) == 0

    @patch(
        "aiperf.services.dataset.composer.synthetic.utils.sample_positive_normal_integer"
    )
    def test_edge_case_statistical_parameters(self, mock_sample, mock_tokenizer_cls):
        """Test behavior with edge case statistical parameters."""
        mock_sample.return_value = 1

        config = DatasetConfig(
            tokenizer=mock_tokenizer_cls.from_pretrained("gpt2"),
            num_conversations=2,
            prompt=PromptConfig(
                mean=1,  # Very small mean
                stddev=0,  # Zero stddev
                prefix_prompt=PrefixPromptConfig(pool_size=0),
            ),
            turn=TurnConfig(
                mean=100,  # Large mean
                stddev=50,  # Large stddev
            ),
        )

        composer = SyntheticDatasetComposer(config)
        conversations = composer.create_dataset()

        # Test with very small/large mean and stddev values
        assert len(conversations) == 2
        assert all(len(conv.turns) == 1 for conv in conversations)  # mocked return

    @pytest.mark.parametrize("num_conversations", [1, 5, 10, 50])
    def test_different_conversation_counts(self, basic_config, num_conversations):
        """Test dataset creation with different conversation counts."""
        basic_config.num_conversations = num_conversations

        composer = SyntheticDatasetComposer(basic_config)
        conversations = composer.create_dataset()

        # Parametrized test for different num_conversations values
        assert len(conversations) == num_conversations

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    @patch(
        "aiperf.services.dataset.composer.synthetic.utils.sample_positive_normal_integer"
    )
    def test_different_batch_sizes(self, mock_sample, basic_config, batch_size):
        """Test dataset creation with different batch sizes."""
        mock_sample.return_value = 1

        basic_config.prompt.batch_size = batch_size

        composer = SyntheticDatasetComposer(basic_config)
        conversations = composer.create_dataset()

        # Parametrized test for different batch_size values
        turn = conversations[0].turns[0]
        assert len(turn.text) == 1  # single text field per turn

        text_payload = turn.text[0]
        assert len(text_payload.content) == batch_size

    # ============================================================================
    # Miscellaneous Tests
    # ============================================================================

    def test_missing_required_generators(self, basic_config):
        """Test behavior when required generators are missing."""
        composer = SyntheticDatasetComposer(basic_config)

        # Test error handling when generators are not properly initialized
        # Simulate missing tokenizer in generator
        composer.prompt_generator = None

        with pytest.raises(AttributeError):
            composer.create_dataset()

    def test_reproducibility_with_fixed_seed(self, config_with_multimodal):
        """Test that dataset generation is reproducible with fixed random seed."""
        config_with_multimodal.prompt.stddev = 2
        config_with_multimodal.image.width_stddev = 2
        config_with_multimodal.image.height_stddev = 2
        config_with_multimodal.audio.length_stddev = 2
        config_with_multimodal.turn = TurnConfig(
            mean=2, stddev=2, delay=TurnDelayConfig(mean=1500, stddev=2)
        )

        random.seed(42)
        np.random.seed(42)
        composer1 = SyntheticDatasetComposer(config_with_multimodal)
        conversations1 = composer1.create_dataset()

        random.seed(42)
        np.random.seed(42)
        composer2 = SyntheticDatasetComposer(config_with_multimodal)
        conversations2 = composer2.create_dataset()

        # Basic structure should be the same
        assert len(conversations1) == len(conversations2)
        assert len(conversations1[0].turns) == len(conversations2[0].turns)

        # Both should have generated the same number of conversations and turns
        for conv1, conv2 in zip(conversations1, conversations2, strict=True):
            assert len(conv1.turns) == len(conv2.turns)
            for turn1, turn2 in zip(conv1.turns, conv2.turns, strict=True):
                assert len(turn1.text) == len(turn2.text)
                assert len(turn1.image) == len(turn2.image)
                assert len(turn1.audio) == len(turn2.audio)
                assert turn1.text[0].content == turn2.text[0].content
                assert turn1.image[0].content == turn2.image[0].content
                assert turn1.audio[0].content == turn2.audio[0].content
                assert turn1.delay == turn2.delay

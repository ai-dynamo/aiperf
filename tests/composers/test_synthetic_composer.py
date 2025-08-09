# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import patch

import numpy as np
import pytest

from aiperf.common.config import (
    AudioConfig,
    AudioLengthConfig,
    ConversationConfig,
    ImageConfig,
    ImageHeightConfig,
    ImageWidthConfig,
    InputConfig,
    InputTokensConfig,
    PrefixPromptConfig,
    PromptConfig,
    TurnConfig,
    TurnDelayConfig,
)
from aiperf.common.config.user_config import EndpointConfig, UserConfig
from aiperf.common.models import Audio, Conversation, Image, Text, Turn
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer


class TestSyntheticDatasetComposer:
    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_basic_config(self, synthetic_config, mock_tokenizer):
        """Test that SyntheticDatasetComposer can be instantiated with basic config."""
        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)

        assert composer.config == synthetic_config
        assert composer.config.input.conversation.num == 5
        assert composer.prompt_generator is not None
        assert composer.include_image is False
        assert composer.include_audio is False

    def test_initialization_with_images(self, image_config, mock_tokenizer):
        """Test initialization with image generation enabled."""
        composer = SyntheticDatasetComposer(image_config, mock_tokenizer)

        assert composer.config.input.image.width.mean == 10
        assert composer.config.input.image.height.mean == 10
        assert composer.include_image is True
        assert composer.include_audio is False

    def test_initialization_with_audio(self, audio_config, mock_tokenizer):
        """Test initialization with audio generation enabled."""
        composer = SyntheticDatasetComposer(audio_config, mock_tokenizer)

        assert composer.config.input.audio.length.mean == 2
        assert composer.include_image is False
        assert composer.include_audio is True

    def test_initialization_with_multimodal(self, multimodal_config, mock_tokenizer):
        """Test initialization with both image and audio enabled."""
        composer = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)

        assert composer.include_image is True
        assert composer.include_audio is True
        input_config = composer.config.input
        assert input_config.image.batch_size == 2
        assert input_config.audio.batch_size == 2
        assert input_config.image.width.mean == 10
        assert input_config.image.height.mean == 10
        assert input_config.audio.length.mean == 2

    def test_initialization_with_all_zero_mean(self, mock_tokenizer):
        """Test initialization with no generators enabled."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test_model"],
            ),
            input=InputConfig(
                conversation=ConversationConfig(num=5),
                prompt=PromptConfig(input_tokens=InputTokensConfig(mean=0)),
                image=ImageConfig(
                    width=ImageWidthConfig(mean=0), height=ImageHeightConfig(mean=0)
                ),
                audio=AudioConfig(length=AudioLengthConfig(mean=0)),
            ),
        )

        with pytest.raises(ValueError):
            SyntheticDatasetComposer(config, mock_tokenizer)

    # ============================================================================
    # Create Dataset Method Tests
    # ============================================================================

    @patch("aiperf.dataset.composer.synthetic.utils.sample_positive_normal_integer")
    def test_create_dataset_basic(self, mock_sample, synthetic_config, mock_tokenizer):
        """Test basic dataset creation with text-only conversations."""
        # Mock the number of turns per conversation
        mock_sample.return_value = 2

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
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
                assert len(turn.texts) == 1  # single text field per turn
                assert len(turn.texts[0].contents) == 1  # batch_size = 1
                assert len(turn.images) == 0  # no images
                assert len(turn.audios) == 0  # no audio

    @patch("aiperf.dataset.composer.synthetic.utils.sample_positive_normal_integer")
    def test_create_dataset_with_images(
        self, mock_sample, image_config, mock_tokenizer
    ):
        """Test dataset creation with image generation enabled."""
        mock_sample.return_value = 1

        composer = SyntheticDatasetComposer(image_config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Test conversations include image payloads
        assert len(conversations) == 3
        for conversation in conversations:
            for turn in conversation.turns:
                assert len(turn.texts) == 1  # single text field per turn
                assert len(turn.texts[0].contents) == 1  # batch_size = 1
                assert len(turn.images) == 1  # single image field per turn
                assert len(turn.images[0].contents) == 1  # batch_size = 1
                assert len(turn.audios) == 0  # no audio

                # Check image properties
                image = turn.images[0]
                assert isinstance(image, Image)
                assert image.name == "image_url"

    @patch("aiperf.dataset.composer.synthetic.utils.sample_positive_normal_integer")
    def test_create_dataset_with_audio(self, mock_sample, audio_config, mock_tokenizer):
        """Test dataset creation with audio generation enabled."""
        mock_sample.return_value = 1

        composer = SyntheticDatasetComposer(audio_config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Test conversations include audio payloads
        assert len(conversations) == 3
        for conversation in conversations:
            for turn in conversation.turns:
                assert len(turn.texts) == 1  # single text field per turn
                assert len(turn.texts[0].contents) == 1  # batch_size = 1
                assert len(turn.images) == 0  # no images
                assert len(turn.audios) == 1  # single audio field per turn
                assert len(turn.audios[0].contents) == 1  # batch_size = 1

                # Check audio properties
                audio = turn.audios[0]
                assert isinstance(audio, Audio)

    @patch("aiperf.dataset.composer.synthetic.utils.sample_positive_normal_integer")
    def test_create_dataset_multimodal(
        self, mock_sample, multimodal_config, mock_tokenizer
    ):
        """Test dataset creation with both image and audio enabled."""
        mock_sample.return_value = 1

        composer = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Test conversations include both image and audio payloads
        assert len(conversations) == multimodal_config.input.conversation.num
        for conversation in conversations:
            for turn in conversation.turns:
                # Test correct batch sizes for all modalities
                assert len(turn.texts) == 1  # single text field per turn
                assert len(turn.texts[0].contents) == 2  # batch_size = 2
                assert len(turn.images) == 1  # single image field per turn
                assert len(turn.images[0].contents) == 2  # batch_size = 2
                assert len(turn.audios) == 1  # single audio field per turn
                assert len(turn.audios[0].contents) == 2  # batch_size = 2

    @patch("aiperf.dataset.composer.synthetic.utils.sample_positive_normal_integer")
    @patch("aiperf.dataset.generator.prompt.PromptGenerator.get_random_prefix_prompt")
    def test_create_dataset_with_prefix_prompts(
        self, mock_prefix, mock_sample, prefix_prompt_config, mock_tokenizer
    ):
        """Test dataset creation with prefix prompts enabled."""
        mock_sample.return_value = 2  # 2 turns per conversation
        mock_prefix.return_value = "Prefix prompt:"

        composer = SyntheticDatasetComposer(prefix_prompt_config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 5
        for conversation in conversations:
            # Test first turns include prefix prompts
            first_turn = conversation.turns[0]
            first_text_content = first_turn.texts[0].contents[0]
            assert "Prefix prompt:" in first_text_content

            # Test subsequent turns don't include prefix prompts (if they exist)
            if len(conversation.turns) > 1:
                subsequent_turn = conversation.turns[1]
                subsequent_text_content = subsequent_turn.texts[0].contents[0]
                assert "Prefix prompt:" not in subsequent_text_content

    def test_create_dataset_multiple_turns(self, multiturn_config, mock_tokenizer):
        """Test dataset creation with multiple turns and delays."""
        composer = SyntheticDatasetComposer(multiturn_config, mock_tokenizer)
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

    def test_create_first_turn(self, synthetic_config, mock_tokenizer):
        """Test _create_turn method for first turn in conversation."""
        synthetic_config.input.conversation.turn.delay.mean = 1500
        synthetic_config.input.conversation.turn.delay.stddev = 0

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)

        # Test first turn creation
        turn = composer._create_turn(is_first=True)

        assert isinstance(turn, Turn)
        assert len(turn.texts) == 1  # single text field per turn
        assert len(turn.images) == 0  # no images
        assert len(turn.audios) == 0  # no audio
        assert turn.delay is None  # first turn has no delay

    def test_create_turn_subsequent_turn(self, multiturn_config, mock_tokenizer):
        """Test _create_turn method for subsequent turns in conversation."""
        composer = SyntheticDatasetComposer(multiturn_config, mock_tokenizer)

        # Test subsequent turn creation
        turn = composer._create_turn(is_first=False)

        assert isinstance(turn, Turn)
        assert len(turn.texts) == 1
        # Test subsequent turns have delays
        assert turn.delay == 1500

    def test_create_turn_with_all_modalities(self, multimodal_config, mock_tokenizer):
        """Test _create_turn method with text, image, and audio."""
        multimodal_config.input.conversation.turn.delay.mean = 1500
        multimodal_config.input.conversation.turn.delay.stddev = 0

        composer = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)

        turn = composer._create_turn(is_first=True)

        assert isinstance(turn, Turn)
        assert len(turn.texts) == 1  # single text field per turn
        assert len(turn.texts[0].contents) == 2  # batch_size = 2
        assert len(turn.images) == 1  # single image field per turn
        assert len(turn.images[0].contents) == 2  # batch_size = 2
        assert len(turn.audios) == 1  # single audio field per turn
        assert len(turn.audios[0].contents) == 2  # batch_size = 2
        assert turn.delay is None  # first turn has no delay

        # Test subsequent turn creation
        turn = composer._create_turn(is_first=False)

        assert isinstance(turn, Turn)
        assert turn.delay == 1500

    # ============================================================================
    # Generate Payload Methods Tests
    # ============================================================================

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_basic(
        self, mock_generate, synthetic_config, mock_tokenizer
    ):
        """Test _generate_text_payloads method with basic configuration."""
        mock_generate.return_value = "Generated text content"

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)

        # Test text payload generation
        turn = Turn()
        text = composer._generate_text_payloads(is_first=True)
        turn.texts.append(text)

        # Test correct number of text payloads based on batch_size
        assert len(turn.texts) == 1  # batch_size = 1

        # Test text content is generated using prompt generator
        text_payload = turn.texts[0]
        assert isinstance(text_payload, Text)
        assert text_payload.name == "text"
        assert text_payload.contents == ["Generated text content"]

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.get_random_prefix_prompt")
    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_first_turn_with_prefix(
        self, mock_generate, mock_prefix, prefix_prompt_config, mock_tokenizer
    ):
        """Test _generate_text_payloads for first turn with prefix prompts."""
        mock_generate.return_value = "User message"
        mock_prefix.return_value = "Prefix prompt:"

        composer = SyntheticDatasetComposer(prefix_prompt_config, mock_tokenizer)

        # Test prefix prompt is added to first turn
        turn = Turn()
        text = composer._generate_text_payloads(is_first=True)
        turn.texts.append(text)

        text_payload = turn.texts[0]
        # Test prefix prompt format ("prefix prompt")
        assert text_payload.contents == ["Prefix prompt: User message"]

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_subsequent_turn_no_prefix(
        self, mock_generate, prefix_prompt_config, mock_tokenizer
    ):
        """Test _generate_text_payloads for subsequent turns without prefix prompts."""
        mock_generate.return_value = "User message"

        composer = SyntheticDatasetComposer(prefix_prompt_config, mock_tokenizer)

        # Test no prefix prompt is added to subsequent turns
        turn = Turn()
        text = composer._generate_text_payloads(is_first=False)
        turn.texts.append(text)

        text_payload = turn.texts[0]
        assert text_payload.contents == ["User message"]  # No prefix

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_multiple_batch_size(
        self, mock_generate, synthetic_config, mock_tokenizer
    ):
        """Test _generate_text_payloads with batch_size > 1."""
        mock_generate.return_value = "Generated text"
        synthetic_config.input.prompt.batch_size = 3

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)

        # Test multiple text payloads are generated per turn
        turn = Turn()
        text = composer._generate_text_payloads(is_first=True)
        turn.texts.append(text)

        assert len(turn.texts) == 1  # single text field per turn
        assert len(turn.texts[0].contents) == 3  # batch_size = 3

        # Batched text payloads
        text_payload = turn.texts[0]
        assert text_payload.contents == [
            "Generated text",
            "Generated text",
            "Generated text",
        ]

    @patch("aiperf.dataset.generator.image.ImageGenerator.generate")
    def test_generate_image_payloads(self, mock_generate, image_config, mock_tokenizer):
        """Test _generate_image_payloads method."""
        mock_generate.return_value = "fake_image_data"

        composer = SyntheticDatasetComposer(image_config, mock_tokenizer)

        # Test image payload generation
        turn = Turn()
        image = composer._generate_image_payloads()
        turn.images.append(image)

        # Test correct number of image payloads based on batch_size
        assert len(turn.images) == 1  # batch_size = 1

        # Test image content is generated using image generator
        image_payload = turn.images[0]
        assert isinstance(image_payload, Image)
        assert image_payload.name == "image_url"
        assert image_payload.contents == ["fake_image_data"]

    @patch("aiperf.dataset.generator.audio.AudioGenerator.generate")
    def test_generate_audio_payloads(self, mock_generate, audio_config, mock_tokenizer):
        """Test _generate_audio_payloads method."""
        mock_generate.return_value = "fake_audio_data"

        composer = SyntheticDatasetComposer(audio_config, mock_tokenizer)

        # Test audio payload generation
        turn = Turn()
        audio = composer._generate_audio_payloads()
        turn.audios.append(audio)

        # Test correct number of audio payloads based on batch_size
        assert len(turn.audios) == 1  # batch_size = 1

        audio_payload = turn.audios[0]
        assert audio_payload.name == "input_audio"
        assert audio_payload.contents == ["fake_audio_data"]

    # ============================================================================
    # Configuration Variations Tests
    # ============================================================================

    def test_zero_conversations(self, synthetic_config, mock_tokenizer):
        """Test behavior with zero conversations requested."""
        synthetic_config.input.conversation.num = 0

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 0

    @patch("aiperf.dataset.composer.synthetic.utils.sample_positive_normal_integer")
    def test_edge_case_statistical_parameters(self, mock_sample, mock_tokenizer):
        """Test behavior with edge case statistical parameters."""
        mock_sample.return_value = 1

        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
            ),
            input=InputConfig(
                conversation=ConversationConfig(num=2),
                prompt=PromptConfig(
                    mean=1,  # Very small mean
                    stddev=0,  # Zero stddev
                    prefix_prompt=PrefixPromptConfig(pool_size=0),
                ),
                turn=TurnConfig(
                    mean=100,  # Large mean
                    stddev=50,  # Large stddev
                ),
            ),
        )

        composer = SyntheticDatasetComposer(config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Test with very small/large mean and stddev values
        assert len(conversations) == 2
        assert all(len(conv.turns) == 1 for conv in conversations)  # mocked return

    @pytest.mark.parametrize("num_conversations", [1, 5, 10, 50])
    def test_different_conversation_counts(
        self, synthetic_config, num_conversations, mock_tokenizer
    ):
        """Test dataset creation with different conversation counts."""
        synthetic_config.input.conversation.num = num_conversations

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Parametrized test for different num_conversations values
        assert len(conversations) == num_conversations

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    @patch("aiperf.dataset.composer.synthetic.utils.sample_positive_normal_integer")
    def test_different_batch_sizes(
        self, mock_sample, synthetic_config, batch_size, mock_tokenizer
    ):
        """Test dataset creation with different batch sizes."""
        mock_sample.return_value = 1

        synthetic_config.input.prompt.batch_size = batch_size

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Parametrized test for different batch_size values
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 1  # single text field per turn

        text_payload = turn.texts[0]
        assert len(text_payload.contents) == batch_size

    # ============================================================================
    # Miscellaneous Tests
    # ============================================================================

    def test_missing_required_generators(self, synthetic_config, mock_tokenizer):
        """Test behavior when required generators are missing."""
        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)

        # Test error handling when generators are not properly initialized
        # Simulate missing tokenizer in generator
        composer.prompt_generator = None

        with pytest.raises(AttributeError):
            composer.create_dataset()

    def test_reproducibility_with_fixed_seed(self, multimodal_config, mock_tokenizer):
        """Test that dataset generation is reproducible with fixed random seed."""
        multimodal_config.input.prompt.input_tokens.stddev = 2
        multimodal_config.input.image.width.stddev = 2
        multimodal_config.input.image.height.stddev = 2
        multimodal_config.input.audio.length.stddev = 2
        multimodal_config.input.conversation.turn = TurnConfig(
            mean=2, stddev=2, delay=TurnDelayConfig(mean=1500, stddev=2)
        )

        random.seed(42)
        np.random.seed(42)
        composer1 = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)
        conversations1 = composer1.create_dataset()

        random.seed(42)
        np.random.seed(42)
        composer2 = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)
        conversations2 = composer2.create_dataset()

        # Basic structure should be the same
        assert len(conversations1) == len(conversations2)
        assert len(conversations1[0].turns) == len(conversations2[0].turns)

        # Both should have generated the same number of conversations and turns
        for conv1, conv2 in zip(conversations1, conversations2, strict=True):
            assert len(conv1.turns) == len(conv2.turns)
            for turn1, turn2 in zip(conv1.turns, conv2.turns, strict=True):
                assert len(turn1.texts) == len(turn2.texts)
                assert len(turn1.images) == len(turn2.images)
                assert len(turn1.audios) == len(turn2.audios)
                assert turn1.texts[0].contents == turn2.texts[0].contents
                assert turn1.images[0].contents == turn2.images[0].contents
                assert turn1.audios[0].contents == turn2.audios[0].contents
                assert turn1.delay == turn2.delay

    # ============================================================================
    # Model Selection Strategy Tests
    # ============================================================================

    @patch("random.choice", return_value="test-model-1")
    def test_model_selection_random(self, mock_choice, custom_config, mock_tokenizer):
        """Test random model selection strategy."""

        custom_config.endpoint.model_selection_strategy = "random"
        composer = SyntheticDatasetComposer(custom_config, mock_tokenizer)

        conversations = composer.create_dataset()

        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.model == "test-model-1"

    def test_model_selection_round_robin(self, custom_config, mock_tokenizer):
        custom_config.endpoint.model_selection_strategy = "round_robin"
        custom_config.endpoint.model_names = ["test-model-1", "test-model-2"]

        composer = SyntheticDatasetComposer(custom_config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Check that models are selected in round-robin fashion
        for i, conversation in enumerate(conversations):
            for j, turn in enumerate(conversation.turns):
                expected_model = "test-model-1" if (i + j) % 2 == 0 else "test-model-2"
                assert turn.model == expected_model

    # ============================================================================
    # Max Token Tests
    # ============================================================================

    def test_max_tokens_integration_with_mean(self, custom_config, mock_tokenizer):
        custom_config.input.prompt.output_tokens.mean = 100
        custom_config.input.prompt.output_tokens.stddev = 5.0

        composer = SyntheticDatasetComposer(custom_config, mock_tokenizer)

        with patch(
            "aiperf.dataset.utils.sample_positive_normal_integer", return_value=98
        ):
            conversations = composer.create_dataset()

        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens == 98

    def test_max_tokens_not_set_when_mean_none(self, custom_config, mock_tokenizer):
        custom_config.input.prompt.output_tokens.mean = None
        custom_config.input.prompt.output_tokens.stddev = None

        composer = SyntheticDatasetComposer(custom_config, mock_tokenizer)
        conversations = composer.create_dataset()

        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens is None

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
from aiperf.common.dataset_models import Audio, Conversation, Image, Text, Turn
from aiperf.services.dataset.composer.synthetic import SyntheticDatasetComposer


class TestSyntheticDatasetComposer:
    # ============================================================================
    # Initialization Tests
    # ============================================================================

    async def test_pre_initialization_basic(self, synthetic_config, mock_tokenizer):
        """Test SyntheticDatasetComposer initialization with basic configuration."""
        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)

        assert composer.config == synthetic_config
        assert composer.config.conversation.num == 5
        assert composer.include_image is False
        assert composer.include_audio is False

    async def test_pre_initialization_with_images(self, image_config, mock_tokenizer):
        """Test initialization with image generation enabled."""
        composer = SyntheticDatasetComposer(image_config, mock_tokenizer)

        assert composer.config.image.width.mean == 10
        assert composer.config.image.height.mean == 10
        assert composer.include_image is True
        assert composer.include_audio is False

    async def test_pre_initialization_with_audio(self, audio_config, mock_tokenizer):
        """Test initialization with audio generation enabled."""
        composer = SyntheticDatasetComposer(audio_config, mock_tokenizer)

        assert composer.config.audio.length.mean == 2
        assert composer.include_image is False
        assert composer.include_audio is True

    async def test_pre_initialization_with_multimodal(
        self, multimodal_config, mock_tokenizer
    ):
        """Test initialization with both image and audio enabled."""
        composer = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)

        assert composer.include_image is True
        assert composer.include_audio is True
        assert composer.config.image.batch_size == 2
        assert composer.config.audio.batch_size == 2
        assert composer.config.image.width.mean == 10
        assert composer.config.image.height.mean == 10
        assert composer.config.audio.length.mean == 2

    async def test_pre_initialization_with_all_zero_mean(self, mock_tokenizer):
        """Test initialization with no generators enabled."""
        config = InputConfig(
            conversation=ConversationConfig(num=5),
            prompt=PromptConfig(input_tokens=InputTokensConfig(mean=0)),
            image=ImageConfig(
                width=ImageWidthConfig(mean=0), height=ImageHeightConfig(mean=0)
            ),
            audio=AudioConfig(length=AudioLengthConfig(mean=0)),
        )

        with pytest.raises(ValueError):
            SyntheticDatasetComposer(config, mock_tokenizer)

    async def test_initialization(self, synthetic_config, mock_tokenizer):
        """Test SyntheticDatasetComposer initialization with basic configuration."""
        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        assert composer.prompt_generator is None
        assert composer.image_generator is None
        assert composer.audio_generator is None

        await composer.initialize()

        assert composer.prompt_generator is not None
        assert composer.image_generator is not None
        assert composer.audio_generator is not None

    # ============================================================================
    # Create Dataset Method Tests
    # ============================================================================

    async def test_create_dataset_basic(self, initialized_synthetic_composer):
        """Test basic dataset creation with text-only conversations."""
        # Fix the number of turns per conversation
        initialized_synthetic_composer.config.conversation.turn.mean = 2
        initialized_synthetic_composer.config.conversation.turn.stddev = 0

        conversations = await initialized_synthetic_composer.create_dataset()

        # Test create_dataset returns correct number of conversations
        assert len(conversations) == 5  # num_conversations

        # Test each conversation has correct structure (id, turns)
        for conversation in conversations:
            assert isinstance(conversation, Conversation)
            assert conversation.id is not None
            assert len(conversation.turns) == 2

            for turn in conversation.turns:
                assert isinstance(turn, Turn)
                assert len(turn.texts) == 1  # single text field per turn
                assert len(turn.texts[0].contents) == 1  # batch_size = 1
                assert len(turn.images) == 0  # no images
                assert len(turn.audios) == 0  # no audio

    async def test_create_dataset_with_images(self, image_config, mock_tokenizer):
        """Test dataset creation with image generation enabled."""
        image_config.conversation.turn.mean = 1
        image_config.conversation.turn.stddev = 0

        composer = SyntheticDatasetComposer(image_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

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

    async def test_create_dataset_with_audio(self, audio_config, mock_tokenizer):
        """Test dataset creation with audio generation enabled."""
        audio_config.conversation.turn.mean = 1
        audio_config.conversation.turn.stddev = 0

        composer = SyntheticDatasetComposer(audio_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

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

    async def test_create_dataset_multimodal(self, multimodal_config, mock_tokenizer):
        """Test dataset creation with both image and audio enabled."""
        multimodal_config.conversation.turn.mean = 1
        multimodal_config.conversation.turn.stddev = 0

        composer = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

        # Test conversations include both image and audio payloads
        assert len(conversations) == multimodal_config.conversation.num
        for conversation in conversations:
            for turn in conversation.turns:
                # Test correct batch sizes for all modalities
                assert len(turn.texts) == 1  # single text field per turn
                assert len(turn.texts[0].contents) == 2  # batch_size = 2
                assert len(turn.images) == 1  # single image field per turn
                assert len(turn.images[0].contents) == 2  # batch_size = 2
                assert len(turn.audios) == 1  # single audio field per turn
                assert len(turn.audios[0].contents) == 2  # batch_size = 2

    @patch(
        "aiperf.services.dataset.generator.prompt.PromptGenerator.sample_prefix_prompt"
    )
    async def test_create_dataset_with_prefix_prompts(
        self, mock_prefix, prefix_prompt_config, mock_tokenizer
    ):
        """Test dataset creation with prefix prompts enabled."""
        prefix_prompt_config.conversation.turn.mean = 2  # 2 turns per conversation
        prefix_prompt_config.conversation.turn.stddev = 0
        mock_prefix.return_value = "Prefix prompt:"

        composer = SyntheticDatasetComposer(prefix_prompt_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

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

    async def test_create_dataset_multiple_turns(
        self, multiturn_config, mock_tokenizer
    ):
        """Test dataset creation with multiple turns and delays."""
        composer = SyntheticDatasetComposer(multiturn_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

        # Test conversations have multiple turns
        assert len(conversations) == 3

        for conversation in conversations:
            assert len(conversation.turns) == 2
            assert conversation.turns[0].delay is None  # first turn has no delay
            assert conversation.turns[1].delay == 1500  # subsequent turns have delays

    # ============================================================================
    # Create Turn Method Tests
    # ============================================================================

    async def test_create_first_turn(self, synthetic_config, mock_tokenizer):
        """Test _create_turn method for first turn in conversation."""
        synthetic_config.conversation.turn.delay.mean = 1500
        synthetic_config.conversation.turn.delay.stddev = 0

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        await composer.initialize()

        # Test first turn creation
        turn = await composer._create_turn(is_first=True)

        assert isinstance(turn, Turn)
        assert len(turn.texts) == 1  # single text field per turn
        assert len(turn.images) == 0  # no images
        assert len(turn.audios) == 0  # no audio
        assert turn.delay is None  # first turn has no delay

    async def test_create_turn_subsequent_turn(self, multiturn_config, mock_tokenizer):
        """Test _create_turn method for subsequent turns in conversation."""
        composer = SyntheticDatasetComposer(multiturn_config, mock_tokenizer)
        await composer.initialize()

        # Test subsequent turn creation
        turn = await composer._create_turn(is_first=False)

        assert isinstance(turn, Turn)
        assert len(turn.texts) == 1
        # Test subsequent turns have delays
        assert turn.delay == 1500

    async def test_create_turn_with_all_modalities(
        self, multimodal_config, mock_tokenizer
    ):
        """Test _create_turn method with text, image, and audio."""
        multimodal_config.conversation.turn.delay.mean = 1500
        multimodal_config.conversation.turn.delay.stddev = 0

        composer = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)
        await composer.initialize()

        turn = await composer._create_turn(is_first=True)

        assert isinstance(turn, Turn)
        assert len(turn.texts) == 1  # single text field per turn
        assert len(turn.texts[0].contents) == 2  # batch_size = 2
        assert len(turn.images) == 1  # single image field per turn
        assert len(turn.images[0].contents) == 2  # batch_size = 2
        assert len(turn.audios) == 1  # single audio field per turn
        assert len(turn.audios[0].contents) == 2  # batch_size = 2
        assert turn.delay is None  # first turn has no delay

        # Test subsequent turn creation
        turn = await composer._create_turn(is_first=False)

        assert isinstance(turn, Turn)
        assert turn.delay == 1500

    # ============================================================================
    # Generate Payload Methods Tests
    # ============================================================================

    @patch("aiperf.services.dataset.generator.prompt.PromptGenerator.generate")
    async def test_generate_text_payloads_basic(
        self, mock_generate, synthetic_config, mock_tokenizer
    ):
        """Test _generate_text_payloads method with basic configuration."""
        mock_generate.return_value = "Generated text content"

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        await composer.initialize()

        # Test text payload generation
        turn = Turn()
        text = await composer._generate_text_payloads(is_first=True)
        turn.texts.append(text)

        # Test correct number of text payloads based on batch_size
        assert len(turn.texts) == 1  # batch_size = 1

        # Test text content is generated using prompt generator
        text_payload = turn.texts[0]
        assert isinstance(text_payload, Text)
        assert text_payload.name == "text"
        assert text_payload.contents == ["Generated text content"]

    @patch(
        "aiperf.services.dataset.generator.prompt.PromptGenerator.sample_prefix_prompt"
    )
    @patch("aiperf.services.dataset.generator.prompt.PromptGenerator.generate")
    async def test_generate_text_payloads_first_turn_with_prefix(
        self, mock_generate, mock_prefix, prefix_prompt_config, mock_tokenizer
    ):
        """Test _generate_text_payloads for first turn with prefix prompts."""
        mock_generate.return_value = "User message"
        mock_prefix.return_value = "Prefix prompt:"

        composer = SyntheticDatasetComposer(prefix_prompt_config, mock_tokenizer)
        await composer.initialize()

        # Test prefix prompt is added to first turn
        turn = Turn()
        text = await composer._generate_text_payloads(is_first=True)
        turn.texts.append(text)

        text_payload = turn.texts[0]
        # Test prefix prompt format ("prefix prompt")
        assert text_payload.contents == ["Prefix prompt: User message"]

    @patch("aiperf.services.dataset.generator.prompt.PromptGenerator.generate")
    async def test_generate_text_payloads_subsequent_turn_no_prefix(
        self, mock_generate, prefix_prompt_config, mock_tokenizer
    ):
        """Test _generate_text_payloads for subsequent turns without prefix prompts."""
        mock_generate.return_value = "User message"

        composer = SyntheticDatasetComposer(prefix_prompt_config, mock_tokenizer)
        await composer.initialize()

        # Test no prefix prompt is added to subsequent turns
        turn = Turn()
        text = await composer._generate_text_payloads(is_first=False)
        turn.texts.append(text)

        text_payload = turn.texts[0]
        assert text_payload.contents == ["User message"]  # No prefix

    @patch("aiperf.services.dataset.generator.prompt.PromptGenerator.generate")
    async def test_generate_text_payloads_multiple_batch_size(
        self, mock_generate, synthetic_config, mock_tokenizer
    ):
        """Test _generate_text_payloads with batch_size > 1."""
        mock_generate.return_value = "Generated text"
        synthetic_config.prompt.batch_size = 3

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        await composer.initialize()

        # Test multiple text payloads are generated per turn
        turn = Turn()
        text = await composer._generate_text_payloads(is_first=True)
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

    @patch("aiperf.services.dataset.generator.image.ImageGenerator.generate")
    async def test_generate_image_payloads(
        self, mock_generate, image_config, mock_tokenizer
    ):
        """Test _generate_image_payloads method."""
        mock_generate.return_value = "fake_image_data"

        composer = SyntheticDatasetComposer(image_config, mock_tokenizer)
        await composer.initialize()

        # Test image payload generation
        turn = Turn()
        image = await composer._generate_image_payloads()
        turn.images.append(image)

        # Test correct number of image payloads based on batch_size
        assert len(turn.images) == 1  # batch_size = 1

        # Test image content is generated using image generator
        image_payload = turn.images[0]
        assert isinstance(image_payload, Image)
        assert image_payload.name == "image_url"
        assert image_payload.contents == ["fake_image_data"]

    @patch("aiperf.services.dataset.generator.audio.AudioGenerator.generate")
    async def test_generate_audio_payloads(
        self, mock_generate, audio_config, mock_tokenizer
    ):
        """Test _generate_audio_payloads method."""
        mock_generate.return_value = "fake_audio_data"

        composer = SyntheticDatasetComposer(audio_config, mock_tokenizer)
        await composer.initialize()

        # Test audio payload generation
        turn = Turn()
        audio = await composer._generate_audio_payloads()
        turn.audios.append(audio)

        # Test correct number of audio payloads based on batch_size
        assert len(turn.audios) == 1  # batch_size = 1

        audio_payload = turn.audios[0]
        assert audio_payload.name == "input_audio"
        assert audio_payload.contents == ["fake_audio_data"]

    # ============================================================================
    # Configuration Variations Tests
    # ============================================================================

    async def test_zero_conversations(self, synthetic_config, mock_tokenizer):
        """Test behavior with zero conversations requested."""
        synthetic_config.conversation.num = 0

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

        assert len(conversations) == 0

    @patch(
        "aiperf.services.dataset.composer.synthetic.utils.sample_positive_normal_integer"
    )
    async def test_edge_case_statistical_parameters(self, mock_sample, mock_tokenizer):
        """Test behavior with edge case statistical parameters."""
        mock_sample.return_value = 1

        config = InputConfig(
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
        )

        composer = SyntheticDatasetComposer(config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

        # Test with very small/large mean and stddev values
        assert len(conversations) == 2
        assert all(len(conv.turns) == 1 for conv in conversations)  # mocked return

    @pytest.mark.parametrize("num_conversations", [1, 5, 10, 50])
    async def test_different_conversation_counts(
        self, synthetic_config, num_conversations, mock_tokenizer
    ):
        """Test dataset creation with different conversation counts."""
        synthetic_config.conversation.num = num_conversations

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

        # Parametrized test for different num_conversations values
        assert len(conversations) == num_conversations

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    @patch(
        "aiperf.services.dataset.composer.synthetic.utils.sample_positive_normal_integer"
    )
    async def test_different_batch_sizes(
        self, mock_sample, synthetic_config, batch_size, mock_tokenizer
    ):
        """Test dataset creation with different batch sizes."""
        mock_sample.return_value = 1

        synthetic_config.prompt.batch_size = batch_size

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        await composer.initialize()
        conversations = await composer.create_dataset()

        # Parametrized test for different batch_size values
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 1  # single text field per turn

        text_payload = turn.texts[0]
        assert len(text_payload.contents) == batch_size

    # ============================================================================
    # Miscellaneous Tests
    # ============================================================================

    async def test_reproducibility_with_fixed_seed(
        self, multimodal_config, mock_tokenizer
    ):
        """Test that dataset generation is reproducible with fixed random seed."""
        multimodal_config.prompt.input_tokens.stddev = 2
        multimodal_config.image.width.stddev = 2
        multimodal_config.image.height.stddev = 2
        multimodal_config.audio.length.stddev = 2
        multimodal_config.conversation.turn = TurnConfig(
            mean=2, stddev=2, delay=TurnDelayConfig(mean=1500, stddev=2)
        )

        random.seed(42)
        np.random.seed(42)
        composer1 = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)
        await composer1.initialize()
        conversations1 = await composer1.create_dataset()

        random.seed(42)
        np.random.seed(42)
        composer2 = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)
        await composer2.initialize()
        conversations2 = await composer2.create_dataset()

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

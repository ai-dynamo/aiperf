# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from aiperf.common import random_generator as rng
from aiperf.common.models import Audio, Conversation, Image, Text, Turn
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from tests.unit.dataset.composer.conftest import make_run


class TestSyntheticDatasetComposer:
    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_basic_config(self, synthetic_config, mock_tokenizer):
        """Test that SyntheticDatasetComposer can be instantiated with basic config."""
        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )

        assert composer.run.cfg.get_default_dataset().entries == 5
        assert composer.prompt_generator is not None
        assert composer.include_image is False
        assert composer.include_audio is False

    def test_initialization_with_images(self, image_config, mock_tokenizer):
        """Test initialization with image generation enabled."""
        composer = SyntheticDatasetComposer(
            run=make_run(image_config), tokenizer=mock_tokenizer
        )

        images = composer._synthetic_images
        assert images is not None
        assert images.width.expected_value == 10
        assert images.height.expected_value == 10
        assert composer.include_image is True
        assert composer.include_audio is False

    def test_initialization_with_audio(self, audio_config, mock_tokenizer):
        """Test initialization with audio generation enabled."""
        composer = SyntheticDatasetComposer(
            run=make_run(audio_config), tokenizer=mock_tokenizer
        )

        assert composer._synthetic_audio is not None
        assert composer._synthetic_audio.length.expected_value == 2
        assert composer.include_image is False
        assert composer.include_audio is True

    def test_initialization_with_multimodal(self, multimodal_config, mock_tokenizer):
        """Test initialization with both image and audio enabled."""
        composer = SyntheticDatasetComposer(
            run=make_run(multimodal_config), tokenizer=mock_tokenizer
        )

        assert composer.include_image is True
        assert composer.include_audio is True
        assert composer._synthetic_images.batch_size == 2
        assert composer._synthetic_audio.batch_size == 2
        assert composer._synthetic_images.width.expected_value == 10
        assert composer._synthetic_images.height.expected_value == 10
        assert composer._synthetic_audio.length.expected_value == 2

    def test_initialization_with_all_zero_mean(self, mock_tokenizer):
        """Test initialization with no generators enabled."""
        config = CLIConfig(
            model_names=["test_model"],
            conversation_num_dataset_entries=5,
            prompt_input_tokens_mean=0,
            image_width_mean=0,
            image_height_mean=0,
            audio_length_mean=0,
        )

        with pytest.raises(ValueError):
            SyntheticDatasetComposer(run=make_run(config), tokenizer=mock_tokenizer)

    # ============================================================================
    # Create Dataset Method Tests
    # ============================================================================

    def test_create_dataset_basic(self, synthetic_config, mock_tokenizer):
        """Test basic dataset creation with text-only conversations."""
        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Test create_dataset returns correct number of conversations
        assert len(conversations) == 5  # num_conversations

        # Test each conversation has correct structure (session_id, turns)
        for conversation in conversations:
            assert isinstance(conversation, Conversation)
            assert conversation.session_id is not None
            # With global RNG seed 42, verify structure without mocking
            assert len(conversation.turns) >= 1  # at least one turn per conversation

            for turn in conversation.turns:
                assert isinstance(turn, Turn)
                assert len(turn.texts) == 1  # single text field per turn
                assert len(turn.texts[0].contents) == 1  # batch_size = 1
                assert len(turn.images) == 0  # no images
                assert len(turn.audios) == 0  # no audio

    def test_create_dataset_with_images(self, image_config, mock_tokenizer):
        """Test dataset creation with image generation enabled."""
        composer = SyntheticDatasetComposer(
            run=make_run(image_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Test conversations include image payloads
        assert len(conversations) == 3
        for conversation in conversations:
            assert len(conversation.turns) >= 1
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

    def test_create_dataset_with_audio(self, audio_config, mock_tokenizer):
        """Test dataset creation with audio generation enabled."""
        composer = SyntheticDatasetComposer(
            run=make_run(audio_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Test conversations include audio payloads
        assert len(conversations) == 3
        for conversation in conversations:
            assert len(conversation.turns) >= 1
            for turn in conversation.turns:
                assert len(turn.texts) == 1  # single text field per turn
                assert len(turn.texts[0].contents) == 1  # batch_size = 1
                assert len(turn.images) == 0  # no images
                assert len(turn.audios) == 1  # single audio field per turn
                assert len(turn.audios[0].contents) == 1  # batch_size = 1

                # Check audio properties
                audio = turn.audios[0]
                assert isinstance(audio, Audio)

    def test_create_dataset_multimodal(self, multimodal_config, mock_tokenizer):
        """Test dataset creation with both image and audio enabled."""
        composer = SyntheticDatasetComposer(
            run=make_run(multimodal_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Test conversations include both image and audio payloads
        assert len(conversations) == multimodal_config.conversation_num_dataset_entries
        for conversation in conversations:
            assert len(conversation.turns) >= 1
            for turn in conversation.turns:
                # Test correct batch sizes for all modalities
                assert len(turn.texts) == 1  # single text field per turn
                assert len(turn.texts[0].contents) == 2  # batch_size = 2
                assert len(turn.images) == 1  # single image field per turn
                assert len(turn.images[0].contents) == 2  # batch_size = 2
                assert len(turn.audios) == 1  # single audio field per turn
                assert len(turn.audios[0].contents) == 2  # batch_size = 2

    def test_create_dataset_with_prefix_prompts(
        self, prefix_prompt_config, mock_tokenizer
    ):
        """Test dataset creation with prefix prompts enabled."""
        composer = SyntheticDatasetComposer(
            run=make_run(prefix_prompt_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        assert len(conversations) == 5
        for conversation in conversations:
            assert len(conversation.turns) >= 1
            # Test that first turns have text content (prefix prompt should be added)
            first_turn = conversation.turns[0]
            first_text_content = first_turn.texts[0].contents[0]
            # Verify text content exists (prefix prompt handling is tested elsewhere)
            assert len(first_text_content) > 0
            assert isinstance(first_text_content, str)

    def test_create_dataset_multiple_turns(self, multiturn_config, mock_tokenizer):
        """Test dataset creation with multiple turns and delays."""
        composer = SyntheticDatasetComposer(
            run=make_run(multiturn_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Test conversations have multiple turns
        assert len(conversations) == 4

        for conversation in conversations:
            assert len(conversation.turns) == 2
            assert conversation.turns[0].delay is None  # first turn has no delay
            assert conversation.turns[1].delay == 1500  # subsequent turns have delays

    # ============================================================================
    # Create Turn Method Tests
    # ============================================================================

    def test_create_first_turn(self, synthetic_config, mock_tokenizer):
        """Test _create_turn method for first turn in conversation."""
        synthetic_config.conversation_turn_delay_mean = 1500
        synthetic_config.conversation_turn_delay_stddev = 0

        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )

        # Test first turn creation
        turn = composer._create_turn(is_first=True)

        assert isinstance(turn, Turn)
        assert len(turn.texts) == 1  # single text field per turn
        assert len(turn.images) == 0  # no images
        assert len(turn.audios) == 0  # no audio
        assert turn.delay is None  # first turn has no delay

    def test_create_turn_subsequent_turn(self, multiturn_config, mock_tokenizer):
        """Test _create_turn method for subsequent turns in conversation."""
        composer = SyntheticDatasetComposer(
            run=make_run(multiturn_config), tokenizer=mock_tokenizer
        )

        # Test subsequent turn creation
        turn = composer._create_turn(is_first=False)

        assert isinstance(turn, Turn)
        assert len(turn.texts) == 1
        # Test subsequent turns have delays
        assert turn.delay == 1500

    def test_create_turn_with_all_modalities(self, multimodal_config, mock_tokenizer):
        """Test _create_turn method with text, image, and audio."""
        multimodal_config.conversation_turn_delay_mean = 1500
        multimodal_config.conversation_turn_delay_stddev = 0

        composer = SyntheticDatasetComposer(
            run=make_run(multimodal_config), tokenizer=mock_tokenizer
        )

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

    def test_create_turn_with_delay_ratio(self, multiturn_config, mock_tokenizer):
        """Test _create_turn method applies delay ratio correctly."""
        multiturn_config.conversation_turn_delay_mean = 2000
        multiturn_config.conversation_turn_delay_stddev = 0
        multiturn_config.conversation_turn_delay_ratio = 0.5

        composer = SyntheticDatasetComposer(
            run=make_run(multiturn_config), tokenizer=mock_tokenizer
        )

        # Test subsequent turn creation
        turn = composer._create_turn(is_first=False)

        assert isinstance(turn, Turn)
        # Delay should be mean * ratio
        assert turn.delay == 1000  # 2000 * 0.5

    def test_turn_delays_from_config_options(self, mock_tokenizer):
        """Test that delays configured via CLI options properly show up in Turn.delay."""
        # Configure delays using the same options available via CLI
        config = CLIConfig(
            model_names=["test_model"],
            conversation_num_dataset_entries=5,
            conversation_turn_mean=3,
            conversation_turn_stddev=0,
            conversation_turn_delay_mean=2500,  # --conversation-turn-delay-mean 2500
            conversation_turn_delay_stddev=500,  # --conversation-turn-delay-stddev 500
            conversation_turn_delay_ratio=1.0,
            prompt_input_tokens_mean=100,
        )
        rng.reset()
        rng.init(42)  # Set seed for reproducibility
        composer = SyntheticDatasetComposer(
            run=make_run(config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Verify conversations were created
        assert len(conversations) == 5

        # Check each conversation
        for conversation in conversations:
            assert len(conversation.turns) == 3  # mean=3, stddev=0

            # First turn should have no delay
            assert conversation.turns[0].delay is None

            # Subsequent turns should have delays
            for turn_idx in range(1, 3):
                turn = conversation.turns[turn_idx]
                assert turn.delay is not None
                assert turn.delay > 0
                # With stddev=500 and seed=42, delays should vary around mean=2500
                # but generally be in reasonable range (e.g., 1000-4000 ms)
                assert 1000 <= turn.delay <= 4000

        # Test with ratio scaling
        config.conversation_turn_delay_ratio = 0.5
        rng.reset()
        rng.init(42)  # Reset seed
        composer = SyntheticDatasetComposer(
            run=make_run(config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        for conversation in conversations:
            # First turn should still have no delay
            assert conversation.turns[0].delay is None

            # Check that ratio is applied (delays should be roughly half)
            for turn_idx in range(1, 3):
                turn = conversation.turns[turn_idx]
                assert turn.delay is not None
                # With ratio=0.5, delays should be roughly halved
                assert 500 <= turn.delay <= 2000

    def test_turn_delays_with_zero_mean(self, mock_tokenizer):
        """Test that zero mean delay results in no delays on turns."""
        config = CLIConfig(
            model_names=["test_model"],
            conversation_num_dataset_entries=3,
            conversation_turn_mean=2,
            conversation_turn_stddev=0,
            conversation_turn_delay_mean=0,  # No delay
            conversation_turn_delay_stddev=0,
            conversation_turn_delay_ratio=1.0,
            prompt_input_tokens_mean=100,
        )
        rng.reset()
        rng.init(42)
        composer = SyntheticDatasetComposer(
            run=make_run(config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        for conversation in conversations:
            # All turns should have None delay when mean=0
            for turn in conversation.turns:
                assert turn.delay is None

    # ============================================================================
    # Generate Payload Methods Tests
    # ============================================================================

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_basic(
        self, mock_generate, synthetic_config, mock_tokenizer
    ):
        """Test _generate_text_payloads method with basic configuration."""
        mock_generate.return_value = "Generated text content"

        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )

        # Test text payload generation
        turn = Turn()
        text = composer._generate_text_payloads(turn, is_first=True)
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

        composer = SyntheticDatasetComposer(
            run=make_run(prefix_prompt_config), tokenizer=mock_tokenizer
        )

        # Test prefix prompt is added to first turn
        turn = Turn()
        text = composer._generate_text_payloads(turn, is_first=True)
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

        composer = SyntheticDatasetComposer(
            run=make_run(prefix_prompt_config), tokenizer=mock_tokenizer
        )

        # Test no prefix prompt is added to subsequent turns
        turn = Turn()
        text = composer._generate_text_payloads(turn, is_first=False)
        turn.texts.append(text)

        text_payload = turn.texts[0]
        assert text_payload.contents == ["User message"]  # No prefix

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_multiple_batch_size(
        self, mock_generate, synthetic_config, mock_tokenizer
    ):
        """Test _generate_text_payloads with batch_size > 1."""
        mock_generate.return_value = "Generated text"
        synthetic_config.prompt_batch_size = 3

        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )

        # Test multiple text payloads are generated per turn
        turn = Turn()
        text = composer._generate_text_payloads(turn, is_first=True)
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

        composer = SyntheticDatasetComposer(
            run=make_run(image_config), tokenizer=mock_tokenizer
        )

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

        composer = SyntheticDatasetComposer(
            run=make_run(audio_config), tokenizer=mock_tokenizer
        )

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
        # v2 forbids entries=0 at validation; mutate the resolved dataset
        # post-construction to verify the create_dataset() loop handles 0.
        run = make_run(synthetic_config)
        run.cfg.datasets[0].entries = 0

        composer = SyntheticDatasetComposer(run=run, tokenizer=mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 0

    def test_edge_case_statistical_parameters(self, mock_tokenizer):
        """Test behavior with edge case statistical parameters."""
        # v1 PrefixPromptConfig allowed pool_size=0 (disabled); v2 forbids it
        # (ge=1). Drop the prefix config — the v1 default is no prefix anyway.
        config = CLIConfig(
            model_names=["test-model"],
            conversation_num_dataset_entries=2,
            prompt_input_tokens_mean=1,  # Very small mean
            prompt_input_tokens_stddev=0,  # Zero stddev
        )

        composer = SyntheticDatasetComposer(
            run=make_run(config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Test with very small/large mean and stddev values
        assert len(conversations) == 2
        # With large mean/stddev for turns, should create valid conversations
        assert all(len(conv.turns) >= 1 for conv in conversations)

    def test_multi_turn_does_not_control_dataset_entries(self, mock_tokenizer):
        """Test that multi-turn settings do not affect num_dataset_entries."""
        config = CLIConfig(
            model_names=["test-model"],
            conversation_num_dataset_entries=10,
            conversation_num=2,
        )

        composer = SyntheticDatasetComposer(
            run=make_run(config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Verify that num_dataset_entries controls the number of conversations generated
        assert len(conversations) == 10

    @pytest.mark.parametrize("num_conversations", [1, 5, 10, 50])
    def test_different_conversation_counts(
        self, synthetic_config, num_conversations, mock_tokenizer
    ):
        """Test dataset creation with different conversation counts."""
        synthetic_config.conversation_num_dataset_entries = num_conversations

        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Parametrized test for different num_conversations values
        assert len(conversations) == num_conversations

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    def test_different_batch_sizes(self, synthetic_config, batch_size, mock_tokenizer):
        """Test dataset creation with different batch sizes."""
        synthetic_config.prompt_batch_size = batch_size

        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Parametrized test for different batch_size values
        assert len(conversations) > 0
        assert len(conversations[0].turns) >= 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 1  # single text field per turn

        text_payload = turn.texts[0]
        assert len(text_payload.contents) == batch_size

    # ============================================================================
    # Miscellaneous Tests
    # ============================================================================

    def test_missing_required_generators(self, synthetic_config, mock_tokenizer):
        """Test behavior when required generators are missing."""
        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )

        # Test error handling when generators are not properly initialized
        # Simulate missing tokenizer in generator
        composer.prompt_generator = None

        with pytest.raises(
            ValueError, match="Text prompt generation requires a tokenizer"
        ):
            composer.create_dataset()

    def test_reproducibility_with_fixed_seed(self, multimodal_config, mock_tokenizer):
        """Test that dataset generation is reproducible with fixed random seed."""
        multimodal_config.prompt_input_tokens_stddev = 2
        multimodal_config.image_width_stddev = 2
        multimodal_config.image_height_stddev = 2
        multimodal_config.audio_length_stddev = 2
        multimodal_config.conversation_turn_mean = 2
        multimodal_config.conversation_turn_stddev = 2
        multimodal_config.conversation_turn_delay_mean = 1500
        multimodal_config.conversation_turn_delay_stddev = 2

        rng.reset()
        rng.init(42)
        composer1 = SyntheticDatasetComposer(
            run=make_run(multimodal_config), tokenizer=mock_tokenizer
        )
        conversations1 = composer1.create_dataset()

        rng.reset()
        rng.init(42)
        composer2 = SyntheticDatasetComposer(
            run=make_run(multimodal_config), tokenizer=mock_tokenizer
        )
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

    def test_model_selection_random(self, synthetic_config, mock_tokenizer):
        """Test random model selection strategy."""
        synthetic_config.model_selection_strategy = "random"
        synthetic_config.model_names = ["test-model-1", "test-model-2"]
        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )

        conversations = composer.create_dataset()

        # With random selection, verify models are from the valid set
        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.model in ["test-model-1", "test-model-2"]

    def test_model_selection_round_robin(self, synthetic_config, mock_tokenizer):
        synthetic_config.model_selection_strategy = "round_robin"
        synthetic_config.model_names = ["test-model-1", "test-model-2"]

        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # Check that models are selected in round-robin fashion
        for i, conversation in enumerate(conversations):
            for j, turn in enumerate(conversation.turns):
                expected_model = "test-model-1" if (i + j) % 2 == 0 else "test-model-2"
                assert turn.model == expected_model

    # ============================================================================
    # Max Token Tests
    # ============================================================================

    def test_max_tokens_integration_with_mean(self, synthetic_config, mock_tokenizer):
        synthetic_config.prompt_output_tokens_mean = 100
        synthetic_config.prompt_output_tokens_stddev = 5.0

        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        # With global RNG, verify max_tokens is set to a positive integer
        # around the mean of 100
        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens is not None
                assert turn.max_tokens > 0
                assert isinstance(turn.max_tokens, int)
                # Should be roughly around the mean of 100 (within 3 stddev)
                assert 85 <= turn.max_tokens <= 115

    def test_max_tokens_not_set_when_mean_none(self, synthetic_config, mock_tokenizer):
        synthetic_config.prompt_output_tokens_mean = None
        synthetic_config.prompt_output_tokens_stddev = None

        composer = SyntheticDatasetComposer(
            run=make_run(synthetic_config), tokenizer=mock_tokenizer
        )
        conversations = composer.create_dataset()

        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens is None

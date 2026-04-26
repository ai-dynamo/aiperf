# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from aiperf.common import random_generator as rng
from aiperf.common.models import Audio, Conversation, Image, Text, Turn
from aiperf.config import AIPerfConfig
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from tests.unit.dataset.composer.conftest import _make_run

# ---------------------------------------------------------------------------
# Local helpers mirroring conftest._config / _BASE for tests that need
# to construct configs inline (cannot use fixtures).
# ---------------------------------------------------------------------------
_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    phases=[
        {"name": "default", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
)


def _make_config(**dataset_overrides) -> AIPerfConfig:
    """Build an AIPerfConfig with a single synthetic dataset, merging overrides."""
    dataset: dict = {
        "type": "synthetic",
        "entries": 100,
        "prompts": {"isl": 128, "osl": 64},
    }
    dataset.update(dataset_overrides)
    return AIPerfConfig(**_BASE, datasets=[{"name": "default", **dataset}])


class TestSyntheticDatasetComposer:
    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_basic_config(self, synthetic_config, mock_tokenizer):
        """Test that SyntheticDatasetComposer can be instantiated with basic config."""
        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)

        assert composer.run == synthetic_config
        assert composer.dataset_config.entries == 5
        assert composer.prompt_generator is not None
        assert composer.include_image is False
        assert composer.include_audio is False

    def test_initialization_with_images(self, image_config, mock_tokenizer):
        """Test initialization with image generation enabled."""
        composer = SyntheticDatasetComposer(image_config, mock_tokenizer)

        assert composer.dataset_config.images.width.mean == 10
        assert composer.dataset_config.images.height.mean == 10
        assert composer.include_image is True
        assert composer.include_audio is False

    def test_initialization_with_audio(self, audio_config, mock_tokenizer):
        """Test initialization with audio generation enabled."""
        composer = SyntheticDatasetComposer(audio_config, mock_tokenizer)

        assert composer.dataset_config.audio.length.mean == 2
        assert composer.include_image is False
        assert composer.include_audio is True

    def test_initialization_with_multimodal(self, multimodal_config, mock_tokenizer):
        """Test initialization with both image and audio enabled."""
        composer = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)

        assert composer.include_image is True
        assert composer.include_audio is True
        dc = composer.dataset_config
        assert dc.images.batch_size == 2
        assert dc.audio.batch_size == 2
        assert dc.images.width.mean == 10
        assert dc.images.height.mean == 10
        assert dc.audio.length.mean == 2

    def test_initialization_with_all_zero_mean(self, mock_tokenizer):
        """Test initialization with no generators enabled."""
        config = AIPerfConfig(
            models=["test_model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 5,
                    "prompts": {"isl": {"mean": 0}},
                    "images": {
                        "width": {"mean": 0},
                        "height": {"mean": 0},
                    },
                    "audio": {"length": {"mean": 0}},
                }
            ],
            phases=[
                {
                    "name": "default",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        )

        with pytest.raises(ValueError):
            SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

    # ============================================================================
    # Create Dataset Method Tests
    # ============================================================================

    def test_create_dataset_basic(self, synthetic_config, mock_tokenizer):
        """Test basic dataset creation with text-only conversations."""
        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Test create_dataset returns correct number of conversations
        assert len(conversations) == 5  # entries from synthetic_config fixture

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
        composer = SyntheticDatasetComposer(image_config, mock_tokenizer)
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
        composer = SyntheticDatasetComposer(audio_config, mock_tokenizer)
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
        composer = SyntheticDatasetComposer(multimodal_config, mock_tokenizer)
        conversations = composer.create_dataset()

        # Test conversations include both image and audio payloads
        assert len(conversations) == multimodal_config.cfg.get_default_dataset().entries
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
        composer = SyntheticDatasetComposer(prefix_prompt_config, mock_tokenizer)
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
        composer = SyntheticDatasetComposer(multiturn_config, mock_tokenizer)
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

    def test_create_first_turn(self, mock_tokenizer):
        """Test _create_turn method for first turn in conversation."""
        config = _make_config(
            entries=5,
            prompts={"isl": {"mean": 10, "stddev": 2}, "osl": 64},
            turn_delay={"mean": 1500, "stddev": 0},
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

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

    def test_create_turn_with_all_modalities(self, mock_tokenizer):
        """Test _create_turn method with text, image, and audio."""
        config = _make_config(
            entries=2,
            prompts={
                "isl": {"mean": 10, "stddev": 2},
                "osl": 64,
                "batch_size": 2,
            },
            prefix_prompts={"pool_size": 2, "length": 15},
            images={
                "batch_size": 2,
                "width": {"mean": 10},
                "height": {"mean": 10},
            },
            audio={"batch_size": 2, "length": {"mean": 2}},
            turn_delay={"mean": 1500, "stddev": 0},
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

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

    def test_create_turn_with_delay_ratio(self, mock_tokenizer):
        """Test _create_turn method applies delay ratio correctly."""
        config = _make_config(
            entries=4,
            prompts={"isl": {"mean": 10, "stddev": 2}, "osl": 64},
            turns={"mean": 2, "stddev": 0},
            turn_delay={"mean": 2000, "stddev": 0},
            turn_delay_ratio=0.5,
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        # Test subsequent turn creation
        turn = composer._create_turn(is_first=False)

        assert isinstance(turn, Turn)
        # Delay should be mean * ratio
        assert turn.delay == 1000  # 2000 * 0.5

    def test_turn_delays_from_config_options(self, mock_tokenizer):
        """Test that delays configured via CLI options properly show up in Turn.delay."""
        config = _make_config(
            entries=5,
            prompts={"isl": {"mean": 100, "stddev": 0}, "osl": 64},
            turns={"mean": 3, "stddev": 0},
            turn_delay={"mean": 2500, "stddev": 500},
            turn_delay_ratio=1.0,
        )
        rng.reset()
        rng.init(42)  # Set seed for reproducibility
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
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
        config_with_ratio = _make_config(
            entries=5,
            prompts={"isl": {"mean": 100, "stddev": 0}, "osl": 64},
            turns={"mean": 3, "stddev": 0},
            turn_delay={"mean": 2500, "stddev": 500},
            turn_delay_ratio=0.5,
        )
        rng.reset()
        rng.init(42)  # Reset seed
        composer = SyntheticDatasetComposer(
            _make_run(config_with_ratio), mock_tokenizer
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
        config = _make_config(
            entries=3,
            prompts={"isl": {"mean": 100, "stddev": 0}, "osl": 64},
            turns={"mean": 2, "stddev": 0},
            turn_delay={"mean": 0, "stddev": 0},
            turn_delay_ratio=1.0,
        )
        rng.reset()
        rng.init(42)
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
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

        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)

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

        composer = SyntheticDatasetComposer(prefix_prompt_config, mock_tokenizer)

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

        composer = SyntheticDatasetComposer(prefix_prompt_config, mock_tokenizer)

        # Test no prefix prompt is added to subsequent turns
        turn = Turn()
        text = composer._generate_text_payloads(turn, is_first=False)
        turn.texts.append(text)

        text_payload = turn.texts[0]
        assert text_payload.contents == ["User message"]  # No prefix

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_text_payloads_multiple_batch_size(
        self, mock_generate, mock_tokenizer
    ):
        """Test _generate_text_payloads with batch_size > 1."""
        mock_generate.return_value = "Generated text"
        config = _make_config(
            entries=5,
            prompts={
                "isl": {"mean": 10, "stddev": 2},
                "osl": 64,
                "batch_size": 3,
            },
        )

        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

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

    def test_one_conversation(self, mock_tokenizer):
        """Test behavior with one conversation requested (minimum allowed)."""
        config = _make_config(
            entries=1,
            prompts={"isl": {"mean": 10, "stddev": 2}, "osl": 64},
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        conversations = composer.create_dataset()

        assert len(conversations) == 1

    def test_edge_case_statistical_parameters(self, mock_tokenizer):
        """Test behavior with edge case statistical parameters."""
        config = _make_config(
            entries=2,
            prompts={"isl": {"mean": 1, "stddev": 0}, "osl": 64},
            prefix_prompts={"pool_size": 1, "length": 1},
            turns={"mean": 100, "stddev": 50},
        )

        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        conversations = composer.create_dataset()

        # Test with very small/large mean and stddev values
        assert len(conversations) == 2
        # With large mean/stddev for turns, should create valid conversations
        assert all(len(conv.turns) >= 1 for conv in conversations)

    def test_multi_turn_does_not_control_dataset_entries(self, mock_tokenizer):
        """Test that multi-turn settings do not affect num_dataset_entries."""
        config = _make_config(
            entries=10,
            prompts={"isl": 128, "osl": 64},
            turns={"mean": 2, "stddev": 0},
        )

        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        conversations = composer.create_dataset()

        # Verify that entries controls the number of conversations generated
        assert len(conversations) == 10

    @pytest.mark.parametrize("num_conversations", [1, 5, 10, 50])
    def test_different_conversation_counts(self, num_conversations, mock_tokenizer):
        """Test dataset creation with different conversation counts."""
        config = _make_config(
            entries=num_conversations,
            prompts={"isl": {"mean": 10, "stddev": 2}, "osl": 64},
        )

        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        conversations = composer.create_dataset()

        # Parametrized test for different num_conversations values
        assert len(conversations) == num_conversations

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
    def test_different_batch_sizes(self, batch_size, mock_tokenizer):
        """Test dataset creation with different batch sizes."""
        config = _make_config(
            entries=5,
            prompts={
                "isl": {"mean": 10, "stddev": 2},
                "osl": 64,
                "batch_size": batch_size,
            },
        )

        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
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
        composer = SyntheticDatasetComposer(synthetic_config, mock_tokenizer)

        # Test error handling when generators are not properly initialized
        # Simulate missing tokenizer in generator
        composer.prompt_generator = None

        with pytest.raises(
            ValueError, match="Text prompt generation requires a tokenizer"
        ):
            composer.create_dataset()

    def test_reproducibility_with_fixed_seed(self, mock_tokenizer):
        """Test that dataset generation is reproducible with fixed random seed."""
        config = _make_config(
            entries=2,
            prompts={
                "isl": {"mean": 10, "stddev": 2},
                "osl": 64,
                "batch_size": 2,
            },
            prefix_prompts={"pool_size": 2, "length": 15},
            images={
                "batch_size": 2,
                "width": {"mean": 10, "stddev": 2},
                "height": {"mean": 10, "stddev": 2},
            },
            audio={
                "batch_size": 2,
                "length": {"mean": 2, "stddev": 2},
            },
            turns={"mean": 2, "stddev": 2},
            turn_delay={"mean": 1500, "stddev": 2},
        )

        rng.reset()
        rng.init(42)
        composer1 = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        conversations1 = composer1.create_dataset()

        rng.reset()
        rng.init(42)
        composer2 = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
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

    def test_model_selection_random(self, mock_tokenizer):
        """Test random model selection strategy."""
        config = AIPerfConfig(
            models={
                "items": [{"name": "test-model-1"}, {"name": "test-model-2"}],
                "strategy": "random",
            },
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 5,
                    "prompts": {
                        "isl": {"mean": 10, "stddev": 2},
                        "osl": 64,
                    },
                }
            ],
            phases=[
                {
                    "name": "default",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        conversations = composer.create_dataset()

        # With random selection, verify models are from the valid set
        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.model in ["test-model-1", "test-model-2"]

    def test_model_selection_round_robin(self, mock_tokenizer):
        """Test round-robin model selection strategy."""
        config = AIPerfConfig(
            models={
                "items": [{"name": "test-model-1"}, {"name": "test-model-2"}],
                "strategy": "round_robin",
            },
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets=[
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 5,
                    "prompts": {
                        "isl": {"mean": 10, "stddev": 2},
                        "osl": 64,
                    },
                }
            ],
            phases=[
                {
                    "name": "default",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        )

        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        conversations = composer.create_dataset()

        # Check that models are selected in round-robin fashion
        for i, conversation in enumerate(conversations):
            for j, turn in enumerate(conversation.turns):
                expected_model = "test-model-1" if (i + j) % 2 == 0 else "test-model-2"
                assert turn.model == expected_model

    # ============================================================================
    # Max Token Tests
    # ============================================================================

    def test_max_tokens_integration_with_mean(self, mock_tokenizer):
        """Test max_tokens is set based on osl mean."""
        config = _make_config(
            entries=5,
            prompts={
                "isl": {"mean": 10, "stddev": 2},
                "osl": {"mean": 100, "stddev": 5},
            },
        )

        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
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

    def test_max_tokens_not_set_when_mean_none(self, mock_tokenizer):
        """Test max_tokens is None when osl is not configured."""
        config = _make_config(
            entries=5,
            prompts={"isl": {"mean": 10, "stddev": 2}},
        )

        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        conversations = composer.create_dataset()

        for conversation in conversations:
            for turn in conversation.turns:
                assert turn.max_tokens is None


# ============================================================================
# ISL sampling end-to-end regression tests
#
# The composer's text path has two places that could sample from the ISL
# distribution:
#   1. BaseDatasetComposer._get_turn_sequence_lengths  (turn level, cached)
#   2. PromptGenerator.generate(mean, stddev)           (per-prompt)
#
# Only (1) should sample. If (2) also samples (stddev > 0 passed from the
# composer), variance inflates to stddev * sqrt(2). These tests lock that
# contract down: the composer must pass stddev=0 to the prompt generator
# and pass the turn-level ISL through as `mean`.
# ============================================================================


class TestIslFlowNoDoubleSample:
    """Lock down: turn-level samples ISL once, prompt generator does NOT re-sample."""

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_passed_stddev_zero_for_constant_isl(
        self, mock_generate, mock_tokenizer
    ):
        """Fixed ISL: generate() must receive stddev=0 and mean=<configured>."""
        mock_generate.return_value = "text"
        config = _make_config(entries=1, prompts={"isl": 128, "osl": 64})
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        composer._generate_text_payloads(Turn(), is_first=True)

        mock_generate.assert_called_once_with(mean=128, stddev=0)

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_passed_stddev_zero_for_normal_isl(
        self, mock_generate, mock_tokenizer
    ):
        """Normal ISL: generate() must STILL receive stddev=0 (sampling already done)."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        composer._generate_text_payloads(Turn(), is_first=True)

        assert mock_generate.call_count == 1
        _, kwargs = mock_generate.call_args
        assert kwargs["stddev"] == 0, (
            f"generate() got stddev={kwargs['stddev']} — double-sampling bug!"
        )
        assert isinstance(kwargs["mean"], int)
        assert 200 <= kwargs["mean"] <= 800

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_generate_mean_matches_cached_turn_level_sample(
        self, mock_generate, mock_tokenizer
    ):
        """The `mean` kwarg to generate() must equal the turn-level ISL cache entry."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turn = Turn()
        composer._generate_text_payloads(turn, is_first=True)

        turn_id = id(turn)
        cached_isl, _ = composer._turn_sequence_cache[turn_id]
        _, kwargs = mock_generate.call_args
        assert kwargs["mean"] == cached_isl

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_batch_size_reuses_same_turn_isl(self, mock_generate, mock_tokenizer):
        """All prompts in a single turn's batch must share the turn's ISL sample."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1,
            prompts={
                "isl": {"mean": 500, "stddev": 100},
                "osl": 64,
                "batch_size": 4,
            },
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        composer._generate_text_payloads(Turn(), is_first=True)

        assert mock_generate.call_count == 4
        means = [c.kwargs["mean"] for c in mock_generate.call_args_list]
        stddevs = [c.kwargs["stddev"] for c in mock_generate.call_args_list]
        assert len(set(means)) == 1, f"batch should share one ISL sample, got {means}"
        assert stddevs == [0, 0, 0, 0]

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_different_turns_get_independent_isl_samples(
        self, mock_generate, mock_tokenizer
    ):
        """Each new turn must trigger a fresh turn-level sample."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 100}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(50)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        means = [c.kwargs["mean"] for c in mock_generate.call_args_list]
        assert len(means) == 50
        assert len(set(means)) > 20, (
            f"turn-level samples stuck: only {len(set(means))} unique ISLs in 50 turns"
        )

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_normal_isl_empirical_stddev_matches_config_not_inflated(
        self, mock_generate, mock_tokenizer
    ):
        """REGRESSION: empirical stddev of ISLs passed to generate() must
        match configured stddev, not stddev * sqrt(2).

        Pre-fix: we'd see ~sqrt(2) * 50 ≈ 70.7.
        Post-fix: we see ~50.
        """
        import statistics

        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(500)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        means = [c.kwargs["mean"] for c in mock_generate.call_args_list]
        empirical_mean = statistics.fmean(means)
        empirical_stddev = statistics.stdev(means)

        assert 490 <= empirical_mean <= 510, f"mean drift: {empirical_mean:.2f}"
        assert 42 <= empirical_stddev <= 58, (
            f"stddev={empirical_stddev:.2f} — expected ~50. "
            f"If ~70 then the double-sampling bug regressed. "
            f"If ~0 then turn-level sampling was dropped."
        )

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_lognormal_isl_passes_sampled_values_not_mean(
        self, mock_generate, mock_tokenizer
    ):
        """LogNormal must produce varied means — a collapse to 1000 would
        mean the turn-level sample was removed (Option-A regression)."""
        import statistics

        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 1000, "median": 400}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(300)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        means = [c.kwargs["mean"] for c in mock_generate.call_args_list]
        assert statistics.stdev(means) > 100, (
            "LogNormal ISL collapsed — turn-level sampling missing?"
        )
        assert statistics.median(means) < 800

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_multimodal_isl_reaches_both_peaks_through_flow(
        self, mock_generate, mock_tokenizer
    ):
        """Multimodal ISL must reach both peaks by the time it hits generate()."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1,
            prompts={
                "isl": {
                    "peaks": [
                        {"mean": 100, "stddev": 5, "weight": 50},
                        {"mean": 2000, "stddev": 50, "weight": 50},
                    ]
                },
                "osl": 64,
            },
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(200)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        means = [c.kwargs["mean"] for c in mock_generate.call_args_list]
        low = sum(1 for m in means if m < 500)
        high = sum(1 for m in means if m > 1500)
        middle = sum(1 for m in means if 500 <= m <= 1500)
        assert low > 50, f"low peak underpopulated: {low}"
        assert high > 50, f"high peak underpopulated: {high}"
        assert middle < 20, f"{middle} samples in gap — shape lost through flow"

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_empirical_isl_only_configured_values_through_flow(
        self, mock_generate, mock_tokenizer
    ):
        """Empirical distribution values must flow through unchanged."""
        mock_generate.return_value = "text"
        allowed = {128, 512, 2048}
        config = _make_config(
            entries=1,
            prompts={
                "isl": {
                    "points": [
                        {"value": 128, "weight": 1},
                        {"value": 512, "weight": 1},
                        {"value": 2048, "weight": 1},
                    ]
                },
                "osl": 64,
            },
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(200)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        means = {c.kwargs["mean"] for c in mock_generate.call_args_list}
        assert means == allowed, f"mean set {means} != configured {allowed}"

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_prefix_prompt_does_not_cause_resampling(
        self, mock_generate, mock_tokenizer
    ):
        """Prefix prompts don't perturb the stddev=0 contract."""
        mock_generate.return_value = "content"
        with patch(
            "aiperf.dataset.generator.prompt.PromptGenerator.get_random_prefix_prompt",
            return_value="pfx:",
        ):
            config = _make_config(
                entries=1,
                prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64},
                prefix_prompts={"pool_size": 2, "length": 10},
            )
            composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
            composer._generate_text_payloads(Turn(), is_first=True)

        for call in mock_generate.call_args_list:
            assert call.kwargs["stddev"] == 0

    def test_isl_osl_paired_consistently_across_text_and_max_tokens(
        self, mock_tokenizer
    ):
        """ISL used for prompt and OSL used for max_tokens must come from the
        same cached turn sample (no drift between the two uses)."""
        config = _make_config(
            entries=1,
            prompts={
                "isl": {"mean": 500, "stddev": 50},
                "osl": {"mean": 100, "stddev": 20},
            },
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turn = Turn()
        with patch(
            "aiperf.dataset.generator.prompt.PromptGenerator.generate",
            return_value="text",
        ) as mock_generate:
            composer._generate_text_payloads(turn, is_first=True)
            turn_id = id(turn)
            cached_pair = composer._turn_sequence_cache[turn_id]

        mean_used = mock_generate.call_args.kwargs["mean"]
        assert mean_used == cached_pair[0]

        composer._set_max_tokens(turn)
        assert turn.max_tokens == int(cached_pair[1])

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_normal_stddev_zero_is_deterministic_through_flow(
        self, mock_generate, mock_tokenizer
    ):
        """stddev=0 in config must produce the mean literally at generate()."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 256, "stddev": 0}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(20)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        for call in mock_generate.call_args_list:
            assert call.kwargs == {"mean": 256, "stddev": 0}

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_sequence_distribution_path_also_uses_stddev_zero(
        self, mock_generate, mock_tokenizer
    ):
        """When a joint sequence_distribution is configured, the ISL still flows
        through generate() with stddev=0 (no double sampling here either)."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1,
            prompts={
                "isl": {"mean": 500, "stddev": 50},  # ignored when seq_dist is set
                "osl": 64,
                "sequence_distribution": [
                    {"isl": 100, "osl": 25, "probability": 50},
                    {"isl": 200, "osl": 50, "probability": 50},
                ],
            },
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(50)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        seen_means = set()
        for call in mock_generate.call_args_list:
            assert call.kwargs["stddev"] == 0
            seen_means.add(call.kwargs["mean"])
        assert seen_means == {100, 200}, f"seq_dist ISLs not respected: {seen_means}"


# ============================================================================
# Adversarial ISL-sampling tests
#
# These tests attack the invariants from unusual angles: cache tampering,
# config mutation, turn-id reuse, extreme values, exotic distribution
# shapes, error propagation, and cross-composer isolation. Every one of
# them should pass with the correct implementation and fail in a distinct,
# informative way under a plausible regression.
# ============================================================================


class TestIslAdversarial:
    """Break things on purpose. Assert the flow holds anyway."""

    # ------------------------------------------------------------------
    # Cache-tampering attacks
    # ------------------------------------------------------------------

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_pre_populated_cache_is_respected(
        self, mock_generate, mock_tokenizer
    ):
        """Pre-populating the cache must override sampling — the composer
        must honor the cached (ISL, OSL) pair rather than resample."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turn = Turn()
        composer._turn_sequence_cache[id(turn)] = (9999, 777)

        composer._generate_text_payloads(turn, is_first=True)

        mock_generate.assert_called_once_with(mean=9999, stddev=0)

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_repeated_calls_same_turn_reuse_isl(
        self, mock_generate, mock_tokenizer
    ):
        """Calling _generate_text_payloads twice on the SAME Turn must reuse
        the cached ISL — no second sample, no drift."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 100}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turn = Turn()
        composer._generate_text_payloads(turn, is_first=True)
        first_mean = mock_generate.call_args.kwargs["mean"]

        composer._generate_text_payloads(turn, is_first=False)
        second_mean = mock_generate.call_args.kwargs["mean"]

        assert first_mean == second_mean, (
            f"ISL drifted on repeated call: {first_mean} -> {second_mean}"
        )
        assert mock_generate.call_count == 2

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_clear_cache_forces_resample(
        self, mock_generate, mock_tokenizer
    ):
        """Clearing the cache between calls must cause a fresh sample on
        the same turn — distinct mean (with overwhelming probability)."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 200}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turn = Turn()
        composer._generate_text_payloads(turn, is_first=True)
        first_mean = mock_generate.call_args.kwargs["mean"]

        composer._clear_turn_cache(id(turn))
        composer._generate_text_payloads(turn, is_first=True)
        second_mean = mock_generate.call_args.kwargs["mean"]

        assert first_mean != second_mean, (
            "Cache clear did not trigger resample (cache stickiness bug?)"
        )

    # ------------------------------------------------------------------
    # Config-mutation attacks
    # ------------------------------------------------------------------

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_mutating_isl_stddev_after_init_keeps_stddev_zero(
        self, mock_generate, mock_tokenizer
    ):
        """Even if someone mutates prompts.isl.stddev after composer init,
        the stddev kwarg to generate() must stay 0. Guards against a future
        regression that re-reads the attribute on every call."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        run = _make_run(config)
        composer = SyntheticDatasetComposer(run, mock_tokenizer)

        composer.dataset_config.prompts.isl.stddev = 999_999

        composer._generate_text_payloads(Turn(), is_first=True)
        assert mock_generate.call_args.kwargs["stddev"] == 0

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_swap_isl_to_normal_with_huge_stddev_still_zero(
        self, mock_generate, mock_tokenizer
    ):
        """Swap the entire isl distribution to a NormalDistribution with an
        absurd stddev — the generate() stddev kwarg must stay 0."""
        from aiperf.config.distributions import NormalDistribution

        mock_generate.return_value = "text"
        config = _make_config(entries=1, prompts={"isl": 256, "osl": 64})
        run = _make_run(config)
        composer = SyntheticDatasetComposer(run, mock_tokenizer)

        composer.dataset_config.prompts.isl = NormalDistribution(
            mean=256, stddev=10_000
        )

        composer._generate_text_payloads(Turn(), is_first=True)
        assert mock_generate.call_args.kwargs["stddev"] == 0

    # ------------------------------------------------------------------
    # Extreme / degenerate values
    # ------------------------------------------------------------------

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_extreme_stddev_truncates_to_min_one(
        self, mock_generate, mock_tokenizer
    ):
        """N(mean=1, stddev=10000): negative draws are clipped to min 1. No
        zero or negative ISLs should ever flow into generate()."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 1, "stddev": 10_000}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(200)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        means = [c.kwargs["mean"] for c in mock_generate.call_args_list]
        assert all(m >= 1 for m in means), (
            f"zero or negative ISL leaked: min={min(means)}"
        )

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_isl_equal_to_one_flows_cleanly(
        self, mock_generate, mock_tokenizer
    ):
        """ISL=1 is the smallest legal prompt length. Must flow end-to-end."""
        mock_generate.return_value = "text"
        config = _make_config(entries=1, prompts={"isl": 1, "osl": 64})
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        composer._generate_text_payloads(Turn(), is_first=True)
        mock_generate.assert_called_once_with(mean=1, stddev=0)

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_very_large_isl(self, mock_generate, mock_tokenizer):
        """Extreme but legal ISL (~1M tokens) flows through without overflow."""
        mock_generate.return_value = "text"
        config = _make_config(entries=1, prompts={"isl": 1_000_000, "osl": 64})
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        composer._generate_text_payloads(Turn(), is_first=True)
        mock_generate.assert_called_once_with(mean=1_000_000, stddev=0)

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_huge_batch_size_one_sample(
        self, mock_generate, mock_tokenizer
    ):
        """batch_size=500 produces 500 generate() calls, all sharing one ISL."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1,
            prompts={
                "isl": {"mean": 500, "stddev": 100},
                "osl": 64,
                "batch_size": 500,
            },
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        composer._generate_text_payloads(Turn(), is_first=True)

        assert mock_generate.call_count == 500
        means = {c.kwargs["mean"] for c in mock_generate.call_args_list}
        stddevs = {c.kwargs["stddev"] for c in mock_generate.call_args_list}
        assert len(means) == 1
        assert stddevs == {0}

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_multimodal_99_1_weight_imbalance_still_hits_minority(
        self, mock_generate, mock_tokenizer
    ):
        """With 99:1 weights over many samples, the minority peak must still
        appear at least once (proves we aren't silently collapsing to majority)."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1,
            prompts={
                "isl": {
                    "peaks": [
                        {"mean": 100, "stddev": 1, "weight": 99},
                        {"mean": 5000, "stddev": 1, "weight": 1},
                    ]
                },
                "osl": 64,
            },
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(1000)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        means = [c.kwargs["mean"] for c in mock_generate.call_args_list]
        assert any(m > 2000 for m in means), (
            "minority peak never sampled — weight logic broken?"
        )
        low = sum(1 for m in means if m < 500)
        assert low > 900

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_empirical_single_point_is_deterministic(
        self, mock_generate, mock_tokenizer
    ):
        """Empirical distribution with exactly one point is effectively Fixed."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1,
            prompts={
                "isl": {"points": [{"value": 777, "weight": 1}]},
                "osl": 64,
            },
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(20)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        for call in mock_generate.call_args_list:
            assert call.kwargs == {"mean": 777, "stddev": 0}

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_empirical_tiny_weight_barely_sampled(
        self, mock_generate, mock_tokenizer
    ):
        """A point with an infinitesimal weight must be sampled rarely (or
        never) relative to peers — proves weights gate sampling, not just
        membership. Pydantic forbids weight=0 outright, so we probe with 1e-6."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1,
            prompts={
                "isl": {
                    "points": [
                        {"value": 100, "weight": 1_000_000},
                        {"value": 999, "weight": 0.000_001},
                        {"value": 200, "weight": 1_000_000},
                    ]
                },
                "osl": 64,
            },
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        turns = [Turn() for _ in range(200)]
        for t in turns:
            composer._generate_text_payloads(t, is_first=True)

        means = [c.kwargs["mean"] for c in mock_generate.call_args_list]
        hits_999 = sum(1 for m in means if m == 999)
        hits_100_or_200 = sum(1 for m in means if m in (100, 200))
        assert hits_999 <= 1, (
            f"tiny-weight point oversampled: {hits_999}/200 hits on value 999"
        )
        assert hits_100_or_200 >= 195

    # ------------------------------------------------------------------
    # Error propagation
    # ------------------------------------------------------------------

    def test_adversarial_sample_int_error_propagates_does_not_silently_fallback(
        self, mock_tokenizer
    ):
        """If isl.sample_int raises, the error surfaces — no silent fallback
        to mean or default. A swallowed exception would mask config bugs."""
        from aiperf.config.distributions import NormalDistribution

        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        run = _make_run(config)
        composer = SyntheticDatasetComposer(run, mock_tokenizer)

        with (
            patch.object(
                NormalDistribution,
                "sample_int",
                side_effect=RuntimeError("sampling broken"),
            ),
            pytest.raises(RuntimeError, match="sampling broken"),
        ):
            composer._generate_text_payloads(Turn(), is_first=True)

    # ------------------------------------------------------------------
    # Cross-composer isolation
    # ------------------------------------------------------------------

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_two_composers_do_not_share_cache(
        self, mock_generate, mock_tokenizer
    ):
        """Separate composer instances keep separate turn caches."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        composer_a = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        composer_b = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)

        composer_a._turn_sequence_cache[42] = (111, 22)
        assert 42 not in composer_b._turn_sequence_cache

    # ------------------------------------------------------------------
    # Multi-turn conversation correctness
    # ------------------------------------------------------------------

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_multiturn_conversation_isl_per_turn_independent(
        self, mock_generate, mock_tokenizer
    ):
        """In a multi-turn conversation each turn must draw its own ISL.
        Guards against a regression that caches at conversation scope."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=5,
            prompts={"isl": {"mean": 500, "stddev": 100}, "osl": 64},
            turns={"mean": 4, "stddev": 0},
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        composer.create_dataset()

        assert mock_generate.call_count == 20  # 5 conversations × 4 turns
        means = [c.kwargs["mean"] for c in mock_generate.call_args_list]
        assert len(set(means)) >= 10
        assert {c.kwargs["stddev"] for c in mock_generate.call_args_list} == {0}

    # ------------------------------------------------------------------
    # End-to-end (create_dataset) statistical invariant
    # ------------------------------------------------------------------

    def test_adversarial_end_to_end_create_dataset_stddev_not_inflated(
        self, mock_tokenizer
    ):
        """Drive create_dataset() end-to-end. The ISL stream passed to
        generate() must have empirical stddev ≈ 50, not ≈ 70.7."""
        import statistics

        config = _make_config(
            entries=500, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        with patch(
            "aiperf.dataset.generator.prompt.PromptGenerator.generate",
            return_value="text",
        ) as mg:
            conversations = composer.create_dataset()
            means = [c.kwargs["mean"] for c in mg.call_args_list]

        assert len(conversations) == 500
        empirical_stddev = statistics.stdev(means)
        assert 42 <= empirical_stddev <= 58, (
            f"create_dataset() ISL stddev={empirical_stddev:.2f} — "
            f"expected ~50, regressed if ~70."
        )

    # ------------------------------------------------------------------
    # Kwarg hygiene
    # ------------------------------------------------------------------

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_generate_called_with_kwargs_only(
        self, mock_generate, mock_tokenizer
    ):
        """Composer must invoke generate() by keyword, not positional.
        Positional args are fragile to signature reorderings."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        composer._generate_text_payloads(Turn(), is_first=True)

        args, kwargs = mock_generate.call_args
        assert args == (), f"unexpected positional args: {args}"
        assert set(kwargs.keys()) == {"mean", "stddev"}

    @patch("aiperf.dataset.generator.prompt.PromptGenerator.generate")
    def test_adversarial_generate_stddev_is_literal_int_zero_not_none(
        self, mock_generate, mock_tokenizer
    ):
        """stddev=0 must be an int 0, not None/False/other-falsy. Downstream
        `sample_positive_normal_integer` checks `stddev <= 0` — None would crash."""
        mock_generate.return_value = "text"
        config = _make_config(
            entries=1, prompts={"isl": {"mean": 500, "stddev": 50}, "osl": 64}
        )
        composer = SyntheticDatasetComposer(_make_run(config), mock_tokenizer)
        composer._generate_text_payloads(Turn(), is_first=True)

        stddev = mock_generate.call_args.kwargs["stddev"]
        assert stddev == 0
        assert stddev is not None
        assert isinstance(stddev, int)

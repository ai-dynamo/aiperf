# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from aiperf.common.enums import CustomDatasetType
from aiperf.common.models import Image, Text
from aiperf.services.dataset.loader.models import SingleTurn
from aiperf.services.dataset.loader.single_turn import SingleTurnDatasetLoader


class TestSingleTurn:
    """Basic functionality tests for SingleTurn model."""

    def test_create_with_text_only(self):
        """Test creating SingleTurn with text."""
        data = SingleTurn(text="What is deep learning?")

        assert data.text == "What is deep learning?"
        assert data.texts is None
        assert data.type == CustomDatasetType.SINGLE_TURN

    def test_create_with_multimodal_data(self):
        """Test creating SingleTurn with text and image."""
        data = SingleTurn(
            text="What is in the image?",
            image="/path/to/image.png",
            audio="/path/to/audio.wav",
        )

        assert data.text == "What is in the image?"
        assert data.texts is None
        assert data.image == "/path/to/image.png"
        assert data.images is None
        assert data.audio == "/path/to/audio.wav"
        assert data.audios is None

    def test_create_with_batched_inputs(self):
        """Test creating SingleTurn with batched inputs."""
        data = SingleTurn(
            texts=["What is the weather today?", "What is deep learning?"],
            images=["/path/to/image1.png", "/path/to/image2.png"],
        )

        assert data.texts == ["What is the weather today?", "What is deep learning?"]
        assert data.images == ["/path/to/image1.png", "/path/to/image2.png"]
        assert data.audios is None

    def test_create_with_fixed_schedule(self):
        """Test creating SingleTurn with fixed schedule (timestamp)."""
        data = SingleTurn(text="What is deep learning?", timestamp=1000)

        assert data.text == "What is deep learning?"
        assert data.timestamp == 1000
        assert data.delay is None

    def test_create_with_delay(self):
        """Test creating SingleTurn with delay."""
        data = SingleTurn(text="Who are you?", delay=1234)

        assert data.text == "Who are you?"
        assert data.delay == 1234
        assert data.timestamp is None

    def test_create_with_full_featured_version(self):
        """Test creating SingleTurn with full-featured version."""
        data = SingleTurn(
            texts=[
                Text(name="text_field_A", content=["Hello", "World"]),
                Text(name="text_field_B", content=["Hi there"]),
            ],
            images=[
                Image(name="image_field_A", content=["/path/1.png", "/path/2.png"]),
                Image(name="image_field_B", content=["/path/3.png"]),
            ],
        )

        assert len(data.texts) == 2
        assert len(data.images) == 2
        assert data.audios is None

        assert data.texts[0].name == "text_field_A"
        assert data.texts[0].content == ["Hello", "World"]
        assert data.texts[1].name == "text_field_B"
        assert data.texts[1].content == ["Hi there"]

        assert data.images[0].name == "image_field_A"
        assert data.images[0].content == ["/path/1.png", "/path/2.png"]
        assert data.images[1].name == "image_field_B"
        assert data.images[1].content == ["/path/3.png"]

    def test_validation_errors(self):
        """Test that at least one modality must be provided."""
        # No modality provided
        with pytest.raises(ValueError):
            SingleTurn()

    @pytest.mark.parametrize(
        "text, texts, image, images, audio, audios, timestamp, delay",
        [
            ("foo", ["bar"], None, None, None, None, None, None),  # text and texts
            (None, None, "foo", ["bar"], None, None, None, None),  # image and images
            (None, None, None, None, "foo", ["bar"], None, None),  # audio and audios
            (None, None, None, None, None, None, 1000, 500),  # timestamp and delay
        ],
    )
    def test_validation_mutual_exclusion(
        self, text, texts, image, images, audio, audios, timestamp, delay
    ):
        """Test that mutual exclusion among fields is preserved."""
        with pytest.raises(ValueError):
            SingleTurn(
                text=text,
                texts=texts,
                image=image,
                images=images,
                audio=audio,
                audios=audios,
                timestamp=timestamp,
                delay=delay,
            )


class TestSingleTurnDatasetLoader:
    """Basic functionality tests for SingleTurnDatasetLoader."""

    def test_load_dataset_basic_functionality(self, create_jsonl_file):
        """Test basic JSONL file loading."""
        content = [
            '{"text": "What is deep learning?"}',
            '{"text": "What is in the image?", "image": "/path/to/image.png"}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        dataset = loader.load_dataset()

        assert isinstance(dataset, dict)
        assert len(dataset) == 2

        # Check that each session has single turn
        for _, turns in dataset.items():
            assert len(turns) == 1

        turn1, turn2 = list(dataset.values())
        assert turn1[0].text == "What is deep learning?"
        assert turn1[0].image is None
        assert turn1[0].audio is None

        assert turn2[0].text == "What is in the image?"
        assert turn2[0].image == "/path/to/image.png"
        assert turn2[0].audio is None

    def test_load_dataset_skips_empty_lines(self, create_jsonl_file):
        """Test that empty lines are skipped."""
        content = [
            '{"text": "Hello"}',
            "",  # Empty line
            '{"text": "World"}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        dataset = loader.load_dataset()

        assert len(dataset) == 2  # Should skip empty line

    def test_load_dataset_with_batched_inputs(self, create_jsonl_file):
        """Test loading dataset with batched inputs."""
        content = [
            '{"texts": ["What is the weather?", "What is AI?"], "images": ["/path/1.png", "/path/2.png"]}',
            '{"texts": ["Summarize the podcast", "What is audio about?"], "audios": ["/path/3.wav", "/path/4.wav"]}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        dataset = loader.load_dataset()

        # Check that there are two sessions
        assert len(dataset) == 2

        turn1, turn2 = list(dataset.values())
        assert turn1[0].texts == ["What is the weather?", "What is AI?"]
        assert turn1[0].images == ["/path/1.png", "/path/2.png"]
        assert turn1[0].audios is None

        assert turn2[0].texts == ["Summarize the podcast", "What is audio about?"]
        assert turn2[0].images is None
        assert turn2[0].audios == ["/path/3.wav", "/path/4.wav"]

    def test_load_dataset_with_timestamp(self, create_jsonl_file):
        """Test loading dataset with timestamp field."""
        content = [
            '{"text": "What is deep learning?", "timestamp": 1000}',
            '{"text": "Who are you?", "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        dataset = loader.load_dataset()

        assert len(dataset) == 2

        turn1, turn2 = list(dataset.values())
        assert turn1[0].text == "What is deep learning?"
        assert turn1[0].timestamp == 1000
        assert turn1[0].delay is None

        assert turn2[0].text == "Who are you?"
        assert turn2[0].timestamp == 2000
        assert turn2[0].delay is None

    def test_load_dataset_with_delay(self, create_jsonl_file):
        """Test loading dataset with delay field."""
        content = [
            '{"text": "What is deep learning?", "delay": 0}',
            '{"text": "Who are you?", "delay": 1234}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        dataset = loader.load_dataset()

        assert len(dataset) == 2

        turn1, turn2 = list(dataset.values())
        assert turn1[0].text == "What is deep learning?"
        assert turn1[0].delay == 0
        assert turn1[0].timestamp is None

        assert turn2[0].text == "Who are you?"
        assert turn2[0].delay == 1234
        assert turn2[0].timestamp is None

    def test_load_dataset_with_full_featured_version(self, create_jsonl_file):
        """Test loading dataset with full-featured version."""

        content = [
            json.dumps(
                {
                    "texts": [
                        {"name": "text_field_A", "content": ["Hello", "World"]},
                        {"name": "text_field_B", "content": ["Hi there"]},
                    ],
                    "images": [
                        {
                            "name": "image_field_A",
                            "content": ["/path/1.png", "/path/2.png"],
                        },
                        {"name": "image_field_B", "content": ["/path/3.png"]},
                    ],
                }
            )
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        dataset = loader.load_dataset()

        assert len(dataset) == 1

        turn = list(dataset.values())[0]
        assert len(turn[0].texts) == 2
        assert len(turn[0].images) == 2

        assert turn[0].texts[0].name == "text_field_A"
        assert turn[0].texts[0].content == ["Hello", "World"]
        assert turn[0].texts[1].name == "text_field_B"
        assert turn[0].texts[1].content == ["Hi there"]

        assert turn[0].images[0].name == "image_field_A"
        assert turn[0].images[0].content == ["/path/1.png", "/path/2.png"]
        assert turn[0].images[1].name == "image_field_B"
        assert turn[0].images[1].content == ["/path/3.png"]

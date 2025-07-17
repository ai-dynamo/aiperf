# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.dataset_models import Image, Text
from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.loader.models import SingleTurnCustomData
from aiperf.services.dataset.loader.single_turn import SingleTurnDatasetLoader


class TestSingleTurnCustomData:
    """Basic functionality tests for SingleTurnCustomData model."""

    def test_create_with_text_only(self):
        """Test creating SingleTurnCustomData with text."""
        data = SingleTurnCustomData(text="What is deep learning?")

        assert data.text == "What is deep learning?"
        assert data.image is None
        assert data.audio is None
        assert data.type == CustomDatasetType.SINGLE_TURN

    def test_create_with_multimodal_data(self):
        """Test creating SingleTurnCustomData with text and image."""
        data = SingleTurnCustomData(
            text="What is in the image?",
            image="/path/to/image.png",
            audio="/path/to/audio.wav",
        )

        assert data.text == "What is in the image?"
        assert data.image == "/path/to/image.png"
        assert data.audio == "/path/to/audio.wav"

    def test_create_with_batched_inputs(self):
        """Test creating SingleTurnCustomData with batched inputs."""
        data = SingleTurnCustomData(
            text=["What is the weather today?", "What is deep learning?"],
            image=["/path/to/image1.png", "/path/to/image2.png"],
        )

        assert data.text == ["What is the weather today?", "What is deep learning?"]
        assert data.image == ["/path/to/image1.png", "/path/to/image2.png"]
        assert data.audio is None

    def test_create_with_fixed_schedule(self):
        """Test creating SingleTurnCustomData with fixed schedule (timestamp)."""
        data = SingleTurnCustomData(text="What is deep learning?", timestamp=1000)

        assert data.text == "What is deep learning?"
        assert data.timestamp == 1000
        assert data.delay is None

    def test_create_with_delay(self):
        """Test creating SingleTurnCustomData with delay."""
        data = SingleTurnCustomData(text="Who are you?", delay=1234)

        assert data.text == "Who are you?"
        assert data.delay == 1234
        assert data.timestamp is None

    def test_create_with_full_featured_version(self):
        """Test creating SingleTurnCustomData with full-featured version."""
        data = SingleTurnCustomData(
            text=[
                Text(name="text_field_A", content=["Hello", "World"]),
                Text(name="text_field_B", content=["Hi there"]),
            ],
            image=[
                Image(name="image_field_A", content=["/path/1.png", "/path/2.png"]),
                Image(name="image_field_B", content=["/path/3.png"]),
            ],
        )

        assert len(data.text) == 2
        assert len(data.image) == 2
        assert data.audio is None

        assert data.text[0].name == "text_field_A"
        assert data.text[0].content == ["Hello", "World"]
        assert data.text[1].name == "text_field_B"
        assert data.text[1].content == ["Hi there"]

        assert data.image[0].name == "image_field_A"
        assert data.image[0].content == ["/path/1.png", "/path/2.png"]
        assert data.image[1].name == "image_field_B"
        assert data.image[1].content == ["/path/3.png"]

    def test_validation_errors(self):
        """Test that at least one modality must be provided."""
        # No modality provided
        with pytest.raises(ValueError):
            SingleTurnCustomData()

        # Timestamp and delay cannot be set together
        with pytest.raises(ValueError):
            SingleTurnCustomData(
                text="What is deep learning?", timestamp=1000, delay=1234
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
        result = loader.load_dataset()

        assert isinstance(result, dict)
        assert len(result) == 2

        # Check that each session has one turn
        for _, turns in result.items():
            assert len(turns) == 1

        data1, data2 = list(result.values())
        assert data1.text == "What is deep learning?"
        assert data1.image is None
        assert data1.audio is None

        assert data2.text == "What is in the image?"
        assert data2.image == "/path/to/image.png"
        assert data2.audio is None

    def test_load_dataset_skips_empty_lines(self, create_jsonl_file):
        """Test that empty lines are skipped."""
        content = [
            '{"text": "Hello"}',
            "",  # Empty line
            '{"text": "World"}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        result = loader.load_dataset()

        assert len(result) == 2  # Should skip empty line

    def test_load_dataset_with_batched_inputs(self, create_jsonl_file):
        """Test loading dataset with batched inputs."""
        content = [
            '{"text": ["What is the weather?", "What is AI?"], "image": ["/path/1.png", "/path/2.png"]}'
            '{"text": ["Summarize the podcast", "What is audio about?"], "audio": ["/path/3.wav", "/path/4.wav"]}'
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        result = loader.load_dataset()

        # Check that there are two sessions
        assert len(result) == 2

        data1, data2 = list(result.values())
        assert data1.text == ["What is the weather?", "What is AI?"]
        assert data1.image == ["/path/1.png", "/path/2.png"]
        assert data1.audio is None

        assert data2.text == ["Summarize the podcast", "What is audio about?"]
        assert data2.image is None
        assert data2.audio == ["/path/3.wav", "/path/4.wav"]

    def test_load_dataset_with_timestamp(self, create_jsonl_file):
        """Test loading dataset with timestamp field."""
        content = [
            '{"text": "What is deep learning?", "timestamp": 1000}',
            '{"text": "Who are you?", "timestamp": 2000}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        result = loader.load_dataset()

        assert len(result) == 2

        data1, data2 = list(result.values())
        assert data1.text == "What is deep learning?"
        assert data1.timestamp == 1000
        assert data1.delay is None

        assert data2.text == "Who are you?"
        assert data2.timestamp == 2000
        assert data2.delay is None

    def test_load_dataset_with_delay(self, create_jsonl_file):
        """Test loading dataset with delay field."""
        content = [
            '{"text": "What is deep learning?", "delay": 0}',
            '{"text": "Who are you?", "delay": 1234}',
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        result = loader.load_dataset()

        assert len(result) == 2

        data1, data2 = list(result.values())
        assert data1.text == "What is deep learning?"
        assert data1.delay == 0
        assert data1.timestamp is None

        assert data2.text == "Who are you?"
        assert data2.delay == 1234
        assert data2.timestamp is None

    def test_load_dataset_with_full_featured_version(self, create_jsonl_file):
        """Test loading dataset with full-featured version."""

        content = [
            """{
                "text": [
                    {"name": "text_field_A", "content": ["Hello", "World"]},
                    {"name": "text_field_B", "content": ["Hi there"]}
                ],
                "image": [
                    {"name": "image_field_A", "content": ["/path/1.png", "/path/2.png"]},
                    {"name": "image_field_B", "content": ["/path/3.png"]}
                ]
            }"""
        ]
        filename = create_jsonl_file(content)

        loader = SingleTurnDatasetLoader(filename)
        result = loader.load_dataset()

        assert len(result) == 1

        data = list(result.values())[0]
        assert len(data.text) == 2
        assert len(data.image) == 2
        assert data.audio is None

        assert data.text[0].name == "text_field_A"
        assert data.text[0].content == ["Hello", "World"]
        assert data.text[1].name == "text_field_B"
        assert data.text[1].content == ["Hi there"]

        assert data.image[0].name == "image_field_A"
        assert data.image[0].content == ["/path/1.png", "/path/2.png"]
        assert data.image[1].name == "image_field_B"
        assert data.image[1].content == ["/path/3.png"]

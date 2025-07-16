# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.loader.models import SingleTurnCustomData
from aiperf.services.dataset.loader.single_turn import SingleTurnDatasetLoader


@pytest.fixture
def create_jsonl_file():
    """Create a temporary JSONL file with custom content."""
    filename = None

    def _create_file(content_lines):
        nonlocal filename
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in content_lines:
                f.write(line + "\n")
            filename = f.name
        return filename

    yield _create_file

    # Cleanup all created files
    if filename:
        Path(filename).unlink(missing_ok=True)


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

    def test_validation_requires_at_least_one_modality(self):
        """Test that at least one modality must be provided."""
        with pytest.raises(ValueError):
            SingleTurnCustomData()


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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.loader.models import RandomPool
from aiperf.services.dataset.loader.random_pool import RandomPoolDatasetLoader


class TestRandomPool:
    """Tests for RandomPool model validation and functionality."""

    def test_create_with_text_only(self):
        """Test creating RandomPool with simple text."""
        data = RandomPool(text="What is machine learning?")

        assert data.text == "What is machine learning?"
        assert data.texts is None
        assert data.type == CustomDatasetType.RANDOM_POOL

    def test_create_with_multimodal_data(self):
        """Test creating RandomPool with multiple modalities."""
        data = RandomPool(
            text="Describe this audio",
            image="/path/to/chart.png",
            audio="/path/to/recording.wav",
        )

        assert data.text == "Describe this audio"
        assert data.texts is None
        assert data.image == "/path/to/chart.png"
        assert data.images is None
        assert data.audio == "/path/to/recording.wav"
        assert data.audios is None

    def test_create_with_batched_inputs(self):
        """Test creating RandomPool with batched content."""
        data = RandomPool(
            texts=["What is AI?", "Explain neural networks"],
            images=["/path/image1.jpg", "/path/image2.jpg"],
            audios=["/path/audio1.wav", "/path/audio2.wav"],
        )

        assert data.text is None
        assert data.texts == ["What is AI?", "Explain neural networks"]
        assert data.image is None
        assert data.images == ["/path/image1.jpg", "/path/image2.jpg"]
        assert data.audio is None
        assert data.audios == ["/path/audio1.wav", "/path/audio2.wav"]

    def test_validation_at_least_one_modality_required(self):
        """Test that at least one modality must be provided."""
        with pytest.raises(ValueError):
            RandomPool()

    @pytest.mark.parametrize(
        "text,texts,image,images,audio,audios",
        [
            ("hello", ["world"], None, None, None, None),  # text and texts
            (None, None, "img.png", ["img2.png"], None, None),  # image and images
            (None, None, None, None, "audio.wav", ["audio2.wav"]),  # audio and audios
        ],
    )
    def test_validation_mutually_exclusive_fields(
        self, text, texts, image, images, audio, audios
    ):
        """Test that mutually exclusive fields cannot be set together."""
        with pytest.raises(ValueError):
            RandomPool(
                text=text,
                texts=texts,
                image=image,
                images=images,
                audio=audio,
                audios=audios,
            )


class TestRandomPoolDatasetLoader:
    """Tests for RandomPoolDatasetLoader functionality."""

    def test_load_simple_single_file(self, create_jsonl_file):
        """Test loading from a single file with simple content."""
        content = [
            '{"text": "What is deep learning?"}',
            '{"text": "Explain neural networks", "image": "/chart.png"}',
        ]
        filepath = create_jsonl_file(content)

        loader = RandomPoolDatasetLoader(filepath)
        dataset = loader.load_dataset()

        filename = Path(filepath).name
        assert isinstance(dataset, dict)
        assert len(dataset) == 1  # Single file loaded
        assert filename in dataset

        dataset_pool = dataset[filename]
        assert len(dataset_pool) == 2
        assert dataset_pool[0].text == "What is deep learning?"
        assert dataset_pool[1].text == "Explain neural networks"
        assert dataset_pool[1].image == "/chart.png"

    def test_load_multimodal_single_file(self, create_jsonl_file):
        """Test loading multimodal content from single file."""
        content = [
            '{"text": "Analyze this image", "image": "/data.png"}',
            '{"text": "Transcribe audio", "audio": "/recording.wav"}',
            '{"texts": ["Query 1", "Query 2"], "images": ["/img1.jpg", "/img2.jpg"]}',
        ]
        filepath = create_jsonl_file(content)

        loader = RandomPoolDatasetLoader(filepath)
        dataset = loader.load_dataset()

        filename = Path(filepath).name
        dataset_pool = dataset[filename]
        assert len(dataset_pool) == 3
        assert dataset_pool[0].text == "Analyze this image"
        assert dataset_pool[0].image == "/data.png"
        assert dataset_pool[1].audio == "/recording.wav"
        assert dataset_pool[2].texts == ["Query 1", "Query 2"]
        assert dataset_pool[2].images == ["/img1.jpg", "/img2.jpg"]

    def test_load_dataset_skips_empty_lines(self, create_jsonl_file):
        """Test that empty lines are skipped during loading."""
        content = [
            '{"text": "First entry"}',
            "",  # Empty line
            '{"text": "Second entry"}',
            "   ",  # Whitespace only
            '{"text": "Third entry"}',
        ]
        filepath = create_jsonl_file(content)

        loader = RandomPoolDatasetLoader(filepath)
        dataset = loader.load_dataset()

        filename = Path(filepath).name
        dataset_pool = dataset[filename]
        assert len(dataset_pool) == 3  # Should skip empty lines

    def test_load_directory_with_multiple_files(self):
        """Test loading from directory with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create first file - queries
            queries_file = temp_path / "queries.jsonl"
            with open(queries_file, "w") as f:
                f.write(
                    '{"texts": [{"name": "query", "contents": ["Who are you?"]}]}\n'
                )
                f.write('{"texts": [{"name": "query", "contents": ["What is AI?"]}]}\n')

            # Create second file - passages
            passages_file = temp_path / "passages.jsonl"
            with open(passages_file, "w") as f:
                f.write(
                    '{"texts": [{"name": "passage", "contents": ["I am an AI assistant."]}]}\n'
                )
                f.write(
                    '{"texts": [{"name": "passage", "contents": ["AI is artificial intelligence."]}]}\n'
                )

            # Create third file - images
            images_file = temp_path / "images.jsonl"
            with open(images_file, "w") as f:
                f.write(
                    '{"images": [{"name": "image", "content": ["/path/to/image1.png"]}]}\n'
                )
                f.write(
                    '{"images": [{"name": "image", "content": ["/path/to/image2.png"]}]}\n'
                )

            loader = RandomPoolDatasetLoader(str(temp_path))
            dataset = loader.load_dataset()

            assert len(dataset) == 3
            assert "queries.jsonl" in dataset
            assert "passages.jsonl" in dataset
            assert "images.jsonl" in dataset

            # Check queries file content
            queries_pool = dataset["queries.jsonl"]
            assert len(queries_pool) == 2
            assert all(item.texts[0].name == "query" for item in queries_pool)
            assert queries_pool[0].texts[0].contents == ["Who are you?"]
            assert queries_pool[1].texts[0].contents == ["What is AI?"]

            # Check passages file content
            passages_pool = dataset["passages.jsonl"]
            assert len(passages_pool) == 2
            assert all(item.texts[0].name == "passage" for item in passages_pool)
            assert passages_pool[0].texts[0].contents == ["I am an AI assistant."]
            assert passages_pool[1].texts[0].contents == [
                "AI is artificial intelligence."
            ]

            # Check images file content
            images_pool = dataset["images.jsonl"]
            assert len(images_pool) == 2
            assert all(item.images[0].name == "image" for item in images_pool)
            assert images_pool[0].images[0].content == ["/path/to/image1.png"]
            assert images_pool[1].images[0].content == ["/path/to/image2.png"]

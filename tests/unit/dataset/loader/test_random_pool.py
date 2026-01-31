# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from aiperf.common.models import Text
from aiperf.dataset.loader.models import RandomPool
from aiperf.dataset.loader.random_pool import RandomPoolDatasetLoader
from aiperf.plugin.enums import CustomDatasetType


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

    def test_load_simple_single_file(self, create_jsonl_file, default_user_config):
        """Test loading from a single file with simple content."""
        content = [
            '{"text": "What is deep learning?"}',
            '{"text": "Explain neural networks", "image": "/chart.png"}',
        ]
        filepath = create_jsonl_file(content)

        loader = RandomPoolDatasetLoader(
            filename=filepath, user_config=default_user_config
        )
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

    def test_load_multimodal_single_file(self, create_jsonl_file, default_user_config):
        """Test loading multimodal content from single file."""
        content = [
            '{"text": "Analyze this image", "image": "/data.png"}',
            '{"text": "Transcribe audio", "audio": "/recording.wav"}',
            '{"texts": ["Query 1", "Query 2"], "images": ["/img1.jpg", "/img2.jpg"]}',
        ]
        filepath = create_jsonl_file(content)

        loader = RandomPoolDatasetLoader(
            filename=filepath, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        filename = Path(filepath).name
        dataset_pool = dataset[filename]
        assert len(dataset_pool) == 3
        assert dataset_pool[0].text == "Analyze this image"
        assert dataset_pool[0].image == "/data.png"
        assert dataset_pool[1].audio == "/recording.wav"
        assert dataset_pool[2].texts == ["Query 1", "Query 2"]
        assert dataset_pool[2].images == ["/img1.jpg", "/img2.jpg"]

    def test_load_dataset_skips_empty_lines(
        self, create_jsonl_file, default_user_config
    ):
        """Test that empty lines are skipped during loading."""
        content = [
            '{"text": "First entry"}',
            "",  # Empty line
            '{"text": "Second entry"}',
            "   ",  # Whitespace only
            '{"text": "Third entry"}',
        ]
        filepath = create_jsonl_file(content)

        loader = RandomPoolDatasetLoader(
            filename=filepath, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        filename = Path(filepath).name
        dataset_pool = dataset[filename]
        assert len(dataset_pool) == 3  # Should skip empty lines

    def test_load_directory_with_multiple_files(self, default_user_config):
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
                    '{"images": [{"name": "image", "contents": ["/path/to/image1.png"]}]}\n'
                )
                f.write(
                    '{"images": [{"name": "image", "contents": ["/path/to/image2.png"]}]}\n'
                )

            loader = RandomPoolDatasetLoader(
                filename=str(temp_path), user_config=default_user_config
            )
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
            assert images_pool[0].images[0].contents == ["/path/to/image1.png"]
            assert images_pool[1].images[0].contents == ["/path/to/image2.png"]

    def test_convert_simple_pool_data(self, default_user_config):
        """Test converting simple random pool data to conversations."""
        data = {"file1.jsonl": [RandomPool(text="Hello world")]}

        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        assert len(conversations[0].turns) == 1
        assert conversations[0].turns[0].texts[0].contents == ["Hello world"]

    def test_convert_multimodal_pool_data(self, default_user_config):
        """Test converting multimodal random pool data."""
        data = {
            "multimodal.jsonl": [
                RandomPool(
                    text="What's in this image?",
                    image="https://example.com/image.png",
                    audio="https://example.com/audio.wav",
                )
            ]
        }

        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 1
        assert turn.texts[0].contents == ["What's in this image?"]
        assert len(turn.images) == 1
        assert turn.images[0].contents == ["https://example.com/image.png"]
        assert len(turn.audios) == 1
        assert turn.audios[0].contents == ["https://example.com/audio.wav"]

    def test_convert_batched_pool_data(self, default_user_config):
        """Test converting pool data with batched content."""
        data = {
            "batched.jsonl": [
                RandomPool(
                    texts=["First question", "Second question"],
                    images=[
                        "https://example.com/image1.png",
                        "https://example.com/image2.png",
                    ],
                )
            ]
        }

        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 1
        assert turn.texts[0].contents == ["First question", "Second question"]
        assert len(turn.images) == 1
        assert turn.images[0].contents == [
            "https://example.com/image1.png",
            "https://example.com/image2.png",
        ]

    def test_convert_multiple_files_no_name_specified(self, default_user_config):
        """Test converting data from multiple files without name specified."""
        # Simplified version with no name specified
        data = {
            "queries.jsonl": [
                RandomPool(text="What is AI?"),
            ],
            "contexts.jsonl": [RandomPool(text="AI is artificial intelligence")],
        }

        loader = RandomPoolDatasetLoader(
            filename="dummy_dir", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1  # merged queries & contexts
        assert len(conversations[0].turns) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 2
        assert turn.texts[0].name == "queries"  # use filename if not specified
        assert turn.texts[0].contents == ["What is AI?"]
        assert turn.texts[1].name == "contexts"  # use filename if not specified
        assert turn.texts[1].contents == ["AI is artificial intelligence"]

    def test_convert_multiple_files_with_name_specified(self, default_user_config):
        """Test converting data from multiple files with name specified."""
        data = {
            "queries.jsonl": [
                RandomPool(texts=[Text(name="abc123", contents=["What is AI?"])]),
            ],
            "contexts.jsonl": [
                RandomPool(
                    texts=[
                        Text(name="def456", contents=["AI is artificial intelligence"])
                    ]
                )
            ],
        }

        loader = RandomPoolDatasetLoader(
            filename="dummy_dir", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1  # merged queries & contexts
        assert len(conversations[0].turns) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 2
        assert turn.texts[0].name == "abc123"  # uses name from Text object
        assert turn.texts[0].contents == ["What is AI?"]
        assert turn.texts[1].name == "def456"  # uses name from Text object
        assert turn.texts[1].contents == ["AI is artificial intelligence"]

    def test_convert_multiple_files_with_multiple_samples(self, default_user_config):
        """Test converting data from multiple files with multiple samples."""
        data = {
            "queries.jsonl": [
                RandomPool(text="text1", image="https://example.com/image1.png"),
                RandomPool(text="text2", image="https://example.com/image2.png"),
            ],
            "contexts.jsonl": [
                RandomPool(text="text3", image="https://example.com/image3.png"),
                RandomPool(text="text4", image="https://example.com/image4.png"),
            ],
        }

        loader = RandomPoolDatasetLoader(
            filename="dummy_dir", user_config=default_user_config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2

        # make sure it's single turn
        conv1, conv2 = conversations
        assert len(conv1.turns) == 1
        assert len(conv2.turns) == 1

        # each turn contains 2 text & image data from the two files
        # (e.g. queries and contexts)
        turn1, turn2 = conv1.turns[0], conv2.turns[0]
        assert len(turn1.texts) == 2
        assert len(turn1.images) == 2
        assert len(turn2.texts) == 2
        assert len(turn2.images) == 2

        possible_text_contents = {
            ("text1", "text3"),
            ("text1", "text4"),
            ("text2", "text3"),
            ("text2", "text4"),
        }
        possible_image_contents = {
            ("https://example.com/image1.png", "https://example.com/image3.png"),
            ("https://example.com/image1.png", "https://example.com/image4.png"),
            ("https://example.com/image2.png", "https://example.com/image3.png"),
            ("https://example.com/image2.png", "https://example.com/image4.png"),
        }

        text_contents = tuple(t.contents[0] for t in turn1.texts)
        image_contents = tuple(i.contents[0] for i in turn1.images)
        assert text_contents in possible_text_contents
        assert image_contents in possible_image_contents

        text_contents = tuple(t.contents[0] for t in turn2.texts)
        image_contents = tuple(i.contents[0] for i in turn2.images)
        assert text_contents in possible_text_contents
        assert image_contents in possible_image_contents

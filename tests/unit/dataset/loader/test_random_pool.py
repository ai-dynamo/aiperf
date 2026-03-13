# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import (
    AudioConfig,
    ImageConfig,
    InputConfig,
    PromptConfig,
    VideoConfig,
)
from aiperf.common.models import Audio, Image, Text, Video
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


class TestRandomPoolBatchSize:
    """Tests for batch size support in RandomPoolDatasetLoader."""

    def _make_config(
        self,
        batch_size_image=1,
        batch_size_text=1,
        batch_size_audio=1,
        batch_size_video=1,
    ):
        from aiperf.common.config import EndpointConfig, UserConfig

        return UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                image=ImageConfig(batch_size=batch_size_image),
                prompt=PromptConfig(batch_size=batch_size_text),
                audio=AudioConfig(batch_size=batch_size_audio),
                video=VideoConfig(batch_size=batch_size_video),
            ),
        )

    def test_batch_size_image_produces_correct_image_count(self, default_user_config):
        """Each conversation should contain batch_size_image images sampled from the flat pool."""
        config = self._make_config(batch_size_image=3)
        data = {
            "images.jsonl": [
                RandomPool(image="https://example.com/img1.png"),
                RandomPool(image="https://example.com/img2.png"),
                RandomPool(image="https://example.com/img3.png"),
                RandomPool(image="https://example.com/img4.png"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        for conv in conversations:
            assert len(conv.turns) == 1
            turn = conv.turns[0]
            assert len(turn.images) == 1
            assert len(turn.images[0].contents) == 3

    def test_batch_size_text_produces_correct_text_count(self, default_user_config):
        """Each conversation should contain batch_size_text texts sampled from the flat pool."""
        config = self._make_config(batch_size_text=4)
        data = {
            "texts.jsonl": [
                RandomPool(text="query1"),
                RandomPool(text="query2"),
                RandomPool(text="query3"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        for conv in conversations:
            assert len(conv.turns) == 1
            turn = conv.turns[0]
            assert len(turn.texts) == 1
            assert len(turn.texts[0].contents) == 4

    def test_batch_mode_images_sampled_from_pool(self, default_user_config):
        """Sampled images should come from the pool entries."""
        config = self._make_config(batch_size_image=2)
        pool_images = [
            "https://example.com/a.png",
            "https://example.com/b.png",
            "https://example.com/c.png",
        ]
        data = {"f.jsonl": [RandomPool(image=img) for img in pool_images]}
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=5
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            for img_content in conv.turns[0].images[0].contents:
                assert img_content in pool_images

    def test_batch_mode_texts_sampled_from_pool(self, default_user_config):
        """Sampled texts should come from the pool entries."""
        config = self._make_config(batch_size_text=2)
        pool_texts = ["alpha", "beta", "gamma"]
        data = {"f.jsonl": [RandomPool(text=t) for t in pool_texts]}
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=5
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            for txt_content in conv.turns[0].texts[0].contents:
                assert txt_content in pool_texts

    def test_batch_mode_images_flattened_from_images_field(self, default_user_config):
        """Images specified via 'images' list field should be included in the flat pool."""
        config = self._make_config(batch_size_image=2)
        data = {
            "f.jsonl": [
                RandomPool(
                    images=["https://example.com/x.png", "https://example.com/y.png"]
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        expected = {"https://example.com/x.png", "https://example.com/y.png"}
        for conv in conversations:
            for img_content in conv.turns[0].images[0].contents:
                assert img_content in expected

    def test_batch_mode_both_image_and_text(self, default_user_config):
        """When both batch sizes > 1, conversations contain both image and text batches."""
        config = self._make_config(batch_size_image=2, batch_size_text=3)
        data = {
            "f.jsonl": [
                RandomPool(image="https://example.com/img1.png", text="text1"),
                RandomPool(image="https://example.com/img2.png", text="text2"),
                RandomPool(image="https://example.com/img3.png", text="text3"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        for conv in conversations:
            turn = conv.turns[0]
            assert len(turn.images[0].contents) == 2
            assert len(turn.texts[0].contents) == 3

    def test_default_batch_size_uses_existing_behavior(self, default_user_config):
        """When batch_size is 1 (default), the existing per-entry sampling path is used."""
        data = {"f.jsonl": [RandomPool(text="hello"), RandomPool(text="world")]}
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        # Existing behavior: each conversation has exactly 1 text object with 1 content
        assert len(conversations) == 2
        for conv in conversations:
            assert len(conv.turns) == 1
            turn = conv.turns[0]
            assert len(turn.texts) == 1
            assert len(turn.texts[0].contents) == 1

    def test_image_batch_preserves_text_at_default_size(self, default_user_config):
        """When only batch_size_image > 1, text (at default size 1) must still appear."""
        config = self._make_config(batch_size_image=3)
        data = {
            "f.jsonl": [
                RandomPool(image="https://example.com/img1.png", text="query1"),
                RandomPool(image="https://example.com/img2.png", text="query2"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            turn = conv.turns[0]
            # 3 images batched
            assert len(turn.images) == 1
            assert len(turn.images[0].contents) == 3
            # text still present (1 item, not dropped)
            assert len(turn.texts) == 1
            assert len(turn.texts[0].contents) == 1
            assert turn.texts[0].contents[0] in {"query1", "query2"}

    def test_text_batch_preserves_image_at_default_size(self, default_user_config):
        """When only batch_size_text > 1, image (at default size 1) must still appear."""
        config = self._make_config(batch_size_text=4)
        data = {
            "f.jsonl": [
                RandomPool(image="https://example.com/img1.png", text="query1"),
                RandomPool(image="https://example.com/img2.png", text="query2"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            turn = conv.turns[0]
            # 4 texts batched
            assert len(turn.texts) == 1
            assert len(turn.texts[0].contents) == 4
            # image still present (1 item, not dropped)
            assert len(turn.images) == 1
            assert len(turn.images[0].contents) == 1
            assert turn.images[0].contents[0] in {
                "https://example.com/img1.png",
                "https://example.com/img2.png",
            }

    def test_batch_mode_preserves_audio(self, default_user_config):
        """Audio entries must not be dropped when batch mode is triggered by image batch size."""
        config = self._make_config(batch_size_image=2)
        data = {
            "f.jsonl": [
                RandomPool(
                    image="https://example.com/img1.png",
                    audio="https://example.com/a1.wav",
                ),
                RandomPool(
                    image="https://example.com/img2.png",
                    audio="https://example.com/a2.wav",
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        audio_urls = {"https://example.com/a1.wav", "https://example.com/a2.wav"}
        for conv in conversations:
            turn = conv.turns[0]
            assert len(turn.images[0].contents) == 2
            assert len(turn.audios) == 1
            assert turn.audios[0].contents[0] in audio_urls

    def test_batch_mode_preserves_video(self, default_user_config):
        """Video entries must appear in conversations when batch mode is active."""
        config = self._make_config(batch_size_image=2)
        data = {
            "f.jsonl": [
                RandomPool(
                    image="https://example.com/img1.png",
                    video="https://example.com/v1.mp4",
                ),
                RandomPool(
                    image="https://example.com/img2.png",
                    video="https://example.com/v2.mp4",
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        video_urls = {"https://example.com/v1.mp4", "https://example.com/v2.mp4"}
        for conv in conversations:
            turn = conv.turns[0]
            assert len(turn.images[0].contents) == 2
            assert len(turn.videos) == 1
            assert turn.videos[0].contents[0] in video_urls

    def test_batch_mode_named_image_objects_flattened(self, default_user_config):
        """Images specified as named Image objects should have their contents added to the pool."""
        config = self._make_config(batch_size_image=2)
        data = {
            "f.jsonl": [
                RandomPool(
                    images=[
                        Image(
                            name="img",
                            contents=[
                                "https://example.com/a.png",
                                "https://example.com/b.png",
                            ],
                        )
                    ]
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        expected = {"https://example.com/a.png", "https://example.com/b.png"}
        for conv in conversations:
            for img_content in conv.turns[0].images[0].contents:
                assert img_content in expected

    def test_batch_mode_named_text_objects_flattened(self, default_user_config):
        """Texts specified as named Text objects should have their contents added to the pool."""
        config = self._make_config(batch_size_text=2)
        data = {
            "f.jsonl": [
                RandomPool(
                    texts=[Text(name="query", contents=["alpha", "beta", "gamma"])]
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        expected = {"alpha", "beta", "gamma"}
        for conv in conversations:
            for txt_content in conv.turns[0].texts[0].contents:
                assert txt_content in expected

    def test_batch_mode_named_audio_objects_flattened(self, default_user_config):
        """Audios specified as named Audio objects should have their contents added to the pool."""
        config = self._make_config(batch_size_image=2)
        data = {
            "f.jsonl": [
                RandomPool(
                    image="https://example.com/img1.png",
                    audios=[
                        Audio(
                            name="aud",
                            contents=[
                                "https://example.com/a1.wav",
                                "https://example.com/a2.wav",
                            ],
                        )
                    ],
                ),
                RandomPool(
                    image="https://example.com/img2.png",
                    audios=[Audio(name="aud", contents=["https://example.com/a3.wav"])],
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        expected = {
            "https://example.com/a1.wav",
            "https://example.com/a2.wav",
            "https://example.com/a3.wav",
        }
        for conv in conversations:
            turn = conv.turns[0]
            assert len(turn.audios) == 1
            assert turn.audios[0].contents[0] in expected

    def test_batch_mode_named_video_objects_flattened(self, default_user_config):
        """Videos specified as named Video objects should have their contents added to the pool."""
        config = self._make_config(batch_size_image=2)
        data = {
            "f.jsonl": [
                RandomPool(
                    image="https://example.com/img1.png",
                    videos=[
                        Video(
                            name="vid",
                            contents=[
                                "https://example.com/v1.mp4",
                                "https://example.com/v2.mp4",
                            ],
                        )
                    ],
                ),
                RandomPool(
                    image="https://example.com/img2.png",
                    videos=[Video(name="vid", contents=["https://example.com/v3.mp4"])],
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        expected = {
            "https://example.com/v1.mp4",
            "https://example.com/v2.mp4",
            "https://example.com/v3.mp4",
        }
        for conv in conversations:
            turn = conv.turns[0]
            assert len(turn.videos) == 1
            assert turn.videos[0].contents[0] in expected

    def test_batch_mode_plain_string_videos_flattened(self, default_user_config):
        """Videos specified as plain strings should be included in the flat pool."""
        config = self._make_config(batch_size_image=2)
        data = {
            "f.jsonl": [
                RandomPool(
                    image="https://example.com/img1.png",
                    videos=["https://example.com/v1.mp4", "https://example.com/v2.mp4"],
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        expected = {"https://example.com/v1.mp4", "https://example.com/v2.mp4"}
        for conv in conversations:
            turn = conv.turns[0]
            assert len(turn.videos) == 1
            assert turn.videos[0].contents[0] in expected

    def test_batch_size_image_zero_disables_images(self, default_user_config):
        """batch_size_image=0 should suppress image output even when images are in the pool."""
        config = self._make_config(batch_size_image=0, batch_size_text=2)
        data = {
            "f.jsonl": [
                RandomPool(image="https://example.com/img1.png", text="query1"),
                RandomPool(image="https://example.com/img2.png", text="query2"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            turn = conv.turns[0]
            assert turn.images == []
            assert len(turn.texts[0].contents) == 2

    def test_batch_size_text_zero_disables_texts(self, default_user_config):
        """batch_size_text=0 should suppress text output even when texts are in the pool."""
        config = self._make_config(batch_size_image=2, batch_size_text=0)
        data = {
            "f.jsonl": [
                RandomPool(image="https://example.com/img1.png", text="query1"),
                RandomPool(image="https://example.com/img2.png", text="query2"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            turn = conv.turns[0]
            assert turn.texts == []
            assert len(turn.images[0].contents) == 2

    def test_image_zero_text_one_disables_images_via_legacy_path(
        self, default_user_config
    ):
        """batch_size_image=0/text=1 must not emit images even via the legacy sampler path."""
        config = self._make_config(batch_size_image=0, batch_size_text=1)
        data = {
            "f.jsonl": [
                RandomPool(image="https://example.com/img1.png", text="query1"),
                RandomPool(image="https://example.com/img2.png", text="query2"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            turn = conv.turns[0]
            assert turn.images == [], (
                "images should be suppressed when batch_size_image=0"
            )
            assert len(turn.texts) == 1

    def test_image_one_text_zero_disables_texts_via_legacy_path(
        self, default_user_config
    ):
        """batch_size_image=1/text=0 must not emit texts even via the legacy sampler path."""
        config = self._make_config(batch_size_image=1, batch_size_text=0)
        data = {
            "f.jsonl": [
                RandomPool(image="https://example.com/img1.png", text="query1"),
                RandomPool(image="https://example.com/img2.png", text="query2"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            turn = conv.turns[0]
            assert turn.texts == [], "texts should be suppressed when batch_size_text=0"
            assert len(turn.images) == 1

    def test_num_conversations_none_defaults_to_100(self, default_user_config):
        """When num_conversations=None is passed, the loader should default to 100."""
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl",
            user_config=default_user_config,
            num_conversations=None,
        )
        assert loader.num_conversations == 100

    def test_batch_size_audio_produces_correct_audio_count(self, default_user_config):
        """Each conversation should contain batch_size_audio audios sampled from the flat pool."""
        config = self._make_config(batch_size_audio=3)
        data = {
            "audios.jsonl": [
                RandomPool(audio="https://example.com/a1.wav"),
                RandomPool(audio="https://example.com/a2.wav"),
                RandomPool(audio="https://example.com/a3.wav"),
                RandomPool(audio="https://example.com/a4.wav"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        for conv in conversations:
            assert len(conv.turns) == 1
            turn = conv.turns[0]
            assert len(turn.audios) == 1
            assert len(turn.audios[0].contents) == 3

    def test_batch_size_video_produces_correct_video_count(self, default_user_config):
        """Each conversation should contain batch_size_video videos sampled from the flat pool."""
        config = self._make_config(batch_size_video=2)
        data = {
            "videos.jsonl": [
                RandomPool(video="https://example.com/v1.mp4"),
                RandomPool(video="https://example.com/v2.mp4"),
                RandomPool(video="https://example.com/v3.mp4"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        for conv in conversations:
            assert len(conv.turns) == 1
            turn = conv.turns[0]
            assert len(turn.videos) == 1
            assert len(turn.videos[0].contents) == 2

    def test_batch_size_audio_sampled_from_pool(self, default_user_config):
        """Sampled audios should come from the pool entries."""
        config = self._make_config(batch_size_audio=2)
        pool_audios = [
            "https://example.com/a.wav",
            "https://example.com/b.wav",
            "https://example.com/c.wav",
        ]
        data = {"f.jsonl": [RandomPool(audio=a) for a in pool_audios]}
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=5
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            for aud_content in conv.turns[0].audios[0].contents:
                assert aud_content in pool_audios

    def test_batch_size_video_sampled_from_pool(self, default_user_config):
        """Sampled videos should come from the pool entries."""
        config = self._make_config(batch_size_video=2)
        pool_videos = [
            "https://example.com/v1.mp4",
            "https://example.com/v2.mp4",
            "https://example.com/v3.mp4",
        ]
        data = {"f.jsonl": [RandomPool(video=v) for v in pool_videos]}
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=5
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            for vid_content in conv.turns[0].videos[0].contents:
                assert vid_content in pool_videos

    def test_audio_batch_size_triggers_batched_path(self, default_user_config):
        """Setting batch_size_audio != 1 should trigger the batched path."""
        config = self._make_config(batch_size_audio=2)
        data = {
            "f.jsonl": [
                RandomPool(text="query1", audio="https://example.com/a1.wav"),
                RandomPool(text="query2", audio="https://example.com/a2.wav"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        for conv in conversations:
            turn = conv.turns[0]
            assert len(turn.audios) == 1
            assert len(turn.audios[0].contents) == 2

    def test_video_batch_size_triggers_batched_path(self, default_user_config):
        """Setting batch_size_video != 1 should trigger the batched path."""
        config = self._make_config(batch_size_video=2)
        data = {
            "f.jsonl": [
                RandomPool(text="query1", video="https://example.com/v1.mp4"),
                RandomPool(text="query2", video="https://example.com/v2.mp4"),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        for conv in conversations:
            turn = conv.turns[0]
            assert len(turn.videos) == 1
            assert len(turn.videos[0].contents) == 2

    def test_batch_size_audio_zero_disables_audio(self, default_user_config):
        """batch_size_audio=0 should suppress audio output even when audios are in the pool."""
        config = self._make_config(batch_size_image=2, batch_size_audio=0)
        data = {
            "f.jsonl": [
                RandomPool(
                    image="https://example.com/img1.png",
                    audio="https://example.com/a1.wav",
                ),
                RandomPool(
                    image="https://example.com/img2.png",
                    audio="https://example.com/a2.wav",
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            turn = conv.turns[0]
            assert turn.audios == []
            assert len(turn.images[0].contents) == 2

    def test_batch_size_video_zero_disables_video(self, default_user_config):
        """batch_size_video=0 should suppress video output even when videos are in the pool."""
        config = self._make_config(batch_size_image=2, batch_size_video=0)
        data = {
            "f.jsonl": [
                RandomPool(
                    image="https://example.com/img1.png",
                    video="https://example.com/v1.mp4",
                ),
                RandomPool(
                    image="https://example.com/img2.png",
                    video="https://example.com/v2.mp4",
                ),
            ]
        }
        loader = RandomPoolDatasetLoader(
            filename="dummy.jsonl", user_config=config, num_conversations=2
        )
        conversations = loader.convert_to_conversations(data)

        for conv in conversations:
            turn = conv.turns[0]
            assert turn.videos == []
            assert len(turn.images[0].contents) == 2

    def test_audio_video_batch_sizes_read_from_config(self, default_user_config):
        """batch_size_audio and batch_size_video should be read from the user config."""
        config = self._make_config(batch_size_audio=2, batch_size_video=3)
        loader = RandomPoolDatasetLoader(filename="dummy.jsonl", user_config=config)
        assert loader.batch_size_audio == 2
        assert loader.batch_size_video == 3

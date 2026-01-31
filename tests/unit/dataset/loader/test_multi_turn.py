# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from aiperf.common.models import Image, Text
from aiperf.dataset.loader.models import MultiTurn, SingleTurn
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader
from aiperf.plugin.enums import CustomDatasetType


class TestMultiTurn:
    """Tests for MultiTurn model validation and functionality."""

    def test_create_simple_conversation(self):
        """Test creating a basic multi-turn conversation."""
        data = MultiTurn(
            session_id="test_session",
            turns=[
                SingleTurn(text="Hello"),
                SingleTurn(text="Hi there", delay=1000),
            ],
        )

        assert data.session_id == "test_session"
        assert len(data.turns) == 2
        assert data.turns[0].text == "Hello"
        assert data.turns[0].texts is None
        assert data.turns[1].text == "Hi there"
        assert data.turns[1].texts is None
        assert data.turns[1].delay == 1000
        assert data.type == CustomDatasetType.MULTI_TURN

    def test_create_without_session_id(self):
        """Test creating conversation without explicit session_id."""
        data = MultiTurn(turns=[SingleTurn(text="What is AI?")])

        assert data.session_id is None
        assert len(data.turns) == 1
        assert data.turns[0].text == "What is AI?"

    def test_create_with_multimodal_turns(self):
        """Test creating conversation with multimodal turns."""
        data = MultiTurn(
            session_id="multimodal_session",
            turns=[
                SingleTurn(text="Describe this image", image="/path/to/image.png"),
                SingleTurn(text="What about this audio?", audio="/path/to/audio.wav"),
                SingleTurn(text="Summary please", delay=2000),
            ],
        )

        assert len(data.turns) == 3
        assert data.turns[0].image == "/path/to/image.png"
        assert data.turns[1].audio == "/path/to/audio.wav"
        assert data.turns[2].delay == 2000

    def test_create_with_timestamp_scheduling(self):
        """Test creating conversation with timestamp-based scheduling."""
        data = MultiTurn(
            session_id="scheduled_session",
            turns=[
                SingleTurn(text="First message", timestamp=0),
                SingleTurn(text="Second message", timestamp=5000),
                SingleTurn(text="Third message", timestamp=10000),
            ],
        )

        assert all(turn.timestamp is not None for turn in data.turns)
        assert data.turns[0].timestamp == 0
        assert data.turns[1].timestamp == 5000
        assert data.turns[2].timestamp == 10000

    def test_create_with_batched_turns(self):
        """Test creating conversation with batched content in turns."""
        data = MultiTurn(
            session_id="batched_session",
            turns=[
                SingleTurn(
                    texts=["Hello there", "How are you?"],
                    images=["/path/1.png", "/path/2.png"],
                ),
                SingleTurn(texts=["I'm fine", "Thanks for asking"], delay=1500),
            ],
        )

        assert len(data.turns[0].texts) == 2
        assert len(data.turns[0].images) == 2
        assert len(data.turns[1].texts) == 2

    def test_create_with_full_featured_turns(self):
        """Test creating conversation with full-featured turn format."""
        data = MultiTurn(
            session_id="featured_session",
            turns=[
                SingleTurn(
                    texts=[
                        Text(name="question", contents=["What is this?"]),
                        Text(name="context", contents=["Please be detailed"]),
                    ],
                    images=[
                        Image(name="main_image", contents=["/path/main.png"]),
                        Image(name="reference", contents=["/path/ref.png"]),
                    ],
                )
            ],
        )

        assert len(data.turns[0].texts) == 2
        assert len(data.turns[0].images) == 2
        assert data.turns[0].texts[0].name == "question"
        assert data.turns[0].images[0].name == "main_image"

    def test_validation_empty_turns_raises_error(self):
        """Test that empty turns list raises validation error."""
        with pytest.raises(ValueError, match="At least one turn must be provided"):
            MultiTurn(session_id="empty_session", turns=[])


class TestMultiTurnDatasetLoader:
    """Tests for MultiTurnDatasetLoader functionality."""

    def test_load_simple_conversation(self, create_jsonl_file, default_user_config):
        """Test loading a simple multi-turn conversation."""
        content = [
            json.dumps(
                {
                    "session_id": "conv_001",
                    "turns": [
                        {"text": "Hello, how are you?"},
                        {"text": "I'm doing well, thanks!", "delay": 1000},
                    ],
                }
            )
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 1
        assert "conv_001" in dataset

        multi_turn = dataset["conv_001"][0]
        assert isinstance(multi_turn, MultiTurn)
        assert len(multi_turn.turns) == 2
        assert multi_turn.turns[0].text == "Hello, how are you?"
        assert multi_turn.turns[0].texts is None
        assert multi_turn.turns[1].text == "I'm doing well, thanks!"
        assert multi_turn.turns[1].texts is None
        assert multi_turn.turns[1].delay == 1000

    def test_load_multiple_conversations(self, create_jsonl_file, default_user_config):
        """Test loading multiple conversations from file."""
        content = [
            json.dumps(
                {
                    "session_id": "session_A",
                    "turns": [
                        {"text": "First conversation start"},
                    ],
                }
            ),
            json.dumps(
                {
                    "session_id": "session_B",
                    "turns": [
                        {"text": "Second conversation start"},
                        {"text": "Second conversation continues", "delay": 2000},
                    ],
                }
            ),
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2
        assert "session_A" in dataset
        assert "session_B" in dataset
        assert len(dataset["session_A"][0].turns) == 1
        assert len(dataset["session_B"][0].turns) == 2

    def test_load_conversation_without_session_id(
        self, create_jsonl_file, default_user_config
    ):
        """Test loading conversation without explicit session_id generates UUID."""
        content = [
            json.dumps(
                {
                    "turns": [
                        {"text": "Anonymous conversation"},
                        {"text": "Should get auto-generated session_id"},
                    ]
                }
            )
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 1
        session_id = list(dataset.keys())[0]
        # Should be a UUID string (36 characters with hyphens)
        assert len(session_id) == 36
        assert session_id.count("-") == 4

        multi_turn = dataset[session_id][0]
        assert len(multi_turn.turns) == 2

    def test_load_multimodal_conversation(self, create_jsonl_file, default_user_config):
        """Test loading conversation with multimodal content."""
        content = [
            json.dumps(
                {
                    "session_id": "multimodal_chat",
                    "turns": [
                        {
                            "text": "What do you see?",
                            "image": "/path/to/image.jpg",
                        },
                        {
                            "text": "Can you hear this?",
                            "audio": "/path/to/sound.wav",
                            "delay": 3000,
                        },
                    ],
                }
            )
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        multi_turn = dataset["multimodal_chat"][0]
        assert multi_turn.turns[0].text == "What do you see?"
        assert multi_turn.turns[0].texts is None
        assert multi_turn.turns[0].image == "/path/to/image.jpg"
        assert multi_turn.turns[0].images is None
        assert multi_turn.turns[1].text == "Can you hear this?"
        assert multi_turn.turns[1].texts is None
        assert multi_turn.turns[1].audio == "/path/to/sound.wav"
        assert multi_turn.turns[1].audios is None
        assert multi_turn.turns[1].delay == 3000

    def test_load_scheduled_conversation(self, create_jsonl_file, default_user_config):
        """Test loading conversation with timestamp scheduling."""
        content = [
            json.dumps(
                {
                    "session_id": "scheduled_chat",
                    "turns": [
                        {"text": "Message at start", "timestamp": 0},
                        {"text": "Message after 5 seconds", "timestamp": 5000},
                        {"text": "Final message", "timestamp": 10000},
                    ],
                }
            )
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        result = loader.load_dataset()

        conversation = result["scheduled_chat"][0]
        timestamps = [turn.timestamp for turn in conversation.turns]
        assert timestamps == [0, 5000, 10000]

    def test_load_batched_conversation(self, create_jsonl_file, default_user_config):
        """Test loading conversation with batched content."""
        content = [
            json.dumps(
                {
                    "session_id": "batched_chat",
                    "turns": [
                        {
                            "texts": ["Hello", "How are you?"],
                            "images": ["/img1.png", "/img2.png"],
                        },
                        {
                            "texts": ["Fine", "Thanks"],
                            "delay": 1500,
                        },
                    ],
                }
            )
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        multi_turn = dataset["batched_chat"][0]
        assert multi_turn.turns[0].text is None
        assert multi_turn.turns[0].texts == ["Hello", "How are you?"]
        assert multi_turn.turns[0].image is None
        assert multi_turn.turns[0].images == ["/img1.png", "/img2.png"]
        assert multi_turn.turns[1].text is None
        assert multi_turn.turns[1].texts == ["Fine", "Thanks"]

    def test_load_full_featured_conversation(
        self, create_jsonl_file, default_user_config
    ):
        """Test loading conversation with full-featured format."""
        content = [
            json.dumps(
                {
                    "session_id": "full_featured_chat",
                    "turns": [
                        {
                            "texts": [
                                {
                                    "name": "user_query",
                                    "contents": ["Analyze this data"],
                                },
                                {"name": "user_context", "contents": ["Be thorough"]},
                            ],
                            "images": [
                                {"name": "dataset_viz", "contents": ["/chart.png"]},
                                {"name": "raw_data", "contents": ["/data.png"]},
                            ],
                            "timestamp": 1000,
                        },
                    ],
                }
            )
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        multi_turn = dataset["full_featured_chat"][0]
        assert len(multi_turn.turns) == 1

        turn = multi_turn.turns[0]
        assert len(turn.texts) == 2
        assert turn.texts[0].name == "user_query"
        assert turn.texts[0].contents == ["Analyze this data"]
        assert turn.texts[1].name == "user_context"
        assert turn.texts[1].contents == ["Be thorough"]

        assert len(turn.images) == 2
        assert turn.images[0].name == "dataset_viz"
        assert turn.images[0].contents == ["/chart.png"]
        assert turn.images[1].name == "raw_data"
        assert turn.images[1].contents == ["/data.png"]
        assert turn.timestamp == 1000

    def test_load_dataset_skips_empty_lines(
        self, create_jsonl_file, default_user_config
    ):
        """Test that empty lines are skipped during loading."""
        content = [
            json.dumps(
                {
                    "session_id": "test_empty_lines",
                    "turns": [{"text": "First"}],
                }
            ),
            "",  # Empty line
            json.dumps(
                {
                    "session_id": "test_empty_lines_2",
                    "turns": [{"text": "Second"}],
                }
            ),
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 2  # Should skip empty line
        assert "test_empty_lines" in dataset
        assert "test_empty_lines_2" in dataset

    def test_load_duplicate_session_ids_are_grouped(
        self, create_jsonl_file, default_user_config
    ):
        """Test that multiple conversations with same session_id are grouped together."""
        content = [
            json.dumps(
                {
                    "session_id": "shared_session",
                    "turns": [{"text": "First conversation"}],
                }
            ),
            json.dumps(
                {
                    "session_id": "shared_session",
                    "turns": [{"text": "Second conversation"}],
                }
            ),
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(
            filename=filename, user_config=default_user_config
        )
        dataset = loader.load_dataset()

        assert len(dataset) == 1  # Same session_id groups together
        assert len(dataset["shared_session"]) == 2  # Two conversations in same session

        multi_turn = dataset["shared_session"]
        assert multi_turn[0].turns[0].text == "First conversation"
        assert multi_turn[1].turns[0].text == "Second conversation"


class TestMultiTurnDatasetLoaderConvertToConversations:
    """Test convert_to_conversations method for MultiTurnDatasetLoader."""

    def test_convert_simple_multi_turn_data(self, default_user_config):
        """Test converting simple multi-turn data to conversations."""
        data = {
            "session_123": [
                MultiTurn(
                    session_id="session_123",
                    turns=[
                        SingleTurn(text="Hello"),
                        SingleTurn(text="How are you?", delay=1000),
                    ],
                )
            ]
        }

        loader = MultiTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conversation = conversations[0]
        assert conversation.session_id == "session_123"
        assert len(conversation.turns) == 2

        assert conversation.turns[0].texts[0].contents == ["Hello"]
        assert conversation.turns[0].delay is None

        assert conversation.turns[1].texts[0].contents == ["How are you?"]
        assert conversation.turns[1].delay == 1000

    def test_convert_multiple_multi_turn_entries_same_session(
        self, default_user_config
    ):
        """Test converting multiple MultiTurn entries with same session ID."""
        data = {
            "session_123": [
                MultiTurn(session_id="session_123", turns=[SingleTurn(text="First")]),
                MultiTurn(session_id="session_123", turns=[SingleTurn(text="Second")]),
            ]
        }

        loader = MultiTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conversation = conversations[0]
        assert conversation.session_id == "session_123"
        assert len(conversation.turns) == 2
        assert conversation.turns[0].texts[0].contents == ["First"]
        assert conversation.turns[1].texts[0].contents == ["Second"]

    def test_convert_multimodal_multi_turn_data(self, default_user_config):
        """Test converting multimodal multi-turn data."""
        data = {
            "session_1": [
                MultiTurn(
                    session_id="session_1",
                    turns=[
                        SingleTurn(
                            text="What's this?", image="https://example.com/image1.png"
                        ),
                        SingleTurn(
                            text="Follow up", image="https://example.com/image2.png"
                        ),
                    ],
                )
            ]
        }
        loader = MultiTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )

        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        conversation = conversations[0]
        assert len(conversation.turns) == 2

        # First turn
        first_turn = conversation.turns[0]
        assert first_turn.texts[0].contents == ["What's this?"]
        assert first_turn.images[0].contents == ["https://example.com/image1.png"]

        # Second turn
        second_turn = conversation.turns[1]
        assert second_turn.texts[0].contents == ["Follow up"]
        assert second_turn.images[0].contents == ["https://example.com/image2.png"]

    def test_convert_structured_objects_in_turns(self, default_user_config):
        """Test converting MultiTurn with structured Text objects."""
        data = {
            "session_1": [
                MultiTurn(
                    session_id="session_1",
                    turns=[
                        SingleTurn(
                            texts=[
                                Text(name="query", contents=["What is this?"]),
                                Text(name="context", contents=["Some context"]),
                            ],
                        )
                    ],
                )
            ]
        }

        loader = MultiTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 1
        turn = conversations[0].turns[0]
        assert len(turn.texts) == 2
        assert turn.texts[0].name == "query"
        assert turn.texts[0].contents == ["What is this?"]
        assert turn.texts[1].name == "context"
        assert turn.texts[1].contents == ["Some context"]

    def test_convert_multiple_sessions(self, default_user_config):
        """Test converting multiple sessions."""
        data = {
            "session_1": [
                MultiTurn(session_id="session_1", turns=[SingleTurn(text="First")]),
            ],
            "session_2": [
                MultiTurn(session_id="session_2", turns=[SingleTurn(text="Second")]),
            ],
        }

        loader = MultiTurnDatasetLoader(
            filename="dummy.jsonl", user_config=default_user_config
        )
        conversations = loader.convert_to_conversations(data)

        assert len(conversations) == 2
        assert conversations[0].session_id == "session_1"
        assert conversations[1].session_id == "session_2"
        assert len(conversations[0].turns) == 1
        assert len(conversations[1].turns) == 1
        assert conversations[0].turns[0].texts[0].contents == ["First"]
        assert conversations[1].turns[0].texts[0].contents == ["Second"]

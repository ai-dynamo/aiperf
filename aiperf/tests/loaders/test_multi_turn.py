# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from aiperf.common.dataset_models import Image, Text
from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.loader.models import (
    MultiTurnCustomData,
    SingleTurnCustomData,
)
from aiperf.services.dataset.loader.multi_turn import MultiTurnDatasetLoader


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


class TestMultiTurnCustomData:
    """Tests for MultiTurnCustomData model validation and functionality."""

    def test_create_simple_conversation(self):
        """Test creating a basic multi-turn conversation."""
        turns = [
            SingleTurnCustomData(text="Hello"),
            SingleTurnCustomData(text="Hi there", delay=1000),
        ]
        data = MultiTurnCustomData(session_id="test_session", turns=turns)

        assert data.session_id == "test_session"
        assert len(data.turns) == 2
        assert data.turns[0].text == "Hello"
        assert data.turns[1].text == "Hi there"
        assert data.turns[1].delay == 1000
        assert data.type == CustomDatasetType.MULTI_TURN

    def test_create_without_session_id(self):
        """Test creating conversation without explicit session_id."""
        turns = [SingleTurnCustomData(text="What is AI?")]
        data = MultiTurnCustomData(turns=turns)

        assert data.session_id is None
        assert len(data.turns) == 1
        assert data.turns[0].text == "What is AI?"

    def test_create_with_multimodal_turns(self):
        """Test creating conversation with multimodal turns."""
        turns = [
            SingleTurnCustomData(
                text="Describe this image", image="/path/to/image.png"
            ),
            SingleTurnCustomData(
                text="What about this audio?", audio="/path/to/audio.wav"
            ),
            SingleTurnCustomData(text="Summary please", delay=2000),
        ]
        data = MultiTurnCustomData(session_id="multimodal_session", turns=turns)

        assert len(data.turns) == 3
        assert data.turns[0].image == "/path/to/image.png"
        assert data.turns[1].audio == "/path/to/audio.wav"
        assert data.turns[2].delay == 2000

    def test_create_with_timestamp_scheduling(self):
        """Test creating conversation with timestamp-based scheduling."""
        turns = [
            SingleTurnCustomData(text="First message", timestamp=0),
            SingleTurnCustomData(text="Second message", timestamp=5000),
            SingleTurnCustomData(text="Third message", timestamp=10000),
        ]
        data = MultiTurnCustomData(session_id="scheduled_session", turns=turns)

        assert all(turn.timestamp is not None for turn in data.turns)
        assert data.turns[0].timestamp == 0
        assert data.turns[1].timestamp == 5000
        assert data.turns[2].timestamp == 10000

    def test_create_with_batched_turns(self):
        """Test creating conversation with batched content in turns."""
        turns = [
            SingleTurnCustomData(
                text=["Hello there", "How are you?"],
                image=["/path/1.png", "/path/2.png"],
            ),
            SingleTurnCustomData(text=["I'm fine", "Thanks for asking"], delay=1500),
        ]
        data = MultiTurnCustomData(session_id="batched_session", turns=turns)

        assert len(data.turns[0].text) == 2
        assert len(data.turns[0].image) == 2
        assert len(data.turns[1].text) == 2

    def test_create_with_full_featured_turns(self):
        """Test creating conversation with full-featured turn format."""
        turns = [
            SingleTurnCustomData(
                text=[
                    Text(name="question", content=["What is this?"]),
                    Text(name="context", content=["Please be detailed"]),
                ],
                image=[
                    Image(name="main_image", content=["/path/main.png"]),
                    Image(name="reference", content=["/path/ref.png"]),
                ],
            )
        ]
        data = MultiTurnCustomData(session_id="featured_session", turns=turns)

        assert len(data.turns[0].text) == 2
        assert len(data.turns[0].image) == 2
        assert data.turns[0].text[0].name == "question"
        assert data.turns[0].image[0].name == "main_image"

    def test_validation_empty_turns_raises_error(self):
        """Test that empty turns list raises validation error."""
        with pytest.raises(ValueError, match="At least one turn must be provided"):
            MultiTurnCustomData(session_id="empty_session", turns=[])

    def test_validation_turn_constraints_preserved(self):
        """Test that individual turn validation constraints are preserved."""
        # Test that turns still require at least one modality
        with pytest.raises(ValueError, match="At least one modality"):
            invalid_turn = SingleTurnCustomData()
            MultiTurnCustomData(session_id="invalid_session", turns=[invalid_turn])

        # Test that timestamp/delay mutual exclusion is preserved
        with pytest.raises(
            ValueError, match="timestamp and delay cannot be set together"
        ):
            invalid_turn = SingleTurnCustomData(text="Test", timestamp=1000, delay=500)
            MultiTurnCustomData(session_id="invalid_session", turns=[invalid_turn])


class TestMultiTurnDatasetLoader:
    """Tests for MultiTurnDatasetLoader functionality."""

    def test_load_simple_conversation(self, create_jsonl_file):
        """Test loading a simple multi-turn conversation."""
        content = [
            """{
                "session_id": "conv_001",
                "turns": [
                    {"text": "Hello, how are you?"},
                    {"text": "I'm doing well, thanks!", "delay": 1000}
                ]
            }"""
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(filename)
        result = loader.load_dataset()

        assert len(result) == 1
        assert "conv_001" in result

        conversation = result["conv_001"][0]
        assert isinstance(conversation, MultiTurnCustomData)
        assert len(conversation.turns) == 2
        assert conversation.turns[0].text == "Hello, how are you?"
        assert conversation.turns[1].text == "I'm doing well, thanks!"
        assert conversation.turns[1].delay == 1000

    def test_load_multiple_conversations(self, create_jsonl_file):
        """Test loading multiple conversations from file."""
        content = [
            """{
                "session_id": "session_A",
                "turns": [
                    {"text": "First conversation start"}
                ]
            }""",
            """{
                "session_id": "session_B",
                "turns": [
                    {"text": "Second conversation start"},
                    {"text": "Second conversation continues", "delay": 2000}
                ]
            }""",
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(filename)
        result = loader.load_dataset()

        assert len(result) == 2
        assert "session_A" in result
        assert "session_B" in result
        assert len(result["session_A"][0].turns) == 1
        assert len(result["session_B"][0].turns) == 2

    def test_load_conversation_without_session_id(self, create_jsonl_file):
        """Test loading conversation without explicit session_id generates UUID."""
        content = [
            """{
                "turns": [
                    {"text": "Anonymous conversation"},
                    {"text": "Should get auto-generated session_id"}
                ]
            }"""
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(filename)
        result = loader.load_dataset()

        assert len(result) == 1
        session_id = list(result.keys())[0]
        # Should be a UUID string (36 characters with hyphens)
        assert len(session_id) == 36
        assert session_id.count("-") == 4

        conversation = result[session_id][0]
        assert len(conversation.turns) == 2

    def test_load_multimodal_conversation(self, create_jsonl_file):
        """Test loading conversation with multimodal content."""
        content = [
            """{
                "session_id": "multimodal_chat",
                "turns": [
                    {
                        "text": "What do you see?",
                        "image": "/path/to/image.jpg"
                    },
                    {
                        "text": "Can you hear this?",
                        "audio": "/path/to/sound.wav",
                        "delay": 3000
                    }
                ]
            }"""
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(filename)
        result = loader.load_dataset()

        conversation = result["multimodal_chat"][0]
        assert conversation.turns[0].text == "What do you see?"
        assert conversation.turns[0].image == "/path/to/image.jpg"
        assert conversation.turns[1].text == "Can you hear this?"
        assert conversation.turns[1].audio == "/path/to/sound.wav"
        assert conversation.turns[1].delay == 3000

    def test_load_scheduled_conversation(self, create_jsonl_file):
        """Test loading conversation with timestamp scheduling."""
        content = [
            """{
                "session_id": "scheduled_chat",
                "turns": [
                    {"text": "Message at start", "timestamp": 0},
                    {"text": "Message after 5 seconds", "timestamp": 5000},
                    {"text": "Final message", "timestamp": 10000}
                ]
            }"""
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(filename)
        result = loader.load_dataset()

        conversation = result["scheduled_chat"][0]
        timestamps = [turn.timestamp for turn in conversation.turns]
        assert timestamps == [0, 5000, 10000]

    def test_load_batched_conversation(self, create_jsonl_file):
        """Test loading conversation with batched content."""
        content = [
            """{
                "session_id": "batched_chat",
                "turns": [
                    {
                        "text": ["Hello", "How are you?"],
                        "image": ["/img1.png", "/img2.png"]
                    },
                    {
                        "text": ["Fine", "Thanks"],
                        "delay": 1500
                    }
                ]
            }"""
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(filename)
        result = loader.load_dataset()

        conversation = result["batched_chat"][0]
        assert conversation.turns[0].text == ["Hello", "How are you?"]
        assert conversation.turns[0].image == ["/img1.png", "/img2.png"]
        assert conversation.turns[1].text == ["Fine", "Thanks"]

    def test_load_full_featured_conversation(self, create_jsonl_file):
        """Test loading conversation with full-featured format."""
        content = [
            """{
                "session_id": "full_featured_chat",
                "turns": [
                    {
                        "text": [
                            {"name": "user_query", "content": ["Analyze this data"]},
                            {"name": "user_context", "content": ["Be thorough"]}
                        ],
                        "image": [
                            {"name": "dataset_viz", "content": ["/chart.png"]},
                            {"name": "raw_data", "content": ["/data.png"]}
                        ],
                        "timestamp": 1000
                    }
                ]
            }"""
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(filename)
        result = loader.load_dataset()

        conversation = result["full_featured_chat"][0]
        turn = conversation.turns[0]

        assert len(turn.text) == 2
        assert len(turn.image) == 2
        assert turn.text[0].name == "user_query"
        assert turn.text[0].content == ["Analyze this data"]
        assert turn.image[0].name == "dataset_viz"
        assert turn.timestamp == 1000

    def test_load_dataset_skips_empty_lines(self, create_jsonl_file):
        """Test that empty lines are skipped during loading."""
        content = [
            """{
                "session_id": "test_empty_lines",
                "turns": [{"text": "First"}]
            }""",
            "",  # Empty line
            """{
                "session_id": "test_empty_lines_2",
                "turns": [{"text": "Second"}]
            }""",
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(filename)
        result = loader.load_dataset()

        assert len(result) == 2  # Should skip empty line
        assert "test_empty_lines" in result
        assert "test_empty_lines_2" in result

    def test_load_duplicate_session_ids_are_grouped(self, create_jsonl_file):
        """Test that multiple conversations with same session_id are grouped together."""
        content = [
            """{
                "session_id": "shared_session",
                "turns": [{"text": "First conversation"}]
            }""",
            """{
                "session_id": "shared_session",
                "turns": [{"text": "Second conversation"}]
            }""",
        ]
        filename = create_jsonl_file(content)

        loader = MultiTurnDatasetLoader(filename)
        result = loader.load_dataset()

        assert len(result) == 1  # Same session_id groups together
        assert len(result["shared_session"]) == 2  # Two conversations in same session

        conversations = result["shared_session"]
        assert conversations[0].turns[0].text == "First conversation"
        assert conversations[1].turns[0].text == "Second conversation"

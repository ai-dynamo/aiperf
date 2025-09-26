# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for dataset manager testing.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.dataset_manager import DatasetManager


@pytest.fixture
def user_config(tmp_path: Path) -> UserConfig:
    """Create a UserConfig for testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
        ),
        output=OutputConfig(artifact_directory=tmp_path),
    )


@pytest.fixture
def sample_conversations() -> dict[str, Conversation]:
    """Create sample conversations for testing."""
    conversations = {
        "session_1": Conversation(
            session_id="session_1",
            turns=[
                Turn(
                    texts=[Text(contents=["Hello, world!"])],
                    role="user",
                    model="test-model",
                ),
                Turn(
                    texts=[Text(contents=["How can I help you?"])],
                    role="assistant",
                    model="test-model",
                ),
            ],
        ),
        "session_2": Conversation(
            session_id="session_2",
            turns=[
                Turn(
                    texts=[Text(contents=["What is AI?"])],
                    role="user",
                    model="test-model",
                    max_tokens=100,
                ),
            ],
        ),
    }
    return conversations


@pytest.fixture
def empty_dataset_manager(
    user_config: UserConfig,
) -> DatasetManager:
    """Create a DatasetManager instance with empty dataset."""
    manager = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )
    manager.dataset = {}
    return manager


@pytest.fixture
def populated_dataset_manager(
    user_config: UserConfig,
    sample_conversations: dict[str, Conversation],
) -> DatasetManager:
    """Create a DatasetManager instance with sample data."""
    manager = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )
    manager.dataset = sample_conversations
    return manager


@pytest.fixture
def capture_file_writes():
    """Provide a fixture to capture file write operations for testing purposes."""

    class FileWriteCapture:
        def __init__(self):
            self.written_content = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def write(self, content: str):
            self.written_content = content

    class AsyncContextManager:
        def __init__(self, capture):
            self.capture = capture

        async def __aenter__(self):
            return self.capture

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    capture = FileWriteCapture()

    def mock_open(file_path, mode="r"):
        return AsyncContextManager(capture)

    with patch("aiperf.dataset.dataset_manager.aiofiles.open", mock_open):
        yield capture

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.dataset_models import Conversation, Text, Turn
from aiperf.common.messages import (
    ConversationRequestMessage,
    DatasetTimingRequest,
)
from aiperf.tests.utils.async_test_utils import async_fixture

# ============================================================================
# Dataset Manager Test Fixtures
# ============================================================================


@pytest.fixture
def sample_conversations():
    """Pre-built conversation objects with various structures for testing."""
    conversations = []

    # Conversation 1: Single turn with timestamp
    conv1 = Conversation(session_id="session-1")
    turn1 = Turn(
        timestamp=1000, text=[Text(name="text", content=["Hello, how are you?"])]
    )
    conv1.turns.append(turn1)
    conversations.append(conv1)

    # Conversation 2: Multiple turns with timestamps and delays
    conv2 = Conversation(session_id="session-2")
    turn2a = Turn(timestamp=2000, text=[Text(name="text", content=["First message"])])
    turn2b = Turn(
        timestamp=2500, delay=500, text=[Text(name="text", content=["Second message"])]
    )
    conv2.turns.extend([turn2a, turn2b])
    conversations.append(conv2)

    return conversations


@pytest.fixture
def conversation_request_message():
    """Valid ConversationRequestMessage instances for testing."""
    return ConversationRequestMessage(
        service_id="test-requester", request_id="req-123", conversation_id="session-1"
    )


@pytest.fixture
def timing_request_message():
    """Valid DatasetTimingRequest instances for testing."""
    return DatasetTimingRequest(
        service_id="test-requester", request_id="timing-req-456"
    )


@pytest.fixture
async def populated_dataset_manager(initialized_service, sample_conversations):
    """Pre-configured DatasetManager with sample data for testing."""
    manager = await async_fixture(initialized_service)
    manager.dataset = {conv.session_id: conv for conv in sample_conversations}
    return manager

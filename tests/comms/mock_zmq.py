# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Create mock modules to prevent actual ZMQ imports before any real imports happen
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from aiperf.common.enums import MessageType
from aiperf.common.messages import Message
from aiperf.zmq.zmq_comms import BaseZMQCommunication


class MockCommunicationData(BaseModel):
    """Data structure to hold state information for mock communication objects."""

    published_messages: dict[MessageType, list[Message]] = Field(default_factory=dict)
    subscriptions: dict[str, Callable[[Message], Coroutine[Any, Any, None]]] = Field(
        default_factory=dict
    )
    pull_callbacks: dict[MessageType, Callable[[Message], None]] = Field(
        default_factory=dict
    )
    push_messages: dict[MessageType, Message] = Field(default_factory=dict)
    requests: dict[str, Message] = Field(default_factory=dict)
    responses: dict[str, Message] = Field(default_factory=dict)

    def clear(self) -> None:
        self.published_messages.clear()
        self.subscriptions.clear()
        self.pull_callbacks.clear()
        self.push_messages.clear()
        self.requests.clear()
        self.responses.clear()


@pytest.fixture
def mock_zmq_communication() -> MagicMock:
    """
    Create a mock communication object for testing service communication.

    This mock tracks published messages, subscriptions, pull callbacks,
    push messages, and requests and responses for verification in tests.

    Returns:
        A MagicMock configured to behave like ZMQCommunication
    """
    mock_comm = MagicMock(spec=BaseZMQCommunication)

    # Configure basic behavior
    mock_comm.initialize.return_value = None
    mock_comm.shutdown.return_value = None

    mock_comm.mock_data = MockCommunicationData()

    return mock_comm

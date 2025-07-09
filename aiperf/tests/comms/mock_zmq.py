# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Create mock modules to prevent actual ZMQ imports before any real imports happen
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from aiperf.common.comms.zmq import BaseZMQCommunication
from aiperf.common.enums import MessageType, ServiceState, ServiceType
from aiperf.common.messages import Message, StatusMessage


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

    async def mock_publish(topic: MessageType, message: Message) -> None:
        """Mock implementation of publish that stores messages by topic."""
        if topic not in mock_comm.mock_data.published_messages:
            mock_comm.mock_data.published_messages[topic] = []

        mock_comm.mock_data.published_messages[topic].append(message)

    mock_comm.publish.side_effect = mock_publish

    async def mock_subscribe(
        topic: str, callback: Callable[[Message], Coroutine[Any, Any, None]]
    ) -> None:
        """Mock implementation of subscribe that stores callbacks by topic."""
        mock_comm.mock_data.subscriptions[topic] = callback

    mock_comm.subscribe.side_effect = mock_subscribe

    async def mock_pull(
        message_type: MessageType, callback: Callable[[Message], None]
    ) -> None:
        """Mock implementation of pull that stores callbacks by topic."""
        mock_comm.mock_data.pull_callbacks[message_type] = callback

    mock_comm.register_pull_callback.side_effect = mock_pull

    async def mock_push(topic: MessageType, message: Message) -> None:
        """Mock implementation of push that stores messages by topic."""
        mock_comm.mock_data.push_messages[topic] = message

    mock_comm.push.side_effect = mock_push

    async def mock_request(target: str, request_data: Message) -> Message:
        """Mock implementation of request that stores requests by target."""
        mock_comm.mock_data.requests[target] = request_data

        # Return a fake mock response
        return StatusMessage(
            service_id="mock_service_id",
            service_type=ServiceType.TEST,
            state=ServiceState.READY,
        )

    mock_comm.request.side_effect = mock_request

    return mock_comm

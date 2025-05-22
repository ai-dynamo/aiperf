#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# Create mock modules to prevent actual ZMQ imports before any real imports happen
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from aiperf.common.comms.zmq import ZMQCommunication
from aiperf.common.enums.comms import Topic
from aiperf.common.models.message import BaseMessage, Message
from aiperf.common.models.payload import DataPayload


class MockCommunicationData(BaseModel):
    """Data structure to hold state information for mock communication objects."""

    published_messages: dict[Topic, list[Message]] = Field(default_factory=dict)
    subscriptions: dict[str, Callable[[Message], Coroutine[Any, Any, None]]] = Field(
        default_factory=dict
    )
    pull_callbacks: dict[Topic, Callable[[Message], None]] = Field(default_factory=dict)
    push_messages: dict[Topic, Message] = Field(default_factory=dict)
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
    mock_comm = MagicMock(spec=ZMQCommunication)

    # Configure basic behavior
    mock_comm.initialize.return_value = None
    mock_comm.shutdown.return_value = None
    mock_comm.create_clients.return_value = None

    mock_comm.mock_data = MockCommunicationData()

    async def mock_publish(topic: Topic, message: Message) -> None:
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

    async def mock_pull(topic: Topic, callback: Callable[[Message], None]) -> None:
        """Mock implementation of pull that stores callbacks by topic."""
        mock_comm.mock_data.pull_callbacks[topic] = callback

    mock_comm.pull.side_effect = mock_pull

    async def mock_push(topic: Topic, message: Message) -> None:
        """Mock implementation of push that stores messages by topic."""
        mock_comm.mock_data.push_messages[topic] = message

    mock_comm.push.side_effect = mock_push

    async def mock_request(target: str, request_data: Message) -> Message:
        """Mock implementation of request that stores requests by target."""
        mock_comm.mock_data.requests[target] = request_data

        # Return a fake mock response
        return BaseMessage(
            payload=DataPayload(),
        )

    mock_comm.request.side_effect = mock_request

    async def mock_respond(target: str, response: Message) -> None:
        """Mock implementation of respond that stores responses by target."""
        mock_comm.mock_data.responses[target] = response

    mock_comm.respond.side_effect = mock_respond

    return mock_comm

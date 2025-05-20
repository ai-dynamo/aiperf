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
"""
Shared fixtures for testing AIPerf services.

This file contains fixtures that are automatically discovered by pytest
and made available to test functions in the same directory and subdirectories.
"""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.tests.comms.mock_zmq import (
    mock_zmq_communication,  # noqa: F401 : used as a fixture
)


@pytest.fixture
def mock_zmq_socket() -> Generator[MagicMock, None, None]:
    """Fixture to provide a mock ZMQ socket."""
    zmq_socket = MagicMock()
    with patch("zmq.Socket", new_callable=zmq_socket):
        yield zmq_socket


@pytest.fixture
def mock_zmq_context() -> Generator[MagicMock, None, None]:
    """Fixture to provide a mock ZMQ context."""
    zmq_context = MagicMock()
    zmq_context.socket.return_value = mock_zmq_socket()

    with patch("zmq.Context", new_callable=zmq_context):
        yield zmq_context


@pytest.fixture
def mock_communication(mock_zmq_communication: MagicMock) -> MagicMock:  # noqa: F811 : used as a fixture
    """
    Create a mock communication object for testing service communication.

    This mock tracks published messages and subscriptions for verification in tests.

    Returns:
        An MagicMock configured to behave like ZMQCommunication
    """
    return mock_zmq_communication

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing AIPerf services.

This file contains fixtures that are automatically discovered by pytest
and made available to test functions in the same directory and subdirectories.
"""

import logging
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.tokenizer import Tokenizer
from aiperf.tests.comms.mock_zmq import (
    mock_zmq_communication,  # noqa: F401 : used as a fixture
)

logging.basicConfig(level=logging.DEBUG)


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


@pytest.fixture
def mock_tokenizer_cls() -> type[Tokenizer]:
    """Mock our Tokenizer class to avoid HTTP requests during testing.

    This fixture patches AutoTokenizer.from_pretrained and provides a realistic
    mock tokenizer that can encode, decode, and handle special tokens.

    Usage in tests:
        def test_something(mock_tokenizer_cls):
            tokenizer = mock_tokenizer_cls.from_pretrained("any-model-name")
            # tokenizer is now mocked and won't make HTTP requests
    """

    class MockTokenizer(Tokenizer):
        """A thin mocked wrapper around AIPerf Tokenizer for testing."""

        def __init__(self, mock_tokenizer: MagicMock):
            super().__init__()
            self._tokenizer = mock_tokenizer

            # Create MagicMock methods that you can assert on
            self.encode = MagicMock(side_effect=self._mock_encode)
            self.decode = MagicMock(side_effect=self._mock_decode)

        @classmethod
        def from_pretrained(
            cls, name: str, trust_remote_code: bool = False, revision: str = "main"
        ):
            # Create a mock tokenizer around HF AutoTokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.bos_token_id = 1
            return cls(mock_tokenizer)

        def __call__(self, text, **kwargs):
            return self._mock_call(text, **kwargs)

        def _mock_call(self, text, **kwargs):
            base_tokens = list(range(10, 10 + len(text.split())))
            return {"input_ids": base_tokens}

        def _mock_encode(self, text, **kwargs):
            return self._mock_call(text, **kwargs)["input_ids"]

        def _mock_decode(self, token_ids, **kwargs):
            return " ".join([f"token_{t}" for t in token_ids])

    return MockTokenizer

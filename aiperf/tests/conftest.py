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
def mock_hf_tokenizer() -> Generator[MagicMock, None, None]:
    """Mock Hugging Face tokenizer to avoid HTTP requests during testing.

    This fixture patches AutoTokenizer.from_pretrained and provides a realistic
    mock tokenizer that can encode, decode, and handle special tokens.

    Usage in tests:
        def test_something(mock_hf_tokenizer):
            tokenizer = Tokenizer.from_pretrained("any-model-name")
            # tokenizer is now mocked and won't make HTTP requests
    """
    # Create a mock tokenizer with realistic behavior
    mock_tokenizer = MagicMock()
    mock_tokenizer.bos_token_id = 1

    def mock_call(text, **kwargs):
        base_tokens = list(range(10, 10 + len(text.split())))
        return {"input_ids": base_tokens}

    def mock_encode(text, **kwargs):
        return mock_call(text, **kwargs)["input_ids"]

    def mock_decode(token_ids, **kwargs):
        return " ".join([str(t) for t in token_ids])

    mock_tokenizer.side_effect = mock_call
    mock_tokenizer.encode = mock_encode
    mock_tokenizer.decode = mock_decode

    with patch(
        "aiperf.common.tokenizer.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    ):
        yield mock_tokenizer

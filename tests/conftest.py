# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing AIPerf services.

This file contains fixtures that are automatically discovered by pytest
and made available to test functions in the same directory and subdirectories.
"""

import asyncio
import logging
from collections.abc import Callable, Generator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.aiperf_logger import _TRACE
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceRunType
from aiperf.common.enums.communication_enums import CommunicationBackend
from aiperf.common.messages import Message
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.types import MessageTypeT
from tests.comms.mock_zmq import (
    mock_zmq_communication as mock_zmq_communication,  # import fixture globally
)

real_sleep = (
    asyncio.sleep
)  # save the real sleep so we can use it in the no_sleep fixture
from tests.utils.time_traveler import (  # noqa: E402
    time_traveler as time_traveler,  # import fixture globally
)

logging.basicConfig(level=_TRACE)


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests (disabled by default)",
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (disabled by default)",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests (disabled by default, use --performance to enable)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (disabled by default, use --integration to enable)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip performance and integration tests unless their respective options are given."""
    performance_enabled = config.getoption("--performance")
    integration_enabled = config.getoption("--integration")

    skip_performance = pytest.mark.skip(
        reason="performance tests disabled (use --performance to enable)"
    )
    skip_integration = pytest.mark.skip(
        reason="integration tests disabled (use --integration to enable)"
    )

    for item in items:
        if "performance" in item.keywords and not performance_enabled:
            item.add_marker(skip_performance)
        if "integration" in item.keywords and not integration_enabled:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch) -> None:
    """
    Patch asyncio.sleep with a no-op to prevent test delays.

    This ensures tests don't need to wait for real sleep calls.
    """

    async def fast_sleep(*args, **kwargs):
        await real_sleep(
            0
        )  # relinquish time slice to other tasks to avoid blocking the event loop

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)


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


@pytest.fixture
def user_config() -> UserConfig:
    config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
    return config


@pytest.fixture
def service_config() -> ServiceConfig:
    return ServiceConfig(
        service_run_type=ServiceRunType.MULTIPROCESSING,
        comm_backend=CommunicationBackend.ZMQ_IPC,
    )


class MockPubClient:
    """Mock pub client."""

    def __init__(self):
        self.publish_calls = []

    async def publish(self, message: Message) -> None:
        self.publish_calls.append(message)


@pytest.fixture
def mock_pub_client() -> MockPubClient:
    """Create a mock pub client."""
    return MockPubClient()


class MockSubClient:
    """Mock sub client."""

    def __init__(self):
        self.subscribe_calls = []
        self.subscribe_all_calls = []

    async def subscribe(
        self, message_type: MessageTypeT, callback: Callable[[Message], None]
    ) -> None:
        self.subscribe_calls.append((message_type, callback))

    async def subscribe_all(
        self, message_callback_map: dict[MessageTypeT, Callable[[Message], None]]
    ) -> None:
        self.subscribe_all_calls.append(message_callback_map)


@pytest.fixture
def mock_sub_client() -> MockSubClient:
    """Create a mock sub client."""
    return MockSubClient()

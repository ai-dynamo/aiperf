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
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aiperf.common.aiperf_logger import _TRACE
from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.enums import CommunicationBackend, ServiceRunType
from aiperf.common.messages import Message
from aiperf.common.models import (
    Conversation,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
    Text,
    TextResponseData,
    Turn,
)
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

# Shared test constants for request/response records
DEFAULT_START_TIME_NS = 1_000_000
DEFAULT_FIRST_RESPONSE_NS = 1_050_000
DEFAULT_LAST_RESPONSE_NS = 1_100_000
DEFAULT_INPUT_TOKENS = 5
DEFAULT_OUTPUT_TOKENS = 2


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
            mock_tokenizer.eos_token_id = 2
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


@pytest.fixture
def create_mooncake_trace_file():
    """Create a temporary mooncake trace file with custom content."""
    import tempfile
    from pathlib import Path

    filenames = []

    def _create_file(entries_or_count, include_timestamps=None):
        """Create a mooncake trace file.

        Args:
            entries_or_count: Either a list of JSON string entries, or an integer count
            include_timestamps: Only used when entries_or_count is an integer.
                               If True, adds timestamps to generated entries.
                               If False, omits timestamps.
                               If None, entries are used as-is.

        Returns:
            str: Path to the created temporary file
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            if isinstance(entries_or_count, int):
                # Generate entries based on count
                entry_count = entries_or_count
                for i in range(entry_count):
                    if include_timestamps is True:
                        entry = f'{{"input_length": {100 + i * 50}, "hash_ids": [{i}], "timestamp": {1000 + i * 1000}}}'
                    elif include_timestamps is False:
                        entry = f'{{"input_length": {100 + i * 50}, "hash_ids": [{i}]}}'
                    else:
                        # Default behavior when include_timestamps is None
                        entry = f'{{"input_length": {100 + i * 50}, "hash_ids": [{i}]}}'
                    f.write(f"{entry}\n")
            else:
                # Use provided entries list
                for entry in entries_or_count:
                    f.write(f"{entry}\n")

            filename = f.name
            filenames.append(filename)
            return filename

    yield _create_file

    # Cleanup all created files
    for filename in filenames:
        Path(filename).unlink(missing_ok=True)


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
def sample_request_record() -> RequestRecord:
    """Create a sample RequestRecord for testing."""
    return RequestRecord(
        conversation_id="test-conversation",
        turn_index=0,
        model_name="test-model",
        start_perf_ns=DEFAULT_START_TIME_NS,
        timestamp_ns=DEFAULT_START_TIME_NS,
        end_perf_ns=DEFAULT_LAST_RESPONSE_NS,
        error=None,
    )


@pytest.fixture
def sample_parsed_record(sample_request_record: RequestRecord) -> ParsedResponseRecord:
    """Create a valid ParsedResponseRecord for testing."""
    responses = [
        ParsedResponse(
            perf_ns=DEFAULT_FIRST_RESPONSE_NS,
            data=TextResponseData(text="Hello"),
        ),
        ParsedResponse(
            perf_ns=DEFAULT_LAST_RESPONSE_NS,
            data=TextResponseData(text=" world"),
        ),
    ]

    return ParsedResponseRecord(
        request=sample_request_record,
        responses=responses,
        input_token_count=DEFAULT_INPUT_TOKENS,
        output_token_count=DEFAULT_OUTPUT_TOKENS,
    )


@pytest.fixture
def mock_aiofiles_stringio():
    """Mock aiofiles.open to write to a StringIO buffer instead of a file.

    Automatically patches aiofiles.open for the duration of the test.

    Returns:
        StringIO: Buffer that captures all writes

    Example:
        def test_something(mock_aiofiles_stringio):
            # aiofiles.open is already patched
            # ... test code that writes to files ...

            # Verify contents
            contents = mock_aiofiles_stringio.getvalue()
            assert "expected" in contents
    """
    string_buffer = StringIO()

    mock_file = AsyncMock()
    mock_file.write = AsyncMock(side_effect=lambda data: string_buffer.write(data))
    mock_file.flush = AsyncMock()
    mock_file.close = AsyncMock()

    async def mock_aiofiles_open(*args, **kwargs):
        return mock_file

    with patch("aiofiles.open", side_effect=mock_aiofiles_open):
        yield string_buffer


@pytest.fixture
def mock_macos_child_process():
    """Mock for simulating a child process on macOS."""
    mock_process = Mock()
    mock_process.name = "DATASET_MANAGER_process"  # Not MainProcess
    return mock_process


@pytest.fixture
def mock_macos_main_process():
    """Mock for simulating the main process on macOS."""
    mock_process = Mock()
    mock_process.name = "MainProcess"
    return mock_process

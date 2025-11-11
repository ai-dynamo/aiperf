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
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import zmq.asyncio

from aiperf.common import random_generator as rng
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
from aiperf.module_loader import ensure_modules_loaded


@pytest.fixture(autouse=True, scope="function")
def mock_zmq_globally(monkeypatch):
    """
    Globally mock ZMQ for all tests to prevent real socket/context creation.

    This fixture runs automatically for every test across the entire test suite.
    It prevents ZMQ from creating real sockets and contexts which could cause
    resource leaks, port conflicts, and test failures.

    Tests in tests/zmq/ will use their own more specific mocking from tests/zmq/conftest.py.
    """

    async def _block_forever():
        """Block forever by awaiting a Future that never completes."""
        await asyncio.Future()  # Never resolves

    # Create mock socket
    mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
    mock_socket.bind = Mock()
    mock_socket.connect = Mock()
    mock_socket.close = Mock()
    mock_socket.setsockopt = Mock()
    mock_socket.send = AsyncMock()
    mock_socket.send_multipart = AsyncMock()
    # Block forever instead of raising zmq.Again in a loop
    mock_socket.recv = AsyncMock(side_effect=_block_forever)
    mock_socket.recv_multipart = AsyncMock(side_effect=_block_forever)
    mock_socket.closed = False

    # Create mock context
    mock_context = MagicMock(spec=zmq.asyncio.Context)
    mock_context.socket = Mock(return_value=mock_socket)
    mock_context.term = Mock()

    # Mock Context.instance() to return our mock context
    monkeypatch.setattr("zmq.asyncio.Context.instance", lambda: mock_context)

    return mock_context


real_sleep = (
    asyncio.sleep
)  # save the real sleep so we can use it in the no_sleep fixture
from tests.unit.utils.time_traveler import (  # noqa: E402
    time_traveler as time_traveler,  # import fixture globally
)

logging.basicConfig(level=_TRACE)

# Shared test constants for request/response records
DEFAULT_START_TIME_NS = 1_000_000
DEFAULT_FIRST_RESPONSE_NS = 1_050_000
DEFAULT_LAST_RESPONSE_NS = 1_100_000
DEFAULT_INPUT_TOKENS = 5
DEFAULT_OUTPUT_TOKENS = 2


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


@pytest.fixture(scope="session", autouse=True)
def load_aiperf_modules() -> None:
    """Load all AIPerf modules for testing.

    This fixture is automatically used for all tests and ensures that all AIPerf
    modules are loaded for registration purposes.
    """
    ensure_modules_loaded()


@pytest.fixture(autouse=True)
def reset_random_generator() -> Generator[None, None, None]:
    """Reset and seed the global random generator for each test.

    This fixture is automatically used for all tests and ensures that:
    1. Each test starts with a fresh random generator state
    2. The random generator is seeded with a fixed value for reproducibility
    3. The state is cleaned up after each test to prevent leakage

    This ensures all tests have consistent, reproducible random behavior.
    """
    # Reset and seed before each test
    rng.reset()
    rng.init(42)  # Use a fixed seed for test reproducibility

    yield  # Run the test

    # Reset after each test to ensure clean state
    rng.reset()


@pytest.fixture(autouse=True)
def reset_singleton_factories():
    """Reset singleton factory instances between tests to prevent state leakage.

    This fixture runs automatically for every test and clears the singleton
    instances in factories like CommunicationFactory. This prevents tests from
    interfering with each other when they create services that use singleton
    communication instances.

    The error "Communication clients must be created before the ZMQIPCCommunication
    class is initialized" occurs when a singleton instance from a previous test
    is reused in an invalid state.
    """
    yield  # Run the test first

    # Clean up after test completes
    from aiperf.common.factories import AIPerfUIFactory, CommunicationFactory

    if hasattr(CommunicationFactory, "_instances"):
        CommunicationFactory._instances.clear()
    if hasattr(CommunicationFactory, "_instances_pid"):
        CommunicationFactory._instances_pid.clear()
    if hasattr(AIPerfUIFactory, "_instances"):
        AIPerfUIFactory._instances.clear()
    if hasattr(AIPerfUIFactory, "_instances_pid"):
        AIPerfUIFactory._instances_pid.clear()


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
    """Mock aiofiles.open to write to a BytesIO buffer instead of a file.

    Automatically patches aiofiles.open for the duration of the test.

    Returns:
        BytesIO: Buffer that captures all writes

    Example:
        def test_something(mock_aiofiles_stringio):
            # aiofiles.open is already patched
            # ... test code that writes to files ...

            # Verify contents
            contents = mock_aiofiles_stringio.getvalue()
            assert b"expected" in contents
    """
    string_buffer = BytesIO()

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


@pytest.fixture
def mock_platform_system():
    """Mock platform.system() for testing OS-specific behavior."""
    with patch("platform.system") as mock:
        yield mock


@pytest.fixture
def mock_platform_darwin(mock_platform_system):
    """Mock platform.system() to return 'Darwin' for macOS testing."""
    mock_platform_system.return_value = "Darwin"
    return mock_platform_system


@pytest.fixture
def mock_platform_linux(mock_platform_system):
    """Mock platform.system() to return 'Linux' for Linux testing."""
    mock_platform_system.return_value = "Linux"
    return mock_platform_system


@pytest.fixture
def mock_multiprocessing_set_start_method():
    """Mock multiprocessing.set_start_method() for testing spawn method setup."""
    with patch("multiprocessing.set_start_method") as mock:
        yield mock


@pytest.fixture
def mock_bootstrap_and_run_service():
    """Mock aiperf.common.bootstrap.bootstrap_and_run_service() for testing."""
    with patch("aiperf.common.bootstrap.bootstrap_and_run_service") as mock:
        yield mock


@pytest.fixture
def mock_get_global_log_queue():
    """Mock aiperf.common.logging.get_global_log_queue() for testing."""
    with patch("aiperf.common.logging.get_global_log_queue") as mock:
        yield mock


@pytest.fixture
def mock_psutil_process():
    """Mock psutil.Process for testing."""
    with patch("psutil.Process") as mock:
        yield mock


@pytest.fixture
def mock_setup_child_process_logging():
    """Mock aiperf.common.logging.setup_child_process_logging() for testing."""
    with patch("aiperf.common.logging.setup_child_process_logging") as mock:
        yield mock


@pytest.fixture
def mock_current_process():
    """Mock multiprocessing.current_process() for testing."""
    with patch("multiprocessing.current_process") as mock:
        yield mock


@pytest.fixture
def mock_darwin_child_process(
    mock_platform_darwin, mock_current_process, mock_macos_child_process
):
    """Mock macOS child process environment (Darwin platform + child process)."""
    mock_current_process.return_value = mock_macos_child_process
    return mock_current_process


@pytest.fixture
def mock_darwin_main_process(
    mock_platform_darwin, mock_current_process, mock_macos_main_process
):
    """Mock macOS main process environment (Darwin platform + main process)."""
    mock_current_process.return_value = mock_macos_main_process
    return mock_current_process


@pytest.fixture
def mock_linux_child_process(
    mock_platform_linux, mock_current_process, mock_macos_child_process
):
    """Mock Linux child process environment (Linux platform + child process)."""
    mock_current_process.return_value = mock_macos_child_process
    return mock_current_process

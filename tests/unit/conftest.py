# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing AIPerf services.

This file contains fixtures that are automatically discovered by pytest
and made available to test functions in the same directory and subdirectories.
"""

import asyncio
from collections.abc import Callable, Generator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import zmq.asyncio

from aiperf.common import random_generator as rng
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.messages import Message
from aiperf.common.models import (
    Conversation,
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
    Text,
    TextResponseData,
    Turn,
)
from aiperf.common.models.record_models import TokenCounts
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.types import MessageTypeT
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.plugin.enums import CommunicationBackend, ServiceRunType
from aiperf.plugin.plugins import _PluginRegistry as PluginRegistry
from tests.harness.fake_tokenizer import FakeTokenizer
from tests.harness.time_traveler import TimeTraveler

# Shared test constants for request/response records
DEFAULT_START_TIME_NS = 1_000_000
DEFAULT_FIRST_RESPONSE_NS = 1_050_000
DEFAULT_LAST_RESPONSE_NS = 1_100_000
DEFAULT_INPUT_TOKENS = 5
DEFAULT_OUTPUT_TOKENS = 2

_REAL_SLEEP = asyncio.sleep


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch, request):
    """Patch asyncio.sleep to do nothing, unless test uses time_traveler fixture.

    Tests using time_traveler (or time_traveler_no_patch_sleep) need looptime
    to handle asyncio.sleep for virtual time advancement, so we skip patching.
    """
    # Check if test uses time_traveler fixtures (which need looptime to work)
    fixture_names = request.fixturenames
    uses_time_traveler = (
        "time_traveler" in fixture_names
        or "time_traveler_no_patch_sleep" in fixture_names
    )
    if not uses_time_traveler:
        monkeypatch.setattr("asyncio.sleep", lambda delay: _REAL_SLEEP(0))
    yield


@pytest.fixture
async def enable_looptime(request):
    """Enable looptime (virtual time).

    This fixture enables looptime on the event loop, making asyncio.sleep/wait_for
    run instantly in virtual time.

    Note: When @pytest.mark.looptime is used, the looptime plugin already enables
    looptime via its context manager. This fixture detects that and becomes a no-op
    to avoid the "already enabled" error.
    """
    # Check if @pytest.mark.looptime is used - if so, looptime plugin handles enabling
    looptime_marker = request.node.get_closest_marker("looptime")
    if looptime_marker is not None:
        # Looptime is already enabled by the plugin's context manager
        yield
        return

    # No marker - try to enable programmatically (fallback for tests without marker)
    try:
        loop = asyncio.get_running_loop()
        # Check if loop is looptime-patched and enable it if not already enabled
        if hasattr(loop, "looptime_on") and not loop.looptime_on:
            # Use the name-mangled private attribute to enable looptime
            # This is the internal flag that looptime checks to decide if it's enabled
            loop._LoopTimeEventLoop__enabled = True
    except RuntimeError:
        # No running loop yet - that's okay, looptime will be enabled when loop starts
        pass

    yield


@pytest.fixture
async def time_traveler(enable_looptime):
    """
    TimeTraveler fixture for virtual time testing.

    Provides:
    - Virtual time tracking (time.time(), time.perf_counter(), etc. return virtual time)
    - Timing assertion utilities (sleeps_for, sleeps_at_least, etc.)
    - Works with looptime

    Usage:
        async def test_timing(time_traveler):
            start = time_traveler.time()
            await asyncio.sleep(10.0)  # Instant in real time!
            elapsed = time_traveler.time() - start
            assert elapsed >= 10.0

        async def test_with_assertion(time_traveler):
            async with time_traveler.sleeps_for(5.0):
                await asyncio.sleep(5.0)
    """
    traveler = TimeTraveler()
    traveler.start_traveling()
    yield traveler
    traveler.stop_traveling()


@pytest.fixture
async def time_traveler_no_patch_sleep(enable_looptime):
    """
    TimeTraveler fixture for virtual time testing with real asyncio.sleep.

    Provides:
    - Virtual time tracking (time.time(), time.perf_counter(), etc. return virtual time)
    - Timing assertion utilities (sleeps_for, sleeps_at_least, etc.)
    - Works with looptime

    Usage:
        async def test_timing(time_traveler):
            start = time_traveler.time()
            await asyncio.sleep(10.0)  # Instant in real time!
            elapsed = time_traveler.time() - start
            assert elapsed >= 10.0

        async def test_with_assertion(time_traveler):
            async with time_traveler.sleeps_for(5.0):
                await asyncio.sleep(5.0)
    """
    traveler = TimeTraveler(patch_sleep=False)
    traveler.start_traveling()
    yield traveler
    traveler.stop_traveling()


@pytest.fixture
def fake_tokenizer():
    """Patch Tokenizer.from_pretrained to use FakeTokenizer."""
    with patch(
        "aiperf.common.tokenizer.Tokenizer.from_pretrained",
        FakeTokenizer.from_pretrained,
    ):
        yield


@pytest.fixture
def skip_service_registration():
    """Patch BaseComponentService._register_service_on_start to do nothing."""
    with patch.object(BaseComponentService, "_register_service_on_start", AsyncMock()):
        yield


@dataclass
class MockZmqFixture:
    """
    Container for mock ZMQ components with send capture and receive queues.

    Attributes:
        context: Mock ZMQ context returned by Context.instance().
        socket: Mock ZMQ socket created by context.socket().
        sent: List capturing all data passed to socket.send().
        sent_multipart: List capturing all parts passed to socket.send_multipart().
        recv_queue: Queue for injecting messages returned by socket.recv().
        recv_multipart_queue: Queue for injecting messages returned by socket.recv_multipart().
    """

    context: MagicMock
    socket: AsyncMock
    sent: list[bytes]
    sent_multipart: list[list[bytes]]
    recv_queue: asyncio.Queue[bytes]
    recv_multipart_queue: asyncio.Queue[list[bytes]]


@pytest.fixture
def mock_zmq(monkeypatch) -> MockZmqFixture:
    """
    Mock ZMQ to prevent real socket/context creation and enable message inspection.

    Prevents ZMQ from creating real sockets and contexts which could cause
    resource leaks, port conflicts, and test failures.

    Send operations are captured in lists. Receive operations pull from queues,
    blocking forever when empty (safe for tests that don't need to receive).

    Example:
    ```python
        async def test_zmq_echo(mock_zmq):
            mock_zmq.recv_queue.put_nowait(b"ping")
            # ... run code that receives and sends ...
            assert mock_zmq.sent == [b"pong"]
    ```
    Note:
        Tests in tests/zmq/ use their own specific mocking from tests/zmq/conftest.py.
    """
    mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
    mock_socket.bind = Mock()
    mock_socket.connect = Mock()
    mock_socket.close = Mock()
    mock_socket.setsockopt = Mock()
    mock_socket.closed = False

    mock_context = MagicMock(spec=zmq.asyncio.Context)
    mock_context.socket = Mock(return_value=mock_socket)
    mock_context.term = Mock()

    fixture = MockZmqFixture(
        context=mock_context,
        socket=mock_socket,
        sent=[],
        sent_multipart=[],
        recv_queue=asyncio.Queue(),
        recv_multipart_queue=asyncio.Queue(),
    )

    async def _capture_send(data: bytes, flags: int = 0) -> None:
        fixture.sent.append(data)

    async def _capture_send_multipart(parts: list[bytes], flags: int = 0) -> None:
        fixture.sent_multipart.append(list(parts))

    async def _recv(flags: int = 0) -> bytes:
        return await fixture.recv_queue.get()

    async def _recv_multipart(flags: int = 0) -> list[bytes]:
        return await fixture.recv_multipart_queue.get()

    mock_socket.send = AsyncMock(side_effect=_capture_send)
    mock_socket.send_multipart = AsyncMock(side_effect=_capture_send_multipart)
    mock_socket.recv = AsyncMock(side_effect=_recv)
    mock_socket.recv_multipart = AsyncMock(side_effect=_recv_multipart)

    monkeypatch.setattr("zmq.asyncio.Context.instance", lambda: mock_context)

    return fixture


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
    instances managed by the Singleton metaclass. This prevents tests from interfering
    with each other when they create services that use singleton communication instances.

    The error "Communication clients must be created before the ZMQIPCCommunication
    class is initialized" occurs when a singleton instance from a previous test
    is reused in an invalid state.
    """
    yield  # Run the test first

    # Clean up after test completes - clear per-process singleton instances
    from aiperf.common.singleton import SingletonMeta

    SingletonMeta._instances.clear()


@pytest.fixture
def temporary_registry() -> Generator[PluginRegistry, None, None]:
    """Fixture for isolated plugin registry testing.

    Creates a temporary registry for the duration of the test,
    then restores the original registry on exit.

    Yields:
        Fresh PluginRegistry instance

    Example:
        def test_custom_plugin(temporary_registry):
            temporary_registry.load_builtin_registry(test_registry_path)
            cls = temporary_registry.get('endpoint', 'test')
            # Original registry restored after test
    """
    # Save the current singleton instance
    old_instance = PluginRegistry._instance

    # Create a fresh singleton
    PluginRegistry._reset_singleton()
    fresh_registry = PluginRegistry()

    yield fresh_registry

    # Restore the original singleton
    PluginRegistry._instance = old_instance


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
def sample_request_info() -> RequestInfo:
    """Create a sample RequestInfo for testing."""
    from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
    from aiperf.common.models.model_endpoint_info import (
        EndpointInfo,
        ModelEndpointInfo,
        ModelInfo,
        ModelListInfo,
    )
    from aiperf.plugin.enums import EndpointType

    return RequestInfo(
        model_endpoint=ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
            ),
        ),
        turns=[
            Turn(
                texts=[Text(contents=["test prompt"])], role="user", model="test-model"
            )
        ],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="test-request-id",
        x_correlation_id="test-correlation-id",
        conversation_id="test-conversation",
    )


@pytest.fixture
def sample_request_record(sample_request_info: RequestInfo) -> RequestRecord:
    """Create a sample RequestRecord for testing."""
    return RequestRecord(
        request_info=sample_request_info,
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
        token_counts=TokenCounts(
            input=DEFAULT_INPUT_TOKENS,
            output=DEFAULT_OUTPUT_TOKENS,
            reasoning=None,
        ),
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


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    """Create a temporary artifact directory for testing."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def create_exporter_config(
    profile_results,
    user_config,
    telemetry_results=None,
    server_metrics_results=None,
    verbose=True,
):
    """Helper to create ExporterConfig with common defaults."""
    return ExporterConfig(
        results=profile_results,
        user_config=user_config,
        service_config=ServiceConfig(verbose=verbose),
        telemetry_results=telemetry_results,
        server_metrics_results=server_metrics_results,
    )

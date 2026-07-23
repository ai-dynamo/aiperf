# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for testing AIPerf services."""

import asyncio
import os
import uuid
from collections.abc import Callable, Generator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import zmq.asyncio

from aiperf.common import random_generator as rng
from aiperf.common.base_component_service import BaseComponentService
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
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.plugin.plugins import _PluginRegistry as PluginRegistry
from tests.harness.fake_tokenizer import FakeTokenizer
from tests.harness.optional_deps import collect_ignore_for_unavailable_deps
from tests.harness.time_traveler import TimeTraveler

collect_ignore: list[str] = collect_ignore_for_unavailable_deps(Path(__file__).parent)

DEFAULT_START_TIME_NS = 1_000_000
DEFAULT_FIRST_RESPONSE_NS = 1_050_000
DEFAULT_LAST_RESPONSE_NS = 1_100_000
DEFAULT_INPUT_TOKENS = 5
DEFAULT_OUTPUT_TOKENS = 2

_REAL_SLEEP = asyncio.sleep


@pytest.fixture(scope="session", autouse=True)
def _hf_offline_mode():
    """Never load a real tokenizer over the network in unit tests."""
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    prev = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch, request):
    """Patch asyncio.sleep to do nothing, unless test uses time_traveler fixture."""
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
    """Enable looptime (virtual time)."""
    looptime_marker = request.node.get_closest_marker("looptime")
    if looptime_marker is not None:
        yield
        return

    try:
        loop = asyncio.get_running_loop()
        if hasattr(loop, "looptime_on") and not loop.looptime_on:
            loop._LoopTimeEventLoop__enabled = True
    except RuntimeError:
        pass

    yield


@pytest.fixture
async def time_traveler(enable_looptime):
    """TimeTraveler fixture for virtual time testing."""
    traveler = TimeTraveler()
    traveler.start_traveling()
    yield traveler
    traveler.stop_traveling()


@pytest.fixture
async def time_traveler_no_patch_sleep(enable_looptime):
    """TimeTraveler fixture for virtual time testing with real asyncio.sleep."""
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
    """Container for mock ZMQ components with send capture and receive queues."""

    context: MagicMock
    socket: AsyncMock
    sent: list[bytes]
    sent_multipart: list[list[bytes]]
    recv_queue: asyncio.Queue[bytes]
    recv_multipart_queue: asyncio.Queue[list[bytes]]


@pytest.fixture
def mock_zmq(monkeypatch) -> MockZmqFixture:
    """Mock ZMQ to prevent real socket/context creation and enable message inspection."""
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
    """Reset and seed the global random generator for each test."""
    rng.reset()
    rng.init(42)

    yield

    rng.reset()


@pytest.fixture(autouse=True)
def reset_singleton_factories():
    """Reset singleton factory instances between tests to prevent state leakage."""
    yield

    from aiperf.common.singleton import SingletonMeta

    SingletonMeta._instances.clear()


@pytest.fixture
def temporary_registry() -> Generator[PluginRegistry, None, None]:
    """Fixture for isolated plugin registry testing."""
    old_instance = PluginRegistry._instance

    PluginRegistry._reset_singleton()
    fresh_registry = PluginRegistry()

    yield fresh_registry

    PluginRegistry._instance = old_instance


@pytest.fixture
def mock_tokenizer_cls() -> type[Tokenizer]:
    """Mock our Tokenizer class to avoid HTTP requests during testing."""

    class MockTokenizer(Tokenizer):
        """A thin mocked wrapper around AIPerf Tokenizer for testing."""

        def __init__(self, mock_tokenizer: MagicMock):
            super().__init__()
            self._tokenizer = mock_tokenizer

            self.encode = MagicMock(side_effect=self._mock_encode)
            self.decode = MagicMock(side_effect=self._mock_decode)

        @classmethod
        def from_pretrained(
            cls, name: str, trust_remote_code: bool = False, revision: str = "main"
        ):
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
def cli_config() -> CLIConfig:
    """Unified CLIConfig fixture combining benchmark + service-runtime fields."""
    return CLIConfig(model_names=["test-model"])


def make_run_from_cli(
    cli_config: CLIConfig,
):
    """Build a v2 ``BenchmarkRun`` from a :class:`CLIConfig` input DTO."""
    from aiperf.config import BenchmarkRun
    from aiperf.config.flags.resolver import resolve_config

    aiperf_config = resolve_config(cli_config, cli_config.config_file)
    return BenchmarkRun(
        benchmark_id=uuid.uuid4().hex,
        cfg=aiperf_config.benchmark,
        artifact_dir=aiperf_config.benchmark.artifacts.dir,
        random_seed=aiperf_config.random_seed,
        variables=dict(aiperf_config.variables),
    )


def make_benchmark_run(
    *,
    model_names: list[str] | None = None,
    endpoint_type: str = "completions",
    streaming: bool = False,
    accuracy: dict | None = None,
    extra: dict | None = None,
):
    """Build a v2 ``BenchmarkRun`` directly without round-tripping through v1."""
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    payload: dict = {
        "models": model_names or ["test-model"],
        "endpoint": {
            "type": endpoint_type,
            "urls": ["http://localhost:8000/v1"],
            "streaming": streaming,
        },
        "datasets": [{"name": "default", "type": "synthetic"}],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "requests": 1,
            }
        ],
    }
    if accuracy is not None:
        payload["accuracy"] = accuracy
    if extra:
        for key, value in extra.items():
            payload[key] = value
    cfg = BenchmarkConfig.model_validate(payload)
    return BenchmarkRun(
        benchmark_id=uuid.uuid4().hex,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )


@pytest.fixture
def benchmark_run(cli_config: CLIConfig):
    """Build a v2 ``BenchmarkRun`` from the existing v1 fixture."""
    return make_run_from_cli(cli_config)


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
        """Create a mooncake trace file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            if isinstance(entries_or_count, int):
                entry_count = entries_or_count
                for i in range(entry_count):
                    if include_timestamps is True:
                        entry = f'{{"input_length": {100 + i * 50}, "hash_ids": [{i}], "timestamp": {1000 + i * 1000}}}'
                    elif include_timestamps is False:
                        entry = f'{{"input_length": {100 + i * 50}, "hash_ids": [{i}]}}'
                    else:
                        entry = f'{{"input_length": {100 + i * 50}, "hash_ids": [{i}]}}'
                    f.write(f"{entry}\n")
            else:
                for entry in entries_or_count:
                    f.write(f"{entry}\n")

            filename = f.name
            filenames.append(filename)
            return filename

    yield _create_file

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
    """Mock aiofiles.open to write to a BytesIO buffer instead of a file."""
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
def mock_parent_process():
    """Mock multiprocessing.parent_process() for testing."""
    with patch("multiprocessing.parent_process") as mock:
        yield mock


@pytest.fixture
def mock_platform_system():
    """Mock platform.system() for testing OS-specific behavior."""
    with patch("platform.system") as mock:
        yield mock


def _patch_platform_constants(*, is_windows: bool, is_macos: bool, is_linux: bool):
    """Patch IS_WINDOWS/IS_MACOS/IS_LINUX in the source module and every consumer."""
    from contextlib import ExitStack

    targets = [
        ("aiperf.common.constants", ("IS_WINDOWS", "IS_MACOS", "IS_LINUX")),
        ("aiperf.common.bootstrap", ("IS_WINDOWS", "IS_MACOS")),
        ("aiperf.common.environment", ("IS_WINDOWS",)),
        ("aiperf.config.comm.ipc", ("IS_WINDOWS",)),
        ("aiperf.controller.system_mixins", ("IS_WINDOWS",)),
    ]
    values = {"IS_WINDOWS": is_windows, "IS_MACOS": is_macos, "IS_LINUX": is_linux}

    stack = ExitStack()
    for module, names in targets:
        for name in names:
            stack.enter_context(patch(f"{module}.{name}", values[name]))
    return stack


@pytest.fixture
def mock_platform_darwin(mock_platform_system: Mock) -> Generator[Mock, None, None]:
    """Mock platform.system() and IS_MACOS/IS_LINUX/IS_WINDOWS for macOS testing."""
    mock_platform_system.return_value = "Darwin"
    with _patch_platform_constants(is_windows=False, is_macos=True, is_linux=False):
        yield mock_platform_system


@pytest.fixture
def mock_platform_linux(mock_platform_system: Mock) -> Generator[Mock, None, None]:
    """Mock platform.system() and IS_MACOS/IS_LINUX/IS_WINDOWS for Linux testing."""
    mock_platform_system.return_value = "Linux"
    with _patch_platform_constants(is_windows=False, is_macos=False, is_linux=True):
        yield mock_platform_system


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
def mock_darwin_child_process(mock_platform_darwin, mock_parent_process):
    """Mock macOS child process environment (Darwin + parent_process() returns non-None)."""
    mock_parent_process.return_value = Mock()
    return mock_parent_process


@pytest.fixture
def mock_darwin_main_process(mock_platform_darwin, mock_parent_process):
    """Mock macOS main process environment (Darwin + parent_process() returns None)."""
    mock_parent_process.return_value = None
    return mock_parent_process


@pytest.fixture
def mock_linux_child_process(mock_platform_linux, mock_parent_process):
    """Mock Linux child process environment (Linux + parent_process() returns non-None)."""
    mock_parent_process.return_value = Mock()
    return mock_parent_process


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    """Create a temporary artifact directory for testing."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def make_cfg_from_v1(
    cli_config: CLIConfig,
    artifact_directory: Path | None = None,
):
    """Build a v2 ``BenchmarkConfig`` from a v1 ``CLIConfig``."""
    cfg = make_run_from_cli(cli_config).cfg
    if artifact_directory is not None:
        cfg.artifacts.dir = Path(artifact_directory)
    return cfg


def create_exporter_config(
    profile_results,
    cli_config,
    telemetry_results=None,
    server_metrics_results=None,
    verbose=True,
):
    """Helper to create ExporterConfig with common defaults."""
    from aiperf.config.config import BenchmarkConfig

    if isinstance(cli_config, BenchmarkConfig):
        cfg = cli_config
    else:
        cli_with_verbose = cli_config.model_copy(update={"verbose": verbose})
        cfg = make_cfg_from_v1(cli_with_verbose)
    return ExporterConfig(
        results=profile_results,
        cfg=cfg,
        telemetry_results=telemetry_results,
        server_metrics_results=server_metrics_results,
    )

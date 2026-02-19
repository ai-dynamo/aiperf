# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

# Suppress factory override warnings BEFORE imports trigger registration
import logging

for _factory_logger in [
    "CommunicationFactory",
    "ServiceManagerFactory",
    "TransportFactory",
    "ZMQProxyFactory",
]:
    logging.getLogger(_factory_logger).setLevel(logging.ERROR)

import platform
import signal
import sys
from collections.abc import Generator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.cli import app
from aiperf.common import random_generator as rng
from aiperf.common.environment import Environment
from aiperf.plugin.enums import CommClientType

# Import fakes for test harness
from tests.harness import (
    FakeCommunication,
    FakeCommunicationBus,
    FakeServiceManager,  # noqa: F401 - imported for test harness
    FakeTokenizer,  # noqa: F401 - imported for test harness
    FakeTransport,  # noqa: F401 - imported for test harness
)
from tests.harness.fake_communication import CapturedPayload
from tests.harness.fake_tokenizer import TOKEN, TOKEN_LEN
from tests.harness.utils import AIPerfCLI, AIPerfRunnerFn, AIPerfRunnerResult


class TeeStream:
    """Write to both the original stream and a capture buffer."""

    def __init__(self, original: object):
        self.original = original
        self.buffer = StringIO()

    def write(self, data: str) -> int:
        self.original.write(data)
        return self.buffer.write(data)

    def flush(self) -> None:
        self.original.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()


@dataclass(frozen=True)
class ComponentIntegrationTestDefaults:
    """Default test parameters."""

    # Default model to use for integration tests.
    # Note that the openai/gpt-oss-120b model crashes on macOS for some reason.
    # Defining the default model differently so we can have more variety in the tests.
    if platform.system() == "Darwin":
        model = "Qwen/Qwen3-0.6B"
        tokenizer = "Qwen/Qwen3-0.6B"
    else:
        model = "openai/gpt-oss-120b"
        tokenizer = "openai/gpt-oss-120b"
    workers_max: int = 1
    concurrency: int = 2
    request_count: int = 10
    timeout: float = 200.0
    ui: str = "simple"


@pytest.fixture(autouse=True, scope="session")
def mock_os_exit():
    """Patch os._exit to raise an exception instead of exiting the process."""
    with patch(
        "os._exit",
        side_effect=lambda code: SystemExit(code),
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def mock_os_kill_sigkill():
    """Patch os.kill to prevent SIGKILL from killing the test process.

    Component integration tests run AIPerf in-process (not as a subprocess).
    When BaseService._kill() calls os.kill(os.getpid(), signal.SIGKILL), it would
    kill the test runner. This fixture intercepts that call and raises SystemExit
    instead, allowing the test framework to handle it gracefully.
    """
    import os

    original_os_kill = os.kill

    def safe_os_kill(pid, sig):
        if pid == os.getpid() and sig == signal.SIGKILL:
            # BaseService._kill calls os.kill(os.getpid(), signal.SIGKILL)
            # Raise SystemExit instead of actually killing the test process
            raise SystemExit(-9)  # Python subprocess convention for SIGKILL
        return original_os_kill(pid, sig)

    with patch("os.kill", safe_os_kill):
        yield


@pytest.fixture(autouse=True, scope="session")
def no_server_metrics_flush_period():
    """Fixture to disable server metrics flush period."""
    original_flush_period = Environment.SERVER_METRICS.COLLECTION_FLUSH_PERIOD
    Environment.SERVER_METRICS.COLLECTION_FLUSH_PERIOD = 0
    yield
    Environment.SERVER_METRICS.COLLECTION_FLUSH_PERIOD = original_flush_period


@pytest.fixture(autouse=True, scope="session")
def mock_tokenizer_from_pretrained():
    """Patch Tokenizer.from_pretrained to use FakeTokenizer."""
    with patch(
        "aiperf.common.tokenizer.Tokenizer.from_pretrained",
        FakeTokenizer.from_pretrained,
    ):
        yield


def _mock_tokenize(text: str) -> tuple[str, ...]:
    """Tokenize using FakeTokenizer logic."""
    if not text:
        return ()
    return (TOKEN,) * round(len(text) / TOKEN_LEN)


@pytest.fixture(autouse=True, scope="session")
def mock_server_tokenize():
    """Patch mock server _tokenize to match FakeTokenizer.

    Also patches CORPUS_TOKENS to None so _cycle_tokens falls back to
    cycling through prompt tokens (which will be 'tok$' strings).
    """
    with (
        patch("aiperf_mock_server.tokens._tokenize", _mock_tokenize),
        patch("aiperf_mock_server.tokens.CORPUS_TOKENS", None),
    ):
        yield


_MOCK_CORPUS_TEXT = TOKEN * 10000  # 10000 tokens of "token$"


@pytest.fixture(autouse=True, scope="session")
def mock_corpus_file():
    """Patch open() to return mock corpus when reading shakespeare.txt."""
    import builtins
    from io import StringIO

    _original_open = builtins.open

    def _patched_open(file, *args, **kwargs):
        if "shakespeare.txt" in str(file):
            return StringIO(_MOCK_CORPUS_TEXT)
        return _original_open(file, *args, **kwargs)

    with patch("builtins.open", _patched_open):
        yield


@pytest.fixture(autouse=True)
def reset_singleton_factories():
    """Reset singleton factory instances between tests to prevent state leakage.

    This fixture runs automatically for every test and clears the singleton
    instances managed by the Singleton metaclass. This prevents tests from interfering
    with each other when they create services that use singleton communication instances.
    """
    yield  # Run the test first

    # Clean up after test completes - clear per-process singleton instances
    from aiperf.common.singleton import SingletonMeta

    SingletonMeta._instances.clear()


@pytest.fixture(autouse=True)
def reset_random_generator() -> Generator[None, None, None]:
    """Reset and seed the global random generator for each test.

    This ensures all tests have consistent, reproducible random behavior
    and prevents test pollution when running with pytest-xdist.
    """
    # Reset and seed before each test
    rng.reset()
    rng.init(42)  # Use a fixed seed for test reproducibility

    yield  # Run the test

    # Reset after each test to ensure clean state
    rng.reset()


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for AIPerf output."""
    output_dir = tmp_path / "aiperf_output"
    output_dir.mkdir()
    return output_dir


@dataclass(frozen=True)
class AIPerfRunnerResultWithSharedBus(AIPerfRunnerResult):
    """AIPerf component integration result with message inspection helpers."""

    # Note: stdout and stderr are inherited from parent with default values
    # shared_bus must have a default value to maintain field ordering
    shared_bus: FakeCommunicationBus | None = None

    @property
    def sent_payloads(self) -> list[CapturedPayload]:
        """Get all sent payloads from all clients across all communications."""
        if self.shared_bus is None:
            return []
        return self.shared_bus.sent_payloads

    @property
    def received_payloads(self) -> list[CapturedPayload]:
        """Get all received payloads from all clients across all communications."""
        return self.shared_bus.received_payloads

    def payloads_by_client_type(
        self, client_type: CommClientType, *, sent: bool = True
    ) -> list[CapturedPayload]:
        """Filter payloads by client type.

        Args:
            client_type: The client type to filter by.
            sent: If True, filter sent payloads; if False, filter received payloads.
        """
        payloads = self.sent_payloads if sent else self.received_payloads
        return [p for p in payloads if p.client_type == client_type]

    def payloads_by_type(
        self, message_type: type, *, sent: bool = True
    ) -> list[CapturedPayload]:
        """Filter payloads by message type.

        Args:
            message_type: The message class to filter by.
            sent: If True, filter sent payloads; if False, filter received payloads.
        """
        payloads = self.sent_payloads if sent else self.received_payloads
        return [p for p in payloads if isinstance(p.payload, message_type)]

    def payloads_by_identity(
        self, identity: str, *, sent: bool = True
    ) -> list[CapturedPayload]:
        """Get payloads sent or received by a specific client identity.

        Args:
            identity: The client identity to filter by.
            sent: If True, get payloads sent by this identity; if False, get payloads received.
        """
        payloads = self.sent_payloads if sent else self.received_payloads
        if sent:
            return [p for p in payloads if p.sender_identity == identity]
        return [p for p in payloads if p.receiver_identity == identity]

    def messages(self, message_type: type, *, sent: bool = True) -> list:
        """Get just the message objects (not wrapped in CapturedPayload) by type.

        Args:
            message_type: The message class to filter by.
            sent: If True, filter sent messages; if False, filter received messages.
        """
        return [p.payload for p in self.payloads_by_type(message_type, sent=sent)]


@pytest.fixture
def aiperf_runner(
    temp_output_dir: Path,
) -> AIPerfRunnerFn:
    """AIPerf subprocess runner."""

    def runner(
        args: list[str], timeout: float = 200.0
    ) -> AIPerfRunnerResultWithSharedBus:  # noqa: ARG001
        full_args = args
        # Only add --artifact-dir for profile command (not for plot)
        if args and args[0] == "profile":
            full_args += [
                "--artifact-dir",
                str(temp_output_dir),
            ]

        # Create a fresh bus for test isolation and capture reference BEFORE running
        # This ensures we have the bus even if clear_shared_bus() is called during shutdown
        test_bus = FakeCommunicationBus()
        FakeCommunication.set_shared_bus(test_bus)

        exit_code = 0
        stdout_tee = TeeStream(sys.stdout)
        stderr_tee = TeeStream(sys.stderr)

        # CLI calls os._exit(0) on success, so we expect SystemExit
        try:
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                app(full_args)
        except SystemExit as e:
            exit_code = e.code

        return AIPerfRunnerResultWithSharedBus(
            exit_code=exit_code,
            output_dir=temp_output_dir,
            shared_bus=test_bus,
            stdout=stdout_tee.getvalue(),
            stderr=stderr_tee.getvalue(),
        )

    return runner


@pytest.fixture
def cli(
    aiperf_runner: AIPerfRunnerFn,
) -> AIPerfCLI:
    """AIPerf CLI wrapper."""
    return AIPerfCLI(aiperf_runner)


# =============================================================================
# DCGM / GPU Telemetry Fixtures
# =============================================================================


@pytest.fixture
def mock_dcgm_endpoints(request):
    """Mock aiohttp.ClientSession for DCGM endpoint requests.

    This fixture uses FakeDCGMMocker from tests.harness to intercept HTTP
    requests to DCGM Prometheus endpoints and return fake metrics.

    Usage:
        1. Use with default endpoints:
           def test_basic(cli, mock_dcgm_endpoints):
               result = cli.run_sync("aiperf profile --gpu-telemetry http://localhost:9401/metrics ...")

        2. Customize endpoints per test:
           @pytest.fixture
           def custom_dcgm_endpoints():
               return [
                   DCGMEndpoint("http://node1:9401/metrics", gpu_name="h100", num_gpus=4),
                   DCGMEndpoint("http://node2:9401/metrics", gpu_name="h200", num_gpus=2),
               ]

           def test_custom(cli, custom_dcgm_endpoints, mock_dcgm_endpoints):
               # Uses your custom endpoints
               ...

        3. Control GPU load dynamically:
           def test_load(cli, mock_dcgm_endpoints):
               faker = mock_dcgm_endpoints["http://localhost:9401/metrics"]
               faker.set_load(0.8)  # 80% GPU load
               ...

    Returns:
        dict[str, DCGMFaker]: Mapping of URL to DCGMFaker instance for load control.
    """
    from tests.harness import FakeDCGMMocker

    # Check if test provides custom endpoints
    custom_endpoints = None
    if hasattr(request, "param"):
        custom_endpoints = request.param
    elif "custom_dcgm_endpoints" in request.fixturenames:
        custom_endpoints = request.getfixturevalue("custom_dcgm_endpoints")

    # Use FakeDCGMMocker context manager
    mocker = FakeDCGMMocker(endpoints=custom_endpoints)
    with mocker as fakers:
        yield fakers


# ============================================================================
# Session Analysis Helpers
# ============================================================================


def group_records_by_session(jsonl_records) -> dict[str, list]:
    """Group JSONL records by session (x_correlation_id).

    This is a common pattern in multi-turn session tests for analyzing
    per-session behavior (turn counts, sticky routing, cancellation, etc.).

    Args:
        jsonl_records: List of AIPerfMetricRecord objects from result.jsonl

    Returns:
        Dictionary mapping x_correlation_id to list of records for that session,
        ordered by turn_index

    Example:
        sessions = group_records_by_session(result.jsonl)
        for session_id, records in sessions.items():
            assert len(records) == expected_turns
            assert all(r.metadata.worker_id == records[0].metadata.worker_id for r in records)
    """
    from collections import defaultdict

    sessions = defaultdict(list)
    for record in jsonl_records:
        session_id = record.metadata.x_correlation_id
        sessions[session_id].append(record)

    # Sort each session's records by turn_index for convenience
    for session_id in sessions:
        sessions[session_id].sort(key=lambda r: r.metadata.turn_index)

    return dict(sessions)


def assert_sticky_routing(jsonl_records) -> dict[str, list]:
    """Validate that each session's turns went to the same worker.

    This validates the core sticky routing invariant: all turns within a session
    must be handled by the same worker to maintain conversation context.

    Args:
        jsonl_records: List of AIPerfMetricRecord objects from result.jsonl

    Returns:
        Dictionary mapping x_correlation_id to list of records for that session

    Raises:
        AssertionError: If any session has turns handled by different workers

    Example:
        sessions = assert_sticky_routing(result.jsonl)
        assert len(sessions) == expected_num_sessions
    """
    sessions = group_records_by_session(jsonl_records)

    for session_id, records in sessions.items():
        worker_ids = {r.metadata.worker_id for r in records}
        assert len(worker_ids) == 1, (
            f"Session {session_id} violated sticky routing: "
            f"workers={worker_ids}, turns={[r.metadata.turn_index for r in records]}"
        )

    return sessions


def assert_turns_sequential(sessions: dict[str, list]) -> None:
    """Assert turn indices are sequential (0, 1, 2, ...) within each session.

    Args:
        sessions: Dictionary mapping session_id to list of records from group_records_by_session()

    Raises:
        AssertionError: If any session has non-sequential turn indices

    Example:
        sessions = group_records_by_session(result.jsonl)
        assert_turns_sequential(sessions)
    """
    for session_id, records in sessions.items():
        turns = [r.metadata.turn_index for r in records]
        expected = list(range(len(turns)))
        assert turns == expected, (
            f"Session {session_id} has non-sequential turns: {turns}, expected {expected}"
        )


def assert_jsonl_turns_sequential(jsonl_records) -> None:
    """Assert that turn indices are sequential (0, 1, 2, ...) within each session.

    Convenience helper that combines group_records_by_session and assert_turns_sequential.
    Validates that each session has complete, sequential turn indices starting
    from 0 with no gaps or duplicates.

    Args:
        jsonl_records: List of AIPerfMetricRecord objects from result.jsonl

    Raises:
        AssertionError: If any session has non-sequential or missing turns

    Example:
        assert_jsonl_turns_sequential(result.jsonl)
    """
    sessions = group_records_by_session(jsonl_records)
    assert_turns_sequential(sessions)


def count_cancelled_requests(jsonl_records) -> int:
    """Count the number of cancelled requests in JSONL records.

    Args:
        jsonl_records: List of AIPerfMetricRecord objects from result.jsonl

    Returns:
        Number of cancelled requests

    Example:
        cancelled_count = count_cancelled_requests(result.jsonl)
        assert cancelled_count > 0, "Expected some cancellations"
    """
    return sum(1 for record in jsonl_records if record.metadata.was_cancelled)


def validate_cancellation_errors(jsonl_records) -> None:
    """Validate that all cancelled requests have proper error details.

    Checks that cancelled requests have:
    - error field is not None
    - error.code == 499
    - error.type == "RequestCancellationError"

    Args:
        jsonl_records: List of AIPerfMetricRecord objects from result.jsonl

    Raises:
        AssertionError: If any cancelled request has invalid error details

    Example:
        validate_cancellation_errors(result.jsonl)
    """
    for record in jsonl_records:
        if record.metadata.was_cancelled:
            assert record.error is not None, (
                f"Cancelled request {record.metadata.request_id} missing error details"
            )
            assert record.error.code == 499, (
                f"Cancelled request {record.metadata.request_id} has wrong error code: "
                f"{record.error.code} (expected 499)"
            )
            assert record.error.type == "RequestCancellationError", (
                f"Cancelled request {record.metadata.request_id} has wrong error type: "
                f"{record.error.type} (expected RequestCancellationError)"
            )


def assert_sessions_complete(jsonl_records, expected_turns: int) -> None:
    """Assert that all sessions have the expected number of turns.

    Validates that every session has completed all expected turns,
    regardless of whether individual turns were cancelled.

    Args:
        jsonl_records: List of AIPerfMetricRecord objects from result.jsonl
        expected_turns: Expected number of turns per session

    Raises:
        AssertionError: If any session doesn't have expected turns

    Example:
        assert_sessions_complete(result.jsonl, expected_turns=5)
    """
    sessions = group_records_by_session(jsonl_records)

    for session_id, records in sessions.items():
        turn_set = {r.metadata.turn_index for r in records}
        expected_set = set(range(expected_turns))
        assert turn_set == expected_set, (
            f"Session {session_id} incomplete: "
            f"has turns {sorted(turn_set)}, expected {sorted(expected_set)}"
        )


def get_session_worker_mapping(jsonl_records) -> dict[str, str]:
    """Get mapping of session IDs to their assigned worker IDs.

    Returns a dictionary mapping each x_correlation_id to the worker_id
    that handled that session. Useful for analyzing sticky routing patterns.

    Args:
        jsonl_records: List of AIPerfMetricRecord objects from result.jsonl

    Returns:
        Dictionary mapping x_correlation_id to worker_id

    Example:
        worker_map = get_session_worker_mapping(result.jsonl)
        workers_used = set(worker_map.values())
        assert len(workers_used) == expected_num_workers
    """
    sessions = group_records_by_session(jsonl_records)
    return {
        session_id: records[0].metadata.worker_id
        for session_id, records in sessions.items()
        if records  # Ensure session has at least one record
    }


# Backward compatibility alias
validate_sticky_routing = assert_sticky_routing

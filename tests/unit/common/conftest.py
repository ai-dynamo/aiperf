# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for common tests, especially bootstrap tests."""

import io
import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import Process
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from aiperf.common.base_service import BaseService
from aiperf.common.subprocess_manager import SubprocessInfo, SubprocessManager
from aiperf.common.tokenizer_display import TokenizerDisplayEntry
from aiperf.config import BenchmarkRun
from aiperf.plugin.enums import ServiceType
from aiperf.timing.manager import TimingManager
from aiperf.workers.worker import Worker
from tests.harness import mock_plugin

# =============================================================================
# Mock Process Fixtures
# =============================================================================


@pytest.fixture
def mock_process_factory() -> Callable[..., MagicMock]:
    """Factory fixture for creating mock processes with custom state.

    Returns a callable that creates mock Process objects with configurable:
    - is_alive: Whether the process appears alive (default: True)
    - pid: Process ID (default: auto-generated)
    - exitcode: Exit code for dead processes (default: None if alive, 0 if dead)

    Example:
        def test_something(mock_process_factory):
            alive_proc = mock_process_factory(is_alive=True, pid=1234)
            dead_proc = mock_process_factory(is_alive=False, exitcode=1)
    """
    _counter = [0]

    def _create(
        is_alive: bool = True,
        pid: int | None = None,
        exitcode: int | None = None,
    ) -> MagicMock:
        _counter[0] += 1
        mock = MagicMock(spec=Process)
        mock.is_alive.return_value = is_alive
        mock.pid = pid if pid is not None else 10000 + _counter[0]
        mock.exitcode = exitcode if exitcode is not None else (None if is_alive else 0)
        return mock

    return _create


@pytest.fixture
def mock_process_alive(mock_process_factory) -> MagicMock:
    """Create a mock process that appears alive."""
    return mock_process_factory(is_alive=True, pid=12345)


@pytest.fixture
def mock_process_dead(mock_process_factory) -> MagicMock:
    """Create a mock process that appears dead with exit code 0."""
    return mock_process_factory(is_alive=False, pid=54321, exitcode=0)


@pytest.fixture
def mock_process_crashed(mock_process_factory) -> MagicMock:
    """Create a mock process that crashed with non-zero exit code."""
    return mock_process_factory(is_alive=False, pid=99999, exitcode=1)


# =============================================================================
# SubprocessManager Fixtures
# =============================================================================


@pytest.fixture
def subprocess_manager(run: BenchmarkRun) -> SubprocessManager:
    """Create a SubprocessManager instance for testing."""
    return SubprocessManager(
        run=run,
        log_queue=None,
        logger=None,
    )


@pytest.fixture
def subprocess_manager_with_logger(
    run: BenchmarkRun,
) -> tuple[SubprocessManager, MagicMock]:
    """Create a SubprocessManager with a mock logger.

    Returns tuple of (manager, mock_logger) for assertions.
    """
    mock_logger = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.warning = MagicMock()
    manager = SubprocessManager(
        run=run,
        log_queue=None,
        logger=mock_logger,
    )
    return manager, mock_logger


# =============================================================================
# SubprocessInfo Helpers
# =============================================================================


def make_subprocess_info(
    service_type: ServiceType = ServiceType.WORKER,
    service_id: str = "test_service",
    process: Process | MagicMock | None = None,
) -> SubprocessInfo:
    """Helper to create SubprocessInfo for tests.

    Args:
        service_type: Type of service (default: WORKER)
        service_id: Service identifier (default: "test_service")
        process: Process object or None (default: None)

    Returns:
        SubprocessInfo instance
    """
    return SubprocessInfo(
        process=process,
        service_type=service_type,
        service_id=service_id,
    )


# Tokenizer Test Helpers
# =============================================================================


def make_display_entry(
    original_name: str,
    resolved_name: str | None = None,
    was_resolved: bool | None = None,
) -> TokenizerDisplayEntry:
    """Factory helper for creating TokenizerDisplayEntry instances.

    Args:
        original_name: The name originally requested by the user.
        resolved_name: The resolved name. Defaults to original_name if not provided.
        was_resolved: Whether resolution occurred. Auto-detected if not provided.

    Returns:
        TokenizerDisplayEntry with the specified attributes.
    """
    if resolved_name is None:
        resolved_name = original_name

    if was_resolved is None:
        was_resolved = original_name != resolved_name

    return TokenizerDisplayEntry(
        original_name=original_name,
        resolved_name=resolved_name,
        was_resolved=was_resolved,
    )


# =============================================================================
# Tokenizer Test Fixtures
# =============================================================================


@pytest.fixture
def console_output():
    """Create a console that writes to a string buffer for testing Rich output.

    Returns:
        Tuple of (Console, StringIO) for capturing and inspecting output.
    """
    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=True, width=120)
    return console, string_io


@pytest.fixture
def mock_logger():
    """Create a mock logger that captures log output for testing.

    Returns:
        Tuple of (mock_logger, list) where list captures info messages.
    """
    messages: list[str] = []
    logger = MagicMock()
    logger.info = MagicMock(side_effect=lambda msg: messages.append(msg))
    return logger, messages


@pytest.fixture
def mock_tokenizer_cls():
    """Mock the Tokenizer class for testing validation without loading real tokenizers."""
    with patch("aiperf.common.tokenizer.Tokenizer") as mock_cls:
        yield mock_cls


@pytest.fixture
def mock_executor():
    """Mock ProcessPoolExecutor for testing subprocess validation.

    Provides a dictionary with:
        - executor: The mocked executor instance
        - future: The mocked future object for setting return values
    """
    mock_future = MagicMock()
    mock_executor_instance = MagicMock()
    mock_executor_instance.submit.return_value = mock_future
    mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
    mock_executor_instance.__exit__ = MagicMock(return_value=False)

    with patch(
        "concurrent.futures.ProcessPoolExecutor", return_value=mock_executor_instance
    ):
        yield {"executor": mock_executor_instance, "future": mock_future}


class DummyService(BaseService):
    """Minimal service for testing bootstrap.

    This service immediately completes when started, allowing tests to
    complete quickly without hanging.
    """

    service_type = "test_dummy"

    async def start(self):
        """Start the service and immediately stop."""
        self.stopped_event.set()

    async def stop(self):
        """Stop the service."""
        self.stopped_event.set()


class DummyWorker(DummyService):
    """Dummy service named 'Worker' to test GC disabling."""

    pass


# Override the class name to simulate the Worker service
DummyWorker.__name__ = Worker.__name__


class DummyTimingManager(DummyService):
    """Dummy service named 'TimingManager' to test GC disabling."""

    pass


# Override the class name to simulate the TimingManager service
DummyTimingManager.__name__ = TimingManager.__name__


@pytest.fixture
def register_dummy_services():
    """Register dummy services in the plugin registry for testing.

    This allows bootstrap tests to use service names instead of classes.
    """
    # Use mock_plugin context managers to register with metadata
    with (
        mock_plugin(
            "service",
            "test_dummy",
            DummyService,
            metadata={"required": False, "auto_start": False, "disable_gc": False},
        ),
        mock_plugin(
            "service",
            "test_worker",
            DummyWorker,
            metadata={"required": False, "auto_start": False, "disable_gc": True},
        ),
        mock_plugin(
            "service",
            "test_timing_manager",
            DummyTimingManager,
            metadata={"required": False, "auto_start": False, "disable_gc": True},
        ),
    ):
        yield


@pytest.fixture
def mock_log_queue() -> MagicMock:
    """Create a mock multiprocessing.Queue for testing."""
    return MagicMock(spec=multiprocessing.Queue)


@pytest.fixture
def service_config_no_uvloop(run: BenchmarkRun, monkeypatch) -> BenchmarkRun:
    """Create a BenchmarkRun with uvloop disabled for testing."""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.SERVICE, "DISABLE_UVLOOP", True)
    return run


@dataclass
class MockGC:
    """Container for mocked GC functions."""

    collect: MagicMock
    """Mock for gc.collect()."""

    freeze: MagicMock
    """Mock for gc.freeze()."""

    set_threshold: MagicMock
    """Mock for gc.set_threshold()."""

    disable: MagicMock
    """Mock for gc.disable()."""

    call_order: list[str] = field(default_factory=list)
    """Ordered list tracking which GC operations were called."""


@pytest.fixture
def mock_gc() -> MockGC:
    """Mock garbage collection functions for testing bootstrap GC behavior.

    Returns a MockGC dataclass with mocked gc functions and a call_order list
    that tracks the order of GC operations.
    """
    call_order: list[str] = []

    def track_collect(*args, **kwargs):
        call_order.append("collect")

    def track_freeze(*args, **kwargs):
        call_order.append("freeze")

    def track_set_threshold(*args, **kwargs):
        call_order.append("set_threshold")

    def track_disable(*args, **kwargs):
        call_order.append("disable")

    with (
        patch("gc.collect", side_effect=track_collect) as mock_collect,
        patch("gc.freeze", side_effect=track_freeze) as mock_freeze,
        patch(
            "gc.set_threshold", side_effect=track_set_threshold
        ) as mock_set_threshold,
        patch("gc.disable", side_effect=track_disable) as mock_disable,
    ):
        yield MockGC(
            collect=mock_collect,
            freeze=mock_freeze,
            set_threshold=mock_set_threshold,
            disable=mock_disable,
            call_order=call_order,
        )

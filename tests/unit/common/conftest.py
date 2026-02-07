# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for common tests, especially bootstrap tests."""

import io
import multiprocessing
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from aiperf.common.base_service import BaseService
from aiperf.common.config import ServiceConfig
from aiperf.common.tokenizer_display import TokenizerDisplayEntry
from aiperf.timing.manager import TimingManager
from aiperf.workers.worker import Worker
from tests.harness import mock_plugin

# =============================================================================
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
def service_config_no_uvloop(
    service_config: ServiceConfig, monkeypatch
) -> ServiceConfig:
    """Create a ServiceConfig with uvloop disabled for testing."""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.SERVICE, "DISABLE_UVLOOP", True)
    return service_config


@dataclass
class MockGC:
    """Container for mocked GC functions."""

    collect: MagicMock
    freeze: MagicMock
    set_threshold: MagicMock
    disable: MagicMock
    call_order: list[str] = field(default_factory=list)


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

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for common tests, especially bootstrap tests."""

import multiprocessing
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.base_service import BaseService
from aiperf.common.config import ServiceConfig
from aiperf.timing.manager import TimingManager
from aiperf.workers.worker import Worker
from tests.harness import mock_plugin


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

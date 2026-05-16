# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing AIPerf controller.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.models import ErrorDetails
from aiperf.common.service_registry import ServiceRegistry
from aiperf.config import BenchmarkRun
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceRunType


@pytest.fixture(autouse=True)
def _reset_service_registry():
    """Reset the process-global ServiceRegistry around every test.

    ``ServiceRegistry`` is a per-process singleton; without an explicit
    reset, ``expected_by_type`` and the wait-state accumulators leak
    between tests, so ``test_X_already_registered`` etc. fail when run
    under ``-n auto`` whenever a sibling test ran first.
    """
    ServiceRegistry.reset()
    yield
    ServiceRegistry.reset()


class MockTestException(Exception):
    """Mock test exception."""


@pytest.fixture
def mock_service_manager() -> AsyncMock:
    """Mock service manager."""
    mock_manager = AsyncMock()
    return mock_manager


@pytest.fixture
def system_controller(
    run: BenchmarkRun,
    mock_service_manager: AsyncMock,
) -> SystemController:
    """Create a SystemController instance with mocked dependencies."""
    mock_ui = AsyncMock()
    mock_comm = AsyncMock()
    # get_address is synchronous — return a plain string so the
    # ZMQRouterReplyClient constructor doesn't receive a coroutine.
    mock_comm.get_address = MagicMock(return_value="ipc:///tmp/test-health-check")

    def mock_get_class(protocol, name):
        if protocol == "service_manager":
            return lambda **kwargs: mock_service_manager
        if protocol == "ui":
            return lambda **kwargs: mock_ui
        if protocol == "communication":
            return lambda **kwargs: mock_comm
        raise ValueError(f"Unknown protocol: {protocol}")

    with (
        patch(
            "aiperf.controller.system_controller.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch("aiperf.controller.system_controller.ProxyManager") as mock_proxy,
        patch(
            "aiperf.controller.system_controller.ZMQStreamingRouterClient",
            return_value=AsyncMock(),
        ),
        patch(
            "aiperf.common.mixins.communication_mixin.plugins.get_class",
            side_effect=mock_get_class,
        ),
    ):  # fmt: skip
        mock_proxy.return_value = AsyncMock()

        controller = SystemController(
            run=run,
            service_id="test_controller",
        )
        # Mock the stop method to avoid actual shutdown
        controller.stop = AsyncMock()
        # Stub the bus publish — _set_system_state and other handlers fan out
        # via publish, but the controller fixture isn't started so pub_client
        # is unset. Tests that want to assert on publish overwrite this.
        controller.publish = AsyncMock()
        return controller


@pytest.fixture
def local_group_run(run: BenchmarkRun) -> BenchmarkRun:
    """BenchmarkRun configured to expose local worker-group adapter capacity."""
    run.cfg.benchmark.runtime.service_run_type = ServiceRunType.MULTIPROCESSING
    run.cfg.benchmark.runtime.workers = 4
    run.cfg.benchmark.runtime.record_processors = 2
    return run


@pytest.fixture
def mock_exception() -> MockTestException:
    """Mock the exception."""
    return MockTestException("Test error")


@pytest.fixture
def error_details(mock_exception: MockTestException) -> ErrorDetails:
    """Mock the error details."""
    return ErrorDetails.from_exception(mock_exception)

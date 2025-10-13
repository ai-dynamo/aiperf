# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for common tests, especially bootstrap tests."""

import multiprocessing
from unittest.mock import MagicMock

import pytest

from aiperf.common.base_service import BaseService
from aiperf.common.config import ServiceConfig


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


@pytest.fixture
def mock_log_queue() -> MagicMock:
    """Create a mock multiprocessing.Queue for testing."""
    return MagicMock(spec=multiprocessing.Queue)


@pytest.fixture
def service_config_no_uvloop(service_config: ServiceConfig) -> ServiceConfig:
    """Create a ServiceConfig with uvloop disabled for testing."""
    service_config.developer.disable_uvloop = True
    return service_config

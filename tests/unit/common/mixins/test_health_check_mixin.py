# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HealthCheckMixin."""

import pytest

from aiperf.common.enums import LifecycleState
from aiperf.common.mixins.health_check_mixin import HealthCheckMixin


class MockService(HealthCheckMixin):
    """Mock service for testing HealthCheckMixin."""

    def __init__(self, state: LifecycleState, service_id: str = "test-service") -> None:
        """Initialize mock service."""
        self.state = state
        self.id = service_id


class TestHealthCheckMixin:
    """Test HealthCheckMixin methods."""

    @pytest.mark.parametrize(
        "state,expected",
        [
            (LifecycleState.CREATED, True),
            (LifecycleState.INITIALIZING, True),
            (LifecycleState.INITIALIZED, True),
            (LifecycleState.STARTING, True),
            (LifecycleState.RUNNING, True),
            (LifecycleState.STOPPING, True),
            (LifecycleState.STOPPED, True),
            (LifecycleState.FAILED, False),  # Only FAILED is unhealthy
        ],
    )
    def test_is_healthy(self, state: LifecycleState, expected: bool) -> None:
        """Test is_healthy returns True for all states except FAILED."""
        service = MockService(state)
        assert service.is_healthy() == expected

    @pytest.mark.parametrize(
        "state,expected",
        [
            (LifecycleState.CREATED, False),
            (LifecycleState.INITIALIZING, False),
            (LifecycleState.INITIALIZED, False),
            (LifecycleState.STARTING, False),
            (LifecycleState.RUNNING, True),  # Only RUNNING is ready
            (LifecycleState.STOPPING, False),
            (LifecycleState.STOPPED, False),
            (LifecycleState.FAILED, False),
        ],
    )
    def test_is_ready(self, state: LifecycleState, expected: bool) -> None:
        """Test is_ready returns True only when RUNNING."""
        service = MockService(state)
        assert service.is_ready() == expected

    def test_get_health_details_running(self) -> None:
        """Test get_health_details when service is running."""
        service = MockService(LifecycleState.RUNNING, "my-service-1")
        details = service.get_health_details()

        assert details.service_id == "my-service-1"
        assert details.state == LifecycleState.RUNNING
        assert details.healthy is True
        assert details.ready is True

    def test_get_health_details_initializing(self) -> None:
        """Test get_health_details when service is initializing."""
        service = MockService(LifecycleState.INITIALIZING, "init-service")
        details = service.get_health_details()

        assert details.service_id == "init-service"
        assert details.state == LifecycleState.INITIALIZING
        assert details.healthy is True
        assert details.ready is False

    def test_get_health_details_failed(self) -> None:
        """Test get_health_details when service has failed."""
        service = MockService(LifecycleState.FAILED, "failed-service")
        details = service.get_health_details()

        assert details.service_id == "failed-service"
        assert details.state == LifecycleState.FAILED
        assert details.healthy is False
        assert details.ready is False

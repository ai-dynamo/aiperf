# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from aiperf.common.enums import LifecycleState
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.mixins import AIPerfLifecycleMixin


class TestAIPerfLifecycleBasic:
    """Test suite for basic AIPerfLifecycleMixin functionality."""

    @pytest.fixture
    def lifecycle_component(self):
        """Create a minimal lifecycle component for testing."""
        return AIPerfLifecycleMixin()

    def test_lifecycle_initialization(self, lifecycle_component):
        """Test that lifecycle components are properly initialized."""
        assert lifecycle_component.state == LifecycleState.CREATED
        assert lifecycle_component.id.startswith("AIPerfLifecycleMixin_")
        assert len(lifecycle_component.id.split("_")[1]) == 8  # UUID hex[:8]

        # Test initial state
        assert not lifecycle_component.was_initialized
        assert not lifecycle_component.was_started
        assert not lifecycle_component.was_stopped
        assert not lifecycle_component.is_running
        assert not lifecycle_component.stop_requested

    def test_lifecycle_custom_id(self):
        """Test that custom IDs are properly set."""
        custom_id = "test_component_123"
        component = AIPerfLifecycleMixin(id=custom_id)
        assert component.id == custom_id

    @pytest.mark.asyncio
    async def test_lifecycle_properties_progression(self, lifecycle_component):
        """Test that lifecycle properties change correctly as component progresses through states."""
        assert not lifecycle_component.was_initialized
        assert not lifecycle_component.was_started
        assert not lifecycle_component.was_stopped
        assert not lifecycle_component.is_running

        await lifecycle_component.initialize()
        assert lifecycle_component.was_initialized
        assert not lifecycle_component.was_started
        assert not lifecycle_component.is_running

        await lifecycle_component.start()
        assert lifecycle_component.was_initialized
        assert lifecycle_component.was_started
        assert lifecycle_component.is_running
        assert not lifecycle_component.was_stopped

        await lifecycle_component.stop()
        assert lifecycle_component.was_initialized
        assert lifecycle_component.was_started
        assert lifecycle_component.was_stopped
        assert not lifecycle_component.is_running

    @pytest.mark.asyncio
    async def test_initialize_success(self, lifecycle_component):
        """Test successful initialization."""
        await lifecycle_component.initialize()

        assert lifecycle_component.state == LifecycleState.INITIALIZED
        assert lifecycle_component.initialized_event.is_set()
        assert lifecycle_component.was_initialized

    @pytest.mark.asyncio
    async def test_initialize_ignored_when_already_advanced(self, lifecycle_component):
        """Test that initialize() is ignored when called from advanced states."""
        await lifecycle_component.initialize()

        with patch.object(lifecycle_component, "debug") as mock_debug:
            await lifecycle_component.initialize()  # Should be ignored
            mock_debug.assert_called_once()

        assert lifecycle_component.state == LifecycleState.INITIALIZED

    @pytest.mark.asyncio
    async def test_initialize_from_invalid_state_raises_error(
        self, lifecycle_component
    ):
        """Test that initialize() raises InvalidStateError from invalid states."""
        await lifecycle_component._set_state(LifecycleState.STOPPED)

        with pytest.raises(InvalidStateError):
            await lifecycle_component.initialize()

    @pytest.mark.asyncio
    async def test_start_success(self, lifecycle_component):
        """Test successful start after initialization."""
        await lifecycle_component.initialize()
        await lifecycle_component.start()

        assert lifecycle_component.state == LifecycleState.RUNNING
        assert lifecycle_component.started_event.is_set()
        assert lifecycle_component.was_started
        assert lifecycle_component.is_running

    @pytest.mark.asyncio
    async def test_start_ignored_when_already_running(self, lifecycle_component):
        """Test that start() is ignored when already running."""
        await lifecycle_component.initialize()
        await lifecycle_component.start()

        with patch.object(lifecycle_component, "debug") as mock_debug:
            await lifecycle_component.start()  # Should be ignored
            mock_debug.assert_called_once()

        assert lifecycle_component.state == LifecycleState.RUNNING

    @pytest.mark.asyncio
    async def test_start_from_invalid_state_raises_error(self, lifecycle_component):
        """Test that start() raises InvalidStateError from invalid states."""
        # Try to start without initializing first
        with pytest.raises(InvalidStateError):
            await lifecycle_component.start()

    @pytest.mark.asyncio
    async def test_stop_success(self, lifecycle_component):
        """Test successful stop after start."""
        await lifecycle_component.initialize()
        await lifecycle_component.start()
        await lifecycle_component.stop()

        assert lifecycle_component.state == LifecycleState.STOPPED
        assert lifecycle_component.stopped_event.is_set()
        assert lifecycle_component.was_stopped
        assert not lifecycle_component.is_running
        assert lifecycle_component.stop_requested

    @pytest.mark.asyncio
    async def test_stop_when_already_requested_ignored(self, lifecycle_component):
        """Test that stop() is ignored when stop is already requested."""
        lifecycle_component.stop_requested = True

        with (
            patch.object(lifecycle_component, "debug") as mock_debug,
            patch.object(lifecycle_component, "run_hooks") as mock_run_hooks,
        ):
            await lifecycle_component.stop()
            mock_debug.assert_called_once()
            mock_run_hooks.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_requested_property(self, lifecycle_component):
        """Test the stop_requested property setter and getter."""
        assert not lifecycle_component.stop_requested

        lifecycle_component.stop_requested = True
        assert lifecycle_component.stop_requested
        assert lifecycle_component._stop_requested_event.is_set()

        lifecycle_component.stop_requested = False
        assert not lifecycle_component.stop_requested
        assert not lifecycle_component._stop_requested_event.is_set()

    @pytest.mark.asyncio
    async def test_initialize_and_start_convenience_method(self, lifecycle_component):
        """Test the initialize_and_start convenience method."""
        with (
            patch.object(lifecycle_component, "initialize") as mock_init,
            patch.object(lifecycle_component, "start") as mock_start,
        ):
            await lifecycle_component.initialize_and_start()

            mock_init.assert_called_once()
            mock_start.assert_called_once()

    def test_lifecycle_string_representation(self, lifecycle_component):
        """Test that the lifecycle has a proper string representation."""
        assert lifecycle_component.id in str(lifecycle_component)
        assert lifecycle_component.id in repr(lifecycle_component)
        assert "state=" in repr(lifecycle_component)

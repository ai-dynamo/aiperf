# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TimingManager service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import TimingMode
from aiperf.common.environment import Environment
from aiperf.common.messages import (
    DatasetTimingRequest,
    DatasetTimingResponse,
    ProfileConfigureCommand,
)
from aiperf.timing.timing_manager import TimingManager


class TestTimingManagerDatasetTimeout:
    """Test suite for TimingManager dataset request timeout behavior."""

    def _create_timing_manager(
        self, service_config: ServiceConfig, user_config: UserConfig
    ) -> TimingManager:
        """Create a TimingManager instance (ZMQ is globally mocked)."""
        return TimingManager(
            service_config=service_config,
            user_config=user_config,
            service_id="test-timing-manager",
        )

    @pytest.fixture
    def service_config(self):
        """Service configuration fixture."""
        return ServiceConfig()

    @pytest.fixture
    def user_config_fixed_schedule(self):
        """User config with fixed schedule timing mode."""
        return UserConfig.model_construct(
            endpoint=MagicMock(),
            _timing_mode=TimingMode.FIXED_SCHEDULE,
        )

    @pytest.fixture
    def user_config_request_rate(self):
        """User config with request rate timing mode."""
        return UserConfig.model_construct(
            endpoint=MagicMock(),
            _timing_mode=TimingMode.REQUEST_RATE,
        )

    @pytest.fixture
    def mock_dataset_client(self):
        """Create a mock dataset request client with a sample response."""
        client = AsyncMock()
        response = DatasetTimingResponse(
            service_id="test-dataset-manager",
            timing_data=[(0, "conv1"), (100, "conv2")],
        )
        client.request = AsyncMock(return_value=response)
        return client

    @pytest.mark.asyncio
    async def test_profile_configure_uses_dataset_timeout_for_fixed_schedule(
        self, service_config, user_config_fixed_schedule, mock_dataset_client
    ):
        """Test that profile configure command uses DATASET.CONFIGURATION_TIMEOUT for fixed schedule mode."""
        manager = self._create_timing_manager(
            service_config, user_config_fixed_schedule
        )
        manager.dataset_request_client = mock_dataset_client

        with patch(
            "aiperf.timing.timing_manager.CreditIssuingStrategyFactory.create_instance"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()

            command = ProfileConfigureCommand.model_construct(
                service_id="test-system-controller",
                config={},
            )

            await manager._profile_configure_command(command)

            # Verify dataset request client was called with correct timeout
            mock_dataset_client.request.assert_called_once()
            call_args = mock_dataset_client.request.call_args

            assert (
                call_args.kwargs["timeout"] == Environment.DATASET.CONFIGURATION_TIMEOUT
            )
            assert isinstance(call_args.kwargs["message"], DatasetTimingRequest)
            assert call_args.kwargs["message"].service_id == "test-timing-manager"

    @pytest.mark.asyncio
    async def test_profile_configure_does_not_call_dataset_for_request_rate(
        self, service_config, user_config_request_rate
    ):
        """Test that profile configure command does not call dataset manager for non-fixed-schedule modes."""
        manager = self._create_timing_manager(service_config, user_config_request_rate)

        mock_dataset_client = AsyncMock()
        manager.dataset_request_client = mock_dataset_client

        with patch(
            "aiperf.timing.timing_manager.CreditIssuingStrategyFactory.create_instance"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()

            command = ProfileConfigureCommand.model_construct(
                service_id="test-system-controller",
                config={},
            )

            await manager._profile_configure_command(command)

            # Verify dataset request client was NOT called for request rate mode
            mock_dataset_client.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_dataset_timeout_uses_environment_constant(
        self, service_config, user_config_fixed_schedule, mock_dataset_client
    ):
        """Test that the timeout value used is exactly Environment.DATASET.CONFIGURATION_TIMEOUT."""
        manager = self._create_timing_manager(
            service_config, user_config_fixed_schedule
        )
        manager.dataset_request_client = mock_dataset_client

        with patch(
            "aiperf.timing.timing_manager.CreditIssuingStrategyFactory.create_instance"
        ) as mock_factory:
            mock_factory.return_value = MagicMock()

            command = ProfileConfigureCommand.model_construct(
                service_id="test-system-controller",
                config={},
            )
            await manager._profile_configure_command(command)

            # Verify timeout matches the dataset configuration timeout constant
            call_args = mock_dataset_client.request.call_args
            assert (
                call_args.kwargs["timeout"] == Environment.DATASET.CONFIGURATION_TIMEOUT
            )

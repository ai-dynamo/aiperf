#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""
Tests for the timing manager service.
"""

import pytest
from pydantic import BaseModel

from aiperf.common.enums import ServiceType
from aiperf.common.service.base_service import BaseService
from aiperf.services.timing_manager.timing_manager import TimingManager
from aiperf.tests.base_test_component_service import BaseTestComponentService
from aiperf.tests.utils.async_test_utils import async_fixture


class TimingManagerTestConfig(BaseModel):
    """Configuration model for timing manager tests."""

    # TODO: Replace this with the actual configuration model once available
    pass


@pytest.mark.asyncio
class TestTimingManager(BaseTestComponentService):
    """
    Tests for the timing manager service.

    This test class extends BaseTestComponentService to leverage common
    component service tests while adding timing manager specific tests.
    """

    @pytest.fixture
    def service_class(self) -> type[BaseService]:
        """Return the service class to be tested."""
        return TimingManager

    @pytest.fixture
    def timing_config(self) -> TimingManagerTestConfig:
        """
        Return a test configuration for the timing manager.
        """
        return TimingManagerTestConfig()

    async def test_timing_manager_initialization(
        self, initialized_service: TimingManager
    ) -> None:
        """
        Test that the timing manager initializes with the correct service type.
        """
        service = await async_fixture(initialized_service)
        assert service.service_type == ServiceType.TIMING_MANAGER

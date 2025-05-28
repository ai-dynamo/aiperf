#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""
Tests for the post processor manager service.
"""

import pytest
from pydantic import BaseModel

from aiperf.common.enums import ServiceType
from aiperf.common.service.base_service import BaseService
from aiperf.services.post_processor_manager.post_processor_manager import (
    PostProcessorManager,
)
from aiperf.tests.base_test_component_service import BaseTestComponentService
from aiperf.tests.utils.async_test_utils import async_fixture


class PostProcessorTestConfig(BaseModel):
    """Configuration model for post processor manager tests."""

    # TODO: Replace this with the actual configuration model once available
    pass


@pytest.mark.asyncio
class TestPostProcessorManager(BaseTestComponentService):
    """
    Tests for the post processor manager service.

    This test class extends BaseTestComponentService to leverage common
    component service tests while adding post processor manager specific tests.
    """

    @pytest.fixture
    def service_class(self) -> type[BaseService]:
        """Return the service class to be tested."""
        return PostProcessorManager

    @pytest.fixture
    def processor_config(self) -> PostProcessorTestConfig:
        """
        Return a test configuration for the post processor manager.
        """
        return PostProcessorTestConfig()

    async def test_post_processor_manager_initialization(
        self, initialized_service: PostProcessorManager
    ) -> None:
        """
        Test that the post processor manager initializes with the correct service type.
        """
        service = await async_fixture(initialized_service)
        assert service.service_type == ServiceType.POST_PROCESSOR_MANAGER

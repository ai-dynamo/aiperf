# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the records manager service.
"""

import pytest
from pydantic import BaseModel

from aiperf.common.enums import ServiceType
from aiperf.common.service.base_service import BaseService
from aiperf.services.records_manager.records_manager import RecordsManager
from aiperf.tests.base_test_component_service import BaseTestComponentService
from aiperf.tests.utils.async_test_utils import async_fixture


class RecordsManagerTestConfig(BaseModel):
    """Configuration model for records manager tests."""

    # TODO: Replace this with the actual configuration model once available
    pass


@pytest.mark.asyncio
class TestRecordsManagerService(BaseTestComponentService):
    """
    Tests for the records manager service.

    This test class extends BaseTestComponentService to leverage common
    component service tests while adding records manager specific tests.
    """

    @pytest.fixture
    def service_class(self) -> type[BaseService]:
        """Return the service class to be tested."""
        return RecordsManager

    @pytest.fixture
    def records_config(self) -> RecordsManagerTestConfig:
        """
        Return a test configuration for the records manager.
        """
        return RecordsManagerTestConfig()

    async def test_records_manager_initialization(
        self, initialized_service: RecordsManager
    ) -> None:
        """
        Test that the records manager initializes with the correct service type.
        """
        service = await async_fixture(initialized_service)
        assert service.service_type == ServiceType.RECORDS_MANAGER

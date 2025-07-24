# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the worker manager service.
"""

import multiprocessing

import pytest
from pydantic import BaseModel

from aiperf.common.enums import ServiceType
from aiperf.services.base_service import BaseService
from aiperf.services.workers import WorkerManager
from tests.base_test_component_service import BaseTestComponentService
from tests.utils.async_test_utils import async_fixture


class WorkerManagerTestConfig(BaseModel):
    """Configuration model for worker manager tests."""

    # TODO: Replace this with the actual configuration model once available
    pass


@pytest.mark.asyncio
class WorkerManagerServiceTest(BaseTestComponentService):
    """
    Tests for the worker manager service.

    This test class extends BaseTestComponentService to leverage common
    component service tests while adding worker manager specific tests.
    """

    @pytest.fixture
    def service_class(self) -> type[BaseService]:
        """Return the service class to be tested."""
        return WorkerManager

    @pytest.fixture
    def worker_manager_config(self) -> WorkerManagerTestConfig:
        """
        Return a test configuration for the worker manager.
        """
        return WorkerManagerTestConfig()

    async def test_worker_manager_initialization(
        self, initialized_service: WorkerManager
    ) -> None:
        """
        Test that the worker manager initializes with the correct attributes.
        """
        service = await async_fixture(initialized_service)
        assert service.service_type == ServiceType.WORKER_MANAGER
        assert hasattr(service, "workers")
        assert hasattr(service, "cpu_count")
        assert service.cpu_count == multiprocessing.cpu_count()
        assert service.worker_count == service.cpu_count

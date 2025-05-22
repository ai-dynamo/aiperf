#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Tests for the worker manager service.
"""

import multiprocessing

import pytest
from pydantic import BaseModel

from aiperf.common.enums import ServiceType
from aiperf.common.service.base_service import BaseService
from aiperf.services.worker_manager.worker_manager import WorkerManager
from aiperf.tests.base_test_component_service import BaseTestComponentService
from aiperf.tests.utils.async_test_utils import async_fixture


class WorkerManagerTestConfig(BaseModel):
    """Configuration model for worker manager tests."""

    # TODO: Replace this with the actual configuration model once available
    pass


@pytest.mark.asyncio
class TestWorkerManager(BaseTestComponentService):
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

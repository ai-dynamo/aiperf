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
Tests for the dataset manager service.
"""

import pytest
from pydantic import BaseModel

from aiperf.common.enums import ServiceType
from aiperf.common.service.base_service import BaseService
from aiperf.services.dataset.dataset_manager import DatasetManager
from aiperf.tests.base_test_component_service import BaseTestComponentService
from aiperf.tests.utils.async_test_utils import async_fixture


class DatasetManagerTestConfig(BaseModel):
    """Configuration model for dataset manager tests."""

    # TODO: Replace this with the actual configuration model once available
    pass


@pytest.mark.asyncio
class TestDatasetManager(BaseTestComponentService):
    """
    Tests for the dataset manager service.

    This test class extends BaseTestComponentService to leverage common
    component service tests while adding dataset manager specific tests.
    """

    @pytest.fixture
    def service_class(self) -> type[BaseService]:
        """Return the service class to be tested."""
        return DatasetManager

    @pytest.fixture
    def dataset_config(self) -> DatasetManagerTestConfig:
        """
        Return a test configuration for the dataset manager.
        """
        return DatasetManagerTestConfig()

    async def test_dataset_manager_initialization(
        self, initialized_service: DatasetManager
    ) -> None:
        """
        Test that the dataset manager initializes with the correct service type.
        """
        service = await async_fixture(initialized_service)
        assert service.service_type == ServiceType.DATASET_MANAGER

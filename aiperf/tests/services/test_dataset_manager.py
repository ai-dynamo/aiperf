# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the dataset manager service.
"""

import pytest
from pydantic import BaseModel

from aiperf.common.enums import (
    ServiceType,
)
from aiperf.common.exceptions import ServiceError
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    DatasetTimingResponse,
)
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
    Tests for dataset manager service functionalities and basic properties.

    This test class extends BaseTestComponentService to leverage common
    component service tests while adding dataset manager specific tests
    for service properties and request handling.
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

    async def test_service_type_property(
        self, initialized_service: DatasetManager
    ) -> None:
        """
        Test that the dataset manager returns the correct service type.

        Verifies that the service_type property returns ServiceType.DATASET_MANAGER
        and that it matches the expected enum value.
        """
        service = await async_fixture(initialized_service)
        assert service.service_type == ServiceType.DATASET_MANAGER

    async def test_initialization_with_service_config(
        self, initialized_service: DatasetManager
    ) -> None:
        """
        Test that the dataset manager initializes properly with service configuration.

        Verifies that the service initializes correctly with the provided service config
        and that all required attributes are properly set during initialization.
        """
        service = await async_fixture(initialized_service)
        assert service.service_config is not None
        assert service.service_id is not None
        assert service.dataset == {}
        assert isinstance(service.dataset, dict)

    # ============================================================================
    # Test Error Handling
    # ============================================================================

    async def test_handle_conversation_request_empty_dataset(
        self, initialized_service, conversation_request_message
    ) -> None:
        """
        Test error handling when dataset is empty.

        Verifies that when a conversation request is made but the dataset
        is empty (not configured), a ServiceError is raised with an
        appropriate error message.
        """
        # Dataset is empty by default
        service = await async_fixture(initialized_service)

        with pytest.raises(ServiceError) as exc_info:
            await service._handle_conversation_request(conversation_request_message)

        assert "Dataset is empty" in str(exc_info.value)

    async def test_handle_conversation_request_missing_conversation(
        self, populated_dataset_manager, conversation_request_message
    ) -> None:
        """
        Test error handling when requested conversation ID is not found.

        Verifies that when a conversation request is made for a conversation_id
        that doesn't exist in the dataset, a ServiceError is raised with
        details about the missing conversation.
        """
        conversation_request_message.conversation_id = "nonexistent-conversation-id"

        with pytest.raises(ServiceError) as exc_info:
            await populated_dataset_manager._handle_conversation_request(
                conversation_request_message
            )

        assert "not found in dataset" in str(exc_info.value)

    async def test_handle_timing_request_empty_dataset(
        self, initialized_service, timing_request_message
    ) -> None:
        """
        Test error handling when dataset is empty during timing request.

        Verifies that when a timing request is made but the dataset is empty,
        a ServiceError is raised with an appropriate error message indicating
        the dataset needs to be configured first.
        """
        service = await async_fixture(initialized_service)

        with pytest.raises(ServiceError) as exc_info:
            await service._handle_dataset_timing_request(timing_request_message)

        assert "Dataset is empty" in str(exc_info.value)

    # ============================================================================
    # Test Request Handling
    # ============================================================================

    async def test_handle_conversation_request(self, populated_dataset_manager) -> None:
        """
        Test conversation retrieval with multiple conversations in dataset.
        """
        # first conversation
        request1 = ConversationRequestMessage(
            service_id="test-requester", request_id="req-1", conversation_id="session-1"
        )

        response1 = await populated_dataset_manager._handle_conversation_request(
            request1
        )
        assert isinstance(response1, ConversationResponseMessage)
        assert response1.conversation.id == request1.conversation_id
        assert response1.conversation.turns[0].timestamp == 1000

        # second conversation
        request2 = ConversationRequestMessage(
            service_id="test-requester", request_id="req-2", conversation_id="session-2"
        )

        response2 = await populated_dataset_manager._handle_conversation_request(
            request2
        )
        assert isinstance(response2, ConversationResponseMessage)
        assert response2.conversation.id == request2.conversation_id
        assert response2.conversation.turns[0].timestamp == 2000
        assert response2.conversation.turns[1].timestamp == 2500

    async def test_handle_timing_request(
        self, populated_dataset_manager, timing_request_message
    ) -> None:
        """
        Test timing data extraction from multiple conversations.

        Verifies that timing data is correctly extracted from all conversations
        in the dataset, and that each timing entry includes the correct
        conversation_id association.
        """
        response = await populated_dataset_manager._handle_dataset_timing_request(
            timing_request_message
        )

        assert isinstance(response, DatasetTimingResponse)
        assert response.service_id == populated_dataset_manager.service_id
        assert response.request_id == timing_request_message.request_id

        expected_timing_data = [
            (1000, "session-1"),
            (2000, "session-2"),
            (2500, "session-2"),
        ]
        assert response.timing_data == expected_timing_data

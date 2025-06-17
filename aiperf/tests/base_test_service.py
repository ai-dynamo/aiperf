# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Base test class for testing AIPerf services.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import CommunicationBackend, ServiceRunType, ServiceState
from aiperf.common.service.base_service import BaseService
from aiperf.tests.utils.async_test_utils import async_fixture, async_noop


class BaseTestService(ABC):
    """
    Base test class for all service tests.

    This class provides common test methods and fixtures for testing
    AIPerf services. Specific service test classes should inherit from
    this class and implement service-specific fixtures and tests.
    """

    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch) -> None:
        """
        Patch asyncio.sleep with a no-op to prevent test delays.

        This ensures tests don't need to wait for real sleep calls.
        """
        monkeypatch.setattr(asyncio, "sleep", async_noop)

    @pytest.fixture(autouse=True)
    def patch_communication_factory(
        self, mock_communication: MagicMock
    ) -> Generator[None, None, None]:
        """
        Patch the communication factory to always return our mock communication.

        This ensures no real communication is attempted during tests.
        """
        with patch(
            "aiperf.common.factories.CommunicationFactory.create_instance",
            return_value=mock_communication,
        ):
            yield

    @abstractmethod
    @pytest.fixture
    def service_class(self) -> type[BaseService]:
        """
        Return the service class to test.

        Must be implemented by subclasses to specify which service is being tested.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @pytest.fixture
    def service_config(self) -> ServiceConfig:
        """
        Create a service configuration for testing.

        Returns:
            A ServiceConfig instance with test settings
        """
        return ServiceConfig(
            service_run_type=ServiceRunType.MULTIPROCESSING,
            comm_backend=CommunicationBackend.ZMQ_TCP,
        )

    @pytest.fixture
    async def uninitialized_service(
        self,
        service_class: type[BaseService],
        service_config: ServiceConfig,
    ) -> AsyncGenerator[BaseService, None]:
        """
        Create an uninitialized instance of the service under test.

        This provides a service instance before initialize() has been called,
        allowing tests to verify initialization behavior.

        Returns:
            An uninitialized instance of the service

        Example usage:
        ```python
        async def test_service_initialization(uninitialized_service: BaseService):
            service = await async_fixture(uninitialized_service)
            await service.initialize()
        ```
        """
        # Patch the heartbeat task otherwise it will run forever
        with patch(
            "aiperf.common.service.base_component_service.BaseComponentService._heartbeat_task",
            lambda: None,
        ):
            service = service_class(service_config=service_config)
            yield service

    @pytest.fixture
    async def initialized_service(
        self,
        uninitialized_service: BaseService,
        mock_communication: MagicMock,
    ) -> AsyncGenerator[BaseService, None]:
        """
        Create and initialize the service under test.

        This fixture sets up a complete service instance ready for testing,
        with the communication layer mocked.

        Returns:
            An initialized instance of the service

        Example usage:
        ```python
        async def test_service_foo(initialized_service: BaseService):
            service = await async_fixture(initialized_service)
            await service.foo()
        ```
        """
        service = await async_fixture(uninitialized_service)

        await service.initialize()

        yield service

    @pytest.mark.asyncio
    async def test_service_initialization(
        self, uninitialized_service: BaseService, mock_communication: MagicMock
    ) -> None:
        """
        Test that the service initializes correctly. This will be executed
        for every service that inherits from BaseTestService.

        This verifies:
        1. The service has a valid ID and type
        2. The service transitions to the correct state during initialization
        3. The service's internal initialization method is called
        """
        service = await async_fixture(uninitialized_service)

        # Check that the service has an ID and type
        assert service.service_id is not None
        assert service.service_type is not None

        # Check that the service is not initialized
        assert service.state == ServiceState.UNKNOWN

        # Initialize the service
        await service.initialize()

        # Check that the service is initialized and in the READY state
        assert service.is_initialized
        assert service.state == ServiceState.READY

        await service.stop()

    @pytest.mark.asyncio
    async def test_service_start_stop(self, initialized_service: BaseService) -> None:
        """
        Test that the service can start and stop correctly. This will be executed
        for every service that inherits from BaseTestService.

        This verifies:
        1. The service transitions to the `ServiceState.RUNNING` state when started
        2. The service transitions to the `ServiceState.STOPPED` state when stopped
        """
        service = await async_fixture(initialized_service)

        # Start the service
        await service.start()
        assert service.state == ServiceState.RUNNING

        # Stop the service
        await service.stop()
        assert service.state == ServiceState.STOPPED

    @pytest.mark.parametrize(
        "state",
        [state for state in ServiceState if state != ServiceState.UNKNOWN],
    )
    @pytest.mark.asyncio
    async def test_service_state_transitions(
        self, initialized_service: BaseService, state: ServiceState
    ) -> None:
        """
        Test that the service can transition to all possible states. This will be executed
        for every service that inherits from BaseTestService.
        """
        service = await async_fixture(initialized_service)

        # Update the service state
        await service.set_state(state)

        # Check that the service state was updated
        assert service.state == state

    @pytest.mark.asyncio
    async def test_service_run_does_not_start(
        self, initialized_service: BaseService
    ) -> None:
        """
        Test that the service does not start when the run method is called (default behavior). This will be executed
        for every service that inherits from BaseTestService.
        """
        service = await async_fixture(MagicMock(wraps=initialized_service))

        service._forever_loop.return_value = None
        await service.run_forever()
        assert not service.start.called

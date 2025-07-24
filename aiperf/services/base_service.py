# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import uuid
from abc import ABC

from aiperf.common.comms.base import (
    BaseCommunication,
    CommunicationClientAddressType,
    CommunicationFactory,
)
from aiperf.common.config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ServiceState, ServiceType
from aiperf.common.exceptions import (
    AIPerfError,
    ServiceError,
)
from aiperf.common.hooks import (
    AIPerfHook,
    AIPerfTaskHook,
    supports_hooks,
)
from aiperf.common.messages import Message
from aiperf.common.mixins import AIPerfLoggerMixin, AIPerfTaskMixin
from aiperf.services.base_service_interface import BaseServiceInterface


@supports_hooks(
    AIPerfHook.ON_INIT,
    AIPerfHook.ON_RUN,
    AIPerfHook.ON_CONFIGURE,
    AIPerfHook.ON_START,
    AIPerfHook.ON_STOP,
    AIPerfHook.ON_CLEANUP,
    AIPerfHook.ON_SET_STATE,
    AIPerfTaskHook.AIPERF_TASK,
)
class BaseService(BaseServiceInterface, ABC, AIPerfTaskMixin, AIPerfLoggerMixin):
    """Base class for all AIPerf services, providing common functionality for
    communication, state management, and lifecycle operations.

    This class provides the foundation for implementing the various services of the
    AIPerf system. Some of the abstract methods are implemented here, while others
    are still required to be implemented by derived classes.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig | None = None,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        self.service_id: str = (
            service_id or f"{self.service_type}_{uuid.uuid4().hex[:8]}"
        )
        self.service_config = service_config
        self.user_config = user_config

        self._state: ServiceState = ServiceState.UNKNOWN

        super().__init__(
            service_id=service_id,
            service_config=service_config,
            user_config=user_config,
            logger_name=self.service_id,
            **kwargs,
        )

        self.debug(
            lambda: f"__init__ {self.service_type} service (id: {self.service_id})"
        )

        self._state: ServiceState = ServiceState.UNKNOWN

        self.stop_event = asyncio.Event()
        self.initialized_event = asyncio.Event()

        self.comms: BaseCommunication = CommunicationFactory.create_instance(
            self.service_config.comm_backend,
            config=self.service_config.comm_config,
        )
        self.sub_client = self.comms.create_sub_client(
            CommunicationClientAddressType.EVENT_BUS_PROXY_BACKEND
        )  # type: ignore
        self.pub_client = self.comms.create_pub_client(
            CommunicationClientAddressType.EVENT_BUS_PROXY_FRONTEND
        )  # type: ignore

        try:
            import setproctitle

            setproctitle.setproctitle(f"aiperf {self.service_id}")
        except Exception:
            # setproctitle is not available on all platforms, so we ignore the error
            self.logger.debug("Failed to set process title, ignoring")

        self.logger.debug(
            "BaseService._init__ finished for %s", self.__class__.__name__
        )

    @property
    def state(self) -> ServiceState:
        """The current state of the service."""
        return self._state

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized.

        Returns:
            True if service is initialized, False otherwise
        """
        return self.initialized_event.is_set()

    def _service_error(self, message: str) -> ServiceError:
        return ServiceError(
            message=message,
            service_type=self.service_type,
            service_id=self.service_id,
        )

    # Note: Not using as a setter so it can be overridden by derived classes and still
    # be async
    async def set_state(self, state: ServiceState) -> None:
        """Set the state of the service. This method implements
        the `BaseServiceInterface.set_state` method.

        This method will:
        - Set the service state to the given state
        - Call all registered `AIPerfHook.ON_SET_STATE` hooks
        """
        self._state = state
        await self.run_hooks(AIPerfHook.ON_SET_STATE, state)

    async def initialize(self) -> None:
        """Initialize the service communication and signal handlers. This method implements
        the `BaseServiceInterface.initialize` method.

        This method will:
        - Set the service to `ServiceState.INITIALIZING` state
        - Initialize communication
        - Call all registered `AIPerfHook.ON_INIT` hooks
        - Set the service to `ServiceState.READY` state
        - Set the initialized asyncio event
        """
        self._state = ServiceState.INITIALIZING

        await self.comms.initialize()

        # Initialize any derived service components
        await self.run_hooks(AIPerfHook.ON_INIT)
        await self.set_state(ServiceState.READY)

        self.initialized_event.set()

    async def run_forever(self) -> None:
        """Run the service in a loop until the stop event is set. This method implements
        the `BaseServiceInterface.run_forever` method.

        This method will:
        - Call the initialize method to initialize the service
        - Call all registered `AIPerfHook.RUN` hooks
        - Wait for the stop event to be set
        - Shuts down the service when the stop event is set

        This method will be called as the main entry point for the service.
        """
        try:
            self.logger.debug(
                "Running %s service (id: %s)", self.service_type, self.service_id
            )

            await self.initialize()
            await self.run_hooks(AIPerfHook.ON_RUN)

        except asyncio.CancelledError:
            self.logger.debug("Service %s execution cancelled", self.service_type)
            return

        except AIPerfError:
            raise  # re-raise it up the stack

        except Exception as e:
            self.logger.exception("Service %s execution failed:", self.service_type)
            _ = await self.set_state(ServiceState.ERROR)
            raise self._service_error("Service execution failed") from e

        await self._forever_loop()

    async def _forever_loop(self) -> None:
        """
        This method will be called by the `run_forever` method to allow the service to run
        indefinitely. This method is not expected to be overridden by derived classes.

        This method will:
        - Wait for the stop event to be set
        - Shuts down the service when the stop event is set
        """
        while not self.stop_event.is_set():
            try:
                self.logger.debug(
                    "Service %s waiting for stop event", self.service_type
                )
                # Wait forever for the stop event to be set
                await self.stop_event.wait()

            except asyncio.CancelledError:
                self.logger.debug(
                    "Service %s received CancelledError, exiting",
                    self.service_type,
                )
                break

            except Exception:
                self.logger.exception(
                    "Caught unexpected exception in service %s execution",
                    self.service_type,
                )

        # Shutdown the service
        try:
            await self.stop()
        except Exception:
            self.logger.exception(
                "Caught unexpected exception in service %s stop",
                self.service_type,
            )

    async def start(self) -> None:
        """Start the service and its components. This method implements
        the `BaseServiceInterface.start` method.

        This method should be called to start the service after it has been initialized
        and configured.

        This method will:
        - Set the service to `ServiceState.STARTING` state
        - Call all registered `AIPerfHook.ON_START` hooks
        - Set the service to `ServiceState.RUNNING` state
        """

        try:
            self.logger.debug(
                "Starting %s service (id: %s)", self.service_type, self.service_id
            )
            _ = await self.set_state(ServiceState.STARTING)

            await self.run_hooks(AIPerfHook.ON_START)

            _ = await self.set_state(ServiceState.RUNNING)

        except asyncio.CancelledError:
            pass

        except Exception as e:
            self._state = ServiceState.ERROR
            raise self._service_error("Failed to start service") from e

    async def stop(self) -> None:
        """Stop the service and clean up its components. This method implements
        the `BaseServiceInterface.stop` method.

        This method will:
        - Set the service to `ServiceState.STOPPING` state
        - Call all registered `AIPerfHook.ON_STOP` hooks
        - Shutdown the service communication component
        - Call all registered `AIPerfHook.ON_CLEANUP` hooks
        - Set the service to `ServiceState.STOPPED` state
        """
        try:
            if self.state == ServiceState.STOPPED:
                self.logger.warning(
                    "Service %s state %s is already STOPPED, ignoring stop request",
                    self.service_type,
                    self.state,
                )
                return

            self._state = ServiceState.STOPPING

            # Signal the run method to exit if it hasn't already
            if not self.stop_event.is_set():
                self.stop_event.set()

            cancelled_error = None
            # Custom stop logic implemented by derived classes
            try:
                await self.run_hooks(AIPerfHook.ON_STOP)
            except asyncio.CancelledError as e:
                cancelled_error = e

            # Shutdown communication component
            if self.comms and not self.comms.stop_requested:
                try:
                    await self.comms.shutdown()
                except asyncio.CancelledError as e:
                    cancelled_error = e

            # Custom cleanup logic implemented by derived classes
            try:
                await self.run_hooks(AIPerfHook.ON_CLEANUP)
            except asyncio.CancelledError as e:
                cancelled_error = e

            # Set the state to STOPPED. Communications are shutdown, so we don't need to
            # publish a status message
            self._state = ServiceState.STOPPED
            if self.service_type not in (
                ServiceType.WORKER,
                ServiceType.WORKER_MANAGER,
            ):
                self.logger.debug(
                    "Service %s (id: %s) stopped", self.service_type, self.service_id
                )

            # Re-raise the cancelled error if it was raised during the stop hooks
            if cancelled_error:
                raise cancelled_error

        except Exception as e:
            self._state = ServiceState.ERROR
            raise self._service_error("Failed to stop service") from e

    async def configure(self, message: Message) -> None:
        """Configure the service with the given configuration. This method implements
        the `BaseServiceInterface.configure` method.

        This method will:
        - Call all registered AIPerfHook.ON_CONFIGURE hooks
        """
        await self.run_hooks(AIPerfHook.ON_CONFIGURE, message)

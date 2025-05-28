# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import asyncio
import contextlib
import logging
import signal
import uuid
from abc import ABC
from collections import defaultdict
from collections.abc import Callable

import setproctitle

from aiperf.common.comms import BaseCommunication, CommunicationFactory
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.decorators import AIPerfHooks
from aiperf.common.enums import ServiceState
from aiperf.common.exceptions import (
    AIPerfMultiError,
    CommunicationClientCreationError,
    CommunicationCreateError,
    CommunicationNotInitializedError,
    ServiceInitializationError,
    ServiceRunError,
    ServiceStartError,
    ServiceStopError,
)
from aiperf.common.models import BaseMessage, Message, Payload
from aiperf.common.service.base_service_interface import BaseServiceInterface
from aiperf.common.service.service_metaclass import ServiceMetaclass
from aiperf.common.utils import call_all_functions_self


class BaseService(BaseServiceInterface, ABC, metaclass=ServiceMetaclass):
    """Base class for all AIPerf services, providing common functionality for
    communication, state management, and lifecycle operations.

    This class provides the foundation for implementing the various services of the
    AIPerf system. Some of the abstract methods are implemented here, while others
    are still required to be implemented by derived classes.
    """

    _aiperf_hooks: dict[str, list[Callable]] = defaultdict(list)

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        self.service_id: str = (
            service_id or f"{self.service_type}_{uuid.uuid4().hex[:8]}"
        )
        self.service_config = service_config

        self.logger = logging.getLogger(self.service_type)
        self.logger.debug(
            f"Initializing {self.service_type} service (id: {self.service_id})"
        )

        self._state: ServiceState = ServiceState.UNKNOWN
        self._heartbeat_interval = self.service_config.heartbeat_interval

        self._task_registry: dict[str, asyncio.Task] = {}

        self.stop_event = asyncio.Event()
        self.initialized_event = asyncio.Event()

        self._comms: BaseCommunication | None = None

        # Set to store signal handler tasks
        self._signal_tasks = set()

        # noinspection PyBroadException
        try:
            setproctitle.setproctitle(f"aiperf {self.service_id}")
        except:  # noqa: E722
            # setproctitle is not available on all platforms, so we ignore the error
            self.logger.debug("Failed to set process title, ignoring")

        self.logger.debug("__init__ finished for %s", self.__class__.__name__)

    @property
    def comms(self) -> BaseCommunication:
        """
        Get the communication object for the service.
        Raises:
            CommunicationNotInitializedError: If the communication is not initialized
        """
        if not self._comms:
            raise CommunicationNotInitializedError()
        return self._comms

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

    @property
    def is_shutdown(self) -> bool:
        """Check if service is shutdown.

        Returns:
            True if service is shutdown, False otherwise
        """
        return self.stop_event.is_set()

    def _get_hooks(self, hook_type: AIPerfHooks) -> list[Callable]:
        """Get all the hooks for the given hook type."""
        self.logger.debug(
            f"Getting hooks for {self.service_id}.{hook_type}: {self._aiperf_hooks[hook_type]}"
        )
        return self._aiperf_hooks[hook_type]

    async def _run_hooks(self, hook_type: AIPerfHooks, *args, **kwargs) -> None:
        """Run all the hooks for the given hook type.

        Args:
            hook_type: The type of hook to run
            *args: The arguments to pass to the hooks
            **kwargs: The keyword arguments to pass to the hooks

        Raises:
            AIPerfMultiError: If any of the hooks raise an exception
        """
        await call_all_functions_self(self, self._get_hooks(hook_type), *args, **kwargs)

    # Note: Not using as a setter so it can be overridden by derived classes and still
    # be async
    async def set_state(self, state: ServiceState) -> None:
        """Set the state of the service. This method implements
        the `BaseServiceInterface.set_state` method.

        This method will:
        - Set the service state to the given state
        - Call all registered `AIPerfHooks.SET_STATE` hooks
        """
        self._state = state
        await self._run_hooks(AIPerfHooks.SET_STATE, state)

    async def initialize(self) -> None:
        """Initialize the service communication and signal handlers. This method implements
        the `BaseServiceInterface.initialize` method.

        This method will:
        - Set the service to `ServiceState.INITIALIZING` state
        - Set up signal handlers for graceful shutdown
        - Allow time for the event loop to start
        - Initialize communication
        - Call all registered `AIPerfHooks.INIT` hooks
        - Set the service to `ServiceState.READY` state
        - Set the initialized asyncio event
        """
        self._state = ServiceState.INITIALIZING
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        # Allow time for the event loop to start
        await asyncio.sleep(0.1)

        # Initialize communication
        try:
            self._comms = CommunicationFactory.create_communication(self.service_config)
        except Exception as e:
            self.logger.exception(
                "Failed to create communication for service %s (id: %s)",
                self.service_type,
                self.service_id,
            )
            raise CommunicationCreateError from e

        try:
            await self._comms.initialize()
        except Exception as e:
            self.logger.exception(
                "Failed to initialize communication for service %s (id: %s)",
                self.service_type,
                self.service_id,
            )
            raise ServiceInitializationError from e

        if len(self.required_clients) > 0:
            # Create the communication clients ahead of time
            self.logger.debug(
                "%s: Creating communication clients (%s)",
                self.service_type,
                self.required_clients,
            )

            try:
                await self._comms.create_clients(*self.required_clients)
            except Exception as e:
                self.logger.exception(
                    "Failed to create communication clients for service %s (id: %s)",
                    self.service_type,
                    self.service_id,
                )
                raise CommunicationClientCreationError from e

        # Initialize any derived service components
        try:
            await self._run_hooks(AIPerfHooks.INIT)
        except AIPerfMultiError as e:
            self.logger.exception(
                "Failed to initialize service %s (id: %s)",
                self.service_type,
                self.service_id,
            )
            raise ServiceInitializationError from e

        await self.set_state(ServiceState.READY)

        self.initialized_event.set()

    async def run_forever(self) -> None:
        """Run the service in a loop until the stop event is set. This method implements
        the `BaseServiceInterface.run_forever` method.

        This method will:
        - Call the initialize method to initialize the service
        - Start all the registered tasks
        - Call all registered `AIPerfHooks.RUN` hooks
        - Wait for the stop event to be set
        - Shuts down the service when the stop event is set

        This method will be called as the main entry point for the service.
        """
        try:
            self.logger.debug(
                "Running %s service (id: %s)", self.service_type, self.service_id
            )

            try:
                await self.initialize()
            except Exception as e:
                self.logger.exception(
                    "Failed to initialize service %s (id: %s)",
                    self.service_type,
                    self.service_id,
                )
                raise ServiceRunError from e

            # Start all registered tasks
            for hook in self._get_hooks(AIPerfHooks.TASK):
                # TODO: support task intervals
                self._task_registry[hook.__name__] = asyncio.create_task(hook(self))

            try:
                await self._run_hooks(AIPerfHooks.RUN)
            except AIPerfMultiError as e:
                self.logger.exception(
                    "Failed to run service %s (id: %s)",
                    self.service_type,
                    self.service_id,
                )
                raise ServiceRunError from e

        except asyncio.CancelledError:
            self.logger.debug("Service %s execution cancelled", self.service_type)
            return

        except Exception as e:
            self.logger.exception("Service %s execution failed:", self.service_type)
            _ = await self.set_state(ServiceState.ERROR)
            raise ServiceRunError(
                "Service %s execution failed", self.service_type
            ) from e

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

            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                self.logger.debug("Service %s execution cancelled", self.service_type)
                self.stop_event.set()

            except Exception:
                self.logger.exception(
                    "Caught unexpected exception in service %s execution",
                    self.service_type,
                )
            finally:
                # Shutdown the service
                try:
                    await self.stop()
                except Exception as e:
                    self.logger.exception(
                        "Exception stopping service %s", self.service_type
                    )
                    raise ServiceStopError(
                        "Exception stopping service %s", self.service_type
                    ) from e

    async def start(self) -> None:
        """Start the service and its components. This method implements
        the `BaseServiceInterface.start` method.

        This method should be called to start the service after it has been initialized
        and configured.

        This method will:
        - Set the service to `ServiceState.STARTING` state
        - Call all registered `AIPerfHooks.START` hooks
        - Set the service to `ServiceState.RUNNING` state
        """

        try:
            self.logger.debug(
                "Starting %s service (id: %s)", self.service_type, self.service_id
            )
            _ = await self.set_state(ServiceState.STARTING)

            await self._run_hooks(AIPerfHooks.START)

            _ = await self.set_state(ServiceState.RUNNING)

        except asyncio.CancelledError:
            self.logger.debug("Service %s execution cancelled", self.service_type)

        except Exception as e:
            self.logger.exception(
                "Failed to start service %s (id: %s)",
                self.service_type,
                self.service_id,
            )
            self._state = ServiceState.ERROR

            raise ServiceStartError(
                "Failed to start service %s (id: %s)",
                self.service_type,
                self.service_id,
            ) from e

    async def stop(self) -> None:
        """Stop the service and clean up its components. This method implements
        the `BaseServiceInterface.stop` method.

        This method will:
        - Set the service to `ServiceState.STOPPING` state
        - Call all registered `AIPerfHooks.STOP` hooks
        - Shutdown the service communication component
        - Cancel all registered tasks
        - Call all registered `AIPerfHooks.CLEANUP` hooks
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

            # ignore if we were unable to send the STOPPING state message
            _ = await self.set_state(ServiceState.STOPPING)

            # Signal the run method to exit if it hasn't already
            if not self.stop_event.is_set():
                self.stop_event.set()

            # Custom stop logic implemented by derived classes
            with contextlib.suppress(asyncio.CancelledError):
                await self._run_hooks(AIPerfHooks.STOP)

            # Shutdown communication component
            if self._comms and not self._comms.is_shutdown:
                await self._comms.shutdown()

            # Cancel all registered tasks
            for task in self._task_registry.values():
                task.cancel()

            # Wait for all tasks to complete
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*self._task_registry.values())

            # Custom cleanup logic implemented by derived classes
            with contextlib.suppress(asyncio.CancelledError):
                await self._run_hooks(AIPerfHooks.CLEANUP)

            # Set the state to STOPPED. Communications are shutdown, so we don't need to
            # publish a status message
            self._state = ServiceState.STOPPED
            self.logger.debug(
                "Service %s (id: %s) stopped", self.service_type, self.service_id
            )

        except Exception as e:
            self.logger.exception(
                "Failed to stop service %s (id: %s)",
                self.service_type,
                self.service_id,
            )
            self._state = ServiceState.ERROR
            raise ServiceStopError(
                "Failed to stop service %s (id: %s)",
                self.service_type,
                self.service_id,
            ) from e

    async def configure(self, message: Message) -> None:
        """Configure the service with the given configuration. This method implements
        the `BaseServiceInterface.configure` method.

        This method will:
        - Call all registered AIPerfHooks.CONFIGURE hooks
        """
        await self._run_hooks(AIPerfHooks.CONFIGURE, message)

    def create_message(
        self, payload: Payload, request_id: str | None = None
    ) -> Message:
        """Create a message of the given type, and pre-fill the service_id.

        Args:
            payload: The payload of the message
            request_id: optional The request id of this message, or the request id of the
                message this is a response to

        Returns:
            A message of the given type
        """
        message = BaseMessage(
            service_id=self.service_id,
            request_id=request_id,
            payload=payload,
        )
        return message

    def _setup_signal_handlers(self) -> None:
        """This method will set up signal handlers for the SIGTERM and SIGINT signals
        in order to trigger a graceful shutdown of the service.
        """
        loop = asyncio.get_running_loop()

        def signal_handler(sig: int) -> None:
            # Create a task and store it so it doesn't get garbage collected
            task = asyncio.create_task(self._handle_signal(sig))

            # Store the task somewhere to prevent it from being garbage collected
            # before it completes
            self._signal_tasks.add(task)
            task.add_done_callback(self._signal_tasks.discard)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    async def _handle_signal(self, sig: int) -> None:
        """Handle received signals by triggering graceful shutdown.

        Args:
            sig: The signal number received
        """
        signal_name = signal.Signals(sig).name
        self.logger.debug(
            "%s: Received signal %s, initiating graceful shutdown",
            self.service_type,
            signal_name,
        )

        self.stop_event.set()

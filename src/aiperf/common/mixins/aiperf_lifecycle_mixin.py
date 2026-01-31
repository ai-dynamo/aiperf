# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiperf.common.enums import LifecycleState
from aiperf.common.exceptions import (
    InvalidStateError,
    LifecycleOperationError,
)
from aiperf.common.hooks import (
    AIPerfHook,
    BackgroundTaskParams,
    on_init,
    on_start,
    on_stop,
    provides_hooks,
)
from aiperf.common.mixins.hooks_mixin import HooksMixin
from aiperf.common.mixins.task_manager_mixin import TaskManagerMixin
from aiperf.common.models import ErrorDetails, ExitErrorInfo
from aiperf.common.protocols import AIPerfLifecycleProtocol


@provides_hooks(
    AIPerfHook.ON_INIT,
    AIPerfHook.ON_START,
    AIPerfHook.ON_STOP,
    AIPerfHook.ON_STATE_CHANGE,
    AIPerfHook.BACKGROUND_TASK,
)
class AIPerfLifecycleMixin(TaskManagerMixin, HooksMixin):
    """This mixin provides a lifecycle state machine, and is the basis for most components in the AIPerf framework.
    It provides a set of hooks that are run at each state transition, and the ability to define background tasks
    that are automatically ran on @on_start, and canceled via @on_stop.

    It exposes to the outside world `initialize`, `start`, and `stop` methods, as well as getting the
    current state of the lifecycle. These simple methods promote a simple interface for users to interact with.
    """

    def __init__(self, id: str | None = None, **kwargs) -> None:
        """
        Args:
            id: The id of the lifecycle. If not provided, a random uuid will be generated.
        """
        self.id = id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self._state = LifecycleState.CREATED
        self.initialized_event = asyncio.Event()
        self.started_event = asyncio.Event()
        self._stop_requested_event = asyncio.Event()
        self.stopped_event = asyncio.Event()  # set on stop or failure
        self._children: list[AIPerfLifecycleProtocol] = []
        self._exit_errors: list[ExitErrorInfo] = []
        if "logger_name" not in kwargs:
            kwargs["logger_name"] = self.id
        super().__init__(**kwargs)

    @property
    def state(self) -> LifecycleState:
        return self._state

    # NOTE: This was moved to not be a property setter, as we want it to be async so we can
    # run the hooks and await them. Otherwise there is issues with creating a task when the
    # lifecycle is trying to stop.
    async def _set_state(self, state: LifecycleState) -> None:
        if state == self._state:
            return
        old_state = self._state
        self._state = state
        if self.is_trace_enabled:
            self.trace(f"State changed from {old_state!r} to {state!r} for {self}")
        await self.run_hooks(
            AIPerfHook.ON_STATE_CHANGE, old_state=old_state, new_state=state
        )

    @property
    def was_initialized(self) -> bool:
        return self.initialized_event.is_set()

    @property
    def was_started(self) -> bool:
        return self.started_event.is_set()

    @property
    def was_stopped(self) -> bool:
        return self.stopped_event.is_set()

    @property
    def is_running(self) -> bool:
        """Whether the lifecycle's current state is LifecycleState.RUNNING."""
        return self.state == LifecycleState.RUNNING

    @property
    def stop_requested(self) -> bool:
        """Whether the lifecycle has been requested to stop."""
        return self._stop_requested_event.is_set()

    @stop_requested.setter
    def stop_requested(self, value: bool) -> None:
        if value:
            self._stop_requested_event.set()
        else:
            self._stop_requested_event.clear()

    async def _execute_state_transition(
        self,
        transient_state: LifecycleState,
        final_state: LifecycleState,
        hook_type: AIPerfHook,
        event: asyncio.Event,
        reverse: bool = False,
    ) -> None:
        """This method wraps the functionality of changing the state of the lifecycle, and running the hooks.
        It is used to ensure that the state change and hook running are atomic, and that the state change is
        only made after the hooks have completed. It also takes in an event that is set when the state change is complete.
        This is useful for external code waiting for the state change to complete before continuing.

        If reverse is True, the hooks are run in reverse order. This is useful for stopping the lifecycle in the reverse order of starting it.
        """
        await self._set_state(transient_state)
        self.debug(lambda: f"{transient_state.title()} {self}")
        try:
            await self.run_hooks(hook_type, reverse=reverse)
            await self._set_state(final_state)
            self.debug(lambda: f"{self} is now {final_state.title()}")
            event.set()
        except Exception as e:
            await self._fail(e)

    async def initialize(self) -> None:
        """Initialize the lifecycle and run the @on_init hooks.

        NOTE: It is generally discouraged from overriding this method.
        Instead, use the @on_init hook to handle your own initialization logic.
        """
        if self.state in (
            LifecycleState.INITIALIZING,
            LifecycleState.INITIALIZED,
            LifecycleState.STARTING,
            LifecycleState.RUNNING,
        ):
            self.debug(
                lambda: f"Ignoring initialize request for {self} in state {self.state}"
            )
            return

        if self.state != LifecycleState.CREATED:
            raise InvalidStateError(
                f"Cannot initialize from state {self.state} for {self}"
            )

        await self._execute_state_transition(
            LifecycleState.INITIALIZING,
            LifecycleState.INITIALIZED,
            AIPerfHook.ON_INIT,
            self.initialized_event,
        )

    async def start(self) -> None:
        """Start the lifecycle and run the @on_start hooks.

        NOTE: It is generally discouraged from overriding this method.
        Instead, use the @on_start hook to handle your own starting logic.
        """
        if self.state in (
            LifecycleState.STARTING,
            LifecycleState.RUNNING,
        ):
            self.debug(
                lambda: f"Ignoring start request for {self} in state {self.state}"
            )
            return

        if self.state != LifecycleState.INITIALIZED:
            raise InvalidStateError(f"Cannot start from state {self.state} for {self}")

        await self._execute_state_transition(
            LifecycleState.STARTING,
            LifecycleState.RUNNING,
            AIPerfHook.ON_START,
            self.started_event,
        )

    @asynccontextmanager
    async def try_operation_or_stop(self, operation: str) -> AsyncIterator[None]:
        """Context manager to try an operation or stop the lifecycle on failure.

        This context manager catches any exception, logs it, and raises a LifecycleOperationError
        which will be caught by the lifecycle system and trigger a graceful shutdown via _fail().

        Args:
            operation: Description of the operation being performed

        Raises:
            LifecycleOperationError: When the operation fails, triggering graceful shutdown
        """
        try:
            yield
        except LifecycleOperationError as e:
            # Log error and re-raise without wrapping to avoid duplicate error info
            self.error(f"Failed to {operation.lower()}: {e}")
            self._exit_errors.append(
                ExitErrorInfo(
                    error_details=ErrorDetails.from_exception(e),
                    operation=operation,
                    service_id=self.id,
                )
            )
            raise
        except Exception as e:
            self.error(f"Failed to {operation.lower()}: {e}")
            error = LifecycleOperationError(
                operation=operation,
                original_exception=e,
                lifecycle_id=self.id,
            )
            self._exit_errors.append(
                ExitErrorInfo.from_lifecycle_operation_error(error)
            )
            raise error from e

    async def initialize_and_start(self) -> None:
        """Initialize and start the lifecycle. This is a convenience method that calls `initialize` and `start` in sequence."""
        async with self.try_operation_or_stop("Initialize"):
            await self.initialize()
        async with self.try_operation_or_stop("Start"):
            await self.start()

    async def stop(self) -> None:
        """Stop the lifecycle and run the @on_stop hooks.

        NOTE: It is generally discouraged from overriding this method.
        Instead, use the @on_stop hook to handle your own stopping logic.
        """
        if self.stop_requested:
            self.debug(
                lambda: f"Ignoring stop request for {self} in state {self.state}"
            )
            return

        self.stop_requested = True
        await self._execute_state_transition(
            LifecycleState.STOPPING,
            LifecycleState.STOPPED,
            AIPerfHook.ON_STOP,
            self.stopped_event,
            reverse=True,  # run the stop hooks in reverse order
        )

    @on_start
    async def _start_background_tasks(self) -> None:
        """Start all tasks that are decorated with the @background_task decorator."""
        for hook in self.get_hooks(AIPerfHook.BACKGROUND_TASK):
            if not isinstance(hook.params, BackgroundTaskParams):
                raise AttributeError(
                    f"Invalid hook parameters for {hook}: {hook.params}. Expected BackgroundTaskParams."
                )
            self.start_background_task(
                hook.func,
                interval=hook.params.interval,
                immediate=hook.params.immediate,
                stop_on_error=hook.params.stop_on_error,
                stop_event=self._stop_requested_event,
            )

    @on_stop
    async def _stop_all_tasks(self) -> None:
        """Stop all tasks that are decorated with the @background_task decorator,
        and any custom ones that were ran using `self.execute_async()`.
        """
        await self.cancel_all_tasks()

    async def _fail(self, e: Exception) -> None:
        """Set the state to FAILED and raise an asyncio.CancelledError.
        This is used when the transition from one state to another fails.
        """
        self.error(f"Failed for {self}: {e}")
        if not isinstance(e, LifecycleOperationError):
            # Only add to exit errors if the exception has not already been added
            self._exit_errors.append(
                ExitErrorInfo(
                    error_details=ErrorDetails.from_exception(e),
                    operation=self.state.title(),
                    service_id=self.id,
                )
            )
        if self.state != LifecycleState.STOPPING:
            self.debug(f"Stopping {self} due to failure")
            await self.stop()
        await self._set_state(LifecycleState.FAILED)
        raise asyncio.CancelledError(f"Failed for {self}: {e}") from e

    def attach_child_lifecycle(self, child: AIPerfLifecycleProtocol) -> None:
        """Attach a child lifecycle to manage. This child will now have its lifecycle managed and
        controlled by this lifecycle. Common use cases are having a Service be a parent lifecycle,
        and having supporting components such as streaming post processors, progress reporters, etc. be children.

        Children will be called in the order they were attached for initialize and start,
        and in reverse order for stop.
        """
        if self.state != LifecycleState.CREATED:
            raise InvalidStateError(
                f"Cannot attach child {child} to {self} in state {self.state}. "
                "Please attach children before initializing or starting the lifecycle."
            )
        self._children.append(child)

    @on_init
    async def _initialize_children(self) -> None:
        """Initialize all children. This is done via the @on_init hook to ensure that the children
        initialize along with the parent hooks, and not after the parent hooks, which would cause
        a race condition.
        """
        for child in self._children:
            await child.initialize()

    @on_start
    async def _start_children(self) -> None:
        """Start all children. This is done via the @on_start hook to ensure that the children
        start along with the parent hooks, and not after the parent hooks, which would cause
        a race condition.
        """
        for child in self._children:
            await child.start()

    @on_stop
    async def _stop_children(self) -> None:
        """Stop all children. This is done via the @on_stop hook to ensure that the children
        are stopped along with the parent hooks, and not after the parent hooks, which would cause
        a race condition.
        """
        for child in reversed(self._children):
            await child.stop()

    def __str__(self) -> str:
        return f"{self.__class__.__name__} (id={self.id})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__qualname__} {self.id} (state={self.state})>"

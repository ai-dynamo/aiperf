# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import uuid

from aiperf.common.enums.service_enums import LifecycleState
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.hooks import (
    AIPerfHook,
    BackgroundTaskParams,
    on_start,
    on_stop,
    provides_hooks,
)
from aiperf.common.mixins.hooks_mixin import HooksMixin
from aiperf.common.mixins.task_manager_mixin import (
    TaskManagerMixin,
)


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
        self.id = id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self._state = LifecycleState.CREATED
        self.initialized_event = asyncio.Event()
        self.started_event = asyncio.Event()
        self._stop_requested_event = asyncio.Event()
        self.stopped_event = asyncio.Event()  # set on stop or failure
        if "logger_name" not in kwargs:
            kwargs["logger_name"] = self.id
        super().__init__(**kwargs)

    @property
    def state(self) -> LifecycleState:
        return self._state

    @state.setter
    def state(self, state: LifecycleState) -> None:
        if state == self._state:
            return
        old_state = self._state
        self._state = state
        self.debug(lambda: f"State changed from {old_state!r} to {state!r} for {self}")

        self.execute_async(
            self.run_hooks(
                AIPerfHook.ON_STATE_CHANGE, old_state=old_state, new_state=state
            )
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

    async def _execute_state_transition(
        self,
        transient_state: LifecycleState,
        final_state: LifecycleState,
        hook_type: AIPerfHook,
        event: asyncio.Event,
        reversed: bool = False,
    ) -> None:
        """This method wraps the functionality of changing the state of the lifecycle, and running the hooks.
        It is used to ensure that the state change and hook running are atomic, and that the state change is
        only made after the hooks have completed. It also take in an event that is set when the state change is complete.
        This is useful for external code waiting for the state change to complete before continuing.
        """
        self.state = transient_state
        self.debug(lambda: f"{transient_state.title()} {self}")
        try:
            await self.run_hooks(hook_type, reversed=reversed)
            self.state = final_state
            self.debug(lambda: f"{self} is now {final_state.title()}")
            event.set()
        except Exception as e:
            self._fail(e)

    async def initialize(self) -> None:
        """Initialize the lifecycle and run the @on_init hooks."""
        if self.state != LifecycleState.CREATED:
            raise InvalidStateError(f"Cannot initialize from state {self.state}")

        await self._execute_state_transition(
            LifecycleState.INITIALIZING,
            LifecycleState.INITIALIZED,
            AIPerfHook.ON_INIT,
            self.initialized_event,
        )

    async def start(self) -> None:
        """Start the lifecycle and run the @on_start hooks."""
        if self.state != LifecycleState.INITIALIZED:
            raise InvalidStateError(f"Cannot start from state {self.state}")

        await self._execute_state_transition(
            LifecycleState.STARTING,
            LifecycleState.RUNNING,
            AIPerfHook.ON_START,
            self.started_event,
        )

    async def initialize_and_start(self) -> None:
        """Initialize and start the lifecycle. This is a convenience method that calls `initialize` and `start` in sequence."""
        await self.initialize()
        await self.start()

    async def stop(self) -> None:
        """Stop the lifecycle and run the @on_stop hooks."""
        if self.stop_requested:
            # If we are already in a stopping state, we need to kill the process to be safe.
            self.warning(f"Attempted to stop {self} in state {self.state}. Killing.")
            await self._kill()
            return

        self._stop_requested_event.set()
        await self._execute_state_transition(
            LifecycleState.STOPPING,
            LifecycleState.STOPPED,
            AIPerfHook.ON_STOP,
            self.stopped_event,
            reversed=True,  # run the stop hooks in reverse order
        )

    @on_start
    async def _start_background_tasks(self) -> None:
        """Start all tasks that are decorated with the @background_task decorator."""
        for hook in self.get_hooks(AIPerfHook.BACKGROUND_TASK):
            if not isinstance(hook.params, BackgroundTaskParams):
                raise AttributeError(
                    f"Invalid hook parameters for {hook}: {hook.params}"
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

    async def _kill(self) -> None:
        """Kill the lifecycle. This is used when the lifecycle is requested to stop, but is already in a stopping state.
        This is a last resort to ensure that the lifecycle is stopped.
        """
        self.state = LifecycleState.FAILED
        self.debug(lambda: f"Killed {self}")
        self.stopped_event.set()
        os.kill(os.getpid(), signal.SIGKILL)
        raise asyncio.CancelledError(f"Killed {self}")

    def _fail(self, e: Exception) -> None:
        """Set the state to FAILED and raise an asyncio.CancelledError.
        This is used when the transition from one state to another fails.
        """
        self.state = LifecycleState.FAILED
        self.exception(f"Failed for {self}: {e}")
        self._stop_requested_event.set()
        self.stopped_event.set()
        raise asyncio.CancelledError(f"Failed for {self}: {e}") from e

    def __str__(self) -> str:
        return f"{self.__class__.__name__} (id={self.id})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__qualname__} {self.id} (state={self.state})>"


# # Add this file as one to be ignored when finding the caller of aiperf_logger.
# # This helps to make it more transparent where the actual function is being called from.
from aiperf.common import aiperf_logger  # noqa: E402 I001

_srcfile = os.path.normcase(AIPerfLifecycleMixin.initialize.__code__.co_filename)
aiperf_logger._ignored_files.append(_srcfile)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This module provides an extensive set of hook definitions for AIPerf. It is designed to be
used in conjunction with the :class:`HooksMixin` for classes to provide support for hooks.
It provides a simple interface for registering hooks.

Classes should inherit from the :class:`HooksMixin`, and specify the provided
hook types by decorating the class with the :func:`provides_hooks` decorator.

The hook functions are registered by decorating functions with the various hook
decorators such as :func:`on_init`, :func:`on_start`, :func:`on_stop`, etc.

More than one hook can be registered for a given hook type, and classes that inherit from
classes with existing hooks will inherit the hooks from the base classes as well.

The hooks are run by calling the :meth:`HooksMixin.run_hooks` method or retrieved via the
:meth:`HooksMixin.get_hooks` method on the class.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Generic

from pydantic import BaseModel, Field

from aiperf.common.enums import (
    CaseInsensitiveStrEnum,
    LifecycleState,
)
from aiperf.common.types import (
    CommandTypeT,
    HookCallableParamsT,
    HookParamsT,
    HooksMixinT,
    MessageTypeT,
    SelfT,
)


class AIPerfHook(CaseInsensitiveStrEnum):
    BACKGROUND_TASK = "@background_task"
    ON_COMMAND = "@on_command"
    ON_INIT = "@on_init"
    ON_MESSAGE = "@on_message"
    ON_REALTIME_METRICS = "@on_realtime_metrics"
    ON_PROFILING_PROGRESS = "@on_profiling_progress"
    ON_PULL_MESSAGE = "@on_pull_message"
    ON_RECORDS_PROGRESS = "@on_records_progress"
    ON_START = "@on_start"
    ON_STATE_CHANGE = "@on_state_change"
    ON_STOP = "@on_stop"
    ON_REQUEST = "@on_request"
    ON_WARMUP_PROGRESS = "@on_warmup_progress"
    ON_WORKER_STATUS_SUMMARY = "@on_worker_status_summary"
    ON_WORKER_UPDATE = "@on_worker_update"


HookType = AIPerfHook | str
"""Type alias for valid hook types. This is a union of the AIPerfHook enum and any user-defined custom strings."""


class HookAttrs:
    """Constant attribute names for hooks.

    When you decorate a function with a hook decorator, the hook type and parameters are
    set as attributes on the function or class.
    """

    HOOK_TYPE = "__aiperf_hook_type__"
    HOOK_PARAMS = "__aiperf_hook_params__"
    PROVIDES_HOOKS = "__provides_hooks__"


class Hook(BaseModel, Generic[HookParamsT]):
    """A hook is a function that is decorated with a hook type and optional parameters.
    The HookParamsT is the type of the parameters. You can either have a static value,
    or a callable that returns the parameters.
    """

    func: Callable
    params: HookParamsT | Callable[[SelfT], HookParamsT] | None = None  # type: ignore

    @property
    def hook_type(self) -> HookType:
        return getattr(self.func, HookAttrs.HOOK_TYPE)

    @property
    def func_name(self) -> str:
        return self.func.__name__

    @property
    def qualified_name(self) -> str:
        return f"{self.func.__qualname__}"

    def resolve_params(self, self_obj: SelfT) -> HookParamsT | None:
        """Resolve the parameters for the hook. If the parameters are a callable, it will be called
        with the self_obj as the argument, otherwise the parameters are returned as is."""
        if self.params is None:
            return None
        # With variable length parameters, you get a tuple with 1 item in it, so we need to check for that.
        if (
            isinstance(self.params, Iterable)
            and len(self.params) == 1
            and callable(self.params[0])
        ):  # type: ignore
            return self.params[0](self_obj)  # type: ignore
        if callable(self.params):
            return self.params(self_obj)
        return self.params  # type: ignore

    async def __call__(self, **kwargs) -> None:
        if asyncio.iscoroutinefunction(self.func):
            await self.func(**kwargs)
        else:
            await asyncio.to_thread(self.func, **kwargs)

    def __str__(self) -> str:
        return f"{self.hook_type} 🡒 {self.qualified_name}"


class BackgroundTaskParams(BaseModel):
    interval: float | Callable[[Any], float] | None = Field(default=None)
    immediate: bool = Field(default=False)
    stop_on_error: bool = Field(default=False)


def _hook_decorator(hook_type: HookType, func: Callable) -> Callable:
    """Generic decorator to specify that the function should be called during
    a specific hook. See :func:`aiperf.common.hooks._hook_decorator_with_params` for a decorator that
    can also set parameters on the function.

    Args:
        hook_type: The hook type to decorate the function with.
        func: The function to decorate.
    Returns:
        The decorated function.
    """
    setattr(func, HookAttrs.HOOK_TYPE, hook_type)
    return func


def _hook_decorator_with_params(
    hook_type: HookType, params: HookCallableParamsT
) -> Callable[[Callable], Callable]:
    """Generic decorator to specify that the function should be called during
    a specific hook, and with the provided parameters. The parameters are set on
    the function as an attribute, that can later be retrieved via the :meth:`HooksMixin.get_hooks` method.

    Args:
        hook_type: The hook type to decorate the function with.
        params: The parameters to set on the function. Can be any data type, or a callable that returns
            the parameters (for dynamic parameters).
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, HookAttrs.HOOK_TYPE, hook_type)
        setattr(func, HookAttrs.HOOK_PARAMS, params)
        return func

    return decorator


def background_task(
    interval: float | Callable[[SelfT], float] | None = None,
    immediate: bool = True,
    stop_on_error: bool = False,
) -> Callable:
    """
    Decorator to mark a method as a background task with automatic management.

    Tasks are automatically started when the service starts and stopped when the service stops.
    The decorated method will be run periodically in the background when the service is running.

    Args:
        interval: Time between task executions in seconds. If None, the task will run once.
            Can be a callable that returns the interval, and will be called with 'self' as the argument.
        immediate: If True, run the task immediately on start, otherwise wait for the interval first.
        stop_on_error: If True, stop the task on any exception, otherwise log and continue.

    Example:
    ```python
    class MyPlugin(AIPerfLifecycleMixin):
        @background_task(interval=1.0)
        def _background_task(self) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._background_task.__aiperf_hook_type__ = AIPerfHook.BACKGROUND_TASK
    MyPlugin._background_task.__aiperf_hook_params__ = BackgroundTaskParams(
        interval=1.0, immediate=True, stop_on_error=False
    )
    ```
    """
    return _hook_decorator_with_params(
        AIPerfHook.BACKGROUND_TASK,
        BackgroundTaskParams(
            interval=interval, immediate=immediate, stop_on_error=stop_on_error
        ),
    )


def provides_hooks(
    *hook_types: HookType,
) -> Callable[[type[HooksMixinT]], type[HooksMixinT]]:
    """Decorator to specify that the class provides a hook of the given type to all of its subclasses.

    Example:
    ```python
    @provides_hooks(AIPerfHook.ON_MESSAGE)
    class MessageBusClientMixin(CommunicationMixin):
        pass
    ```

    The above is the equivalent to setting:
    ```python
    MessageBusClientMixin.__provides_hooks__ = {AIPerfHook.ON_MESSAGE}
    ```
    """

    def decorator(cls: type[HooksMixinT]) -> type[HooksMixinT]:
        setattr(cls, HookAttrs.PROVIDES_HOOKS, set(hook_types))
        return cls

    return decorator


def on_init(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called during initialization.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(AIPerfLifecycleMixin):
        @on_init
        def _init_plugin(self) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._init_plugin.__aiperf_hook_type__ = AIPerfHook.ON_INIT
    ```
    """
    return _hook_decorator(AIPerfHook.ON_INIT, func)


def on_start(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called during start.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(AIPerfLifecycleMixin):
        @on_start
        def _start_plugin(self) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._start_plugin.__aiperf_hook_type__ = AIPerfHook.ON_START
    ```
    """
    return _hook_decorator(AIPerfHook.ON_START, func)


def on_stop(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called during stop.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(AIPerfLifecycleMixin):
        @on_stop
        def _stop_plugin(self) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._stop_plugin.__aiperf_hook_type__ = AIPerfHook.ON_STOP
    ```
    """
    return _hook_decorator(AIPerfHook.ON_STOP, func)


def on_state_change(
    func: Callable[["HooksMixinT", LifecycleState, LifecycleState], Awaitable],
) -> Callable[["HooksMixinT", LifecycleState, LifecycleState], Awaitable]:
    """Decorator to specify that the function is a hook that should be called during the service state change.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(AIPerfLifecycleMixin):
        @on_state_change
        def _on_state_change(self, old_state: LifecycleState, new_state: LifecycleState) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._on_state_change.__aiperf_hook_type__ = AIPerfHook.ON_STATE_CHANGE
    ```
    """
    return _hook_decorator(AIPerfHook.ON_STATE_CHANGE, func)


def on_message(
    *message_types: MessageTypeT | Callable[[SelfT], Iterable[MessageTypeT]],
) -> Callable:
    """Decorator to specify that the function is a hook that should be called when messages of the
    given type(s) (or topics) are received from the message bus.
    See :func:`aiperf.common.hooks._hook_decorator_with_params`.

    Example:
    ```python
    class MyService(MessageBusClientMixin):
        @on_message(MessageType.STATUS)
        def _on_status_message(self, message: StatusMessage) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyService._on_status_message.__aiperf_hook_type__ = AIPerfHook.ON_MESSAGE
    MyService._on_status_message.__aiperf_hook_params__ = (MessageType.STATUS,)
    ```
    """
    return _hook_decorator_with_params(AIPerfHook.ON_MESSAGE, message_types)


def on_realtime_metrics(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when real-time metrics are received.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(RealtimeMetricsMixin):
        @on_realtime_metrics
        def _on_realtime_metrics(self, metrics: list[MetricResult]) -> None:
            pass
    ```
    """
    return _hook_decorator(AIPerfHook.ON_REALTIME_METRICS, func)


def on_pull_message(
    *message_types: MessageTypeT | Callable[[SelfT], Iterable[MessageTypeT]],
) -> Callable:
    """Decorator to specify that the function is a hook that should be called a pull client
    receives a message of the given type(s).
    See :func:`aiperf.common.hooks._hook_decorator_for_message_types`.

    Example:
    ```python
    class MyService(PullClientMixin, BaseComponentService):
        @on_pull_message(MessageType.CREDIT_DROP)
        def _on_credit_drop_pull(self, message: CreditDropMessage) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyService._on_pull_message.__aiperf_hook_type__ = AIPerfHook.ON_PULL_MESSAGE
    MyService._on_pull_message.__aiperf_hook_params__ = (MessageType.CREDIT_DROP,)
    """
    return _hook_decorator_with_params(AIPerfHook.ON_PULL_MESSAGE, message_types)


def on_profiling_progress(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when a profiling progress update is received.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(ProgressTrackerMixin):
        @on_profiling_progress
        def _on_profiling_progress(self, profiling_stats: RequestsStats) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._on_profiling_progress.__aiperf_hook_type__ = AIPerfHook.ON_PROFILING_PROGRESS
    ```
    """
    return _hook_decorator(AIPerfHook.ON_PROFILING_PROGRESS, func)


def on_records_progress(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when a records progress update is received.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(ProgressTrackerMixin):
        @on_records_progress
        def _on_records_progress(self, progress: RecordsStats) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._on_records_progress.__aiperf_hook_type__ = AIPerfHook.ON_RECORDS_PROGRESS
    ```
    """
    return _hook_decorator(AIPerfHook.ON_RECORDS_PROGRESS, func)


def on_request(
    *message_types: MessageTypeT | Callable[[SelfT], Iterable[MessageTypeT]],
) -> Callable:
    """Decorator to specify that the function is a hook that should be called when requests of the
    given type(s) are received from a ReplyClient.
    See :func:`aiperf.common.hooks._hook_decorator_for_message_types`.

    Example:
    ```python
    class MyService(RequestClientMixin, BaseComponentService):
        @on_request(MessageType.CONVERSATION_REQUEST)
        async def _handle_conversation_request(
            self, message: ConversationRequestMessage
        ) -> ConversationResponseMessage:
            return ConversationResponseMessage(
                ...
            )
    ```

    The above is the equivalent to setting:
    ```python
    MyService._handle_conversation_request.__aiperf_hook_type__ = AIPerfHook.ON_REQUEST
    MyService._handle_conversation_request.__aiperf_hook_params__ = (MessageType.CONVERSATION_REQUEST,)
    ```
    """
    return _hook_decorator_with_params(AIPerfHook.ON_REQUEST, message_types)


def on_command(
    *command_types: CommandTypeT | Callable[[SelfT], Iterable[CommandTypeT]],
) -> Callable:
    """Decorator to specify that the function is a hook that should be called when a CommandMessage with the given
    command type(s) is received from the message bus.
    See :func:`aiperf.common.hooks._hook_decorator_for_message_types`.

    Example:
    ```python
    class MyService(BaseComponentService):
        @on_command(CommandType.PROFILE_START)
        def _on_profile_start(self, message: ProfileStartCommand) -> CommandResponse:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyService._on_profile_start.__aiperf_hook_type__ = AIPerfHook.ON_COMMAND
    MyService._on_profile_start.__aiperf_hook_params__ = (CommandType.PROFILE_START,)
    ```
    """
    return _hook_decorator_with_params(AIPerfHook.ON_COMMAND, command_types)


def on_warmup_progress(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when a warmup progress update is received.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(ProgressTrackerMixin):
        @on_warmup_progress
        def _on_warmup_progress(self, warmup_stats: RequestsStats) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._on_warmup_progress.__aiperf_hook_type__ = AIPerfHook.ON_WARMUP_PROGRESS
    ```
    """
    return _hook_decorator(AIPerfHook.ON_WARMUP_PROGRESS, func)


def on_worker_status_summary(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when a worker status summary is received
    from the WorkerManager.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(WorkerTrackerMixin):
        @on_worker_status_summary
        def _on_worker_status_summary(self, worker_statuses: dict[str, WorkerStatus]) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._on_worker_status_summary.__aiperf_hook_type__ = AIPerfHook.ON_WORKER_STATUS_SUMMARY
    ```
    """
    return _hook_decorator(AIPerfHook.ON_WORKER_STATUS_SUMMARY, func)


def on_worker_update(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when a worker update is received.
    See :func:`aiperf.common.hooks._hook_decorator`.

    Example:
    ```python
    class MyPlugin(WorkerTrackerMixin):
        @on_worker_update
        def _on_worker_update(self, worker_id: str, worker_stats: WorkerStats) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._on_worker_update.__aiperf_hook_type__ = AIPerfHook.ON_WORKER_UPDATE
    ```
    """
    return _hook_decorator(AIPerfHook.ON_WORKER_UPDATE, func)

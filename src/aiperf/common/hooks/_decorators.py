# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public hook decorators (``@on_init``, ``@on_start``, ``@background_task``, ...).

These decorators are the primary user-facing API for registering hooks on
classes that inherit from :class:`HooksMixin`. The low-level helpers live in
:mod:`aiperf.common.hooks._core`.
"""

from collections.abc import Awaitable, Callable, Iterable

from aiperf.common.enums import LifecycleState
from aiperf.common.hooks._core import (
    AIPerfHook,
    BackgroundTaskParams,
    HookAttrs,
    HookType,
    _hook_decorator,
    _hook_decorator_with_params,
)
from aiperf.common.types import (
    CommandTypeT,
    HooksMixinT,
    MessageTypeT,
    SelfT,
)


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

    Example:
    ```python
    class MyPlugin(RealtimeMetricsMixin):
        @on_realtime_metrics
        def _on_realtime_metrics(self, metrics: list[MetricResult]) -> None:
            pass
    ```
    """
    return _hook_decorator(AIPerfHook.ON_REALTIME_METRICS, func)


def on_realtime_telemetry_metrics(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when real-time GPU telemetry metrics are received.

    Example:
    ```python
    class MyPlugin(RealtimeMetricsMixin):
        @on_realtime_telemetry_metrics
        def _on_realtime_telemetry_metrics(self, metrics: list[MetricResult]) -> None:
            pass
    ```
    """
    return _hook_decorator(AIPerfHook.ON_REALTIME_TELEMETRY_METRICS, func)


def on_pull_message(
    *message_types: MessageTypeT | Callable[[SelfT], Iterable[MessageTypeT]],
) -> Callable:
    """Decorator to specify that the function is a hook that should be called a pull client
    receives a message of the given type(s).

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


def on_phase_progress(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when any phase progress update is received.

    Example:
    ```python
    class MyPlugin(ProgressTrackerMixin):
        @on_phase_progress
        def _on_phase_progress(self, phase_stats: CombinedPhaseStats) -> None:
            pass
    ```

    The above is the equivalent to setting:
    ```python
    MyPlugin._on_phase_progress.__aiperf_hook_type__ = AIPerfHook.ON_PHASE_PROGRESS
    ```
    """
    return _hook_decorator(AIPerfHook.ON_PHASE_PROGRESS, func)


def on_records_progress(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when a records progress update is received.

    Example:
    ```python
    class MyPlugin(ProgressTrackerMixin):
        @on_records_progress
        def _on_records_progress(self, progress: PhaseRecordsStats) -> None:
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
    """Decorator to register a handler for commands received on the DEALER/ROUTER control channel.

    Example:
    ```python
    class MyService(BaseComponentService):
        @on_command(CommandType.PROFILE_START)
        async def _on_profile_start(self, message: Command) -> None:
            pass
    ```
    """
    return _hook_decorator_with_params(AIPerfHook.ON_COMMAND, command_types)


def on_worker_status_summary(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called when a worker status summary is received
    from the WorkerManager.

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

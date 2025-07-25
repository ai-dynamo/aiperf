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
import warnings
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from typing_extensions import Self

from aiperf.common.constants import DEFAULT_PULL_CLIENT_MAX_CONCURRENCY
from aiperf.common.enums import (
    CaseInsensitiveStrEnum,
    CommandType,
    LifecycleState,
)
from aiperf.common.types import ClassProtocolT, HooksMixinT, MessageTypeT, ProtocolT

if TYPE_CHECKING:
    pass


class AIPerfHook(CaseInsensitiveStrEnum):
    BACKGROUND_TASK = "@background_task"
    COMMAND_HANDLER = "@command_handler"
    ON_INIT = "@on_init"
    ON_MESSAGE = "@on_message"
    ON_PULL_MESSAGE = "@on_pull_message"
    ON_START = "@on_start"
    ON_STATE_CHANGE = "@on_state_change"
    ON_STOP = "@on_stop"
    REQUEST_HANDLER = "@request_handler"


HookType = AIPerfHook | str
"""Type alias for valid hook types. This is a union of the AIPerfHook enum and any user-defined custom strings."""


class HookAttrs:
    """Constant attribute names for hooks.

    When you decorate a function with a hook decorator, the hook type and parameters are
    set as attributes on the function or class.

    Example:

    @on_message(MessageType.STATUS)
    def on_status(self, message: Message) -> None:
        pass

    # This will look like:
    on_status.__aiperf_hook_type__ = AIPerfHook.ON_MESSAGE
    on_status.__aiperf_hook_params__ = MessageHookParams(message_types={MessageType.STATUS})
    """

    HOOK_TYPE = "__aiperf_hook_type__"
    HOOK_PARAMS = "__aiperf_hook_params__"
    PROVIDES_HOOKS = "__provides_hooks__"
    IMPLEMENTS_PROTOCOL = "__implements_protocol__"


class Hook(BaseModel):
    """A hook is a function that is decorated with a hook type and optional parameters."""

    func: Callable
    params: BaseModel | None = None

    @property
    def hook_type(self) -> HookType:
        return getattr(self.func, HookAttrs.HOOK_TYPE)

    @property
    def func_name(self) -> str:
        return self.func.__name__

    @property
    def qual_name(self) -> str:
        return f"{self.func.__module__}.{self.func_name}"

    async def __call__(self, **kwargs) -> None:
        if asyncio.iscoroutinefunction(self.func):
            await self.func(**kwargs)
        else:
            await asyncio.to_thread(self.func, **kwargs)

    def __str__(self) -> str:
        return f"{self.qual_name} ({self.hook_type})"


class BackgroundTaskParams(BaseModel):
    interval: float | Callable[[Any], float] | None = Field(default=None)
    immediate: bool = Field(default=False)
    stop_on_error: bool = Field(default=False)


class MessageHookParams(BaseModel):
    message_types: set[MessageTypeT]


class CommandHookParams(BaseModel):
    command_types: set[CommandType]


class PullHookParams(BaseModel):
    message_types: set[MessageTypeT]
    max_concurrency: int | None = DEFAULT_PULL_CLIENT_MAX_CONCURRENCY


def hook_decorator(hook_type: HookType, func: Callable) -> Callable:
    """Generic decorator to specify that the function should be called during
    a specific hook. See :func:`aiperf.common.hooks.hook_decorator_with_params` for a decorator that
    can also set parameters on the function.

    Args:
        hook_type: The hook type to decorate the function with.
        func: The function to decorate.
    Returns:
        The decorated function.
    """
    setattr(func, HookAttrs.HOOK_TYPE, hook_type)
    return func


def hook_decorator_with_params(
    hook_type: HookType, params: BaseModel
) -> Callable[[Callable], Callable]:
    """Generic decorator to specify that the function should be called during
    a specific hook, and with the provided parameters. The parameters are set on
    the function as an attribute, that can later be retrieved via the :meth:`HooksMixin.get_hooks` method.

    Args:
        hook_type: The hook type to decorate the function with.
        params: The parameters to set on the function.
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, HookAttrs.HOOK_TYPE, hook_type)
        setattr(func, HookAttrs.HOOK_PARAMS, params)
        return func

    return decorator


def provides_hooks(
    *hook_types: HookType,
) -> Callable[[type[HooksMixinT]], type[HooksMixinT]]:
    """Decorator to specify that the class provides a hook of the given type to all of its subclasses."""

    def decorator(cls: type[HooksMixinT]) -> type[HooksMixinT]:
        setattr(cls, HookAttrs.PROVIDES_HOOKS, set(hook_types))
        return cls

    return decorator


def implements_protocol(protocol: type[ProtocolT]) -> Callable:
    """Decorator to specify that the class implements the given protocol."""

    def decorator(cls: type[ClassProtocolT]) -> type[ClassProtocolT]:
        if TYPE_CHECKING:
            if not hasattr(protocol, "_is_runtime_protocol"):
                warnings.warn(
                    f"Protocol {protocol.__name__} is not a runtime protocol. "
                    "Please use the @runtime_checkable decorator to mark it as a runtime protocol.",
                    category=UserWarning,
                    stacklevel=2,
                )
                raise TypeError(
                    f"Protocol {protocol.__name__} is not a runtime protocol. "
                    "Please use the @runtime_checkable decorator to mark it as a runtime protocol."
                )
            if not issubclass(cls, protocol):
                warnings.warn(
                    f"Class {cls.__name__} does not implement the {protocol.__name__} protocol.",
                    category=UserWarning,
                    stacklevel=2,
                )
                raise TypeError(
                    f"Class {cls.__name__} does not implement the {protocol.__name__} protocol."
                )
        setattr(cls, HookAttrs.IMPLEMENTS_PROTOCOL, protocol)
        return cls

    return decorator


def on_init(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called during initialization.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_INIT, func)


def on_start(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called during start.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_START, func)


def on_stop(func: Callable) -> Callable:
    """Decorator to specify that the function is a hook that should be called during stop.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_STOP, func)


def on_state_change(
    func: Callable[["HooksMixinT", LifecycleState, LifecycleState], Awaitable],
) -> Callable[["HooksMixinT", LifecycleState, LifecycleState], Awaitable]:
    """Decorator to specify that the function is a hook that should be called during the service state change.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_STATE_CHANGE, func)


def background_task(
    interval: float | Callable[["Self"], float] | None = None,
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
    """
    return hook_decorator_with_params(
        AIPerfHook.BACKGROUND_TASK,
        BackgroundTaskParams(
            interval=interval, immediate=immediate, stop_on_error=stop_on_error
        ),
    )


def on_message(
    *message_types: MessageTypeT,
) -> Callable:
    """Decorator to specify that the function is a hook that should be called when messages of the
    given type(s) are received from the message bus.
    See :func:`aiperf.common.hooks.hook_decorator_with_params`."""
    return hook_decorator_with_params(
        AIPerfHook.ON_MESSAGE, MessageHookParams(message_types=set(message_types))
    )


def on_pull_message(
    *message_types: MessageTypeT,
    max_concurrency: int | None = DEFAULT_PULL_CLIENT_MAX_CONCURRENCY,
) -> Callable:
    """Decorator to specify that the function is a hook that should be called a pull client
    receives a message of the given type(s).
    See :func:`aiperf.common.hooks.hook_decorator_with_params`."""
    return hook_decorator_with_params(
        AIPerfHook.ON_PULL_MESSAGE,
        PullHookParams(
            message_types=set(message_types),
            max_concurrency=max_concurrency,
        ),
    )


def request_handler(
    *message_types: MessageTypeT,
) -> Callable:
    """Decorator to specify that the function is a hook that should be called when requests of the
    given type(s) are received from a ReplyClient.
    See :func:`aiperf.common.hooks.hook_decorator_with_params`."""
    return hook_decorator_with_params(
        AIPerfHook.REQUEST_HANDLER, MessageHookParams(message_types=set(message_types))
    )


def command_handler(
    *command_types: CommandType,
) -> Callable:
    """Decorator to specify that the function is a hook that should be called when a CommandMessage with the given
    command type(s) is received from the message bus.
    See :func:`aiperf.common.hooks.hook_decorator_with_params`."""
    return hook_decorator_with_params(
        AIPerfHook.COMMAND_HANDLER, CommandHookParams(command_types=set(command_types))
    )

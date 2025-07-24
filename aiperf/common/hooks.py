# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This module provides an extensive hook system for AIPerf. It is designed to be
used as a mixin for classes that support hooks. It provides a simple interface
for registering and running hooks.

Classes should inherit from the :class:`HooksMixin`, and specify the supported
hook types by decorating the class with the :func:`supports_hooks` decorator.

The hook functions are registered by decorating functions with the various hook
decorators such as :func:`on_init`, :func:`on_start`, :func:`on_stop`, etc.

The hooks are run by calling the :meth:`HooksMixin.run_hooks` or
:meth:`HooksMixin.run_hooks_async` methods on the class.

More than one hook can be registered for a given hook type, and classes that inherit from
classes with existing hooks will inherit the hooks from the base classes as well.
"""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable

from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.exceptions import (
    AIPerfError,
    AIPerfMultiError,
    UnsupportedHookError,
)
from aiperf.common.types import LifecycleMixinT

################################################################################
# Hook Types
################################################################################


class AIPerfHook(CaseInsensitiveStrEnum):
    """Enum for the various AIPerf hooks.

    Note: If you add a new hook, you must also add it to the @supports_hooks
    decorator of the class you wish to use the hook in.
    """

    ON_INIT = "__aiperf_on_init__"
    ON_RUN = "__aiperf_on_run__"
    ON_CONFIGURE = "__aiperf_on_configure__"
    ON_START = "__aiperf_on_start__"
    ON_STOP = "__aiperf_on_stop__"
    ON_CLEANUP = "__aiperf_on_cleanup__"

    ON_SET_STATE = "__aiperf_on_set_state__"


class AIPerfTaskHook(CaseInsensitiveStrEnum):
    """Enum for the various AIPerf task hooks."""

    AIPERF_TASK = "__aiperf_task__"
    AIPERF_AUTO_TASK = "__aiperf_auto_task__"
    AIPERF_AUTO_TASK_INTERVAL = "__aiperf_auto_task_interval__"


HookType = AIPerfHook | AIPerfTaskHook | str
"""Type alias for valid hook types. This is a union of the AIPerfHook enum, the AIPerfTaskHook enum, and any user-defined custom strings."""


AIPERF_HOOK_TYPE = "__aiperf_hook_type__"
"""Constant attribute name that marks a function's hook type."""


################################################################################
# Hook System
################################################################################


class HookSystem:
    """
    System for managing hooks.

    This class is responsible for managing the hooks for a class. It will
    store the hooks in a dictionary, and provide methods to register and run
    the hooks.
    """

    def __init__(self, supported_hooks: set[HookType]):
        """
        Initialize the hook system.

        Args:
            supported_hooks: The hook types that the class supports.
        """
        self.logger = logging.getLogger(__class__.__name__)
        self.supported_hooks: set[HookType] = supported_hooks
        self._hooks: dict[HookType, list[Callable]] = {}

    def register_hook(self, hook_type: HookType, func: Callable):
        """Register a hook function for a given hook type.

        Args:
            hook_type: The hook type to register the function for.
            func: The function to register.
        """
        if hook_type not in self.supported_hooks:
            raise UnsupportedHookError(f"Hook {hook_type} is not supported by class.")

        self._hooks.setdefault(hook_type, []).append(func)

    def get_hooks(self, hook_type: HookType) -> list[Callable]:
        """Get all the registered hooks for the given hook type.

        Args:
            hook_type: The hook type to get the hooks for.

        Returns:
            A list of the hooks for the given hook type.
        """
        return self._hooks.get(hook_type, [])

    async def run_hooks(self, hook_type: HookType, *args, **kwargs):
        """
        Run all the hooks for a given hook type serially. This will wait for each
        hook to complete before running the next one.

        Args:
            hook_type: The hook type to run.
            *args: The arguments to pass to the hooks.
            **kwargs: The keyword arguments to pass to the hooks.
        """
        if hook_type not in self.supported_hooks:
            raise UnsupportedHookError(f"Hook {hook_type} is not supported by class.")

        exceptions: list[Exception] = []
        for func in self.get_hooks(hook_type):
            try:
                if inspect.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                self.logger.exception("Error running hook %s: %s", func.__qualname__, e)
                exceptions.append(
                    AIPerfError(
                        f"Error running hook {func.__qualname__}: {e.__class__.__name__} {e}"
                    )
                )

        if exceptions:
            raise AIPerfMultiError("Errors running hooks", exceptions)

    async def run_hooks_async(self, hook_type: HookType, *args, **kwargs):
        """
        Run all the hooks for a given hook type concurrently. This will run all
        the hooks at the same time and return when all the hooks have completed.

        Args:
            hook_type: The hook type to run.
            *args: The arguments to pass to the hooks.
            **kwargs: The keyword arguments to pass to the hooks.
        """
        if hook_type not in self.supported_hooks:
            raise UnsupportedHookError(f"Hook {hook_type} is not supported by class.")

        coroutines: list[Awaitable] = []
        for func in self.get_hooks(hook_type):
            if inspect.iscoroutinefunction(func):
                coroutines.append(func(*args, **kwargs))
            else:
                coroutines.append(asyncio.to_thread(func, *args, **kwargs))

        if coroutines:
            results = await asyncio.gather(*coroutines, return_exceptions=True)

            exceptions = [result for result in results if isinstance(result, Exception)]
            if exceptions:
                raise AIPerfMultiError("Errors running hooks", exceptions)


################################################################################
# Hook Decorators
################################################################################


def supports_hooks(
    *supported_hook_types: HookType,
) -> Callable[[type], type]:
    """Decorator to indicate that a class supports hooks and sets the
    supported hook types.

    Args:
        supported_hook_types: The hook types that the class supports.

    Returns:
        The decorated class
    """

    def decorator(cls: type) -> type:
        # TODO: We can consider creating a HooksMixinProtocol, but it would still
        #       need to exist somewhere both hooks.py and mixins module can access.
        # Import this here to prevent circular imports. Also make sure you use
        # fully qualified import name to avoid partial loaded module errors.
        from aiperf.common.mixins.hooks_mixin import HooksMixin

        # Ensure the class inherits from HooksMixin
        if not issubclass(cls, HooksMixin):
            raise TypeError(f"Class {cls.__name__} does not inherit from HooksMixin.")

        # Inherit any hooks defined by base classes in the MRO (Method Resolution Order).
        base_hooks = [
            base.supported_hooks
            for base in cls.__mro__[1:]  # Skip this class itself (cls)
            if issubclass(
                base, HooksMixin
            )  # Only include classes that inherit from HooksMixin
        ]

        # Set the supported hooks to be the union of the existing base hooks and the new supported hook types.
        cls.supported_hooks = set.union(*base_hooks, set(supported_hook_types))
        return cls

    return decorator


def hook_decorator(hook_type: HookType, func: Callable) -> Callable:
    """Generic decorator to specify that the function should be called during
    a specific hook.

    Args:
        hook_type: The hook type to decorate the function with.
        func: The function to decorate.
    Returns:
        The decorated function.
    """
    setattr(func, AIPERF_HOOK_TYPE, hook_type)
    return func


def hook_kwargs_decorator(
    hook_type: HookType, **kwargs
) -> Callable[[Callable], Callable]:
    """Generic decorator to specify that the function should be called during
    a specific hook, and with the provided keyword arguments. The keyword
    arguments provided are set on the function as attributes.

    Args:
        hook_type: The hook type to decorate the function with.
        **kwargs: The keyword arguments to set on the function.
    """

    def decorator(func: Callable) -> Callable:
        for key, value in kwargs.items():
            setattr(func, key, value)
        return hook_decorator(hook_type, func)

    return decorator


################################################################################
# Syntactic sugar for the hook decorators.
################################################################################


def on_init(func: Callable) -> Callable:
    """Decorator to specify that the function should be called during initialization.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_INIT, func)


def on_start(func: Callable) -> Callable:
    """Decorator to specify that the function should be called during start.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_START, func)


def on_stop(func: Callable) -> Callable:
    """Decorator to specify that the function should be called during stop.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_STOP, func)


def on_configure(func: Callable) -> Callable:
    """Decorator to specify that the function should be called during the service configuration.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_CONFIGURE, func)


def on_cleanup(func: Callable) -> Callable:
    """Decorator to specify that the function should be called during cleanup.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_CLEANUP, func)


def on_run(func: Callable) -> Callable:
    """Decorator to specify that the function should be called during run.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_RUN, func)


def on_set_state(
    func: Callable,
) -> Callable:
    """Decorator to specify that the function should be called when the service state is set.
    See :func:`aiperf.common.hooks.hook_decorator`."""
    return hook_decorator(AIPerfHook.ON_SET_STATE, func)


def aiperf_task(
    func: Callable,
) -> Callable:
    """Decorator to indicate that the function is a task function. It will be started
    and stopped automatically by the base class lifecycle.
    See :func:`aiperf.common.hooks.hook_decorator`.
    """
    return hook_decorator(AIPerfTaskHook.AIPERF_TASK, func)


def aiperf_auto_task(
    interval_sec: float | Callable[[LifecycleMixinT], float] | None,
) -> Callable[[Callable], Callable]:
    """Decorator to indicate that the function is an auto-managed task function. It will be started
    and stopped automatically by the base class lifecycle, and will run at the specified interval.
    See :func:`aiperf.common.hooks.hook_kwargs_decorator`.

    Args:
        interval_sec: The interval in seconds to sleep between runs. Can be a callable that returns a float.
                    If None, the task will run once and then stop.
    """
    return hook_kwargs_decorator(
        AIPerfTaskHook.AIPERF_AUTO_TASK, interval_sec=interval_sec
    )

#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
from collections.abc import Awaitable, Callable
from enum import Enum

################################################################################
# Hook Types
################################################################################


class AIPerfHook(Enum):
    """Enum for the various AIPerf hooks.

    Note: If you add a new hook, you must also add it to the @supports_hooks
    decorator of the class you wish to use the hook in.
    """

    ON_CLEANUP = "__aiperf_on_cleanup__"
    ON_INIT = "__aiperf_on_init__"
    ON_COMMS_INIT = "__aiperf_on_comms_init__"
    ON_STOP = "__aiperf_on_stop__"
    ON_START = "__aiperf_on_start__"
    ON_CONFIGURE = "__aiperf_on_configure__"
    ON_RUN = "__aiperf_on_run__"
    ON_SET_STATE = "__aiperf_on_set_state__"
    AIPERF_TASK = "__aiperf_task__"


HookType = AIPerfHook | str
"""Type alias for valid hook types. This is a union of the AIPerfHook enum and any user-defined custom strings."""


AIPERF_HOOK_TYPE = "__aiperf_hook_type__"
"""Constant attribute name that marks a function's hook type."""


class UnsupportedHookError(Exception):
    """Exception raised when a hook is defined on a class that does not support it."""


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

        for func in self.get_hooks(hook_type):
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                await result

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
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                coroutines.append(result)

        if coroutines:
            await asyncio.gather(*coroutines)


################################################################################
# Hooks Mixin
################################################################################


class HooksMixin:
    """
    Mixin to add hook support to a class. It abstracts away the details of the
    :class:`HookSystem` and provides a simple interface for registering and running hooks.
    """

    # Class attributes that are set by the :func:`supports_hooks` decorator
    supported_hooks: set[HookType] = set()

    def __init__(self):
        """
        Initialize the hook system and register all functions that are decorated with a hook decorator.
        """
        # Initialize the hook system
        self._hook_system = HookSystem(self.supported_hooks)

        # Register all functions that are decorated with a hook decorator
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, AIPERF_HOOK_TYPE):
                # Get the hook type from the function
                hook_type = getattr(attr, AIPERF_HOOK_TYPE)
                # Register the function with the hook type
                self._hook_system.register_hook(hook_type, attr)

    async def run_hooks(self, hook_type: HookType, *args, **kwargs):
        """Run all the hooks serially. See :meth:`HookSystem.run_hooks`."""
        await self._hook_system.run_hooks(hook_type, *args, **kwargs)

    async def run_hooks_async(self, hook_type: HookType, *args, **kwargs):
        """Run all the hooks concurrently. See :meth:`HookSystem.run_hooks_async`."""
        await self._hook_system.run_hooks_async(hook_type, *args, **kwargs)

    def get_hooks(self, hook_type: HookType) -> list[Callable]:
        """Get all the registered hooks for the given hook type. See :meth:`HookSystem.get_hooks`."""
        return self._hook_system.get_hooks(hook_type)


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


def hook_decorator(hook_type: HookType) -> Callable[[Callable], Callable]:
    """Generic decorator to specify that the function should be called during
    a specific hook.

    Args:
        hook_type: The hook type to decorate the function with.

    Returns:
        The decorated function.
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, AIPERF_HOOK_TYPE, hook_type)
        return func

    return decorator


################################################################################
# Syntactic sugar for the hook decorators.
################################################################################

on_init = hook_decorator(AIPerfHook.ON_INIT)
"""Decorator to specify that the function should be called during initialization.
See :func:`aiperf.common.hooks.hook_decorator`."""

on_start = hook_decorator(AIPerfHook.ON_START)
"""Decorator to specify that the function should be called during start.
See :func:`aiperf.common.hooks.hook_decorator`."""

on_stop = hook_decorator(AIPerfHook.ON_STOP)
"""Decorator to specify that the function should be called during stop.
See :func:`aiperf.common.hooks.hook_decorator`."""

on_configure = hook_decorator(AIPerfHook.ON_CONFIGURE)
"""Decorator to specify that the function should be called during the service configuration.
See :func:`aiperf.common.hooks.hook_decorator`."""

on_comms_init = hook_decorator(AIPerfHook.ON_COMMS_INIT)
"""Decorator to specify that the function should be called during communication initialization.
See :func:`aiperf.common.hooks.hook_decorator`."""

on_cleanup = hook_decorator(AIPerfHook.ON_CLEANUP)
"""Decorator to specify that the function should be called during cleanup.
See :func:`aiperf.common.hooks.hook_decorator`."""

on_run = hook_decorator(AIPerfHook.ON_RUN)
"""Decorator to specify that the function should be called during run.
See :func:`aiperf.common.hooks.hook_decorator`."""

on_set_state = hook_decorator(AIPerfHook.ON_SET_STATE)
"""Decorator to specify that the function should be called when the service state is set.
See :func:`aiperf.common.hooks.hook_decorator`."""

aiperf_task = hook_decorator(AIPerfHook.AIPERF_TASK)
"""Decorator to indicate that the function is a task function. It will be started
and stopped automatically by the base class lifecycle.
See :func:`aiperf.common.hooks.hook_decorator`."""

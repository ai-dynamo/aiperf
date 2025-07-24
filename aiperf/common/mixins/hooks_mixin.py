# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

from aiperf.common import aiperf_logger
from aiperf.common.exceptions import AIPerfMultiError, UnsupportedHookError
from aiperf.common.hooks import (
    AIPERF_HOOK_PARAMS,
    AIPERF_HOOK_TYPE,
    PROVIDES_HOOKS,
    Hook,
    HookType,
)
from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin


class HooksMixin(AIPerfLoggerMixin):
    """Mixin for a class to be able to provide hooks to its subclasses, and to be able to run them. A "hook" is a function
    that is decorated with a hook type (AIPerfHook), and optional parameters.

    In order to provide hooks, a class MUST use the `@provides_hooks` decorator to declare the hook types it provides.
    Only list hook types that you call `get_hooks` or `run_hooks` on, to get or run the functions that are decorated
    with those hook types.

    Provided hooks are recursively inherited by subclasses, so if a base class provides a hook,
    all subclasses will also provide that hook (without having to explicitly declare it, or call `get_hooks` or `run_hooks`).
    In fact, you typically should not get or run hooks from the base class, as this may lead to calling hooks twice.

    Hooks are registered in the order they are defined within the same class from top to bottom, and each class's hooks
    are inspected starting with hooks defined in the lowest level of base classes, moving up to the highest subclass.

    IMPORTANT:
    - Hook decorated methods from one class can be named the same as methods in their base classes, and BOTH will be registered.
    Meaning if class A and class B both have a method named `_initialize`, which is decorated with `@on_init`, and class B inherits from class A,
    then both `_initialize` methods will be registered as hooks, and will be run in the order A._initialize, then B._initialize.
    This is done without requiring the user to call `super()._initialize` in the subclass, as the base class hook will be run automatically.
    However, the caveat is that it is not possible to disable the hook from the base class without extra work, and if the user does accidentally
    call `super()._initialize` in the subclass, the base class hook may be called twice.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._provided_hook_types: set[HookType] = set()

        self._hooks: dict[HookType, list[Hook]] = {}
        # Go through the MRO in reverse order to ensure that the hooks are
        # registered in the correct order (base classes first, then subclasses).
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, PROVIDES_HOOKS):
                # As we find base classes that provide hooks, we add them to the
                # set of provided hook types, which is used for validation.
                self._provided_hook_types.update(getattr(cls, PROVIDES_HOOKS))

            # Go through the class's methods to find the hooks.
            for method in cls.__dict__.values():
                if not callable(method):
                    continue

                # If the method has the AIPERF_HOOK_TYPE attribute, it is a hook.
                if hasattr(method, AIPERF_HOOK_TYPE):
                    method_hook_type = getattr(method, AIPERF_HOOK_TYPE)
                    # If the hook type is not provided by any base class, it is an error.
                    # This is to ensure that the hook is only registered if it is provided by a base class.
                    # This is to avoid the case where a developer accidentally uses a hook that is not provided by a base class.
                    if method_hook_type not in self._provided_hook_types:
                        raise UnsupportedHookError(
                            f"Hook {method_hook_type} is not provided by any base class of {self.__class__.__name__}. "
                            f"(Provided Hooks: {[f'{hook_type}' for hook_type in self._provided_hook_types]})"
                        )

                    # Bind the method to the instance ("self"), extract the hook parameters,
                    # and add it to the hooks dictionary.
                    bound_method = method.__get__(self)
                    self._hooks.setdefault(method_hook_type, []).append(
                        Hook(
                            func=bound_method,
                            params=getattr(method, AIPERF_HOOK_PARAMS, None),
                        ),
                    )

        self.debug(
            lambda: f"Provided hook types: {self._provided_hook_types} for {self.__class__.__name__}"
        )

    def get_hooks(self, *hook_types: HookType, reversed: bool = False) -> list[Hook]:
        """Get the hooks that are defined by the class for the given hook type(s), optionally reversed.
        This will return a list of Hook objects that can be inspected for their type and parameters,
        and optionally called."""
        hooks = [
            hook
            for hook_type, hooks in self._hooks.items()
            if not hook_types or hook_type in hook_types
            for hook in hooks
        ]
        if reversed:
            hooks.reverse()
        return hooks

    async def run_hooks(
        self, *hook_types: HookType, reversed: bool = False, **kwargs
    ) -> None:
        """Run the hooks for the given hook type, waiting for each hook to complete before running the next one.
        Hooks are run in the order they are defined by the class, starting with hooks defined in the lowest level
        of base classes, moving up to the top level class. If more than one hook type is provided, the hooks
        from each level of classes will be run in the order of the hook types provided.

        If reversed is True, the hooks will be run in reverse order. This is useful for stop/cleanup hooks, where you
        want to start with the children and ending with the parent.

        The kwargs are passed through as keyword arguments to each hook.
        """
        exceptions: list[Exception] = []
        for hook in self.get_hooks(*hook_types, reversed=reversed):
            self.debug(lambda hook=hook: f"Running hook: {hook!r}")
            try:
                await hook(**kwargs)
            except Exception as e:
                exceptions.append(e)
                self.exception(
                    f"Error running {hook!r} hook for {self.__class__.__name__}: {e}"
                )
        if exceptions:
            raise AIPerfMultiError(
                f"Errors running {hook_types} hooks for {self.__class__.__name__}",
                exceptions,
            )


# Add this file as one to be ignored when finding the caller of aiperf_logger.
# This helps to make it more transparent where the actual function is being called from.
_srcfile = os.path.normcase(HooksMixin.get_hooks.__code__.co_filename)
aiperf_logger._ignored_files.append(_srcfile)

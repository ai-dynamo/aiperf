# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Callable, Iterable
from typing import Any

from aiperf.common import aiperf_logger
from aiperf.common.exceptions import AIPerfMultiError, HookError, UnsupportedHookError
from aiperf.common.hooks import Hook, HookAttrs, HookType
from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.common.types import AnyT, HookParamsT, SelfT


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
            if hasattr(cls, HookAttrs.PROVIDES_HOOKS):
                # As we find base classes that provide hooks, we add them to the
                # set of provided hook types, which is used for validation.
                self._provided_hook_types.update(getattr(cls, HookAttrs.PROVIDES_HOOKS))

            # Go through the class's methods to find the hooks.
            for method in cls.__dict__.values():
                if not callable(method):
                    continue

                # If the method has the AIPERF_HOOK_TYPE attribute, it is a hook.
                if hasattr(method, HookAttrs.HOOK_TYPE):
                    method_hook_type = getattr(method, HookAttrs.HOOK_TYPE)
                    self._check_hook_type_is_provided(method_hook_type)

                    # Bind the method to the instance ("self"), extract the hook parameters,
                    # and add it to the hooks dictionary.
                    bound_method = method.__get__(self)
                    self._hooks.setdefault(method_hook_type, []).append(
                        Hook(
                            func=bound_method,
                            params=getattr(method, HookAttrs.HOOK_PARAMS, None),
                        ),
                    )

        self.trace(
            lambda: f"Provided hook types: {self._provided_hook_types} for {self.__class__.__name__}"
        )

    def _check_hook_type_is_provided(self, hook_type: HookType) -> None:
        """Check if the hook type is provided by any base class of the class.

        Args:
            hook_type: The hook type to check.

        Raises:
            UnsupportedHookError: If the hook type is not provided by any base class of the class.
        """
        # If the hook type is not provided by any base class, it is an error.
        # This is to ensure that the hook is only registered if it is provided by a base class.
        # This is to avoid the case where a developer accidentally uses a hook that is not provided by a base class.
        if hook_type not in self._provided_hook_types:
            raise UnsupportedHookError(
                f"Hook {hook_type} is not provided by any base class of {self.__class__.__name__}. "
                f"(Provided Hooks: {[f'{hook_type}' for hook_type in self._provided_hook_types]})"
            )

    def attach_hook(
        self,
        hook_type: HookType,
        func: Callable,
        params: HookParamsT | Callable[[SelfT], HookParamsT] | None = None,
    ) -> None:
        """Attach a hook to this class. This is useful for attaching hooks to a class directly,
        without having to inherit from this class, or use a decorator.

        Args:
            hook_type: The hook type to attach the hook to.
            func: The function to attach the hook to.
            params: The parameters to attach to the hook.
        """
        if not callable(func):
            raise ValueError(f"Invalid hook function: {func}")

        self._check_hook_type_is_provided(hook_type)
        self._hooks.setdefault(hook_type, []).append(Hook(func=func, params=params))

    def get_hooks(self, *hook_types: HookType, reverse: bool = False) -> list[Hook]:
        """Get the hooks that are defined by the class for the given hook type(s), optionally reversed.
        This will return a list of Hook objects that can be inspected for their type and parameters,
        and optionally called."""
        hooks = [
            hook
            for hook_type, hooks in self._hooks.items()
            if not hook_types or hook_type in hook_types
            for hook in hooks
        ]
        if reverse:
            hooks.reverse()
        return hooks

    def for_each_hook_param(
        self,
        *hook_types: HookType,
        self_obj: Any,
        param_type: AnyT,
        lambda_func: Callable[[Hook, AnyT], None],
        reverse: bool = False,
    ) -> None:
        """Iterate over the hooks for the given hook type(s), optionally reversed.
        If a lambda_func is provided, it will be called for each parameter of the hook,
        and the hook and parameter will be passed as arguments.

        Args:
            hook_types: The hook types to iterate over.
            self_obj: The object to pass to the lambda_func.
            param_type: The type of the parameter to pass to the lambda_func (for validation).
            lambda_func: The function to call for each hook.
            reverse: Whether to iterate over the hooks in reverse order.
        """
        for hook in self.get_hooks(*hook_types, reverse=reverse):
            # in case the hook params are a callable, we need to resolve them to get the actual params
            params = hook.resolve_params(self_obj)
            if not isinstance(params, Iterable):
                raise ValueError(
                    f"Invalid hook params: {params}. Expected Iterable but got {type(params)}"
                )
            for param in params:
                self.trace(
                    lambda param=param,
                    type=param_type: f"param: {param}, param_type: {type}"
                )
                if not isinstance(param, param_type):
                    raise ValueError(
                        f"Invalid hook param: {param}. Expected {param_type} but got {type(param)}"
                    )
                # Call the lambda_func for each parameter of each hook.
                lambda_func(hook, param)

    async def run_hooks(
        self, *hook_types: HookType, reverse: bool = False, **kwargs
    ) -> None:
        """Run the hooks for the given hook type, waiting for each hook to complete before running the next one.
        Hooks are run in the order they are defined by the class, starting with hooks defined in the lowest level
        of base classes, moving up to the top level class. If more than one hook type is provided, the hooks
        from each level of classes will be run in the order of the hook types provided.

        If reverse is True, the hooks will be run in reverse order. This is useful for stop/cleanup hooks, where you
        want to start with the children and ending with the parent.

        The kwargs are passed through as keyword arguments to each hook.
        """
        exceptions: list[Exception] = []
        for hook in self.get_hooks(*hook_types, reverse=reverse):
            if self.is_trace_enabled:
                self.trace(f"Running hook: {hook!r}")
            try:
                await hook(**kwargs)
            except Exception as e:
                exceptions.append(HookError(self.__class__.__name__, hook.func_name, e))
                self.exception(
                    f"Error running {hook!r} hook for {self.__class__.__name__}: {e}"
                )
        if exceptions:
            raise AIPerfMultiError(None, exceptions)


# Add this file as one to be ignored when finding the caller of aiperf_logger.
# This helps to make it more transparent where the actual function is being called from.
_srcfile = os.path.normcase(HooksMixin.get_hooks.__code__.co_filename)
aiperf_logger._ignored_files.append(_srcfile)

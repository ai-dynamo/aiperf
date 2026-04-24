# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core hook primitives: the AIPerfHook enum, HookAttrs, the Hook model,
BackgroundTaskParams, and the low-level decorator builders used by the
public decorators in :mod:`aiperf.common.hooks._decorators`.
"""

import asyncio
from collections.abc import Callable, Iterable
from typing import Any, Generic

from pydantic import BaseModel, Field

from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.types import (
    HookCallableParamsT,
    HookParamsT,
    SelfT,
)


class AIPerfHook(CaseInsensitiveStrEnum):
    BACKGROUND_TASK = "@background_task"
    ON_COMMAND = "@on_command"
    ON_INIT = "@on_init"
    ON_MESSAGE = "@on_message"
    ON_REALTIME_METRICS = "@on_realtime_metrics"
    ON_REALTIME_TELEMETRY_METRICS = "@on_realtime_telemetry_metrics"
    ON_PHASE_PROGRESS = "@on_phase_progress"
    ON_PULL_MESSAGE = "@on_pull_message"
    ON_RECORDS_PROGRESS = "@on_records_progress"
    ON_START = "@on_start"
    ON_STATE_CHANGE = "@on_state_change"
    ON_STOP = "@on_stop"
    ON_REQUEST = "@on_request"
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
    interval: float | Callable[[Any], float] | None = Field(
        default=None,
        description="Seconds between executions, callable returning interval, or None for one-shot.",
    )
    immediate: bool = Field(
        default=False,
        description="Run immediately on start instead of waiting for first interval.",
    )
    stop_on_error: bool = Field(
        default=False,
        description="Stop the background task on any unhandled exception.",
    )


def _hook_decorator(hook_type: HookType, func: Callable) -> Callable:
    """Generic decorator to specify that the function should be called during
    a specific hook. See :func:`_hook_decorator_with_params` for a decorator that
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

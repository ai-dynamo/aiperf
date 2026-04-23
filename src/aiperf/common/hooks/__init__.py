# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiperf.common.hooks._core import (
    AIPerfHook,
    BackgroundTaskParams,
    Hook,
    HookAttrs,
    HookType,
    _hook_decorator,
    _hook_decorator_with_params,
)
from aiperf.common.hooks._decorators import (
    background_task,
    on_command,
    on_init,
    on_message,
    on_phase_progress,
    on_pull_message,
    on_realtime_metrics,
    on_realtime_telemetry_metrics,
    on_records_progress,
    on_request,
    on_start,
    on_state_change,
    on_stop,
    on_worker_status_summary,
    on_worker_update,
    provides_hooks,
)

__all__ = [
    "AIPerfHook",
    "BackgroundTaskParams",
    "Hook",
    "HookAttrs",
    "HookType",
    "_hook_decorator",
    "_hook_decorator_with_params",
    "background_task",
    "on_command",
    "on_init",
    "on_message",
    "on_phase_progress",
    "on_pull_message",
    "on_realtime_metrics",
    "on_realtime_telemetry_metrics",
    "on_records_progress",
    "on_request",
    "on_start",
    "on_state_change",
    "on_stop",
    "on_worker_status_summary",
    "on_worker_update",
    "provides_hooks",
]

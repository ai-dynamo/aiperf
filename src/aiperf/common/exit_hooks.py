# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Work that must happen after a run finishes but before the process is killed.

``SystemController._stop_system_controller`` ends with ``os._exit``, which
bypasses every normal teardown path: ``finally`` blocks, ``atexit`` handlers and
any code after the call that started the controller. That is deliberate --
leftover ZMQ contexts and daemon threads can otherwise keep the interpreter
alive forever -- but it also means a single-run caller never regains control to
do post-run work, so anything it registered simply never ran.

Hooks registered here are drained immediately before that ``os._exit``. Draining
(rather than iterating) makes them run exactly once, so a caller that *does*
regain control -- component-integration tests mock ``os._exit`` to a no-op --
can call the drain again safely and get a no-op instead of duplicate work.
"""

from __future__ import annotations

from collections.abc import Callable

# Takes the exit code the process is about to use, returns the code to use
# instead. Returning the input unchanged means "no opinion".
PreExitHook = Callable[[int], int]


class _PreExitHookRegistry:
    """Process-wide registry of post-run work, drained exactly once."""

    def __init__(self) -> None:
        self._hooks: list[PreExitHook] = []

    def register(self, hook: PreExitHook) -> None:
        self._hooks.append(hook)

    def clear(self) -> None:
        self._hooks.clear()

    def drain(self) -> list[PreExitHook]:
        hooks = list(self._hooks)
        self._hooks.clear()
        return hooks


_REGISTRY = _PreExitHookRegistry()


def register_pre_exit_hook(hook: PreExitHook) -> None:
    """Register work to run immediately before the process is hard-exited."""
    _REGISTRY.register(hook)


def clear_pre_exit_hooks() -> None:
    """Drop all registered hooks without running them."""
    _REGISTRY.clear()


def run_pre_exit_hooks(exit_code: int) -> int:
    """Drain and run every registered hook, returning the final exit code.

    Exceptions propagate. Callbacks already decide their own failure policy --
    ``AIPERF_RAISE_ON_CALLBACK_ERROR`` makes a failing callback raise rather
    than be swallowed -- and catching here would convert that contract into a
    bare exit code. The one caller that cannot afford to propagate is the
    controller, which must still reach ``os._exit``; it catches explicitly.
    """
    for hook in _REGISTRY.drain():
        exit_code = hook(exit_code)
    return exit_code

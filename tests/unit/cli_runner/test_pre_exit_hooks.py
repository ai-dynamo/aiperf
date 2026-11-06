# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-run callbacks must survive the controller's hard exit.

``SystemController._stop_system_controller`` ends in ``os._exit``, which skips
everything after the call that started it. Before pre-exit hooks existed, that
made every ``on_complete`` callback on the single-run path unreachable:
``--auto-plot`` produced no plots, emitted no warning, and exited 0, and
``--plot-required`` could never fail a run (AIP-1975).
"""

from __future__ import annotations

import pytest

from aiperf.common.exit_hooks import (
    clear_pre_exit_hooks,
    register_pre_exit_hook,
    run_pre_exit_hooks,
)


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_pre_exit_hooks()
    yield
    clear_pre_exit_hooks()


def test_registered_hook_runs_and_sees_the_exit_code() -> None:
    seen: list[int] = []
    register_pre_exit_hook(lambda code: seen.append(code) or code)
    assert run_pre_exit_hooks(0) == 0
    assert seen == [0]


def test_hooks_run_only_once_even_if_drained_again() -> None:
    """The controller drains, then the caller may drain again if it survives."""
    calls: list[int] = []
    register_pre_exit_hook(lambda code: calls.append(code) or code)

    run_pre_exit_hooks(0)
    run_pre_exit_hooks(0)

    assert calls == [0], "a surviving caller must not re-run post-run work"


def test_a_hook_can_fail_the_run() -> None:
    """`--plot-required` depends on this: strict mode must change the exit code."""
    register_pre_exit_hook(lambda code: 3)
    assert run_pre_exit_hooks(0) == 3


def test_hooks_are_skipped_by_the_caller_when_the_run_already_failed() -> None:
    """The single-run path registers a hook that no-ops on a non-zero code."""
    ran: list[int] = []

    def only_on_success(code: int) -> int:
        if code != 0:
            return code
        ran.append(code)
        return code

    register_pre_exit_hook(only_on_success)
    assert run_pre_exit_hooks(1) == 1
    assert ran == []


def test_a_raising_hook_propagates() -> None:
    """Strict mode depends on this.

    ``AIPERF_RAISE_ON_CALLBACK_ERROR`` makes a failing callback raise instead
    of being swallowed. Catching inside the drain would silently downgrade that
    contract to a bare exit code, so the drain must not catch; the controller,
    which cannot propagate past ``os._exit``, catches explicitly instead.
    """

    def boom(code: int) -> int:
        raise RuntimeError("callback exploded")

    register_pre_exit_hook(boom)
    with pytest.raises(RuntimeError, match="callback exploded"):
        run_pre_exit_hooks(0)


def test_a_raising_hook_still_drains_so_it_cannot_run_twice() -> None:
    calls: list[int] = []

    def boom(code: int) -> int:
        calls.append(code)
        raise RuntimeError("callback exploded")

    register_pre_exit_hook(boom)
    with pytest.raises(RuntimeError):
        run_pre_exit_hooks(0)
    run_pre_exit_hooks(0)

    assert calls == [0]

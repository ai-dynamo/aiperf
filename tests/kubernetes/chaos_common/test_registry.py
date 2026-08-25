# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`InjectorRegistry` + :py:class:`EchoInjector`.

These tests exercise every code path of the registry contract (dispatch,
LIFO compose, restore-on-exception, idempotent restore, prefix matching,
precondition propagation) without touching a Kubernetes cluster. The
``faults`` fixture from ``chaos_common/conftest.py`` constructs a fresh
registry pre-loaded with :py:class:`EchoInjector`.
"""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos_common.base import FaultPreconditionError
from tests.kubernetes.chaos_common.injectors.echo import EchoInjector
from tests.kubernetes.chaos_common.registry import InjectorRegistry


async def test_inject_single_fault_runs_inject_and_restore() -> None:
    log: list[str] = []
    registry = InjectorRegistry()
    registry.register(EchoInjector(order_log=log))

    async with registry.inject("echo.simple") as applied:
        assert log == ["inject:echo.simple"]
        assert applied.spec.fault_id == "echo.simple"

    assert log == ["inject:echo.simple", "restore:echo.simple"]
    assert applied.metadata["restored"] is True


async def test_compose_orders_restore_lifo() -> None:
    log: list[str] = []
    registry = InjectorRegistry()
    registry.register(EchoInjector(order_log=log))

    async with registry.compose(
        ("echo.one", {"token": "one"}),
        ("echo.two", {"token": "two"}),
        ("echo.three", {"token": "three"}),
    ) as handles:
        assert [h.metadata["token"] for h in handles] == ["one", "two", "three"]
        assert log == ["inject:one", "inject:two", "inject:three"]

    assert log == [
        "inject:one",
        "inject:two",
        "inject:three",
        "restore:three",
        "restore:two",
        "restore:one",
    ]


async def test_resolve_raises_lookup_error_on_unknown_fault_id() -> None:
    registry = InjectorRegistry()
    registry.register(EchoInjector())

    with pytest.raises(LookupError) as excinfo:
        registry.resolve("nope.unknown")

    msg = str(excinfo.value)
    assert "nope.unknown" in msg
    assert "EchoInjector" in msg
    assert "('echo',)" in msg


async def test_restore_runs_even_on_exception() -> None:
    log: list[str] = []
    registry = InjectorRegistry()
    registry.register(EchoInjector(order_log=log))

    with pytest.raises(RuntimeError, match="boom"):
        async with registry.inject("echo.crash") as applied:
            assert log == ["inject:echo.crash"]
            raise RuntimeError("boom")

    assert log == ["inject:echo.crash", "restore:echo.crash"]
    assert applied.metadata["restored"] is True


async def test_restore_is_idempotent() -> None:
    """Calling ``__aexit__`` twice on the same handle restores only once."""
    log: list[str] = []
    injector = EchoInjector(order_log=log)
    spec_handle = await injector.inject(
        # Build a spec directly so we can drive __aexit__ manually.
        # The registry would normally guard against double-exit.
        _make_spec("echo.manual"),
    )
    # First exit -> restore runs.
    await spec_handle.__aexit__(None, None, None)
    # Second exit -> guarded by _restored, no-op.
    await spec_handle.__aexit__(None, None, None)

    assert log == ["inject:echo.manual", "restore:echo.manual"]
    assert spec_handle.metadata["restored"] is True


def _make_spec(fault_id: str):
    """Local helper to avoid leaking FaultSpec into the public test surface."""
    from tests.kubernetes.chaos_common.base import FaultSpec

    return FaultSpec(fault_id=fault_id)


def test_handles_prefix_match() -> None:
    """`HANDLES = ('echo',)` matches `echo`, `echo.simple`; not `echos`/`foo.echo`."""
    assert EchoInjector.handles("echo") is True
    assert EchoInjector.handles("echo.simple") is True
    assert EchoInjector.handles("echo.nested.deeper") is True
    assert EchoInjector.handles("echos") is False
    assert EchoInjector.handles("foo.echo") is False
    assert EchoInjector.handles("") is False


async def test_fault_precondition_error_propagates() -> None:
    """The registry must not swallow `FaultPreconditionError`."""
    registry = InjectorRegistry()
    registry.register(EchoInjector(raise_precondition_for="echo.bad"))

    with pytest.raises(FaultPreconditionError, match="fail-fast"):
        async with registry.inject("echo.bad"):
            pytest.fail("body should not run when inject() raises")


async def test_compose_restores_completed_handles_when_later_inject_raises() -> None:
    """If a mid-compose inject raises, already-applied faults still restore."""
    log: list[str] = []
    registry = InjectorRegistry()
    registry.register(EchoInjector(order_log=log, raise_precondition_for="echo.boom"))

    with pytest.raises(FaultPreconditionError):
        async with registry.compose(
            ("echo.ok", {"token": "ok"}),
            ("echo.boom", {"token": "boom"}),
        ):
            pytest.fail("body should not run when compose mid-stage raises")

    # echo.ok injected and then restored on stack unwind; echo.boom never
    # got an inject entry (it raised), so no restore for it.
    assert log == ["inject:ok", "restore:ok"]


async def test_registry_inject_idempotent_when_caller_double_wraps() -> None:
    """Wrapping the yielded handle in a second `async with` still restores once.

    A defensive caller may do::

        async with registry.inject("echo.x") as applied:
            async with applied:
                ...

    The inner `async with` triggers `AppliedFault.__aexit__` first; the
    outer `registry.inject` finally-block then triggers it again. The
    guard in `AppliedFault.__aexit__` must ensure `restore()` runs exactly
    once across both exits.
    """
    log: list[str] = []
    registry = InjectorRegistry()
    registry.register(EchoInjector(order_log=log))

    async with registry.inject("echo.double") as applied:
        async with applied:
            assert log == ["inject:echo.double"]

        # Inner block exited -> restore ran exactly once.
        assert log == ["inject:echo.double", "restore:echo.double"]
        assert applied.metadata["restored"] is True

    # Outer block exit must not invoke restore a second time.
    assert log == ["inject:echo.double", "restore:echo.double"]

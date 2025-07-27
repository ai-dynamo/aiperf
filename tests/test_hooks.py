# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

import pytest

from aiperf.common.exceptions import UnsupportedHookError
from aiperf.common.hooks import (
    AIPerfHook,
    on_init,
    on_start,
    on_stop,
    provides_hooks,
)
from aiperf.common.mixins import HooksMixin


@provides_hooks(AIPerfHook.ON_INIT, AIPerfHook.ON_STOP)
class MockHookProvider(HooksMixin):
    def __init__(self):
        super().__init__()
        self.called_hooks = set()

    def add_called_hook(self, hook_name: Callable | str):
        self.called_hooks.add(hook_name)
        print(f"Hook called: {self.__class__.__name__}.{hook_name}")

    async def initialize(self) -> None:
        await self.run_hooks(AIPerfHook.ON_INIT)

    async def stop(self) -> None:
        await self.run_hooks(AIPerfHook.ON_STOP)


class MockHooks(MockHookProvider):
    @on_init
    async def on_init_3(self):
        self.add_called_hook(self.on_init_3)

    @on_init
    async def on_init_2(self):
        self.add_called_hook(self.on_init_2)

    @on_init
    async def on_init_1(self):
        self.add_called_hook(self.on_init_1)

    @on_stop
    async def on_stop_1(self):
        self.add_called_hook(self.on_stop_1)


@provides_hooks(AIPerfHook.ON_START)
class MockHooksInheritance(MockHooks):
    @on_init
    async def on_init_4(self):
        self.add_called_hook(self.on_init_4)

    @on_stop
    async def on_stop_2(self):
        self.add_called_hook(self.on_stop_2)

    async def start(self):
        await self.run_hooks(AIPerfHook.ON_START)

    @on_start
    async def on_start_1(self):
        self.add_called_hook(self.on_start_1)


def test_hook_decorators():
    """Test the hook decorators."""
    test_hooks = MockHooks()

    assert [hook.func for hook in test_hooks.get_hooks(AIPerfHook.ON_INIT)] == [
        test_hooks.on_init_3,
        test_hooks.on_init_2,
        test_hooks.on_init_1,
    ], "Init hooks should be registered in the order they are defined"

    assert [hook.func for hook in test_hooks.get_hooks(AIPerfHook.ON_STOP)] == [
        test_hooks.on_stop_1,
    ], "Stop hooks should be registered"


def test_hook_inheritance():
    """Test the hook inheritance."""
    test_hooks_inheritance = MockHooksInheritance()

    assert [
        hook.func for hook in test_hooks_inheritance.get_hooks(AIPerfHook.ON_INIT)
    ] == [
        test_hooks_inheritance.on_init_3,
        test_hooks_inheritance.on_init_2,
        test_hooks_inheritance.on_init_1,
        test_hooks_inheritance.on_init_4,
    ], "Init hooks should be registered in the order they are defined"
    assert [
        hook.func for hook in test_hooks_inheritance.get_hooks(AIPerfHook.ON_STOP)
    ] == [
        test_hooks_inheritance.on_stop_1,
        test_hooks_inheritance.on_stop_2,
    ], "Stop hooks should be registered in the order they are defined"
    assert [
        hook.func for hook in test_hooks_inheritance.get_hooks(AIPerfHook.ON_START)
    ] == [test_hooks_inheritance.on_start_1], "Start hook should be registered"


@pytest.mark.asyncio
async def test_run_hooks_init():
    test_hooks = MockHooks()

    await test_hooks.initialize()

    assert test_hooks.on_init_1 in test_hooks.called_hooks, (
        "Init hook 1 should be called"
    )
    assert test_hooks.on_init_2 in test_hooks.called_hooks, (
        "Init hook 2 should be called"
    )
    assert test_hooks.on_init_3 in test_hooks.called_hooks, (
        "Init hook 3 should be called"
    )


@pytest.mark.asyncio
async def test_run_stop_hooks():
    test_hooks = MockHooksInheritance()

    await test_hooks.stop()

    assert test_hooks.on_stop_1 in test_hooks.called_hooks, (
        "Stop hook 1 should be called"
    )
    assert test_hooks.on_stop_2 in test_hooks.called_hooks, (
        "Stop hook 2 should be called"
    )
    assert test_hooks.on_init_1 not in test_hooks.called_hooks, (
        "Init hook 1 should not be called"
    )


@pytest.mark.asyncio
async def test_inherited_run_hooks_start():
    test_hooks = MockHooksInheritance()

    await test_hooks.start()

    assert test_hooks.on_start_1 in test_hooks.called_hooks, (
        "Start hook should be called"
    )
    assert test_hooks.on_init_1 not in test_hooks.called_hooks, (
        "Init hook should not be called"
    )
    assert test_hooks.on_stop_1 not in test_hooks.called_hooks, (
        "Stop hook should not be called"
    )


def test_unsupported_hook_decorator():
    """Test that an UnsupportedHookError is raised when a hook is defined on a class
    that does not support it.
    """

    @provides_hooks(AIPerfHook.ON_STOP)
    class TestHooksUnsupported(MockHooks):
        @on_start
        async def _on_start_1(self):
            self.add_called_hook(self._on_start_1)

    with pytest.raises(UnsupportedHookError):
        TestHooksUnsupported()  # this should raise an UnsupportedHookError


@pytest.mark.asyncio
async def test_inheritance_hook_order():
    """Test that the hook order is correct when using inheritance."""

    class MockHooksInheritance2(MockHooks):
        @on_init
        async def on_init_99(self):
            assert self.on_init_1 in self.called_hooks
            self.add_called_hook(self.on_init_99)

        @on_init
        async def on_init_0(self):
            assert self.on_init_1 in self.called_hooks
            self.add_called_hook(self.on_init_0)

    test_hooks = MockHooksInheritance2()

    await test_hooks.initialize()

    assert test_hooks.on_init_0 in test_hooks.called_hooks, (
        "Subclass hook should be called"
    )
    assert test_hooks.on_init_1 in test_hooks.called_hooks, "Base hook should be called"


@pytest.mark.asyncio
async def test_inheritance_hook_override():
    """Test that a hook that is overridden in a subclass does not call the base class hook."""

    class MockHooksInheritance3(MockHooks):
        @on_init
        async def on_init_1(self):
            assert MockHooks.on_init_1 not in self.called_hooks
            self.add_called_hook(self.on_init_1)

    test_hooks = MockHooksInheritance3()

    await test_hooks.initialize()

    assert test_hooks.on_init_1 in test_hooks.called_hooks, (
        "Subclass hook should be called"
    )
    assert MockHooks.on_init_1 not in test_hooks.called_hooks, (
        "Base hook should not be called"
    )


def test_hook_ordering():
    """Test that the hook ordering is correct."""

    @provides_hooks(AIPerfHook.ON_INIT)
    class Hooks(HooksMixin):
        @on_init
        async def on_init_2(self):
            pass

        @on_init
        async def on_init_3(self):
            pass

        @on_init
        async def on_init_1(self):
            pass

    hooks = Hooks()

    # Ensure the hooks are added in the order they are defined
    assert [hook.func for hook in hooks.get_hooks(AIPerfHook.ON_INIT)] == [
        hooks.on_init_2,
        hooks.on_init_3,
        hooks.on_init_1,
    ], "Hooks should be registered in the order they are defined"

    class Hooks2(Hooks):
        @on_init
        async def on_init_99(self):
            pass

    hooks2 = Hooks2()

    # Ensure that base hooks are registered before the subclass hooks
    assert [hook.func for hook in hooks2.get_hooks(AIPerfHook.ON_INIT)] == [
        # Base hooks
        hooks2.on_init_2,
        hooks2.on_init_3,
        hooks2.on_init_1,
        # Subclass hooks
        hooks2.on_init_99,
    ], "Base hooks should be registered before subclass hooks"


@pytest.mark.asyncio
async def test_hook_overridden_methods_still_callable():
    """Test that overridden methods are still callable."""

    @provides_hooks(AIPerfHook.ON_INIT)
    class Hooks1(MockHookProvider):
        @on_init
        async def on_init_1(self):
            self.add_called_hook("Hooks1.on_init_1")

    class Hooks2(Hooks1):
        @on_init
        async def on_init_1(self):
            self.add_called_hook("Hooks2.on_init_1")

    hooks2 = Hooks2()
    await hooks2.initialize()

    assert hooks2.called_hooks == {
        "Hooks1.on_init_1",
        "Hooks2.on_init_1",
    }, "Base class hooks should still be called even if overridden"

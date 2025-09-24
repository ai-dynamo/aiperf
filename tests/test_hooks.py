# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

import pytest

from aiperf.common.exceptions import AIPerfMultiError, HookError, UnsupportedHookError
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


@pytest.mark.asyncio
async def test_hook_error_exception():
    """Test that HookError is raised when a hook throws an exception during execution."""

    @provides_hooks(AIPerfHook.ON_INIT)
    class HookWithError(MockHookProvider):
        @on_init
        async def failing_hook(self):
            raise ValueError("Something went wrong in the hook")

    hook_provider = HookWithError()

    # Should raise AIPerfMultiError containing the HookError
    with pytest.raises(AIPerfMultiError) as exc_info:
        await hook_provider.initialize()

    # Verify the multi-error contains our hook error
    assert len(exc_info.value.exceptions) == 1
    hook_error = exc_info.value.exceptions[0]
    assert isinstance(hook_error, HookError)
    assert hook_error.hook_class_name == "HookWithError"
    assert hook_error.hook_func_name == "failing_hook"
    assert isinstance(hook_error.exception, ValueError)
    assert str(hook_error.exception) == "Something went wrong in the hook"


@pytest.mark.asyncio
async def test_multiple_hook_errors():
    """Test that multiple hook errors are collected in AIPerfMultiError."""

    @provides_hooks(AIPerfHook.ON_INIT)
    class MultipleErrorHooks(MockHookProvider):
        @on_init
        async def failing_hook_1(self):
            raise ValueError("First error")

        @on_init
        async def failing_hook_2(self):
            raise RuntimeError("Second error")

    hook_provider = MultipleErrorHooks()

    with pytest.raises(AIPerfMultiError) as exc_info:
        await hook_provider.initialize()

    # Should have collected both hook errors
    assert len(exc_info.value.exceptions) == 2

    errors = exc_info.value.exceptions

    hook_error_1 = errors[0]
    hook_error_2 = errors[1]
    assert isinstance(hook_error_1, HookError)
    assert isinstance(hook_error_2, HookError)

    assert hook_error_1.hook_func_name == "failing_hook_1"
    assert hook_error_2.hook_func_name == "failing_hook_2"
    assert isinstance(hook_error_1.exception, ValueError)
    assert isinstance(hook_error_2.exception, RuntimeError)


def test_hook_error_properties():
    """Test that HookError properties are set correctly."""
    original_exception = ValueError("Test error message")
    hook_error = HookError("TestClass", "test_method", original_exception)

    assert hook_error.hook_class_name == "TestClass"
    assert hook_error.hook_func_name == "test_method"
    assert hook_error.exception is original_exception
    assert str(hook_error) == "TestClass.test_method: Test error message"


def test_unsupported_hook_error_attach_hook():
    """Test that UnsupportedHookError is raised when attaching an unsupported hook type."""

    @provides_hooks(AIPerfHook.ON_INIT)
    class LimitedHookProvider(HooksMixin):
        pass

    provider = LimitedHookProvider()

    # Try to attach a hook type that's not provided
    with pytest.raises(UnsupportedHookError) as exc_info:
        provider.attach_hook(AIPerfHook.ON_START, lambda: None)

    error_message = str(exc_info.value)
    assert all(
        snippet in error_message
        for snippet in [
            "@on_start",
            "LimitedHookProvider",
            "not provided by any base class",
        ]
    )


def test_unsupported_hook_error_message_content():
    """Test that UnsupportedHookError contains helpful information."""

    @provides_hooks(AIPerfHook.ON_INIT, AIPerfHook.ON_STOP)
    class TestHooksUnsupported(MockHooks):
        @on_start  # This hook is not provided
        async def _on_start_1(self):
            pass

    with pytest.raises(UnsupportedHookError) as exc_info:
        TestHooksUnsupported()

    error_message = str(exc_info.value)
    assert all(
        snippet in error_message
        for snippet in ["@on_start", "TestHooksUnsupported", "@on_init", "@on_stop"]
    )


@pytest.mark.asyncio
async def test_hook_execution_continues_after_error():
    """Test that hook execution continues after one hook fails (collecting all errors)."""

    @provides_hooks(AIPerfHook.ON_INIT)
    class MixedHooks(MockHookProvider):
        @on_init
        async def successful_hook_1(self):
            self.add_called_hook("successful_1")

        @on_init
        async def failing_hook(self):
            raise RuntimeError("Hook failed")

        @on_init
        async def successful_hook_2(self):
            self.add_called_hook("successful_2")

    hook_provider = MixedHooks()

    with pytest.raises(AIPerfMultiError):
        await hook_provider.initialize()

    # Both successful hooks should have been called despite the failure
    assert "successful_1" in hook_provider.called_hooks
    assert "successful_2" in hook_provider.called_hooks

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

import pytest

from aiperf.common.exceptions import UnsupportedHookError
from aiperf.common.hooks import (
    AIPerfHook,
    HooksMixin,
    on_cleanup,
    on_init,
    on_start,
    supports_hooks,
)


@supports_hooks(AIPerfHook.ON_INIT, AIPerfHook.ON_CLEANUP)
class BaseClass(HooksMixin):
    def __init__(self):
        super().__init__()
        self.called_hooks = set()

    async def initialize(self) -> None:
        await self.run_hooks_async(AIPerfHook.ON_INIT)

    async def cleanup(self) -> None:
        await self.run_hooks_async(AIPerfHook.ON_CLEANUP)


class MockHooks(BaseClass):
    @on_init
    async def on_init_1(self):
        self.called_hooks.add(self.on_init_1)

    @on_init
    async def on_init_2(self):
        self.called_hooks.add(self.on_init_2)

    @on_init
    async def on_init_3(self):
        self.called_hooks.add(self.on_init_3)

    @on_cleanup
    async def on_cleanup_1(self):
        self.called_hooks.add(self.on_cleanup_1)


@supports_hooks(AIPerfHook.ON_START)
class MockHooksInheritance(MockHooks):
    @on_init
    async def on_init_4(self):
        self.called_hooks.add(self.on_init_4)

    @on_cleanup
    async def on_cleanup_2(self):
        self.called_hooks.add(self.on_cleanup_2)

    async def start(self):
        await self.run_hooks_async(AIPerfHook.ON_START)

    @on_start
    async def on_start_1(self):
        self.called_hooks.add(self.on_start_1)


def test_hook_decorators():
    """Test the hook decorators."""
    test_hooks = MockHooks()

    assert test_hooks.get_hooks(AIPerfHook.ON_INIT) == [
        test_hooks.on_init_1,
        test_hooks.on_init_2,
        test_hooks.on_init_3,
    ]
    assert test_hooks.get_hooks(AIPerfHook.ON_CLEANUP) == [test_hooks.on_cleanup_1]


def test_hook_inheritance():
    """Test the hook inheritance."""
    test_hooks_inheritance = MockHooksInheritance()

    assert test_hooks_inheritance.get_hooks(AIPerfHook.ON_INIT) == [
        test_hooks_inheritance.on_init_1,
        test_hooks_inheritance.on_init_2,
        test_hooks_inheritance.on_init_3,
        test_hooks_inheritance.on_init_4,
    ]
    assert test_hooks_inheritance.get_hooks(AIPerfHook.ON_CLEANUP) == [
        test_hooks_inheritance.on_cleanup_1,
        test_hooks_inheritance.on_cleanup_2,
    ]
    assert test_hooks_inheritance.get_hooks(AIPerfHook.ON_START) == [
        test_hooks_inheritance.on_start_1
    ]


async def test_run_hooks_init():
    test_hooks = MockHooks()

    await test_hooks.initialize()

    assert test_hooks.on_init_1 in test_hooks.called_hooks
    assert test_hooks.on_init_2 in test_hooks.called_hooks
    assert test_hooks.on_init_3 in test_hooks.called_hooks


async def test_run_hooks_cleanup():
    test_hooks = MockHooksInheritance()

    await test_hooks.cleanup()

    assert test_hooks.on_cleanup_1 in test_hooks.called_hooks
    assert test_hooks.on_cleanup_2 in test_hooks.called_hooks
    assert test_hooks.on_init_1 not in test_hooks.called_hooks


async def test_inherited_run_hooks_start():
    test_hooks = MockHooksInheritance()

    await test_hooks.start()

    assert test_hooks.on_start_1 in test_hooks.called_hooks
    assert test_hooks.on_init_1 not in test_hooks.called_hooks
    assert test_hooks.on_cleanup_1 not in test_hooks.called_hooks


def test_unsupported_hook_decorator():
    """Test that an UnsupportedHookError is raised when a hook is defined on a class
    that does not support it.
    """

    @supports_hooks(AIPerfHook.ON_CLEANUP)
    class TestHooksUnsupported(MockHooks):
        @on_start
        async def _on_start_1(self):
            self.called_hooks.add(self._on_start_1)

    with pytest.raises(UnsupportedHookError):
        TestHooksUnsupported()  # this should raise an UnsupportedHookError


async def test_instance_additional_hooks():
    """Test that additional hooks can be added to a class that supports hooks."""
    test_hooks = MockHooksInheritance()

    async def custom_start_hook():
        test_hooks.called_hooks.add(custom_start_hook)

    test_hooks.register_hook(AIPerfHook.ON_START, custom_start_hook)

    assert test_hooks.get_hooks(AIPerfHook.ON_START) == [
        test_hooks.on_start_1,
        custom_start_hook,
    ]

    await test_hooks.start()

    assert custom_start_hook in test_hooks.called_hooks
    assert test_hooks.on_start_1 in test_hooks.called_hooks


async def test_instance_additional_supported_hooks():
    """Test that additional hook types can be supported by a class"""
    test_hooks = MockHooks()

    async def custom_stop_hook():
        test_hooks.called_hooks.add(custom_stop_hook)

    # this should raise an UnsupportedHookError because the hook type is not supported
    with pytest.raises(UnsupportedHookError):
        test_hooks.register_hook(AIPerfHook.ON_STOP, custom_stop_hook)

    # Now we add the hook type to the supported hooks
    test_hooks.supported_hooks.add(AIPerfHook.ON_STOP)

    # Now we can register the hook and it will not raise an UnsupportedHookError
    test_hooks.register_hook(AIPerfHook.ON_STOP, custom_stop_hook)

    # Expect the hook to be in the list of hooks
    assert test_hooks.get_hooks(AIPerfHook.ON_STOP) == [custom_stop_hook]

    async def custom_init_hook():
        test_hooks.called_hooks.add(custom_init_hook)
        # Hack to allow the hook to run the newly added ON_STOP hook
        await test_hooks.run_hooks_async(AIPerfHook.ON_STOP)

    test_hooks.register_hook(
        AIPerfHook.ON_INIT, custom_init_hook
    )  # this should not raise an UnsupportedHookError

    await test_hooks.initialize()

    # Expect the custom init and stop hooks to have been called
    assert custom_init_hook in test_hooks.called_hooks
    assert custom_stop_hook in test_hooks.called_hooks

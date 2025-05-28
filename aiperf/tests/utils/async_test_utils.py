#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""
Utilities for testing asynchronous code.
"""

import contextlib
from collections.abc import AsyncIterator
from typing import Any, TypeVar, cast

T = TypeVar("T")


async def async_noop(*args, **kwargs) -> None:
    """
    A no-op async function for testing purposes.

    Can be used to replace asyncio.sleep, asyncio.wait_for, or other async calls in tests.
    Accepts any arguments but performs no operation and returns immediately.
    """
    return


async def async_fixture(fixture: T) -> T:
    """
    Manually await an async pytest fixture.

    This is necessary because pytest fixtures are not awaited by default in test methods.
    If the fixture is an async generator, this will get the first yielded value.

    Args:
        fixture: The fixture to await

    Returns:
        The awaited fixture value
    """
    if hasattr(fixture, "__aiter__"):
        # If it's an async generator, get the first yielded value
        with contextlib.suppress(StopAsyncIteration):
            async_gen = cast(AsyncIterator[Any], fixture)
            value = await anext(async_gen)
            return cast(T, value)

    # Otherwise return the fixture as is
    return fixture

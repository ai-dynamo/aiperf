# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Absolute-deadline helpers for Kubernetes test assertions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class DeadlineExceededError(AssertionError):
    """An assertion operation exhausted its shared absolute deadline."""


async def await_before_deadline(
    deadline: float,
    operation_name: str,
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Run an awaitable factory without exceeding an absolute loop deadline.

    Raises:
        DeadlineExceededError: The deadline passed before the operation could
            start or while it was in flight.
    """
    loop = asyncio.get_running_loop()
    remaining = deadline - loop.time()
    if remaining <= 0:
        raise DeadlineExceededError(
            f"Cleanup deadline expired before {operation_name} could start"
        )

    timeout = asyncio.timeout_at(deadline)
    try:
        async with timeout:
            return await operation()
    except TimeoutError as exc:
        if not timeout.expired():
            raise
        raise DeadlineExceededError(
            f"Cleanup deadline expired while {operation_name}; "
            f"the operation exceeded the {remaining:.1f}s remaining"
        ) from exc


async def delete_and_observe_until_deadline(
    deadline: float,
    resource_name: str,
    delete_resource: Callable[[], Awaitable[None]],
    resource_exists: Callable[[], Awaitable[bool]],
    poll_interval: float = 1,
) -> None:
    """Delete a predeclared resource repeatedly through a teardown deadline.

    A timed-out create can commit after an earlier delete reported the resource
    absent. Polling covers the bounded failure-teardown window, then a reserved
    final phase deletes once more and freshly observes the resulting state.
    """
    loop = asyncio.get_running_loop()
    remaining = deadline - loop.time()
    if remaining <= 0:
        raise DeadlineExceededError(
            f"Cleanup deadline expired before deleting {resource_name} "
            "during failure teardown could start"
        )

    final_phase_budget = min(poll_interval, remaining / 2)
    polling_deadline = deadline - final_phase_budget
    while loop.time() < polling_deadline:
        try:
            exists = await _delete_and_observe_before_deadline(
                polling_deadline,
                resource_name,
                delete_resource,
                resource_exists,
            )
        except DeadlineExceededError:
            break
        if exists:
            continue
        await asyncio.sleep(min(poll_interval, max(0, polling_deadline - loop.time())))

    exists = await _delete_and_observe_before_deadline(
        deadline,
        resource_name,
        delete_resource,
        resource_exists,
    )

    if exists:
        raise DeadlineExceededError(
            f"Cleanup deadline expired while {resource_name} still existed"
        )


async def _delete_and_observe_before_deadline(
    deadline: float,
    resource_name: str,
    delete_resource: Callable[[], Awaitable[None]],
    resource_exists: Callable[[], Awaitable[bool]],
) -> bool:
    """Delete an exact resource identity and freshly observe its state."""
    await await_before_deadline(
        deadline,
        f"deleting {resource_name} during failure teardown",
        delete_resource,
    )
    return await await_before_deadline(
        deadline,
        f"checking {resource_name} during failure teardown",
        resource_exists,
    )

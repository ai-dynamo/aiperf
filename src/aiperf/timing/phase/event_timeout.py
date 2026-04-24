# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Await an asyncio.Event with optional timeout and logged lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def wait_for_event_with_timeout(
    *,
    name: str,
    event: asyncio.Event,
    timeout: float | None,
    task_to_cancel: asyncio.Task | None,
    set_event_on_timeout: bool,
    info: Callable[[str], None],
    debug: Callable[[Callable[[], str]], None],
    error: Callable[[str], None],
) -> bool:
    """Wait for event with optional timeout.

    Args:
        name: The name of the event to wait for.
        event: The event to wait for.
        timeout: The timeout in seconds.
            If None, the event will be waited for indefinitely.
            If timeout is <= 0, returns immediately with timeout.
        task_to_cancel: The optional task to cancel when the timeout occurs.
        set_event_on_timeout: If True, the event will also be set when the timeout occurs.
        info: Info-level logger.
        debug: Debug-level logger (lazy — accepts a callable).
        error: Error-level logger.

    Returns:
        True if the event timed out, False if the event was set before timeout.
    """
    if timeout is None:
        debug(lambda: f"Waiting for event '{name}' indefinitely")
        await event.wait()
        return False

    def _on_timeout() -> bool:
        info(f"Timeout of {timeout}s elapsed for event '{name}'")
        if set_event_on_timeout:
            event.set()
        if task_to_cancel:
            task_to_cancel.cancel()
        return True

    if timeout <= 0:
        debug(lambda: f"Timeout already elapsed for event '{name}'")
        return _on_timeout()

    try:
        info(f"Waiting for event '{name}' with timeout of {timeout}s")
        await asyncio.wait_for(event.wait(), timeout=timeout)
        debug(lambda: f"Event '{name}' set before timeout of {timeout}s")
        return False

    except asyncio.TimeoutError:
        return _on_timeout()

    except Exception as e:
        error(f"Error waiting for event '{name}' with timeout: {e!r}")
        raise

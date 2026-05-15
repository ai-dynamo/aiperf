# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic concurrency limit primitive and supporting stats dataclass."""

import asyncio
from dataclasses import dataclass


@dataclass(slots=True)
class ConcurrencyStats:
    """Instrumentation for concurrency tracking.

    Tracks acquire/release/wait operations for observability and testing.
    Safe in single-threaded asyncio (operations between awaits are atomic).
    """

    acquire_count: int = 0
    """Number of times a slot was successfully acquired."""

    release_count: int = 0
    """Number of times a slot was released."""

    wait_count: int = 0
    """Number of times an acquire had to wait for a slot."""


class DynamicConcurrencyLimit:
    """Dynamic concurrency limit using semaphore + debt tracking.

    Supports runtime limit adjustment for smooth phase transitions:
    - Increase: Cancels debt first, then adds slots (immediate capacity)
    - Decrease: Drains available slots, remainder as debt (graceful drain)

    Why Debt Tracking:
        We can't make _value negative - asyncio.Semaphore.acquire() has:
        ```python
            if not self.locked():  # locked() = (_value == 0)
                return True        # Negative bypasses blocking
        ```

        Debt keeps `_value >= 0` to preserve the semaphore's invariant.

    See `GlobalPhaseConcurrencyLimiter` for layered concurrency system usage.

    Thread Safety:
        Safe for asyncio single-threaded concurrency.

    Example:
        ```python
        limit = DynamicConcurrencyLimit(10)
        await limit.acquire()
        try:
            await do_work()
        finally:
            limit.release()
        ```
    """

    def __init__(self, initial_limit: int = 0) -> None:
        """Initialize with initial concurrency limit.

        Args:
            initial_limit: Starting number of available slots (default 0).
        """
        self._semaphore = asyncio.Semaphore(initial_limit)
        self._current_limit = initial_limit
        self._debt = 0
        self.stats = ConcurrencyStats()

    @property
    def current_limit(self) -> int:
        """Current concurrency limit."""
        return self._current_limit

    @property
    def debt(self) -> int:
        """Outstanding debt (releases to absorb before freeing slots)."""
        return self._debt

    @property
    def effective_slots(self) -> int:
        """Number of slots currently available in the semaphore, taking into account debt."""
        return max(0, self._semaphore._value - self._debt)

    def set_limit(self, new_limit: int) -> None:
        """Update the concurrency limit.

        Args:
            new_limit: New concurrency limit (must be >= 0)

        Behavior:
            - Increase: Cancels debt first, then adds additional slots
            - Decrease: Drains available slots first (immediate), remainder as debt

        Note:
            This is a synchronous operation. Waiters are scheduled to wake
            when slots are added, but won't run until the caller yields.

        Design Note:
            Hybrid approach: drain available slots directly, remainder as debt.
            Direct _value decrement is safe when _value > 0 (no blocked waiters).
            Debt keeps _value >= 0 to preserve semaphore invariant (see class doc).
        """
        if new_limit < 0:
            raise ValueError(f"Limit must be non-negative, got {new_limit}")

        diff = new_limit - self._current_limit

        if diff > 0:
            # Increase: cancel debt first, then add slots
            cancel = min(diff, self._debt)
            self._debt -= cancel
            slots_to_add = diff - cancel
            for _ in range(slots_to_add):
                self._semaphore.release()
        elif diff < 0:
            # Decrease: track debt for future releases to absorb
            diff = abs(diff)
            if not self._semaphore.locked():
                # If the semaphore is not locked, we can directly decrement the value
                # without waking up any waiters. It is safe to drain it all the way down to 0.
                # However, if diff is less than the current value, we need to drain only the difference.
                to_drain = min(diff, self._semaphore._value)
                self._semaphore._value -= to_drain
                diff -= to_drain
            self._debt += diff

        self._current_limit = new_limit

    async def acquire(self) -> None:
        """Acquire a concurrency slot.

        Blocks until a slot is available. Use with try/finally to ensure
        release() is called.
        """
        # Track if we'll need to wait (semaphore has no available slots)
        if self._semaphore.locked():
            self.stats.wait_count += 1
        await self._semaphore.acquire()
        self.stats.acquire_count += 1

    def release(self) -> None:
        """Release a concurrency slot.

        If there is outstanding debt (from a limit decrease), this release is
        absorbed by the debt instead of freeing a slot for other acquirers.
        """
        self.stats.release_count += 1
        if self._debt > 0:
            self._debt -= 1
        else:
            self._semaphore.release()

    def try_acquire(self) -> bool:
        """Try to acquire a concurrency slot without blocking.

        Returns immediately whether the slot was acquired or not.
        Unlike acquire(), this never blocks or waits for a slot.

        Returns:
            True if slot was acquired, False if no slot available.

        Note:
            Must still call release() if True is returned.
        """
        if self._semaphore.locked():
            return False
        # Semaphore not locked means _value > 0, so acquire won't block
        self._semaphore._value -= 1
        self.stats.acquire_count += 1
        return True

    def locked(self) -> bool:
        """Check if the semaphore is locked."""
        return self._semaphore.locked()

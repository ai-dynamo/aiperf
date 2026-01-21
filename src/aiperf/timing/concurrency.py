# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concurrency limiting for credit issuance.

Provides three layers of concurrency control:
- DynamicConcurrencyLimit: Semaphore with dynamic limit adjustment and debt tracking
- GlobalPhaseConcurrencyLimiter: Phase-specific limits with global coordination
- ConcurrencyManager: High-level manager for session and prefill concurrency

Used by CreditIssuer and CreditCallbackHandler to control how many concurrent
sessions and prefill requests are active during benchmarking.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from aiperf.common.enums import CreditPhase


@dataclass(slots=True)
class ConcurrencyStats:
    """Instrumentation for concurrency tracking.

    Tracks acquire/release/wait operations for observability and testing.
    Safe in single-threaded asyncio (operations between awaits are atomic).

    Attributes:
        acquire_count: The number of times a slot was successfully acquired.
        release_count: The number of times a slot was released.
        wait_count: The number of times an acquire had to wait for a slot.
    """

    acquire_count: int = 0
    release_count: int = 0
    wait_count: int = 0


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


class GlobalPhaseConcurrencyLimiter:
    """Concurrency limiter with phase-specific and global limits.

    Combines a global DynamicConcurrencyLimit with phase-specific limits.
    Requests must acquire both the global slot and the phase-specific limit.

    Both global and phase limits use DynamicConcurrencyLimit for consistent
    stats tracking and dynamic limit adjustment capability.

    Design:
        Phase limits are created fresh via configure_for_phase(), providing
        immediate hard enforcement of the configured limit. The global limit
        uses the hybrid approach (drain available + debt) for graceful drain
        of in-flight requests from previous phases. This layered approach
        ensures new phases respect their limits while allowing old phases
        to complete gracefully.
    """

    def __init__(self) -> None:
        """Initialize as disabled with empty phase limits and a global limit of 0."""
        self._enabled = False
        self._global_limit = DynamicConcurrencyLimit()
        self._phase_limits: dict[CreditPhase, DynamicConcurrencyLimit] = {}

    @property
    def enabled(self) -> bool:
        """Whether concurrency limiting is enabled for this limiter."""
        return self._enabled

    def configure_for_phase(self, phase: CreditPhase, limit: int | None) -> None:
        """Configure limits for a new phase.

        Args:
            phase: The phase to configure
            limit: Maximum concurrent slots for this phase.
                If None, concurrency limiting is disabled globally for this limiter
                (not just for this phase).
        """
        if limit is None:
            self._enabled = False
            return

        self._enabled = True
        self._phase_limits[phase] = DynamicConcurrencyLimit(limit)
        self._global_limit.set_limit(limit)

    async def acquire(
        self, phase: CreditPhase, can_proceed_fn: Callable[[], bool]
    ) -> bool:
        """Acquire a concurrency slot.

        Acquires both global and phase-specific slots. Checks can_proceed_fn()
        after each acquisition to allow early exit if phase is stopping.

        Args:
            phase: The phase to acquire for
            can_proceed_fn: Callable returning True if we should continue

        Returns:
            True if slot was acquired, False if cancelled (can_proceed_fn returned False)

        Raises:
            ValueError: If phase not configured via configure_for_phase()
        """
        if phase not in self._phase_limits:
            raise ValueError(f"Phase {phase} not configured in limiter")

        phase_limit = self._phase_limits[phase]

        acquired_global = False
        acquired_phase = False
        try:
            await self._global_limit.acquire()
            acquired_global = True

            if not can_proceed_fn():
                self._global_limit.release()
                return False

            await phase_limit.acquire()
            acquired_phase = True

            if not can_proceed_fn():
                phase_limit.release()
                self._global_limit.release()
                return False

            return True
        except Exception:
            if acquired_phase:
                phase_limit.release()
            if acquired_global:
                self._global_limit.release()
            raise

    def try_acquire(
        self, phase: CreditPhase, can_proceed_fn: Callable[[], bool]
    ) -> bool:
        """Try to acquire a concurrency slot without blocking.

        Attempts to acquire both global and phase-specific slots immediately.
        Unlike acquire(), this never blocks or waits for slots.

        Args:
            phase: The phase to acquire for
            can_proceed_fn: Callable returning True if we should continue.
                Checked BEFORE attempting slot acquisition.

        Returns:
            True if slots were acquired, False if no slots available or
            can_proceed_fn returned False.

        Raises:
            ValueError: If phase not configured via configure_for_phase()
        """
        if phase not in self._phase_limits:
            raise ValueError(f"Phase {phase} not configured in limiter")

        # Check stop conditions first to avoid unnecessary slot attempts
        if not can_proceed_fn():
            return False

        phase_limit = self._phase_limits[phase]

        # Try global first
        if not self._global_limit.try_acquire():
            return False

        # Try phase - release global if phase fails
        if not phase_limit.try_acquire():
            self._global_limit.release()
            return False

        return True

    def release(self, phase: CreditPhase) -> None:
        """Release a concurrency slot.

        Args:
            phase: The phase to release for

        Raises:
            ValueError: If phase not configured via configure_for_phase()
        """
        if phase not in self._phase_limits:
            raise ValueError(f"Phase {phase} not configured in limiter")

        self._global_limit.release()
        self._phase_limits[phase].release()

    def slot_available(self, phase: CreditPhase) -> bool:
        """Check if a slot is available without blocking.

        Args:
            phase: The phase to check availability for.

        Returns:
            True if both global and phase slots are available (not locked).
            False if no slots available.
        """
        if phase not in self._phase_limits:
            raise ValueError(f"Phase {phase} not configured in limiter")
        return (
            not self._global_limit.locked() and not self._phase_limits[phase].locked()
        )

    def get_held_slots(self, phase: CreditPhase) -> int:
        """Get the number of slots currently held for a specific phase.

        Args:
            phase: The phase to query.

        Returns:
            Number of slots currently acquired (not yet released). Returns 0 if
            phase is not configured.
        """
        if phase not in self._phase_limits:
            return 0

        phase_limit = self._phase_limits[phase]
        return max(0, phase_limit.current_limit - phase_limit.effective_slots)

    @property
    def global_stats(self) -> ConcurrencyStats:
        """Global concurrency stats across all phases."""
        return self._global_limit.stats

    def get_phase_stats(self, phase: CreditPhase) -> ConcurrencyStats | None:
        """Get stats for a specific phase.

        Args:
            phase: The phase to get stats for.

        Returns:
            ConcurrencyStats for the phase, or None if phase is not configured.
        """
        phase_limit = self._phase_limits.get(phase)
        return phase_limit.stats if phase_limit else None


class ConcurrencyManager:
    """Manager for session and prefill concurrency limits.

    Supports two independent concurrency dimensions:
    - Session concurrency: Limits concurrent sessions (conversations). A slot is
      acquired on first turn and released on final turn or phase cleanup.
    - Prefill concurrency: Limits requests waiting for first token (prefill phase).
      A slot is acquired on every turn and released when TTFT is received.

    Each dimension uses a GlobalPhaseConcurrencyLimiter to handle phase-specific
    and global limits.
    """

    def __init__(self) -> None:
        """Initialize the concurrency manager.

        Note: By default, both session and prefill concurrency limiting is disabled.
        They will be enabled or disabled based on the phase config.
        """
        self._session_limiter = GlobalPhaseConcurrencyLimiter()
        self._prefill_limiter = GlobalPhaseConcurrencyLimiter()

    def configure_for_phase(
        self,
        phase: CreditPhase,
        concurrency: int | None,
        prefill_concurrency: int | None,
    ) -> None:
        """Configure concurrency limits for a new phase.

        Must be called before acquiring slots for a phase. Configures both
        session and prefill limiters unconditionally - each limiter internally
        enables/disables based on whether the limit is None.

        Args:
            phase: The phase to configure
            concurrency: Maximum concurrent session slots for this phase.
                If None, session concurrency limiting is disabled globally.
            prefill_concurrency: Maximum concurrent prefill slots for this phase.
                If None, prefill concurrency limiting is disabled globally.
        """
        self._session_limiter.configure_for_phase(phase, concurrency)
        self._prefill_limiter.configure_for_phase(phase, prefill_concurrency)

    def session_slot_available(self, phase: CreditPhase) -> bool:
        """Check if a session slot is available without blocking.

        Returns True when concurrency limiting is disabled (no limit configured).
        """
        if not self._session_limiter.enabled:
            return True  # No limit - always available
        return self._session_limiter.slot_available(phase)

    async def acquire_session_slot(
        self, phase: CreditPhase, can_proceed_fn: Callable[[], bool]
    ) -> bool:
        """Acquire a session concurrency slot.

        Args:
            phase: The credit phase to acquire the slot for.
            can_proceed_fn: Callback that returns True if the operation should proceed.
                Used to check phase stop conditions (request limits, session quotas,
                cancellation). ALWAYS called, even when concurrency limiting is
                disabled, to ensure stop conditions are enforced regardless of
                configuration.

        Returns:
            True if slot was acquired and can_proceed_fn returned True, False otherwise.
        """
        if not self._session_limiter.enabled:
            return can_proceed_fn()
        return await self._session_limiter.acquire(phase, can_proceed_fn)

    def try_acquire_session_slot(
        self, phase: CreditPhase, can_proceed_fn: Callable[[], bool]
    ) -> bool:
        """Try to acquire a session concurrency slot without blocking.

        Non-blocking version of acquire_session_slot. Returns immediately
        whether the slot was acquired or not.

        Args:
            phase: The credit phase to acquire the slot for.
            can_proceed_fn: Callback that returns True if the operation should proceed.
                Checked BEFORE attempting slot acquisition.

        Returns:
            True if slot was acquired, False if no slot available or
            can_proceed_fn returned False.
        """
        if not self._session_limiter.enabled:
            return can_proceed_fn()
        return self._session_limiter.try_acquire(phase, can_proceed_fn)

    def release_session_slot(self, phase: CreditPhase) -> None:
        """Release a session concurrency slot.

        Called when a session ends (final turn completed or cancelled) or during
        phase cleanup for in-flight sessions. No-op if session concurrency is disabled.

        Args:
            phase: The credit phase to release the slot for.
        """
        if self._session_limiter.enabled:
            self._session_limiter.release(phase)

    async def acquire_prefill_slot(
        self, phase: CreditPhase, can_proceed_fn: Callable[[], bool]
    ) -> bool:
        """Acquire a prefill concurrency slot.

        Prefill slots limit how many requests can be waiting for first token.
        Released when TTFT is received (via release_prefill_slot), or when the
        credit is returned without TTFT (deadlock prevention for errors/cancellation).

        Args:
            phase: The credit phase to acquire the slot for.
            can_proceed_fn: Callback that returns True if the operation should proceed.
                Used to check phase stop conditions (request limits, session quotas,
                cancellation). ALWAYS called, even when concurrency limiting is
                disabled, to ensure stop conditions are enforced regardless of
                configuration.

        Returns:
            True if slot was acquired and can_proceed_fn returned True, False otherwise.
        """
        if not self._prefill_limiter.enabled:
            return can_proceed_fn()
        return await self._prefill_limiter.acquire(phase, can_proceed_fn)

    def try_acquire_prefill_slot(
        self, phase: CreditPhase, can_proceed_fn: Callable[[], bool]
    ) -> bool:
        """Try to acquire a prefill concurrency slot without blocking.

        Non-blocking version of acquire_prefill_slot. Returns immediately
        whether the slot was acquired or not.

        Args:
            phase: The credit phase to acquire the slot for.
            can_proceed_fn: Callback that returns True if the operation should proceed.
                Checked BEFORE attempting slot acquisition.

        Returns:
            True if slot was acquired, False if no slot available or
            can_proceed_fn returned False.
        """
        if not self._prefill_limiter.enabled:
            return can_proceed_fn()
        return self._prefill_limiter.try_acquire(phase, can_proceed_fn)

    def release_prefill_slot(self, phase: CreditPhase) -> None:
        """Release a prefill concurrency slot.

        Called when TTFT is received (normal path) or when a credit is returned
        without TTFT (error/cancellation path). No-op if prefill concurrency is disabled.

        Args:
            phase: The credit phase to release the slot for.
        """
        if self._prefill_limiter.enabled:
            self._prefill_limiter.release(phase)

    def release_stuck_slots(self, phase: CreditPhase) -> tuple[int, int]:
        """Release all stuck slots for a phase during force-completion.

        Called when cancel drain timeout expires and credits will never return.
        Prevents slot leaks that would reduce capacity for subsequent phases.

        Args:
            phase: The phase to release stuck slots for.

        Returns:
            Tuple of (session_slots_released, prefill_slots_released)
        """
        session_released = 0
        if self._session_limiter.enabled:
            session_released = self._session_limiter.get_held_slots(phase)
            for _ in range(session_released):
                self._session_limiter.release(phase)

        prefill_released = 0
        if self._prefill_limiter.enabled:
            prefill_released = self._prefill_limiter.get_held_slots(phase)
            for _ in range(prefill_released):
                self._prefill_limiter.release(phase)

        return session_released, prefill_released

    def get_session_stats(
        self, phase: CreditPhase | None = None
    ) -> ConcurrencyStats | None:
        """Get session concurrency stats.

        Args:
            phase: If provided, get phase-specific stats. Otherwise get global stats.

        Returns:
            ConcurrencyStats or None if session limiting not enabled.
        """
        if not self._session_limiter.enabled:
            return None
        if phase is not None:
            return self._session_limiter.get_phase_stats(phase)
        return self._session_limiter.global_stats

    def set_session_limit(self, phase: CreditPhase, limit: int) -> None:
        """Set session concurrency limit for both global and phase.

        Args:
            phase: The phase whose limit should be adjusted along with global.
            limit: The new concurrency limit.
        """
        if not self._session_limiter.enabled:
            return
        self._session_limiter._global_limit.set_limit(limit)
        if phase in self._session_limiter._phase_limits:
            self._session_limiter._phase_limits[phase].set_limit(limit)

    def set_prefill_limit(self, phase: CreditPhase, limit: int) -> None:
        """Set prefill concurrency limit for both global and phase.

        Args:
            phase: The phase whose limit should be adjusted along with global.
            limit: The new concurrency limit.
        """
        if not self._prefill_limiter.enabled:
            return
        self._prefill_limiter._global_limit.set_limit(limit)
        if phase in self._prefill_limiter._phase_limits:
            self._prefill_limiter._phase_limits[phase].set_limit(limit)

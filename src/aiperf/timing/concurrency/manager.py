# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level manager for session and prefill concurrency dimensions."""

from collections.abc import Callable

from aiperf.common.enums import CreditPhase
from aiperf.timing.concurrency.dynamic_limit import ConcurrencyStats
from aiperf.timing.concurrency.global_phase_limiter import (
    GlobalPhaseConcurrencyLimiter,
)


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

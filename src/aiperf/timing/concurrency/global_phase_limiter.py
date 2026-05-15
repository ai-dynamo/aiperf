# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-aware concurrency limiter composing global + per-phase limits."""

from collections.abc import Callable

from aiperf.common.enums import CreditPhase
from aiperf.timing.concurrency.dynamic_limit import (
    ConcurrencyStats,
    DynamicConcurrencyLimit,
)


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

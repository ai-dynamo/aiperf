# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase lifecycle state machine.

Explicit state machine for credit phase lifecycle with validated transitions.
States: CREATED → STARTED → SENDING_COMPLETE → COMPLETE

Cancellation is a flag (can happen at any state), not a state itself.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CaseInsensitiveStrEnum

if TYPE_CHECKING:
    from aiperf.timing.config import CreditPhaseConfig


class PhaseState(CaseInsensitiveStrEnum):
    """Explicit phase lifecycle states."""

    CREATED = "created"
    STARTED = "started"
    SENDING_COMPLETE = "sending_complete"
    COMPLETE = "complete"


_SENDING_COMPLETE_STATES = {PhaseState.SENDING_COMPLETE, PhaseState.COMPLETE}


class PhaseLifecycle:
    """Explicit phase state machine with timestamps.

    Manages phase lifecycle transitions with validation.

    State transitions:
        CREATED → STARTED → SENDING_COMPLETE → COMPLETE

    Cancellation is orthogonal to state - a cancelled phase still completes
    its lifecycle for cleanup.

    Uses wall-clock time (time.time_ns()) for timestamps since they're published
    across services and need to be comparable. perf_counter is process-local.
    """

    def __init__(self, config: CreditPhaseConfig) -> None:
        self._config = config
        self.state: PhaseState = PhaseState.CREATED

        # Timestamps (wall clock - time.time_ns())
        self.started_at_ns: int | None = None
        self.sending_complete_at_ns: int | None = None
        self.complete_at_ns: int | None = None

        # Performance timestamps (process-local - time.perf_counter_ns())
        self.started_at_perf_ns: int | None = None

        # Transition metadata
        self.timeout_triggered: bool = False
        self.grace_period_triggered: bool = False

        # Cancellation flag (orthogonal to state)
        self.was_cancelled: bool = False

    @property
    def started_at_perf_sec(self) -> float | None:
        """The start time of the phase in performance seconds."""
        return (
            self.started_at_perf_ns / NANOS_PER_SECOND
            if self.started_at_perf_ns is not None
            else None
        )

    def start(self) -> None:
        """Transition to STARTED state.

        Raises:
            ValueError: If already started (not in CREATED state).
        """
        if self.state != PhaseState.CREATED:
            raise ValueError("Credit phase already started")
        self.state = PhaseState.STARTED
        perf_ns, time_ns = time.perf_counter_ns(), time.time_ns()
        self.started_at_ns = time_ns
        self.started_at_perf_ns = perf_ns

    def mark_sending_complete(self, *, timeout_triggered: bool = False) -> None:
        """Transition to SENDING_COMPLETE state.

        Raises:
            ValueError: If not started or already in SENDING_COMPLETE/COMPLETE.
        """
        if self.state == PhaseState.CREATED:
            raise ValueError("Credit phase not started. Call start() first.")
        if self.state in _SENDING_COMPLETE_STATES:
            raise ValueError("Credit phase already completed sending")
        self.state = PhaseState.SENDING_COMPLETE
        self.sending_complete_at_ns = time.time_ns()
        if timeout_triggered:
            self.timeout_triggered = True

    def mark_complete(self, *, grace_period_triggered: bool = False) -> None:
        """Transition to COMPLETE state.

        Raises:
            ValueError: If not in SENDING_COMPLETE state.
        """
        if self.state != PhaseState.SENDING_COMPLETE:
            if self.state == PhaseState.COMPLETE:
                raise ValueError("Credit phase already completed")
            raise ValueError(
                "Credit phase has not completed sending. Call mark_sending_complete() first."
            )
        self.state = PhaseState.COMPLETE
        self.complete_at_ns = time.time_ns()
        if grace_period_triggered:
            self.grace_period_triggered = True

    def cancel(self) -> None:
        """Mark phase as cancelled. Can be called at any state."""
        self.was_cancelled = True

    # =========================================================================
    # Time left in phase
    # =========================================================================

    def time_left_in_seconds(self, include_grace_period: bool = False) -> float | None:
        """Calculate remaining time in phase.

        Uses perf_counter for monotonic, high-precision timing.
        This is the authoritative source for time-left calculations.

        Args:
            include_grace_period: If True, includes grace_period_sec in remaining time.
                Used when waiting for in-flight requests to complete.

        Returns:
            Remaining time in seconds, or None if no duration limit configured.
            Returns 0.0 if time has elapsed.
        """
        if self._config.expected_duration_sec is None:
            return None

        if self.started_at_perf_ns is None:
            return None  # Not started yet

        elapsed_ns = time.perf_counter_ns() - self.started_at_perf_ns
        elapsed_sec = elapsed_ns / NANOS_PER_SECOND
        grace = (self._config.grace_period_sec or 0) if include_grace_period else 0
        time_left = self._config.expected_duration_sec + grace - elapsed_sec

        return max(0.0, time_left)

    # =========================================================================
    # Convenience Properties
    # =========================================================================

    @property
    def is_started(self) -> bool:
        """True if phase has started (in any state after CREATED)."""
        return self.state != PhaseState.CREATED

    @property
    def is_sending_complete(self) -> bool:
        """True if all credits have been sent (SENDING_COMPLETE or COMPLETE)."""
        return self.state in _SENDING_COMPLETE_STATES

    @property
    def is_complete(self) -> bool:
        """True if phase is complete."""
        return self.state == PhaseState.COMPLETE

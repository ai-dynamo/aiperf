# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.aiperf_logger import AIPerfLogger


def clamp_inter_turn_delay_ms(
    delay_ms: float | None, cap_seconds: float | None
) -> float | None:
    """Clamp ``delay_ms`` to at most ``cap_seconds * 1000`` ms.

    Returns the input unchanged when either value is ``None`` or when the
    delay is already at or below the cap. Negative values pass through
    unchanged.
    """
    if delay_ms is None or cap_seconds is None:
        return delay_ms
    cap_ms = cap_seconds * 1000.0
    if delay_ms > cap_ms:
        return cap_ms
    return delay_ms


class DelayCapTracker:
    """Per-loader counter that clamps inter-turn delays and logs a summary.

    Note: not a settable cap value — a stateful clamp+counter+summary helper.

    Subscribers call :meth:`clamp` on every per-turn delay value (ms or
    ``None``); the tracker returns the clamped value, increments the
    capped-count when clamping actually fires, and records the largest
    pre-clamp delay seen. Loaders call :meth:`log_summary` once after a
    load completes to emit a single info-level summary if any clamp
    happened.
    """

    __slots__ = ("cap_seconds", "capped_count", "max_observed_ms")

    def __init__(self, cap_seconds: float | None) -> None:
        """Initialize with an optional cap (seconds); ``None`` disables clamping."""
        self.cap_seconds = cap_seconds
        self.capped_count = 0
        self.max_observed_ms = 0.0

    def clamp(self, delay_ms: float | None) -> float | None:
        """Return ``delay_ms`` clamped to the cap, updating counters.

        Returns ``None`` when ``delay_ms`` is ``None``. When ``cap_seconds``
        is ``None`` the input passes through unchanged. Negative values
        also pass through unchanged (matching
        :func:`clamp_inter_turn_delay_ms` and never count toward
        ``capped_count`` or ``max_observed_ms``).
        """
        if delay_ms is None:
            return None
        if self.cap_seconds is None:
            return delay_ms
        if delay_ms > self.max_observed_ms:
            self.max_observed_ms = float(delay_ms)
        cap_ms = self.cap_seconds * 1000.0
        if delay_ms > cap_ms:
            self.capped_count += 1
            return cap_ms
        return delay_ms

    def reset(self) -> None:
        """Zero the capped-count and max-observed counters (cap value untouched)."""
        self.capped_count = 0
        self.max_observed_ms = 0.0

    def log_summary(self, *, logger_name: str) -> None:
        """Emit one info-level summary line if any delays were clamped; otherwise no-op."""
        if self.cap_seconds is None or self.capped_count == 0:
            return
        AIPerfLogger(logger_name).info(
            f"Capped {self.capped_count:,} inter-turn delays exceeding "
            f"{self.cap_seconds}s (max observed: {self.max_observed_ms:,.1f} ms)"
        )

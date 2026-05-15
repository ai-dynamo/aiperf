# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concise log-message formatters for phase lifecycle events."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.models import CreditPhaseStats


def format_phase_started(stats: CreditPhaseStats) -> str:
    """Format a concise log message for phase start."""
    parts = [f"Phase {stats.phase} started"]
    targets = []
    if stats.total_expected_requests:
        targets.append(f"{stats.total_expected_requests:,} requests")
    if stats.expected_duration_sec:
        targets.append(f"{stats.expected_duration_sec:.1f}s duration")
    if stats.expected_num_sessions:
        targets.append(f"{stats.expected_num_sessions:,} sessions")
    if targets:
        parts.append(f"target: {', '.join(targets)}")
    return " | ".join(parts)


def format_phase_sending_complete(stats: CreditPhaseStats) -> str:
    """Format a concise log message for phase sending complete."""
    parts = [f"Phase {stats.phase} sending complete"]
    parts.append(
        f"sent={stats.requests_sent:,}, "
        f"completed={stats.requests_completed:,}, "
        f"in_flight={stats.in_flight_requests:,}"
    )
    if stats.sent_sessions > 0:
        parts.append(
            f"sessions: sent={stats.sent_sessions:,}, "
            f"completed={stats.completed_sessions:,}"
        )
    if stats.timeout_triggered:
        parts.append("timeout_triggered=True")
    return " | ".join(parts)


def format_phase_complete(stats: CreditPhaseStats) -> str:
    """Format a concise log message for phase complete."""
    parts = [f"Phase {stats.phase} complete"]
    parts.append(
        f"completed={stats.final_requests_completed:,}, "
        f"cancelled={stats.final_requests_cancelled:,}, "
        f"errors={stats.final_request_errors:,}"
    )
    if stats.final_sent_sessions and stats.final_sent_sessions > 0:
        parts.append(
            f"sessions: completed={stats.final_completed_sessions:,}, "
            f"cancelled={stats.final_cancelled_sessions:,}"
        )
    elapsed = stats.requests_elapsed_time
    parts.append(f"elapsed={elapsed:.2f}s")
    if stats.grace_period_timeout_triggered:
        parts.append("grace_period_timeout=True")
    if stats.was_cancelled:
        parts.append("was_cancelled=True")
    return " | ".join(parts)

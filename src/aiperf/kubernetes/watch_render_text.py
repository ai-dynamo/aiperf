# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-text renderer for aiperf kube watch.

Emits simple log lines to the console. No TUI, no JSON -- just readable
status updates suitable for CI logs, piped scripts, or terminal sessions
that don't support Rich.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.kubernetes.console import logger

if TYPE_CHECKING:
    from aiperf.kubernetes.watch_models import WatchSnapshot

# Metrics shown each refresh, in order: (field_name, label, unit)
_KEY_METRICS: list[tuple[str, str, str]] = [
    ("request_throughput_rps", "Throughput", "req/s"),
    ("request_latency_avg_ms", "Latency avg", "ms"),
    ("ttft_avg_ms", "TTFT avg", "ms"),
    ("output_token_throughput_tps", "Output tokens", "tok/s"),
]


class TextRenderer:
    """Renders WatchSnapshot as plain log lines."""

    def __init__(self) -> None:
        self._prev_phase: str | None = None
        self._prev_progress_pct: float | None = None
        self._prev_workers: tuple[int, int] | None = None

    def render(self, snapshot: WatchSnapshot) -> None:
        """Emit one round of status lines for the current snapshot."""
        elapsed = _fmt_duration(snapshot.elapsed_seconds)
        phase = snapshot.phase
        current = snapshot.current_phase or phase

        if phase != self._prev_phase:
            logger.info(f"[{elapsed}] Phase: {phase}")
            self._prev_phase = phase

        p = snapshot.progress
        if p and p.requests_total > 0:
            pct = p.percent
            eta = f"  ETA: {_fmt_duration(p.eta_seconds)}" if p.eta_seconds else ""
            if pct != self._prev_progress_pct:
                logger.info(
                    f"[{elapsed}] {current} {p.requests_completed}/{p.requests_total}"
                    f" ({pct:.0f}%){eta}"
                )
                self._prev_progress_pct = pct

        # Workers (only print when count changes)
        w = snapshot.workers
        workers_state = (w.ready, w.total)
        if workers_state != self._prev_workers:
            logger.info(f"[{elapsed}] Workers: {w.ready}/{w.total} ready")
            self._prev_workers = workers_state

        # Key metrics
        m = snapshot.metrics
        if m and m.request_count > 0:
            parts = []
            for attr, label, unit in _KEY_METRICS:
                val = getattr(m, attr, 0.0)
                if val:
                    parts.append(f"{label}: {val:.1f}{unit}")
            if parts:
                logger.info(f"[{elapsed}] {' | '.join(parts)}")

            if m.error_count:
                logger.info(f"[{elapsed}] Errors: {m.error_count}")

        # Diagnosis issues
        if snapshot.diagnosis.issues:
            for issue in snapshot.diagnosis.issues:
                logger.info(
                    f"[{elapsed}] [{issue.severity}] {issue.title}: {issue.detail}"
                )

    def start(self) -> None:
        """Log header."""
        logger.info("Watching benchmark (text mode, Ctrl+C to detach)...")
        logger.info("")

    def stop(self) -> None:
        """No-op."""


def _fmt_duration(seconds: float | None) -> str:
    """Format seconds as Xm Ys or Xs."""
    if seconds is None or seconds < 0:
        return "0s"
    s = int(seconds)
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"

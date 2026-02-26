# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

from tqdm import tqdm

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.hooks import (
    on_profiling_progress,
    on_records_progress,
    on_stop,
    on_warmup_progress,
)
from aiperf.common.mixins import CombinedPhaseStats
from aiperf.ui.base_ui import BaseAIPerfUI

_logger = AIPerfLogger(__name__)


class ProgressBar:
    """A progress bar that can be updated with a progress percentage."""

    BAR_FORMAT_WITH_TOTAL = (
        "{desc}: {n:,.0f}/{total:,} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]"
    )
    BAR_FORMAT_PERCENT_ONLY = (
        "{desc}: {percentage:3.0f}% |{bar}| [{elapsed}<{remaining}]"
    )

    def __init__(
        self,
        desc: str,
        color: str,
        position: int,
        total: int | None = None,
        **kwargs,
    ):
        self.has_known_total = total is not None
        effective_total = total if total is not None else 100
        bar_format = (
            self.BAR_FORMAT_WITH_TOTAL
            if self.has_known_total
            else self.BAR_FORMAT_PERCENT_ONLY
        )

        self.bar = tqdm(
            total=effective_total,
            desc=desc,
            colour=color,
            position=position,
            leave=False,
            dynamic_ncols=False,
            bar_format=bar_format,
            **kwargs,
        )
        self.total = effective_total
        self.update_threshold = Environment.UI.MIN_UPDATE_PERCENT
        self.last_percent = 0.0

    def update(self, progress: int):
        """Update the progress bar with a new progress percentage."""
        if self.bar.disable:
            return
        if progress is None or not self.total:
            return
        # Cap progress to total to prevent tqdm from setting total=None internally
        progress = min(progress, self.total)
        pct = (progress / self.total) * 100.0
        if pct >= self.last_percent + self.update_threshold:
            # Use bar.n (tqdm's actual value) to calculate delta to avoid race conditions
            # where multiple updates arrive before last_percent is updated
            delta = progress - self.bar.n
            if delta <= 0:
                # Already at or past this progress, just update our tracking state
                self.last_percent = pct
                return
            self.bar.update(delta)
            self.last_percent = pct
            self.bar.refresh()

    def disable(self):
        """Disable the progress bar."""
        # NOTE: Closing the progress bar can cause a deadlock from the tqdm library randomly,
        # so we just disable it instead.
        self.bar.disable = True


class TQDMProgressUI(BaseAIPerfUI):
    """A UI that shows progress bars for the records and requests phases."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bars: dict[str, ProgressBar] = {}
        self._stopped = False

    def _disable_all_bars_sync(self):
        """Disable all progress bars (sync helper)."""
        bars_to_close = list(self._bars.values())
        self._bars.clear()  # Clear immediately to prevent further updates
        for bar in bars_to_close:
            if bar is not None:
                bar.disable()

    @on_stop
    async def _disable_all_bars(self):
        """Disable all progress bars and clean up tqdm's semaphore."""
        self._disable_all_bars_sync()
        self._unregister_tqdm_semaphore()

    @staticmethod
    def _unregister_tqdm_semaphore() -> None:
        """Unregister tqdm's internal multiprocessing lock from the resource tracker.

        tqdm lazily creates a multiprocessing.RLock (TRLock.mp_lock) for thread-safe
        output. This semaphore leaks when the process exits via os._exit() since GC
        is skipped.
        """
        from aiperf.common.resource_tracker import unregister_semaphore

        lock = getattr(tqdm, "_lock", None)
        if lock is not None:
            mp_lock = getattr(lock, "mp_lock", None)
            if mp_lock is not None:
                unregister_semaphore(mp_lock)

    def _create_or_update_requests_bar(
        self,
        key: str,
        color: str,
        stats: CombinedPhaseStats,
        callback: Callable[
            [CombinedPhaseStats], tuple[int | None, int | None, float | None]
        ],
    ):
        if self.stop_requested:
            return  # Don't create/update bars after stop
        expected, finished, pct = callback(stats)
        if key not in self._bars:
            # Disable all existing bars to avoid overlap and interleaving of bars.
            self._disable_all_bars_sync()
            # Pass None when expected count is unknown to show percentage-only format
            self._bars[key] = ProgressBar(
                desc=key,
                color=color,
                position=0,
                total=expected,
            )
        else:
            try:
                # If expected is not None, use the finished count as the progress, otherwise use the percentage.
                value = finished if expected is not None else pct
                if value is not None:
                    bar = self._bars[key]
                    bar.update(value)
            except Exception as e:
                _logger.error(f"Error updating progress bar {key}: {e!r}")
                _logger.error(stats)

    GRACE_PERIOD_COLORS = {
        CreditPhase.WARMUP: "cyan",
        CreditPhase.PROFILING: "magenta",
    }
    PHASE_COLORS = {
        CreditPhase.WARMUP: "yellow",
        CreditPhase.PROFILING: "green",
    }
    RECORDS_COLOR = "blue"

    def _on_phase_progress(self, stats: CombinedPhaseStats):
        """Callback for phase progress updates."""
        if stats.timeout_triggered:
            self._create_or_update_requests_bar(
                f"{stats.phase.title()} Grace Period",
                self.GRACE_PERIOD_COLORS[stats.phase],
                stats,
                lambda stats: (
                    stats.requests_sent,
                    stats.requests_completed + stats.requests_cancelled,
                    None,
                ),
            )
        else:
            self._create_or_update_requests_bar(
                f"{stats.phase.title()}",
                self.PHASE_COLORS[stats.phase],
                stats,
                lambda stats: (
                    stats.total_expected_requests,
                    stats.requests_completed,
                    stats.requests_progress_percent,
                ),
            )

    @on_warmup_progress
    def _on_warmup_progress(self, warmup_stats: CombinedPhaseStats):
        """Callback for warmup progress updates."""
        self._on_phase_progress(warmup_stats)

    @on_profiling_progress
    def _on_profiling_progress(self, profiling_stats: CombinedPhaseStats):
        """Callback for profiling progress updates."""
        self._on_phase_progress(profiling_stats)

    @on_records_progress
    def _on_records_progress(self, records_stats: CombinedPhaseStats):
        """Callback for records progress updates."""
        if records_stats.final_requests_completed is None:
            return
        self._create_or_update_requests_bar(
            "Processing Records",
            self.RECORDS_COLOR,
            records_stats,
            lambda stats: (
                stats.final_requests_completed,
                stats.total_records,
                stats.records_progress_percent,
            ),
        )

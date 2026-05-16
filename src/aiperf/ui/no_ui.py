# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_phase_progress, on_records_progress
from aiperf.common.mixins import CombinedPhaseStats
from aiperf.ui.base_ui import BaseAIPerfUI

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun

_logger = AIPerfLogger(__name__)


class NoUI(BaseAIPerfUI):
    """Headless UI that periodically logs benchmark progress.

    Subscribes to phase and records progress via :class:`BaseAIPerfUI` and
    emits a compact status line every ``AIPERF_UI_STATUS_LOG_INTERVAL`` seconds
    (default 30 s).
    """

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(run=run, **kwargs)
        self._last_phase_stats: CombinedPhaseStats | None = None
        self._last_records_stats: CombinedPhaseStats | None = None
        self._phase_complete = False

    @on_phase_progress
    def _on_phase_progress(self, phase_stats: CombinedPhaseStats) -> None:
        self._last_phase_stats = phase_stats
        self._phase_complete = phase_stats.is_requests_complete

    @on_records_progress
    def _on_records_progress(self, records_stats: CombinedPhaseStats) -> None:
        self._last_records_stats = records_stats

    @background_task(
        interval=lambda self: Environment.UI.STATUS_LOG_INTERVAL,
        immediate=False,
    )
    async def _periodic_status_log(self) -> None:
        if self._phase_complete and self._last_records_stats is not None:
            self._log_records_status(self._last_records_stats)
        elif self._last_phase_stats is not None:
            self._log_phase_status(self._last_phase_stats)

    def _log_phase_status(self, s: CombinedPhaseStats) -> None:
        parts: list[str] = [s.phase.title()]

        if s.total_expected_requests is not None:
            parts.append(
                f"{s.requests_completed:,}/{s.total_expected_requests:,} requests"
            )
        else:
            parts.append(f"{s.requests_completed:,} requests")

        if s.requests_progress_percent is not None:
            parts.append(f"{s.requests_progress_percent:.1f}%")

        if s.requests_per_second is not None:
            parts.append(f"{s.requests_per_second:,.1f} req/s")

        if s.request_errors:
            parts.append(f"{s.request_errors:,} errors")

        if s.requests_eta_sec is not None and s.requests_eta_sec > 0:
            parts.append(f"ETA {s.requests_eta_sec:.0f}s")

        _logger.info(" | ".join(parts))

    def _log_records_status(self, s: CombinedPhaseStats) -> None:
        parts: list[str] = ["Processing records"]

        total = s.total_records
        expected = s.final_requests_completed
        if expected is not None:
            parts.append(f"{total:,}/{expected:,}")
        else:
            parts.append(f"{total:,}")

        if s.records_progress_percent is not None:
            parts.append(f"{s.records_progress_percent:.1f}%")

        if s.records_per_second is not None:
            parts.append(f"{s.records_per_second:,.1f} rec/s")

        if s.error_records:
            parts.append(f"{s.error_records:,} errors")

        if s.records_eta_sec is not None and s.records_eta_sec > 0:
            parts.append(f"ETA {s.records_eta_sec:.0f}s")

        _logger.info(" | ".join(parts))

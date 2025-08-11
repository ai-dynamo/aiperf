# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tqdm import tqdm

from aiperf.common.constants import DEFAULT_UI_MIN_UPDATE_PERCENT
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import AIPerfUIType, CreditPhase
from aiperf.common.factories import AIPerfUIFactory
from aiperf.common.hooks import (
    on_records_progress,
    on_requests_phase_progress,
    on_stop,
)
from aiperf.common.models import RecordsStats, RequestsStats
from aiperf.common.protocols import AIPerfUIProtocol
from aiperf.ui.base_ui import BaseAIPerfUI


class ProgressBar:
    """A progress bar that can be updated with a progress percentage."""

    def __init__(
        self,
        desc: str,
        color: str,
        position: int,
        total: int,
        **kwargs,
    ):
        self.bar = tqdm(
            total=total,
            desc=desc,
            colour=color,
            position=position,
            leave=False,
            dynamic_ncols=False,
            unit=" pct",
            bar_format="{desc}: {n:,.0f}/{total:,} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]",
            **kwargs,
        )
        self.total = total
        self.update_threshold = DEFAULT_UI_MIN_UPDATE_PERCENT
        self.last_percent = 0.0
        self.last_value = 0.0

    def update(self, progress: int):
        """Update the progress bar with a new progress percentage."""
        pct = (progress / self.total) * 100.0
        if pct >= self.last_percent + self.update_threshold:
            self.bar.update(progress - self.last_value)
            self.last_percent = pct
            self.last_value = progress

    def close(self):
        """Close the progress bar."""
        self.bar.close()


@implements_protocol(AIPerfUIProtocol)
@AIPerfUIFactory.register(AIPerfUIType.SIMPLE)
class TQDMProgressUI(BaseAIPerfUI):
    """A UI that shows progress bars for the records and requests phases."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._records_bar: ProgressBar | None = None
        self._requests_bar: ProgressBar | None = None
        self._warmup_bar: ProgressBar | None = None

    @on_stop
    def _close_all_bars(self):
        """Close all progress bars."""
        for bar in [self._records_bar, self._requests_bar, self._warmup_bar]:
            if bar is not None:
                bar.close()

    @on_requests_phase_progress
    def _on_requests_phase_progress(
        self, phase: CreditPhase, requests_stats: RequestsStats
    ):
        """Callback for requests phase progress updates."""
        if phase == CreditPhase.WARMUP:
            if self._warmup_bar is None and requests_stats.finished is not None:
                self._warmup_bar = ProgressBar(
                    desc="Warmup",
                    color="yellow",
                    position=0,
                    total=requests_stats.total_expected_requests or 100,
                )

            if self._warmup_bar:
                self._warmup_bar.update(requests_stats.finished)  # type: ignore

        elif phase == CreditPhase.PROFILING:
            if self._requests_bar is None and requests_stats.finished is not None:
                self._requests_bar = ProgressBar(
                    desc="Requests (Profiling)",
                    color="green",
                    position=1,
                    total=requests_stats.total_expected_requests or 100,
                )

            if self._requests_bar:
                self._requests_bar.update(requests_stats.finished)  # type: ignore

    @on_records_progress
    def _on_records_progress(self, records_stats: RecordsStats):
        """Callback for records progress updates."""
        if self._records_bar is None and records_stats.finished is not None:
            self._records_bar = ProgressBar(
                desc="Records (Processing)",
                color="blue",
                position=2,
                total=records_stats.total_expected_requests or 100,
            )

        if self._records_bar:
            self._records_bar.update(records_stats.finished)  # type: ignore

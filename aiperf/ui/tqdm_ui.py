# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tqdm import tqdm

from aiperf.common.constants import DEFAULT_UI_MIN_UPDATE_PERCENT
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import AIPerfUIType
from aiperf.common.factories import AIPerfUIFactory
from aiperf.common.hooks import (
    on_profiling_progress,
    on_records_progress,
    on_stop,
    on_warmup_progress,
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
        self._warmup_bar: ProgressBar | None = None
        self._requests_bar: ProgressBar | None = None
        self._records_bar: ProgressBar | None = None

    @on_stop
    def _close_all_bars(self):
        """Close all progress bars."""
        for bar in [self._records_bar, self._requests_bar, self._warmup_bar]:
            if bar is not None:
                bar.close()

    @on_warmup_progress
    def _on_warmup_progress(self, warmup_stats: RequestsStats):
        """Callback for warmup progress updates."""
        if self._warmup_bar is None and warmup_stats.total_expected_requests:
            self._warmup_bar = ProgressBar(
                desc="Warmup",
                color="yellow",
                position=0,
                total=warmup_stats.total_expected_requests,
            )
        if self._warmup_bar:
            self._warmup_bar.update(warmup_stats.finished)

    @on_profiling_progress
    def _on_profiling_progress(self, profiling_stats: RequestsStats):
        """Callback for profiling progress updates."""
        if self._requests_bar is None and profiling_stats.total_expected_requests:
            self._requests_bar = ProgressBar(
                desc="Requests (Profiling)",
                color="green",
                position=1,
                total=profiling_stats.total_expected_requests,
            )
        if self._requests_bar:
            self._requests_bar.update(profiling_stats.finished)

    @on_records_progress
    def _on_records_progress(self, records_stats: RecordsStats):
        """Callback for records progress updates."""
        if self._records_bar is None and records_stats.total_expected_requests:
            self._records_bar = ProgressBar(
                desc="Records (Processing)",
                color="blue",
                position=2,
                total=records_stats.total_expected_requests,
            )
        if self._records_bar:
            self._records_bar.update(records_stats.finished)

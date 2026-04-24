# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricResult

if TYPE_CHECKING:
    from aiperf.common.metric_records_wire import MetricRecordsData
    from aiperf.config import BenchmarkRun


class AccuracyResultsProcessor(AIPerfLifecycleMixin):
    """Results processor for accuracy benchmarking.

    Aggregates accuracy grading results and computes summary metrics.
    Self-disables when accuracy mode is not enabled.
    """

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        config = run.cfg
        if not (config.accuracy and config.accuracy.enabled):
            raise PostProcessorDisabled(
                "Accuracy results processor is disabled: accuracy mode is not enabled"
            )

        super().__init__(run=run, **kwargs)
        self.run = run

    async def process_result(self, record_data: MetricRecordsData) -> None:
        raise NotImplementedError

    async def summarize(self) -> list[MetricResult]:
        raise NotImplementedError

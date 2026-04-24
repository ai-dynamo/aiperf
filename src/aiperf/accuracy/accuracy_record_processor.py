# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.metric_records_wire import MetricRecordMetadata
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricRecordDict

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class AccuracyRecordProcessor(AIPerfLifecycleMixin):
    """Record processor for accuracy benchmarking.

    Computes per-record accuracy metrics by grading LLM responses against ground truth.
    Self-disables when accuracy mode is not enabled.
    """

    def __init__(
        self,
        service_id: str | None,
        run: BenchmarkRun,
        **kwargs,
    ) -> None:
        config = run.cfg
        if not (config.accuracy and config.accuracy.enabled):
            raise PostProcessorDisabled(
                "Accuracy record processor is disabled: accuracy mode is not enabled"
            )

        super().__init__(service_id=service_id, run=run, **kwargs)
        self.run = run

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> MetricRecordDict:
        raise NotImplementedError

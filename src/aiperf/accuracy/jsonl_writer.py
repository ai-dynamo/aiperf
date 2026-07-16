# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.accuracy.models import AccuracyRecordsData
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models import MetricResult
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class AccuracyJSONLWriter(
    BaseMetricsProcessor, BufferedJSONLWriterMixin[AccuracyRecordsData]
):
    """Exports per-record graded accuracy data to JSONL files.

    Streams each ``AccuracyRecordsData`` as it arrives, writing one JSON line per
    graded response. Each line carries the grade (pass/unparsed/confidence), the
    expected/actual answers, and the grader's reasoning, enabling per-response
    post-hoc analysis.
    """

    # ``record_type`` is a wire-only discriminator (needed to reconstruct the
    # record across the ZMQ boundary on the generic RecordsMessage) -- exclude it
    # from accuracy_export.jsonl so the on-disk output is byte-identical.
    _jsonl_exclude_fields = {"record_type"}

    def __init__(
        self,
        run: BenchmarkRun,
        **kwargs,
    ) -> None:
        if run.cfg.accuracy is None or not run.cfg.accuracy.enabled:
            raise PostProcessorDisabled(
                "Accuracy JSONL export is disabled: accuracy mode is not enabled"
            )

        output_file = run.cfg.artifacts.accuracy_export_jsonl_file

        super().__init__(
            run=run,
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            **kwargs,
        )

        self.info(f"Accuracy JSONL export enabled: {self.output_file}")

    async def process_record(self, record: AccuracyRecordsData) -> None:
        """Write a single graded accuracy record to the JSONL buffer."""
        await self.buffered_write(record)

    async def finalize(self) -> None:
        """Flush any buffered data at end-of-run (``StreamExporterProtocol``)."""
        await self.flush_buffer()

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results. This writer streams records, so returns []."""
        return []

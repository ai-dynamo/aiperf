# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from aiperf.accuracy.models import (
    AccuracyRecordsData,
    AccuracySummary,
    TaskAccuracyStats,
)
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import ExportContext, SummaryContext
    from aiperf.common.enums import CreditPhase
    from aiperf.config.resolution.plan import BenchmarkRun


class AccuracyAccumulator(BaseMetricsProcessor):
    """Accumulate graded accuracy records on the dedicated ``accuracy`` channel.

    Ingests per-graded-response ``AccuracyRecordsData`` and rolls them up into an
    ``AccuracySummary`` (overall + per-task pass rates and unparsed counts).
    Because each record carries its own ``task`` label, this accumulator needs no
    ``on_dataset_configured`` hook — task bucketing is read straight off the
    record.

    Phase scoping mirrors ``ServerMetricsAccumulator``: ``export_results(ctx)``
    filters records to ``ctx.phase`` so warmup grades never leak into the
    profiling summary (and vice versa).

    Raises:
        PostProcessorDisabled: When accuracy mode is not enabled.
    """

    # RecordsManager routes phase-scoped-export accumulators through
    # export_results(ctx) instead of the unscoped summarize().
    supports_phase_scoped_export = True

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        if run.cfg.accuracy is None or not run.cfg.accuracy.enabled:
            raise PostProcessorDisabled(
                "Accuracy accumulator is disabled: accuracy mode is not enabled"
            )
        super().__init__(run=run, **kwargs)
        self.run = run
        self._records: list[AccuracyRecordsData] = []

    async def process_record(self, record: AccuracyRecordsData) -> None:
        """Append a graded record in arrival order."""
        self._records.append(record)

    def query_time_range(self, start_ns: int, end_ns: int) -> list[AccuracyRecordsData]:
        """Return records whose ``timestamp_ns`` is in ``[start_ns, end_ns)``.

        Analyzer query surface. Filters directly rather than bisecting: records
        arrive interleaved from multiple record processors, so ``_records`` is not
        globally sorted by ``timestamp_ns`` and a binary search would slice wrong.
        """
        return [r for r in self._records if start_ns <= r.timestamp_ns < end_ns]

    def _build_summary(self, phase: CreditPhase | None) -> AccuracySummary | None:
        """Roll scoped records into an ``AccuracySummary`` (None when empty).

        ``phase`` selects one phase; ``None`` is phase-agnostic (all records).
        Records with ``task is None`` count toward the overall totals but are
        absent from ``per_task``.
        """
        scoped = (
            self._records
            if phase is None
            else [r for r in self._records if r.benchmark_phase == phase]
        )
        if not scoped:
            return None

        total_evaluated = len(scoped)
        total_passed = sum(1 for r in scoped if r.passed)
        overall_unparsed = sum(1 for r in scoped if r.unparsed)
        accuracy_rate = total_passed / total_evaluated if total_evaluated else 0.0

        task_total: dict[str, int] = defaultdict(int)
        task_passed: dict[str, int] = defaultdict(int)
        task_unparsed: dict[str, int] = defaultdict(int)
        for r in scoped:
            if r.task is None:
                continue
            task_total[r.task] += 1
            if r.passed:
                task_passed[r.task] += 1
            if r.unparsed:
                task_unparsed[r.task] += 1

        per_task: dict[str, TaskAccuracyStats] = {}
        for task, total in task_total.items():
            passed = task_passed.get(task, 0)
            unparsed = task_unparsed.get(task, 0)
            per_task[task] = TaskAccuracyStats(
                total=total,
                passed=passed,
                unparsed=unparsed,
                accuracy_rate=passed / total if total else 0.0,
                unparsed_rate=unparsed / total if total else 0.0,
            )

        return AccuracySummary(
            total_evaluated=total_evaluated,
            total_passed=total_passed,
            accuracy_rate=accuracy_rate,
            overall_unparsed=overall_unparsed,
            grader_name=scoped[0].grader_name,
            per_task=per_task,
        )

    async def export_results(self, ctx: ExportContext) -> AccuracySummary | None:
        """Return an ``AccuracySummary`` scoped to ``ctx.phase`` (all if None).

        Returns None when no records fall in scope so RecordsManager can skip
        publishing, mirroring ``ServerMetricsAccumulator.export_results``.
        """
        return self._build_summary(ctx.phase)

    async def summarize(
        self, ctx: SummaryContext | None = None
    ) -> AccuracySummary | None:
        """Return the phase-agnostic (all-phase) accuracy summary."""
        return self._build_summary(phase=None)

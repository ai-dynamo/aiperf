# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from aiperf.accuracy.models import (
    ACCURACY_OVERALL_TAG,
    ACCURACY_RECORD_CORRECT_KEY,
    ACCURACY_RECORD_UNPARSED_KEY,
    ACCURACY_UNPARSED_TAG,
    accuracy_task_tag,
    accuracy_unparsed_task_tag,
)
from aiperf.common.enums import CreditPhase, MetricConsoleGroup
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricResult

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import ExportContext
    from aiperf.common.messages.inference_messages import MetricRecordsData
    from aiperf.common.models.dataset_models import DatasetMetadata
    from aiperf.config.resolution.plan import BenchmarkRun


class AccuracyResultsProcessor(AIPerfLifecycleMixin):
    """Results processor for accuracy benchmarking.

    Receives task names via on_dataset_configured (called by RecordsManager
    when DatasetConfiguredNotification arrives). Accumulates per-record grading
    results from AccuracyRecordProcessor, then summarizes into per-task and
    overall accuracy MetricResult objects.

    Counts are keyed by ``CreditPhase`` so that ``export_results(ctx)`` can scope
    a summary to a single phase (e.g. PROFILING), matching how MetricsAccumulator
    is phase-scoped. Without this, warmup grades would leak into the profiling
    accuracy summary (and vice versa) since every record increments the counters.
    """

    # RecordsManager routes phase-scoped-export accumulators through
    # export_results(ctx) instead of the unscoped summarize().
    supports_phase_scoped_export = True

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        acc_cfg = run.cfg.accuracy
        if acc_cfg is None or not acc_cfg.enabled:
            raise PostProcessorDisabled(
                "Accuracy results processor is disabled: accuracy mode is not enabled"
            )

        super().__init__(**kwargs)
        self.run = run

        self._tasks: list[str] | None = None
        # Per-phase counts: phase -> task -> count. Overall is a phase -> count.
        self._task_correct: dict[CreditPhase, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._task_total: dict[CreditPhase, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._task_unparsed: dict[CreditPhase, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._overall_correct: dict[CreditPhase, int] = defaultdict(int)
        self._overall_total: dict[CreditPhase, int] = defaultdict(int)
        self._overall_unparsed: dict[CreditPhase, int] = defaultdict(int)

    def on_dataset_configured(self, metadata: DatasetMetadata) -> None:
        """Receive task names from the DatasetConfiguredNotification.

        Called by RecordsManager before any records are processed. Builds the
        ordered list of task names from ConversationMetadata so that
        process_record can bucket records without re-loading the benchmark.
        """
        self._tasks = [
            c.accuracy_task
            for c in metadata.conversations
            if c.accuracy_task is not None
        ]

    async def process_record(self, record_data: MetricRecordsData) -> None:
        """Accumulate per-task accuracy counts from a single record's metrics.

        Reads ``accuracy_correct`` from ``record_data.metrics`` (produced by
        AccuracyRecordProcessor) and increments per-task and overall counters.
        Records missing the ``accuracy_correct`` key are silently skipped.

        Raises:
            RuntimeError: if on_dataset_configured was not called before processing.
        """
        if self._tasks is None:
            raise RuntimeError(
                "AccuracyResultsProcessor: dataset not configured; "
                "on_dataset_configured must be called before process_record"
            )
        metrics = record_data.metrics
        correct = metrics.get(ACCURACY_RECORD_CORRECT_KEY)
        if correct is None:
            return

        phase = record_data.metadata.benchmark_phase
        task = self._tasks[record_data.metadata.session_num % len(self._tasks)]
        is_correct = float(correct) >= 0.5
        is_unparsed = float(metrics.get(ACCURACY_RECORD_UNPARSED_KEY, 0.0)) >= 0.5

        self._overall_total[phase] += 1
        if is_correct:
            self._overall_correct[phase] += 1
        if is_unparsed:
            self._overall_unparsed[phase] += 1

        self._task_total[phase][task] += 1
        if is_correct:
            self._task_correct[phase][task] += 1
        if is_unparsed:
            self._task_unparsed[phase][task] += 1

    def _phase_scoped_counts(
        self, phase: CreditPhase | None
    ) -> tuple[int, int, int, dict[str, int], dict[str, int], dict[str, int]]:
        """Collapse the per-phase counters into a single scope.

        ``phase`` selects one phase; ``None`` sums across every phase (the
        phase-agnostic full-range view). Returns
        ``(overall_correct, overall_total, overall_unparsed,
        task_correct, task_total, task_unparsed)``.
        """
        phases = [phase] if phase is not None else list(self._overall_total.keys())

        overall_correct = sum(self._overall_correct.get(p, 0) for p in phases)
        overall_total = sum(self._overall_total.get(p, 0) for p in phases)
        overall_unparsed = sum(self._overall_unparsed.get(p, 0) for p in phases)

        task_correct: dict[str, int] = defaultdict(int)
        task_total: dict[str, int] = defaultdict(int)
        task_unparsed: dict[str, int] = defaultdict(int)
        for p in phases:
            for task, count in self._task_total.get(p, {}).items():
                task_total[task] += count
            for task, count in self._task_correct.get(p, {}).items():
                task_correct[task] += count
            for task, count in self._task_unparsed.get(p, {}).items():
                task_unparsed[task] += count

        return (
            overall_correct,
            overall_total,
            overall_unparsed,
            task_correct,
            task_total,
            task_unparsed,
        )

    def _build_results(self, phase: CreditPhase | None) -> list[MetricResult]:
        """Build accuracy MetricResults scoped to ``phase`` (or all phases).

        Emits:
        - ``accuracy.overall``: overall correct/total ratio
        - ``accuracy.task.<name>``: per-task correct/total ratio (sorted alphabetically)
        - ``accuracy.unparsed``: overall count of responses that required regex fallback
        - ``accuracy.unparsed.task.<name>``: per-task unparsed counts (sorted alphabetically)

        Returns an empty list if no records were processed in scope.
        """
        (
            overall_correct,
            overall_total,
            overall_unparsed,
            task_correct,
            task_total,
            task_unparsed,
        ) = self._phase_scoped_counts(phase)

        results: list[MetricResult] = []

        if overall_total > 0:
            results.append(
                MetricResult(
                    tag=ACCURACY_OVERALL_TAG,
                    header="Accuracy (Overall)",
                    unit="ratio",
                    count=overall_total,
                    current=overall_correct / overall_total,
                    sum=overall_correct,
                    # Rendered by the dedicated Accuracy Benchmark Results table,
                    # not the main LLM metrics table (a ratio has no avg/p99/etc,
                    # so it would show as a row of N/A there).
                    console_group=MetricConsoleGroup.NONE,
                )
            )

        for task in sorted(task_total.keys()):
            total = task_total[task]
            correct = task_correct.get(task, 0)
            results.append(
                MetricResult(
                    tag=accuracy_task_tag(task),
                    header=f"Accuracy ({task})",
                    unit="ratio",
                    count=total,
                    current=correct / total if total > 0 else 0.0,
                    sum=correct,
                    console_group=MetricConsoleGroup.NONE,
                )
            )

        if overall_total > 0:
            results.append(
                MetricResult(
                    tag=ACCURACY_UNPARSED_TAG,
                    header="Accuracy Unparsed (Overall)",
                    unit="ratio",
                    count=overall_total,
                    current=overall_unparsed / overall_total,
                    sum=overall_unparsed,
                    console_group=MetricConsoleGroup.NONE,
                )
            )

        for task in sorted(task_total.keys()):
            total = task_total[task]
            unparsed = task_unparsed.get(task, 0)
            results.append(
                MetricResult(
                    tag=accuracy_unparsed_task_tag(task),
                    header=f"Accuracy Unparsed ({task})",
                    unit="ratio",
                    count=total,
                    current=unparsed / total if total > 0 else 0.0,
                    sum=unparsed,
                    console_group=MetricConsoleGroup.NONE,
                )
            )

        return results

    async def summarize(self) -> list[MetricResult]:
        """Return phase-agnostic (all-phase) accuracy and unparsed counts.

        Prefer ``export_results(ctx)`` for phase-scoped summaries; this full-range
        view is retained for phase-agnostic callers.
        """
        return self._build_results(phase=None)

    async def export_results(self, ctx: ExportContext) -> list[MetricResult]:
        """Return accuracy counts scoped to ``ctx.phase`` (all phases if None).

        RecordsManager summarizes profiling and warmup separately; scoping here
        keeps warmup grades out of the profiling accuracy summary (and vice
        versa), matching MetricsAccumulator's phase-scoped export.
        """
        return self._build_results(phase=ctx.phase)

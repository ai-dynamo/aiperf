# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from aiperf.accuracy.models import (
    ACCURACY_OVERALL_TAG,
    ACCURACY_UNPARSED_TAG,
    accuracy_task_tag,
    accuracy_unparsed_task_tag,
)
from aiperf.common.config import UserConfig
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricResult

if TYPE_CHECKING:
    from aiperf.common.messages.inference_messages import MetricRecordsData
    from aiperf.common.models.dataset_models import DatasetMetadata


class AccuracyResultsProcessor(AIPerfLifecycleMixin):
    """Results processor for accuracy benchmarking.

    Receives task names via on_dataset_configured (called by RecordsManager
    when DatasetConfiguredNotification arrives). Accumulates per-record grading
    results from AccuracyRecordProcessor, then summarizes into per-task and
    overall accuracy MetricResult objects.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        if not user_config.accuracy.enabled:
            raise PostProcessorDisabled(
                "Accuracy results processor is disabled: accuracy mode is not enabled"
            )

        super().__init__(user_config=user_config, **kwargs)
        self.user_config = user_config

        self._tasks: list[str] | None = None
        self._task_correct: dict[str, int] = defaultdict(int)
        self._task_total: dict[str, int] = defaultdict(int)
        self._task_unparsed: dict[str, int] = defaultdict(int)
        self._overall_correct: int = 0
        self._overall_total: int = 0
        self._overall_unparsed: int = 0

    def on_dataset_configured(self, metadata: DatasetMetadata) -> None:
        """Receive task names from the DatasetConfiguredNotification.

        Called by RecordsManager before any records are processed. Builds the
        ordered list of task names from ConversationMetadata so that
        process_result can bucket results without re-loading the benchmark.
        """
        self._tasks = [
            c.accuracy_task
            for c in metadata.conversations
            if c.accuracy_task is not None
        ]

    async def process_result(self, record_data: MetricRecordsData) -> None:
        """Accumulate per-task accuracy counts from a single record's metrics.

        Reads ``accuracy.correct`` from ``record_data.metrics`` (produced by
        AccuracyRecordProcessor) and increments per-task and overall counters.
        Records missing the ``accuracy.correct`` key are silently skipped.

        Raises:
            RuntimeError: if on_dataset_configured was not called before processing.
        """
        if self._tasks is None:
            raise RuntimeError(
                "AccuracyResultsProcessor: dataset not configured; "
                "on_dataset_configured must be called before process_result"
            )
        metrics = record_data.metrics
        correct = metrics.get("accuracy.correct")
        if correct is None:
            return

        task = self._tasks[record_data.metadata.session_num % len(self._tasks)]
        is_correct = float(correct) >= 0.5
        is_unparsed = float(metrics.get("accuracy.unparsed", 0.0)) >= 0.5

        self._overall_total += 1
        if is_correct:
            self._overall_correct += 1
        if is_unparsed:
            self._overall_unparsed += 1

        self._task_total[task] += 1
        if is_correct:
            self._task_correct[task] += 1
        if is_unparsed:
            self._task_unparsed[task] += 1

    async def summarize(self) -> list[MetricResult]:
        """Return overall and per-task accuracy and unparsed counts as MetricResult list.

        Emits:
        - ``accuracy.overall``: overall correct/total ratio
        - ``accuracy.task.<name>``: per-task correct/total ratio (sorted alphabetically)
        - ``accuracy.unparsed``: overall count of responses that required regex fallback
        - ``accuracy.unparsed.task.<name>``: per-task unparsed counts (sorted alphabetically)

        Returns an empty list if no records were processed.
        """
        results: list[MetricResult] = []

        if self._overall_total > 0:
            overall_acc = self._overall_correct / self._overall_total
            results.append(
                MetricResult(
                    tag=ACCURACY_OVERALL_TAG,
                    header="Accuracy (Overall)",
                    unit="ratio",
                    count=self._overall_total,
                    current=overall_acc,
                    sum=self._overall_correct,
                )
            )

        for task in sorted(self._task_total.keys()):
            total = self._task_total[task]
            correct = self._task_correct[task]
            acc = correct / total if total > 0 else 0.0
            results.append(
                MetricResult(
                    tag=accuracy_task_tag(task),
                    header=f"Accuracy ({task})",
                    unit="ratio",
                    count=total,
                    current=acc,
                    sum=correct,
                )
            )

        if self._overall_total > 0:
            results.append(
                MetricResult(
                    tag=ACCURACY_UNPARSED_TAG,
                    header="Accuracy Unparsed (Overall)",
                    unit="ratio",
                    count=self._overall_total,
                    current=self._overall_unparsed / self._overall_total,
                    sum=self._overall_unparsed,
                )
            )

        for task in sorted(self._task_total.keys()):
            total = self._task_total[task]
            unparsed = self._task_unparsed.get(task, 0)
            results.append(
                MetricResult(
                    tag=accuracy_unparsed_task_tag(task),
                    header=f"Accuracy Unparsed ({task})",
                    unit="ratio",
                    count=total,
                    current=unparsed / total if total > 0 else 0.0,
                    sum=unparsed,
                )
            )

        return results

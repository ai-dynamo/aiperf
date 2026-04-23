# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

from aiperf.accuracy.benchmark_loader import load_benchmark_problems
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import UserConfig
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricResult

if TYPE_CHECKING:
    from aiperf.common.messages.inference_messages import MetricRecordsData


class AccuracyResultsProcessor(AIPerfLifecycleMixin):
    """Results processor for accuracy benchmarking.

    Loads benchmark problems to resolve task names via session_num.
    Accumulates per-record grading results from AccuracyRecordProcessor,
    then summarizes into per-task and overall accuracy MetricResult objects.
    """

    def __init__(self, user_config: UserConfig, **kwargs) -> None:
        if not user_config.accuracy.enabled:
            raise PostProcessorDisabled(
                "Accuracy results processor is disabled: accuracy mode is not enabled"
            )

        super().__init__(user_config=user_config, **kwargs)
        self.user_config = user_config

        self._problems: list[BenchmarkProblem] | None = None
        self._problems_lock = asyncio.Lock()
        self._task_correct: dict[str, int] = defaultdict(int)
        self._task_total: dict[str, int] = defaultdict(int)
        self._overall_correct: int = 0
        self._overall_total: int = 0

    async def _ensure_problems_loaded(self) -> None:
        if self._problems is not None:
            return
        async with self._problems_lock:
            if self._problems is None:
                self._problems = await load_benchmark_problems(self.user_config)

    async def process_result(self, record_data: MetricRecordsData) -> None:
        await self._ensure_problems_loaded()
        metrics = record_data.metrics
        correct = metrics.get("accuracy.correct")
        if correct is None:
            return

        task = self._problems[
            record_data.metadata.session_num % len(self._problems)
        ].task
        is_correct = float(correct) >= 0.5

        self._overall_total += 1
        if is_correct:
            self._overall_correct += 1

        self._task_total[task] += 1
        if is_correct:
            self._task_correct[task] += 1

    async def summarize(self) -> list[MetricResult]:
        results: list[MetricResult] = []

        if self._overall_total > 0:
            overall_acc = self._overall_correct / self._overall_total
            results.append(
                MetricResult(
                    tag="accuracy.overall",
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
                    tag=f"accuracy.task.{task}",
                    header=f"Accuracy ({task})",
                    unit="ratio",
                    count=total,
                    current=acc,
                    sum=correct,
                )
            )

        return results

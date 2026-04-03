# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import UserConfig
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricResult
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

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

        acc_cfg = user_config.accuracy
        self._benchmark_cls = plugins.get_class(
            PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark
        )
        self._n_shots = acc_cfg.n_shots
        if self._n_shots == 0:
            meta = plugins.get_metadata(
                PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark
            )
            default_n = meta.get("default_n_shots")
            if default_n is not None:
                self._n_shots = default_n

        self._problems: list[BenchmarkProblem] | None = None
        self._task_correct: dict[str, int] = defaultdict(int)
        self._task_total: dict[str, int] = defaultdict(int)
        self._overall_correct: int = 0
        self._overall_total: int = 0

    async def _ensure_problems_loaded(self) -> None:
        if self._problems is not None:
            return
        acc_cfg = self.user_config.accuracy
        benchmark = self._benchmark_cls(user_config=self.user_config)
        self._problems = await benchmark.load_problems(
            tasks=acc_cfg.tasks,
            n_shots=self._n_shots,
            enable_cot=acc_cfg.enable_cot,
        )

    async def process_result(self, record_data: MetricRecordsData) -> None:
        await self._ensure_problems_loaded()
        metrics = record_data.metrics
        correct = metrics.get("accuracy.correct")
        if correct is None:
            return

        idx = record_data.metadata.session_num
        task = (
            self._problems[idx].task
            if idx < len(self._problems)
            else "_unknown"
        )
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

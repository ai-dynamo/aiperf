# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiperf.accuracy.benchmark_loader import load_benchmark_problems
from aiperf.accuracy.models import BenchmarkProblem, GradingResult
from aiperf.common.config import UserConfig
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricRecordMetadata, ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.accuracy.graders.base import BaseGrader


class AccuracyRecordProcessor(AIPerfLifecycleMixin):
    """Record processor for accuracy benchmarking.

    Lazily loads benchmark problems on first process_record call, then grades
    each response against the corresponding ground truth. Maps each response to
    its problem via session_num % len(problems), supporting both single-pass and
    multi-pass runs.
    """

    def __init__(
        self,
        service_id: str | None,
        user_config: UserConfig,
        **kwargs,
    ) -> None:
        if not user_config.accuracy.enabled:
            raise PostProcessorDisabled(
                "Accuracy record processor is disabled: accuracy mode is not enabled"
            )

        super().__init__(service_id=service_id, user_config=user_config, **kwargs)
        self.user_config = user_config

        acc_cfg = user_config.accuracy
        benchmark_name = acc_cfg.benchmark
        grader_name = acc_cfg.grader

        if grader_name is None:
            meta = plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, benchmark_name)
            grader_name = meta.get("default_grader", "multiple_choice")

        grader_cls = plugins.get_class(PluginType.ACCURACY_GRADER, grader_name)
        self.grader: BaseGrader = grader_cls(user_config=user_config)

        self.problems: list[BenchmarkProblem] | None = None
        self._problems_lock = asyncio.Lock()

    async def _ensure_problems_loaded(self) -> None:
        if self.problems is not None:
            return
        async with self._problems_lock:
            if self.problems is None:
                problems = await load_benchmark_problems(self.user_config)
                if not problems:
                    acc_cfg = self.user_config.accuracy
                    msg = (
                        f"Benchmark '{acc_cfg.benchmark}' returned 0 problems "
                        f"(tasks={acc_cfg.tasks}, n_shots={acc_cfg.n_shots}). "
                        f"Check that --accuracy-tasks names a valid subtask "
                        f"(see docs/accuracy/accuracy_benchmarking.md) or omit "
                        f"the flag to evaluate all tasks."
                    )
                    self.error(msg)
                    raise ValueError(msg)
                self.problems = problems

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> MetricRecordDict:
        """Grade a single response against its corresponding benchmark problem.

        Maps ``metadata.session_num % len(self.problems)`` to a BenchmarkProblem,
        runs the configured grader, and returns a MetricRecordDict containing
        ``accuracy.correct`` (1.0 if correct, 0.0 otherwise).

        Raises:
            ValueError: if the benchmark returned 0 problems (e.g., bad --accuracy-tasks).
        """
        await self._ensure_problems_loaded()
        record_metrics = MetricRecordDict()

        problem = self.problems[metadata.session_num % len(self.problems)]
        response_text = self._extract_response_text(record)

        result: GradingResult = await self.grader.grade(
            response_text, problem.ground_truth
        )

        record_metrics["accuracy.correct"] = 1.0 if result.correct else 0.0

        return record_metrics

    @staticmethod
    def _extract_response_text(record: ParsedResponseRecord) -> str:
        parts: list[str] = []
        for resp in record.content_responses:
            if resp.data:
                text = resp.data.get_text()
                if text:
                    parts.append(text)
        return "".join(parts)

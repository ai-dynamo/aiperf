# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

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
    each response against the corresponding ground truth. Uses session_num
    from metadata to index into the problems list.
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

        self._benchmark_cls = plugins.get_class(
            PluginType.ACCURACY_BENCHMARK, benchmark_name
        )

        if grader_name is None:
            meta = plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, benchmark_name)
            grader_name = meta.get("default_grader", "multiple_choice")

        grader_cls = plugins.get_class(PluginType.ACCURACY_GRADER, grader_name)
        self.grader: BaseGrader = grader_cls(user_config=user_config)

        self._n_shots = acc_cfg.n_shots
        if self._n_shots == 0:
            meta = plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, benchmark_name)
            default_n = meta.get("default_n_shots")
            if default_n is not None:
                self._n_shots = default_n

        self.problems: list[BenchmarkProblem] | None = None

    async def _ensure_problems_loaded(self) -> None:
        if self.problems is not None:
            return
        acc_cfg = self.user_config.accuracy
        benchmark = self._benchmark_cls(user_config=self.user_config)
        self.problems = await benchmark.load_problems(
            tasks=acc_cfg.tasks,
            n_shots=self._n_shots,
            enable_cot=acc_cfg.enable_cot,
        )

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> MetricRecordDict:
        await self._ensure_problems_loaded()
        record_metrics = MetricRecordDict()

        idx = metadata.session_num
        if idx >= len(self.problems):
            return record_metrics

        problem = self.problems[idx]
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

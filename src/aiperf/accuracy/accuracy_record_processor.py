# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.accuracy.models import GradingResult
from aiperf.common.config import UserConfig
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricRecordMetadata, ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.accuracy.graders.base import BaseGrader
    from aiperf.common.models.dataset_models import DatasetMetadata


class AccuracyRecordProcessor(AIPerfLifecycleMixin):
    """Record processor for accuracy benchmarking.

    Receives ground-truth answers via on_dataset_configured (called by
    RecordProcessorService when DatasetConfiguredNotification arrives) and
    grades each response against the corresponding ground truth. Maps each
    response to its problem via session_num % len(_ground_truths), supporting
    both single-pass and multi-pass runs.
    """

    def __init__(
        self,
        service_id: str | None,
        user_config: UserConfig,
        **kwargs: Any,
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

        self._ground_truths: list[str] | None = None

    def on_dataset_configured(self, metadata: DatasetMetadata) -> None:
        """Receive ground-truth answers from the DatasetConfiguredNotification.

        Called by RecordProcessorService before any records are processed.
        Builds the ordered list of ground-truth answers from ConversationMetadata
        so that process_record can grade without re-loading the benchmark.
        """
        self._ground_truths = [
            c.accuracy_ground_truth
            for c in metadata.conversations
            if c.accuracy_ground_truth is not None
        ]

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> MetricRecordDict:
        """Grade a single response against its corresponding benchmark problem.

        Maps ``metadata.session_num % len(_ground_truths)`` to the ground-truth
        answer, runs the configured grader, and returns a MetricRecordDict
        containing ``accuracy.correct`` and ``accuracy.unparsed``.

        Raises:
            RuntimeError: if on_dataset_configured was not called before processing.
        """
        if not self._ground_truths:
            raise RuntimeError(
                "AccuracyRecordProcessor: dataset not configured; "
                "on_dataset_configured must be called before process_record"
            )
        record_metrics = MetricRecordDict()

        ground_truth = self._ground_truths[
            metadata.session_num % len(self._ground_truths)
        ]
        response_text = self._extract_response_text(record)

        result: GradingResult = await self.grader.grade(response_text, ground_truth)

        record_metrics["accuracy.correct"] = 1.0 if result.correct else 0.0
        record_metrics["accuracy.unparsed"] = 1.0 if result.unparsed else 0.0

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

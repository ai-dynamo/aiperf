# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.accuracy.models import (
    AccuracyRecordsData,
    GradingResult,
)
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricRecordMetadata, ParsedResponseRecord
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.accuracy.graders.base import BaseGrader
    from aiperf.common.models.dataset_models import DatasetMetadata
    from aiperf.config.resolution.plan import BenchmarkRun


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
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        acc_cfg = run.cfg.accuracy
        if acc_cfg is None or not acc_cfg.enabled:
            raise PostProcessorDisabled(
                "Accuracy record processor is disabled: accuracy mode is not enabled"
            )

        super().__init__(service_id=service_id, **kwargs)
        self.run = run

        benchmark_name = acc_cfg.benchmark
        grader_name = acc_cfg.grader

        if grader_name is None:
            meta = plugins.get_metadata(PluginType.ACCURACY_BENCHMARK, benchmark_name)
            grader_name = meta.get("default_grader", "multiple_choice")

        grader_cls = plugins.get_class(PluginType.ACCURACY_GRADER, grader_name)
        self.grader: BaseGrader = grader_cls(run=run)

        self._grader_name = grader_name
        self._verbose = acc_cfg.verbose
        self._ground_truths: list[str] | None = None
        self._tasks: list[str] | None = None

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
        self._tasks = [
            c.accuracy_task
            for c in metadata.conversations
            if c.accuracy_task is not None
        ]

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> AccuracyRecordsData:
        """Grade a single response against its corresponding benchmark problem.

        Maps ``metadata.session_num % len(_ground_truths)`` to the ground-truth
        answer, runs the configured grader, and returns a typed
        ``AccuracyRecordsData`` that flows on the dedicated ``accuracy`` channel.

        Raises:
            RuntimeError: if on_dataset_configured was not called before processing.
        """
        if not self._ground_truths:
            raise RuntimeError(
                "AccuracyRecordProcessor: dataset not configured; "
                "on_dataset_configured must be called before process_record"
            )

        ground_truth = self._ground_truths[
            metadata.session_num % len(self._ground_truths)
        ]
        response_text = self._extract_response_text(record)

        result: GradingResult = await self.grader.grade(response_text, ground_truth)

        task = (
            self._tasks[metadata.session_num % len(self._tasks)]
            if self._tasks
            else None
        )

        self._log_grading_detail(metadata.session_num, response_text, result)

        model_output, model_thinking = self._extract_output_and_thinking(record)

        return AccuracyRecordsData(
            session_num=metadata.session_num,
            conversation_id=metadata.conversation_id,
            x_request_id=metadata.x_request_id,
            worker_id=metadata.worker_id,
            benchmark_phase=metadata.benchmark_phase,
            timestamp_ns=metadata.request_end_ns,
            task=task,
            grader_name=self._grader_name,
            passed=result.correct,
            unparsed=result.unparsed,
            confidence=result.confidence,
            expected=result.ground_truth,
            actual=result.extracted_answer,
            reasoning=result.reasoning,
            model_output=model_output,
            model_thinking=model_thinking,
        )

    def _log_grading_detail(
        self, session_num: int, response_text: str, result: GradingResult
    ) -> None:
        """Surface per-problem grading diagnostics.

        Every grader fills in ``reasoning`` (why a response was graded
        correct/unparsed) and ``extracted_answer``, but only ``correct`` and
        ``unparsed`` reach the metrics. Without this, a run reporting 100%
        unparsed gives no clue whether the response was empty, the answer
        format didn't match, or grading raised an exception (e.g. LCB's
        sandboxed execution failing to fork from the daemon record processor).

        Emits at info level under ``--accuracy-verbose`` (the flag's
        documented "per-problem grading details") and always at debug, so the
        reason is recoverable from logs without re-running.
        """
        if not self._verbose and not self.is_debug_enabled:
            return

        def _detail() -> str:
            preview = response_text.strip().replace("\n", "\\n")[:200]
            return (
                f"[accuracy] session={session_num} correct={result.correct} "
                f"unparsed={result.unparsed} reason={result.reasoning!r} "
                f"extracted={result.extracted_answer[:120]!r} "
                f"response_len={len(response_text)} response_preview={preview!r}"
            )

        if self._verbose:
            self.info(_detail)
        else:
            self.debug(_detail)

    @staticmethod
    def _extract_response_text(record: ParsedResponseRecord) -> str:
        parts: list[str] = []
        for resp in record.content_responses:
            if resp.data:
                text = resp.data.get_text()
                if text:
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_output_and_thinking(
        record: ParsedResponseRecord,
    ) -> tuple[str, str | None]:
        """Split the response into visible answer content and reasoning/thinking.

        ``model_output`` is the answer channel (``TextResponseData.text`` or
        ``ReasoningResponseData.content``); ``model_thinking`` is the concatenated
        ``reasoning_content`` from any ``ReasoningResponseData`` chunks, or None
        when the model emitted no separate reasoning channel.
        """
        output_parts: list[str] = []
        thinking_parts: list[str] = []
        for resp in record.content_responses:
            data = resp.data
            if data is None:
                continue
            reasoning = getattr(data, "reasoning", None)
            if reasoning:
                thinking_parts.append(reasoning)
            content = getattr(data, "content", None)
            if content is not None:
                output_parts.append(content)
            elif reasoning is None:
                # Plain text (or tool-call) data with no reasoning channel.
                output_parts.append(data.get_text())
        thinking = "".join(thinking_parts) if thinking_parts else None
        return "".join(output_parts), thinking

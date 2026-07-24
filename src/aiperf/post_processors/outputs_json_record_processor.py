# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record observer that captures model response text and per-request metrics for outputs.json export."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from aiperf.common.enums import CreditPhase, MetricValueTypeT
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.config.artifacts import OutputDefaults
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.shard_writer import ShardWriterMixin

if TYPE_CHECKING:
    from aiperf.post_processors.record_observer_context import RecordObserverContext


class OutputFragment(AIPerfBaseModel):
    """A single output fragment capturing response text and request identifiers."""

    session_num: int = Field(ge=0, description="The session number of the request.")
    turn_index: int = Field(ge=0, description="The turn index within the conversation.")
    conversation_id: str = Field(description="The conversation identifier.")
    x_request_id: str = Field(description="The unique request identifier.")
    response_text: str | None = Field(
        default=None,
        description="The concatenated generated text from the model response.",
    )
    request_start_ns: int = Field(
        ge=0, description="Request start timestamp in nanoseconds."
    )
    request_end_ns: int = Field(
        ge=0, description="Request end timestamp in nanoseconds."
    )
    metrics: dict[str, MetricValueTypeT] = Field(
        default_factory=dict,
        description="Allowlisted per-request metrics in display units, captured "
        "at record-processing time so outputs.json does not depend on the "
        "records JSONL export being enabled.",
    )


class OutputsJsonRecordProcessor(
    ShardWriterMixin, BufferedJSONLWriterMixin[OutputFragment]
):
    """Captures model response text per request and writes fragment files.

    Enabled when --export-outputs-json is set. Writes per-processor fragment
    files that are later aggregated by the OutputsJsonExporter.
    """

    # Per-request metrics paired with each generated response in outputs.json.
    # Streaming-only metrics (TTFT, inter-token latency) are simply absent from
    # non-streaming records -- to_display_dict omits what the record lacks.
    _METRIC_ALLOWLIST = (
        "input_sequence_length",
        "output_token_count",
        "output_sequence_length",
        "request_latency",
        "time_to_first_token",
        "inter_token_latency",
    )

    def __init__(
        self,
        service_id: str | None,
        run: BenchmarkRun,
        **kwargs,
    ) -> None:
        self.cfg = run.cfg

        if not self.cfg.artifacts.export_outputs_json:
            raise PostProcessorDisabled(
                "OutputsJsonRecordProcessor is disabled (--export-outputs-json not set)"
            )

        output_file = self.shard_output_file(
            self.cfg.artifacts.dir,
            OutputDefaults.OUTPUT_FRAGMENTS_FOLDER,
            prefix="output_fragments",
            ext="jsonl",
            service_id=service_id,
        )

        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            service_id=service_id,
            cfg=self.cfg,
            **kwargs,
        )

        self.info(f"OutputsJsonRecordProcessor initialized: {self.output_file}")

    async def observe(self, ctx: RecordObserverContext) -> None:
        """Extract response text and allowlisted metrics, and write an output fragment."""
        record = ctx.record
        metadata = ctx.metadata
        if metadata.benchmark_phase != CreditPhase.PROFILING:
            return

        parts: list[str] = []
        for resp in record.content_responses:
            if resp.data:
                text = resp.data.get_text()
                if text:
                    parts.append(text)
        response_text = "".join(parts) or None

        # Capture the allowlisted metrics straight off the producer output, in the
        # same display units the records JSONL export uses (see RecordShardJSONLWriter).
        # This keeps outputs.json self-contained: metrics no longer require the
        # records JSONL to have been written, so --export-level summary (or a YAML
        # records: false) still yields fully-populated per-request metrics.
        metrics: dict[str, MetricValueTypeT] = {}
        if ctx.metrics is not None:
            display = MetricRecordDict(ctx.metrics.metrics).to_display_dict(
                MetricRegistry
            )
            for tag in self._METRIC_ALLOWLIST:
                metric_value = display.get(tag)
                if metric_value is not None:
                    metrics[tag] = metric_value.value

        fragment = OutputFragment(
            session_num=metadata.session_num,
            turn_index=metadata.turn_index or 0,
            conversation_id=metadata.conversation_id or "",
            x_request_id=metadata.x_request_id or "",
            response_text=response_text,
            request_start_ns=metadata.request_start_ns or 0,
            request_end_ns=metadata.request_end_ns or 0,
            metrics=metrics,
        )

        await self.buffered_write(fragment)

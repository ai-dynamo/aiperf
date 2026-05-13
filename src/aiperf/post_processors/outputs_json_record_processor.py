# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record processor that captures model response text per request for outputs.json export."""

from pydantic import Field

from aiperf.common.config import UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import BufferedJSONLWriterMixin
from aiperf.common.models import MetricRecordMetadata, ParsedResponseRecord
from aiperf.common.models.base_models import AIPerfBaseModel


class OutputFragment(AIPerfBaseModel):
    """A single output fragment capturing response text and request identifiers."""

    session_num: int = Field(description="The session number of the request.")
    turn_index: int = Field(description="The turn index within the conversation.")
    conversation_id: str = Field(description="The conversation identifier.")
    x_request_id: str = Field(description="The unique request identifier.")
    response_text: str | None = Field(
        default=None,
        description="The concatenated generated text from the model response.",
    )
    request_start_ns: int = Field(description="Request start timestamp in nanoseconds.")
    request_end_ns: int = Field(description="Request end timestamp in nanoseconds.")


class OutputsJsonRecordProcessor(BufferedJSONLWriterMixin[OutputFragment]):
    """Captures model response text per request and writes fragment files.

    Enabled when --export-outputs-json is set. Writes per-processor fragment
    files that are later aggregated by the OutputsJsonExporter.
    """

    def __init__(
        self,
        service_id: str | None,
        user_config: UserConfig,
        **kwargs,
    ) -> None:
        self.user_config = user_config

        if not self.user_config.output.export_outputs_json:
            raise PostProcessorDisabled(
                "OutputsJsonRecordProcessor is disabled (--export-outputs-json not set)"
            )

        output_dir = (
            user_config.output.artifact_directory
            / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_id = (
            (service_id or "processor")
            .replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )
        output_file = output_dir / f"output_fragments_{safe_id}.jsonl"

        # Clear own file from a previous failed run (safe: each processor has a unique ID)
        output_file.unlink(missing_ok=True)

        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            service_id=service_id,
            user_config=user_config,
            **kwargs,
        )

        self.info(f"OutputsJsonRecordProcessor initialized: {self.output_file}")

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> None:
        """Extract response text and write an output fragment."""
        if metadata.benchmark_phase != CreditPhase.PROFILING:
            return

        parts: list[str] = []
        for resp in record.content_responses:
            if resp.data:
                text = resp.data.get_text()
                if text:
                    parts.append(text)
        response_text = "".join(parts) or None

        fragment = OutputFragment(
            session_num=metadata.session_num,
            turn_index=metadata.turn_index or 0,
            conversation_id=metadata.conversation_id or "",
            x_request_id=metadata.x_request_id or "",
            response_text=response_text,
            request_start_ns=metadata.request_start_ns or 0,
            request_end_ns=metadata.request_end_ns or 0,
        )

        await self.buffered_write(fragment)

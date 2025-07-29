# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from aiperf.common.enums import (
    PostProcessorType,
    StreamingPostProcessorType,
)
from aiperf.common.enums.message_enums import MessageType
from aiperf.common.enums.timing_enums import CreditPhase
from aiperf.common.factories import PostProcessorFactory, StreamingPostProcessorFactory
from aiperf.common.hooks import on_message
from aiperf.common.messages.credit_messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseStartMessage,
)
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    ParsedResponseRecord,
)
from aiperf.common.models.record_models import ProfileResults
from aiperf.services.records_manager.post_processors.streaming_post_processor import (
    BaseStreamingPostProcessor,
)


@StreamingPostProcessorFactory.register(StreamingPostProcessorType.BASIC_METRICS)
class BasicMetricsStreamer(BaseStreamingPostProcessor):
    """Streamer for basic metrics."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.valid_count: int = 0
        self.error_count: int = 0
        self.start_time_ns: int = time.time_ns()
        self.error_summary: dict[ErrorDetails, int] = {}
        self.end_time_ns: int | None = None
        self.total_expected: int | None = None
        self.metric_summary = PostProcessorFactory.create_instance(
            PostProcessorType.METRIC_SUMMARY,
            endpoint_type=self.user_config.endpoint.type,
        )

    async def stream_record(self, record: ParsedResponseRecord) -> None:
        """Stream a record."""
        if record.request.valid:
            self.valid_count += 1
            self.metric_summary.process_record(record)
        else:
            self.error_count += 1
            self.warning(f"Received invalid inference results: {record.request.error}")
            if record.request.error is not None:
                self.error_summary.setdefault(record.request.error, 0)
                self.error_summary[record.request.error] += 1

    def get_error_summary(self) -> list[ErrorDetailsCount]:
        """Generate a summary of the error records."""
        return [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self.error_summary.items()
        ]

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(
        self, phase_start_msg: CreditPhaseStartMessage
    ) -> None:
        """Handle a credit phase start message."""
        if phase_start_msg.phase != CreditPhase.PROFILING:
            return
        self.start_time_ns = phase_start_msg.start_ns
        self.total_expected = phase_start_msg.total_expected_requests
        self.info(
            f"Credit phase start: {phase_start_msg.phase} with {self.total_expected} expected requests"
        )

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, phase_complete_msg: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message."""
        if phase_complete_msg.phase != CreditPhase.PROFILING:
            return
        self.end_time_ns = phase_complete_msg.end_ns
        if self.total_expected is None:
            self.total_expected = phase_complete_msg.completed

    async def process_records(
        self, cancelled: bool
    ) -> ProfileResults | ErrorDetails | None:
        """Process the records.

        This method is called when the records manager receives a command to process the records.
        """
        if self.valid_count + self.error_count == 0:
            self.warning("No records to process")
            return None

        self.notice("Processing records")
        try:
            self.info(
                f"Processing {self.valid_count} successful records and {self.error_count} error records"
            )
            return ProfileResults(
                total_expected=self.total_expected,
                completed=self.valid_count + self.error_count,
                start_ns=self.start_time_ns,
                end_ns=self.end_time_ns or time.time_ns(),
                records=self.metric_summary.post_process(),
                error_summary=self.get_error_summary(),
                was_cancelled=cancelled,
            )
        except Exception as e:
            self.exception(f"Error processing records: {e}")
            return ErrorDetails.from_exception(e)

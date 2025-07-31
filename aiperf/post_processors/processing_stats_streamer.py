# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from aiperf.common.enums import CreditPhase, MessageType, StreamingPostProcessorType
from aiperf.common.factories import StreamingPostProcessorFactory
from aiperf.common.hooks import background_task, on_message
from aiperf.common.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseStartMessage,
)
from aiperf.common.messages.progress_messages import (
    AllRecordsReceivedMessage,
    RecordsProcessingStatsMessage,
)
from aiperf.common.models import ParsedResponseRecord, PhaseProcessingStats
from aiperf.post_processors.streaming_post_processor import (
    BaseStreamingPostProcessor,
)


@StreamingPostProcessorFactory.register(StreamingPostProcessorType.PROCESSING_STATS)
class ProcessingStatsStreamer(BaseStreamingPostProcessor):
    """This streamer is used to track the number of records processed and the number of errors.
    It is also used to track the number of requests expected and the number of requests completed.
    It will send a notification message when all expected requests have been received.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.start_time_ns: int | None = None
        self.processing_stats: PhaseProcessingStats = PhaseProcessingStats()
        self.final_request_count: int | None = None
        self.end_time_ns: int | None = None

        # Track per-worker statistics
        self.worker_stats: dict[str, PhaseProcessingStats] = {}

    async def stream_record(self, record: ParsedResponseRecord) -> None:
        """Stream a record."""
        self.trace(lambda: f"Received parsed inference results: {record}")
        if record.request.credit_phase != CreditPhase.PROFILING:
            self.debug(
                lambda: f"Skipping non-profiling record: {record.request.credit_phase}"
            )
            return

        worker_id = record.worker_id
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = PhaseProcessingStats()

        if record.request.valid:
            self.worker_stats[worker_id].processed += 1
            self.processing_stats.processed += 1
        else:
            self.worker_stats[worker_id].errors += 1
            self.processing_stats.errors += 1

        if (
            self.final_request_count is not None
            and self.processing_stats.total_records >= self.final_request_count
        ):
            self.info(
                lambda: f"Processed {self.processing_stats.processed} valid requests and {self.processing_stats.errors} errors ({self.processing_stats.total_records} total)."
            )
            # Make sure everyone knows the final stats, including the worker stats
            await self.publish_processing_stats()

            # Send a message to the event bus to signal that we received all the records
            await self.publish(
                AllRecordsReceivedMessage(
                    service_id=self.service_id,
                    request_ns=time.time_ns(),
                    final_processing_stats=self.processing_stats,
                )
            )

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(
        self, phase_start_msg: CreditPhaseStartMessage
    ) -> None:
        """Handle a credit phase start message."""
        if phase_start_msg.phase == CreditPhase.PROFILING:
            self.processing_stats.total_expected_requests = (
                phase_start_msg.total_expected_requests
            )

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, phase_complete_msg: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message."""
        if phase_complete_msg.phase == CreditPhase.PROFILING:
            # This will equate to how many records we expect to receive,
            # and once we receive that many records, we know to stop.
            self.final_request_count = phase_complete_msg.completed
            self.end_time_ns = phase_complete_msg.end_ns
            self.info(f"Updating final request count to {self.final_request_count}")

    @background_task(
        interval=lambda self: self.service_config.progress_report_interval,
        immediate=False,
    )
    async def _report_records_task(self) -> None:
        """Report the records."""
        if self.processing_stats.processed > 0 or self.processing_stats.errors > 0:
            # Only publish stats if there are records to report
            await self.publish_processing_stats()

    async def publish_processing_stats(self) -> None:
        """Publish the profile processing stats."""
        await self.publish(
            RecordsProcessingStatsMessage(
                service_id=self.service_id,
                request_ns=time.time_ns(),
                processing_stats=self.processing_stats,
                worker_stats=self.worker_stats,
            ),
        )

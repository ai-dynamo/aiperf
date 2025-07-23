# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from aiperf.common.comms.base import SubClientProtocol
from aiperf.common.enums import CreditPhase, StreamingPostProcessorType
from aiperf.common.enums.message_enums import MessageType
from aiperf.common.factories import StreamingPostProcessorFactory
from aiperf.common.hooks import aiperf_auto_task, on_init
from aiperf.common.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseStartMessage,
)
from aiperf.common.messages.progress_messages import (
    AllRecordsReceivedMessage,
    RecordsProcessingStatsMessage,
)
from aiperf.common.models import ParsedResponseRecord, PhaseProcessingStats
from aiperf.services.records_manager.post_processors.streaming_post_processor import (
    BaseStreamingPostProcessor,
)


@StreamingPostProcessorFactory.register(StreamingPostProcessorType.PROCESSING_STATS)
class ProcessingStatsStreamer(BaseStreamingPostProcessor):
    """This streamer is used to track the number of records processed and the number of errors.
    It is also used to track the number of requests expected and the number of requests completed.
    It will send a notification message when all expected requests have been received.
    """

    def __init__(self, sub_client: SubClientProtocol, **kwargs) -> None:
        self.sub_client = sub_client
        super().__init__(sub_client=sub_client, **kwargs)
        self.processing_stats: PhaseProcessingStats = PhaseProcessingStats()
        self.final_request_count: int | None = None

        # Track per-worker statistics
        self.worker_stats: dict[str, PhaseProcessingStats] = {}

    @on_init
    async def _initialize(self) -> None:
        """Initialize the processing stats streamer."""
        _subscriptions = {
            MessageType.CREDIT_PHASE_START: self._on_credit_phase_start,
            MessageType.CREDIT_PHASE_COMPLETE: self._on_credit_phase_complete,
        }
        await self.sub_client.subscribe_all(_subscriptions)

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
            self.warning(f"Received invalid inference results: {record}")
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
            await self.pub_client.publish(
                AllRecordsReceivedMessage(
                    service_id=self.service_id,
                    request_ns=time.time_ns(),
                    final_processing_stats=self.processing_stats,
                )
            )

    async def _on_credit_phase_start(
        self, phase_start_msg: CreditPhaseStartMessage
    ) -> None:
        """Handle a credit phase start message."""
        if phase_start_msg.phase == CreditPhase.PROFILING:
            self.processing_stats.total_expected_requests = (
                phase_start_msg.total_expected_requests
            )

    async def _on_credit_phase_complete(
        self, phase_complete_msg: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message."""
        if phase_complete_msg.phase == CreditPhase.PROFILING:
            self.info(f"Updating final request count to {phase_complete_msg.completed}")
            # This will equate to how many records we expect to receive,
            # and once we receive that many records, we know to stop.
            self.final_request_count = phase_complete_msg.completed

    @aiperf_auto_task(
        interval_sec=lambda self: self.service_config.progress_report_interval_seconds
    )
    async def _report_records_task(self) -> None:
        """Report the records."""
        if self.processing_stats.processed > 0 or self.processing_stats.errors > 0:
            # Only publish stats if there are records to report
            await self.publish_processing_stats()

    async def publish_processing_stats(self) -> None:
        """Publish the profile processing stats."""
        await self.pub_client.publish(
            RecordsProcessingStatsMessage(
                service_id=self.service_id,
                request_ns=time.time_ns(),
                processing_stats=self.processing_stats,
                worker_stats=self.worker_stats,
            ),
        )

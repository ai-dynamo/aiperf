# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import sys
import time
from collections import deque

from aiperf.common.comms.base import (
    CommunicationClientAddressType,
    PullClientProtocol,
)
from aiperf.common.config import ServiceConfig, ServiceDefaults, UserConfig
from aiperf.common.enums import CommandType, CreditPhase, MessageType, ServiceType
from aiperf.common.factories import ServiceFactory, StreamingPostProcessorFactory
from aiperf.common.hooks import (
    aiperf_task,
    on_cleanup,
    on_init,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    CreditPhaseCompleteMessage,
    CreditPhaseStartMessage,
    ParsedInferenceResultsMessage,
    ProcessRecordsCommandData,
    ProfileResultsMessage,
    RecordsProcessingStatsMessage,
)
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    ParsedResponseRecord,
    PhaseProcessingStats,
)
from aiperf.common.service import BaseComponentService
from aiperf.data_exporter.exporter_manager import ExporterManager
from aiperf.services.records_manager.post_processors import BaseStreamingPostProcessor
from aiperf.services.records_manager.post_processors.metric_summary import MetricSummary

DEFAULT_MAX_RECORDS_CONCURRENCY = 100_000
"""The default maximum concurrency for the records manager pull client."""


@ServiceFactory.register(ServiceType.RECORDS_MANAGER)
class RecordsManager(BaseComponentService):
    """
    The RecordsManager service is primarily responsible for holding the
    results returned from the workers.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )
        self.user_config: UserConfig | None = None
        self.configured_event = asyncio.Event()

        # TODO: we do not want to keep all the data forever
        self.records: deque[ParsedResponseRecord] = deque()
        self.error_records: deque[ParsedResponseRecord] = deque()

        self.total_expected_requests: int | None = None
        self.error_records_count: int = 0
        self.records_count: int = 0
        self.final_request_count: int | None = None

        # Track per-worker statistics
        self.worker_success_counts: dict[str, int] = {}
        self.worker_error_counts: dict[str, int] = {}

        self.start_time_ns: int = time.time_ns()
        self.end_time_ns: int | None = None

        self.streaming_post_processors: list[BaseStreamingPostProcessor] = []

        self.response_results_client: PullClientProtocol = (
            self.comms.create_pull_client(
                CommunicationClientAddressType.RECORDS,
                bind=True,
            )
        )

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.RECORDS_MANAGER

    @on_init
    async def _initialize(self) -> None:
        """Initialize records manager-specific components."""
        self.debug("Initializing records manager")
        self.register_command_callback(
            CommandType.PROCESS_RECORDS,
            self.process_records,
        )
        await self.response_results_client.register_pull_callback(
            message_type=MessageType.PARSED_INFERENCE_RESULTS,
            callback=self._on_parsed_inference_results,
            max_concurrency=DEFAULT_MAX_RECORDS_CONCURRENCY,
        )

        await self.sub_client.subscribe(
            MessageType.CREDIT_PHASE_START,
            self._on_credit_phase_start,
        )

    @on_init
    async def _initialize_streaming_post_processors(self) -> None:
        """Initialize the streaming post processors and start their lifecycle."""
        for streamer_type in StreamingPostProcessorFactory.get_all_class_types():
            streamer = StreamingPostProcessorFactory.create_instance(
                class_type=streamer_type,
                pub_client=self.pub_client,
                sub_client=self.sub_client,
                service_id=self.service_id,
                service_config=self.service_config,
                user_config=self.user_config,
            )
            self.debug(f"Initializing streaming post processor: {streamer_type}")
            self.streaming_post_processors.append(streamer)
            self.debug(
                lambda streamer=streamer: f"Starting lifecycle for {streamer.__class__.__name__}"
            )
            await streamer.run_async()

    @on_stop
    async def _stop_streaming_post_processors(self) -> None:
        """Stop the streaming post processors."""
        await asyncio.gather(
            *[streamer.shutdown() for streamer in self.streaming_post_processors]
        )

    @on_cleanup
    async def _cleanup(self) -> None:
        """Cleanup the records manager."""
        await asyncio.gather(
            *[
                streamer.wait_for_shutdown()
                for streamer in self.streaming_post_processors
            ]
        )

    @aiperf_task
    async def _report_records_task(self) -> None:
        """Report the records."""
        while not self.stop_event.is_set():
            await asyncio.sleep(ServiceDefaults.PROGRESS_REPORT_INTERVAL_SECONDS)
            await self.publish_processing_stats()

    async def publish_processing_stats(self) -> None:
        """Publish the profile stats."""
        await self.pub_client.publish(
            RecordsProcessingStatsMessage(
                service_id=self.service_id,
                processing_stats=PhaseProcessingStats(
                    processed=self.records_count,
                    errors=self.error_records_count,
                    total_expected_requests=self.total_expected_requests,
                ),
                worker_stats={
                    worker_id: PhaseProcessingStats(
                        processed=self.worker_success_counts[worker_id],
                        errors=self.worker_error_counts[worker_id],
                    )
                    for worker_id in self.worker_success_counts
                },
                request_ns=time.time_ns(),
            ),
        )

    async def _on_credit_phase_start(self, message: CreditPhaseStartMessage) -> None:
        """Handle a credit phase start message."""
        if message.phase == CreditPhase.PROFILING:
            self.total_expected_requests = message.total_expected_requests

    async def _on_credit_phase_complete(
        self, message: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message."""
        if message.phase == CreditPhase.PROFILING:
            self.final_request_count = message.completed

    async def _on_parsed_inference_results(
        self, message: ParsedInferenceResultsMessage
    ) -> None:
        """Handle a parsed inference results message."""
        self.trace(lambda: f"Received parsed inference results: {message}")

        if message.record.request.credit_phase != CreditPhase.PROFILING:
            self.debug(
                lambda: f"Skipping non-profiling record: {message.record.request.credit_phase}"
            )
            return

        # Stream the record to all of the streaming post processors
        for streamer in self.streaming_post_processors:
            try:
                self.debug(
                    lambda name=streamer.__class__.__name__: f"Putting record into queue for streamer {name}"
                )
                streamer.records_queue.put_nowait(message.record)
            except asyncio.QueueFull:
                self.error(
                    f"Streaming post processor {streamer.__class__.__name__} is unable to keep up with the rate of incoming records."
                )
                self.warning(
                    f"Waiting for queue to be available for streamer {streamer.__class__.__name__}. This will cause back pressure on the records manager."
                )
                await streamer.records_queue.put(message.record)

        worker_id = message.record.worker_id
        if worker_id not in self.worker_success_counts:
            self.worker_success_counts[worker_id] = 0
        if worker_id not in self.worker_error_counts:
            self.worker_error_counts[worker_id] = 0

        if message.record.request.has_error:
            self.warning(lambda: f"Received error inference results: {message}")
            # TODO: we do not want to keep all the data forever
            self.error_records.append(message.record)
            self.worker_error_counts[worker_id] += 1
            self.error_records_count += 1
        elif message.record.request.valid:
            # TODO: we do not want to keep all the data forever
            self.records.append(message.record)
            self.worker_success_counts[worker_id] += 1
            self.records_count += 1
        else:
            self.warning(lambda: f"Received invalid inference results: {message}")
            # TODO: we do not want to keep all the data forever
            self.error_records.append(message.record)
            self.worker_error_counts[worker_id] += 1
            self.error_records_count += 1

        if (
            self.final_request_count is not None
            and self.records_count >= self.final_request_count
        ):
            self.info(
                lambda: f"Processed {self.records_count} requests and {self.error_records_count} errors."
            )
            await self.publish_processing_stats()
            # TODO: Publish PROFILE_RESULTS_COMPLETE message

    async def get_error_summary(self) -> list[ErrorDetailsCount]:
        """Generate a summary of the error records."""
        summary: dict[ErrorDetails, int] = {}
        for record in self.error_records:
            if record.request.error is None:
                continue
            if record.request.error not in summary:
                summary[record.request.error] = 0
            summary[record.request.error] += 1

        return [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in summary.items()
        ]

    async def process_records(self, message: CommandMessage) -> None:
        """Process the records.

        This method is called when the records manager receives a command to process the records.
        """
        self.notice(lambda: f"Processing records: {message}")
        self.was_cancelled = (
            message.data.cancelled
            if isinstance(message.data, ProcessRecordsCommandData)
            else False
        )
        self.end_time_ns = time.time_ns()
        # TODO: Implement records processing
        self.info(
            lambda: f"Processed {len(self.records)} successful records and {len(self.error_records)} error records"
        )

        profile_results = await self.post_process_records()
        self.info(lambda: f"Profile results: {profile_results}")

        if profile_results:
            await self.pub_client.publish(
                profile_results,
            )

            if self.user_config:
                await ExporterManager(
                    results=profile_results, input_config=self.user_config
                ).export_all()

        else:
            self.error("No profile results to publish")
            await self.pub_client.publish(
                ProfileResultsMessage(
                    service_id=self.service_id,
                    total=0,
                    completed=0,
                    start_ns=self.start_time_ns,
                    end_ns=self.end_time_ns,
                    records=[],
                    errors_by_type=[],
                    was_cancelled=self.was_cancelled,
                ),
            )

    async def post_process_records(self) -> ProfileResultsMessage | None:
        """Post process the records."""
        self.trace("Post processing records")

        if not self.records:
            self.warning("No successful records to process")
            return ProfileResultsMessage(
                service_id=self.service_id,
                total=len(self.records),
                completed=len(self.records) + len(self.error_records),
                start_ns=self.start_time_ns or time.time_ns(),
                end_ns=self.end_time_ns or time.time_ns(),
                records=[],
                errors_by_type=await self.get_error_summary(),
                was_cancelled=self.was_cancelled,
            )

        self.trace(
            lambda: f"Token counts: {', '.join([str(r.output_token_count) for r in self.records])}"
        )
        metric_summary = MetricSummary()
        metric_summary.process(list(self.records))
        metrics_summary = metric_summary.get_metrics_summary()

        # Create and return ProfileResultsMessage
        return ProfileResultsMessage(
            service_id=self.service_id,
            total=len(self.records),
            completed=len(self.records) + len(self.error_records),
            start_ns=self.start_time_ns or time.time_ns(),
            end_ns=self.end_time_ns or time.time_ns(),
            records=metrics_summary,
            errors_by_type=await self.get_error_summary(),
            was_cancelled=self.was_cancelled,
        )


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(RecordsManager)


if __name__ == "__main__":
    sys.exit(main())

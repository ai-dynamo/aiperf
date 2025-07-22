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
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommandType, CreditPhase, MessageType, ServiceType
from aiperf.common.factories import ServiceFactory, StreamingPostProcessorFactory
from aiperf.common.hooks import (
    on_cleanup,
    on_init,
    on_stop,
)
from aiperf.common.messages import (
    ParsedInferenceResultsMessage,
    ProfileResultsMessage,
)
from aiperf.common.messages.progress_messages import AllRecordsReceivedMessage
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    ParsedResponseRecord,
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
            MessageType.ALL_RECORDS_RECEIVED,
            self._on_all_records_received,
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

    async def _on_all_records_received(
        self, message: AllRecordsReceivedMessage
    ) -> None:
        """Handle a all records received message."""
        self.debug(lambda: f"Received all records: {message}")
        await self.process_records(cancelled=False)

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

        if message.record.request.valid:
            # TODO: we do not want to keep all the data forever
            self.records.append(message.record)
        else:
            self.warning(lambda: f"Received invalid inference results: {message}")
            # TODO: we do not want to keep all the data forever
            self.error_records.append(message.record)

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

    async def process_records(self, cancelled: bool = False) -> None:
        """Process the records.

        This method is called when the records manager receives a command to process the records.
        """
        self.notice(lambda: f"Processing records: {cancelled=}")
        self.was_cancelled = cancelled
        self.end_time_ns = time.time_ns()

        if not self.records:
            self.warning("No records to process")
            return

        profile_results = await self.post_process_records()
        self.info(
            lambda: f"Processed {len(self.records)} successful records and {len(self.error_records)} error records"
        )

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

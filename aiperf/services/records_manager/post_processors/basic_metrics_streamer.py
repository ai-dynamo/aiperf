# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import deque

from aiperf.common.enums import StreamingPostProcessorType
from aiperf.common.enums.message_enums import MessageType
from aiperf.common.factories import StreamingPostProcessorFactory
from aiperf.common.hooks import on_init
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    CommandMessage,
    ProfileResultsMessage,
)
from aiperf.common.messages.command_messages import (
    ProcessRecordsCommandData,
)
from aiperf.common.models import ErrorDetails, ErrorDetailsCount, ParsedResponseRecord
from aiperf.data_exporter.exporter_manager import ExporterManager
from aiperf.services.records_manager.post_processors.metric_summary import MetricSummary
from aiperf.services.records_manager.post_processors.streaming_post_processor import (
    BaseStreamingPostProcessor,
)


@StreamingPostProcessorFactory.register(StreamingPostProcessorType.BASIC_METRICS)
class BasicMetricsStreamer(BaseStreamingPostProcessor):
    """Streamer for basic metrics."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # TODO: we do not want to keep all the data forever
        self.records: deque[ParsedResponseRecord] = deque()
        self.error_records: deque[ParsedResponseRecord] = deque()
        self.start_time_ns: int = time.time_ns()
        self.end_time_ns: int | None = None

    @on_init
    async def _initialize(self) -> None:
        """Initialize the streamer."""
        await self.sub_client.subscribe(
            MessageType.ALL_RECORDS_RECEIVED, self._on_all_records_received
        )

    async def stream_record(self, record: ParsedResponseRecord) -> None:
        """Stream a record."""
        if record.request.valid:
            # TODO: we do not want to keep all the data forever
            self.records.append(record)
        else:
            self.warning(f"Received invalid inference results: {record}")
            # TODO: we do not want to keep all the data forever
            self.error_records.append(record)

    # TODO: This could be done on the fly as we process the records
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

    async def _on_all_records_received(
        self, message: AllRecordsReceivedMessage
    ) -> None:
        """Handle a all records received message."""
        self.debug(lambda: f"Received all records: {message}")

        # Even though all the records have been received, we need to ensure that
        # all the records have been processed on our side.
        await self.records_queue.join()

        await self.process_records(cancelled=False)

    async def on_process_records_command(
        self, message: CommandMessage
    ) -> ProfileResultsMessage | None:
        """Handle the process records command."""
        cancelled = False
        if isinstance(message.data, ProcessRecordsCommandData):
            cancelled = message.data.cancelled
        return await self.process_records(cancelled=cancelled)

    async def process_records(self, cancelled: bool) -> None:
        """Process the records.

        This method is called when the records manager receives a command to process the records.
        """
        self.notice("Processing records")
        self.end_time_ns = time.time_ns()

        profile_results = await self.post_process_records(cancelled=cancelled)

        self.info(
            f"Processed {len(self.records)} successful records and {len(self.error_records)} error records"
        )
        self.info(f"Profile results: {profile_results}")

        if profile_results:
            await self.pub_client.publish(
                profile_results,
            )

            # TODO: HACK: Figure out a better place to put the exporter logic
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
                    was_cancelled=cancelled,
                ),
            )

    async def post_process_records(
        self, cancelled: bool
    ) -> ProfileResultsMessage | None:
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
                was_cancelled=cancelled,
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
            was_cancelled=cancelled,
        )

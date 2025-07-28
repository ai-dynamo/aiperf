# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from aiperf.common.enums import (
    MessageType,
    PostProcessorType,
    StreamingPostProcessorType,
)
from aiperf.common.factories import PostProcessorFactory, StreamingPostProcessorFactory
from aiperf.common.hooks import on_init
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    CommandMessage,
    ProcessRecordsCommandData,
    ProfileResultsMessage,
)
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    MetricResult,
    ParsedResponseRecord,
)
from aiperf.data_exporter.exporter_manager import ExporterManager
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
        self.metric_summary = PostProcessorFactory.create_instance(
            PostProcessorType.METRIC_SUMMARY,
            endpoint_type=self.user_config.endpoint.type,
        )

    @on_init
    async def _initialize(self) -> None:
        """Initialize the streamer."""
        await self.sub_client.subscribe(
            MessageType.ALL_RECORDS_RECEIVED, self._on_all_records_received
        )

    async def stream_record(self, record: ParsedResponseRecord) -> None:
        """Stream a record."""
        if record.request.valid:
            self.valid_count += 1
            self.metric_summary.process_record(record)
        else:
            self.error_count += 1
            self.warning(f"Received invalid inference results: {record}")
            if record.request.error is not None:
                self.error_summary.setdefault(record.request.error, 0)
                self.error_summary[record.request.error] += 1

    def get_error_summary(self) -> list[ErrorDetailsCount]:
        """Generate a summary of the error records."""
        return [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self.error_summary.items()
        ]

    async def _on_all_records_received(
        self, message: AllRecordsReceivedMessage
    ) -> None:
        """Handle a all records received message."""
        self.debug(lambda: f"Received all records: {message}")

        # Even though all the records have been received, we need to ensure that
        # all the records have been processed on our side.
        await self.records_queue.join()

        try:
            await self.process_records(cancelled=False)
        except Exception as e:
            self.error(f"Error processing records: {e}")
            # TODO: What to do here?

    async def on_process_records_command(
        self, message: CommandMessage
    ) -> list[MetricResult] | ErrorDetails | None:
        """Handle the process records command."""
        cancelled = (
            isinstance(message.data, ProcessRecordsCommandData)
            and message.data.cancelled
        )
        return await self.process_records(cancelled)

    async def process_records(
        self, cancelled: bool
    ) -> list[MetricResult] | ErrorDetails | None:
        """Process the records.

        This method is called when the records manager receives a command to process the records.
        """
        self.notice("Processing records")
        self.end_time_ns = self.end_time_ns or time.time_ns()

        profile_results = ProfileResultsMessage(
            service_id=self.service_id,
            total=self.valid_count,
            completed=self.valid_count + self.error_count,
            start_ns=self.start_time_ns,
            end_ns=self.end_time_ns,
            records=None,
            errors_by_type=self.get_error_summary(),
            was_cancelled=cancelled,
        )

        try:
            if self.valid_count == 0:
                self.warning("No successful records to process")
                return None

            self.info(
                f"Processing {self.valid_count} successful records and {self.error_count} error records"
            )
            self.metric_summary.post_process()
            profile_results.records = self.metric_summary.get_results()
            return profile_results.records
        except Exception as e:
            self.exception(f"Error processing records: {e}")
            profile_results.records = ErrorDetails.from_exception(e)
            return profile_results.records
        finally:
            # always publish the profile results
            self.execute_async(self.pub_client.publish(profile_results))
            # TODO: HACK: Figure out a better place to put the exporter logic
            if self.user_config:
                await ExporterManager(
                    results=profile_results, input_config=self.user_config
                ).export_all()

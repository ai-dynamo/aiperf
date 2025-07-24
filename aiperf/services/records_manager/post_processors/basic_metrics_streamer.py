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
from aiperf.common.models.record_models import MetricResult
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
            total=len(self.records or []),
            completed=len(self.records or []) + len(self.error_records or []),
            start_ns=self.start_time_ns,
            end_ns=self.end_time_ns,
            records=None,
            errors_by_type=await self.get_error_summary(),
            was_cancelled=cancelled,
        )

        try:
            if not self.records:
                self.warning("No successful records to process")
                return None
            else:
                self.info(
                    f"Processing {len(self.records)} successful records and {len(self.error_records)} error records"
                )
                endpoint_type = None
                if hasattr(self, "user_config") and self.user_config is not None:
                    endpoint_type = getattr(self.user_config.endpoint, "type", None)
                metric_summary = MetricSummary(endpoint_type=endpoint_type)
                metric_summary.process(list(self.records))
                profile_results.records = metric_summary.get_metrics_summary()
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

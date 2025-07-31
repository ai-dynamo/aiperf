# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import DEFAULT_PULL_CLIENT_MAX_CONCURRENCY
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
    ServiceType,
)
from aiperf.common.factories import ServiceFactory, StreamingPostProcessorFactory
from aiperf.common.hooks import on_command, on_message, on_pull_message
from aiperf.common.messages import (
    ParsedInferenceResultsMessage,
    ProcessRecordsCommand,
)
from aiperf.common.messages.command_messages import ProfileCancelCommand
from aiperf.common.messages.progress_messages import (
    AllRecordsReceivedMessage,
    ProcessRecordsResultMessage,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults
from aiperf.common.protocols import ServiceProtocol, StreamingPostProcessorProtocol


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.RECORDS_MANAGER)
class RecordsManager(PullClientMixin, BaseComponentService):
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
            pull_client_address=CommAddress.RECORDS,
            pull_client_bind=True,
            pull_client_max_concurrency=DEFAULT_PULL_CLIENT_MAX_CONCURRENCY,
        )
        self._profile_cancelled = False
        self.streaming_post_processors: list[StreamingPostProcessorProtocol] = []
        for streamer_type in StreamingPostProcessorFactory.get_all_class_types():
            streamer = StreamingPostProcessorFactory.create_instance(
                class_type=streamer_type,
                service_id=self.service_id,
                service_config=self.service_config,
                user_config=self.user_config,
            )
            self.debug(
                f"Created streaming post processor: {streamer_type}: {streamer.__class__.__name__}"
            )
            self.streaming_post_processors.append(streamer)
            self.attach_child_lifecycle(streamer)

    @on_pull_message(MessageType.PARSED_INFERENCE_RESULTS)
    async def _on_parsed_inference_results(
        self, message: ParsedInferenceResultsMessage
    ) -> None:
        """Handle a parsed inference results message."""
        self.trace(lambda: f"Received parsed inference results: {message}")

        if self._profile_cancelled:
            self.debug("Skipping record because profiling is cancelled")
            return

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

    @on_command(CommandType.PROCESS_RECORDS)
    async def _on_process_records_command(
        self, message: ProcessRecordsCommand
    ) -> ProcessRecordsResult:
        """Handle the process records command by forwarding it to all of the streaming post processors, and returning the results."""
        self.debug(lambda: f"Received process records command: {message}")
        return await self._process_records(cancelled=message.cancelled)

    @on_command(CommandType.PROFILE_CANCEL)
    async def _on_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> ProcessRecordsResult:
        """Handle the profile cancel command by cancelling the streaming post processors."""
        self.debug(lambda: f"Received profile cancel command: {message}")
        self._profile_cancelled = True
        for streamer in self.streaming_post_processors:
            streamer.cancellation_event.set()
        return await self._process_records(cancelled=True)

    @on_message(MessageType.ALL_RECORDS_RECEIVED)
    async def _on_all_records_received(
        self, message: AllRecordsReceivedMessage
    ) -> None:
        """Handle a all records received message."""
        self.debug(lambda: f"Received all records: {message}, processing now...")
        await self._process_records(cancelled=self._profile_cancelled)

    async def _process_records(self, cancelled: bool) -> ProcessRecordsResult:
        """Process the records."""
        self.debug(lambda: f"Processing records (cancelled: {cancelled})")

        # Even though all the records have been received, we need to ensure that
        # all the records have been processed through the streaming post processors.
        await asyncio.gather(
            *[
                streamer.records_queue.join()
                for streamer in self.streaming_post_processors
            ]
        )

        # Process the records through the streaming post processors
        results = await asyncio.gather(
            *[
                streamer.process_records(cancelled)
                for streamer in self.streaming_post_processors
            ],
            return_exceptions=True,
        )
        self.debug(lambda: f"Processed records results: {results}")

        records_results = [
            result for result in results if isinstance(result, ProfileResults)
        ]
        error_results = [
            result for result in results if isinstance(result, ErrorDetails)
        ]

        result = ProcessRecordsResult(records=records_results, errors=error_results)
        self.debug(lambda: f"Processed records result: {result}")
        await self.publish(
            ProcessRecordsResultMessage(
                service_id=self.service_id,
                process_records_result=result,
            )
        )
        return result


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(RecordsManager)


if __name__ == "__main__":
    main()

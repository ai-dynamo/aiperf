# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import sys
from typing import Any

from aiperf.common.comms.base import (
    CommAddress,
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
)
from aiperf.common.messages.command_messages import CommandMessage
from aiperf.services.base_component_service import BaseComponentService
from aiperf.services.records_manager.post_processors import BaseStreamingPostProcessor

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
        self.streaming_post_processors: list[BaseStreamingPostProcessor] = []

        self.response_results_client: PullClientProtocol = (
            self.comms.create_pull_client(
                CommAddress.RECORDS,
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
        await self.response_results_client.register_pull_callback(
            message_type=MessageType.PARSED_INFERENCE_RESULTS,
            callback=self._on_parsed_inference_results,
            max_concurrency=DEFAULT_MAX_RECORDS_CONCURRENCY,
        )
        self.register_command_callback(
            CommandType.PROCESS_RECORDS, self._on_process_records_command
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

    async def _on_process_records_command(self, message: CommandMessage) -> list[Any]:
        """Handle the process records command by forwarding it to all of the streaming post processors, and returning the results."""
        self.debug(lambda: f"Received process records command: {message}")
        results = await asyncio.gather(
            *[
                streamer.on_process_records_command(message)
                for streamer in self.streaming_post_processors
            ]
        )
        return results


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(RecordsManager)


if __name__ == "__main__":
    sys.exit(main())

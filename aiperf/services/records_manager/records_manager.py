# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import sys
import time
from collections import deque

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import CommandType, ServiceType
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_cleanup,
    on_configure,
    on_init,
    on_start,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    InferenceResultsMessage,
    Message,
    ParsedInferenceResultsMessage,
    ProcessRecordsCommandData,
    ProfileResultsMessage,
)
from aiperf.common.record_models import (
    ErrorDetails,
    ErrorDetailsCount,
    ParsedResponseRecord,
)
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.services.records_manager.post_processors.metric_summary import MetricSummary


@ServiceFactory.register(ServiceType.RECORDS_MANAGER)
class RecordsManager(BaseComponentService):
    """
    The RecordsManager service is primarily responsible for holding the
    results returned from the workers.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing records manager")

        self.records: deque[ParsedResponseRecord] = deque()
        self.error_records: deque[ParsedResponseRecord] = deque()
        self.error_records_count: int = 0
        self.records_count: int = 0
        # Track per-worker statistics
        self.worker_success_counts: dict[str, int] = {}
        self.worker_error_counts: dict[str, int] = {}

        self.start_time_ns: int = time.time_ns()
        self.end_time_ns: int | None = None
        self.incoming_records: asyncio.Queue[InferenceResultsMessage] = asyncio.Queue()

        # TODO: Enable after ZMQ clients refactoring
        # self.response_results_client: PullClient = self.comms.create_pull_client(
        #     ClientAddressType.INFERENCE_RESULTS_PUSH_PULL,
        #     bind=True,
        # )

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.RECORDS_MANAGER

    @on_init
    async def _initialize(self) -> None:
        """Initialize records manager-specific components."""
        self.logger.debug("Initializing records manager")
        self.register_command_callback(
            CommandType.PROCESS_RECORDS,
            self.process_records,
        )

    @on_start
    async def _start(self) -> None:
        """Start the records manager."""
        self.logger.debug("Starting records manager")
        # TODO: Implement records manager start

    @on_stop
    async def _stop(self) -> None:
        """Stop the records manager."""
        self.logger.debug("Stopping records manager")
        # TODO: Implement records manager stop

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up records manager-specific components."""
        self.logger.debug("Cleaning up records manager")
        # TODO: Implement records manager cleanup

    @on_configure
    async def _configure(self, message: Message) -> None:
        """Configure the records manager."""
        self.logger.debug(f"Configuring records manager with message: {message}")
        # TODO: Implement records manager configuration

    async def _on_inference_results(self, message: InferenceResultsMessage) -> None:
        """Handle an inference results message."""
        self.incoming_records.put_nowait(message)

    async def _on_post_process_results(
        self, message: ParsedInferenceResultsMessage
    ) -> None:
        """Handle a post process results message."""
        self.logger.debug("Received post process results: %s", message)

        worker_id = message.record.worker_id
        if worker_id not in self.worker_success_counts:
            self.worker_success_counts[worker_id] = 0
        if worker_id not in self.worker_error_counts:
            self.worker_error_counts[worker_id] = 0

        if message.record.request.has_error:
            self.logger.warning("Received error post process results: %s", message)
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
            self.logger.warning("Received invalid post process results: %s", message)
            # TODO: we do not want to keep all the data forever
            self.error_records.append(message.record)
            self.worker_error_counts[worker_id] += 1
            self.error_records_count += 1

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
        self.logger.debug("Processing records")
        self.was_cancelled = (
            message.data.cancelled
            if isinstance(message.data, ProcessRecordsCommandData)
            else False
        )
        self.end_time_ns = time.time_ns()
        # TODO: Implement records processing
        self.logger.info(
            "Processed %d successful records and %d error records",
            len(self.records),
            len(self.error_records),
        )

        profile_results = await self.post_process_records()
        self.logger.info("Profile results: %s", profile_results)

        # TODO: Enable after ZMQ Clients refactor
        # if profile_results:
        #     await self.pub_client.publish(
        #         profile_results,
        #     )
        # else:
        #     self.logger.error("No profile results to publish")
        #     await self.pub_client.publish(
        #         ProfileResultsMessage(
        #             service_id=self.service_id,
        #             total=0,
        #             completed=0,
        #             start_ns=self.start_time_ns,
        #             end_ns=self.end_time_ns,
        #             records=[],
        #             errors_by_type=[],
        #             was_cancelled=self.was_cancelled,
        #         ),
        #     )

    async def post_process_records(self) -> ProfileResultsMessage | None:
        """Post process the records."""
        self.logger.debug("Post processing records")

        if not self.records:
            self.logger.warning("No successful records to process")
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

        self.logger.info("Token counts: %s", [r.token_count for r in self.records])
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

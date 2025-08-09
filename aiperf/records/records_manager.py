# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time

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
from aiperf.common.enums.metric_enums import MetricValueTypeT
from aiperf.common.factories import (
    ResultsProcessorFactory,
    ServiceFactory,
)
from aiperf.common.hooks import background_task, on_command, on_message, on_pull_message
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    CreditPhaseCompleteMessage,
    CreditPhaseStartMessage,
    MetricRecordsMessage,
    ProcessRecordsCommand,
    ProcessRecordsResultMessage,
    ProfileCancelCommand,
    RecordsProcessingStatsMessage,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    PhaseProcessingStats,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.common.protocols import ResultsProcessorProtocol, ServiceProtocol
from aiperf.common.types import MetricTagT


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

        self.start_time_ns: int | None = None
        self.processing_stats: PhaseProcessingStats = PhaseProcessingStats()
        self.final_request_count: int | None = None
        self.end_time_ns: int | None = None
        self.error_summary: dict[ErrorDetails, int] = {}
        # Track per-worker statistics
        self.worker_stats: dict[str, PhaseProcessingStats] = {}

        self._results_processors: list[ResultsProcessorProtocol] = []
        for results_processor_type in ResultsProcessorFactory.get_all_class_types():
            results_processor = ResultsProcessorFactory.create_instance(
                class_type=results_processor_type,
                service_id=self.service_id,
                service_config=self.service_config,
                user_config=self.user_config,
            )
            self.debug(
                f"Created results processor: {results_processor_type}: {results_processor.__class__.__name__}"
            )
            self._results_processors.append(results_processor)

    @on_pull_message(MessageType.METRIC_RECORDS)
    async def _on_metric_records(self, message: MetricRecordsMessage) -> None:
        """Handle a metric records message."""
        self.trace(lambda: f"Received metric records: {message}")

        if message.credit_phase != CreditPhase.PROFILING:
            self.debug(lambda: f"Skipping non-profiling record: {message.credit_phase}")
            return

        worker_id = message.worker_id
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = PhaseProcessingStats()

        if message.valid:
            self.worker_stats[worker_id].processed += 1
            self.processing_stats.processed += 1
        else:
            self.worker_stats[worker_id].errors += 1
            self.processing_stats.errors += 1
            if message.error:
                self.error_summary[message.error] = (
                    self.error_summary.get(message.error, 0) + 1
                )

        await self._send_results_to_results_processors(message.results)

        if (
            self.final_request_count is not None
            and self.processing_stats.total_records >= self.final_request_count
        ):
            self.info(
                lambda: f"Processed {self.processing_stats.processed} valid requests and {self.processing_stats.errors} errors ({self.processing_stats.total_records} total)."
            )
            # Make sure everyone knows the final stats, including the worker stats
            await self._publish_processing_stats()

            # Send a message to the event bus to signal that we received all the records
            await self.publish(
                AllRecordsReceivedMessage(
                    service_id=self.service_id,
                    request_ns=time.time_ns(),
                    final_processing_stats=self.processing_stats,
                )
            )

            self.debug(lambda: f"Received all records: {message}, processing now...")
            await self._process_records(cancelled=self._profile_cancelled)

    async def _send_results_to_results_processors(
        self, results: list[dict[MetricTagT, MetricValueTypeT]]
    ) -> None:
        """Send the results to each of the results processors."""
        await asyncio.gather(
            *[
                results_processor.process_result(result)
                for results_processor in self._results_processors
                for result in results
            ]
        )

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(
        self, phase_start_msg: CreditPhaseStartMessage
    ) -> None:
        """Handle a credit phase start message in order to track the total number of expected requests."""
        if phase_start_msg.phase == CreditPhase.PROFILING:
            self.start_time_ns = phase_start_msg.start_ns or time.time_ns()
            self.processing_stats.total_expected_requests = (
                phase_start_msg.total_expected_requests
            )

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, phase_complete_msg: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message in order to track the final request count."""
        if phase_complete_msg.phase != CreditPhase.PROFILING:
            return
        # This will equate to how many records we expect to receive,
        # and once we receive that many records, we know to stop.
        self.final_request_count = phase_complete_msg.completed
        self.end_time_ns = phase_complete_msg.end_ns or time.time_ns()
        self.info(f"Updating final request count to {self.final_request_count}")
        self.notice(
            f"All requests have completed, please wait for the results to be processed (currently {self.processing_stats.total_records} of {self.final_request_count} records processed)..."
        )
        if self.final_request_count == self.processing_stats.total_records:
            await self._process_records(cancelled=False)

    @background_task(
        interval=lambda self: self.service_config.progress_report_interval,
        immediate=False,
    )
    async def _report_records_task(self) -> None:
        """Report the records processing stats."""
        if self.processing_stats.processed > 0 or self.processing_stats.errors > 0:
            # Only publish stats if there are records to report
            await self._publish_processing_stats()

    async def _publish_processing_stats(self) -> None:
        """Publish the profile processing stats."""
        await self.publish(
            RecordsProcessingStatsMessage(
                service_id=self.service_id,
                request_ns=time.time_ns(),
                processing_stats=self.processing_stats,
                worker_stats=self.worker_stats,
            ),
        )

    @on_command(CommandType.PROCESS_RECORDS)
    async def _on_process_records_command(
        self, message: ProcessRecordsCommand
    ) -> ProcessRecordsResult:
        """Handle the process records command by forwarding it to all of the results processors, and returning the results."""
        self.debug(lambda: f"Received process records command: {message}")
        return await self._process_records(cancelled=message.cancelled)

    @on_command(CommandType.PROFILE_CANCEL)
    async def _on_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> ProcessRecordsResult:
        """Handle the profile cancel command by cancelling the streaming post processors."""
        self.debug(lambda: f"Received profile cancel command: {message}")
        self._profile_cancelled = True
        return await self._process_records(cancelled=True)

    async def _process_records(self, cancelled: bool) -> ProcessRecordsResult:
        """Process the records."""
        self.debug(lambda: f"Processing records (cancelled: {cancelled})")

        self.info("Processing records results...")
        # Process the records through the results processors.
        results = await asyncio.gather(
            *[
                results_processor.summarize()
                for results_processor in self._results_processors
            ],
            return_exceptions=True,
        )

        records_results, error_results = [], []
        for result in results:
            if isinstance(result, list):
                records_results.extend(result)
            elif isinstance(result, ErrorDetails):
                error_results.append(result)
            elif isinstance(result, BaseException):
                error_results.append(ErrorDetails.from_exception(result))

        result = ProcessRecordsResult(
            results=ProfileResults(
                records=records_results,
                completed=len(records_results),
                start_ns=self.start_time_ns or time.time_ns(),
                end_ns=self.end_time_ns or time.time_ns(),
                error_summary=self.get_error_summary(),
                was_cancelled=cancelled,
            ),
            errors=error_results,
        )
        self.debug(lambda: f"Process records result: {result}")
        await self.publish(
            ProcessRecordsResultMessage(
                service_id=self.service_id,
                results=result,
            )
        )
        return result

    def get_error_summary(self) -> list[ErrorDetailsCount]:
        """Generate a summary of the error records."""
        return [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self.error_summary.items()
        ]


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(RecordsManager)


if __name__ == "__main__":
    main()

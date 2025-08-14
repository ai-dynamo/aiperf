# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import time

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import (
    DEFAULT_PULL_CLIENT_MAX_CONCURRENCY,
    DEFAULT_REALTIME_METRICS_INTERVAL,
)
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
    ServiceType,
)
from aiperf.common.enums.metric_enums import MetricValueTypeT
from aiperf.common.enums.ui_enums import AIPerfUIType
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
    RealtimeMetricsMessage,
    RecordsProcessingStatsMessage,
)
from aiperf.common.messages.command_messages import RealtimeMetricsCommand
from aiperf.common.messages.credit_messages import CreditPhaseSendingCompleteMessage
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    ProcessingStats,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.common.models.record_models import MetricResult
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

        #########################################################
        # Protected by processing_status_lock
        self.processing_status_lock: asyncio.Lock = asyncio.Lock()
        self.start_time_ns: int | None = None
        self.processing_stats: ProcessingStats = ProcessingStats()
        self.final_request_count: int | None = None
        self.end_time_ns: int | None = None
        self.sent_all_records_received: bool = False
        self.profile_cancelled: bool = False
        #########################################################

        self.error_summary: dict[ErrorDetails, int] = {}
        self.error_summary_lock: asyncio.Lock = asyncio.Lock()
        # Track per-worker statistics
        self.worker_stats: dict[str, ProcessingStats] = {}
        self.worker_stats_lock: asyncio.Lock = asyncio.Lock()

        self._previous_realtime_records: int | None = None

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
        if self.is_trace_enabled:
            self.trace(f"Received metric records: {message}")

        if message.credit_phase != CreditPhase.PROFILING:
            self.debug(lambda: f"Skipping non-profiling record: {message.credit_phase}")
            return

        await self._send_results_to_results_processors(message.results)

        worker_id = message.worker_id

        if message.valid:
            async with self.worker_stats_lock:
                self.worker_stats.setdefault(
                    worker_id, ProcessingStats()
                ).processed += 1
            async with self.processing_status_lock:
                self.processing_stats.processed += 1
        else:
            async with self.worker_stats_lock:
                self.worker_stats.setdefault(worker_id, ProcessingStats()).errors += 1
            async with self.processing_status_lock:
                self.processing_stats.errors += 1
            if message.error:
                async with self.error_summary_lock:
                    self.error_summary[message.error] = (
                        self.error_summary.get(message.error, 0) + 1
                    )

        await self._check_if_all_records_received()

    async def _check_if_all_records_received(self) -> None:
        """Check if all records have been received, and if so, publish a message and process the records."""
        all_records_received = False
        async with self.processing_status_lock:
            if (
                self.final_request_count is not None
                and self.processing_stats.total_records >= self.final_request_count
            ):
                if self.processing_stats.total_records > self.final_request_count:
                    self.warning(
                        f"Processed {self.processing_stats.total_records:,} records, but only expected {self.final_request_count:,} records"
                    )

                all_records_received = True
                if self.sent_all_records_received:
                    return
                self.sent_all_records_received = True

        if all_records_received:
            self.info(
                lambda: f"Processed {self.processing_stats.processed} valid requests and {self.processing_stats.errors} errors ({self.processing_stats.total_records} total)."
            )
            # Make sure everyone knows the final stats, including the worker stats
            await self._publish_processing_stats()

            async with self.processing_status_lock:
                cancelled = self.profile_cancelled
                proc_stats = copy.deepcopy(self.processing_stats)

            # Send a message to the event bus to signal that we received all the records
            await self.publish(
                AllRecordsReceivedMessage(
                    service_id=self.service_id,
                    request_ns=time.time_ns(),
                    final_processing_stats=proc_stats,
                )
            )

            self.debug("Received all records, processing now...")
            await self._process_results(cancelled=cancelled)

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
        if phase_start_msg.phase != CreditPhase.PROFILING:
            return
        async with self.processing_status_lock:
            self.start_time_ns = phase_start_msg.start_ns
            self.processing_stats.total_expected_requests = (
                phase_start_msg.total_expected_requests
            )

    @on_message(MessageType.CREDIT_PHASE_SENDING_COMPLETE)
    async def _on_credit_phase_sending_complete(
        self, message: CreditPhaseSendingCompleteMessage
    ) -> None:
        """Handle a credit phase sending complete message in order to track the final request count."""
        if message.phase != CreditPhase.PROFILING:
            return
        # This will equate to how many records we expect to receive,
        # and once we receive that many records, we know to stop.
        async with self.processing_status_lock:
            self.final_request_count = message.sent
            self.info(
                f"Sent {self.final_request_count:,} requests. Waiting for completion..."
            )

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, message: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message in order to track the end time, and check if all records have been received."""
        if message.phase != CreditPhase.PROFILING:
            return
        async with self.processing_status_lock:
            if self.final_request_count is None:
                # If for whatever reason the final request count was not set, use the number of completed requests.
                # This would only happen if the credit phase sending complete message was not received by the service.
                self.warning(
                    f"Final request count was not set for profiling phase, using {message.completed:,} as the final request count"
                )
                self.final_request_count = message.completed
            self.end_time_ns = message.end_ns
            self.notice(
                f"All requests have completed, please wait for the results to be processed "
                f"(currently {self.processing_stats.total_records:,} of {self.final_request_count:,} records processed)..."
            )
        # This check is to prevent a race condition where the timing manager processes
        # all records before we have the final request count set.
        await self._check_if_all_records_received()

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

        async with self.processing_status_lock, self.worker_stats_lock:
            proc_stats = copy.deepcopy(self.processing_stats)
            worker_stats = copy.deepcopy(self.worker_stats)

        message = RecordsProcessingStatsMessage(
            service_id=self.service_id,
            request_ns=time.time_ns(),
            processing_stats=proc_stats,
            worker_stats=worker_stats,
        )
        await self.publish(message)

    @on_command(CommandType.PROCESS_RECORDS)
    async def _on_process_records_command(
        self, message: ProcessRecordsCommand
    ) -> ProcessRecordsResult:
        """Handle the process records command by forwarding it to all of the results processors, and returning the results."""
        self.debug(lambda: f"Received process records command: {message}")
        return await self._process_results(cancelled=message.cancelled)

    @on_command(CommandType.PROFILE_CANCEL)
    async def _on_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> ProcessRecordsResult:
        """Handle the profile cancel command by cancelling the streaming post processors."""
        self.debug(lambda: f"Received profile cancel command: {message}")
        async with self.processing_status_lock:
            self.profile_cancelled = True
        return await self._process_results(cancelled=True)

    @background_task(interval=None, immediate=True)
    async def _report_realtime_metrics_task(self) -> None:
        """Report the real-time metrics at a regular interval (only if the UI type is dashboard)."""
        if self.service_config.ui_type != AIPerfUIType.DASHBOARD:
            return
        while not self.stop_requested:
            await asyncio.sleep(DEFAULT_REALTIME_METRICS_INTERVAL)
            async with self.processing_status_lock:
                if (
                    self.processing_stats.total_records
                    == self._previous_realtime_records
                ):
                    continue  # No new records have been processed, so no need to update the metrics
                self._previous_realtime_records = self.processing_stats.processed
            await self._report_realtime_metrics()

    @on_command(CommandType.REALTIME_METRICS)
    async def _on_realtime_metrics_command(
        self, message: RealtimeMetricsCommand
    ) -> None:
        """Handle a real-time metrics command."""
        await self._report_realtime_metrics()

    async def _report_realtime_metrics(self) -> None:
        """Report the real-time metrics."""
        metrics = await self._generate_realtime_metrics()
        if not metrics:
            return
        await self.publish(
            RealtimeMetricsMessage(
                service_id=self.service_id,
                metrics=metrics,
            )
        )

    async def _generate_realtime_metrics(self) -> list[MetricResult]:
        """Generate the real-time metrics for the profile run."""
        results = await asyncio.gather(
            *[
                results_processor.summarize()
                for results_processor in self._results_processors
            ],
            return_exceptions=True,
        )
        return [
            res
            for result in results
            if isinstance(result, list)
            for res in result
            if isinstance(res, MetricResult)
        ]

    async def _process_results(self, cancelled: bool) -> ProcessRecordsResult:
        """Process the results."""
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
                error_summary=await self.get_error_summary(),
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

    async def get_error_summary(self) -> list[ErrorDetailsCount]:
        """Generate a summary of the error records."""
        async with self.error_summary_lock:
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

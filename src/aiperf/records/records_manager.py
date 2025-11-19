# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import time
from collections import defaultdict
from dataclasses import dataclass, field

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    AIPerfUIType,
    CommAddress,
    CommandType,
    CreditPhase,
    GPUTelemetryMode,
    MessageType,
    ResultsProcessorType,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import FactoryCreationError, PostProcessorDisabled
from aiperf.common.factories import ResultsProcessorFactory, ServiceFactory
from aiperf.common.hooks import background_task, on_command, on_message, on_pull_message
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    CreditPhaseCompleteMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    MetricRecordsData,
    MetricRecordsMessage,
    ProcessRecordsCommand,
    ProcessRecordsResultMessage,
    ProcessTelemetryResultMessage,
    ProfileCancelCommand,
    RealtimeMetricsCommand,
    RealtimeMetricsMessage,
    RealtimeTelemetryMetricsMessage,
    RecordsProcessingStatsMessage,
    ServerMetricsRecordsMessage,
    StartRealtimeTelemetryCommand,
    TelemetryRecordsMessage,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    MetricResult,
    ProcessingStats,
    ProcessRecordsResult,
    ProcessTelemetryResult,
    ProfileResults,
    ServerMetricsRecord,
    TelemetryHierarchy,
    TelemetryRecord,
    TelemetryResults,
)
from aiperf.common.protocols import (
    ResultsProcessorProtocol,
    ServerMetricsResultsProcessorProtocol,
    ServiceProtocol,
    TelemetryResultsProcessorProtocol,
)
from aiperf.records.phase_completion import PhaseCompletionChecker


@dataclass
class ErrorTrackingState:
    """Base class for tracking errors with counts and thread-safe access.

    Provides common error tracking functionality for all metrics subsystems
    (telemetry, server metrics, regular metrics).
    """

    error_counts: dict[ErrorDetails, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    error_counts_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class TelemetryTrackingState(ErrorTrackingState):
    """
    Tracks telemetry-related state and performance metrics.

    Consolidates error tracking, warnings, endpoint status, and performance
    statistics for GPU telemetry collection and processing.
    """

    task_runs: int = 0
    total_gen_time_ms: float = 0.0
    total_pub_time_ms: float = 0.0
    total_metrics_generated: int = 0
    mode_enabled_time: float | None = None
    last_metric_values: dict[str, float | None] | None = None


# Type aliases for clarity - all use the base ErrorTrackingState
ServerMetricsTrackingState = ErrorTrackingState
MetricTrackingState = ErrorTrackingState


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
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
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
        self.timeout_triggered: bool = False
        self.expected_duration_sec: float | None = None
        #########################################################

        self._completion_checker = PhaseCompletionChecker()

        self.error_summary: dict[ErrorDetails, int] = {}
        self.error_summary_lock: asyncio.Lock = asyncio.Lock()
        # Track per-worker statistics
        self.worker_stats: dict[str, ProcessingStats] = {}
        self.worker_stats_lock: asyncio.Lock = asyncio.Lock()

        self._previous_realtime_records: int | None = None

        self._telemetry_state = TelemetryTrackingState()
        self._server_metrics_state = ServerMetricsTrackingState()
        self._metric_state = MetricTrackingState()
        self._telemetry_enable_event = asyncio.Event()

        self._metric_results_processors: list[ResultsProcessorProtocol] = []
        self._telemetry_results_processors: list[TelemetryResultsProcessorProtocol] = []
        self._server_metrics_results_processors: list[
            ServerMetricsResultsProcessorProtocol
        ] = []
        self._telemetry_accumulator: TelemetryResultsProcessorProtocol | None = None

        for results_processor_type in ResultsProcessorFactory.get_all_class_types():
            try:
                results_processor = ResultsProcessorFactory.create_instance(
                    class_type=results_processor_type,
                    service_id=self.service_id,
                    service_config=self.service_config,
                    user_config=self.user_config,
                )
                self.attach_child_lifecycle(results_processor)

                if isinstance(results_processor, TelemetryResultsProcessorProtocol):
                    self._telemetry_results_processors.append(results_processor)

                    # Store the accumulating processor separately for hierarchy access
                    if results_processor_type == ResultsProcessorType.TELEMETRY_RESULTS:
                        self._telemetry_accumulator = results_processor
                elif isinstance(
                    results_processor, ServerMetricsResultsProcessorProtocol
                ):
                    self._server_metrics_results_processors.append(results_processor)
                else:
                    self._metric_results_processors.append(results_processor)

                self.debug(
                    f"Created results processor: {results_processor_type}: {results_processor.__class__.__name__}"
                )
            except FactoryCreationError as e:
                if isinstance(e.__cause__, PostProcessorDisabled):
                    self.debug(
                        f"Results processor {results_processor_type} is disabled and will not be used"
                    )

    @on_pull_message(MessageType.METRIC_RECORDS)
    async def _on_metric_records(self, message: MetricRecordsMessage) -> None:
        """Handle a metric records message."""
        if self.is_trace_enabled:
            self.trace(f"Received metric records: {message}")

        if message.metadata.benchmark_phase != CreditPhase.PROFILING:
            self.debug(
                lambda: f"Skipping non-profiling record: {message.metadata.benchmark_phase}"
            )
            return

        record_data = message.to_data()

        should_include_request = self._should_include_request_by_duration(record_data)

        if should_include_request:
            await self._send_results_to_results_processors(record_data)

        worker_id = message.metadata.worker_id

        if record_data.valid and should_include_request:
            # Valid record
            async with self.worker_stats_lock:
                worker_stats = self.worker_stats.setdefault(
                    worker_id, ProcessingStats()
                )
                worker_stats.processed += 1
            async with self.processing_status_lock:
                self.processing_stats.processed += 1
        elif record_data.valid and not should_include_request:
            # Timed out record
            self.debug(
                f"Filtered out record from worker {worker_id} - response received after duration"
            )
        else:
            # Invalid record
            async with self.worker_stats_lock:
                worker_stats = self.worker_stats.setdefault(
                    worker_id, ProcessingStats()
                )
                worker_stats.errors += 1
            async with self.processing_status_lock:
                self.processing_stats.errors += 1
            if record_data.error:
                async with self.error_summary_lock:
                    self.error_summary[record_data.error] = (
                        self.error_summary.get(record_data.error, 0) + 1
                    )

        await self._check_if_all_records_received()

    @on_pull_message(MessageType.TELEMETRY_RECORDS)
    async def _on_telemetry_records(self, message: TelemetryRecordsMessage) -> None:
        """Handle telemetry records message from Telemetry Manager.
        The RecordsManager acts as the central hub for all record processing,
        whether inference metrics or GPU telemetry.

        Args:
            message: Batch of telemetry records from a DCGM collector
        """
        if message.valid:
            try:
                await self._send_telemetry_to_results_processors(message.records)
            except Exception as e:
                error_details = ErrorDetails(
                    message=f"Telemetry processor error: {str(e)}"
                )
                async with self._telemetry_state.error_counts_lock:
                    self._telemetry_state.error_counts[error_details] += 1
                self.debug(f"Failed to process telemetry batch: {e}")
        else:
            if message.error:
                async with self._telemetry_state.error_counts_lock:
                    self._telemetry_state.error_counts[message.error] += 1

    @on_pull_message(MessageType.SERVER_METRICS_RECORDS)
    async def _on_server_metrics_records(
        self, message: ServerMetricsRecordsMessage
    ) -> None:
        """Handle server metrics records message from Server Metrics Manager.

        Forwards full records to results processors.

        Args:
            message: Batch of server metrics records from a Prometheus collector
        """
        if message.valid:
            # Forward full records to results processors
            for record in message.records:
                await self._send_server_metrics_to_results_processors(record)
        else:
            if message.error:
                async with self._server_metrics_state.error_counts_lock:
                    self._server_metrics_state.error_counts[message.error] += 1

    def _should_include_request_by_duration(
        self, record_data: MetricRecordsData
    ) -> bool:
        """Determine if the request should be included based on benchmark duration.

        Args:
            record_data: MetricRecordsData for a single request

        Returns:
            True if the request should be included, else False
        """
        if not self.expected_duration_sec:
            return True

        grace_period_sec = self.user_config.loadgen.benchmark_grace_period
        duration_end_ns = self.start_time_ns + int(
            (self.expected_duration_sec + grace_period_sec) * NANOS_PER_SECOND
        )

        # Check if any response in this request was received after the duration
        # If so, filter out the entire request (all-or-nothing approach)
        if record_data.metadata.request_end_ns > duration_end_ns:
            self.debug(
                f"Filtering out timed-out request - response received "
                f"{record_data.metadata.request_end_ns - duration_end_ns} ns after timeout"
            )
            return False

        return True

    async def _check_if_all_records_received(self) -> None:
        """Check if all records have been received, and if so, publish a message and process the records."""
        all_records_received = False

        async with self.processing_status_lock:
            # Use the Strategy pattern for completion checking
            is_complete, completion_reason = self._completion_checker.is_complete(
                processing_stats=self.processing_stats,
                final_request_count=self.final_request_count,
                timeout_triggered=self.timeout_triggered,
                expected_duration_sec=self.expected_duration_sec,
            )
            all_records_received = is_complete

            if all_records_received:
                if (
                    self.final_request_count is not None
                    and self.processing_stats.total_records > self.final_request_count
                ):
                    self.warning(
                        f"Processed {self.processing_stats.total_records:,} records, but only expected {self.final_request_count:,} records"
                    )

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
        self, record_data: MetricRecordsData
    ) -> None:
        """Send the results to each of the metric results processors."""
        await asyncio.gather(
            *[
                results_processor.process_result(record_data)
                for results_processor in self._metric_results_processors
            ]
        )

    async def _send_telemetry_to_results_processors(
        self, telemetry_records: list[TelemetryRecord]
    ) -> None:
        """Send individual telemetry records to telemetry results processors only.

        Args:
            telemetry_records: Batch of records from single collection cycle
        """
        await asyncio.gather(
            *[
                processor.process_telemetry_record(record)
                for processor in self._telemetry_results_processors
                for record in telemetry_records  # Process each record individually
            ]
        )

    async def _send_server_metrics_to_results_processors(
        self, record: ServerMetricsRecord
    ) -> None:
        """Send individual server metrics records to server metrics results processors only.

        Args:
            record: ServerMetricsRecord from single collection cycle
        """
        errors = await asyncio.gather(
            *[
                processor.process_server_metrics_record(record)
                for processor in self._server_metrics_results_processors
            ],
            return_exceptions=True,
        )
        for error in errors:
            if isinstance(error, BaseException):
                self.exception(f"Failed to process server metrics record: {error!r}")
                async with self._server_metrics_state.error_counts_lock:
                    self._server_metrics_state.error_counts[
                        ErrorDetails.from_exception(error)
                    ] += 1

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(
        self, phase_start_msg: CreditPhaseStartMessage
    ) -> None:
        """Handle a credit phase start message in order to track the total number of expected requests."""
        if phase_start_msg.phase != CreditPhase.PROFILING:
            return
        async with self.processing_status_lock:
            self.start_time_ns = phase_start_msg.start_ns
            self.expected_duration_sec = phase_start_msg.expected_duration_sec
            self.processing_stats.total_expected_requests = (
                phase_start_msg.total_expected_requests
            )

    @on_message(MessageType.CREDIT_PHASE_SENDING_COMPLETE)
    async def _on_credit_phase_sending_complete(
        self, message: CreditPhaseSendingCompleteMessage
    ) -> None:
        """Handle a credit phase sending complete message in order to track the final request count."""
        if message.phase == CreditPhase.PROFILING:
            self.info(
                f"Sent {message.sent:,} conversations. Waiting for all to complete..."
            )

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, message: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message in order to track the end time, and check if all records have been received."""
        if message.phase != CreditPhase.PROFILING:
            return
        async with self.processing_status_lock:
            self.final_request_count = message.final_request_count
            self.end_time_ns = message.end_ns
            self.timeout_triggered = message.timeout_triggered

            self.notice(
                f"All requests have completed, please wait for the results to be processed "
                f"(currently {self.processing_stats.total_records:,} of {self.final_request_count:,} records processed)..."
            )
        # This check is to prevent a race condition where the timing manager processes
        # all records before we have the final request count set.
        await self._check_if_all_records_received()

    @background_task(
        interval=Environment.RECORD.PROGRESS_REPORT_INTERVAL, immediate=False
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
    async def _report_realtime_inference_metrics_task(self) -> None:
        """Report inference metrics at regular intervals (dashboard only)."""
        if self.service_config.ui_type != AIPerfUIType.DASHBOARD:
            return

        while not self.stop_requested:
            await asyncio.sleep(Environment.UI.REALTIME_METRICS_INTERVAL)
            async with self.processing_status_lock:
                if (
                    self.processing_stats.total_records
                    == self._previous_realtime_records
                ):
                    continue  # No new records have been processed, so no need to update the metrics
                self._previous_realtime_records = self.processing_stats.processed
            await self._report_realtime_metrics()

    @background_task(interval=None, immediate=True)
    async def _report_realtime_telemetry_metrics_task(self) -> None:
        """Report telemetry metrics - sleeps when disabled, resumes on command."""
        if self.service_config.ui_type != AIPerfUIType.DASHBOARD:
            return

        while not self.stop_requested:
            if (
                self.user_config.gpu_telemetry_mode
                != GPUTelemetryMode.REALTIME_DASHBOARD
            ):
                # Disabled - sleep until command wakes us
                await self._telemetry_enable_event.wait()
                self._telemetry_enable_event.clear()

            telemetry_metrics = await self._generate_realtime_telemetry_metrics()
            self._telemetry_state.total_metrics_generated += len(telemetry_metrics)

            if telemetry_metrics:
                # Only publish if values have changed - extract once for efficiency
                new_values = {m.tag: m.current for m in telemetry_metrics}
                if (
                    self._telemetry_state.last_metric_values is None
                    or new_values != self._telemetry_state.last_metric_values
                ):
                    await self.publish(
                        RealtimeTelemetryMetricsMessage(
                            service_id=self.service_id,
                            metrics=telemetry_metrics,
                        )
                    )
                    self._telemetry_state.last_metric_values = new_values

            await asyncio.sleep(Environment.UI.REALTIME_METRICS_INTERVAL)

    @on_command(CommandType.REALTIME_METRICS)
    async def _on_realtime_metrics_command(
        self, message: RealtimeMetricsCommand
    ) -> None:
        """Handle a real-time metrics command."""
        await self._report_realtime_metrics()

    @on_command(CommandType.START_REALTIME_TELEMETRY)
    async def _on_start_realtime_telemetry_command(
        self, message: StartRealtimeTelemetryCommand
    ) -> None:
        """Handle command to start the realtime telemetry background task.

        This is called when the user dynamically enables the telemetry dashboard
        by pressing the telemetry option in the UI without having passed the 'dashboard' parameter
        at startup.
        """
        self.info("Received START_REALTIME_TELEMETRY command")

        self.user_config.gpu_telemetry_mode = GPUTelemetryMode.REALTIME_DASHBOARD

        # Wake up the sleeping telemetry task
        self._telemetry_enable_event.set()

    async def _report_realtime_metrics(self) -> None:
        """Report both inference and telemetry metrics (used by command handler)."""
        metrics = await self._generate_realtime_metrics()
        if metrics:
            await self.publish(
                RealtimeMetricsMessage(
                    service_id=self.service_id,
                    metrics=metrics,
                )
            )

        if self.user_config.gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD:
            telemetry_metrics = await self._generate_realtime_telemetry_metrics()
            if telemetry_metrics:
                await self.publish(
                    RealtimeTelemetryMetricsMessage(
                        service_id=self.service_id,
                        metrics=telemetry_metrics,
                    )
                )

    async def _generate_realtime_metrics(self) -> list[MetricResult]:
        """Generate the real-time metrics for the profile run."""
        results = await asyncio.gather(
            *[
                results_processor.summarize()
                for results_processor in self._metric_results_processors
            ],
            return_exceptions=True,
        )

        # Flatten results: each processor returns list[MetricResult], so we have
        # list[list[MetricResult] | Exception]. Flatten to single list[MetricResult].
        metric_results = [
            res
            for result in results
            if isinstance(result, list)
            for res in result
            if isinstance(res, MetricResult)
        ]

        return metric_results

    async def _generate_realtime_telemetry_metrics(self) -> list[MetricResult]:
        """Generate the real-time GPU telemetry metrics."""
        telemetry_results = await asyncio.gather(
            *[
                results_processor.summarize()
                for results_processor in self._telemetry_results_processors
            ],
            return_exceptions=True,
        )

        # Flatten results: each processor returns list[MetricResult], so we have
        # list[list[MetricResult] | Exception]. Flatten to single list[MetricResult].
        telemetry_metrics = [
            res
            for result in telemetry_results
            if isinstance(result, list)
            for res in result
            if isinstance(res, MetricResult)
        ]
        return telemetry_metrics

    async def _process_results(self, cancelled: bool) -> ProcessRecordsResult:
        """Process the results."""
        self.debug(lambda: f"Processing records (cancelled: {cancelled})")

        self.info("Processing records results...")
        # Process the records through the metric results processors only.
        results = await asyncio.gather(
            *[
                results_processor.summarize()
                for results_processor in self._metric_results_processors
            ],
            return_exceptions=True,
        )

        records_results, timeslice_metric_results, error_results = [], {}, []
        for result in results:
            if isinstance(result, list):
                records_results.extend(result)
            elif isinstance(result, dict):
                timeslice_metric_results = result
            elif isinstance(result, ErrorDetails):
                error_results.append(result)
            elif isinstance(result, BaseException):
                error_results.append(ErrorDetails.from_exception(result))

        result = ProcessRecordsResult(
            results=ProfileResults(
                records=records_results,
                timeslice_metric_results=timeslice_metric_results,
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

        await self._publish_telemetry_results()

        return result

    async def export_telemetry_independently(self) -> TelemetryResults | None:
        """Export telemetry data independently from inference results.

        This method provides a separate export path for telemetry data that doesn't
        interfere with the inference results pipeline.

        Retrieves the accumulated telemetry hierarchy from the TelemetryResultsProcessor,
        which serves as the single source of truth for all telemetry data.

        Returns:
            TelemetryResults if telemetry data was collected, None otherwise
        """
        if not self._telemetry_accumulator:
            return None

        # Get hierarchy from the accumulating processor
        telemetry_hierarchy = self._telemetry_accumulator.get_telemetry_hierarchy()

        if not telemetry_hierarchy.dcgm_endpoints:
            return None

        telemetry_results = TelemetryResults(
            telemetry_data=telemetry_hierarchy,
            start_ns=self.start_time_ns or time.time_ns(),
            end_ns=self.end_time_ns or time.time_ns(),
            endpoints_tested=list(telemetry_hierarchy.dcgm_endpoints.keys()),
            endpoints_successful=list(telemetry_hierarchy.dcgm_endpoints.keys()),
            error_summary=await self.get_telemetry_error_summary(),
        )

        return telemetry_results

    async def _process_telemetry_results(self) -> ProcessTelemetryResult:
        """Process telemetry results by calling summarize on all telemetry processors.

        Collects summarized results from all telemetry processors and exports the full
        telemetry hierarchy. Combines processor errors with collection errors to provide
        complete error reporting.

        Returns:
            ProcessTelemetryResult: Contains TelemetryResults with GPU telemetry data hierarchy
                and any errors encountered during collection or processing
        """
        self.debug("Processing telemetry results...")
        results = await asyncio.gather(
            *[
                results_processor.summarize()
                for results_processor in self._telemetry_results_processors
            ],
            return_exceptions=True,
        )

        error_results = []
        for result in results:
            if isinstance(result, ErrorDetails):
                error_results.append(result)
            elif isinstance(result, BaseException):
                error_results.append(ErrorDetails.from_exception(result))

        telemetry_results = await self.export_telemetry_independently()
        if not telemetry_results:
            telemetry_results = TelemetryResults(
                telemetry_data=TelemetryHierarchy(),
                start_ns=self.start_time_ns or time.time_ns(),
                end_ns=self.end_time_ns or time.time_ns(),
            )

        async with self._telemetry_state.error_counts_lock:
            unique_errors = list(self._telemetry_state.error_counts.keys())

        return ProcessTelemetryResult(
            results=telemetry_results,
            errors=error_results + unique_errors,
        )

    async def _publish_telemetry_results(self) -> None:
        """Publish telemetry results independently from inference results.

        Processes and publishes telemetry data via ProcessTelemetryResultMessage.
        Called at the end of _process_results to keep telemetry separate from
        inference metrics in the results pipeline.
        """
        telemetry_result = await self._process_telemetry_results()
        await self.publish(
            ProcessTelemetryResultMessage(
                service_id=self.service_id,
                telemetry_result=telemetry_result,
            )
        )

    async def get_error_summary(self) -> list[ErrorDetailsCount]:
        """Generate a summary of the error records."""
        async with self.error_summary_lock:
            return [
                ErrorDetailsCount(error_details=error_details, count=count)
                for error_details, count in self.error_summary.items()
            ]

    async def get_telemetry_error_summary(self) -> list[ErrorDetailsCount]:
        """Generate a summary of the telemetry error records."""
        async with self._telemetry_state.error_counts_lock:
            return [
                ErrorDetailsCount(error_details=error_details, count=count)
                for error_details, count in self._telemetry_state.error_counts.items()
            ]


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(RecordsManager)


if __name__ == "__main__":
    main()

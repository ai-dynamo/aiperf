# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.accumulator_protocols import (
    AccumulatorProtocol,
    AnalyzerProtocol,
    StreamExporterProtocol,
    SummaryContext,
)
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.channel_codecs import RECORDS_CODEC
from aiperf.config.zmq import ZMQDualBindConfig

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.control_structs import Command
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import (
    background_task,
    on_command,
    on_message,
    on_pull_message,
)
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    DatasetConfiguredNotification,
    NetworkLatencyRecordMessage,
    ProcessAllResultsMessage,
    ProcessRecordsResultMessage,
    ProcessTelemetryResultMessage,
    RealtimeMetricsMessage,
    RecordsProcessingStatsMessage,
    TelemetryRecordsMessage,
)
from aiperf.common.metric_records_wire import (
    MetricRecordsBatchWireMessage,
    MetricRecordsData,
    MetricRecordsWireMessage,
    wire_error_to_domain_error,
    wire_message_to_record_data,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    BranchStats,
    ErrorDetails,
    ErrorDetailsCount,
    ErrorTrackingState,
    MetricResult,
    NetworkLatencySample,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProcessTelemetryResult,
    WorkerProcessingStats,
)
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.network_latency.accumulator import NetworkLatencyAccumulator
from aiperf.network_latency.protocols import NetworkLatencyProcessorProtocol
from aiperf.plugin.enums import (
    AccumulatorType,
    AnalyzerType,
    ServiceRunType,
    StreamExporterType,
    UIType,
)
from aiperf.post_processors.protocols import ResultsProcessorProtocol
from aiperf.records.error_tracker import ErrorTracker
from aiperf.records.records_manager_export import (
    current_results_record_count,
    write_partial_checkpoint,
)
from aiperf.records.records_manager_processing import (
    accumulators_for_record_type,
    bucket_summarize_results,
    build_process_records_result,
    compute_analyzer_outputs,
    filter_display_metrics,
    generate_realtime_metrics,
    load_accumulators,
    load_analyzers,
    load_network_latency_processors,
    load_results_processors,
    load_stream_exporters,
    make_network_latency_accumulator,
    stream_exporters_for_record_type,
)
from aiperf.records.records_tracker import RecordsTracker


class RecordsManager(PullClientMixin, BaseComponentService):
    """Collects and processes benchmark results from workers.

    The RecordsManager receives metric records from workers and accumulates them
    for final processing. The timing manager is the ground truth for what requests
    completed within the benchmark window - when it signals phase completion with
    a final_completed_count, the RecordsManager waits until it has processed that
    many records before finalizing results.
    """

    # The "enable cache reporting" server-knob hint fires at most once per run.
    _warned_missing_cache_reporting: bool = False

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            pull_client_address=CommAddress.RECORDS,
            pull_client_bind=True,
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
            pull_client_additional_bind_address=self._resolve_additional_bind_address(
                run
            ),
            pull_client_codec=RECORDS_CODEC,
            **kwargs,
        )

        self._records_tracker = RecordsTracker()
        self._error_tracker = ErrorTracker()

        # Latest DAG branch-orchestration counters, captured from the
        # PROFILING phase's CreditPhaseCompleteMessage and threaded into
        # ProfileResults.branch_stats for the JSON export. None on non-DAG runs.
        self._latest_branch_stats: BranchStats | None = None

        self._previous_realtime_records: int | None = None

        self._metric_state = ErrorTrackingState()
        # GPU telemetry records arrive on a side channel (CommAddress.RECORDS)
        # pushed by the gpu_telemetry_manager. RecordsManager is the central
        # hub: it feeds them to the gpu_telemetry accumulator and publishes the
        # final ProcessTelemetryResultMessage the SystemController exports.
        self._telemetry_state = ErrorTrackingState()
        self._telemetry_result_published = False

        self._metric_results_processors: list[ResultsProcessorProtocol] = (
            load_results_processors(self)
        )
        self._init_network_latency()
        # Parallel accumulator pipeline (metrics-accumulator branch). The
        # legacy ``_metric_results_processors`` path stays live — both
        # consume the same record stream so analyzers (steady-state, energy
        # efficiency) get a columnar source while existing exporters keep
        # working unchanged. For ``metric_records`` records, both paths run.
        # Disabled / failed plugins are dropped silently — see loaders.
        self._accumulators: dict[AccumulatorType, AccumulatorProtocol] = (
            load_accumulators(self)
        )
        self._stream_exporters: dict[StreamExporterType, StreamExporterProtocol] = (
            load_stream_exporters(self)
        )
        self._analyzers: dict[AnalyzerType, AnalyzerProtocol] = load_analyzers(self)

        # Pre-compute the per-record-type dispatch lists so the hot path is
        # a dict lookup, not an O(N plugins) iter every record.
        self._metric_record_accumulators: list[AccumulatorProtocol] = (
            accumulators_for_record_type(self._accumulators, "metric_records")
        )
        self._metric_record_stream_exporters: list[StreamExporterProtocol] = (
            stream_exporters_for_record_type(self._stream_exporters, "metric_records")
        )
        # GPU telemetry records arrive on a side channel (TELEMETRY_RECORDS) and
        # must be fanned out to the gpu_telemetry stream exporters
        # (GPUTelemetryJSONLWriter writes gpu_telemetry_export.jsonl). On
        # origin/main this was the GPUTelemetryProcessorProtocol fan-out; the
        # stream-exporter rewrite kept the accumulator wiring but dropped the
        # writer dispatch, so the JSONL file was never produced.
        self._gpu_telemetry_stream_exporters: list[StreamExporterProtocol] = (
            stream_exporters_for_record_type(self._stream_exporters, "gpu_telemetry")
        )

        self._last_checkpoint_records: int = 0

    @staticmethod
    def _resolve_additional_bind_address(run: BenchmarkRun) -> str | None:
        """Resolve the extra TCP bind for dual-bind (Kubernetes) mode.

        When ZMQDualBindConfig has no controller_host, this manager process is
        the dual-bind controller and must add a TCP bind on top of IPC so remote
        record processors can reach it. Workers (controller_host set) need none.
        """
        comm_config = run.resolved.comm_config or run.cfg.comm_config
        if (
            isinstance(comm_config, ZMQDualBindConfig)
            and not comm_config.controller_host
        ):
            return comm_config.records_push_pull_tcp_bind_address
        return None

    def _init_network_latency(self) -> None:
        """Wire the network-latency RTT-probe pipeline.

        Network latency processors (e.g. NetworkLatencyJSONLWriter) consume RTT
        samples via process_network_latency_sample, not the metric-record
        protocol, so they load through their own loader. The in-process
        accumulator computes the run-level mean RTT delivered to each
        MetricResultsProcessor via set_network_rtt_ns before summarize(); it is
        None unless network latency probing is active.
        """
        self._network_latency_processors: list[NetworkLatencyProcessorProtocol] = (
            load_network_latency_processors(self)
        )
        self._network_latency_accumulator: NetworkLatencyAccumulator | None = (
            make_network_latency_accumulator(self)
        )
        self._network_latency_state = ErrorTrackingState()

    async def _process_metric_record_data(self, record_data: MetricRecordsData) -> None:
        """Process one metric record payload."""
        self._records_tracker.update_from_record_data(record_data)

        if not self._records_tracker.is_phase_excluded(
            record_data.metadata.benchmark_phase
        ):
            await self._send_results_to_results_processors(record_data)
            # Parallel accumulator path — see ``__init__`` for why both run.
            await self._send_record_to_accumulators(record_data)
        if record_data.error:
            domain_error = wire_error_to_domain_error(record_data.error)
            if domain_error is not None:
                self._error_tracker.increment_error_count_for_phase(
                    record_data.metadata.benchmark_phase, domain_error
                )

        if self._records_tracker.check_and_set_all_records_received_for_phase(
            record_data.metadata.benchmark_phase
        ):
            await self._handle_all_records_received(
                record_data.metadata.benchmark_phase
            )

    async def _send_record_to_accumulators(
        self, record_data: MetricRecordsData
    ) -> None:
        """Dispatch a metric record to all accumulators + stream exporters subscribed.

        Mirrors the legacy ``_send_results_to_results_processors`` fan-out but
        targets the new ``AccumulatorProtocol`` / ``StreamExporterProtocol``
        pipeline. Per-handler exceptions are caught so one bad accumulator
        does not abort the others. GPU telemetry / server metrics records
        are routed via their own side-channel pipelines on K8s
        (``gpu_telemetry_processor`` / ``server_metrics_processor`` plugin
        categories) and do **not** flow through here.
        """
        targets: list[Any] = [
            *self._metric_record_accumulators,
            *self._metric_record_stream_exporters,
        ]
        if not targets:
            return
        results = await asyncio.gather(
            *[t.process_record(record_data) for t in targets],
            return_exceptions=True,
        )
        for target, result in zip(targets, results, strict=True):
            if isinstance(result, BaseException):
                self.error(
                    f"Accumulator {target.__class__.__name__} failed for "
                    f"metric_records: {result!r}"
                )

    @on_pull_message(MessageType.METRIC_RECORDS)
    async def _on_metric_records(
        self, message: MetricRecordsWireMessage | MetricRecordsBatchWireMessage
    ) -> None:
        """Handle a metric records message."""
        if self.is_trace_enabled:
            self.trace(f"Received metric records: {message}")

        if isinstance(message, MetricRecordsBatchWireMessage):
            for record_data in message.records:
                await self._process_metric_record_data(record_data)
            return

        await self._process_metric_record_data(wire_message_to_record_data(message))

    @on_pull_message(MessageType.TELEMETRY_RECORDS)
    async def _on_telemetry_records(self, message: TelemetryRecordsMessage) -> None:
        """Handle a GPU telemetry records batch from the gpu_telemetry_manager.

        The gpu_telemetry_manager pushes raw DCGM samples to CommAddress.RECORDS;
        RecordsManager is the central hub that feeds them to the gpu_telemetry
        accumulator. Error batches (empty records + error) only bump the
        telemetry error counts so the export carries an accurate error summary.
        """
        if message.error is not None and not message.records:
            self._telemetry_state.error_counts[message.error] += 1
            return
        accumulator = self._gpu_telemetry_accumulator
        # The accumulator powers the summary export; the stream exporters
        # (GPUTelemetryJSONLWriter) write the per-sample gpu_telemetry_export.jsonl.
        # Both consume process_telemetry_record(record); feeding only the
        # accumulator (as the stream-exporter rewrite did) silently drops the
        # JSONL file.
        targets = self._gpu_telemetry_stream_exporters
        if accumulator is None and not targets:
            return
        for record in message.records:
            if accumulator is not None:
                try:
                    await accumulator.process_telemetry_record(record)
                except Exception as e:  # noqa: BLE001 - one bad record must not abort the batch
                    self._telemetry_state.error_counts[
                        ErrorDetails.from_exception(e)
                    ] += 1
                    self.debug(lambda e=e: f"Failed to process telemetry record: {e!r}")
            for exporter in targets:
                try:
                    await exporter.process_telemetry_record(record)
                except Exception as e:  # noqa: BLE001 - one bad exporter must not abort the batch
                    self.debug(
                        lambda e=e,
                        exporter=exporter: f"Telemetry stream exporter {exporter.__class__.__name__} failed: {e!r}"
                    )

    @property
    def _gpu_telemetry_accumulator(self) -> Any | None:
        """The loaded GPU telemetry accumulator, or None when telemetry is off."""
        return self._accumulators.get(AccumulatorType.GPU_TELEMETRY)

    @on_pull_message(MessageType.NETWORK_LATENCY_RECORD)
    async def _on_network_latency_records(
        self, message: NetworkLatencyRecordMessage
    ) -> None:
        """Handle a network latency RTT probe sample from the NetworkLatencyManager.

        Accumulates the sample for the run-level mean RTT (delivered to the
        metric processors before summarize) and forwards it to the JSONL writer.
        A transport-level delivery error is tracked separately.

        Args:
            message: Network latency probe sample from a probe collector
        """
        if message.valid:
            if self._network_latency_accumulator is not None:
                self._network_latency_accumulator.add_sample(message.sample)
            await self._send_network_latency_to_results_processors(message.sample)
        else:
            if message.error:
                self._network_latency_state.error_counts[message.error] += 1

    async def _send_network_latency_to_results_processors(
        self, sample: NetworkLatencySample
    ) -> None:
        """Forward a probe sample to the network latency results processors."""
        if not self._network_latency_processors:
            return
        errors = await asyncio.gather(
            *[
                processor.process_network_latency_sample(sample)
                for processor in self._network_latency_processors
            ],
            return_exceptions=True,
        )
        for error in errors:
            if isinstance(error, BaseException):
                self.exception(f"Failed to process network latency sample: {error!r}")
                self._network_latency_state.error_counts[
                    ErrorDetails.from_exception(error)
                ] += 1

    async def _handle_all_records_received(self, phase: CreditPhase) -> None:
        """Handle the case where all records have been received for a phase."""
        phase_stats = self._records_tracker.create_stats_for_phase(phase)
        self.info(
            lambda: f"Phase '{phase}': processed {phase_stats.success_records} valid requests and {phase_stats.error_records} errors ({phase_stats.total_records} total)."
        )

        if phase_stats.exclude_from_results:
            self.debug(
                lambda: f"Phase '{phase}' excluded from results, skipping finalization"
            )
            return

        if not self._records_tracker.are_all_results_phases_complete():
            self.debug(
                lambda: f"Phase '{phase}' complete but waiting for other results phases"
            )
            return

        self.info("All results phases complete, processing now...")
        self.execute_async(
            self._finalize_and_process_results(
                cancelled=any(
                    self._records_tracker.was_phase_cancelled(p)
                    for p in self._records_tracker.get_results_phases()
                ),
            )
        )
        await yield_to_event_loop()

    async def _finalize_and_process_results(self, cancelled: bool) -> None:
        """Finalize server metrics collection and process results.

        This runs as a background task to avoid blocking the message pump.
        Aggregates across all non-excluded phases.
        """
        # Use the first results phase for the AllRecordsReceived message
        results_phases = self._records_tracker.get_results_phases()
        phase_stats = self._records_tracker.create_stats_for_phase(
            results_phases[0] if results_phases else "profiling"
        )

        await self.publish(
            AllRecordsReceivedMessage(
                service_id=self.service_id,
                request_ns=time.time_ns(),
                final_processing_stats=phase_stats,
            )
        )

        # Trigger final server metrics scrape and wait for completion.
        # Include the results time window so side-channel managers can compute
        # their export window from the same authoritative source.
        # The relayed scrape can take 10-30s on contended clusters
        # (Prometheus query + Parquet write); a TimeoutError must not abort
        # _finalize_and_process_results, because that would skip
        # _process_results and the resulting ProcessRecordsResultMessage —
        # the system controller would then never run _export_results_data
        # and the .aiperf_results_ready.json marker would never be written,
        # which the operator's results fetch loop interprets as a failed run.
        start_ns, end_ns = self._records_tracker.get_results_time_window()
        try:
            response = await self.control_client.request(
                Command(
                    cid=uuid.uuid4().hex,
                    cmd=CommandType.PROFILE_COMPLETE,
                    payload=orjson.dumps({"start_ns": start_ns, "end_ns": end_ns}),
                ),
                timeout=Environment.SERVER_METRICS.PROFILE_COMPLETE_RELAY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            self.warning(
                "Server metrics final scrape timed out after "
                f"{Environment.SERVER_METRICS.PROFILE_COMPLETE_RELAY_TIMEOUT}s; "
                "continuing with results processing"
            )
        else:
            if isinstance(response, ErrorDetails):
                self.warning(
                    f"Server metrics final scrape timed out or failed: {response}"
                )

        flush_period = Environment.SERVER_METRICS.COLLECTION_FLUSH_PERIOD
        flush_end_ns = (end_ns or time.time_ns()) + (
            (flush_period or 0) * NANOS_PER_SECOND
        )
        sleep_dur_sec = (flush_end_ns - time.time_ns()) / NANOS_PER_SECOND
        if sleep_dur_sec > 0:
            self.info(
                f"Waiting {sleep_dur_sec:.1f}s for server metrics flush period..."
            )
            await asyncio.sleep(sleep_dur_sec)

        await self._process_results(cancelled=cancelled)

    async def _send_results_to_results_processors(
        self, record_data: MetricRecordsData
    ) -> None:
        """Send the results to each of the metric results processors."""
        if not self._metric_results_processors:
            return
        if len(self._metric_results_processors) == 1:
            await self._metric_results_processors[0].process_result(record_data)
            return
        await asyncio.gather(
            *[
                results_processor.process_result(record_data)
                for results_processor in self._metric_results_processors
            ]
        )

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        for processor in self._metric_results_processors:
            if hasattr(processor, "on_dataset_configured"):
                processor.on_dataset_configured(message.metadata)

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(
        self, phase_start_msg: CreditPhaseStartMessage
    ) -> None:
        """Handle a credit phase start message in order to track the total number of expected requests."""
        self._records_tracker.update_phase_info(phase_start_msg.stats)
        self.info(f"Credit phase start: {phase_start_msg.config.phase}")

    @on_message(MessageType.CREDIT_PHASE_SENDING_COMPLETE)
    async def _on_credit_phase_sending_complete(
        self, message: CreditPhaseSendingCompleteMessage
    ) -> None:
        """Handle a credit phase sending complete message in order to track the final request count."""
        self.info(
            f"Phase '{message.stats.phase}': sent {message.stats.final_requests_sent:,} requests. Waiting for all to complete..."
        )
        self._records_tracker.update_phase_info(message.stats)

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, message: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message in order to track the end time, and check if all records have been received."""
        # Capture DAG branch-orchestration counters for the PROFILING phase so
        # they reach ProfileResults.branch_stats / the JSON export. None on
        # non-DAG runs. ``getattr`` guards minimal test doubles that omit the
        # field; real CreditPhaseCompleteMessage always carries it (default None).
        if (
            getattr(message, "branch_stats", None) is not None
            and message.stats.phase == CreditPhase.PROFILING
        ):
            self._latest_branch_stats = message.branch_stats
        self._records_tracker.update_phase_info(message.stats)
        phase_stats = self._records_tracker.create_stats_for_phase(message.stats.phase)
        self.info(
            lambda: f"Phase '{message.stats.phase}' complete: {phase_stats.total_records:,} records"
        )
        if not phase_stats.exclude_from_results:
            self.notice(
                f"Phase '{message.stats.phase}' requests complete, please wait for results "
                f"({phase_stats.total_records:,} of {phase_stats.final_requests_completed:,} records processed)..."
            )

        # This check is to prevent a race condition where the records manager processes
        # all records before the timing manager has sent the final completed count.
        if self._records_tracker.check_and_set_all_records_received_for_phase(
            message.stats.phase
        ):
            await self._handle_all_records_received(message.stats.phase)

    @on_message(MessageType.CREDITS_COMPLETE)
    async def _on_credits_complete(self, message: CreditsCompleteMessage) -> None:
        """Handle a credits complete message in order to track the end time, and check if all records have been received."""
        self.info(
            "All credits complete, please wait for the results to be processed..."
        )
        # Check all results phases for completion
        for phase in self._records_tracker.get_results_phases():
            if self._records_tracker.check_and_set_all_records_received_for_phase(
                phase
            ):
                await self._handle_all_records_received(phase)

    @background_task(
        interval=Environment.RECORD.PROGRESS_REPORT_INTERVAL, immediate=False
    )
    async def _report_records_task(self) -> None:
        """Report the records processing stats for active non-excluded phases."""
        for phase in self._records_tracker.get_results_phases():
            active_phase_stats = self._records_tracker.create_stats_for_phase(phase)
            if active_phase_stats.total_records == 0:
                continue
            overall_worker_stats = self._records_tracker.create_overall_worker_stats()
            await self._publish_processing_stats(
                active_phase_stats, overall_worker_stats
            )
            break  # Report first active results phase

    async def _publish_processing_stats(
        self,
        phase_stats: PhaseRecordsStats,
        worker_stats: dict[str, WorkerProcessingStats],
    ) -> None:
        """Publish the profile processing stats."""
        message = RecordsProcessingStatsMessage(
            service_id=self.service_id,
            request_ns=time.time_ns(),
            processing_stats=phase_stats,
            worker_stats=worker_stats,
        )
        await self.publish(message)

    @on_command(CommandType.PROCESS_RECORDS)
    async def _on_process_records_command(
        self, message: Command
    ) -> ProcessRecordsResult:
        """Handle the process records command by forwarding it to all of the results processors, and returning the results."""
        self.debug(lambda: f"Received process records command: {message}")
        payload = orjson.loads(message.payload) if message.payload else {}
        cancelled = payload.get("cancelled", False)
        return await self._process_results(cancelled=cancelled)

    @on_command(CommandType.PROFILE_CANCEL)
    async def _on_profile_cancel_command(
        self, message: Command
    ) -> ProcessRecordsResult:
        """Handle the profile cancel command by processing current results.

        This marks the phase as cancelled in the records tracker and processes
        all currently received records. Called when user presses Ctrl+C.
        """
        self.warning(f"Received profile cancel command: {message}")

        # Mark all non-excluded phases as cancelled
        for phase in self._records_tracker.get_results_phases():
            self._records_tracker.mark_phase_cancelled(phase)

        return await self._process_results(cancelled=True)

    @background_task(interval=None, immediate=True)
    async def _report_realtime_inference_metrics_task(self) -> None:
        """Report inference metrics at regular intervals."""
        if (
            self.run.cfg.ui_type != UIType.DASHBOARD
            and not self.run.cfg.runtime.api_port
            and not Environment.UI.REALTIME_METRICS_ENABLED
        ):
            return

        while not self.stop_requested:
            await asyncio.sleep(Environment.UI.REALTIME_METRICS_INTERVAL)
            total_records = current_results_record_count(self._records_tracker)
            if total_records == self._previous_realtime_records:
                continue  # No new records have been processed, so no need to update the metrics
            self._previous_realtime_records = total_records
            await self._report_realtime_metrics()

    @on_command(CommandType.REALTIME_METRICS)
    async def _on_realtime_metrics_command(self, message: Command) -> None:
        """Handle a real-time metrics command."""
        await self._report_realtime_metrics()

    async def _report_realtime_metrics(self) -> None:
        """Report inference metrics (used by command handler).

        Filters out hidden metrics (INTERNAL/EXPERIMENTAL) and converts all
        metrics to display units before publishing. This ensures all consumers
        receive consistent, pre-processed metrics.
        """
        raw_metrics = await generate_realtime_metrics(self._metric_results_processors)
        if not raw_metrics:
            return

        display_metrics = filter_display_metrics(raw_metrics)
        if display_metrics:
            await self.publish(
                RealtimeMetricsMessage(
                    service_id=self.service_id,
                    metrics=display_metrics,
                )
            )

    @background_task(interval=Environment.RECORD.CHECKPOINT_INTERVAL, immediate=False)
    async def _write_partial_checkpoint_task(self) -> None:
        """Periodically persist a partial aggregate snapshot for recovery."""
        if self.run.cfg.runtime.service_run_type != ServiceRunType.KUBERNETES:
            return

        new_count = await write_partial_checkpoint(
            tracker=self._records_tracker,
            error_tracker=self._error_tracker,
            processors=self._metric_results_processors,
            benchmark_config=self.run.cfg,
            checkpoint_path=self.run.cfg.artifacts.profile_export_partial_json_file,
            last_checkpoint_records=self._last_checkpoint_records,
        )
        if new_count != self._last_checkpoint_records:
            self._last_checkpoint_records = new_count
            self.debug(
                lambda: f"Wrote partial checkpoint to {self.run.cfg.artifacts.profile_export_partial_json_file}"
            )

    def _deliver_network_rtt_to_processors(self) -> None:
        """Set the run-level mean network RTT (ns) on each metric results processor.

        Two cases, resolved here just before MetricResultsProcessor.summarize():

        1. Manual mean (``--network-latency-mean``): if ``network_latency.mean_ms``
           is set, the NetworkLatencyManager service is never spawned; convert the
           mean ms to ns and deliver it directly.
        2. Automatic (``--network-latency-automatic``): the accumulator computed a
           mean over successful probe samples. If zero successful samples were
           collected, log a warning and apply no adjustment.

        A resolved RTT of 0 (or no RTT) is a no-op: the adjustment would emit
        network_adjusted_* metrics identical to the raw ones, so it is skipped.
        Also a no-op when network latency calibration is disabled entirely.
        """
        network_cfg = self.run.cfg.network_latency
        if not network_cfg.enabled:
            return

        if network_cfg.mean_ms is not None:
            rtt_ns: float | None = network_cfg.mean_ms * 1e6
        else:
            rtt_ns = (
                self._network_latency_accumulator.mean_rtt_ns
                if self._network_latency_accumulator is not None
                else None
            )
            if rtt_ns is None:
                self.warning(
                    "Network latency calibration enabled but no successful RTT "
                    "probes were collected; skipping network_adjusted_* metrics."
                )

        # A resolved RTT of 0/None is a no-op (adjusted == raw): skip injection so we
        # don't emit duplicate network_adjusted_* metrics. The None case already warned.
        if not rtt_ns:
            return

        if network_cfg.mean_ms is not None:
            self.notice(
                f"Network latency calibration: subtracting a fixed mean RTT of "
                f"{rtt_ns / 1e6:.3f} ms from latency metrics (network_adjusted_* metrics)."
            )
        else:
            sample_count = self._network_latency_accumulator.successful_sample_count
            self.notice(
                f"Network latency calibration: subtracting measured mean RTT of "
                f"{rtt_ns / 1e6:.3f} ms (over {sample_count} TCP-handshake probes) "
                "from latency metrics (network_adjusted_* metrics)."
            )

        for processor in self._metric_results_processors:
            set_rtt = getattr(processor, "set_network_rtt_ns", None)
            if callable(set_rtt):
                set_rtt(rtt_ns)

    async def _process_results(self, cancelled: bool) -> ProcessRecordsResult:
        """Process the results across all non-excluded phases."""
        self.debug(lambda: f"Processing records (cancelled: {cancelled})")
        self.info("Processing records results...")

        # Deliver the run-level mean network RTT to each metric results processor
        # BEFORE summarize() so network_adjusted_* metrics can be injected.
        self._deliver_network_rtt_to_processors()

        results, multi_turn_ttft_trend = await self._summarize_all_processors()
        await self._finalize_all_processors()
        # Flush per-record stream exporters (JSONL / CSV writers) BEFORE
        # publishing ProcessRecordsResultMessage. The publish triggers the
        # SystemController's shutdown + readiness-marker write, which races
        # the exporters' own ``@on_stop`` close — when records arrive late
        # (all after CREDIT_PHASE_COMPLETE) the on_stop hook is cancelled
        # mid-shutdown and the buffered records are lost, leaving no
        # profile_export.jsonl on disk. Flushing here mirrors the legacy
        # results-processor path in ``_finalize_all_processors`` (which is
        # also pre-publish) so the per-record files are durable before the
        # controller is told results are ready.
        await self._finalize_stream_exporters()

        result = self._build_records_result(
            results, cancelled=cancelled, multi_turn_ttft_trend=multi_turn_ttft_trend
        )
        await self.publish(
            ProcessRecordsResultMessage(
                service_id=self.service_id,
                results=result,
            )
        )

        # GPU telemetry is published independently of inference results so the
        # SystemController's shutdown coordination (which waits for telemetry
        # when it was enabled) is satisfied and the telemetry_data section is
        # written to the JSON export.
        await self._publish_telemetry_results()

        # Run analyzers and publish ProcessAllResultsMessage so the
        # SystemController's `_on_process_all_results_message` handler picks
        # up steady-state / energy-efficiency summaries for ExporterManager.
        # Failures here must not break the legacy path above — the
        # PROCESS_RECORDS_RESULT message has already been published.
        analyzer_outputs = await self._run_analyzers(
            result=result,
            cancelled=cancelled,
        )
        await self._publish_all_results(result, analyzer_outputs)
        return result

    def _process_telemetry_results(self) -> ProcessTelemetryResult:
        """Export accumulated GPU telemetry into a ProcessTelemetryResult.

        ``end_ns`` is intentionally left unset so the final telemetry scrape
        taken after PROFILE_COMPLETE (but before this export) is included in
        the stats. If no profiling phase ran, all collected data is included.
        """
        accumulator = self._gpu_telemetry_accumulator
        if accumulator is None:
            return ProcessTelemetryResult(results=None)

        error_summary = [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self._telemetry_state.error_counts.items()
        ]
        phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        telemetry_export_data = accumulator.export_results(
            start_ns=phase_stats.start_ns,
            error_summary=error_summary,
        )
        return ProcessTelemetryResult(results=telemetry_export_data)

    async def _publish_telemetry_results(self) -> None:
        """Publish GPU telemetry results via ProcessTelemetryResultMessage.

        Idempotent: only the first call publishes. Telemetry is published even
        when no accumulator is loaded (results=None) so the SystemController's
        wait-for-telemetry coordination always resolves.
        """
        if self._telemetry_result_published:
            return
        self._telemetry_result_published = True
        try:
            telemetry_result = self._process_telemetry_results()
            await self.publish(
                ProcessTelemetryResultMessage(
                    service_id=self.service_id,
                    telemetry_result=telemetry_result,
                )
            )
        except Exception as e:  # noqa: BLE001 - publish failure must not abort the result path
            self.error(f"Failed to publish telemetry results: {e!r}")

    async def _summarize_all_processors(
        self,
    ) -> tuple[list[Any], dict[int, MetricResult] | None]:
        async def _summarize_with_logging(
            processor: ResultsProcessorProtocol, idx: int
        ) -> list[MetricResult] | BaseException:
            name = processor.__class__.__name__
            self.debug(f"Starting summarize for processor {idx}: {name}")
            try:
                result = await asyncio.wait_for(
                    processor.summarize(),
                    timeout=Environment.RECORD.PROCESS_RECORDS_TIMEOUT,
                )
                self.debug(f"Completed summarize for processor {idx}: {name}")
                return result
            except Exception as e:
                self.error(f"Error in summarize for processor {idx}: {name}: {e!r}")
                raise

        results = await asyncio.gather(
            *[
                _summarize_with_logging(processor, idx)
                for idx, processor in enumerate(self._metric_results_processors)
            ],
            return_exceptions=True,
        )
        # The legacy ``MetricResultsProcessor`` was deleted by the
        # accumulator-pipeline port; ``MetricsAccumulator`` (registered under
        # the ``accumulator`` plugin category, not ``results_processor``) now
        # owns the metric percentile rollup. Bridge it into the legacy
        # ``ProcessRecordsResultMessage`` shape by appending its
        # ``list[MetricResult]`` and timeslices dict to the bucketable output
        # so ``bucket_summarize_results`` picks them up alongside whatever the
        # JSONL / CSV / accuracy processors returned. ``multi_turn_ttft_trend``
        # is returned separately because it's a dict[int, MetricResult] —
        # ``bucket_summarize_results`` would mis-route it to the timeslices
        # dict slot.
        multi_turn_ttft_trend: dict[int, MetricResult] | None = None
        metrics_acc = self._accumulators.get(AccumulatorType.METRIC_RESULTS)
        if metrics_acc is not None:
            try:
                acc_summary = await metrics_acc.summarize()
                results.append(list(acc_summary.results.values()))
                if acc_summary.timeslices is not None:
                    results.append(acc_summary.timeslices)
                multi_turn_ttft_trend = acc_summary.multi_turn_ttft_trend
            except Exception as e:  # noqa: BLE001 - accumulator failure must not abort legacy bucketing
                self.error(f"Error in MetricsAccumulator.summarize: {e!r}")
                results.append(e)
        return results, multi_turn_ttft_trend

    async def _finalize_all_processors(self) -> None:
        # Final flush of per-record streaming files BEFORE publishing the
        # result. Otherwise the controller writes the readiness marker and
        # flips results_exported=True while the JSONL/CSV files are still
        # mid-flush — the operator's progress poll then races a partial
        # per-record export.
        finalize_results = await asyncio.gather(
            *[processor.finalize() for processor in self._metric_results_processors],
            return_exceptions=True,
        )
        for exc in finalize_results:
            if isinstance(exc, BaseException):
                self.error(f"Error finalizing results processor: {exc!r}")

    def _build_records_result(
        self,
        results: list[Any],
        *,
        cancelled: bool,
        multi_turn_ttft_trend: dict[int, MetricResult] | None = None,
    ) -> ProcessRecordsResult:
        (
            records_results,
            timeslice_metric_results,
            error_results,
            raw_exceptions,
        ) = bucket_summarize_results(results)
        for exc in raw_exceptions:
            self.error(f"Exception processing results: {exc!r}")
            error_results.append(ErrorDetails.from_exception(exc))

        return build_process_records_result(
            records_results=records_results,
            timeslice_metric_results=timeslice_metric_results,
            error_results=error_results,
            tracker=self._records_tracker,
            error_tracker=self._error_tracker,
            cancelled=cancelled,
            multi_turn_ttft_trend=multi_turn_ttft_trend,
            branch_stats=self._latest_branch_stats,
        )

    async def _publish_all_results(
        self, result: ProcessRecordsResult, analyzer_outputs: dict[Any, Any]
    ) -> None:
        try:
            await self.publish(
                ProcessAllResultsMessage(
                    service_id=self.service_id,
                    results=result,
                    steady_state_results=analyzer_outputs.get(AnalyzerType.STEADY_STATE)
                    if hasattr(AnalyzerType, "STEADY_STATE")
                    else None,
                    # Populated controller-side in SystemController._export_results_data;
                    # the energy analyzer can't run records-manager-side because the
                    # GPU telemetry accumulator lives in a separate process.
                    energy_efficiency_results=None,
                )
            )
        except Exception as e:  # noqa: BLE001 - publish failure must not abort the legacy result path
            self.error(f"Failed to publish ProcessAllResultsMessage: {e!r}")

    async def _finalize_stream_exporters(self) -> None:
        """Flush all stream exporters concurrently; log per-exporter errors.

        Mirrors the legacy ``processor.finalize()`` fan-out in
        ``_process_results``. Stream exporters (e.g. JSONL writers) buffer
        records; without this flush the publish below races partial files
        the same way the legacy comment in ``_process_results`` describes.
        """
        if not self._stream_exporters:
            return
        results = await asyncio.gather(
            *[exporter.finalize() for exporter in self._stream_exporters.values()],
            return_exceptions=True,
        )
        for (exp_type, _), result in zip(
            self._stream_exporters.items(), results, strict=True
        ):
            if isinstance(result, BaseException):
                self.error(f"Stream exporter {exp_type} finalize failed: {result!r}")

    async def _run_analyzers(
        self,
        result: ProcessRecordsResult,
        cancelled: bool,
    ) -> dict[AnalyzerType, Any]:
        """Run all loaded analyzers in dependency order via SummaryContext.

        Returns the analyzer outputs map for callers to attach to outgoing
        messages. ``ProcessRecordsResult`` is passed in so we can use its
        time window — the records-tracker source of truth — without
        re-deriving it. Disabled / failing analyzers are skipped per
        ``compute_analyzer_outputs``'s policy.
        """
        if not self._analyzers:
            return {}

        profile_results = result.results
        start_ns = profile_results.start_ns if profile_results else 0
        end_ns = profile_results.end_ns if profile_results else 0

        summary_ctx = SummaryContext(
            accumulators=dict(self._accumulators),
            accumulator_outputs={},
            start_ns=start_ns or 0,
            end_ns=end_ns or 0,
            cancelled=cancelled,
        )
        return await compute_analyzer_outputs(
            self._analyzers,
            summary_ctx,
            log_error=self.error,
            log_debug=self.debug,
        )


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.RECORDS_MANAGER)


if __name__ == "__main__":
    main()

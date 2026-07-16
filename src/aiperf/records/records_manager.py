# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aiperf.accuracy.models import AccuracySummary, ProcessAccuracyResult
from aiperf.common.accumulator_protocols import (
    AccumulatorProtocol,
    ExportContext,
    StreamExporterProtocol,
    SummaryContext,
)
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_command, on_message, on_pull_message
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    DatasetConfiguredNotification,
    NetworkLatencyRecordMessage,
    ProcessAccuracyResultMessage,
    ProcessAllResultsMessage,
    ProcessRecordsCommand,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
    ProcessTelemetryResultMessage,
    ProfileCancelCommand,
    ProfileCompleteCommand,
    RealtimeMetricsCommand,
    RealtimeMetricsMessage,
    RecordsMessage,
    RecordsProcessingStatsMessage,
    ServerMetricsRecordMessage,
    StartRealtimeTelemetryCommand,
    TelemetryRecordsMessage,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    BranchStats,
    ErrorDetails,
    ErrorDetailsCount,
    MetricResult,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProcessServerMetricsResult,
    ProcessTelemetryResult,
    ProfileResults,
    TimesliceResult,
    WorkerProcessingStats,
)
from aiperf.common.utils import yield_to_event_loop
from aiperf.config.comm import ZMQDualBindConfig
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.gpu_telemetry.protocols import GPUTelemetryAccumulatorProtocol
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.network_latency.accumulator import NetworkLatencyAccumulator
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    AccumulatorType,
    PluginType,
    StreamExporterType,
    UIType,
)
from aiperf.records import records_manager_processing
from aiperf.records.dataset_gate import await_dataset_configured
from aiperf.records.error_tracker import ErrorTracker
from aiperf.records.records_manager_processing import (
    LoadedAnalyzer,
    generate_realtime_metrics,
    load_accumulators,
    load_analyzers,
    load_stream_exporters,
)
from aiperf.records.records_tracker import RecordsTracker
from aiperf.server_metrics.protocols import ServerMetricsAccumulatorProtocol

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.resolution.plan import BenchmarkRun


_LATENCY_LINE_LABELS: tuple[tuple[str, str], ...] = (
    ("ttft", "time_to_first_token"),
    # Use the scalar per-record metric (avg gap across the response), not the
    # list-valued ``inter_chunk_latency``. List metrics don't aggregate into
    # displayable percentiles in the realtime path, so the row used to show
    # only dashes mid-run even when the per-record JSONL had real values.
    ("itl", "inter_token_latency"),
    ("e2e", "request_latency"),
)
_INTERACTIVITY_LABEL: tuple[str, str] = (
    "intvty",
    "output_token_throughput_per_user",
)
_SEQ_LENGTH_LABELS: tuple[tuple[str, str], ...] = (
    ("isl", "input_sequence_length"),
    ("osl", "output_sequence_length"),
)
# Each block line is its own log record (carries its own log prefix), so the
# continuation rows sit at a small fixed indent under the header line rather
# than aligning under the old inline "[realtime MM:SS profiling] " text.
_REALTIME_ROW_INDENT = 2
# Percentile names per row group. Latency/interactivity rows report p95 in the
# third column; sequence-length rows report p90 there (the agentic long-tail is
# more interesting at p90 for token counts). Each row keeps its own ``pNN=``
# labels, so the column can hold p95 on one row and p90 on the next.
_LATENCY_PERCENTILES: tuple[str, ...] = ("p50", "p75", "p95", "p99")
_TOKEN_PERCENTILES: tuple[str, ...] = ("p50", "p75", "p90", "p99")


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    if total < 3600:
        return f"{total // 60:02d}:{total % 60:02d}"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1.0:
        return "<1ms"
    return f"{int(round(value)):,}ms"


def _format_int(value: float | None) -> str:
    """Compact int formatter for token-rate percentiles. Returns ``-`` for None."""
    if value is None:
        return "-"
    return f"{int(round(value)):,}"


def _render_realtime_block(
    metric_results: list[MetricResult],
    phase_stats: PhaseRecordsStats,
    prev_snapshot: tuple[int, float] | None,
    server_snapshot: dict[str, float] | None = None,
) -> str:
    """Render a compact realtime stats block for the aiperf logger.

    Format (``[realtime MM:SS profiling]`` header, a summary counter row, then
    one labeled percentile row per metric)::

        [realtime 00:49 profiling]
          rps=14.2 (avg 13.1)  tput_in=1,097,271/s  tput_out=10,441/s  done=641 ok=641 err=0
          ttft    p50=   30ms  p75=   48ms  p95=   106ms  p99=   155ms
          itl     p50=    5ms  p75=    5ms  p95=     5ms  p99=     5ms
          e2e     p50=2,241ms  p75=4,853ms  p95=13,526ms  p99=22,003ms
          intvty  p50=    200  p75=    201  p95=     211  p99=     254  (1/tpot tok/s)
          isl     p50= 67,234  p75= 97,141  p90= 179,564  p99= 384,325  (tokens)
          osl     p50=    443  p75=    967  p90=   2,034  p99=   4,396  (tokens)
          tot     in=53,555,186  out=509,605

    The header sits on its own line and the summary counters drop to the
    first indented row so the line no longer wraps in narrow terminals; each
    line is emitted as a separate log record (see ``_report_realtime_metrics``).
    Every row keeps its own ``pNN=`` labels (so it stays readable even when log
    lines from other services interleave), while the values are right-aligned in
    per-column widths so the digits and ``ms`` suffixes line up into a grid.

    Latency MetricResult percentile values are already in display units
    (milliseconds for time-based metrics, see ``to_display_unit`` and the
    accumulator's ``summarize`` path), so ``_format_ms`` consumes them as-is.
    Returns an empty string when no requests have completed yet so callers
    can suppress the block entirely on the first tick.

    Records-side stats only — ``in_flight_requests`` is a credit-side concept
    that this function doesn't have access to and is therefore omitted from
    the output.
    """
    if phase_stats.total_records == 0:
        return ""

    by_tag: dict[str, MetricResult] = {m.tag: m for m in metric_results}
    elapsed = phase_stats.records_elapsed_time

    rps_avg_mr = by_tag.get("request_throughput")
    rps_avg = getattr(rps_avg_mr, "avg", None)
    rps_avg_str = f"{rps_avg:.1f}" if rps_avg is not None else "-"

    if prev_snapshot is not None:
        prev_completed, prev_elapsed = prev_snapshot
        dt = elapsed - prev_elapsed
        rps_delta = (phase_stats.total_records - prev_completed) / dt if dt > 0 else 0.0
        rps_delta_str = f"{rps_delta:.1f}"
    else:
        rps_delta_str = rps_avg_str

    tput_out_mr = by_tag.get("output_token_throughput")
    tput_out_avg = getattr(tput_out_mr, "avg", None)
    tput_out_str = f"{int(round(tput_out_avg)):,}" if tput_out_avg is not None else "-"

    tput_in_mr = by_tag.get("input_token_throughput")
    tput_in_avg = getattr(tput_in_mr, "avg", None)
    tput_in_str = f"{int(round(tput_in_avg)):,}" if tput_in_avg is not None else "-"

    header = f"[realtime {_format_elapsed(elapsed)} profiling]"

    indent = " " * _REALTIME_ROW_INDENT

    # Build the percentile rows as (label, percentile_names, value_strings,
    # suffix) tuples first, so column widths can be derived from the actual
    # rendered values before any line is formatted. Latency/interactivity rows
    # use ms-formatted values; sequence-length rows use comma-grouped ints.
    #
    # Interactivity = 1 / inter-token-latency per request, percentiled across
    # requests. Characterizes the user-perceived decode speed; tail (low
    # percentile) is the slowest-decoding user, head (high percentile) is the
    # snappiest. Aggregate tput_in/tput_out on line 1 are bandwidth.
    StatRow = tuple[str, tuple[str, ...], list[str], str]
    stat_rows: list[StatRow] = []
    for label, tag in _LATENCY_LINE_LABELS:
        mr = by_tag.get(tag)
        values = [_format_ms(getattr(mr, p, None)) for p in _LATENCY_PERCENTILES]
        stat_rows.append((label, _LATENCY_PERCENTILES, values, ""))
    intvty_label, intvty_tag = _INTERACTIVITY_LABEL
    mr = by_tag.get(intvty_tag)
    stat_rows.append(
        (
            intvty_label,
            _LATENCY_PERCENTILES,
            [_format_int(getattr(mr, p, None)) for p in _LATENCY_PERCENTILES],
            "(1/tpot tok/s)",
        )
    )

    # Sequence-length distribution rows — useful for spotting long-tail
    # agentic prompts mid-run. Reads the same MetricResults the aggregator
    # already publishes; no extra plumbing. A row is omitted entirely when its
    # metric has no data, rather than rendering a row of dashes.
    for label, tag in _SEQ_LENGTH_LABELS:
        mr = by_tag.get(tag)
        values = [_format_int(getattr(mr, p, None)) for p in _TOKEN_PERCENTILES]
        if all(v == "-" for v in values):
            continue
        stat_rows.append((label, _TOKEN_PERCENTILES, values, "(tokens)"))

    label_w = max(len(label) for label, *_ in stat_rows)
    col_w = [max(len(values[i]) for _, _, values, _ in stat_rows) for i in range(4)]

    rows: list[str] = [
        f"{indent}rps={rps_delta_str} (avg {rps_avg_str})  "
        f"tput_in={tput_in_str}/s  "
        f"tput_out={tput_out_str}/s  "
        f"done={phase_stats.total_records:,} "
        f"ok={phase_stats.success_records:,} "
        f"err={phase_stats.error_records:,}"
    ]
    for label, percentiles, values, suffix in stat_rows:
        cells = "  ".join(
            f"{name}={value.rjust(col_w[i])}"
            for i, (name, value) in enumerate(zip(percentiles, values, strict=True))
        )
        line = f"{indent}{label:<{label_w}}  {cells}"
        rows.append(f"{line}  {suffix}" if suffix else line)

    # Cumulative token totals — running counters, useful for spotting
    # whether the ratio of output:input tokens is matching the workload's
    # expected agentic pattern.
    total_isl_mr = by_tag.get("total_isl")
    total_osl_mr = by_tag.get("total_osl")
    total_isl = getattr(total_isl_mr, "avg", None)
    total_osl = getattr(total_osl_mr, "avg", None)
    if total_isl is not None or total_osl is not None:
        in_str = f"{int(round(total_isl)):,}" if total_isl is not None else "-"
        out_str = f"{int(round(total_osl)):,}" if total_osl is not None else "-"
        rows.append(f"{indent}{'tot':<{label_w}}  in={in_str}  out={out_str}")

    # Server-side row — cumulative cache hit rate, KV usage, and scheduler
    # queue depth from the live ServerMetricsAccumulator snapshot. Sourced
    # from the /metrics scrape, so populates only when server-metrics
    # collection is enabled and the inference server actually serves
    # Prometheus. Each part is rendered only when its backing metric is
    # present, so e.g. cpu_kv / ext_cache_hit show up only on offload=cpu
    # runs.
    if server_snapshot:
        srv_parts: list[str] = []
        if "prefix_cache_hit_rate" in server_snapshot:
            srv_parts.append(
                f"prefix_cache_hit={server_snapshot['prefix_cache_hit_rate']:.1f}%"
            )
        if "unique_input_tokens_srv" in server_snapshot:
            srv_parts.append(
                f"unique_in_srv={int(round(server_snapshot['unique_input_tokens_srv'])):,}"
            )
        if "external_prefix_cache_hit_rate" in server_snapshot:
            srv_parts.append(
                f"ext_cache_hit={server_snapshot['external_prefix_cache_hit_rate']:.1f}%"
            )
        if "kv_cache_usage_pct" in server_snapshot:
            srv_parts.append(f"kv_usage={server_snapshot['kv_cache_usage_pct']:.1f}%")
        if "cpu_kv_cache_usage_pct" in server_snapshot:
            srv_parts.append(
                f"cpu_kv_usage={server_snapshot['cpu_kv_cache_usage_pct']:.1f}%"
            )
        if "num_running" in server_snapshot or "num_waiting" in server_snapshot:
            running = int(server_snapshot.get("num_running", 0))
            waiting = int(server_snapshot.get("num_waiting", 0))
            srv_parts.append(f"queue={running}r/{waiting}w")
        if "input_token_throughput_srv" in server_snapshot:
            srv_parts.append(
                f"tput_in_srv={int(round(server_snapshot['input_token_throughput_srv'])):,}/s"
            )
        if "output_token_throughput_srv" in server_snapshot:
            srv_parts.append(
                f"tput_out_srv={int(round(server_snapshot['output_token_throughput_srv'])):,}/s"
            )
        if srv_parts:
            rows.append(f"{indent}{'srv':<{label_w}} {' '.join(srv_parts)}")

    return "\n".join([header, *rows])


@dataclass
class ErrorTrackingState:
    """State container for tracking errors with counts and thread-safe access.

    Provides common error tracking functionality for all metrics subsystems
    (telemetry, server metrics, regular metrics).
    """

    error_counts: dict[ErrorDetails, int] = field(
        default_factory=lambda: defaultdict(int)
    )


class RecordsManager(PullClientMixin, BaseComponentService):
    """Collects and processes benchmark results from workers.

    The RecordsManager receives metric records from workers and accumulates them
    for final processing. The timing manager is the ground truth for what requests
    completed within the benchmark window - when it signals phase completion with
    a final_completed_count, the RecordsManager waits until it has processed that
    many records before finalizing results.
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        # For dual-bind mode (Kubernetes), also bind to TCP for remote record processors.
        # Controller binds to IPC + TCP; workers connect via TCP.
        additional_bind_address: str | None = None
        comm_config = run.cfg.comm_config
        if (
            isinstance(comm_config, ZMQDualBindConfig)
            and not comm_config.controller_host
        ):
            additional_bind_address = comm_config.records_push_pull_tcp_bind_address

        super().__init__(
            run=run,
            service_id=service_id,
            pull_client_address=CommAddress.RECORDS,
            pull_client_bind=True,
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
            pull_client_additional_bind_address=additional_bind_address,
            **kwargs,
        )

        self._records_tracker = RecordsTracker()
        self._error_tracker = ErrorTracker()

        # DatasetConfiguredNotification (SUB) and metric records (PULL) arrive on
        # independent channels with no ordering guarantee. Gate record processing on
        # this event so results processors are configured (e.g. accuracy task names)
        # before any record is accumulated.
        self._dataset_configured_event: asyncio.Event = asyncio.Event()

        self._previous_realtime_records: int | None = None
        # Server-metric snapshot from the prior realtime tick. The realtime block
        # must re-render when live server metrics (cache hit rate, KV usage,
        # queue depth) move even while the record count is momentarily static --
        # gating on record count alone froze the server row during lulls.
        self._previous_realtime_server_snapshot: dict[str, float] | None = None
        # (completed_records, elapsed_seconds) from the prior realtime tick, used
        # to render the instantaneous (delta) RPS in the realtime stats block.
        self._prev_realtime_snapshot: tuple[int, float] | None = None

        # Latest BranchStats snapshot received via CreditPhaseCompleteMessage
        # for the PROFILING phase. None for non-DAG runs (TimingManager
        # publishes None when no BranchOrchestrator is wired). Spliced
        # into ProfileResults when the records pipeline finalizes.
        self._latest_branch_stats: BranchStats | None = None

        # Per-phase BranchStats snapshots. Populated on every
        # CreditPhaseCompleteMessage that carries a non-None
        # ``branch_stats``; ``_snapshot_branch_stats`` reads back the
        # value for a specific phase. Used by analyzer paths that need
        # warmup vs profiling separation.
        self._phase_branch_stats: dict[CreditPhase, BranchStats] = {}
        self._complete_credit_phases: set[CreditPhase] = set()

        self._telemetry_state = ErrorTrackingState()
        self._server_metrics_state = ErrorTrackingState()

        self._gpu_telemetry_accumulator: GPUTelemetryAccumulatorProtocol | None = None
        self._server_metrics_accumulator: ServerMetricsAccumulatorProtocol | None = None

        # In-process accumulator for RTT probe samples. Computes the run-level
        # mean RTT delivered to MetricsAccumulator before summarize(). None unless
        # network latency probing is active.
        self._network_latency_accumulator: NetworkLatencyAccumulator | None = (
            NetworkLatencyAccumulator(benchmark_id=self.run.benchmark_id)
            if self.run.cfg.network_latency.should_probe
            else None
        )
        self._network_latency_state = ErrorTrackingState()

        self._accumulators: dict[AccumulatorType, AccumulatorProtocol] = (
            load_accumulators(self)
        )
        self._stream_exporters: dict[StreamExporterType, StreamExporterProtocol] = (
            load_stream_exporters(self)
        )
        # Summarize-time cross-accumulator analyzers (e.g. energy efficiency),
        # each carrying its live-instance and summary dependencies.
        self._analyzers: list[LoadedAnalyzer] = load_analyzers(self)
        self._routing_table = self._build_routing_table()
        self._warned_unrouted_record_types: set[str] = set()
        self._log_routing_table()

        # Single-flight guard for _process_results: the background finalize task,
        # the PROCESS_RECORDS command, and PROFILE_CANCEL can all reach it and
        # would otherwise double-publish and double-finalize stream exporters.
        self._process_results_lock = asyncio.Lock()
        self._processed_results: dict[CreditPhase, ProcessRecordsResult] = {}

        self._metric_record_accumulators = [
            accumulator
            for accumulator in self._accumulators.values()
            if accumulator in self._routing_table.get("metric_records", [])
        ]
        self._gpu_telemetry_accumulator = self._accumulators.get(
            AccumulatorType.GPU_TELEMETRY
        )
        self._server_metrics_accumulator = self._accumulators.get(
            AccumulatorType.SERVER_METRICS
        )
        self._accuracy_accumulator = self._accumulators.get(AccumulatorType.ACCURACY)

    def _build_routing_table(self) -> dict[str, list[Any]]:
        """Build record_type string -> handler mapping from plugin metadata."""
        table: dict[str, list[Any]] = {}
        for entry in plugins.iter_entries(PluginType.ACCUMULATOR):
            handler = self._accumulators.get(AccumulatorType(entry.name))
            if handler is None:
                continue
            record_types = (
                entry.metadata.get("record_types", []) if entry.metadata else []
            )
            for record_type in record_types:
                table.setdefault(record_type, []).append(handler)

        for entry in plugins.iter_entries(PluginType.STREAM_EXPORTER):
            handler = self._stream_exporters.get(StreamExporterType(entry.name))
            if handler is None:
                continue
            record_types = (
                entry.metadata.get("record_types", []) if entry.metadata else []
            )
            for record_type in record_types:
                table.setdefault(record_type, []).append(handler)
        return table

    async def _dispatch_record(self, record: Any) -> list[BaseException]:
        """Dispatch one typed record to all handlers registered for its record_type."""
        record_type = getattr(record, "record_type", None)
        if record_type is None:
            error = TypeError(f"Record {type(record).__name__} has no record_type")
            self.error(str(error))
            return [error]

        handlers = self._routing_table.get(record_type, [])
        if not handlers:
            # Warn once per unrouted type: records silently vanish here while the
            # request still counts as a success, so this must not stay debug-only.
            if record_type not in self._warned_unrouted_record_types:
                self._warned_unrouted_record_types.add(record_type)
                self.warning(
                    f"No handlers registered for record type {record_type!r}; "
                    "records of this type are being dropped. Check that a producer's "
                    "record_type matches an accumulator/stream_exporter record_types "
                    "entry in plugins.yaml."
                )
            return []

        results = await asyncio.gather(
            *[handler.process_record(record) for handler in handlers],
            return_exceptions=True,
        )
        errors: list[BaseException] = []
        for handler, result in zip(handlers, results, strict=True):
            # A handler-level CancelledError (captured by return_exceptions) means
            # one handler's coroutine was cancelled, NOT this task -- genuine task
            # cancellation makes the gather itself raise and never reaches here. We
            # must count it like any other handler failure rather than re-raising,
            # or the caller skips the tracker update + (timeout-less) completion
            # barrier and the phase never converges.
            if isinstance(result, BaseException):
                self.error(
                    f"Handler {handler.__class__.__name__} failed for "
                    f"{record_type}: {result!r}"
                )
                errors.append(result)
        return errors

    def _log_routing_table(self) -> None:
        """Log the metadata-derived record routing table."""
        self.debug(
            lambda: (
                f"Routing table: {len(self._accumulators)} accumulators, "
                f"{len(self._stream_exporters)} stream exporters, "
                f"{len(self._routing_table)} record types"
            )
        )
        for record_type, handlers in self._routing_table.items():
            handler_names = [handler.__class__.__name__ for handler in handlers]
            self.debug(lambda rt=record_type, hn=handler_names: f"  {rt} -> {hn}")

    @on_pull_message(MessageType.RECORDS)
    async def _on_records(self, message: RecordsMessage) -> None:
        """Handle a per-request records envelope generically.

        One ``RecordsMessage`` == one inference request. Each contained record
        self-identifies via its serialized ``record_type`` field and is dispatched
        to its registered handlers; the per-request lockstep keys off the message
        envelope (``message.metadata`` / ``message.error``), never off sniffing a
        record type.
        """
        if not await await_dataset_configured(self, self._dataset_configured_event):
            return
        if self.is_trace_enabled:
            self.trace(f"Received records: {message}")

        dispatch_errors: list[BaseException] = []
        for record in message.records:
            dispatch_errors.extend(await self._dispatch_record(record))

        phase = message.metadata.benchmark_phase
        self._records_tracker.update_from_request(message.metadata, message.error)
        if message.error:
            self._error_tracker.increment_error_count_for_phase(phase, message.error)
        # A metric accumulator/exporter that failed to ingest this record yields
        # incomplete metrics; surface it in the phase error summary rather than
        # marking the record cleanly processed and silently dropping the failure.
        for error in dispatch_errors:
            self._error_tracker.increment_error_count_for_phase(
                phase, ErrorDetails.from_exception(error)
            )

        if (
            phase in self._complete_credit_phases
            and self._records_tracker.check_and_set_all_records_received_for_phase(
                phase
            )
        ):
            await self._handle_all_records_received(phase)

    @on_pull_message(MessageType.TELEMETRY_RECORDS)
    async def _on_telemetry_records(self, message: TelemetryRecordsMessage) -> None:
        """Handle telemetry records message from Telemetry Manager."""
        if message.valid:
            for record in message.records:
                for error in await self._dispatch_record(record):
                    self._telemetry_state.error_counts[
                        ErrorDetails.from_exception(error)
                    ] += 1
        elif message.error:
            self._telemetry_state.error_counts[message.error] += 1

    @on_pull_message(MessageType.SERVER_METRICS_RECORD)
    async def _on_server_metrics_records(
        self, message: ServerMetricsRecordMessage
    ) -> None:
        """Handle server metrics record message from Server Metrics Manager."""
        if message.valid:
            for error in await self._dispatch_record(message.record):
                self._server_metrics_state.error_counts[
                    ErrorDetails.from_exception(error)
                ] += 1
        elif message.error:
            self._server_metrics_state.error_counts[message.error] += 1

    @on_pull_message(MessageType.NETWORK_LATENCY_RECORD)
    async def _on_network_latency_records(
        self, message: NetworkLatencyRecordMessage
    ) -> None:
        """Handle a network latency RTT probe sample from the NetworkLatencyManager."""
        if message.valid:
            if self._network_latency_accumulator is not None:
                self._network_latency_accumulator.add_sample(message.sample)
            for error in await self._dispatch_record(message.sample):
                self._network_latency_state.error_counts[
                    ErrorDetails.from_exception(error)
                ] += 1
        elif message.error:
            self._network_latency_state.error_counts[message.error] += 1

    async def _handle_all_records_received(self, phase: CreditPhase) -> None:
        """Handle the case where all records have been received."""
        if phase != CreditPhase.PROFILING:
            self.debug(lambda: f"Skipping non-profiling phase: {phase}")
            return

        phase_stats = self._records_tracker.create_stats_for_phase(phase)
        self.info(
            lambda: (
                f"Processed {phase_stats.success_records} valid requests and {phase_stats.error_records} errors ({phase_stats.total_records} total)."
            )
        )

        self.info("Received all records, processing now...")
        self.execute_async(
            self._finalize_and_process_results(
                phase=phase,
                cancelled=self._records_tracker.was_phase_cancelled(phase),
            )
        )
        await yield_to_event_loop()

    async def _finalize_and_process_results(
        self, phase: CreditPhase, cancelled: bool
    ) -> None:
        """Finalize server metrics collection and process results.

        This runs as a background task to avoid blocking the message pump.
        """
        phase_stats = self._records_tracker.create_stats_for_phase(phase)

        # Send a message to the event bus to signal that we received all the records
        await self.publish(
            AllRecordsReceivedMessage(
                service_id=self.service_id,
                request_ns=time.time_ns(),
                final_processing_stats=phase_stats,
            )
        )

        # Trigger final server metrics scrape and wait for completion
        # This ensures final metrics are pushed before we export results
        response = await self.send_command_and_wait_for_response(
            ProfileCompleteCommand(service_id=self.service_id), timeout=10.0
        )

        if isinstance(response, ErrorDetails):
            self.warning(f"Server metrics final scrape timed out or failed: {response}")
        else:
            self.debug("Server metrics final scrape completed")

        self.debug("Waiting for server metrics flush period...")
        # Wait for server metrics flush period to allow final metrics to be collected
        # This ensures metrics that are still being processed by the server are captured
        flush_period = Environment.SERVER_METRICS.COLLECTION_FLUSH_PERIOD
        phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        flush_end_ns = (phase_stats.requests_end_ns or time.time_ns()) + (
            (flush_period or 0) * NANOS_PER_SECOND
        )
        sleep_dur_sec = (flush_end_ns - time.time_ns()) / NANOS_PER_SECOND
        if sleep_dur_sec > 0:
            self.info(
                f"Waiting {sleep_dur_sec:.1f}s for server metrics flush period..."
            )
            await asyncio.sleep(sleep_dur_sec)

        self.debug("Server metrics flush period complete, processing now...")
        await self._process_results(phase=phase, cancelled=cancelled)
        self.info("_finalize_and_process_results completed")

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        for handler in (*self._accumulators.values(), *self._stream_exporters.values()):
            if hasattr(handler, "on_dataset_configured"):
                handler.on_dataset_configured(message.metadata)
        self._dataset_configured_event.set()

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(
        self, phase_start_msg: CreditPhaseStartMessage
    ) -> None:
        """Handle a credit phase start message in order to track the total number of expected requests."""
        self._records_tracker.update_phase_info(phase_start_msg.stats)
        await self._dispatch_record(phase_start_msg.stats)
        self.info(f"Credit phase start: {phase_start_msg.config.phase}")

    @on_message(MessageType.CREDIT_PHASE_PROGRESS)
    async def _on_credit_phase_progress(
        self, message: CreditPhaseProgressMessage
    ) -> None:
        """Handle a credit phase progress message to track and stream live timing snapshots."""
        self._records_tracker.update_phase_info(message.stats)
        await self._dispatch_record(message.stats)

    @on_message(MessageType.CREDIT_PHASE_SENDING_COMPLETE)
    async def _on_credit_phase_sending_complete(
        self, message: CreditPhaseSendingCompleteMessage
    ) -> None:
        """Handle a credit phase sending complete message in order to track the final request count."""
        if message.stats.phase == CreditPhase.PROFILING:
            self.info(
                f"Sent {message.stats.final_requests_sent:,} requests. Waiting for all to complete..."
            )
        self._records_tracker.update_phase_info(message.stats)
        await self._dispatch_record(message.stats)

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, message: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message in order to track the end time, and check if all records have been received."""
        self._records_tracker.update_phase_info(message.stats)
        await self._dispatch_record(message.stats)
        self._complete_credit_phases.add(message.stats.phase)
        # Capture per-phase BranchStats for any phase that publishes them.
        if message.branch_stats is not None:
            self._phase_branch_stats[message.stats.phase] = message.branch_stats
        if message.stats.phase == CreditPhase.PROFILING:
            # Capture the BranchStats snapshot so it flows into
            # ProfileResults when the records pipeline finalizes.
            # Non-DAG runs publish None and leave this unset.
            if message.branch_stats is not None:
                self._latest_branch_stats = message.branch_stats
            phase_stats = self._records_tracker.create_stats_for_phase(
                message.stats.phase
            )
            self.info(
                lambda: (
                    f"Received CREDIT_PHASE_COMPLETE message, Phase complete: {phase_stats!r}"
                )
            )
            self.notice(
                f"All requests have completed, please wait for the results to be processed "
                f"(currently {phase_stats.total_records:,} of {phase_stats.final_requests_completed:,} records processed)..."
            )

        # This check is to prevent a race condition where the records manager processes
        # all records before the timing manager has sent the final completed count.
        if self._records_tracker.check_and_set_all_records_received_for_phase(
            message.stats.phase
        ):
            await self._handle_all_records_received(message.stats.phase)

    def _snapshot_branch_stats(self, phase: CreditPhase) -> BranchStats | None:
        """Return the orchestrator-published BranchStats for ``phase``.

        Returns ``None`` for non-DAG runs or for phases where the
        TimingManager never published sub-agent counters on
        ``CreditPhaseCompleteMessage``.
        """
        return self._phase_branch_stats.get(phase)

    @on_message(MessageType.CREDITS_COMPLETE)
    async def _on_credits_complete(self, message: CreditsCompleteMessage) -> None:
        """Handle a credits complete message in order to track the end time, and check if all records have been received."""
        self.info(
            "All credits complete, please wait for the results to be processed..."
        )
        if (
            CreditPhase.PROFILING in self._complete_credit_phases
            and self._records_tracker.check_and_set_all_records_received_for_phase(
                CreditPhase.PROFILING
            )
        ):
            await self._handle_all_records_received(CreditPhase.PROFILING)

    @background_task(
        interval=Environment.RECORD.PROGRESS_REPORT_INTERVAL, immediate=False
    )
    async def _report_records_task(self) -> None:
        """Report the records processing stats."""
        active_phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        if active_phase_stats.total_records == 0:
            return  # TODO: What about worker stats?
        overall_worker_stats = self._records_tracker.create_overall_worker_stats()
        await self._publish_processing_stats(active_phase_stats, overall_worker_stats)

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
        self, message: ProcessRecordsCommand
    ) -> ProcessRecordsResult:
        """Handle the process records command by forwarding it to all of the results processors, and returning the results."""
        self.debug(lambda: f"Received process records command: {message}")
        return await self._process_results(
            phase=CreditPhase.PROFILING, cancelled=message.cancelled
        )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _on_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> ProcessRecordsResult:
        """Handle the profile cancel command by processing current results.

        This marks the phase as cancelled in the records tracker and processes
        all currently received records. Called when user presses Ctrl+C.
        """
        self.warning(f"Received profile cancel command: {message}")

        # Mark the phase as cancelled in the tracker
        self._records_tracker.mark_phase_cancelled(CreditPhase.PROFILING)

        return await self._process_results(phase=CreditPhase.PROFILING, cancelled=True)

    @property
    def service_config(self) -> BenchmarkConfig:
        """The resolved benchmark config for this run.

        Compatibility accessor for the realtime-stats path: the renderer gate
        reads ``service_config.ui_type`` so headless runs emit the per-tick log
        block while ``--ui dashboard`` suppresses it (the dashboard renders the
        same metrics itself).
        """
        return self.run.cfg

    @background_task(interval=None, immediate=True)
    async def _report_realtime_inference_metrics_task(self) -> None:
        """Report inference metrics at regular intervals.

        The dashboard/realtime gate is checked inside the loop so the framework's
        ``interval=None`` semantics (run body once and break) don't permanently
        kill the task when the gate is currently False — see
        ``task_manager_mixin.py`` rule for ``interval=None``.

        ``--stats-interval 0`` disables only the per-tick log block. The
        ``RealtimeMetricsMessage`` keeps publishing for dashboards / k8s job-WS
        subscribers at the per-UI default cadence (the value
        ``realtime_metrics_interval`` returns when unset), so a
        ``--ui dashboard --stats-interval 0`` run still drives the live panel.
        """
        configured_interval = self.run.cfg.runtime.realtime_metrics_interval(
            self.run.cfg.ui_type
        )
        log_block_enabled = configured_interval != 0
        # When the log block is disabled (interval 0), still tick the publish
        # loop at a sane per-UI default cadence instead of busy-spinning.
        interval = (
            configured_interval
            if log_block_enabled
            else self._default_realtime_interval()
        )
        while not self.stop_requested:
            await asyncio.sleep(interval)

            if (
                self.run.cfg.ui_type != UIType.DASHBOARD
                and not Environment.UI.REALTIME_METRICS_ENABLED
            ):
                continue

            phase_stats = self._records_tracker.create_stats_for_phase(
                CreditPhase.PROFILING
            )
            server_snapshot = self._collect_realtime_server_snapshot(
                start_ns=phase_stats.start_ns
            )
            if not self._has_realtime_update(
                phase_stats.total_records, server_snapshot
            ):
                continue
            self._previous_realtime_records = phase_stats.total_records
            self._previous_realtime_server_snapshot = dict(server_snapshot)
            await self._report_realtime_metrics(
                server_snapshot=server_snapshot,
                emit_log_block=log_block_enabled,
            )

    def _default_realtime_interval(self) -> float:
        """Resolve the per-UI default realtime cadence (interval-0 publish fallback).

        Mirrors ``realtime_metrics_interval`` when ``REALTIME_METRICS_INTERVAL``
        is unset: 5.0s under ``--ui dashboard``, 30.0s otherwise. Used so the
        dashboard keeps polling even when the log block is disabled with
        ``--stats-interval 0``.
        """
        return 5.0 if self.run.cfg.ui_type == UIType.DASHBOARD else 30.0

    def _has_realtime_update(
        self, total_records: int, server_snapshot: dict[str, float]
    ) -> bool:
        """Whether the realtime block needs rebuilding this tick.

        True when EITHER the record count OR the live server-metrics snapshot
        (cache hit rate, KV usage, queue depth) changed since the last emit.
        Gating on record count alone froze the server-metrics row whenever the
        count was momentarily static during a lull.
        """
        return (
            total_records != self._previous_realtime_records
            or server_snapshot != self._previous_realtime_server_snapshot
        )

    @on_command(CommandType.START_REALTIME_TELEMETRY)
    async def _on_start_realtime_telemetry_command(
        self, message: StartRealtimeTelemetryCommand
    ) -> None:
        """Handle command to start the realtime telemetry background task.

        This is called when the user dynamically enables the telemetry dashboard
        by pressing the telemetry option in the UI without having passed the 'dashboard' parameter
        at startup.
        """
        if self._gpu_telemetry_accumulator:
            self._gpu_telemetry_accumulator.start_realtime_telemetry()
        else:
            self.error(
                "GPU telemetry accumulator not found, cannot start realtime telemetry"
            )

    @on_command(CommandType.REALTIME_METRICS)
    async def _on_realtime_metrics_command(
        self, message: RealtimeMetricsCommand
    ) -> None:
        """Handle a real-time metrics command."""
        await self._report_realtime_metrics()

    def _collect_realtime_server_snapshot(
        self, start_ns: int | None = None
    ) -> dict[str, float]:
        """Return the current live server metrics snapshot, if available."""
        server_snapshot: dict[str, float] = {}
        if self._server_metrics_accumulator is None:
            return server_snapshot
        try:
            snapshot_fn = getattr(
                self._server_metrics_accumulator,
                "realtime_snapshot",
                None,
            )
            if callable(snapshot_fn):
                server_snapshot = snapshot_fn(start_ns=start_ns) or {}
        except Exception as exc:  # noqa: BLE001
            self.debug(lambda exc=exc: f"server_snapshot failed: {exc!r}")
        return server_snapshot

    async def _report_realtime_metrics(
        self,
        server_snapshot: dict[str, float] | None = None,
        emit_log_block: bool = True,
    ) -> None:
        """Report inference metrics (used by command handler).

        Publishes a ``RealtimeMetricsMessage`` for the dashboard / k8s job-WS
        subscribers, then — for non-dashboard UIs — renders and emits the
        per-tick realtime stats log block (one log record per line so the rows
        don't interleave with other services' writes on the shared console
        stream). The dashboard renders the same metrics itself, so the log
        block is suppressed under ``--ui dashboard``. ``emit_log_block=False``
        (set when ``--stats-interval 0`` disables the log block) suppresses the
        log line while still publishing the message for dashboards.
        """
        # Realtime metrics only need the metric_records accumulators —
        # GPU telemetry / server metrics live on separate fan-outs.
        raw_metrics = await generate_realtime_metrics(self._metric_record_accumulators)
        if not raw_metrics:
            return

        display_metrics = records_manager_processing.filter_display_metrics(raw_metrics)
        if not display_metrics:
            return
        await self.publish(
            RealtimeMetricsMessage(
                service_id=self.service_id,
                metrics=display_metrics,
            )
        )

        phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        # Realtime block uses the *raw* (unfiltered) metric set so per-user
        # throughput rows can show ``prefill_throughput_per_user`` etc. —
        # those have ``console_group=NONE`` (hidden from the dashboard table)
        # and ``filter_display_metrics`` strips them, leaving the row blank.
        if server_snapshot is None:
            server_snapshot = self._collect_realtime_server_snapshot(
                start_ns=phase_stats.start_ns
            )

        rendered = _render_realtime_block(
            raw_metrics,
            phase_stats,
            self._prev_realtime_snapshot,
            server_snapshot=server_snapshot,
        )
        if rendered:
            self._prev_realtime_snapshot = (
                phase_stats.total_records,
                phase_stats.records_elapsed_time,
            )
            if emit_log_block and self.run.cfg.ui_type != UIType.DASHBOARD:
                # One record per line: multi-line records interleave with
                # other services' writes on the shared console stream.
                for line in rendered.splitlines():
                    self.info(line)

    async def _run_analyzers(self, ctx: SummaryContext) -> list[MetricResult]:
        """Run summarize-time analyzer plugins that join across accumulators.

        An analyzer is skipped unless every accumulator it needs a LIVE instance
        of (``required_accumulators``) is loaded AND every accumulator whose
        SUMMARY it reads (``required_summaries``) was produced — e.g. the
        energy-efficiency analyzer queries the live GPU accumulator and reads the
        metrics summary. One analyzer's failure is logged and does not abort the
        rest. Returns the flattened MetricResults to merge into the summary.
        """
        if not self._analyzers:
            return []
        loaded = {str(acc_type) for acc_type in self._accumulators}
        summarized = {str(acc_type) for acc_type in ctx.accumulator_outputs}
        results: list[MetricResult] = []
        for loaded_analyzer in self._analyzers:
            analyzer = loaded_analyzer.analyzer
            name = analyzer.__class__.__name__
            missing_acc = [
                r for r in loaded_analyzer.required_accumulators if r not in loaded
            ]
            missing_sum = [
                r for r in loaded_analyzer.required_summaries if r not in summarized
            ]
            if missing_acc or missing_sum:
                self.debug(
                    lambda n=name, a=missing_acc, s=missing_sum: (
                        f"Skipping analyzer {n}: missing accumulators {a}, summaries {s}"
                    )
                )
                continue
            try:
                results.extend(await analyzer.analyze(ctx))
            except Exception as e:  # noqa: BLE001 - one analyzer must not abort the summary
                self.error(f"Analyzer {name} failed: {e!r}")
        return results

    async def _summarize_one_accumulator(
        self,
        acc_type: AccumulatorType,
        accumulator: AccumulatorProtocol,
        ctx: ExportContext,
    ) -> tuple[AccumulatorType, object]:
        """Run summarize/export_results on a single accumulator with timeout.

        Returns the result (or exception object) so a single bad accumulator
        cannot abort the rest. Accumulators that support phase/window-scoped
        export (marked with ``supports_phase_scoped_export`` — MetricsAccumulator)
        get ``export_results(ctx)`` so warmup
        records are excluded from profiling summaries; otherwise prefers
        ``summarize()`` and falls back to ``export_results(ctx)``.
        """
        name = accumulator.__class__.__name__
        self.debug(f"Starting summarize for accumulator {acc_type}: {name}")
        try:
            # ``is True`` (not truthiness) so a MagicMock's auto-created attribute
            # does not spuriously route mock accumulators through export_results.
            if getattr(accumulator, "supports_phase_scoped_export", False) is True and (
                hasattr(accumulator, "export_results")
            ):
                res = await asyncio.wait_for(
                    accumulator.export_results(ctx),
                    timeout=Environment.RECORD.PROCESS_RECORDS_TIMEOUT,
                )
            elif hasattr(accumulator, "summarize"):
                res = await asyncio.wait_for(
                    accumulator.summarize(),
                    timeout=Environment.RECORD.PROCESS_RECORDS_TIMEOUT,
                )
            else:
                res = await asyncio.wait_for(
                    accumulator.export_results(ctx),
                    timeout=Environment.RECORD.PROCESS_RECORDS_TIMEOUT,
                )
            self.debug(f"Completed summarize for accumulator {acc_type}: {name}")
            return acc_type, res
        except Exception as e:  # noqa: BLE001 - one bad accumulator must not abort the rest
            self.error(f"Error in summarize for accumulator {acc_type} ({name}): {e!r}")
            return acc_type, e

    def _bucket_accumulator_summary(
        self,
        acc_type: AccumulatorType,
        summary: object,
        records_results: list[MetricResult],
        error_results: list[ErrorDetails],
    ) -> list[TimesliceResult]:
        """Route a single accumulator summary into the right ProfileResults bucket."""
        timeslices: list[TimesliceResult] = []
        if isinstance(summary, BaseException):
            error_results.append(ErrorDetails.from_exception(summary))
        elif isinstance(summary, AccumulatorMetricsSummary):
            records_results.extend(summary.results.values())
            if summary.timeslices is not None:
                timeslices = summary.timeslices
        elif isinstance(summary, list):
            records_results.extend(r for r in summary if isinstance(r, MetricResult))
        elif isinstance(summary, ErrorDetails):
            error_results.append(summary)
        else:
            self.debug(
                lambda s=summary, a=acc_type: (
                    f"Accumulator {a} returned unrecognized shape: {type(s).__name__}"
                )
            )
        return timeslices

    async def _summarize_metric_record_accumulators(
        self, phase: CreditPhase, cancelled: bool
    ) -> tuple[
        list[MetricResult], list[TimesliceResult], list[ErrorDetails], SummaryContext
    ]:
        """Summarize the metric_records accumulators (the byte-exact engine).

        Telemetry / server-metrics accumulators are summarized separately via
        the dedicated ``_publish_telemetry_results`` / ``_publish_server_metrics_results``
        side-channels so they are not double-processed here.

        Also returns a populated ``SummaryContext`` — every loaded accumulator
        instance plus the metric_records summaries keyed by ``AccumulatorType`` —
        so summarize-time ``analyzer`` plugins can join across accumulators
        (e.g. energy efficiency joins GPU telemetry to inference tokens).
        """
        records_results: list[MetricResult] = []
        timeslices: list[TimesliceResult] = []
        error_results: list[ErrorDetails] = []

        phase_stats = self._records_tracker.create_stats_for_phase(phase)
        summary_ctx = SummaryContext(
            accumulators=dict(self._accumulators),
            start_ns=phase_stats.start_ns or 0,
            end_ns=phase_stats.requests_end_ns or 0,
            phase=phase,
            cancelled=cancelled,
        )

        # Only the metric_records-typed accumulators feed the summary records.
        acc_items = [
            (acc_type, acc)
            for acc_type, acc in self._accumulators.items()
            if acc in self._metric_record_accumulators
        ]
        if not acc_items:
            return records_results, timeslices, error_results, summary_ctx

        ctx = ExportContext(
            start_ns=phase_stats.start_ns,
            end_ns=phase_stats.requests_end_ns,
            phase=phase,
            error_summary=self._error_tracker.get_error_summary_for_phase(phase),
            cancelled=cancelled,
        )
        summaries = await asyncio.gather(
            *[
                self._summarize_one_accumulator(acc_type, acc, ctx)
                for acc_type, acc in acc_items
            ],
            return_exceptions=False,
        )
        for acc_type, summary in summaries:
            # Expose each accumulator's summary for cross-accumulator analyzers.
            summary_ctx.accumulator_outputs[acc_type] = summary
            ts = self._bucket_accumulator_summary(
                acc_type, summary, records_results, error_results
            )
            if ts:
                timeslices = ts
        return records_results, timeslices, error_results, summary_ctx

    def _has_records_for_phase(self, phase: CreditPhase) -> bool:
        phase_trackers = getattr(self._records_tracker, "_phase_trackers", {})
        if not isinstance(phase_trackers, dict):
            return False
        tracker = phase_trackers.get(phase)
        if tracker is None:
            return False
        return tracker.total_records > 0

    async def _summarize_warmup_metric_records(self) -> list[MetricResult] | None:
        """Return warmup-only inference metrics, or None when no warmup records exist."""
        if not self._has_records_for_phase(CreditPhase.WARMUP):
            return None

        (
            records_results,
            _,
            error_results,
            _summary_ctx,
        ) = await self._summarize_metric_record_accumulators(
            CreditPhase.WARMUP,
            self._records_tracker.was_phase_cancelled(CreditPhase.WARMUP),
        )
        if error_results:
            for error in error_results:
                self.error(f"Warmup metric summary error: {error}")

        return records_results or None

    async def _finalize_stream_exporters(self) -> None:
        """Flush all stream exporters concurrently; log per-exporter errors.

        Without this flush the publish below races partial files — the
        controller could write the readiness marker while the JSONL/CSV files
        were still mid-flush.
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

    async def _publish_all_results(
        self,
        result: ProcessRecordsResult,
    ) -> None:
        """Publish ProcessAllResultsMessage for the SystemController fan-in."""
        try:
            await self.publish(
                ProcessAllResultsMessage(
                    service_id=self.service_id,
                    results=result,
                )
            )
        except Exception as e:  # noqa: BLE001 - publish failure must not abort the per-record result path
            self.error(f"Failed to publish ProcessAllResultsMessage: {e!r}")

    def _deliver_network_rtt_to_accumulators(self) -> None:
        """Set the run-level mean network RTT (ns) on each metric-record accumulator.

        Two cases, resolved here just before MetricsAccumulator.summarize():

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

        # Deliver to the primary MetricsAccumulator engine, which injects
        # network_adjusted_* in its own summarize() from the columnar latency arrays.
        for target in self._metric_record_accumulators:
            set_rtt = getattr(target, "set_network_rtt_ns", None)
            if callable(set_rtt):
                set_rtt(rtt_ns)

    async def _process_results(
        self, phase: CreditPhase, cancelled: bool
    ) -> ProcessRecordsResult:
        """Process the accumulated records into final benchmark results.

        Single-flight: the natural finalize task and the PROCESS_RECORDS /
        PROFILE_CANCEL commands can race. The lock serializes them and the
        per-phase cache makes every call after the first return the same result
        instead of re-publishing and re-finalizing the stream exporters.
        """
        async with self._process_results_lock:
            cached = self._processed_results.get(phase)
            if cached is not None:
                self.debug(
                    lambda: f"Results for phase {phase} already processed; "
                    "returning cached result"
                )
                return cached
            result = await self._process_results_impl(phase, cancelled)
            self._processed_results[phase] = result
            return result

    async def _process_results_impl(
        self, phase: CreditPhase, cancelled: bool
    ) -> ProcessRecordsResult:
        """Process the accumulated records into final benchmark results."""
        self.debug(lambda: f"Processing records (cancelled: {cancelled})")
        self.info("Processing records results...")

        # Deliver the run-level mean network RTT before summarize() so
        # network_adjusted_* metrics can be injected.
        self._deliver_network_rtt_to_accumulators()

        (
            records_results,
            timeslices,
            error_results,
            summary_ctx,
        ) = await self._summarize_metric_record_accumulators(phase, cancelled)

        warmup_records_results = await self._summarize_warmup_metric_records()

        await self._finalize_stream_exporters()

        phase_stats = self._records_tracker.create_stats_for_phase(phase)
        # Snapshot count BEFORE extending with derived aggregates (efficiency,
        # analyzers) — `completed` reports request-derived records only.
        records_completed = len(records_results)

        # Cross-accumulator analyzer plugins (e.g. energy efficiency) run after
        # all accumulators have summarized, reading peers via the SummaryContext.
        records_results.extend(await self._run_analyzers(summary_ctx))

        result = ProcessRecordsResult(
            results=ProfileResults(
                records=records_results,
                warmup_records=warmup_records_results,
                timeslices=timeslices or None,
                completed=records_completed,
                start_ns=phase_stats.start_ns or time.time_ns(),
                end_ns=phase_stats.requests_end_ns or time.time_ns(),
                error_summary=self._error_tracker.get_error_summary_for_phase(phase),
                was_cancelled=cancelled,
                successful_request_count=phase_stats.success_records,
                error_request_count=phase_stats.error_records,
                branch_stats=self._latest_branch_stats
                if phase == CreditPhase.PROFILING
                else None,
            ),
            errors=error_results,
        )
        self.debug(lambda: f"Process records result: {result}")
        self.debug("Publishing ProcessRecordsResultMessage...")
        await self.publish(
            ProcessRecordsResultMessage(
                service_id=self.service_id,
                results=result,
            )
        )
        self.debug("ProcessRecordsResultMessage published")

        if self.run.cfg.gpu_telemetry_disabled:
            self.debug("GPU telemetry collection is disabled, skipping publish")
        else:
            try:
                self.debug("Starting _publish_telemetry_results...")
                await self._publish_telemetry_results(phase)
                self.debug("_publish_telemetry_results completed")
            except Exception as e:
                self.exception(f"Failed to publish telemetry results: {e!r}")

        if self.run.cfg.server_metrics_disabled:
            self.debug("Server metrics collection is disabled, skipping publish")
        else:
            try:
                self.debug("Starting _publish_server_metrics_results...")
                await self._publish_server_metrics_results()
                self.debug("_publish_server_metrics_results completed")
            except Exception as e:
                self.exception(f"Failed to publish server metrics results: {e!r}")

        accuracy_enabled = (
            self.run.cfg.accuracy is not None and self.run.cfg.accuracy.enabled
        )
        if accuracy_enabled and phase == CreditPhase.PROFILING:
            try:
                await self._publish_accuracy_results(phase)
            except Exception as e:
                self.exception(f"Failed to publish accuracy results: {e!r}")
        else:
            self.debug("Accuracy publish skipped (disabled or non-profiling phase)")

        # Publish the unified ProcessAllResultsMessage over the populated
        # accumulators. The per-stream result messages above remain the
        # shutdown trigger; this is supplementary.
        await self._publish_all_results(result)

        self.debug("_process_results completed, returning result")
        return result

    async def _process_telemetry_results(self) -> ProcessTelemetryResult:
        """Process telemetry results by exporting the accumulated telemetry data.

        Returns:
            ProcessTelemetryResult: Contains TelemetryExportData with pre-computed GPU telemetry stats and any errors encountered
        """
        self.debug("Processing telemetry results...")

        error_summary = [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self._telemetry_state.error_counts.items()
        ]

        if not self._gpu_telemetry_accumulator:
            self.debug(
                "GPU telemetry accumulator not found, cannot process telemetry results"
            )
            return ProcessTelemetryResult(
                results=None,
            )

        # Get timing from profiling phase stats
        # Note: end_ns is left None to include the final telemetry scrape that
        # occurs after PROFILE_COMPLETE but before export_results is called.
        # If start_ns is None (no profiling phase), include all data.
        phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        telemetry_export_data = await self._gpu_telemetry_accumulator.export_results(
            ExportContext(start_ns=phase_stats.start_ns, error_summary=error_summary)
        )

        return ProcessTelemetryResult(
            results=telemetry_export_data,
        )

    async def _publish_telemetry_results(self, phase: CreditPhase) -> None:
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

    async def _publish_accuracy_results(self, phase: CreditPhase) -> None:
        """Publish phase-scoped accuracy results on the dedicated accuracy channel.

        Mirrors ``_publish_telemetry_results``: exports the phase-scoped accuracy
        summary from the AccuracyAccumulator and publishes it independently from
        inference results.

        Exactly-once contract: this method only runs when accuracy is enabled and
        the phase is PROFILING. It attempts to publish exactly one
        ``ProcessAccuracyResultMessage``; the SystemController clears
        ``_should_wait_for_accuracy`` only on receipt of that message. A summary
        that fails to export still publishes a terminal ``results=None`` message so
        the gate is released. The publish itself is the only unrecoverable point:
        if the message bus raises here the gate cannot be released from this side,
        so we log it at error level (rather than swallowing) to make the cause of
        any resulting shutdown stall diagnosable.
        """
        summary: AccuracySummary | None = None
        if self._accuracy_accumulator is not None:
            try:
                summary = await self._accuracy_accumulator.export_results(
                    ExportContext(phase=phase)
                )
            except Exception as e:  # noqa: BLE001 - must still publish a terminal message
                self.exception(f"Accuracy summary export failed: {e!r}")
                summary = None
        try:
            await self.publish(
                ProcessAccuracyResultMessage(
                    service_id=self.service_id,
                    accuracy_result=ProcessAccuracyResult(results=summary),
                )
            )
        except Exception as e:  # noqa: BLE001
            self.error(
                "Failed to publish ProcessAccuracyResultMessage; the controller's "
                f"accuracy shutdown gate may not release: {e!r}"
            )
            raise

    async def _process_server_metrics_results(self) -> ProcessServerMetricsResult:
        """Process server metrics results by exporting the accumulated server metrics data.

        Returns:
            ProcessServerMetricsResult: Contains ServerMetricsResults with server metrics data hierarchy and any errors encountered
        """
        self.debug("Processing server metrics results...")

        error_summary = [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self._server_metrics_state.error_counts.items()
        ]

        if not self._server_metrics_accumulator:
            return ProcessServerMetricsResult(
                results=None,
                error_summary=error_summary,
            )

        # Get timing from profiling phase stats (warmup is automatically excluded)
        # TimeFilter will be constructed per-endpoint in accumulator with per-endpoint end times
        phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        profiling_start_ns = phase_stats.start_ns or time.time_ns()
        profiling_end_ns = phase_stats.requests_end_ns or time.time_ns()
        warmup_phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.WARMUP
        )

        server_metrics_export_data = (
            await self._server_metrics_accumulator.export_results(
                ExportContext(
                    start_ns=profiling_start_ns,
                    end_ns=profiling_end_ns,
                    error_summary=error_summary,
                    warmup_start_ns=warmup_phase_stats.start_ns,
                    warmup_end_ns=warmup_phase_stats.requests_end_ns,
                )
            )
        )

        return ProcessServerMetricsResult(
            results=server_metrics_export_data,
            error_summary=error_summary,
        )

    async def _publish_server_metrics_results(self) -> None:
        """Publish server metrics results independently from inference results.

        Processes and publishes server metrics data via ProcessServerMetricsResultMessage.
        Called at the end of _process_results to keep server metrics separate from
        inference metrics in the results pipeline.
        """
        self.debug(
            "_publish_server_metrics_results: calling _process_server_metrics_results..."
        )
        try:
            server_metrics_result = await self._process_server_metrics_results()
        except Exception as e:  # noqa: BLE001
            self.exception(f"Failed to process server metrics results: {e!r}")
            server_metrics_result = ProcessServerMetricsResult(results=None)
        self.debug(
            "_publish_server_metrics_results: publishing ProcessServerMetricsResultMessage..."
        )
        await self.publish(
            ProcessServerMetricsResultMessage(
                service_id=self.service_id,
                server_metrics_result=server_metrics_result,
            )
        )
        self.debug(
            "_publish_server_metrics_results: published ProcessServerMetricsResultMessage"
        )


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.RECORDS_MANAGER)


if __name__ == "__main__":
    main()

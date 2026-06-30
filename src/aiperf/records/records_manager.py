# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import inspect
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from aiperf.common.accumulator_protocols import (
    AccumulatorProtocol,
    AnalyzerProtocol,
    ExportContext,
    StreamExporterProtocol,
    SummaryContext,
)
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.zmq_config import ZMQDualBindConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
    ProfileCancelReason,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_command, on_message, on_pull_message
from aiperf.common.messages import (
    AllRecordsReceivedMessage,
    DatasetConfiguredNotification,
    MetricRecordsData,
    MetricRecordsMessage,
    ProcessAllResultsMessage,
    ProcessRecordsCommand,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
    ProcessTelemetryResultMessage,
    ProfileCancelCommand,
    ProfileCompleteCommand,
    RealtimeMetricsCommand,
    RealtimeMetricsMessage,
    RecordsProcessingStatsMessage,
    ServerMetricsRecordMessage,
    StartRealtimeTelemetryCommand,
    TelemetryRecordsMessage,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    MetricResult,
    PhaseRecordsStats,
    ProcessRecordsResult,
    ProcessServerMetricsResult,
    ProcessTelemetryResult,
    ServerMetricsRecord,
    TelemetryRecord,
    TimesliceResult,
    WorkerProcessingStats,
)
from aiperf.common.models.branch_stats import BranchStats
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.plugin.enums import (
    AccumulatorType,
    AnalyzerType,
    StreamExporterType,
    UIType,
)
from aiperf.records.error_tracker import ErrorTracker
from aiperf.records.records_manager_processing import (
    accumulators_for_record_type,
    build_process_records_result,
    compute_analyzer_outputs,
    filter_display_metrics,
    generate_realtime_metrics,
    load_accumulators,
    load_analyzers,
    load_stream_exporters,
    stream_exporters_for_record_type,
)
from aiperf.records.records_tracker import RecordsTracker

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
# Continuation rows sit at a small fixed indent under the header line rather
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
    server_snapshot: dict[str, float] | dict[str, dict[str, float]] | None = None,
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
          trace   theoretical_prefix_cache_hit=97.5%

    The header sits on its own line and the summary counters drop to the
    first indented row so the line no longer wraps in narrow terminals; each
    block is emitted as one log record (see ``_report_realtime_metrics``), so
    the timestamp, level, and source location appear once. Every row keeps its
    own ``pNN=`` labels, while the values are right-aligned in per-column widths
    so the digits and ``ms`` suffixes line up into a grid.

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

    theoretical_prefix_mr = by_tag.get("theoretical_prefix_cache_hit")
    theoretical_prefix_hit = getattr(theoretical_prefix_mr, "current", None)
    if theoretical_prefix_hit is None:
        theoretical_prefix_hit = getattr(theoretical_prefix_mr, "avg", None)
    if theoretical_prefix_hit is not None:
        rows.append(
            f"{indent}{'trace':<{label_w}} theoretical_prefix_cache_hit={theoretical_prefix_hit:.1f}%"
        )

    # Server-side row — cumulative cache hit rate, KV usage, and scheduler
    # queue depth from the live ServerMetricsAccumulator snapshot. Sourced
    # from the /metrics scrape, so populates only when server-metrics
    # collection is enabled and the inference server actually serves
    # Prometheus. Each part is rendered only when its backing metric is
    # present, so e.g. cpu_kv / ext_cache_hit show up only on offload=cpu
    # runs.
    if server_snapshot:
        if all(isinstance(value, dict) for value in server_snapshot.values()):
            labeled_snapshots = server_snapshot.items()
        else:
            labeled_snapshots = [("", server_snapshot)]

    else:
        labeled_snapshots = []

    for server_label, snapshot in labeled_snapshots:
        srv_parts: list[str] = []
        if "prefix_cache_hit_rate" in snapshot:
            srv_parts.append(
                f"prefix_cache_hit={snapshot['prefix_cache_hit_rate']:.1f}%"
            )
        if "unique_input_tokens_srv" in snapshot:
            srv_parts.append(
                f"unique_in_srv={int(round(snapshot['unique_input_tokens_srv'])):,}"
            )
        if "external_prefix_cache_hit_rate" in snapshot:
            srv_parts.append(
                f"ext_cache_hit={snapshot['external_prefix_cache_hit_rate']:.1f}%"
            )
        if "kv_cache_usage_pct" in snapshot:
            srv_parts.append(f"kv_usage={snapshot['kv_cache_usage_pct']:.1f}%")
        if "cpu_kv_cache_usage_pct" in snapshot:
            srv_parts.append(f"cpu_kv_usage={snapshot['cpu_kv_cache_usage_pct']:.1f}%")
        if "num_running" in snapshot or "num_waiting" in snapshot:
            running = int(snapshot.get("num_running", 0))
            waiting = int(snapshot.get("num_waiting", 0))
            srv_parts.append(f"queue={running}r/{waiting}w")
        if "input_token_throughput_srv" in snapshot:
            srv_parts.append(
                f"tput_in_srv={int(round(snapshot['input_token_throughput_srv'])):,}/s"
            )
        if "output_token_throughput_srv" in snapshot:
            srv_parts.append(
                f"tput_out_srv={int(round(snapshot['output_token_throughput_srv'])):,}/s"
            )
        if srv_parts:
            row_label = "srv" if not server_label else f"srv {server_label}"
            rows.append(f"{indent}{row_label:<{label_w}} {' '.join(srv_parts)}")

    return "\n".join([header, *rows])


@dataclass
class ErrorTrackingState:
    """State container for tracking errors with counts.

    Provides common error tracking functionality for telemetry / server
    metrics / regular-metric subsystems.
    """

    error_counts: dict[ErrorDetails, int] = field(
        default_factory=lambda: defaultdict(int)
    )


class RecordsManager(PullClientMixin, BaseComponentService):
    """Collects and processes benchmark results from workers.

    The RecordsManager receives metric records from workers and routes them
    through the new ``accumulator`` / ``stream_exporter`` plugin pipeline.
    The timing manager is the ground truth for what requests completed
    within the benchmark window — when it signals phase completion with a
    final completed count, the RecordsManager waits until it has processed
    that many records before finalizing results.

    At ``_process_results`` time the manager:

    1. Calls ``summarize()`` on every accumulator (typed
       :class:`AccumulatorMetricsSummary` from :class:`MetricsAccumulator`,
       ``list[MetricResult]`` from GPU telemetry / server metrics
       accumulators) and bridges both shapes into the
       ``ProcessRecordsResultMessage`` payload.
    2. Finalizes stream exporters concurrently (JSONL flush).
    3. Runs every loaded analyzer with a single :class:`SummaryContext`
       and publishes a :class:`ProcessAllResultsMessage` carrying the
       analyzer outputs for the SystemController fan-in.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        # For dual-bind mode (Kubernetes), also bind to TCP for remote record processors.
        # Controller binds to IPC + TCP; workers connect via TCP.
        additional_bind_address: str | None = None
        comm_config = service_config.comm_config
        if (
            isinstance(comm_config, ZMQDualBindConfig)
            and not comm_config.controller_host
        ):
            additional_bind_address = comm_config.records_push_pull_tcp_bind_address

        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            pull_client_address=CommAddress.RECORDS,
            pull_client_bind=True,
            pull_client_max_concurrency=Environment.ZMQ.PULL_MAX_CONCURRENCY,
            pull_client_additional_bind_address=additional_bind_address,
            **kwargs,
        )

        self._records_tracker = RecordsTracker()
        self._error_tracker = ErrorTracker()

        self._previous_realtime_records: int | None = None
        self._previous_realtime_server_snapshot: (
            dict[str, float] | dict[str, dict[str, float]] | None
        ) = None
        self._prev_realtime_snapshot: tuple[int, float] | None = None

        self._telemetry_state = ErrorTrackingState()
        self._server_metrics_state = ErrorTrackingState()
        self._metric_state = ErrorTrackingState()
        self._skipped_context_overflow_count = 0
        self._skipped_context_overflow_counts_by_phase: dict[CreditPhase, int] = {
            CreditPhase.WARMUP: 0,
            CreditPhase.PROFILING: 0,
        }

        # Orchestrator-emitted DAG sub-agent stats, received via
        # CreditPhaseCompleteMessage. Keyed by phase so ProfileResults for the
        # profiling phase can include the orchestrator's final counters.
        self._phase_branch_stats: dict[CreditPhase, BranchStats] = {}

        # Failed-request threshold (in-flight abort). When the rolling
        # ``error_records / total_records`` ratio exceeds the user-supplied
        # threshold after the grace floor is passed, broadcast a
        # ProfileCancelCommand to terminate the run early. The grace floor
        # is max(concurrency, 10) records so a single early failure (e.g.,
        # the very first request) cannot trip a tiny-N threshold.
        self._failed_request_threshold: float | None = (
            user_config.loadgen.failed_request_threshold
        )
        conc_val = user_config.loadgen.concurrency
        conc_int = int(conc_val) if isinstance(conc_val, int | float) else 1
        self._failed_request_grace_floor: int = max(conc_int, 10)
        self._failed_request_abort_triggered: bool = False

        # New accumulator + analyzer pipeline. Three sibling categories:
        #   accumulator:    process_record + summarize (MetricsAccumulator,
        #                   GPUTelemetryAccumulator, ServerMetricsAccumulator)
        #   stream_exporter: process_record + finalize (RecordExportJSONLWriter, ...)
        #   analyzer:        summarize(ctx) only — no record ingestion
        # Disabled / failed plugins are dropped silently — see loaders.
        self._accumulators: dict[AccumulatorType, AccumulatorProtocol] = (
            load_accumulators(self)
        )
        self._stream_exporters: dict[StreamExporterType, StreamExporterProtocol] = (
            load_stream_exporters(self)
        )
        self._analyzers: dict[AnalyzerType, AnalyzerProtocol] = load_analyzers(self)

        # Per-record-type dispatch lists so the hot path is a list iteration,
        # not an O(N plugins) re-iteration of plugin metadata per record.
        self._metric_record_accumulators: list[AccumulatorProtocol] = (
            accumulators_for_record_type(self._accumulators, "metric_records")
        )
        self._metric_record_stream_exporters: list[StreamExporterProtocol] = (
            stream_exporters_for_record_type(self._stream_exporters, "metric_records")
        )
        self._gpu_telemetry_stream_exporters: list[StreamExporterProtocol] = (
            stream_exporters_for_record_type(self._stream_exporters, "gpu_telemetry")
        )
        self._server_metrics_stream_exporters: list[StreamExporterProtocol] = (
            stream_exporters_for_record_type(self._stream_exporters, "server_metrics")
        )

        # Side-channel accumulators (telemetry / server-metrics) keep
        # single-instance handles for the controller fan-in path — both are
        # still subclasses of BaseMetricsProcessor and conform to
        # the existing `process_telemetry_record` / `process_server_metrics_record`
        # interface, looked up by AccumulatorType.
        self._gpu_telemetry_accumulator = self._accumulators.get(
            AccumulatorType.GPU_TELEMETRY
        )
        self._server_metrics_accumulator = self._accumulators.get(
            AccumulatorType.SERVER_METRICS
        )

    @on_pull_message(MessageType.METRIC_RECORDS)
    async def _on_metric_records(self, message: MetricRecordsMessage) -> None:
        """Handle a metric records message."""
        if self.is_trace_enabled:
            self.trace(f"Received metric records: {message}")

        record_data = message.to_data()

        # Context-overflow records in AGENTIC_REPLAY scenarios bypass normal
        # user-facing per-record processing but still advance the records-side
        # success counter so the completion barrier converges. Keep only a
        # narrow aggregate side-channel count for runtime submission validation.
        if getattr(record_data.metadata, "context_overflow_skip", False):
            phase = record_data.metadata.benchmark_phase
            if not hasattr(self, "_skipped_context_overflow_counts_by_phase"):
                self._skipped_context_overflow_counts_by_phase = {}
            self._skipped_context_overflow_counts_by_phase[phase] = (
                self._skipped_context_overflow_counts_by_phase.get(phase, 0) + 1
            )
            if phase == CreditPhase.PROFILING:
                self._skipped_context_overflow_count += 1
            phase_tracker = self._records_tracker._get_phase_tracker(phase)
            phase_tracker.increment_success_records()
            phase_tracker.increment_worker_success_records(
                record_data.metadata.worker_id
            )
            if self._records_tracker.check_and_set_all_records_received_for_phase(
                phase
            ):
                await self._handle_all_records_received(phase)
            return

        await self._send_record_to_accumulators(record_data)

        self._records_tracker.update_from_record_data(record_data)
        if record_data.error:
            self._error_tracker.increment_error_count_for_phase(
                record_data.metadata.benchmark_phase, record_data.error
            )

        await self._maybe_trigger_failed_request_abort(
            record_data.metadata.benchmark_phase
        )

        if self._records_tracker.check_and_set_all_records_received_for_phase(
            record_data.metadata.benchmark_phase
        ):
            await self._handle_all_records_received(
                record_data.metadata.benchmark_phase
            )

    async def _maybe_trigger_failed_request_abort(self, phase: CreditPhase) -> None:
        """Abort the run when the PROFILING failure rate exceeds the threshold.

        No-op when ``--failed-request-threshold`` is unset, when this method
        already fired once for this run, or when the total record count has
        not yet crossed the grace floor (``max(concurrency, 10)``). Otherwise
        broadcasts ProfileCancelCommand on the message bus -- the existing
        cancel-path handlers in timing_manager, server_metrics manager, and
        gpu_telemetry manager stop their work; this manager's own
        _on_profile_cancel_command marks the phase cancelled and finalizes
        results with cancelled=True, which surfaces in the run's exit code
        via the standard cancel flow.
        """
        if self._failed_request_threshold is None:
            return
        if self._failed_request_abort_triggered:
            return
        if phase != CreditPhase.PROFILING:
            return

        stats = self._records_tracker.create_stats_for_phase(phase)
        total = stats.total_records
        if total < self._failed_request_grace_floor:
            return

        error_records = stats.error_records
        rate = error_records / total if total > 0 else 0.0
        if rate <= self._failed_request_threshold:
            return

        self._failed_request_abort_triggered = True
        self.warning(
            f"--failed-request-threshold exceeded: "
            f"{error_records}/{total} = {rate:.3f} > "
            f"{self._failed_request_threshold:.3f} "
            f"(grace floor {self._failed_request_grace_floor}). "
            "Broadcasting ProfileCancelCommand to terminate the run."
        )
        try:
            await self.publish(
                ProfileCancelCommand(
                    service_id=self.service_id,
                    reason=ProfileCancelReason.FAILED_REQUEST_THRESHOLD,
                )
            )
        except Exception as exc:  # noqa: BLE001
            # Publish failure must not abort the per-record path; if the
            # broadcast doesn't land, the run will continue and the
            # threshold violation will be re-evaluated and re-published on
            # the next record.
            self.warning(
                f"Failed to publish ProfileCancelCommand for threshold abort: {exc!r}"
            )
            self._failed_request_abort_triggered = False

    async def _send_record_to_accumulators(
        self, record_data: MetricRecordsData
    ) -> None:
        """Dispatch a metric record to all metric_records accumulators + stream exporters.

        Per-handler exceptions are caught so one bad accumulator does not
        abort the others. GPU telemetry / server metrics records are routed
        via their own ``@on_pull_message`` handlers and do **not** flow
        through here — the dispatch is metadata-driven via plugin
        ``record_types``.
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

    @on_pull_message(MessageType.TELEMETRY_RECORDS)
    async def _on_telemetry_records(self, message: TelemetryRecordsMessage) -> None:
        """Handle telemetry records message from Telemetry Manager.

        The RecordsManager acts as the central hub for all record processing,
        whether inference metrics or GPU telemetry. Routes the batch to the
        ``accumulator:gpu_telemetry`` plugin instance.
        """
        if message.valid:
            try:
                await self._send_telemetry_to_accumulator(message.records)
            except Exception as e:
                error_details = ErrorDetails(
                    message=f"Telemetry processor error: {str(e)}"
                )
                self._telemetry_state.error_counts[error_details] += 1
                self.debug(f"Failed to process telemetry batch: {e}")
        else:
            if message.error:
                self._telemetry_state.error_counts[message.error] += 1

    @on_pull_message(MessageType.SERVER_METRICS_RECORD)
    async def _on_server_metrics_records(
        self, message: ServerMetricsRecordMessage
    ) -> None:
        """Handle server metrics record message from Server Metrics Manager.

        Forwards full record to the ``accumulator:server_metrics`` plugin
        instance.
        """
        if message.valid:
            await self._send_server_metrics_to_accumulator(message.record)
        else:
            if message.error:
                self._server_metrics_state.error_counts[message.error] += 1

    async def _send_telemetry_to_accumulator(
        self, telemetry_records: list[TelemetryRecord]
    ) -> None:
        """Dispatch each telemetry record to the GPU telemetry accumulator."""
        if self._gpu_telemetry_accumulator is None:
            return
        for record in telemetry_records:
            try:
                await self._gpu_telemetry_accumulator.process_telemetry_record(record)
            except BaseException as exc:  # noqa: BLE001
                self.exception(f"Failed to process telemetry record: {exc!r}")
                self._telemetry_state.error_counts[
                    ErrorDetails.from_exception(exc)
                ] += 1
        for exporter in self._gpu_telemetry_stream_exporters:
            for record in telemetry_records:
                try:
                    await exporter.process_record(record)
                except BaseException as exc:  # noqa: BLE001
                    self.error(
                        f"Stream exporter {exporter.__class__.__name__} failed for "
                        f"gpu_telemetry record: {exc!r}"
                    )

    async def _send_server_metrics_to_accumulator(
        self, record: ServerMetricsRecord
    ) -> None:
        """Dispatch a server metrics record to the server metrics accumulator."""
        if self._server_metrics_accumulator is None:
            return
        try:
            await self._server_metrics_accumulator.process_server_metrics_record(record)
        except BaseException as exc:  # noqa: BLE001
            self.exception(f"Failed to process server metrics record: {exc!r}")
            self._server_metrics_state.error_counts[
                ErrorDetails.from_exception(exc)
            ] += 1
        for exporter in self._server_metrics_stream_exporters:
            try:
                await exporter.process_record(record)
            except BaseException as exc:  # noqa: BLE001
                self.error(
                    f"Stream exporter {exporter.__class__.__name__} failed for "
                    f"server_metrics record: {exc!r}"
                )

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

        Runs as a background task to avoid blocking the message pump.
        """
        phase_stats = self._records_tracker.create_stats_for_phase(phase)

        await self.publish(
            AllRecordsReceivedMessage(
                service_id=self.service_id,
                request_ns=time.time_ns(),
                final_processing_stats=phase_stats,
            )
        )

        # Trigger final server metrics scrape and wait for completion.
        # A TimeoutError must not abort _finalize_and_process_results, because
        # that would skip _process_results and the resulting
        # ProcessRecordsResultMessage — the system controller would then
        # never run _export_results_data.
        response = await self.send_command_and_wait_for_response(
            ProfileCompleteCommand(service_id=self.service_id), timeout=10.0
        )

        if isinstance(response, ErrorDetails):
            self.warning(f"Server metrics final scrape timed out or failed: {response}")
        else:
            self.debug("Server metrics final scrape completed")

        self.debug("Waiting for server metrics flush period...")
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
        """Forward dataset metadata to any accumulator that wants it.

        Only the ``accumulator:metric_results`` (``MetricsAccumulator``) cares
        about dataset metadata today, but the dispatch is duck-typed so a
        future plugin can opt in by exposing ``on_dataset_configured``.
        """
        for accumulator in self._accumulators.values():
            if hasattr(accumulator, "on_dataset_configured"):
                accumulator.on_dataset_configured(message.metadata)

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
        if message.stats.phase == CreditPhase.PROFILING:
            self.info(
                f"Sent {message.stats.final_requests_sent:,} requests. Waiting for all to complete..."
            )
        self._records_tracker.update_phase_info(message.stats)

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, message: CreditPhaseCompleteMessage
    ) -> None:
        """Handle a credit phase complete message in order to track the end time, and check if all records have been received."""
        self._records_tracker.update_phase_info(message.stats)
        if message.branch_stats is not None:
            self._phase_branch_stats[message.stats.phase] = message.branch_stats
        if message.stats.phase == CreditPhase.PROFILING:
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

        # This check is to prevent a race condition where the records manager
        # processes all records before the timing manager has sent the final
        # completed count.
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
        if self._records_tracker.check_and_set_all_records_received_for_phase(
            CreditPhase.PROFILING
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
            return
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
        """Handle the process records command by running the unified pipeline and returning the results."""
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

        self._records_tracker.mark_phase_cancelled(CreditPhase.PROFILING)

        return await self._process_results(phase=CreditPhase.PROFILING, cancelled=True)

    @background_task(interval=None, immediate=True)
    async def _report_realtime_inference_metrics_task(self) -> None:
        """Report inference metrics at regular intervals.

        Always runs so subscribers (dashboard, k8s job-WS, headless log summary)
        get snapshots regardless of the active UI type. ``--stats-interval 0``
        disables both the publish and the log line by short-circuiting here.
        """
        interval = Environment.UI.realtime_metrics_interval(self.service_config.ui_type)
        if interval == 0:
            return
        while not self.stop_requested:
            await asyncio.sleep(interval)
            phase_stats = self._records_tracker.create_stats_for_phase(
                CreditPhase.PROFILING
            )
            server_snapshot = self._collect_realtime_server_snapshot(
                start_ns=phase_stats.start_ns
            )
            if not self._has_realtime_update(phase_stats, server_snapshot):
                continue  # No changed records or server metrics; skip the rebuild.
            emitted = await self._report_realtime_metrics(
                server_snapshot=server_snapshot
            )
            if emitted:
                self._previous_realtime_records = phase_stats.total_records
                self._previous_realtime_server_snapshot = dict(server_snapshot)

    @on_command(CommandType.START_REALTIME_TELEMETRY)
    async def _on_start_realtime_telemetry_command(
        self, message: StartRealtimeTelemetryCommand
    ) -> None:
        """Handle command to start the realtime telemetry background task.

        This is called when the user dynamically enables the telemetry dashboard
        by pressing the telemetry option in the UI without having passed the
        ``dashboard`` parameter at startup.
        """
        if self._gpu_telemetry_accumulator is not None and hasattr(
            self._gpu_telemetry_accumulator, "start_realtime_telemetry"
        ):
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
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        """Return the current live server metrics snapshot, if available."""
        server_snapshot: dict[str, float] | dict[str, dict[str, float]] = {}
        if self._server_metrics_accumulator is None:
            return server_snapshot
        try:
            snapshot_fn = getattr(
                self._server_metrics_accumulator,
                "realtime_snapshots",
                None,
            )
            if not callable(snapshot_fn):
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

    def _has_realtime_update(
        self,
        phase_stats: PhaseRecordsStats,
        server_snapshot: dict[str, float] | dict[str, dict[str, float]],
    ) -> bool:
        """Return whether realtime metrics need rebuilding for the current tick."""
        return (
            phase_stats.total_records != self._previous_realtime_records
            or server_snapshot != self._previous_realtime_server_snapshot
        )

    async def _report_realtime_metrics(
        self,
        server_snapshot: dict[str, float] | dict[str, dict[str, float]] | None = None,
    ) -> bool:
        """Report inference metrics (used by command handler).

        Filters out hidden metrics (INTERNAL/EXPERIMENTAL) and converts all
        metrics to display units before publishing. This ensures all consumers
        receive consistent, pre-processed metrics.
        """
        # Realtime metrics only need the metric_records accumulators —
        # GPU telemetry / server metrics live on separate fan-outs.
        raw_metrics = await generate_realtime_metrics(self._metric_record_accumulators)
        if not raw_metrics:
            return False

        display_metrics = filter_display_metrics(raw_metrics)
        if not display_metrics:
            return False
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
            if self.service_config.ui_type != UIType.DASHBOARD:
                # Keep the block atomic and avoid repeating the logger's
                # timestamp, level, and source suffix on every metrics row.
                self.info(rendered)
        return True

    def _snapshot_branch_stats(self, phase: CreditPhase) -> BranchStats | None:
        """Return the orchestrator-published BranchStats for ``phase``.

        Returns ``None`` for non-DAG runs or for phases where the TimingManager
        never published sub-agent counters on ``CreditPhaseCompleteMessage``.
        """
        return self._phase_branch_stats.get(phase)

    async def _process_results(
        self, phase: CreditPhase, cancelled: bool
    ) -> ProcessRecordsResult:
        """Run the full unified records pipeline.

        Steps (each one logs and continues on per-handler failure — the
        controller-side ``ProcessRecordsResultMessage`` consumer must not be
        starved by a single bad accumulator/exporter/analyzer):

        1. ``summarize()`` every accumulator and bucket the output (handles
           both ``AccumulatorMetricsSummary`` and ``list[MetricResult]``
           shapes).
        2. ``finalize()`` every stream exporter (JSONL flush) before the
           controller writes the readiness marker.
        3. Build :class:`ProcessRecordsResult` and publish
           :class:`ProcessRecordsResultMessage`.
        4. Run analyzers over a single :class:`SummaryContext` and publish
           :class:`ProcessAllResultsMessage` (steady-state, energy efficiency
           — populated controller-side).
        """
        self.debug(lambda: f"Processing records (cancelled: {cancelled})")
        self.info("Processing records results...")

        (
            records_results,
            timeslices,
            error_results,
        ) = await self._summarize_all_accumulators(phase=phase, cancelled=cancelled)
        warmup_records_results: list[MetricResult] | None = None
        summarize_warmup = getattr(self, "_summarize_warmup_metric_records", None)
        if callable(summarize_warmup):
            maybe_warmup_records_results = summarize_warmup()
            if inspect.isawaitable(maybe_warmup_records_results):
                warmup_records_results = await maybe_warmup_records_results
        await self._finalize_stream_exporters()

        result = build_process_records_result(
            records_results=records_results,
            warmup_records_results=warmup_records_results,
            timeslices=timeslices,
            error_results=error_results,
            tracker=self._records_tracker,
            error_tracker=self._error_tracker,
            cancelled=cancelled,
            branch_stats=self._snapshot_branch_stats(phase),
            context_overflow_count=self._skipped_context_overflow_count,
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

        # Side-channel telemetry / server-metrics fan-out.
        if self.user_config.gpu_telemetry_disabled:
            self.debug("GPU telemetry collection is disabled, skipping publish")
        else:
            try:
                self.debug("Starting _publish_telemetry_results...")
                await self._publish_telemetry_results(phase)
                self.debug("_publish_telemetry_results completed")
            except Exception as e:
                self.exception(f"Failed to publish telemetry results: {e!r}")

        if self.user_config.server_metrics_disabled:
            self.debug("Server metrics collection is disabled, skipping publish")
        else:
            try:
                self.debug("Starting _publish_server_metrics_results...")
                await self._publish_server_metrics_results()
                self.debug("_publish_server_metrics_results completed")
            except Exception as e:
                self.exception(f"Failed to publish server metrics results: {e!r}")

        # Analyzer pipeline + ProcessAllResultsMessage — bridges the
        # records-side accumulators to the SystemController fan-in.
        # Failures here must not break the publishes above; the
        # ProcessRecordsResultMessage has already been published.
        analyzer_outputs = await self._run_analyzers(
            result=result,
            cancelled=cancelled,
        )
        await self._publish_all_results(result, analyzer_outputs)

        self.debug("_process_results completed, returning result")
        return result

    async def _summarize_one_accumulator(
        self,
        acc_type: AccumulatorType,
        accumulator: AccumulatorProtocol,
        ctx: ExportContext,
    ) -> tuple[AccumulatorType, Any]:
        """Run summarize/export_results on a single accumulator with timeout.

        Returns the result (or exception object) so a single bad accumulator
        cannot abort the rest. Prefers ``summarize()`` because it is cheaper
        for the metric_results path (no ExportContext window math) and falls
        back to ``export_results(ctx)`` when ``summarize`` is missing.
        """
        name = accumulator.__class__.__name__
        self.debug(f"Starting summarize for accumulator {acc_type}: {name}")
        try:
            if accumulator.__class__.__name__ in {
                "MetricsAccumulator",
                "TheoreticalPrefixCacheAccumulator",
            } and hasattr(accumulator, "export_results"):
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
        summary: Any,
        records_results: list[MetricResult],
        error_results: list[ErrorDetails],
    ) -> list[TimesliceResult]:
        """Route a single summary into the right ProfileResults bucket.

        Returns the timeslices contributed by this summary; the caller
        merges them into the per-call accumulator state. Each
        :class:`TimesliceResult` bundles the slice's window bounds with
        its metric results in chronological order.
        """
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

    async def _summarize_all_accumulators(
        self,
        *,
        phase: CreditPhase,
        cancelled: bool,
        accumulators: dict[AccumulatorType, AccumulatorProtocol] | None = None,
    ) -> tuple[
        list[MetricResult],
        list[TimesliceResult],
        list[ErrorDetails],
    ]:
        """Summarize every accumulator and bucket the results by shape.

        Returns ``(records, timeslices, errors)``. Tolerates both
        :class:`AccumulatorMetricsSummary` (returned by
        :class:`MetricsAccumulator`) and the simpler ``list[MetricResult]``
        shape still returned by GPU telemetry / server metrics accumulators.
        The list shape is appended to ``records``; ``timeslices`` come from
        the typed summary path only.
        """
        records_results: list[MetricResult] = []
        timeslices: list[TimesliceResult] = []
        error_results: list[ErrorDetails] = []

        selected_accumulators = (
            accumulators if accumulators is not None else self._accumulators
        )

        if not selected_accumulators:
            self.debug("No accumulators configured, returning empty result")
            return (
                records_results,
                timeslices,
                error_results,
            )

        phase_stats = self._records_tracker.create_stats_for_phase(phase)
        ctx = ExportContext(
            start_ns=phase_stats.start_ns,
            end_ns=phase_stats.requests_end_ns,
            phase=phase,
            error_summary=self._error_tracker.get_error_summary_for_phase(phase),
            cancelled=cancelled,
        )

        summaries = await asyncio.gather(
            *[
                self._summarize_one_accumulator(acc_type, accumulator, ctx)
                for acc_type, accumulator in selected_accumulators.items()
            ],
            return_exceptions=False,
        )

        for acc_type, summary in summaries:
            ts = self._bucket_accumulator_summary(
                acc_type, summary, records_results, error_results
            )
            if ts:
                timeslices = ts

        return (
            records_results,
            timeslices,
            error_results,
        )

    def _has_records_for_phase(self, phase: CreditPhase) -> bool:
        phase_trackers = getattr(self._records_tracker, "_phase_trackers", {})
        if not isinstance(phase_trackers, dict):
            return False
        tracker = phase_trackers.get(phase)
        if tracker is None:
            return False
        return tracker.total_records > 0

    def _metric_record_accumulator_map(
        self,
    ) -> dict[AccumulatorType, AccumulatorProtocol]:
        metric_record_accumulators = {
            id(acc) for acc in getattr(self, "_metric_record_accumulators", [])
        }
        return {
            acc_type: accumulator
            for acc_type, accumulator in self._accumulators.items()
            if id(accumulator) in metric_record_accumulators
        }

    async def _summarize_warmup_metric_records(self) -> list[MetricResult] | None:
        """Return warmup-only inference metrics, or None when no warmup records exist."""
        if not self._has_records_for_phase(CreditPhase.WARMUP):
            return None

        records_results, _, error_results = await self._summarize_all_accumulators(
            phase=CreditPhase.WARMUP,
            cancelled=self._records_tracker.was_phase_cancelled(CreditPhase.WARMUP),
            accumulators=self._metric_record_accumulator_map(),
        )
        if error_results:
            for error in error_results:
                self.error(f"Warmup metric summary error: {error}")

        warmup_context_overflow_count = (
            self._skipped_context_overflow_counts_by_phase.get(CreditPhase.WARMUP, 0)
        )
        if warmup_context_overflow_count:
            records_results.append(
                MetricResult(
                    tag="context_overflow_count",
                    header="Context Overflow Count",
                    unit="requests",
                    avg=float(warmup_context_overflow_count),
                    count=1,
                )
            )
        return records_results or None

    async def _finalize_stream_exporters(self) -> None:
        """Flush all stream exporters concurrently; log per-exporter errors.

        Stream exporters (e.g. JSONL writers) buffer records; without this
        flush the publish below races partial files — the controller could
        write the readiness marker while the JSONL/CSV files were still
        mid-flush.
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
        """Run all loaded analyzers over a single :class:`SummaryContext`.

        Returns the analyzer outputs map for callers to attach to outgoing
        messages. Time bounds come from ``result.results`` so the analyzers
        see exactly the window the records-tracker reported. Disabled /
        failing analyzers are skipped per :func:`compute_analyzer_outputs`'s
        policy.
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

    async def _publish_all_results(
        self,
        result: ProcessRecordsResult,
        analyzer_outputs: dict[AnalyzerType, Any],
    ) -> None:
        """Publish :class:`ProcessAllResultsMessage` with analyzer outputs."""
        try:
            await self.publish(
                ProcessAllResultsMessage(
                    service_id=self.service_id,
                    results=result,
                )
            )
        except Exception as e:  # noqa: BLE001 - publish failure must not abort the per-record result path
            self.error(f"Failed to publish ProcessAllResultsMessage: {e!r}")

    def _process_telemetry_results(self) -> ProcessTelemetryResult:
        """Process telemetry results by exporting the accumulated telemetry data."""
        self.debug("Processing telemetry results...")

        error_summary = [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self._telemetry_state.error_counts.items()
        ]

        if not self._gpu_telemetry_accumulator:
            self.debug(
                "GPU telemetry accumulator not found, cannot process telemetry results"
            )
            return ProcessTelemetryResult(results=None)

        # end_ns is intentionally omitted to include the final telemetry scrape
        # that occurs after PROFILE_COMPLETE but before export_results is called.
        phase_stats = self._records_tracker.create_stats_for_phase(
            CreditPhase.PROFILING
        )
        telemetry_export_data = self._gpu_telemetry_accumulator.export_results(
            start_ns=phase_stats.start_ns,
            error_summary=error_summary,
        )

        return ProcessTelemetryResult(results=telemetry_export_data)

    async def _publish_telemetry_results(self, phase: CreditPhase) -> None:
        """Publish telemetry results independently from inference results."""
        telemetry_result = self._process_telemetry_results()
        await self.publish(
            ProcessTelemetryResultMessage(
                service_id=self.service_id,
                telemetry_result=telemetry_result,
            )
        )

    async def _process_server_metrics_results(self) -> ProcessServerMetricsResult:
        """Process server metrics results by exporting the accumulated server metrics data."""
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
                start_ns=profiling_start_ns,
                end_ns=profiling_end_ns,
                error_summary=error_summary,
                warmup_start_ns=warmup_phase_stats.start_ns,
                warmup_end_ns=warmup_phase_stats.requests_end_ns,
            )
        )

        return ProcessServerMetricsResult(
            results=server_metrics_export_data,
            error_summary=error_summary,
        )

    async def _publish_server_metrics_results(self) -> None:
        """Publish server metrics results independently from inference results."""
        self.debug(
            "_publish_server_metrics_results: calling _process_server_metrics_results..."
        )
        server_metrics_result = await self._process_server_metrics_results()
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

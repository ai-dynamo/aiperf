# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numpy-backed metrics accumulator with columnar storage and dynamic timeslicing."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import (
    AggregationKind,
    MetricFlags,
    MetricType,
    MetricValueTypeT,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import MetricResult, TimesliceResult
from aiperf.common.types import MetricTagT
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.metrics.accumulator_sweeps import compute_sweep_curves
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.cache_reporting_hint import (
    CACHE_REPORTING_HINT,
    usage_without_cache_in_record,
)
from aiperf.metrics.column_store import ColumnStore
from aiperf.metrics.derived_latency import (
    inject_adjusted_latency_metrics,
    inject_derived_latency_metrics,
)
from aiperf.metrics.display_units import to_display_unit
from aiperf.metrics.metric_dicts import MetricResultsDict, metric_result_from_array
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.network_adjusted_analyzer import (
    compute_network_adjusted_arrays,
    inject_network_adjusted_from_arrays,
)
from aiperf.metrics.replay_sched_lag_analyzer import inject_replay_sched_lag_metrics
from aiperf.metrics.types.replay_sched_lag_metrics import (
    REPLAY_SCHED_DEGRADED_THRESHOLD_MS,
)
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import SummaryContext
    from aiperf.config.resolution.plan import BenchmarkRun


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


_AGGREGATE_FUNCS: dict[AggregationKind, Callable[[np.ndarray], float]] = {
    AggregationKind.SUM: lambda a: float(np.sum(a)),
    AggregationKind.MAX: lambda a: float(np.max(a)),
    AggregationKind.MIN: lambda a: float(np.min(a)),
}


class _MetricClassLookup:
    def __init__(self, metric_classes: dict[MetricTagT, Any]) -> None:
        self._metric_classes = metric_classes

    def get_class(self, tag: MetricTagT) -> Any:
        metric_class = self._metric_classes.get(tag)
        if metric_class is None:
            raise KeyError(tag)
        return metric_class


class MetricsAccumulator(BaseMetricsProcessor):
    """Numpy-backed accumulator for inference metrics.

    Session_num-indexed NaN-sparse columnar storage; RECORD metrics get
    per-value stats, AGGREGATE metrics one scalar via :class:`AggregationKind`,
    DERIVED metrics computed from those at summarize time.
    """

    # RecordsManager routes phase-scoped-export accumulators through
    # export_results(ctx) so warmup records are excluded from profiling summaries.
    supports_phase_scoped_export = True

    def __init__(
        self,
        run: BenchmarkRun,
        **kwargs: Any,
    ) -> None:
        super().__init__(run=run, **kwargs)

        self._column_store = ColumnStore(initial_capacity=1024)
        # Credit/session numbers are scoped per phase and can restart at zero
        # for warmup and profiling. Use an internal append-only row index for
        # storage so warmup record 0 cannot be overwritten by profiling record 0.
        self._next_record_idx = 0

        # Run-level mean network RTT (ns), delivered by the RecordsManager before
        # summarize() when network latency calibration is active. None = no-op.
        self._network_rtt_ns: float | None = None

        # One-shot latch for the run-level "replay schedule degraded" warning
        # emitted by inject_replay_sched_lag_metrics (fires at most once per run).
        self._replay_degraded_warned: bool = False

        # One-shot latch for the mid-run "enable cache reporting" server-knob hint.
        self._warned_missing_cache_reporting: bool = False

        # Pooled speculative-decoding acceptance histogram, keyed by
        # (benchmark_phase, phase_index) so a phase-scoped export pools exactly
        # the same record set the masked scalar totals do -- the export mask
        # filters by phase kind and, when a concrete instance is requested, by
        # phase_index too. Keeps the pooled counts reconciled with
        # ``total_spec_decode_steps``. Dict aggregation has no home in the numpy
        # columnar store, so it lives here as the one dedicated reducer.
        self._acceptance_pool_by_phase: dict[
            tuple[str, int | None], dict[int, int]
        ] = {}

        # Derive functions for DERIVED metrics
        # _setup_metrics includes transitive dependencies (RECORD/AGGREGATE),
        # so filter to only metrics that actually have derive_value.
        # Bind derive_value off fresh per-run instances (not MetricRegistry
        # singletons) so any metric instance state (e.g. warn-once latches) is
        # scoped to this run rather than shared process-wide.
        self._derive_funcs: dict[
            MetricTagT, Callable[[MetricResultsDict], MetricValueTypeT]
        ] = {}
        # Derived tags anchored to a run-global reference (``timeslice_derivable
        # = False``, e.g. the replay send-lag family): re-deriving them per
        # timeslice would re-anchor each slice at its own reference and erase the
        # run-wide signal, so the per-slice derivation skips them.
        self._non_timeslice_derived_tags: set[MetricTagT] = set()
        for metric in self._setup_metrics(MetricType.DERIVED):
            if metric.type != MetricType.DERIVED:
                continue
            self._derive_funcs[metric.tag] = type(metric)().derive_value  # type: ignore
            if not getattr(metric, "timeslice_derivable", True):
                self._non_timeslice_derived_tags.add(metric.tag)

        _all_metric_classes: list[type[BaseMetric]] = MetricRegistry.all_classes()
        self._tags_to_types: dict[MetricTagT, MetricType] = {
            metric.tag: metric.type for metric in _all_metric_classes
        }

        # Aggregation kind per AGGREGATE tag — for vectorized windowed aggregation
        self._aggregation_kinds: dict[MetricTagT, AggregationKind] = {
            metric.tag: getattr(metric, "aggregation_kind", AggregationKind.SUM)
            for metric in _all_metric_classes
            if metric.type == MetricType.AGGREGATE
        }

        self._metric_classes: dict[MetricTagT, type[BaseMetric]] = {
            tag: MetricRegistry.get_class(tag) for tag in MetricRegistry.all_tags()
        }

        slice_dur = run.cfg.artifacts.slice_duration
        self._slice_duration_ns: int | None = (
            int(slice_dur * NANOS_PER_SECOND) if slice_dur else None
        )

    @property
    def column_store(self) -> ColumnStore:
        """Read-only access to the underlying columnar store for analyzers."""
        return self._column_store

    @property
    def record_count(self) -> int:
        """Number of records ingested so far."""
        n = self._column_store.count
        if n == 0:
            return 0
        return int(np.count_nonzero(~np.isnan(self._column_store.start_ns[:n])))

    async def process_record(self, record: MetricRecordsData) -> None:
        """Ingest a single ``MetricRecordsData`` into columnar storage."""
        self._maybe_hint_missing_cache_reporting(record)
        self._pool_spec_decode_record(record)
        idx = self._next_record_idx
        self._next_record_idx += 1
        meta = record.metadata

        # Compute generation_start_ns from wall-clock start + TTFT duration
        ttft_ns = record.metrics.get("time_to_first_token")
        gen_start = (
            float(meta.request_start_ns + int(ttft_ns)) if ttft_ns is not None else None
        )

        self._column_store.ingest(
            idx=idx,
            record_metrics=record.metrics,
            start_ns=float(meta.request_start_ns),
            end_ns=float(meta.request_end_ns),
            generation_start_ns=gen_start,
        )

        # Per-record metadata routing — see ``ColumnStore.ingest_metadata`` for
        # storage-type rationale. ``x_request_id`` is intentionally dropped:
        # cardinality == n_records (no grouping value) and per-record exporters
        # read it off the live record struct, never the column store.
        self._column_store.ingest_metadata(
            idx=idx,
            metadata_numeric={
                "session_num": meta.session_num,
                "credit_issued_ns": meta.credit_issued_ns,
                "request_ack_ns": meta.request_ack_ns,
                "cancellation_time_ns": meta.cancellation_time_ns,
                "turn_index": meta.turn_index,
            },
            metadata_string={},
            metadata_bool={
                "was_cancelled": meta.was_cancelled,
                "has_error": record.error is not None,
            },
            metadata_categorical={
                "worker_id": meta.worker_id,
                "record_processor_id": meta.record_processor_id,
                "benchmark_phase": str(meta.benchmark_phase),
                "phase_index": str(meta.phase_index)
                if meta.phase_index is not None
                else None,
                "x_correlation_id": meta.x_correlation_id,
                "conversation_id": meta.conversation_id,
            },
        )

    def _maybe_hint_missing_cache_reporting(self, record: MetricRecordsData) -> None:
        """Warn once, mid-run, when the server reports token usage but no prompt-cache
        reads — the signature of a cache-capable server that hasn't been told to
        report ``cached_tokens``. Fires on the first qualifying record so a long run
        can be aborted and re-launched with the flag set; the end-of-run console
        exporter emits the same hint for anyone who only reads the final summary.
        """
        if self._warned_missing_cache_reporting:
            return
        if usage_without_cache_in_record(record.metrics):
            self._warned_missing_cache_reporting = True
            self.warning(CACHE_REPORTING_HINT)

    def _pool_spec_decode_record(self, record: MetricRecordsData) -> None:
        """Sum a request's accepted-draft histogram into its
        ``(benchmark_phase, phase_index)`` pool.

        Skips records with no spec-decode stats and error records -- the latter
        never contribute the ``spec_decode_steps`` metric either, so excluding
        them here keeps the pooled counts reconciled with the masked scalar
        ``total_spec_decode_steps``.
        """
        spec = record.spec_decode_acceptance
        if spec is None or record.error is not None:
            return
        key = (str(record.metadata.benchmark_phase), record.metadata.phase_index)
        pool = self._acceptance_pool_by_phase.setdefault(key, {})
        for accepted_draft_count, steps in spec.acceptance_histogram.items():
            pool[accepted_draft_count] = pool.get(accepted_draft_count, 0) + steps

    def _pooled_acceptance_histogram(
        self, ctx: ExportContext | None
    ) -> dict[int, int] | None:
        """Return the pooled histogram for the exported phase, key-sorted.

        Mirrors the scalar export mask: a phase-scoped export selects pools by
        phase kind, and by ``phase_index`` too when a concrete instance was
        requested; otherwise it merges every same-kind instance. Windowed
        (realtime/timeslice) exports return None -- the pooled histogram is a
        run-level artifact, not defined per rolling window. A fully-unbounded
        export pools every phase.
        """
        if not self._acceptance_pool_by_phase:
            return None
        if ctx is not None and ctx.phase is not None:
            phase = str(ctx.phase)
            selected = [
                pool
                for (
                    pool_phase,
                    pool_index,
                ), pool in self._acceptance_pool_by_phase.items()
                if pool_phase == phase
                and (ctx.phase_index is None or pool_index == ctx.phase_index)
            ]
        elif ctx is not None and (ctx.start_ns is not None or ctx.end_ns is not None):
            return None
        else:
            selected = list(self._acceptance_pool_by_phase.values())
        merged: dict[int, int] = {}
        for pool in selected:
            for accepted_draft_count, steps in pool.items():
                merged[accepted_draft_count] = (
                    merged.get(accepted_draft_count, 0) + steps
                )
        if not merged:
            return None
        return {j: merged[j] for j in sorted(merged)}

    def query_time_range(self, start_ns: int, end_ns: int) -> BoolArray:
        """Return a boolean mask where True marks records in [start_ns, end_ns)."""
        n = self._column_store.count
        if n == 0:
            return np.array([], dtype=bool)
        ts = self._column_store.start_ns[:n]
        return ~np.isnan(ts) & (ts >= start_ns) & (ts < end_ns)

    def _mask_for_export_context(self, ctx: ExportContext | None) -> BoolArray | None:
        """Build a record mask for the requested export phase/window.

        Phase-scoped contexts select by the per-record credit-phase tag alone:
        the tag is authoritative for phase membership, so the wall-clock bounds
        are only redundant with it. Applying the half-open time window on top
        drops legitimate boundary records on coarse clocks — Windows
        ``time.time_ns()`` before Python 3.13 updates in ~0.5-15.6ms ticks, so a
        straggler's ``request_start_ns`` can equal the phase's
        ``requests_end_ns`` and fail a strict ``start_ns < end_ns`` check.
        Phase-less contexts (realtime rolling windows) keep the half-open
        ``[start_ns, end_ns)`` semantics so adjacent windows never overlap.
        """
        if ctx is None:
            return None
        n = self._column_store.count
        if n == 0:
            return np.zeros(0, dtype=bool)

        mask = ~np.isnan(self._column_store.start_ns[:n])
        if ctx.phase is not None:
            phase_value = str(ctx.phase)
            mask &= self._column_store.mask_for_categorical(
                "benchmark_phase", phase_value
            )
            if ctx.phase_index is not None:
                mask &= self._column_store.mask_for_categorical(
                    "phase_index", str(ctx.phase_index)
                )
            return mask
        if ctx.start_ns is not None:
            mask &= self._column_store.start_ns[:n] >= ctx.start_ns
        if ctx.end_ns is not None:
            mask &= self._column_store.start_ns[:n] < ctx.end_ns
        return mask

    def _aggregate_values(self, tag: MetricTagT, values: np.ndarray) -> float:
        """Apply the tag's aggregation function to an array of values."""
        kind = self._aggregation_kinds.get(tag, AggregationKind.SUM)
        return _AGGREGATE_FUNCS[kind](values)

    def _compute_results(
        self,
        mask: BoolArray | None = None,
        *,
        window_start_ns: int | None = None,
        window_end_ns: int | None = None,
        is_timeslice: bool = False,
    ) -> dict[MetricTagT, MetricResult]:
        """Phases: collect scalars/arrays, resolve derived, build MetricResults.

        For metrics flagged ``PERCENTILE_INCLUDES_FAILED_REQUESTS`` (issue #688),
        appends a separate ``adj_<tag>`` MetricResult with the failure-inflated
        distribution after the regular build pass.

        ``is_timeslice`` skips derived metrics anchored to a run-global reference
        (``timeslice_derivable = False``); see ``_non_timeslice_derived_tags``.
        """
        scalar_dict: MetricResultsDict = MetricResultsDict()
        scalar_dict.window_start_ns = window_start_ns
        scalar_dict.window_end_ns = window_end_ns
        record_arrays: dict[MetricTagT, tuple[FloatArray, float]] = {}
        sketch_results: dict[MetricTagT, MetricResult] = {}

        self._collect_scalars_and_arrays(
            mask, scalar_dict, record_arrays, sketch_results
        )
        self._resolve_derived_metrics(scalar_dict, is_timeslice=is_timeslice)

        output = self._build_metric_results(scalar_dict, record_arrays, sketch_results)

        n = self._column_store.count
        if n > 0:
            is_error = self._column_store.metadata_bool("has_error")[:n] == 1
            if mask is not None:
                is_error = is_error & mask
            error_count = int(is_error.sum())
            inject_adjusted_latency_metrics(
                output, record_arrays, error_count, self._metric_classes
            )
        return output

    def _build_metric_results(
        self,
        scalar_dict: MetricResultsDict,
        record_arrays: dict[MetricTagT, tuple[FloatArray, float]],
        sketch_results: dict[MetricTagT, MetricResult],
    ) -> dict[MetricTagT, MetricResult]:
        """Convert scalar_dict + record_arrays + sketch_results into a result dict."""
        output: dict[MetricTagT, MetricResult] = {}
        for tag, value in scalar_dict.items():
            if tag in sketch_results:
                output[tag] = sketch_results[tag]
                continue
            mc = self._metric_classes.get(tag)
            if mc is None:
                continue
            if tag in record_arrays:
                arr, arr_sum = record_arrays[tag]
                output[tag] = metric_result_from_array(
                    tag, mc.header, str(mc.unit), arr, arr_sum
                )
            elif isinstance(value, (int, float)):
                output[tag] = MetricResult(
                    tag=tag,
                    header=mc.header,
                    unit=str(mc.unit),
                    avg=value,
                    count=1,
                )
        return output

    def _collect_scalars_and_arrays(
        self,
        mask: BoolArray | None,
        scalar_dict: MetricResultsDict,
        record_arrays: dict[MetricTagT, tuple[FloatArray, float]],
        sketch_results: dict[MetricTagT, MetricResult],
    ) -> None:
        """Iterate columns, populating scalar_dict and record_arrays in-place."""
        store = self._column_store
        full_dataset = mask is None

        for tag in store.numeric_tags():
            if full_dataset:
                col = store.numeric(tag)
                clean = col[~np.isnan(col)]
            else:
                values = store.numeric(tag)[mask]
                clean = values[~np.isnan(values)]
            if len(clean) == 0:
                continue

            metric_type = self._tags_to_types.get(tag)
            if metric_type == MetricType.RECORD:
                # O(1) running sum for the full dataset; np.sum for windowed
                s = store.numeric_sum(tag) if full_dataset else float(np.sum(clean))
                scalar_dict[tag] = s
                record_arrays[tag] = (clean, s)
            elif metric_type == MetricType.AGGREGATE:
                scalar_dict[tag] = self._aggregate_values(tag, clean)

        for tag in store.ragged_tags():
            self._collect_one_list_column(
                tag,
                mask=mask,
                full_dataset=full_dataset,
                scalar_dict=scalar_dict,
                record_arrays=record_arrays,
                sketch_results=sketch_results,
            )

    def _collect_one_list_column(
        self,
        tag: MetricTagT,
        *,
        mask: BoolArray | None,
        full_dataset: bool,
        scalar_dict: MetricResultsDict,
        record_arrays: dict[MetricTagT, tuple[FloatArray, float]],
        sketch_results: dict[MetricTagT, MetricResult],
    ) -> None:
        """Forks on the backend's ``SUPPORTS_PER_RECORD_REPLAY`` flag.

        Replay-capable backends (RaggedSeries) emit (values, sum) into
        ``record_arrays``. Sketch backends (t-digest) emit a pre-built
        MetricResult into ``sketch_results`` and skip windowed (timeslice)
        computation entirely — the sketch has no per-record indices.
        """
        backend = self._column_store.ragged(tag)
        if getattr(backend, "SUPPORTS_PER_RECORD_REPLAY", False):
            # metric_result_from_array sorts its input in place; backend.values is
            # a view into the ragged buffer that compute_sweep_curves reads later
            # (against unsorted offsets/record_indices), so the full-dataset branch
            # must copy. get_values_for_mask already returns a fresh masked copy.
            filtered = (
                backend.values.copy()
                if full_dataset
                else backend.get_values_for_mask(mask)
            )
            if len(filtered) == 0:
                return
            s = float(np.sum(filtered))
            scalar_dict[tag] = s
            record_arrays[tag] = (filtered, s)
            return
        if not full_dataset or len(backend) == 0:
            return
        mc = self._metric_classes.get(tag)
        if mc is None:
            return
        sketch_results[tag] = backend.to_result(tag, mc.header, str(mc.unit))
        # Expose the running sum so derived-sum metrics can reach it
        # uniformly via the scalar_dict.
        scalar_dict[tag] = float(backend.sum)

    def _warn_replay_degraded(self, p50: float, p90: float, p99: float) -> None:
        """Emit the run-level replay-schedule-degraded warning at most once."""
        if self._replay_degraded_warned:
            return
        self._replay_degraded_warned = True
        self.warning(
            f"Replay schedule degraded: anchored send lag p50={p50:.0f} ms, "
            f"p90={p90:.0f} ms, p99={p99:.0f} ms exceeds "
            f"{REPLAY_SCHED_DEGRADED_THRESHOLD_MS:.0f} ms. Request timing no longer "
            f"tracks the recorded schedule; consider lowering replay_speedup or the "
            f"offered load."
        )

    def _resolve_derived_metrics(
        self, scalar_dict: MetricResultsDict, *, is_timeslice: bool = False
    ) -> None:
        """Run derive functions over the scalar dict, logging failures.

        When ``is_timeslice`` is True, derived metrics anchored to a run-global
        reference (``_non_timeslice_derived_tags``) are skipped so each slice is
        not re-anchored at its own reference.
        """
        for tag, derive_func in self._derive_funcs.items():
            if is_timeslice and tag in self._non_timeslice_derived_tags:
                continue
            try:
                scalar_dict[tag] = derive_func(scalar_dict)
            except NoMetricValue as e:
                self.debug(f"No metric value for derived metric '{tag}': {e!r}")
            except Exception as e:  # noqa: BLE001 - one bad derive must not abort the rest of the summary
                self.warning(f"Error deriving metric '{tag}': {e!r}")

    def compute_results_for_mask(
        self,
        mask: BoolArray,
        *,
        window_start_ns: int | None = None,
        window_end_ns: int | None = None,
    ) -> dict[MetricTagT, MetricResult]:
        """Build, derive, and convert metric results for an arbitrary boolean mask.

        Public interface for analyzers that need windowed metric computation
        without accessing private methods. Results are converted to display
        units before returning.
        """
        raw = self._compute_results(
            mask, window_start_ns=window_start_ns, window_end_ns=window_end_ns
        )
        return self._convert_display_units(raw)

    def _convert_display_units(
        self,
        results: dict[MetricTagT, MetricResult],
    ) -> dict[MetricTagT, MetricResult]:
        """Convert all metric results from native units to display units."""
        registry = _MetricClassLookup(self._metric_classes)
        return {
            tag: to_display_unit(result, registry) for tag, result in results.items()
        }

    def _should_include_in_summary(self, tag: MetricTagT) -> bool:
        """Return False for hidden internal/experimental metrics."""
        metric_class = self._metric_classes.get(tag)
        if metric_class is None:
            return True
        has_flags = getattr(metric_class, "has_flags", None)
        if not callable(has_flags):
            return True
        if (
            has_flags(MetricFlags.INTERNAL)
            and not Environment.DEV.SHOW_INTERNAL_METRICS
        ):
            return False
        return not (
            has_flags(MetricFlags.EXPERIMENTAL)
            and not Environment.DEV.SHOW_EXPERIMENTAL_METRICS
        )

    def _filter_hidden_metrics(
        self, results: dict[MetricTagT, MetricResult]
    ) -> dict[MetricTagT, MetricResult]:
        """Drop computed metrics that should not appear in summary exports."""
        return {
            tag: result
            for tag, result in results.items()
            if self._should_include_in_summary(tag)
        }

    def set_network_rtt_ns(self, rtt_ns: float | None) -> None:
        """Set the run-level mean network RTT (ns) to subtract from latency metrics.

        Called by the RecordsManager before summarize() when network latency
        calibration is enabled (or a manual override was provided). A falsy value
        disables the adjustment (no network_adjusted_* metrics are emitted).
        """
        self._network_rtt_ns = rtt_ns

    async def summarize(
        self, ctx: SummaryContext | None = None
    ) -> AccumulatorMetricsSummary:
        """Compute and return aggregated metric results.

        If slice_duration is configured, also computes per-timeslice results
        by partitioning the data into time windows. Always derives the
        coordinated-omission-aware ``effective_latency`` and the
        ``credit_to_start_latency`` queue-wait metric from stored timestamps,
        plus a per-``turn_index`` TTFT trend that surfaces KV-cache effectiveness.
        """
        export_ctx: ExportContext | None = None
        if ctx is not None and (ctx.start_ns or ctx.end_ns or ctx.phase is not None):
            export_ctx = ExportContext(
                start_ns=ctx.start_ns or None,
                end_ns=ctx.end_ns or None,
                phase=ctx.phase,
                phase_index=ctx.phase_index,
            )
        return self._summarize_for_export_context(export_ctx)

    def _summarize_for_export_context(
        self, ctx: ExportContext | None = None
    ) -> AccumulatorMetricsSummary:
        mask = self._mask_for_export_context(ctx)

        window_start_ns = ctx.start_ns if ctx is not None else None
        window_end_ns = ctx.end_ns if ctx is not None else None
        overall_results = self._compute_results(
            mask,
            window_start_ns=window_start_ns,
            window_end_ns=window_end_ns,
        )

        timeslices: list[TimesliceResult] | None = None
        adjusted_arrays: dict[str, FloatArray] | None = None

        has_records = self._column_store.count > 0 and (
            mask is None or bool(mask.any())
        )
        if has_records:
            # Compute sweeps once for both overall and timeslice injection.
            sweeps = compute_sweep_curves(self._column_store, mask=mask)
            self._inject_sweep_metrics(
                overall_results,
                sweeps,
                window_start_ns=window_start_ns,
                window_end_ns=window_end_ns,
            )
            # Network-RTT-adjusted latency: the per-record subtraction is
            # window-independent, so compute the clamped arrays ONCE here and let
            # the overall summary and every timeslice aggregate masked views. No-op
            # unless the RecordsManager delivered a (truthy) RTT via set_network_rtt_ns.
            if self._network_rtt_ns:
                adjusted_arrays = compute_network_adjusted_arrays(
                    self._column_store, self._network_rtt_ns
                )
            if self._slice_duration_ns is not None:
                timeslices = self._compute_timeslices(
                    sweeps, mask=mask, adjusted_arrays=adjusted_arrays
                )

        overall_results = self._convert_display_units(overall_results)

        # Derived latency metrics — already in display units (ms), so injected
        # after _convert_display_units to bypass the registry lookup.
        if has_records:
            inject_derived_latency_metrics(
                self._column_store, overall_results, mask=mask
            )
            if adjusted_arrays is not None:
                inject_network_adjusted_from_arrays(
                    adjusted_arrays,
                    overall_results,
                    self._network_rtt_ns,
                    mask=mask,
                )
            # Run-scoped replay send-lag family (fixed-schedule only): anchored at
            # the run-global least-late request, so computed once over the masked
            # offset column, never per timeslice.
            inject_replay_sched_lag_metrics(
                self._column_store,
                overall_results,
                mask=mask,
                warn_degraded=self._warn_replay_degraded,
            )

        overall_results = self._filter_hidden_metrics(overall_results)
        self.debug(lambda: f"Summarized {len(overall_results)} metric results")
        return AccumulatorMetricsSummary(
            results=overall_results,
            timeslices=timeslices,
            pooled_spec_decode_acceptance_histogram=self._pooled_acceptance_histogram(
                ctx
            ),
        )

    async def export_results(self, ctx: ExportContext) -> AccumulatorMetricsSummary:
        """Export final metrics results for the requested phase/window."""
        return self._summarize_for_export_context(ctx)

    def _inject_sweep_metrics(
        self,
        results: dict[MetricTagT, MetricResult],
        sweeps: Any,
        *,
        window_start_ns: int | None = None,
        window_end_ns: int | None = None,
    ) -> None:
        """Inject time-weighted sweep metrics into results.

        ``sweeps`` is the ``SweepLineCurves`` bundle from
        ``aiperf.analysis.sweepline``.
        """
        if len(sweeps.concurrency_ts) == 0:
            return
        window_start = (
            float(window_start_ns)
            if window_start_ns is not None
            else float(sweeps.concurrency_ts[0])
        )
        window_end = (
            float(window_end_ns)
            if window_end_ns is not None
            else float(sweeps.concurrency_ts[-1])
        )
        results.update(sweeps.compute_metrics(window_start, window_end))

    def _compute_timeslices(
        self,
        sweeps: Any,
        mask: BoolArray | None = None,
        adjusted_arrays: dict[str, FloatArray] | None = None,
    ) -> list[TimesliceResult]:
        """Compute per-timeslice results by partitioning the time range.

        Sweeps are pre-computed once in ``summarize()`` and windowed per
        timeslice via ``compute_time_weighted_stats`` — O(T log M) total.

        Slice grid is sized to span [min(start_ns), max(end_ns)], the actual
        wall-clock span of activity. The last slice's window_end is clipped
        to max(end_ns) so the window covers only real activity (otherwise
        sweep metrics like throughput / concurrency get diluted by phantom
        idle padding past the run end). Partial slices are flagged via
        ``TimesliceResult.is_complete=False`` so consumers can filter them.

        Returns:
            Per-slice results in chronological order. Each entry bundles
            window bounds with metric results in display units. Empty bins
            (slices with no records) are skipped, so list position is dense
            even if the underlying grid has gaps.
        """
        assert self._slice_duration_ns is not None

        store = self._column_store
        n = store.count
        start_ns = store.start_ns[:n]
        end_ns = store.end_ns[:n]
        filled = ~np.isnan(start_ns)
        if mask is not None:
            filled &= mask
        filled_ts = start_ns[filled]

        if len(filled_ts) == 0:
            return []

        min_ts = float(np.nanmin(filled_ts))
        # Use the latest of any record's start or end to size the grid: the run
        # ends when the last record ends. Real data has end_ns >= start_ns, but
        # take the max of both so artificial fixtures with end < start still
        # bucket every record. Falls back to max(start_ns) if no end_ns is
        # recorded.
        max_start_ts = float(np.nanmax(filled_ts))
        filled_end = ~np.isnan(end_ns)
        if filled_end.any():
            max_ts = max(max_start_ts, float(np.nanmax(end_ns[filled_end])))
        else:
            max_ts = max_start_ts

        # Build slice edges — compute n_slices first to avoid np.arange stop-exclusion issues
        n_slices = int((max_ts - min_ts) / self._slice_duration_ns) + 1
        edges = min_ts + np.arange(n_slices + 1) * self._slice_duration_ns

        # Assign each record to a bin — O(n) total via digitize
        bins = np.digitize(filled_ts, edges) - 1

        timeslices: list[TimesliceResult] = []
        filled_indices = np.where(filled)[0]

        for bin_idx in range(len(edges) - 1):
            bin_mask_local = bins == bin_idx
            if not bin_mask_local.any():
                continue
            # Expand local mask to full-array mask
            full_mask = np.zeros(n, dtype=bool)
            full_mask[filled_indices[bin_mask_local]] = True

            raw_window_end = float(edges[bin_idx + 1])
            window_start = float(edges[bin_idx])
            # Clip the last slice's end to the run end so sweep metrics aren't
            # diluted by idle padding. is_complete distinguishes clipped slices
            # from full-duration ones for downstream consumers.
            is_complete = raw_window_end <= max_ts
            window_end = raw_window_end if is_complete else max_ts

            results = self._compute_results(
                full_mask,
                window_start_ns=int(window_start),
                window_end_ns=int(window_end),
                is_timeslice=True,
            )
            if len(results) == 0:
                continue
            results.update(sweeps.compute_metrics(window_start, window_end))
            results = self._convert_display_units(results)
            # Network-RTT-adjusted latency metrics per window, aggregated from the
            # arrays precomputed once in summarize() — this window just slices its
            # records out via full_mask. None unless a run-level RTT was delivered.
            if adjusted_arrays is not None:
                inject_network_adjusted_from_arrays(
                    adjusted_arrays,
                    results,
                    self._network_rtt_ns,
                    mask=full_mask,
                )
            results = self._filter_hidden_metrics(results)
            timeslices.append(
                TimesliceResult(
                    start_ns=int(window_start),
                    end_ns=int(window_end),
                    is_complete=None if is_complete else False,
                    metric_results=results,
                )
            )

        return timeslices

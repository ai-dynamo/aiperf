# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numpy-backed metrics accumulator with columnar storage and dynamic timeslicing."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import (
    AggregationKind,
    MetricType,
    MetricValueTypeT,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import MetricResult, TimesliceResult
from aiperf.common.types import MetricTagT
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.metrics.accumulator_sweeps import compute_sweep_curves
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.column_store import ColumnStore
from aiperf.metrics.derived_latency import (
    inject_adjusted_latency_metrics,
    inject_derived_latency_metrics,
)
from aiperf.metrics.display_units import to_display_unit
from aiperf.metrics.metric_dicts import MetricResultsDict, metric_result_from_array
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import ExportContext, SummaryContext


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


_AGGREGATE_FUNCS: dict[AggregationKind, Callable[[np.ndarray], float]] = {
    AggregationKind.SUM: lambda a: float(np.sum(a)),
    AggregationKind.MAX: lambda a: float(np.max(a)),
    AggregationKind.MIN: lambda a: float(np.min(a)),
}


class MetricsAccumulator(BaseMetricsProcessor):
    """Numpy-backed accumulator for inference metrics.

    Session_num-indexed NaN-sparse columnar storage; RECORD metrics get
    per-value stats, AGGREGATE metrics one scalar via :class:`AggregationKind`,
    DERIVED metrics computed from those at summarize time.
    """

    def __init__(
        self,
        user_config: UserConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(user_config=user_config, **kwargs)

        self._column_store = ColumnStore(initial_capacity=1024)

        # Derive functions for DERIVED metrics
        # _setup_metrics includes transitive dependencies (RECORD/AGGREGATE),
        # so filter to only metrics that actually have derive_value.
        self._derive_funcs: dict[
            MetricTagT, Callable[[MetricResultsDict], MetricValueTypeT]
        ] = {
            metric.tag: metric.derive_value  # type: ignore
            for metric in self._setup_metrics(MetricType.DERIVED)
            if metric.type == MetricType.DERIVED
        }

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

        slice_dur = user_config.output.slice_duration
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
        idx = record.metadata.session_num
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
                "x_correlation_id": meta.x_correlation_id,
                "conversation_id": meta.conversation_id,
            },
        )

    def query_time_range(self, start_ns: int, end_ns: int) -> BoolArray:
        """Return a boolean mask where True marks records in [start_ns, end_ns)."""
        n = self._column_store.count
        if n == 0:
            return np.array([], dtype=bool)
        ts = self._column_store.start_ns[:n]
        return ~np.isnan(ts) & (ts >= start_ns) & (ts < end_ns)

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
    ) -> dict[MetricTagT, MetricResult]:
        """Phases: collect scalars/arrays, resolve derived, build MetricResults.

        For metrics flagged ``PERCENTILE_INCLUDES_FAILED_REQUESTS`` (issue #688),
        appends a separate ``adj_<tag>`` MetricResult with the failure-inflated
        distribution after the regular build pass.
        """
        scalar_dict: MetricResultsDict = MetricResultsDict()
        scalar_dict.window_start_ns = window_start_ns
        scalar_dict.window_end_ns = window_end_ns
        record_arrays: dict[MetricTagT, tuple[FloatArray, float]] = {}
        sketch_results: dict[MetricTagT, MetricResult] = {}

        self._collect_scalars_and_arrays(
            mask, scalar_dict, record_arrays, sketch_results
        )
        self._resolve_derived_metrics(scalar_dict)

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

    def _resolve_derived_metrics(self, scalar_dict: MetricResultsDict) -> None:
        """Run derive functions over the scalar dict, logging failures."""
        for tag, derive_func in self._derive_funcs.items():
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

    @staticmethod
    def _convert_display_units(
        results: dict[MetricTagT, MetricResult],
    ) -> dict[MetricTagT, MetricResult]:
        """Convert all metric results from native units to display units."""
        return {
            tag: to_display_unit(result, MetricRegistry)
            for tag, result in results.items()
        }

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
        overall_results = self._compute_results()

        timeslices: list[TimesliceResult] | None = None

        if self._column_store.count > 0:
            # Compute sweeps once for both overall and timeslice injection.
            sweeps = compute_sweep_curves(self._column_store)
            self._inject_sweep_metrics(overall_results, sweeps)
            if self._slice_duration_ns is not None:
                timeslices = self._compute_timeslices(sweeps)

        overall_results = self._convert_display_units(overall_results)

        # Derived latency metrics — already in display units (ms), so injected
        # after _convert_display_units to bypass the registry lookup.
        if self._column_store.count > 0:
            inject_derived_latency_metrics(self._column_store, overall_results)

        self.debug(lambda: f"Summarized {len(overall_results)} metric results")
        return AccumulatorMetricsSummary(
            results=overall_results,
            timeslices=timeslices,
        )

    async def export_results(self, ctx: ExportContext) -> AccumulatorMetricsSummary:
        """Export final metrics results. Delegates to summarize()."""
        return await self.summarize()

    def _inject_sweep_metrics(
        self,
        results: dict[MetricTagT, MetricResult],
        sweeps: Any,
    ) -> None:
        """Inject time-weighted sweep metrics into results.

        ``sweeps`` is the ``SweepLineCurves`` bundle from
        ``aiperf.analysis.sweepline``.
        """
        if len(sweeps.concurrency_ts) == 0:
            return
        window_start = float(sweeps.concurrency_ts[0])
        window_end = float(sweeps.concurrency_ts[-1])
        results.update(sweeps.compute_metrics(window_start, window_end))

    def _compute_timeslices(
        self,
        sweeps: Any,
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
            )
            if len(results) == 0:
                continue
            results.update(sweeps.compute_metrics(window_start, window_end))
            results = self._convert_display_units(results)
            timeslices.append(
                TimesliceResult(
                    start_ns=int(window_start),
                    end_ns=int(window_end),
                    is_complete=None if is_complete else False,
                    metric_results=results,
                )
            )

        return timeslices

    async def full_metrics(self) -> dict[MetricTagT, MetricResult]:
        """Returns the full metrics results, including derived metrics."""
        return self._compute_results()

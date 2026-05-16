# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state detection and windowed metric computation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from aiperf.analysis.ramp_detection import (
    cusum_steady_state_window,
    detect_steady_state_window,
    manual_steady_state_window,
    mser5_boundary_ns,
)
from aiperf.analysis.stationarity import batch_means_trend_test
from aiperf.analysis.sweepline import (
    SweepLineCurves,
    concurrency_sweep_line,
    divide_step_functions,
    prefill_throughput_sweep_line,
    throughput_sweep_line,
    total_throughput_sweep_line,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PluginDisabled
from aiperf.common.models import MetricResult
from aiperf.metrics.accumulator_sweeps import (
    icl_aware_throughput,
    icl_aware_tokens_in_flight,
)
from aiperf.post_processors.steady_state_models import (
    SteadyStateSummary,
    SteadyStateWindowMetadata,
)

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import SummaryContext
    from aiperf.config import BenchmarkRun
    from aiperf.metrics.accumulator import MetricsAccumulator
    from aiperf.plugin.enums import AccumulatorType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SteadyStateDataset:
    """Per-record arrays + filled mask snapshotted from the accumulator."""

    store: Any
    start_ns: np.ndarray
    end_ns: np.ndarray
    filled: np.ndarray
    latency: np.ndarray
    ttft: np.ndarray
    generation_start_ns: np.ndarray
    output_tokens: np.ndarray
    input_tokens: np.ndarray


def _build_steady_state_summary(
    *,
    windowed_results: dict[str, MetricResult],
    sweep_results: dict[str, MetricResult],
    metadata: SteadyStateWindowMetadata,
) -> SteadyStateSummary:
    """Compose a SteadyStateSummary from windowed metrics + sweep curves + metadata."""
    return SteadyStateSummary(
        results=windowed_results,
        effective_concurrency=sweep_results["effective_concurrency"],
        effective_throughput=sweep_results["effective_throughput"],
        effective_prefill_throughput=sweep_results["effective_prefill_throughput"],
        effective_generation_concurrency=sweep_results[
            "effective_generation_concurrency"
        ],
        effective_prefill_concurrency=sweep_results["effective_prefill_concurrency"],
        effective_total_throughput=sweep_results["effective_total_throughput"],
        effective_throughput_per_user=sweep_results["effective_throughput_per_user"],
        effective_prefill_throughput_per_user=sweep_results[
            "effective_prefill_throughput_per_user"
        ],
        tokens_in_flight=sweep_results["tokens_in_flight"],
        window_metadata=metadata,
    )


class SteadyStateAnalyzer:
    """Event-based steady-state detection and windowed metric computation.

    Implements AnalyzerProtocol. No record ingestion — reads columnar
    arrays from MetricsAccumulator at summarize time.
    """

    required_accumulators: ClassVar[set[AccumulatorType]] = {"metric_results"}
    summary_dependencies: ClassVar[list[AccumulatorType]] = ["metric_results"]

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        ss_config = run.cfg.output.steady_state
        if not ss_config.enabled:
            raise PluginDisabled("Steady-state analysis is disabled")

        env_ss = Environment.STEADY_STATE
        self._min_window_pct = (
            ss_config.min_window_pct
            if "min_window_pct" in ss_config.model_fields_set
            else env_ss.MIN_WINDOW_PCT
        )
        self._start_pct = ss_config.start_pct
        self._end_pct = ss_config.end_pct
        self._bootstrap_iterations = (
            ss_config.bootstrap_iterations
            if "bootstrap_iterations" in ss_config.model_fields_set
            else env_ss.BOOTSTRAP_ITERATIONS
        )

    async def summarize(self, ctx: SummaryContext) -> SteadyStateSummary:
        """Detect steady-state window and compute windowed metrics."""
        from aiperf.metrics.accumulator import MetricsAccumulator
        from aiperf.plugin.enums import AccumulatorType

        metrics_acc: MetricsAccumulator | None = ctx.get_accumulator(
            AccumulatorType.METRIC_RESULTS
        )
        if metrics_acc is None or not isinstance(metrics_acc, MetricsAccumulator):
            raise PluginDisabled("MetricsAccumulator not available")

        ds = self._load_dataset(metrics_acc)
        sorted_c_ts, concurrency = concurrency_sweep_line(ds.start_ns, ds.end_ns)
        sorted_t_ts, tput = throughput_sweep_line(
            ds.generation_start_ns, ds.end_ns, ds.output_tokens
        )

        window_start, window_end, detection_method, signal_diag = (
            self._detect_or_override_window(
                start_ns=ds.start_ns,
                end_ns=ds.end_ns,
                filled=ds.filled,
                latency=ds.latency,
                ttft=ds.ttft,
                sorted_c_ts=sorted_c_ts,
                concurrency=concurrency,
                sorted_t_ts=sorted_t_ts,
                tput=tput,
            )
        )

        sweeps = self._build_sweep_curves(
            store=ds.store,
            metrics_acc=metrics_acc,
            start_ns=ds.start_ns,
            end_ns=ds.end_ns,
            generation_start_ns=ds.generation_start_ns,
            input_tokens=ds.input_tokens,
            output_tokens=ds.output_tokens,
            sorted_c_ts=sorted_c_ts,
            concurrency=concurrency,
        )
        sweep_results = sweeps.compute_metrics(window_start, window_end)

        ss_mask = ds.filled & (ds.start_ns >= window_start) & (ds.end_ns <= window_end)
        metadata = self._build_window_metadata(
            window_start=window_start,
            window_end=window_end,
            total_requests=int(ds.filled.sum()),
            steady_state_requests=int(ss_mask.sum()),
            detection_method=detection_method,
            signal_diag=signal_diag,
            latency=ds.latency,
            ttft=ds.ttft,
            ss_mask=ss_mask,
            start_ns=ds.start_ns,
            end_ns=ds.end_ns,
            generation_start_ns=ds.generation_start_ns,
            output_tokens=ds.output_tokens,
        )

        windowed_results = metrics_acc.compute_results_for_mask(
            ss_mask,
            window_start_ns=int(window_start),
            window_end_ns=int(window_end),
        )
        return _build_steady_state_summary(
            windowed_results=windowed_results,
            sweep_results=sweep_results,
            metadata=metadata,
        )

    @staticmethod
    def _load_dataset(metrics_acc: Any) -> _SteadyStateDataset:
        store = metrics_acc.column_store
        n = store.count
        if n == 0:
            raise PluginDisabled("No records available for steady-state detection")

        start_ns = store.start_ns[:n]
        end_ns = store.end_ns[:n]
        filled = ~np.isnan(start_ns) & ~np.isnan(end_ns)
        if not filled.any():
            raise PluginDisabled("No valid records for steady-state detection")

        return _SteadyStateDataset(
            store=store,
            start_ns=start_ns,
            end_ns=end_ns,
            filled=filled,
            latency=store.numeric("request_latency"),
            ttft=store.numeric("time_to_first_token"),
            generation_start_ns=store.generation_start_ns[:n],
            output_tokens=store.numeric("output_sequence_length"),
            input_tokens=store.numeric("input_sequence_length"),
        )

    def _build_window_metadata(
        self,
        *,
        window_start: float,
        window_end: float,
        total_requests: int,
        steady_state_requests: int,
        detection_method: str,
        signal_diag: dict[str, float | None],
        latency: np.ndarray,
        ttft: np.ndarray,
        ss_mask: np.ndarray,
        start_ns: np.ndarray,
        end_ns: np.ndarray,
        generation_start_ns: np.ndarray,
        output_tokens: np.ndarray,
    ) -> SteadyStateWindowMetadata:
        sample_quality = self._sample_quality(total_requests, steady_state_requests)
        stationarity = self._stationarity(latency, ss_mask)
        bootstrap = self._maybe_bootstrap(
            start_ns=start_ns,
            end_ns=end_ns,
            latency=latency,
            ttft=ttft,
            generation_start_ns=generation_start_ns,
            output_tokens=output_tokens,
        )
        return SteadyStateWindowMetadata(
            ramp_up_end_ns=window_start,
            ramp_down_start_ns=window_end,
            steady_state_duration_ns=window_end - window_start,
            total_requests=total_requests,
            steady_state_requests=steady_state_requests,
            detection_method=detection_method,
            fraction_retained=sample_quality["fraction_retained"],
            variance_inflation_factor=sample_quality["variance_inflation_factor"],
            effective_p99_sample_size=sample_quality["effective_p99_sample_size"],
            sample_size_warning=sample_quality["sample_size_warning"],
            trend_correlation=stationarity["trend_correlation"],
            trend_p_value=stationarity["trend_p_value"],
            stationarity_warning=stationarity["stationarity_warning"],
            cusum_ramp_up_end_ns=signal_diag["cusum_start"],
            cusum_ramp_down_start_ns=signal_diag["cusum_end"],
            mser5_latency_ramp_up_end_ns=signal_diag["lat_start"],
            mser5_latency_ramp_down_start_ns=signal_diag["lat_end"],
            mser5_ttft_ramp_up_end_ns=signal_diag["ttft_start"],
            mser5_ttft_ramp_down_start_ns=signal_diag["ttft_end"],
            cusum_throughput_ramp_up_end_ns=signal_diag["tput_start"],
            cusum_throughput_ramp_down_start_ns=signal_diag["tput_end"],
            bootstrap_ci_ramp_up_ns=bootstrap["ci_ramp_up"],
            bootstrap_ci_ramp_down_ns=bootstrap["ci_ramp_down"],
            bootstrap_ci_mean_latency=bootstrap["ci_mean_lat"],
            bootstrap_ci_p99_latency=bootstrap["ci_p99_lat"],
            bootstrap_n_iterations=bootstrap["n_iterations"],
        )

    def _detect_or_override_window(
        self,
        *,
        start_ns: np.ndarray,
        end_ns: np.ndarray,
        filled: np.ndarray,
        latency: np.ndarray,
        ttft: np.ndarray,
        sorted_c_ts: np.ndarray,
        concurrency: np.ndarray,
        sorted_t_ts: np.ndarray,
        tput: np.ndarray,
    ) -> tuple[float, float, str, dict[str, float | None]]:
        """Pick window via user override or automatic detection, returning per-signal diagnostics."""
        diag: dict[str, float | None] = {
            "cusum_start": None,
            "cusum_end": None,
            "lat_start": None,
            "lat_end": None,
            "ttft_start": None,
            "ttft_end": None,
            "tput_start": None,
            "tput_end": None,
        }
        if self._start_pct is not None and self._end_pct is not None:
            min_ts = float(np.nanmin(start_ns[filled]))
            max_ts = float(np.nanmax(end_ns[filled]))
            window_start, window_end = manual_steady_state_window(
                min_ts, max_ts, self._start_pct, self._end_pct
            )
            return window_start, window_end, "user_override", diag

        window_start, window_end, method = detect_steady_state_window(
            sorted_c_ts,
            concurrency,
            start_ns,
            end_ns,
            latency=latency,
            ttft=ttft,
            min_window_pct=self._min_window_pct,
            sorted_tput_ts=sorted_t_ts if len(sorted_t_ts) > 0 else None,
            throughput=tput if len(sorted_t_ts) > 0 else None,
        )
        diag["cusum_start"], diag["cusum_end"] = cusum_steady_state_window(
            sorted_c_ts, concurrency, min_window_pct=0.0
        )
        diag["lat_start"], diag["lat_end"] = mser5_boundary_ns(
            latency, start_ns, end_ns, filled
        )
        diag["ttft_start"], diag["ttft_end"] = mser5_boundary_ns(
            ttft, start_ns, end_ns, filled
        )
        if len(sorted_t_ts) > 0:
            diag["tput_start"], diag["tput_end"] = cusum_steady_state_window(
                sorted_t_ts, tput, min_window_pct=0.0
            )
        return window_start, window_end, method, diag

    @staticmethod
    def _build_sweep_curves(
        *,
        store: Any,
        metrics_acc: Any,
        start_ns: np.ndarray,
        end_ns: np.ndarray,
        generation_start_ns: np.ndarray,
        input_tokens: np.ndarray,
        output_tokens: np.ndarray,
        sorted_c_ts: np.ndarray,
        concurrency: np.ndarray,
    ) -> SweepLineCurves:
        tput_ts, tput_vals = icl_aware_throughput(
            store, generation_start_ns, end_ns, output_tokens
        )
        sorted_p_ts, prefill_tput = prefill_throughput_sweep_line(
            start_ns, generation_start_ns, input_tokens
        )
        gen_conc_ts, gen_conc = concurrency_sweep_line(generation_start_ns, end_ns)
        pre_conc_ts, pre_conc = concurrency_sweep_line(start_ns, generation_start_ns)
        total_ts, total_tput = total_throughput_sweep_line(
            start_ns,
            generation_start_ns,
            end_ns,
            input_tokens,
            output_tokens=output_tokens,
        )
        tpu_ts, tpu_vals = divide_step_functions(
            tput_ts, tput_vals, gen_conc_ts, gen_conc
        )
        ptpu_ts, ptpu_vals = divide_step_functions(
            sorted_p_ts, prefill_tput, pre_conc_ts, pre_conc
        )
        tif_ts, tif_vals = icl_aware_tokens_in_flight(
            store,
            start_ns,
            generation_start_ns,
            end_ns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return SweepLineCurves(
            concurrency_ts=sorted_c_ts,
            concurrency=concurrency,
            throughput_ts=tput_ts,
            throughput=tput_vals,
            prefill_throughput_ts=sorted_p_ts,
            prefill_throughput=prefill_tput,
            generation_concurrency_ts=gen_conc_ts,
            generation_concurrency=gen_conc,
            prefill_concurrency_ts=pre_conc_ts,
            prefill_concurrency=pre_conc,
            total_throughput_ts=total_ts,
            total_throughput=total_tput,
            throughput_per_user_ts=tpu_ts,
            throughput_per_user=tpu_vals,
            prefill_throughput_per_user_ts=ptpu_ts,
            prefill_throughput_per_user=ptpu_vals,
            tokens_in_flight_ts=tif_ts,
            tokens_in_flight=tif_vals,
        )

    @staticmethod
    def _sample_quality(
        total_requests: int, steady_state_requests: int
    ) -> dict[str, float | int | bool]:
        fraction_retained = (
            steady_state_requests / total_requests if total_requests > 0 else 0.0
        )
        variance_inflation_factor = (
            total_requests / steady_state_requests
            if steady_state_requests > 0
            else float("inf")
        )
        effective_p99_sample_size = int(steady_state_requests * 0.01)
        return {
            "fraction_retained": fraction_retained,
            "variance_inflation_factor": variance_inflation_factor,
            "effective_p99_sample_size": effective_p99_sample_size,
            "sample_size_warning": effective_p99_sample_size < 10,
        }

    @staticmethod
    def _stationarity(
        latency: np.ndarray, ss_mask: np.ndarray
    ) -> dict[str, float | bool | None]:
        windowed_latency = latency[ss_mask]
        valid_latency = windowed_latency[~np.isnan(windowed_latency)]
        if len(valid_latency) < 10:
            return {
                "trend_correlation": None,
                "trend_p_value": None,
                "stationarity_warning": False,
            }
        trend_rho, trend_p = batch_means_trend_test(valid_latency)
        return {
            "trend_correlation": trend_rho,
            "trend_p_value": trend_p,
            "stationarity_warning": abs(trend_rho) > 0.65 and trend_p < 0.05,
        }

    def _maybe_bootstrap(
        self,
        *,
        start_ns: np.ndarray,
        end_ns: np.ndarray,
        latency: np.ndarray,
        ttft: np.ndarray,
        generation_start_ns: np.ndarray,
        output_tokens: np.ndarray,
    ) -> dict[str, Any]:
        empty: dict[str, Any] = {
            "ci_ramp_up": None,
            "ci_ramp_down": None,
            "ci_mean_lat": None,
            "ci_p99_lat": None,
            "n_iterations": None,
        }
        if not self._bootstrap_iterations or self._bootstrap_iterations <= 0:
            return empty
        from aiperf.analysis.bootstrap import bootstrap_detection

        boot = bootstrap_detection(
            start_ns,
            end_ns,
            latency,
            ttft,
            n_iterations=self._bootstrap_iterations,
            min_window_pct=self._min_window_pct,
            generation_start_ns=generation_start_ns,
            output_tokens=output_tokens,
        )
        return {
            "ci_ramp_up": boot.ci_ramp_up_ns,
            "ci_ramp_down": boot.ci_ramp_down_ns,
            "ci_mean_lat": boot.ci_mean_latency,
            "ci_p99_lat": boot.ci_p99_latency,
            "n_iterations": boot.n_iterations,
        }

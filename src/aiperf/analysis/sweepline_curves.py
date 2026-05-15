# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-computed sweep-line curve container and per-window metric reduction."""

from __future__ import annotations

from dataclasses import dataclass

from aiperf.analysis.sweepline import (
    ACTIVE_VARIANT_SPECS,
    SWEEP_LINE_METRIC_SPECS,
    FloatArray,
)
from aiperf.analysis.sweepline_stats import (
    compute_active_weighted_stats,
    compute_time_weighted_stats,
    metric_result_from_sweep_line_stats,
)
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import MetricResult


@dataclass(frozen=True, slots=True)
class SweepLineCurves:
    """Pre-computed sweep-line curves for concurrency, throughput, and prefill throughput."""

    concurrency_ts: FloatArray
    concurrency: FloatArray
    throughput_ts: FloatArray
    throughput: FloatArray
    prefill_throughput_ts: FloatArray
    prefill_throughput: FloatArray
    generation_concurrency_ts: FloatArray
    generation_concurrency: FloatArray
    prefill_concurrency_ts: FloatArray
    prefill_concurrency: FloatArray
    total_throughput_ts: FloatArray
    total_throughput: FloatArray
    throughput_per_user_ts: FloatArray
    throughput_per_user: FloatArray
    prefill_throughput_per_user_ts: FloatArray
    prefill_throughput_per_user: FloatArray
    tokens_in_flight_ts: FloatArray
    tokens_in_flight: FloatArray

    def curves(
        self,
    ) -> tuple[tuple[FloatArray, FloatArray], ...]:
        """Return (ts, values) pairs in SWEEP_LINE_METRIC_SPECS order."""
        return (
            (self.concurrency_ts, self.concurrency),
            (self.throughput_ts, self.throughput),
            (self.prefill_throughput_ts, self.prefill_throughput),
            (self.generation_concurrency_ts, self.generation_concurrency),
            (self.prefill_concurrency_ts, self.prefill_concurrency),
            (self.total_throughput_ts, self.total_throughput),
            (self.throughput_per_user_ts, self.throughput_per_user),
            (self.prefill_throughput_per_user_ts, self.prefill_throughput_per_user),
            (self.tokens_in_flight_ts, self.tokens_in_flight),
        )

    def compute_metrics(
        self, window_start: float, window_end: float
    ) -> dict[str, MetricResult]:
        """Compute all sweep-line MetricResults for a time window."""
        results: dict[str, MetricResult] = {}
        for spec, (ts, values) in zip(
            SWEEP_LINE_METRIC_SPECS, self.curves(), strict=True
        ):
            stats = compute_time_weighted_stats(ts, values, window_start, window_end)
            results[spec.tag] = metric_result_from_sweep_line_stats(
                spec.tag, spec.header, spec.unit, stats, scale=spec.scale
            )

        # Active-only variants: time-weight only over segments where the
        # corresponding phase has at least one record in flight. Per-user
        # variants need this too — divide_step_functions zeros them during
        # idle gaps, biasing percentiles otherwise.
        for spec in ACTIVE_VARIANT_SPECS:
            stats = compute_active_weighted_stats(
                getattr(self, spec.rate_attr + "_ts"),
                getattr(self, spec.rate_attr),
                getattr(self, spec.mask_attr + "_ts"),
                getattr(self, spec.mask_attr),
                window_start=window_start,
                window_end=window_end,
            )
            results[spec.tag] = metric_result_from_sweep_line_stats(
                spec.tag, spec.header, spec.unit, stats, scale=NANOS_PER_SECOND
            )

        return results

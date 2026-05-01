#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-baseline per-request memory constants in
``aiperf.kubernetes._memory_estimator.constants`` against the current
on-disk model classes.

Measures pympler deep-size of the model classes the estimator references
(``RequestRecord``, ``Turn``, ``Text``, ``SSEMessage``, ``SSEField``,
``TextResponse``), linear-regresses base-plus-per-token costs, and prints
a diff vs. the constants currently in tree. Use whenever the model
classes change shape (added fields, swapped Pydantic for msgspec, etc.)
to keep the estimator's per-request formula honest.

Why this exists separately from ``calibrate_memory_estimates.py``:
``calibrate_memory_estimates.py`` validates the *aggregate* estimator
against scenario-shaped workloads. This script validates each *constant*
in isolation — far smaller surface, faster to run, gives you a per-line
update list rather than a single end-to-end ratio.

SSE per-chunk cost is measured with **unique** chunk values to defeat
Python's string interning — identical-value chunks produce a
deduplicated deep-size that under-counts real-world per-chunk RSS by
~5%.

Usage:
    uv run python -m aiperf.analysis.rebaseline_memory_constants
    uv run python -m aiperf.analysis.rebaseline_memory_constants --tolerance 0.05
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from pympler import asizeof

from aiperf.common.enums import SSEFieldType
from aiperf.common.models._server_metrics_records import MetricSample
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.record_models import (
    RequestRecord,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.telemetry_timeseries import GpuMetricTimeSeries
from aiperf.kubernetes._memory_estimator import constants as k
from aiperf.kubernetes._memory_estimator.utils import _ceil_pow2
from aiperf.server_metrics._storage_histogram import HistogramTimeSeries
from aiperf.server_metrics._storage_scalar import ScalarTimeSeries

_CHARS_PER_TOKEN = 4


# =============================================================================
# Object factories
# =============================================================================


def _make_turn(isl: int) -> Turn:
    if isl == 0:
        return Turn(role="user", texts=[])
    return Turn(role="user", texts=[Text(contents=["x" * (isl * _CHARS_PER_TOKEN)])])


def _make_sse_message_unique(osl: int) -> SSEMessage:
    """SSEMessage where each chunk's `value` is unique — defeats interning."""
    now = time.perf_counter_ns()
    return SSEMessage(
        perf_ns=now,
        packets=[
            SSEField(
                name=SSEFieldType.DATA,
                value=f'{{"choices":[{{"delta":{{"content":"tok{i:08d}"}}}}]}}',
            )
            for i in range(osl)
        ],
    )


def _make_text_response(osl: int) -> TextResponse:
    if osl == 0:
        return TextResponse(perf_ns=time.perf_counter_ns(), text="")
    body = (
        f'{{"choices":[{{"message":{{"content":"{"y" * (osl * _CHARS_PER_TOKEN)}"}}}}],'
        f'"usage":{{"prompt_tokens":512,"completion_tokens":{osl}}}}}'
    )
    return TextResponse(perf_ns=time.perf_counter_ns(), text=body)


def _make_empty_record() -> RequestRecord:
    return RequestRecord(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        timestamp_ns=time.time_ns(),
        start_perf_ns=time.perf_counter_ns(),
        end_perf_ns=time.perf_counter_ns(),
        status=200,
        responses=[],
        turns=[],
    )


def _make_record(
    isl: int, osl: int, *, streaming: bool, turns: int = 1
) -> RequestRecord:
    now_ns = time.perf_counter_ns()
    return RequestRecord(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        timestamp_ns=time.time_ns(),
        start_perf_ns=now_ns,
        end_perf_ns=now_ns + 2_000_000_000,
        recv_start_perf_ns=now_ns + 100_000_000 if streaming else None,
        status=200,
        responses=[
            _make_sse_message_unique(osl) if streaming else _make_text_response(osl)
        ],
        turns=[_make_turn(isl) for _ in range(turns)],
    )


def _make_gpu_time_series(num_metrics: int, num_snapshots: int) -> GpuMetricTimeSeries:
    """Fill a GpuMetricTimeSeries with N snapshots of M unique metrics."""
    ts = GpuMetricTimeSeries()
    metrics_template = {
        f"DCGM_FI_DEV_METRIC_{i}": float(i) * 10.0 for i in range(num_metrics)
    }
    for snap in range(num_snapshots):
        ts.append_snapshot(metrics_template, timestamp_ns=snap * 1_000_000_000)
    return ts


def _make_scalar_time_series(num_samples: int) -> ScalarTimeSeries:
    ts = ScalarTimeSeries()
    for i in range(num_samples):
        ts.append(i * 1_000_000_000, MetricSample(value=float(i)))
    return ts


def _make_histogram_time_series(
    num_samples: int, num_buckets: int
) -> HistogramTimeSeries:
    ts = HistogramTimeSeries()
    bucket_template = {
        f"{0.001 * (2**b):.6f}": float(b) for b in range(num_buckets - 1)
    }
    bucket_template["+Inf"] = float(num_buckets)
    for i in range(num_samples):
        ts.append(
            i * 1_000_000_000,
            MetricSample(buckets=dict(bucket_template), sum=float(i), count=float(i)),
        )
    return ts


# =============================================================================
# Linear regression on base + per-token cost
# =============================================================================


@dataclass
class Calibration:
    """One calibration finding: a constant + measured value + drift assessment."""

    name: str
    current: float
    measured: float
    unit: str

    @property
    def drift_pct(self) -> float:
        if self.current == 0:
            return 0.0
        return (self.measured - self.current) / self.current * 100

    def status(self, tolerance_pct: float) -> str:
        if abs(self.drift_pct) <= tolerance_pct:
            return "OK"
        return "DRIFT"

    def render(self, tolerance_pct: float) -> str:
        tag = self.status(tolerance_pct)
        sign = "+" if self.drift_pct >= 0 else ""
        return (
            f"  [{tag:<5}] {self.name:<32} current={self.current:>8.1f} {self.unit}  "
            f"measured={self.measured:>8.1f} {self.unit}  drift={sign}{self.drift_pct:>5.1f}%"
        )


def _linear_fit(
    factory, sizes: list[int], at_zero: int | None = None
) -> tuple[float, float]:
    """Fit ``size(N) = base + N * per_token`` to a small set of samples."""
    measurements = {n: asizeof.asizeof(factory(n)) for n in sizes}
    if at_zero is not None:
        measurements[0] = at_zero
    elif 0 not in measurements:
        measurements[0] = asizeof.asizeof(factory(0))
    base = measurements[0]
    largest_n = max(measurements)
    per_token = (measurements[largest_n] - base) / largest_n
    return base, per_token


# =============================================================================
# Main
# =============================================================================


def collect_calibrations() -> tuple[list[Calibration], dict[str, float]]:
    """Run measurements, return (calibrations, raw fit values)."""
    raw: dict[str, float] = {}

    # Empty RequestRecord
    raw["empty_record"] = asizeof.asizeof(_make_empty_record())

    # Turn(ISL) linear fit
    turn_base, turn_per = _linear_fit(_make_turn, [0, 64, 128, 512, 1024, 2048, 4096])
    raw["turn_base"] = turn_base
    raw["turn_per_token"] = turn_per

    # SSEMessage(OSL) linear fit — UNIQUE chunk values
    sse_base, sse_per = _linear_fit(
        _make_sse_message_unique, [0, 1, 64, 128, 512, 1024, 2048, 4096]
    )
    raw["sse_base"] = sse_base
    raw["sse_per_chunk"] = sse_per

    # TextResponse(OSL) linear fit
    text_base, text_per = _linear_fit(
        _make_text_response, [0, 64, 128, 512, 1024, 2048, 4096]
    )
    raw["text_base"] = text_base
    raw["text_per_token"] = text_per

    cals = [
        Calibration(
            "_REQUEST_RECORD_BASE_BYTES",
            k._REQUEST_RECORD_BASE_BYTES,
            raw["empty_record"],
            "B",
        ),
        Calibration("_TURN_BASE_BYTES", k._TURN_BASE_BYTES, turn_base, "B"),
        Calibration(
            "_TURN_BYTES_PER_TOKEN", k._TURN_BYTES_PER_TOKEN, turn_per, "B/tok"
        ),
        Calibration(
            "_SSE_MESSAGE_BASE_BYTES", k._SSE_MESSAGE_BASE_BYTES, sse_base, "B"
        ),
        Calibration("_SSE_BYTES_PER_CHUNK", k._SSE_BYTES_PER_CHUNK, sse_per, "B/chunk"),
        Calibration(
            "_TEXT_RESPONSE_BASE_BYTES", k._TEXT_RESPONSE_BASE_BYTES, text_base, "B"
        ),
        Calibration(
            "_TEXT_RESPONSE_BYTES_PER_TOKEN",
            k._TEXT_RESPONSE_BYTES_PER_TOKEN,
            text_per,
            "B/tok",
        ),
    ]
    return cals, raw


def validate_full_record(raw: dict[str, float]) -> None:
    """Verify the per-request formula matches measured RequestRecord."""
    print()
    print("=" * 78)
    print("  Formula validation: predicted-from-fit vs measured RequestRecord")
    print("=" * 78)

    def predict(isl: int, osl: int, streaming: bool) -> float:
        turn = raw["turn_base"] + isl * raw["turn_per_token"]
        if streaming:
            resp = raw["sse_base"] + osl * raw["sse_per_chunk"]
        else:
            resp = raw["text_base"] + osl * raw["text_per_token"]
        return raw["empty_record"] + turn + resp

    print(
        f"  {'shape':<22} {'mode':<5} {'predicted':>10} {'measured':>10} {'ratio':>6}"
    )
    print("  " + "-" * 60)
    ratios = []
    for isl, osl in [(128, 64), (512, 128), (1024, 1024), (2048, 512), (4096, 2048)]:
        for streaming in (True, False):
            rec = _make_record(isl, osl, streaming=streaming)
            meas = asizeof.asizeof(rec)
            pred = predict(isl, osl, streaming)
            ratio = pred / meas
            ratios.append(ratio)
            mode = "SSE" if streaming else "txt"
            print(
                f"  ISL={isl:<4} OSL={osl:<5}     {mode:<5} "
                f"{pred:>9.0f}B {meas:>9}B {ratio:>5.2f}x"
            )
    import math
    import statistics

    geo = math.exp(statistics.mean(math.log(r) for r in ratios))
    print(f"\n  Geo-mean ratio: {geo:.3f}x  (1.000 = exact match)")


def emit_constants_block(raw: dict[str, float]) -> None:
    """Print a copy-pasteable constants block reflecting the fresh measurements."""
    print()
    print("=" * 78)
    print("  Suggested constants block (round to clean integers as desired)")
    print("=" * 78)
    print(f"_REQUEST_RECORD_BASE_BYTES = {round(raw['empty_record'])}")
    print(f"_TURN_BASE_BYTES = {round(raw['turn_base'])}")
    print(f"_TURN_BYTES_PER_TOKEN = {round(raw['turn_per_token'])}")
    print(f"_SSE_MESSAGE_BASE_BYTES = {round(raw['sse_base'])}")
    print(f"_SSE_BYTES_PER_CHUNK = {round(raw['sse_per_chunk'])}")
    print(f"_TEXT_RESPONSE_BASE_BYTES = {round(raw['text_base'])}")
    print(f"_TEXT_RESPONSE_BYTES_PER_TOKEN = {round(raw['text_per_token'])}")


def validate_time_series(tolerance_pct: float) -> int:
    """Cross-check the GPU telemetry and Prometheus-storage formulas in
    ``components.py`` against measured deep-size of the actual storage
    classes.

    These formulas don't have direct constant counterparts in
    ``constants.py`` (they're derived from ``_INT64_BYTES``,
    ``_FLOAT64_BYTES``, ``_GROWABLE_ARRAY_OVERHEAD``), so we report the
    bytes/snapshot the measurements imply alongside what the formula
    would predict, and flag DRIFT when the ratio falls outside tolerance.

    Cells at minimum-fill capacity (``N == _INITIAL_CAPACITY`` of the
    underlying storage class) are reported as ``INFO`` rather than DRIFT
    when out of tolerance — the formula misses fixed wrapper-class
    overhead that doesn't scale with N. The absolute byte gap at minimum
    fill is small (<20 KB per series), and real estimator predictions
    are at much larger N where the formula is accurate.

    Returns the number of DRIFT findings (excludes small-N INFO findings).
    """
    print()
    print("=" * 78)
    print("  Time-series storage cross-check (GPU telemetry + Prometheus)")
    print("=" * 78)
    print()

    drift = 0

    def _classify(measured: int, predicted: float, *, small_n: bool) -> str:
        ratio = predicted / max(measured, 1)
        if abs(1 - ratio) * 100 <= tolerance_pct:
            return "OK"
        return "INFO" if small_n else "DRIFT"

    # ---- GPU telemetry ---------------------------------------------------
    # Formula in components.py:
    #   per_gpu_bytes = (capacity * INT64) + (num_metrics * capacity * FLOAT64)
    #   total = num_gpus * per_gpu_bytes * GROWABLE_ARRAY_OVERHEAD
    # Initial capacity for GpuMetricTimeSeries is 128.
    print(f"  {'shape':<32} {'measured':>10} {'predicted':>10} {'ratio':>6}")
    print("  " + "-" * 60)
    for n_metrics, n_snapshots in [(12, 256), (12, 1024), (12, 8192), (32, 1024)]:
        ts = _make_gpu_time_series(n_metrics, n_snapshots)
        measured = asizeof.asizeof(ts)
        capacity = _ceil_pow2(n_snapshots)
        per_gpu = (capacity * k._INT64_BYTES) + (
            n_metrics * capacity * k._FLOAT64_BYTES
        )
        predicted = per_gpu * k._GROWABLE_ARRAY_OVERHEAD
        ratio = predicted / max(measured, 1)
        tag = _classify(measured, predicted, small_n=n_snapshots <= 128)
        if tag == "DRIFT":
            drift += 1
        print(
            f"  [{tag:<5}] GPU M={n_metrics:<2} N={n_snapshots:<5}  "
            f"{measured:>9}B {predicted:>9.0f}B {ratio:>5.2f}x"
        )

    # ---- Server metrics: ScalarTimeSeries --------------------------------
    # Formula models scalar bytes as: count * capacity * (INT64 + FLOAT64)
    # multiplied by GROWABLE_ARRAY_OVERHEAD outside (in components.py).
    # ScalarTimeSeries _INITIAL_CAPACITY = 256.
    print()
    for n_samples in [256, 1024, 8192, 65536]:
        ts = _make_scalar_time_series(n_samples)
        measured = asizeof.asizeof(ts)
        capacity = _ceil_pow2(max(n_samples, 256))
        predicted_per_series = capacity * (k._INT64_BYTES + k._FLOAT64_BYTES)
        predicted = predicted_per_series * k._GROWABLE_ARRAY_OVERHEAD
        ratio = predicted / max(measured, 1)
        tag = _classify(measured, predicted, small_n=n_samples <= 256)
        if tag == "DRIFT":
            drift += 1
        print(
            f"  [{tag:<5}] Scalar N={n_samples:<6}            "
            f"{measured:>9}B {predicted:>9.0f}B {ratio:>5.2f}x"
        )

    # ---- Server metrics: HistogramTimeSeries -----------------------------
    # Formula: capacity * (INT64 + 2*FLOAT64 + buckets * FLOAT64) * OVERHEAD
    # HistogramTimeSeries _INITIAL_CAPACITY = 256.
    print()
    for n_samples, n_buckets in [(256, 10), (1024, 10), (1024, 20), (8192, 10)]:
        ts = _make_histogram_time_series(n_samples, n_buckets)
        measured = asizeof.asizeof(ts)
        capacity = _ceil_pow2(max(n_samples, 256))
        per_sample = (
            k._INT64_BYTES + 2 * k._FLOAT64_BYTES + n_buckets * k._FLOAT64_BYTES
        )
        predicted = capacity * per_sample * k._GROWABLE_ARRAY_OVERHEAD
        ratio = predicted / max(measured, 1)
        tag = _classify(measured, predicted, small_n=n_samples <= 256)
        if tag == "DRIFT":
            drift += 1
        print(
            f"  [{tag:<5}] Hist N={n_samples:<5} B={n_buckets:<3}      "
            f"{measured:>9}B {predicted:>9.0f}B {ratio:>5.2f}x"
        )

    print()
    print(
        "  Note: ratios > 1.0 mean the formula over-allocates (conservative);"
        " < 1.0 means under-allocation."
    )
    print(
        f"  _GROWABLE_ARRAY_OVERHEAD={k._GROWABLE_ARRAY_OVERHEAD} captures wrapper-"
        f"class overhead atop the numpy bytes (``_ceil_pow2(N)`` already handles"
        " doubling-strategy waste)."
    )
    print(
        "  INFO rows (small-N at minimum fill) are wrapper-overhead artifacts —"
        " absolute byte gap stays small at production-relevant N (≥ 1024)."
    )
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-baseline per-request memory constants against current models"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Drift threshold in %% to flag DRIFT vs OK (default: 10)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the predicted-vs-measured RequestRecord cross-check",
    )
    parser.add_argument(
        "--no-time-series",
        action="store_true",
        help=(
            "Skip the GPU-telemetry/Prometheus-storage formula cross-check "
            "(``GpuMetricTimeSeries`` / ``ScalarTimeSeries`` / ``HistogramTimeSeries``)"
        ),
    )
    args = parser.parse_args()

    print("=" * 78)
    print("  Per-request memory constants re-baseline")
    print(
        f"  Tolerance: ±{args.tolerance:.1f}%  (constants outside this band print DRIFT)"
    )
    print("=" * 78)
    print()

    cals, raw = collect_calibrations()
    drift_count = 0
    for cal in cals:
        print(cal.render(args.tolerance))
        if cal.status(args.tolerance) == "DRIFT":
            drift_count += 1

    if not args.no_validate:
        validate_full_record(raw)

    if not args.no_time_series:
        drift_count += validate_time_series(args.tolerance)

    if drift_count > 0:
        emit_constants_block(raw)
        print()
        print(
            f"!! {drift_count} constant(s) outside ±{args.tolerance:.1f}% tolerance — "
            "update ``aiperf/kubernetes/_memory_estimator/constants.py``."
        )
        return 1

    print()
    print(
        f"All {len(cals)} per-request constants and time-series formulas "
        f"within ±{args.tolerance:.1f}% tolerance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

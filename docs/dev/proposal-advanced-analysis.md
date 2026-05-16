<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Proposal: Advanced Analysis — Changepoints, Stationarity, Spectral Methods

**Phase:** P3 (Advanced Analysis)
**Depends on:** P0, P1
**Architecture:** Optional AnalyzerProtocol plugins

## Overview

This document covers three families of advanced statistical methods that would
strengthen AIPerf's analytical capabilities. Each is proposed as an optional
analyzer (AnalyzerProtocol) that can be enabled via CLI flags or plugins,
keeping the core accumulator pipeline unchanged.

---

## 1. PELT Changepoint Detection

### What It Is

PELT (Pruned Exact Linear Time) is an optimal multiple changepoint detection
algorithm (Killick et al., 2012). Unlike CUSUM which finds exactly two
boundaries (ramp-up end, ramp-down start), PELT can detect an arbitrary number
of regime changes in a time series.

### Why It Matters

Real benchmark runs often have more than two phases:
- Server warmup → steady state → scaling event → new steady state → ramp-down
- JIT compilation phase → optimized phase
- KV cache filling → KV cache eviction regime change

CUSUM + MSER-5 forces a two-boundary model. PELT would reveal the true
segmentation structure.

### Algorithm

PELT minimizes a penalized cost function over all possible segmentations:

```
Q(τ) = Σ [C(y_{τ_i+1:τ_{i+1}})] + β × m
```

Where:
- C is a segment cost (e.g., Gaussian log-likelihood: n × log(variance))
- β is a penalty per changepoint (BIC: β = log(n), or user-specified)
- m is the number of changepoints

The "pruning" insight: if adding a changepoint at t doesn't improve the cost,
no future data can make it optimal. This gives O(n) average-case complexity.

### Implementation Sketch

```python
class PELTChangepoints:
    """PELT changepoint detection for metric time series."""

    def detect(
        self,
        values: NDArray[np.float64],
        penalty: str = "bic",  # "bic", "aic", or float
        min_segment_length: int = 10,
    ) -> list[int]:
        """Return changepoint indices in the original array."""
```

**Cost function options:**
- Gaussian (change in mean): fastest, works for latency/throughput
- Poisson (change in rate): works for request counts
- Non-parametric (empirical CDF difference): most robust, slowest

### Integration

New analyzer plugin: `PELTAnalyzer` implementing AnalyzerProtocol.
- Reads ColumnStore from MetricsAccumulator (like SteadyStateAnalyzer)
- Runs PELT on concurrency sweep, latency time series, throughput sweep
- Reports: changepoint timestamps, per-segment statistics, segment labels
- Output: `PELTSummary` with per-segment MetricResults

### When to Use

- Exploratory analysis: "What happened during this run?"
- Multi-phase benchmarks: ramp → stable → scale-up → stable
- Anomaly detection: unexpected regime changes during steady state

### Effort

High — PELT is a non-trivial algorithm. The core dynamic programming loop is
~100 lines, but robust penalty selection and multi-metric correlation add
complexity. Consider using the `ruptures` Python package as a reference
implementation.

---

## 2. Formal Stationarity Testing (ADF / KPSS)

### What We Have Now

`batch_means_trend_test()` in `stationarity.py` detects linear trends via
Spearman rank correlation of k=10 batch means. This catches monotonic trends
but misses:
- Unit roots (random walk behavior)
- Trend-stationarity (deterministic trend component)
- Cyclical non-stationarity (periodic level shifts)

### What the Literature Recommends

**Augmented Dickey-Fuller (ADF) test:**
- Tests H₀: the series has a unit root (non-stationary)
- Rejection → evidence of stationarity
- Industry standard in econometrics and simulation analysis

**KPSS test (Kwiatkowski-Phillips-Schmidt-Shin):**
- Tests H₀: the series IS stationary
- Rejection → evidence of non-stationarity
- Complementary to ADF (different null hypothesis)

**Combined interpretation:**

| ADF rejects | KPSS doesn't reject | → Stationary |
|---|---|---|
| ADF doesn't reject | KPSS rejects | → Non-stationary |
| Both reject | | → Trend-stationary (remove trend) |
| Neither rejects | | → Inconclusive |

**Heidelberger-Welch test:**
- Specifically designed for simulation output analysis
- Phase 1: tests stationarity using Cramér-von Mises statistic on
  spectral density at frequency zero
- Phase 2: iteratively truncates the initial transient
- Most appropriate test for our use case (simulation-like benchmark runs)

### Implementation Options

**Option A — Pure numpy (preferred for zero-dependency):**
- ADF: requires computing lag-augmented OLS regression coefficients.
  This is ~50 lines of numpy (X'X inverse × X'y with lagged differences).
  Critical values from published tables (MacKinnon, 1996).
- KPSS: similar complexity. Requires long-run variance estimation via
  Bartlett kernel.
- Heidelberger-Welch: requires spectral density estimation at f=0 via
  Bartlett method. Moderate complexity.

**Option B — Optional scipy dependency:**
- `scipy.stats` has `adfuller` (via statsmodels) but not directly
- `statsmodels.tsa.stattools.adfuller` and `kpss` are available
- Adds a heavy dependency for a small feature

**Recommendation:** Pure numpy for ADF (most common test), with an optional
`statsmodels` path for KPSS and Heidelberger-Welch when available.

### Integration

Extend `SteadyStateWindowMetadata` with:

```python
adf_statistic: float | None
adf_p_value: float | None
adf_is_stationary: bool | None  # True if H0 (unit root) rejected at 5%

kpss_statistic: float | None
kpss_p_value: float | None
kpss_is_stationary: bool | None  # True if H0 (stationary) NOT rejected at 5%

stationarity_classification: str | None  # "stationary", "non_stationary",
                                          # "trend_stationary", "inconclusive"
```

### Effort

Medium for ADF (pure numpy, ~100 lines + critical value tables).
Medium-High for KPSS + Heidelberger-Welch.

---

## 3. Spectral / Periodicity Analysis

### What It Is

Welch's method estimates the power spectral density (PSD) of a time series,
revealing which frequencies dominate the signal. Applied to latency or
throughput time series, it can detect periodic patterns.

### Why It Matters

Common periodic patterns in LLM benchmark data:
- **GPU boost clock cycling** (~1-10 second period): thermal management causes
  periodic throughput oscillation
- **Batch scheduling** (~10-100ms period): the server's internal batching
  creates a sawtooth latency pattern
- **Garbage collection** (~1-60 second period): Python/Java GC pauses in the
  inference server create periodic spikes
- **Kubernetes HPA scaling** (~60-300 second period): pod scaling creates
  step-change + periodic patterns

Detecting these helps users diagnose *why* their latency variance is high and
whether it's an artifact of the measurement or a real server behavior.

### Algorithm

Welch's method:
1. Divide the time series into overlapping segments (50% overlap)
2. Apply a Hanning window to each segment
3. Compute FFT of each windowed segment
4. Average the periodograms

This produces a smoother PSD estimate than a single FFT.

### Implementation

```python
def welch_psd(
    values: NDArray[np.float64],
    sample_rate_hz: float,
    nperseg: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute power spectral density using Welch's method.

    Pure numpy implementation (no scipy.signal dependency).

    Returns:
        (frequencies_hz, power_spectral_density)
    """
```

The core is ~30 lines of numpy (windowing + FFT + averaging). The challenge is
determining sample_rate_hz: latency observations are not equally spaced in
time. Two approaches:
- Resample to a regular grid using linear interpolation
- Use Lomb-Scargle periodogram (designed for unequally spaced data, but more
  complex)

**Recommendation:** Resample to regular grid. The interpolation error is
negligible for the frequency ranges we care about (0.01 Hz to 100 Hz).

### Output

```python
@dataclass
class PeriodicityAnalysis:
    """Results from spectral analysis of a metric time series."""
    dominant_frequency_hz: float | None
    dominant_period_seconds: float | None
    dominant_amplitude: float | None
    is_periodic: bool  # True if dominant peak > 3× median PSD
    spectrum_frequencies_hz: list[float]
    spectrum_power: list[float]
```

### Integration

- Optional analyzer plugin: `SpectralAnalyzer` implementing AnalyzerProtocol
- Runs on request_latency, TTFT, and effective_concurrency time series
- Results included in JSON export under a `spectral_analysis` key
- Console: only print if periodicity detected ("Periodic pattern detected:
  ~2.3s period in request_latency")

### Effort

Medium — Welch's method is well-defined. The main work is resampling and
determining appropriate segment lengths. ~150 lines of numpy.

---

## 4. Multi-Run Aggregation

### What It Is

Running the same benchmark N times and computing statistics across runs:
mean, std, min, max, coefficient of variation (CV) for each metric.

### Why It Matters

- MLPerf requires multiple runs for submission
- Single-run results are subject to run-to-run variance from thermal state,
  OS scheduling, network conditions, and server-side non-determinism
- CV < 5% for throughput metrics indicates stable measurement; CV > 10%
  suggests environmental instability

### Implementation

This is NOT an accumulator feature — it's a CLI post-processing tool:

```bash
aiperf aggregate \
    --runs results/run1/profile_export.json \
           results/run2/profile_export.json \
           results/run3/profile_export.json \
    --output results/aggregate_summary.json
```

**Algorithm:**
1. Load per-run MetricResults from JSON exports
2. For each metric tag, collect the per-run avg/p50/p99/etc. values
3. Compute across-run statistics (mean, std, min, max, CV)
4. Flag metrics with CV > 10% as "high variance"
5. Optionally run modified Z-score outlier detection on per-run values

**Output:**

```json
{
  "n_runs": 5,
  "metrics": {
    "request_latency": {
      "avg": {"mean": 142.3, "std": 2.1, "cv_pct": 1.5, "runs": [140.1, 142.8, ...]},
      "p99": {"mean": 312.5, "std": 8.7, "cv_pct": 2.8, "runs": [305.2, 318.1, ...]},
    }
  },
  "warnings": [
    "time_to_first_token.p99 has CV=12.3% across runs (consider more runs or longer duration)"
  ]
}
```

### Effort

Medium — needs a new CLI command and result file parsing. The statistical
computation is trivial.

---

## 5. Per-GPU Normalization

### What It Is

Throughput metrics normalized by the number of GPUs serving the model.

### Proposed Metrics

- `throughput_per_gpu` = output_token_throughput / num_gpus
- `prefill_throughput_per_gpu` = prefill throughput / num_gpus

### Implementation

GPU count is available from GPU telemetry (number of unique GPU devices
discovered). When GPU telemetry is disabled, these metrics are None.

**Challenge:** Multi-node deployments may have GPUs across nodes. The GPU
telemetry collects per-endpoint, per-hostname, per-device data, so the
count is reliable for discovered endpoints.

### Effort

Low — a DERIVED metric that reads GPU count from telemetry accumulator output
via SummaryContext.

---

## Summary & Prioritization

| Feature | Complexity | Dependencies | Value |
|---|---|---|---|
| PELT changepoints | High | None (standalone) | Multi-regime analysis |
| ADF stationarity | Medium | None (numpy-only) | Rigorous validation |
| KPSS stationarity | Medium-High | Optional (statsmodels) | Combined with ADF |
| Welch spectral analysis | Medium | None (numpy FFT) | Diagnostic |
| Multi-run aggregation | Medium | None (CLI tool) | MLPerf parity |
| Per-GPU normalization | Low | GPU telemetry | Cross-deployment comparison |

All of these are proposed as optional/pluggable features that don't affect the
core accumulator pipeline. They activate when the user opts in or when the
relevant data is available.

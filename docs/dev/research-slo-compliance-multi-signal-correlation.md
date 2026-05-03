<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SLO Compliance & Multi-Signal Correlation for Capacity Planning

**Status**: Research

**Authors**: Anthony Casagrande

**Category**: Research — Correlation Analysis

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [SLO Definition for LLM Inference](#1-slo-definition-for-llm-inference)
3. [SLO Compliance Surface](#2-slo-compliance-surface)
4. [Leading Indicator Correlation](#3-leading-indicator-correlation)
5. [Capacity Planning via Correlation](#4-capacity-planning-via-correlation)
6. [Multi-Signal Anomaly Detection](#5-multi-signal-anomaly-detection)
7. [Degradation Waterfall](#6-degradation-waterfall)
8. [Benchmark Validity Assessment](#7-benchmark-validity-assessment)
9. [Comparative Analysis](#8-comparative-analysis)
10. [AIPerf Implementation Roadmap](#aiperf-implementation-roadmap)
11. [References](#references)

---

## Executive Summary

LLM inference benchmarking produces three distinct signal families — client-side
latency metrics, server-side operational metrics, and GPU hardware telemetry —
yet they are typically analyzed in isolation. This document researches how to
unify these signals into a coherent framework for **SLO compliance analysis**,
**capacity planning**, and **automated root cause identification**.

The core thesis: a single metric (e.g., P99 TTFT) tells you *what happened*. The
correlation structure across all three signal families tells you *why it happened*
and *when it will happen again*. By mapping compliance surfaces, ranking leading
indicators, and tracing degradation waterfalls, AIPerf can move from "here are
your benchmark numbers" to "here is your safe operating envelope and here is what
will break first."

**Key outcomes of this research:**

- Formal SLO definition framework for composite LLM inference objectives
- SLO compliance surface algorithm using AIPerf's existing sweep metrics
- Ranked leading indicator table with cross-correlation methodology
- Capacity planning regression model: P99_TTFT = f(concurrency, kv_cache, gpu_util)
- Multi-signal anomaly scoring via PCA on the correlation matrix
- Automated degradation waterfall tracing through the causal chain
- Benchmark validity scoring using Little's Law and metric reconciliation
- Comparative meta-analysis framework for cross-configuration insights

---

## 1. SLO Definition for LLM Inference

### 1.1 What Makes LLM SLOs Different

Traditional web service SLOs are simple: P99 latency < 200ms, availability >
99.9%. LLM inference introduces three complications:

1. **Streaming decomposition.** A single request has two distinct latency
   components — time-to-first-token (TTFT, the prefill phase) and inter-token
   latency (ITL, the decode phase). Users perceive both independently: TTFT
   determines "how long before I see anything" and ITL determines "how smooth is
   the typing experience."

2. **Throughput is per-request, not per-service.** Each request generates a
   variable number of tokens. A service handling 10 req/s at 100 tokens/req
   produces the same throughput as 100 req/s at 10 tokens/req, but the user
   experience is radically different.

3. **Resource contention is structural.** The KV cache is a finite, shared
   resource whose pressure directly causes preemption and recomputation. Unlike
   CPU-bound services where load increases latency smoothly, KV cache exhaustion
   creates cliff effects.

### 1.2 Atomic SLO Definitions

An atomic SLO constrains a single metric at a single percentile:

```
SLO_atomic := (metric, percentile, comparator, threshold, unit)
```

**Common atomic SLOs for LLM inference:**

| SLO Name | Metric | Percentile | Comparator | Threshold | Unit |
|----------|--------|-----------|------------|-----------|------|
| Fast first token | time_to_first_token | P99 | < | 500 | ms |
| Smooth streaming | inter_token_latency | P99 | < | 50 | ms |
| Bounded e2e | request_latency | P99 | < | 5000 | ms |
| Minimum throughput | output_token_throughput_per_user | P50 | >= | 30 | tokens/s |
| Prefill throughput | effective_prefill_throughput | mean | >= | 1000 | tokens/s |

**Formal definition.** Given an ordered sample X_{(1)} <= X_{(2)} <= ... <= X_{(n)}
of metric values from a benchmark run:

```
SLO_met(metric, p, <, T) := X_{(ceil(n * p/100))} < T

SLO_met(metric, p, >=, T) := X_{(floor(n * p/100))} >= T
```

Where p is the percentile (e.g., 99) and T is the threshold. The SLO is either
met or not met for the entire run.

### 1.3 Composite SLO Definitions

Real deployments require multiple SLOs to hold simultaneously. A composite SLO
combines atomic SLOs with logical operators:

```
SLO_composite := SLO_1 AND SLO_2 AND ... AND SLO_k
```

**Example: Production LLM deployment SLO**

```
SLO_production :=
    P99(TTFT) < 500ms
    AND P99(ITL) < 50ms
    AND P99(request_latency) < 10s
    AND mean(output_token_throughput_per_user) >= 30 tokens/s
```

This is a conjunction — all conditions must hold. In practice, disjunctive SLOs
("either fast TTFT OR high throughput") are rare because they allow degenerate
solutions.

**Per-request compliance.** Beyond aggregate percentile SLOs, we can define
per-request compliance:

```
compliant(request_i) := ALL(metric_j(request_i) meets SLO_j for j in 1..k)
```

A request is compliant only if ALL its metrics meet ALL atomic thresholds. This
is the definition used by the DistServe goodput metric, and it is what AIPerf's
`--goodput` flag already implements.

### 1.4 SLO Compliance Rate

The SLO compliance rate is the fraction of requests that are fully compliant:

```
compliance_rate = count(compliant requests) / count(total requests)
```

The distinction from percentile-based SLOs is important:

- **Percentile SLO:** "P99 TTFT < 500ms" — 99% of requests have TTFT < 500ms
- **Compliance rate:** "99% of requests meet ALL SLOs simultaneously"

These are NOT equivalent. If TTFT and ITL violations are correlated (they
usually are — both worsen under load), the compliance rate can be significantly
lower than the least-met individual SLO.

**Example:** Suppose P99 TTFT < 500ms is met (99% compliant on TTFT alone) and
P99 ITL < 50ms is met (99% compliant on ITL alone). If violations are perfectly
correlated (the same 1% of requests violate both), composite compliance = 99%.
If violations are independent, composite compliance = 98.01%. If violations are
anti-correlated, composite compliance could be as low as 98%.

### 1.5 Relationship to AIPerf's Existing Goodput

AIPerf already implements per-request SLO compliance via `--goodput`:

```bash
aiperf benchmark \
    --goodput "time_to_first_token:500 inter_token_latency:50"
```

This computes goodput = (count of requests where ALL specified metrics meet
their thresholds) / benchmark_duration. The goodput metric is exactly the SLO
compliance rate multiplied by the request throughput:

```
goodput = compliance_rate * request_throughput
```

The research in this document extends this foundation in three directions:

1. **Compliance surface** — map goodput across the (concurrency, throughput) space
2. **Leading indicators** — predict compliance violations before they happen
3. **Capacity planning** — find the maximum concurrency that sustains target compliance

---

## 2. SLO Compliance Surface

### 2.1 Concept

The SLO compliance surface is a 2D map showing compliance rate as a function of
the operating point. The natural axes for LLM inference are:

- **x-axis:** Effective concurrency (from `effective_concurrency` sweep metric)
- **y-axis:** Effective throughput (from `effective_throughput` sweep metric)
- **z-axis (color):** SLO compliance rate at that operating point

This surface reveals the **safe operating envelope** — the region of the
(concurrency, throughput) space where compliance exceeds the target (e.g., 99%).

### 2.2 Construction Algorithm

AIPerf's sweep-line infrastructure already produces time-weighted concurrency and
throughput curves. The compliance surface algorithm overlays per-request SLO
compliance on this temporal grid.

**Step 1: Temporal binning.**

Divide the benchmark timeline into N bins of equal duration delta_t. For each
bin b:

```
bin_start(b) = run_start + b * delta_t
bin_end(b) = run_start + (b + 1) * delta_t
```

**Step 2: Per-bin operating point.**

For each bin, compute the time-weighted average concurrency and throughput from
the sweep curves:

```python
def bin_operating_point(
    concurrency_ts: NDArray[np.float64],
    concurrency: NDArray[np.float64],
    throughput_ts: NDArray[np.float64],
    throughput: NDArray[np.float64],
    bin_start_ns: float,
    bin_end_ns: float,
) -> tuple[float, float]:
    """Compute (avg_concurrency, avg_throughput) for a time bin."""
    c_mask = (concurrency_ts >= bin_start_ns) & (concurrency_ts < bin_end_ns)
    t_mask = (throughput_ts >= bin_start_ns) & (throughput_ts < bin_end_ns)
    avg_c = time_weighted_mean(concurrency_ts[c_mask], concurrency[c_mask], bin_end_ns)
    avg_t = time_weighted_mean(throughput_ts[t_mask], throughput[t_mask], bin_end_ns)
    return avg_c, avg_t
```

**Step 3: Per-bin compliance.**

For each bin, identify all requests that completed within the bin (or started
within the bin — choice depends on the attribution model). Compute the fraction
of those requests meeting the composite SLO:

```python
def bin_compliance(
    store: ColumnStore,
    slo_thresholds: dict[str, tuple[str, float]],  # tag -> (comparator, threshold)
    bin_start_ns: float,
    bin_end_ns: float,
) -> float:
    """Fraction of requests completing in [bin_start, bin_end) that meet all SLOs."""
    mask = (store.end_ns >= bin_start_ns) & (store.end_ns < bin_end_ns)
    mask &= ~np.isnan(store.end_ns)
    n_requests = int(np.sum(mask))
    if n_requests == 0:
        return float("nan")  # No data in this bin

    compliant = np.ones(n_requests, dtype=bool)
    for tag, (comparator, threshold) in slo_thresholds.items():
        values = store.numeric(tag)[mask]
        if comparator == "<":
            compliant &= values < threshold
        elif comparator == ">=":
            compliant &= values >= threshold

    return float(np.sum(compliant)) / n_requests
```

**Step 4: Grid assembly.**

Collect (concurrency, throughput, compliance) triples for all bins. This is the
raw compliance surface. For visualization, interpolate onto a regular 2D grid
using nearest-neighbor or bilinear interpolation.

### 2.3 Identifying the Safe Operating Envelope

The safe operating envelope is the contour at `compliance = target`:

```
envelope(target) = {(c, t) : compliance(c, t) >= target}
```

For capacity planning, the key quantity is the **maximum concurrency on the
envelope boundary**:

```
max_safe_concurrency(target) = max{c : exists t such that compliance(c, t) >= target}
```

This can be extracted directly from the compliance surface by finding the
rightmost point on the target contour.

### 2.4 Bin Size Selection

The bin size delta_t controls the granularity of the surface:

- **Too small** (< 1s): Many bins have zero or very few requests, producing a
  noisy surface. For a service handling 50 req/s, a 100ms bin contains ~5
  requests — insufficient for reliable compliance estimation.
- **Too large** (> 30s): Operating point variation within a bin is large,
  smearing the surface. The concurrency ramp-up covers many bins worth of
  operating points.
- **Recommended:** delta_t = max(1s, 10 / request_rate). This ensures each bin
  contains at least ~10 requests for a meaningful compliance fraction.

### 2.5 Temporal Attribution Model

A subtle choice: when a request spans two bins (starts in bin b, ends in bin
b+1), which bin does it count toward?

- **Completion attribution** (recommended): Assign to the bin where the request
  completed. This is simple, matches how results are typically aggregated, and
  avoids double-counting.
- **Start attribution:** Assign to the bin where the request started. Better for
  correlating with the operating point at the time of submission.
- **Proportional attribution:** Split the request across bins proportional to the
  time spent in each. Most accurate but significantly more complex.

For AIPerf, completion attribution is the natural choice because the ColumnStore
stores `end_ns` per request and metric values (TTFT, ITL, latency) are finalized
at completion.

### 2.6 Compliance Surface Visualization

The compliance surface maps naturally to several visualization types:

**Heatmap:** 2D grid with concurrency on x-axis, throughput on y-axis, compliance
as color intensity. The safe envelope is the boundary between green (compliant)
and red (non-compliant) regions.

**Contour plot:** Iso-compliance lines at 99.9%, 99%, 95%, 90%. The spacing
between contours reveals how sharply compliance degrades — tight spacing means a
cliff, wide spacing means a gradual degradation.

**1D projection:** For a fixed throughput (or letting throughput vary naturally),
plot compliance vs. concurrency. This is the simplest view and directly answers
"at what concurrency does SLO compliance drop below my target?"

---

## 3. Leading Indicator Correlation

### 3.1 The Prediction Problem

SLO violations are reactive measurements — by the time P99 TTFT exceeds 500ms,
the damage is done. For capacity planning and autoscaling, we need **leading
indicators**: server-side or GPU metrics that predict SLO violations *before*
they manifest in client-side latency.

The formal question: given server metric S(t) and client metric C(t), what is
the cross-correlation at time lag tau?

```
rho(tau) = corr(S(t), C(t + tau))
```

A positive rho at positive tau means S leads C: when S increases at time t,
C increases at time t + tau. The lag tau with maximum |rho| is the **predictive
lead time** — how far in advance the server metric warns of the client-side
violation.

### 3.2 Available Signal Inventory

AIPerf collects three signal families, each with different temporal resolution
and information content:

**Client-side metrics (per-request, ~10-100ms resolution):**

| Metric | Tag | What It Measures |
|--------|-----|-----------------|
| Time to first token | `time_to_first_token` | Prefill latency + network |
| Inter-token latency | `inter_token_latency` | Decode latency per token |
| Request latency | `request_latency` | End-to-end request duration |
| Effective concurrency | `effective_concurrency` | Instantaneous active requests (sweep) |
| Effective throughput | `effective_throughput` | Instantaneous output token rate (sweep) |
| Request throughput | `request_throughput` | Requests completed per second |
| Goodput | `goodput` | SLO-compliant requests per second |
| Generation concurrency | `effective_generation_concurrency` | Active decode-phase requests (sweep) |
| Prefill concurrency | `effective_prefill_concurrency` | Active prefill-phase requests (sweep) |
| Tokens in flight | `tokens_in_flight` | Total tokens being processed (sweep) |
| Throughput per user | `effective_throughput_per_user` | Per-user output token rate (sweep) |

**Server-side metrics (polled, ~1-15s resolution):**

| Metric | What It Measures | Leading? |
|--------|-----------------|----------|
| `num_requests_running` | Batch size on the engine | Strong |
| `num_requests_waiting` | Queue depth | Very strong |
| `kv_cache_usage_perc` | KV cache memory pressure | Very strong |
| `request_queue_time_seconds` (histogram) | Server-side queue wait distribution | Strong |
| Latency histograms (TTFT, ITL, e2e) | Server-observed latency distribution | Coincident |

**GPU telemetry (polled, ~1-5s resolution via DCGM):**

| Metric | What It Measures | Leading? |
|--------|-----------------|----------|
| `gpu_utilization` | SM activity percentage | Moderate |
| `sm_utilization` | Streaming multiprocessor utilization | Moderate |
| `mem_utilization` | Memory controller utilization | Moderate |
| `gpu_power_usage` | Power draw in watts | Weak |
| `gpu_temperature` | Junction temperature | Very weak (thermal inertia) |
| `gpu_memory_used` | VRAM usage in bytes | Moderate |
| `energy_consumption` | Cumulative energy | Very weak (cumulative) |

### 3.3 Cross-Correlation Methodology

**Resampling.** The three signal families have different temporal resolutions.
Before computing cross-correlation, resample all signals to a common time base:

```python
def resample_to_common_grid(
    signals: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],  # tag -> (timestamps, values)
    grid_interval_ns: float,
    method: str = "linear",
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Resample all signals to a common regular time grid.

    Uses linear interpolation for continuous signals (GPU metrics)
    and sample-and-hold for discrete signals (queue depth, request counts).
    """
    all_ts = np.concatenate([ts for ts, _ in signals.values()])
    grid_start = np.nanmin(all_ts)
    grid_end = np.nanmax(all_ts)
    common_ts = np.arange(grid_start, grid_end, grid_interval_ns)

    resampled: dict[str, NDArray[np.float64]] = {}
    for tag, (ts, vals) in signals.items():
        resampled[tag] = np.interp(common_ts, ts, vals)

    return common_ts, resampled
```

**Normalized cross-correlation.** For signals x(t) and y(t) resampled to the
same grid:

```python
def normalized_cross_correlation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_lag: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Compute normalized cross-correlation for lags in [-max_lag, +max_lag].

    Returns (lags, correlations) where correlations[i] = corr(x[t], y[t + lags[i]]).
    Positive lag means x leads y.
    """
    n = len(x)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return np.arange(-max_lag, max_lag + 1), np.zeros(2 * max_lag + 1)

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.empty(len(lags), dtype=np.float64)
    for i, lag in enumerate(lags):
        if lag >= 0:
            correlations[i] = np.dot(x_centered[:n - lag], y_centered[lag:]) / (n * sx * sy)
        else:
            correlations[i] = np.dot(x_centered[-lag:], y_centered[:n + lag]) / (n * sx * sy)

    return lags, correlations
```

**Optimal lag extraction.** For each (server_metric, client_metric) pair, find
the lag with maximum absolute correlation:

```python
def optimal_lag(
    lags: NDArray[np.int64],
    correlations: NDArray[np.float64],
) -> tuple[int, float]:
    """Return (best_lag, max_correlation) from a cross-correlation result."""
    idx = np.argmax(np.abs(correlations))
    return int(lags[idx]), float(correlations[idx])
```

### 3.4 Expected Leading Indicator Ranking

Based on the causal structure of LLM inference servers (primarily vLLM and
TensorRT-LLM), the expected ranking of leading indicators for P99 TTFT
violations is:

| Rank | Server/GPU Metric | Expected Lead Time | Mechanism |
|------|------------------|-------------------|-----------|
| 1 | `num_requests_waiting` (queue depth) | 2-10s | Direct cause: queued requests wait longer before prefill starts |
| 2 | `kv_cache_usage_perc` | 5-20s | KV cache pressure triggers preemption → recomputation → higher TTFT |
| 3 | `num_requests_running` (batch size) | 1-5s | Larger batches → more prefill competition → longer TTFT |
| 4 | `gpu_utilization` | 0-3s | High utilization → less spare capacity for new prefills |
| 5 | `mem_utilization` | 0-5s | Memory bandwidth contention → slower attention computation |
| 6 | `sm_utilization` | 0-3s | Compute saturation → prefill takes longer |
| 7 | `gpu_power_usage` | 0-2s | Near TDP → potential frequency throttling |
| 8 | `gpu_temperature` | 30-120s | Thermal throttling is real but slow-acting |

For P99 ITL violations, the ranking shifts:

| Rank | Server/GPU Metric | Expected Lead Time | Mechanism |
|------|------------------|-------------------|-----------|
| 1 | `kv_cache_usage_perc` | 3-15s | KV cache pressure → preemption during decode → recompute |
| 2 | `num_requests_running` (batch size) | 1-5s | Batch size directly determines decode iteration time |
| 3 | `num_requests_waiting` | 2-10s | Queue pressure → aggressive batching → larger batches |
| 4 | `gpu_utilization` | 0-3s | Decode is compute-bound per iteration |
| 5 | `sm_utilization` | 0-3s | SM contention in multi-query attention |

### 3.5 Correlation Significance Testing

With N temporal bins, the null hypothesis (no correlation) has a standard error
of 1/sqrt(N) for the Pearson correlation coefficient. A correlation is
significant at the alpha level if:

```
|rho| > z_{alpha/2} / sqrt(N)
```

For N = 100 bins and alpha = 0.05: significance threshold = 0.196.
For N = 300 bins and alpha = 0.01: significance threshold = 0.149.

**Multiple comparison correction.** When testing all (server_metric,
client_metric, lag) combinations, the Bonferroni correction adjusts alpha:

```
alpha_corrected = alpha / (n_server_metrics * n_client_metrics * n_lags)
```

For 8 server metrics * 5 client metrics * 20 lags = 800 tests:
alpha_corrected = 0.05 / 800 = 6.25e-5, requiring |rho| > 0.40 at N = 100.

A less conservative alternative is the Benjamini-Hochberg procedure (False
Discovery Rate control), which is more appropriate when we expect many true
correlations:

```
1. Sort p-values: p_{(1)} <= p_{(2)} <= ... <= p_{(m)}
2. Find largest k such that p_{(k)} <= k/m * alpha
3. Reject all hypotheses with p_{(i)} <= p_{(k)}
```

### 3.6 Practical Considerations

**Temporal resolution mismatch.** Server metrics are polled every 1-15 seconds.
Client metrics arrive per-request (potentially 100+ per second). The
cross-correlation must account for this:

- Bin client metrics to match server polling interval
- Use the median (not mean) of client metrics per bin to reduce sensitivity to
  outliers
- For sweep metrics (effective_concurrency, etc.), use the time-weighted value
  over the bin, which is already what the sweep-line algorithm produces

**Non-stationarity.** During ramp-up and ramp-down, all metrics are trending.
Trend correlation is spurious — two metrics can appear correlated simply because
both increase during ramp-up. Solutions:

- Restrict correlation analysis to the steady-state window (use AIPerf's
  existing CUSUM + MSER-5 detection)
- Detrend signals before correlating: x'(t) = x(t) - trend(x, t)
- Use first-differences: delta_x(t) = x(t) - x(t-1), which removes trends

**Recommendation for AIPerf:** Compute correlations only within the steady-state
window. If no steady-state is detected, detrend using a rolling mean with window
size = 10% of run duration.

---

## 4. Capacity Planning via Correlation

### 4.1 The Capacity Question

The fundamental capacity planning question is:

> Given SLO constraints and a target compliance rate, what is the maximum
> concurrency (load) this server configuration can sustain?

This is an optimization problem:

```
maximize    concurrency
subject to  compliance_rate(concurrency) >= target
            SLO_1, SLO_2, ..., SLO_k are defined
```

### 4.2 Empirical Approach: Sweep Curve Intersection

The simplest approach uses the compliance surface from Section 2. Project the
surface onto the concurrency axis (marginalizing over throughput):

```
compliance(c) = mean compliance over all bins with effective_concurrency in [c - delta, c + delta]
```

Then find the intersection:

```
max_safe_concurrency = max{c : compliance(c) >= target}
```

This is a lookup, not a model — it only works within the range of concurrencies
actually observed in the benchmark. If the benchmark ran at concurrency 1-64,
we cannot extrapolate to concurrency 128.

**Algorithm:**

```python
def max_safe_concurrency_empirical(
    concurrency_bins: NDArray[np.float64],
    compliance_bins: NDArray[np.float64],
    target: float = 0.99,
) -> float | None:
    """Find maximum concurrency where compliance >= target.

    Args:
        concurrency_bins: Per-bin average concurrency values.
        compliance_bins: Per-bin SLO compliance rates.
        target: Minimum acceptable compliance rate.

    Returns:
        Maximum safe concurrency, or None if never met.
    """
    compliant = compliance_bins >= target
    if not np.any(compliant):
        return None
    return float(np.max(concurrency_bins[compliant]))
```

### 4.3 Regression Approach: Predictive Model

To extrapolate beyond observed operating points, build a regression model
relating client-side SLO metrics to server-side operating conditions.

**Model family: P99 TTFT as a function of operating point.**

The dependent variable is P99 TTFT (computed per time bin). The independent
variables are:

- Effective concurrency (c)
- KV cache usage percentage (kv)
- GPU utilization (gpu)
- Queue depth (qd)

**Linear model (baseline):**

```
P99_TTFT = beta_0 + beta_1 * c + beta_2 * kv + beta_3 * gpu + beta_4 * qd + epsilon
```

This is interpretable but unlikely to capture the non-linear cliff effects
typical of LLM inference (e.g., KV cache hitting 95% causes preemption).

**Log-linear model (recommended):**

```
log(P99_TTFT) = beta_0 + beta_1 * c + beta_2 * kv + beta_3 * kv^2 + beta_4 * gpu + beta_5 * qd + epsilon
```

The log transform stabilizes variance (latency distributions are right-skewed),
and the kv^2 term captures the non-linear KV cache cliff.

**Piecewise model (most accurate):**

```
P99_TTFT = {
    beta_0 + beta_1 * c + beta_2 * kv           if kv < kv_threshold
    gamma_0 + gamma_1 * c + gamma_2 * (kv - kv_threshold)^2   if kv >= kv_threshold
}
```

The KV cache threshold (typically 85-95%) marks the regime change where
preemption begins. This threshold can be estimated from the data by fitting a
segmented regression and selecting the breakpoint that minimizes total residual
variance.

### 4.4 Solving for Maximum Concurrency

Given the regression model, solve for the maximum concurrency that keeps P99
TTFT below the SLO threshold:

**For the linear model:**

```
P99_TTFT_target = beta_0 + beta_1 * c_max + beta_2 * kv(c_max) + beta_3 * gpu(c_max) + beta_4 * qd(c_max)
```

This requires knowing how kv, gpu, and qd depend on concurrency. If we have a
separate regression for each:

```
kv(c) = alpha_kv_0 + alpha_kv_1 * c
gpu(c) = alpha_gpu_0 + alpha_gpu_1 * c
qd(c) = alpha_qd_0 + alpha_qd_1 * c
```

Then substitution gives a single equation in c_max:

```
c_max = (P99_TTFT_target - beta_0 - beta_2 * alpha_kv_0 - beta_3 * alpha_gpu_0 - beta_4 * alpha_qd_0) /
        (beta_1 + beta_2 * alpha_kv_1 + beta_3 * alpha_gpu_1 + beta_4 * alpha_qd_1)
```

**For the log-linear model:** Solve numerically using bisection or Newton's
method on the log-transformed equation.

### 4.5 Confidence Interval on Capacity Estimate

The capacity estimate inherits uncertainty from the regression coefficients.
Using the delta method:

```
Var(c_max) ≈ (partial c_max / partial beta)^T * Cov(beta) * (partial c_max / partial beta)
```

Where Cov(beta) is the covariance matrix of the regression coefficients (readily
available from OLS). This gives a 95% CI on c_max.

For production use, a bootstrap approach is more robust:

```python
def bootstrap_max_concurrency(
    features: NDArray[np.float64],   # (n_bins, n_features)
    targets: NDArray[np.float64],    # (n_bins,) P99 TTFT per bin
    slo_threshold: float,
    n_iterations: int = 1000,
) -> tuple[float, float, float]:
    """Bootstrap CI for maximum safe concurrency.

    Returns (estimate, ci_lower, ci_upper) at 95% confidence.
    """
    n = len(targets)
    estimates = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        beta = np.linalg.lstsq(features[idx], targets[idx], rcond=None)[0]
        # Solve for c_max given beta and slo_threshold
        estimates[i] = solve_for_max_concurrency(beta, slo_threshold)

    return float(np.median(estimates)), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))
```

### 4.6 Multi-SLO Capacity Planning

When the composite SLO includes multiple metrics (TTFT AND ITL AND throughput),
the capacity limit is the minimum across individual limits:

```
c_max_composite = min(c_max_ttft, c_max_itl, c_max_throughput)
```

Each individual c_max comes from its own regression model. The binding
constraint (the one producing the smallest c_max) is the **bottleneck SLO** —
the SLO that fails first as load increases. Identifying the bottleneck SLO is
actionable: it tells operators which aspect of the system to optimize.

**Typical bottleneck hierarchy:**

1. **TTFT** is usually the first to fail at high concurrency, because prefill
   is compute-intensive and gets deferred when the batch is full.
2. **ITL** fails next if KV cache preemption occurs during decode.
3. **Throughput** SLOs fail last because throughput scales nearly linearly with
   concurrency until saturation.

### 4.7 Decision Tree for Capacity Planning

```
START: Define composite SLO
  |
  v
Run benchmark with increasing concurrency
(e.g., --concurrency 1,2,4,8,16,32,64,128)
  |
  v
Compute compliance surface (Section 2)
  |
  +--> Is compliance >= target at max tested concurrency?
  |      |
  |      YES --> Capacity is at least max_tested.
  |      |       Run at higher concurrency to find the limit.
  |      |
  |      NO --> Find max_safe_concurrency_empirical()
  |              |
  |              v
  |         Fit regression model (Section 4.3)
  |              |
  |              v
  |         Identify bottleneck SLO
  |              |
  |              +--> TTFT bottleneck?
  |              |      --> Investigate: KV cache, prefill batch size, model parallelism
  |              |
  |              +--> ITL bottleneck?
  |              |      --> Investigate: Batch size, KV cache preemption, decode scheduling
  |              |
  |              +--> Throughput bottleneck?
  |                     --> Investigate: GPU utilization, memory bandwidth, batch scheduling
  |
  v
Report: max_safe_concurrency, bottleneck SLO, leading indicators, confidence interval
```

---

## 5. Multi-Signal Anomaly Detection

### 5.1 Why Single-Metric Monitoring Fails

Single-metric thresholds produce false positives and miss real problems:

- **False positive:** GPU utilization at 95% triggers an alert, but throughput
  and latency are fine — the server is efficiently processing a large batch.
- **False negative:** GPU utilization is 60%, but throughput has dropped 50% —
  the server is idle-waiting on KV cache operations or experiencing a scheduling
  pathology.

The solution is to monitor the **correlation structure** between metrics, not
individual metric values. An anomaly is a deviation from the normal correlation
pattern.

### 5.2 Normal Correlation Structure

During healthy operation, LLM inference metrics exhibit characteristic
correlations:

| Metric Pair | Expected Correlation | Mechanism |
|-------------|---------------------|-----------|
| concurrency ↔ throughput | Positive (0.7-0.95) | More concurrent requests → more tokens produced |
| concurrency ↔ TTFT | Positive (0.3-0.8) | More load → longer queue wait → higher TTFT |
| concurrency ↔ kv_cache | Positive (0.8-0.99) | More active requests → more KV cache allocated |
| kv_cache ↔ ITL | Positive (0.2-0.6) | KV cache pressure → preemption → decode restarts |
| gpu_utilization ↔ throughput | Positive (0.6-0.9) | More compute → more tokens produced |
| queue_depth ↔ TTFT | Positive (0.7-0.95) | Queue wait directly adds to TTFT |
| gpu_power ↔ gpu_utilization | Positive (0.8-0.99) | More compute → more power |
| throughput ↔ ITL | Negative (-0.3 to -0.7) | Higher per-user throughput → lower ITL |

### 5.3 PCA-Based Anomaly Scoring

Principal Component Analysis on the multi-metric correlation matrix identifies
the normal mode of variation. Anomalies project strongly onto the minor
principal components.

**Algorithm:**

```python
def multi_signal_anomaly_score(
    metric_matrix: NDArray[np.float64],  # (n_time_bins, n_metrics)
    n_normal_components: int = 3,
) -> NDArray[np.float64]:
    """Compute anomaly scores for each time bin via PCA residual.

    High scores indicate time bins where the metric correlation
    structure deviates from the normal pattern.

    Args:
        metric_matrix: Standardized (zero mean, unit variance) metric values.
            Each row is a time bin, each column is a metric.
        n_normal_components: Number of principal components considered "normal."
            Typical: 2-3 for LLM inference (load + efficiency modes).

    Returns:
        Per-bin anomaly scores (sum of squared residuals in minor PC space).
    """
    # Standardize columns
    means = np.mean(metric_matrix, axis=0)
    stds = np.std(metric_matrix, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero for constant metrics
    Z = (metric_matrix - means) / stds

    # SVD decomposition
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)

    # Project onto minor components (the "unusual" directions)
    minor_components = Vt[n_normal_components:]  # (n_metrics - n_normal, n_metrics)
    residuals = Z @ minor_components.T            # (n_bins, n_metrics - n_normal)

    # Anomaly score = sum of squared residuals
    scores = np.sum(residuals**2, axis=1)         # (n_bins,)
    return scores
```

**Interpreting scores:**

- Scores follow approximately a chi-squared distribution with (n_metrics -
  n_normal_components) degrees of freedom under normality.
- Use the 99th percentile of the chi-squared distribution as the anomaly
  threshold: `threshold = chi2.ppf(0.99, df=n_metrics - n_normal_components)`.
- Time bins exceeding this threshold warrant investigation.

### 5.4 Weighted Scoring Alternative

PCA is optimal when metrics are roughly normally distributed. For LLM inference
metrics (which often have heavy tails), a simpler weighted scoring approach may
be more robust:

```python
def weighted_anomaly_score(
    metrics: dict[str, NDArray[np.float64]],  # tag -> per-bin values
    expected_correlations: dict[tuple[str, str], float],  # (tag_a, tag_b) -> expected_rho
    window_size: int = 20,
) -> NDArray[np.float64]:
    """Detect anomalies via rolling correlation deviation.

    Computes rolling Pearson correlation between each pair of metrics
    and flags time bins where the observed correlation deviates
    significantly from the expected correlation.
    """
    n_bins = len(next(iter(metrics.values())))
    total_deviation = np.zeros(n_bins)

    for (tag_a, tag_b), expected_rho in expected_correlations.items():
        a = metrics[tag_a]
        b = metrics[tag_b]
        for i in range(window_size, n_bins):
            window_a = a[i - window_size : i]
            window_b = b[i - window_size : i]
            observed_rho = np.corrcoef(window_a, window_b)[0, 1]
            deviation = (observed_rho - expected_rho) ** 2
            total_deviation[i] += deviation

    return total_deviation
```

### 5.5 Anomaly Pattern Library

Specific anomaly signatures map to known failure modes:

| Pattern | Metrics Involved | Normal | Anomalous | Likely Cause |
|---------|-----------------|--------|-----------|--------------|
| Throughput collapse | gpu_util HIGH, throughput LOW | Positive corr | Negative corr | KV cache thrashing, preemption storm |
| Silent queue | queue_depth HIGH, TTFT NORMAL | Positive corr | Decoupled | Requests queued but dequeued in bursts |
| Thermal throttle | temperature HIGH, gpu_util DROPPING | Independent | Negative corr | GPU thermal throttling |
| Memory leak | memory_used MONOTONE UP, throughput STABLE | Uncorrelated | Positive trend | Memory fragmentation, cache not freed |
| Batch scheduling pathology | concurrency STABLE, ITL BIMODAL | Smooth | Multimodal | Alternating large/small batch sizes |
| Prefill starvation | TTFT SPIKE, ITL NORMAL | Correlated | Decoupled | Prefill deprioritized for decode |

For each pattern, the detection criterion can be formalized:

```python
ANOMALY_PATTERNS = {
    "throughput_collapse": {
        "condition": lambda m: (
            m["gpu_utilization"].mean() > 80
            and m["effective_throughput"].mean() < m["effective_throughput"].std()
        ),
        "explanation": "GPU utilization is high but throughput is abnormally low. "
                       "This suggests compute is being spent on non-productive work "
                       "(e.g., KV cache recomputation after preemption).",
    },
    "silent_queue_buildup": {
        "condition": lambda m: (
            np.corrcoef(m["num_requests_waiting"], m["time_to_first_token"])[0, 1] < 0.2
            and m["num_requests_waiting"].max() > 10
        ),
        "explanation": "Queue depth is high but TTFT is not proportionally affected. "
                       "Requests are being batched aggressively, hiding individual queue wait.",
    },
}
```

### 5.6 Implementation Considerations

**Metric alignment.** Server metrics and GPU telemetry are polled at different
intervals (1-15s). Client metrics arrive per-request. All must be resampled to
a common grid before correlation analysis (see Section 3.3).

**Minimum data requirement.** PCA requires at least n_metrics + 1 time bins to
produce meaningful results. For 12 metrics, this means at least 13 bins. With
5-second bins, the minimum run duration is ~65 seconds. For reliable anomaly
detection, aim for at least 60 bins (5 minutes at 5-second binning).

**Online vs. offline.** The algorithms above are batch (offline) analyses. For
real-time anomaly detection during a benchmark run, use an exponentially weighted
moving correlation matrix and flag when its eigenvalue distribution shifts.

---

## 6. Degradation Waterfall

### 6.1 The Root Cause Problem

When an SLO violation occurs, the user needs to know *why*. A flat list of
correlated metrics is not enough — the user needs the **causal chain** from root
cause to symptom.

In LLM inference, the canonical degradation waterfall is:

```
High concurrency
  → Queue buildup (num_requests_waiting increases)
    → KV cache pressure (kv_cache_usage_perc > 90%)
      → Preemption (requests evicted from batch)
        → Recomputation (evicted requests re-prefill)
          → High TTFT (P99 exceeds SLO)
          → Throughput drop (effective tokens/sec drops)
```

This is a directed acyclic graph (DAG) of causal relationships. Each edge has a
**characteristic lag** (how long after the cause before the effect manifests).

### 6.2 Causal Graph Definition

Define the causal graph as a set of directed edges:

```python
@dataclass
class CausalEdge:
    """A directed causal relationship between two metrics."""
    cause: str            # Metric tag of the cause
    effect: str           # Metric tag of the effect
    expected_lag_s: float # Expected time lag in seconds
    mechanism: str        # Human-readable explanation

INFERENCE_CAUSAL_GRAPH: list[CausalEdge] = [
    CausalEdge(
        cause="effective_concurrency",
        effect="num_requests_waiting",
        expected_lag_s=1.0,
        mechanism="Concurrency exceeds server batch capacity → requests queue",
    ),
    CausalEdge(
        cause="num_requests_waiting",
        effect="kv_cache_usage_perc",
        expected_lag_s=2.0,
        mechanism="Queued requests admitted to batch → KV cache allocation",
    ),
    CausalEdge(
        cause="kv_cache_usage_perc",
        effect="time_to_first_token",
        expected_lag_s=5.0,
        mechanism="KV cache pressure → preemption → recomputation → higher TTFT",
    ),
    CausalEdge(
        cause="kv_cache_usage_perc",
        effect="inter_token_latency",
        expected_lag_s=3.0,
        mechanism="KV cache pressure → preemption during decode → ITL spikes",
    ),
    CausalEdge(
        cause="num_requests_running",
        effect="inter_token_latency",
        expected_lag_s=0.5,
        mechanism="Larger batch → more attention computation per decode step",
    ),
    CausalEdge(
        cause="num_requests_running",
        effect="gpu_utilization",
        expected_lag_s=0.1,
        mechanism="Larger batch → more GPU compute per iteration",
    ),
    CausalEdge(
        cause="gpu_utilization",
        effect="gpu_power_usage",
        expected_lag_s=0.1,
        mechanism="Higher compute → higher power draw (near-instantaneous)",
    ),
    CausalEdge(
        cause="gpu_power_usage",
        effect="gpu_temperature",
        expected_lag_s=30.0,
        mechanism="Sustained power draw → thermal mass heats up (slow)",
    ),
    CausalEdge(
        cause="gpu_temperature",
        effect="gpu_utilization",
        expected_lag_s=5.0,
        mechanism="Temperature exceeds throttle threshold → clock reduction",
    ),
    CausalEdge(
        cause="effective_concurrency",
        effect="effective_throughput",
        expected_lag_s=0.5,
        mechanism="More concurrent requests → more tokens in flight → higher throughput",
    ),
    CausalEdge(
        cause="kv_cache_usage_perc",
        effect="effective_throughput",
        expected_lag_s=5.0,
        mechanism="KV cache saturation → preemption → throughput degradation",
    ),
]
```

### 6.3 Waterfall Trace Algorithm

Given an SLO violation event at time t_violation, trace backward through the
causal graph to identify the root cause:

```python
def trace_degradation_waterfall(
    violation_metric: str,
    violation_time_ns: float,
    causal_graph: list[CausalEdge],
    metric_signals: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    anomaly_threshold_sigma: float = 2.0,
) -> list[WaterfallStep]:
    """Trace backward from an SLO violation to find the root cause chain.

    For each causal edge pointing to the violation metric, check if the
    cause metric was anomalously high at (violation_time - expected_lag).
    Recurse on any activated causes.

    Args:
        violation_metric: The metric that violated the SLO (e.g., "time_to_first_token").
        violation_time_ns: Timestamp of the violation.
        causal_graph: Directed causal edges between metrics.
        metric_signals: All available metric time series.
        anomaly_threshold_sigma: How many sigma above mean to consider "activated."

    Returns:
        Ordered list of WaterfallStep from root cause to symptom.
    """
    steps: list[WaterfallStep] = []
    visited: set[str] = set()

    def _trace(metric: str, time_ns: float, depth: int) -> None:
        if metric in visited or depth > 10:
            return
        visited.add(metric)

        # Find all edges where this metric is the effect
        incoming = [e for e in causal_graph if e.effect == metric]
        for edge in incoming:
            if edge.cause not in metric_signals:
                continue
            cause_ts, cause_vals = metric_signals[edge.cause]
            cause_time_ns = time_ns - edge.expected_lag_s * 1e9

            # Check if cause metric was anomalously high at the expected time
            idx = np.searchsorted(cause_ts, cause_time_ns)
            idx = min(idx, len(cause_vals) - 1)
            value = cause_vals[idx]
            mean = np.mean(cause_vals)
            std = np.std(cause_vals)

            if std > 0 and (value - mean) / std > anomaly_threshold_sigma:
                steps.append(WaterfallStep(
                    cause=edge.cause,
                    effect=edge.effect,
                    lag_s=edge.expected_lag_s,
                    mechanism=edge.mechanism,
                    cause_value=value,
                    cause_z_score=(value - mean) / std,
                ))
                _trace(edge.cause, cause_time_ns, depth + 1)

    _trace(violation_metric, violation_time_ns, 0)
    steps.reverse()  # Root cause first
    return steps
```

### 6.4 Waterfall Report Format

The waterfall report presents the causal chain in human-readable form:

```
=== Degradation Waterfall: P99 TTFT > 500ms at t=120.5s ===

[ROOT] effective_concurrency = 48.2 (z=3.1, normally ~32.0)
  |
  |-- +1.0s --> num_requests_waiting = 12.4 (z=2.8, normally ~2.1)
  |               Queue depth spiked as concurrency exceeded batch capacity
  |
  |-- +2.0s --> kv_cache_usage_perc = 94.2% (z=2.5, normally ~72.0%)
  |               Queued requests consumed KV cache beyond preemption threshold
  |
  |-- +5.0s --> time_to_first_token = 623ms (P99, SLO threshold: 500ms)
                  KV cache pressure caused preemption and recomputation

Bottleneck: KV cache capacity
Recommendation: Increase KV cache block size, reduce max sequence length,
                or add tensor parallelism to increase KV cache memory.
```

### 6.5 Automated Recommendation Engine

Based on the root cause identified in the waterfall, map to actionable
recommendations:

| Root Cause | Bottleneck | Recommendation |
|-----------|-----------|----------------|
| High concurrency → queue depth | Server batch capacity | Reduce concurrency target or increase tensor parallelism |
| Queue depth → KV cache pressure | KV cache memory | Increase `--gpu-memory-utilization`, reduce `--max-model-len`, enable chunked prefill |
| KV cache → TTFT | Preemption | Enable prefix caching, reduce batch size, use priority scheduling |
| Batch size → ITL | Decode compute | Reduce `--max-num-batched-tokens`, increase tensor parallelism |
| GPU utilization → thermal throttle | Cooling | Improve cooling, reduce power limit, add more GPUs |
| Concurrency → throughput drop (no queue) | Memory bandwidth | The model is memory-bandwidth bound; increase tensor parallelism or use faster GPUs |

---

## 7. Benchmark Validity Assessment

### 7.1 Why Validity Matters

A benchmark result is only useful if the measurement is valid. Invalid
benchmarks produce misleading numbers — a tool might report P99 TTFT = 50ms
when the real value is 500ms because the benchmark was misconfigured or the
measurement methodology has a systematic error.

Validity assessment uses **internal consistency checks**: relationships between
metrics that must hold if the measurement is correct. Each check produces a
pass/warn/fail result, and the aggregate gives a validity score.

### 7.2 Little's Law Check

Little's Law (L = lambda * W) relates three quantities:

- L: Average number of items in the system (effective concurrency)
- lambda: Arrival rate (request throughput)
- W: Average time in the system (mean request latency)

All three are independently measured by AIPerf. If the measurement is correct,
the relationship must hold:

```
littles_law_error = |L - lambda * W| / L
```

**Thresholds:**

| Error | Assessment | Explanation |
|-------|-----------|-------------|
| < 5% | Pass | Normal measurement uncertainty |
| 5-15% | Warn | Possible measurement or attribution issue |
| > 15% | Fail | Something is wrong — check timestamps, request counting |

**Implementation using existing AIPerf metrics:**

```python
def littles_law_check(
    effective_concurrency: float,  # From sweep metric (time-weighted avg)
    request_throughput: float,     # From derived metric
    mean_request_latency_s: float, # From record metric (mean)
) -> tuple[float, str]:
    """Check Little's Law consistency: L = lambda * W.

    Returns (error_fraction, assessment).
    """
    predicted_L = request_throughput * mean_request_latency_s
    if effective_concurrency == 0:
        return 0.0, "pass"  # No load — trivially valid
    error = abs(effective_concurrency - predicted_L) / effective_concurrency
    if error < 0.05:
        return error, "pass"
    elif error < 0.15:
        return error, "warn"
    else:
        return error, "fail"
```

**Why it might fail:**

- **Ramp-up/ramp-down artifacts.** During ramp-up, concurrency is increasing
  but throughput hasn't caught up → L > lambda * W. This is expected and is
  another argument for steady-state windowed analysis.
- **Failed requests.** Requests that error out are counted in concurrency (they
  occupied a slot) but may not be counted in throughput (they didn't produce
  tokens). AIPerf's separate good_request_count vs. request_count can detect
  this.
- **Clock skew.** If request_start_ns and request_end_ns use different clocks,
  latency measurements will be biased. This is unlikely in a single-process
  benchmark but possible in distributed setups.

### 7.3 Client-Server Latency Reconciliation

When both client-side latency (from AIPerf) and server-side latency histograms
(from Prometheus) are available, they should approximately agree:

```
client_latency ≈ server_latency + network_round_trip
```

For streaming requests:
```
client_TTFT ≈ server_TTFT + network_one_way
client_ITL ≈ server_ITL  (network latency cancels between consecutive tokens)
```

**Reconciliation check:**

```python
def latency_reconciliation(
    client_p50_ms: float,   # Client-measured P50 request_latency
    server_p50_ms: float,   # Server-reported P50 from histogram
    network_rtt_ms: float,  # Estimated network RTT (from HTTP trace dns + connect)
) -> tuple[float, str]:
    """Check that client latency ≈ server latency + network.

    Returns (discrepancy_ms, assessment).
    """
    expected_client = server_p50_ms + network_rtt_ms
    discrepancy = abs(client_p50_ms - expected_client)

    # Allow up to 20% discrepancy (Prometheus histogram interpolation introduces error)
    relative_error = discrepancy / max(client_p50_ms, 1.0)
    if relative_error < 0.10:
        return discrepancy, "pass"
    elif relative_error < 0.20:
        return discrepancy, "warn"
    else:
        return discrepancy, "fail"
```

### 7.4 Token Count Reconciliation

The total tokens generated should be consistent across metrics:

```
total_output_tokens ≈ sum(output_sequence_length per request)
                    ≈ effective_throughput * benchmark_duration
                    ≈ request_count * mean(output_sequence_length)
```

**Check:**

```python
def token_count_reconciliation(
    total_output_tokens: float,        # From aggregate metric
    throughput_tokens_per_sec: float,   # From derived metric
    benchmark_duration_s: float,        # From aggregate metric
) -> tuple[float, str]:
    """Check that total tokens ≈ throughput * duration.

    Returns (error_fraction, assessment).
    """
    predicted = throughput_tokens_per_sec * benchmark_duration_s
    if total_output_tokens == 0:
        return 0.0, "pass"
    error = abs(total_output_tokens - predicted) / total_output_tokens
    if error < 0.05:
        return error, "pass"
    elif error < 0.15:
        return error, "warn"
    else:
        return error, "fail"
```

### 7.5 GPU Utilization Plausibility

GPU utilization should be consistent with the observed throughput and model
characteristics:

```
expected_gpu_util ≈ throughput * flops_per_token / gpu_peak_flops
```

This requires knowing the model's approximate FLOPs per token (a function of
model size, precision, and whether prefill or decode). For a rough check:

| Model Size | Approx FLOPs/token (decode, FP16) | H100 Peak (FP16) | Expected util at 1000 tok/s |
|-----------|----------------------------------|------------------|---------------------------|
| 7B | 14 GFLOP | 990 TFLOP/s | 0.0014% |
| 70B | 140 GFLOP | 990 TFLOP/s | 0.014% |
| 405B | 810 GFLOP | 990 TFLOP/s | 0.082% |

Note: Decode is memory-bandwidth bound, so GPU SM utilization is inherently low.
Prefill is compute-bound, so utilization during prefill is much higher. A
validity check needs to account for the prefill/decode ratio.

**Simplified plausibility check:**

```python
def gpu_utilization_plausibility(
    avg_gpu_util: float,           # Observed GPU utilization %
    effective_concurrency: float,  # Active requests
) -> tuple[str, str]:
    """Basic plausibility: GPU utilization should correlate with load.

    Returns (assessment, explanation).
    """
    if effective_concurrency > 1 and avg_gpu_util < 5.0:
        return "warn", "GPU utilization is very low despite active requests. Possible measurement issue."
    if effective_concurrency < 0.5 and avg_gpu_util > 80.0:
        return "warn", "GPU utilization is very high with near-zero load. Another workload may be running."
    return "pass", "GPU utilization is plausible for the observed load level."
```

### 7.6 Aggregate Validity Score

Combine individual checks into an overall validity score:

```python
@dataclass
class ValidityAssessment:
    """Aggregate benchmark validity assessment."""
    littles_law: tuple[float, str]
    latency_reconciliation: tuple[float, str] | None  # None if no server metrics
    token_reconciliation: tuple[float, str]
    gpu_plausibility: tuple[str, str] | None          # None if no GPU telemetry

    @property
    def score(self) -> str:
        """Aggregate: 'valid', 'caution', or 'suspect'."""
        assessments = [
            self.littles_law[1],
            self.token_reconciliation[1],
        ]
        if self.latency_reconciliation is not None:
            assessments.append(self.latency_reconciliation[1])
        if self.gpu_plausibility is not None:
            assessments.append(self.gpu_plausibility[0])

        if any(a == "fail" for a in assessments):
            return "suspect"
        if any(a == "warn" for a in assessments):
            return "caution"
        return "valid"

    @property
    def explanation(self) -> str:
        """Human-readable summary of all checks."""
        lines = []
        error, status = self.littles_law
        lines.append(f"Little's Law: {status} (error={error:.1%})")
        error, status = self.token_reconciliation
        lines.append(f"Token reconciliation: {status} (error={error:.1%})")
        if self.latency_reconciliation is not None:
            disc, status = self.latency_reconciliation
            lines.append(f"Latency reconciliation: {status} (discrepancy={disc:.1f}ms)")
        if self.gpu_plausibility is not None:
            status, detail = self.gpu_plausibility
            lines.append(f"GPU utilization: {status} ({detail})")
        return "\n".join(lines)
```

### 7.7 Correlation-Based Validity Extension

Beyond point checks, the correlation matrix itself serves as a validity
indicator. A valid benchmark should exhibit:

1. **Positive correlation between concurrency and throughput** (unless the server
   is saturated). If these are negatively correlated during steady state,
   something is wrong.

2. **Positive correlation between concurrency and queue depth.** If queue depth
   is zero regardless of concurrency, the server may be so fast that requests
   never queue (possible but unusual), or the queue metric is not being reported.

3. **KV cache usage scales with concurrency.** Each active request allocates KV
   cache. If kv_cache_usage_perc doesn't change with concurrency, the metric
   may be stale or misconfigured.

4. **GPU utilization responds to load.** If gpu_utilization is constant
   regardless of concurrency, the GPU metric may be reporting a cached value.

Each correlation check has an expected range. Deviations beyond 2 sigma from
expected constitute a validity warning:

```python
EXPECTED_CORRELATIONS: dict[tuple[str, str], tuple[float, float]] = {
    # (metric_a, metric_b): (expected_rho, tolerance)
    ("effective_concurrency", "effective_throughput"): (0.85, 0.20),
    ("effective_concurrency", "kv_cache_usage_perc"): (0.90, 0.15),
    ("effective_concurrency", "num_requests_running"): (0.95, 0.10),
    ("num_requests_waiting", "time_to_first_token"): (0.70, 0.25),
    ("gpu_utilization", "effective_throughput"): (0.75, 0.25),
}
```

---

## 8. Comparative Analysis

### 8.1 Cross-Configuration Correlation Stability

Different server configurations (model sizes, hardware, serving frameworks)
produce different absolute metric values but may share the same correlation
structure. Identifying which correlations are **universal** (hold across all
configurations) vs. **configuration-specific** is valuable for building robust
monitoring and alerting.

**Methodology:**

1. Run benchmarks across K configurations.
2. For each configuration, compute the steady-state correlation matrix R_k
   (metrics x metrics).
3. Compute the element-wise mean and standard deviation across configurations:
   ```
   R_mean = (1/K) * sum(R_k)
   R_std = std(R_k)
   ```
4. A correlation is **universal** if R_std[i,j] < 0.15 (stable across configs).
5. A correlation is **configuration-specific** if R_std[i,j] > 0.30.

**Expected universal correlations:**

| Pair | Expected | Why Universal |
|------|----------|--------------|
| concurrency ↔ kv_cache | Strong positive | KV cache allocation is proportional to active requests regardless of model |
| queue_depth ↔ TTFT | Strong positive | Queuing theory holds universally |
| gpu_power ↔ gpu_utilization | Strong positive | Physics: more compute = more power |
| batch_size ↔ ITL | Moderate positive | Larger batches always take longer to decode |

**Expected configuration-specific correlations:**

| Pair | Expected | Why Variable |
|------|----------|-------------|
| concurrency ↔ throughput | Varies | Depends on saturation point, which is hardware-specific |
| gpu_utilization ↔ ITL | Varies | Depends on compute vs. memory-bandwidth boundedness |
| kv_cache ↔ ITL | Varies | Depends on preemption policy (vLLM vs. TRT-LLM differ) |

### 8.2 Model Size Scaling Laws for Correlations

How do correlations change with model size?

**Hypothesis: KV cache pressure becomes more dominant at larger model sizes.**

- 7B model: KV cache is small per request → kv_cache_usage_perc has weak
  correlation with TTFT (preemption is rare).
- 70B model: KV cache is larger per request → kv_cache_usage_perc has strong
  correlation with TTFT (preemption is common under load).
- 405B model: KV cache dominates → kv_cache_usage_perc becomes the primary
  predictor of both TTFT and ITL.

**Hypothesis: GPU utilization becomes less informative at larger model sizes.**

- 7B model: Decode is memory-bandwidth bound → GPU SM utilization is low and
  uninformative.
- 405B model: With tensor parallelism, aggregate GPU utilization is higher and
  more informative, but the per-GPU utilization may be less correlated with
  per-request latency.

**Verification approach:** Run the same benchmark suite across model sizes on
the same hardware. Compare the leading indicator ranking (Section 3.4) for each
model size. If the ranking changes systematically, document the scaling law.

### 8.3 Hardware Comparison

Different GPU generations have different bottleneck characteristics:

| Hardware | Primary Bottleneck | Expected Top Indicator |
|----------|-------------------|----------------------|
| A100 (80GB) | Memory bandwidth | gpu_utilization (more informative, frequently high) |
| H100 (80GB) | Often compute-bound for prefill | kv_cache_usage_perc (memory less constrained) |
| H200 (141GB) | Rarely memory-limited | queue_depth (server scheduling becomes bottleneck) |

**Methodology:** For each hardware configuration, compute:

1. The SLO compliance surface (Section 2)
2. The leading indicator ranking (Section 3)
3. The capacity limit and bottleneck SLO (Section 4)

Report these side-by-side. Differences in the bottleneck SLO reveal where each
hardware configuration hits its limit first.

### 8.4 Serving Framework Comparison

vLLM and TensorRT-LLM use different scheduling strategies, which may change
the correlation structure:

- **vLLM:** Continuous batching with preemption. KV cache preemption is a
  primary degradation mechanism. Expected: strong kv_cache ↔ TTFT correlation.
- **TensorRT-LLM:** In-flight batching with (optionally) paged attention.
  Scheduling policy differences may change which metrics are most predictive.
- **Triton Inference Server:** Adds an additional queuing layer. Expected:
  queue_depth becomes an even stronger leading indicator.

**Methodology:** Same benchmark, same hardware, different serving frameworks.
Compare correlation matrices. Differences reveal framework-specific optimization
opportunities.

### 8.5 Meta-Analysis Report Format

A comparative meta-analysis report should include:

```
=== Meta-Analysis: 3 Configurations ===

Configuration A: Llama-70B on 4xH100 (vLLM 0.6.0)
Configuration B: Llama-70B on 4xH100 (TRT-LLM 0.12)
Configuration C: Llama-70B on 8xA100 (vLLM 0.6.0)

--- SLO Compliance ---
| Config | Max Safe Concurrency (P99 TTFT < 500ms) | Bottleneck SLO |
|--------|------------------------------------------|----------------|
| A      | 48 (CI: 44-52)                           | TTFT           |
| B      | 56 (CI: 51-61)                           | ITL            |
| C      | 38 (CI: 34-42)                           | TTFT           |

--- Leading Indicators (TTFT) ---
| Rank | Config A     | Config B         | Config C     |
|------|-------------|------------------|-------------|
| 1    | kv_cache    | num_requests_running | kv_cache    |
| 2    | queue_depth | kv_cache         | gpu_util    |
| 3    | gpu_util    | queue_depth      | queue_depth |

--- Universal Correlations (stable across all configs) ---
| Pair                         | Mean rho | Std  |
|------------------------------|----------|------|
| concurrency ↔ kv_cache       | 0.92     | 0.03 |
| queue_depth ↔ TTFT           | 0.78     | 0.08 |
| gpu_power ↔ gpu_util         | 0.95     | 0.02 |

--- Configuration-Specific Correlations ---
| Pair                    | Config A | Config B | Config C |
|------------------------|----------|----------|----------|
| kv_cache ↔ ITL         | 0.72     | 0.31     | 0.68     |
| gpu_util ↔ throughput  | 0.85     | 0.91     | 0.62     |

Insight: Config B (TRT-LLM) shows weaker kv_cache ↔ ITL correlation,
suggesting its scheduling policy handles KV cache pressure with less
impact on decode latency. Config C (8xA100) shows weaker gpu_util ↔
throughput, consistent with memory-bandwidth-bound operation on A100.
```

---

## AIPerf Implementation Roadmap

### Phase 1: SLO Compliance Infrastructure (Foundation)

**Goal:** Extend the existing `--goodput` mechanism to support full SLO
compliance analysis.

**New module:** `src/aiperf/analysis/slo_compliance.py`

```python
"""SLO compliance analysis: per-request compliance, temporal compliance curves,
and compliance surface construction.

All functions operate on ColumnStore arrays — no record objects, no Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AtomicSLO:
    """A single SLO constraint on one metric."""

    metric_tag: str
    percentile: float      # 0-100
    comparator: str        # "<" or ">="
    threshold: float       # In metric's base unit
    display_unit: str      # For reporting


@dataclass(frozen=True, slots=True)
class CompositeSLO:
    """A conjunction of atomic SLOs — all must be met simultaneously."""

    name: str
    slos: tuple[AtomicSLO, ...]

    def per_request_compliance(
        self,
        store: "ColumnStore",
        mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.bool_]:
        """Boolean mask: True for requests meeting ALL SLOs."""
        # ... implementation per Section 2.2 ...
```

**Integration points:**

- Parse `--goodput "ttft:500 itl:50"` into `CompositeSLO` (extend existing parser)
- `MetricsAccumulator` gains a `compliance_mask()` method that returns per-request compliance
- Compliance rate is computed in `_compute_results()` alongside existing goodput

**Effort:** Medium. Most infrastructure exists (goodput parsing, ColumnStore
masking). New code is the CompositeSLO abstraction and temporal compliance
curve.

### Phase 2: Compliance Surface and Capacity Planning

**Goal:** Construct the SLO compliance surface and extract capacity limits.

**New module:** `src/aiperf/analysis/capacity_planning.py`

```python
"""Capacity planning via SLO compliance surface analysis.

Combines sweep curves (concurrency, throughput) with per-request SLO
compliance to map the safe operating envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ComplianceSurface:
    """2D compliance map over (concurrency, throughput) space."""

    concurrency_bins: NDArray[np.float64]
    throughput_bins: NDArray[np.float64]
    compliance: NDArray[np.float64]  # (n_bins,) compliance rate per temporal bin
    max_safe_concurrency: float | None
    bottleneck_slo: str | None


@dataclass(frozen=True, slots=True)
class CapacityEstimate:
    """Maximum safe concurrency with confidence interval."""

    concurrency: float
    ci_lower: float
    ci_upper: float
    bottleneck_slo: str
    regression_r_squared: float
```

**Integration points:**

- `SteadyStateAnalyzer` already reads `MetricsAccumulator` at summarize time.
  The capacity planner follows the same pattern: an AnalyzerProtocol plugin
  that reads ColumnStore + sweep curves + server metrics.
- New plugin in `plugins.yaml` under the `analyzer` category (if added) or as
  an optional analysis pass in `RecordsManager`.
- Export: `ConsoleCapacityExporter`, `CapacityCsvExporter`, `CapacityJsonExporter`.

**Effort:** High. Requires temporal binning, multi-source signal alignment,
regression, and bootstrap CI.

### Phase 3: Cross-Correlation Analysis

**Goal:** Compute and rank leading indicators.

**New module:** `src/aiperf/analysis/correlation.py`

```python
"""Cross-correlation analysis between client, server, and GPU metric signals.

Resamples multi-resolution signals to a common grid and computes
lagged Pearson correlation with significance testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    """Cross-correlation between two metrics."""

    cause_tag: str
    effect_tag: str
    optimal_lag_s: float
    max_correlation: float
    p_value: float
    significant: bool


def rank_leading_indicators(
    results: list[CorrelationResult],
    effect_tag: str,
) -> list[CorrelationResult]:
    """Rank indicators by predictive power for a specific target metric."""
    relevant = [r for r in results if r.effect_tag == effect_tag and r.significant]
    return sorted(relevant, key=lambda r: abs(r.max_correlation), reverse=True)
```

**Integration points:**

- Reads from all three accumulators (MetricsAccumulator, ServerMetricsAccumulator,
  GPUTelemetryAccumulator) via the SummaryContext.
- Requires server metrics time series extraction (currently stored hierarchically
  in ServerMetricsAccumulator — would need a `timeseries()` accessor).
- Requires GPU telemetry time series extraction (GrowableArray in
  GPUTelemetryAccumulator — already accessible).

**Effort:** High. The cross-correlation algorithm itself is straightforward
(numpy), but signal alignment across three different temporal resolutions is the
main challenge.

### Phase 4: Anomaly Detection and Degradation Waterfall

**Goal:** Multi-signal anomaly scoring and automated root cause tracing.

**New modules:**

- `src/aiperf/analysis/anomaly.py` — PCA-based and weighted anomaly scoring
- `src/aiperf/analysis/waterfall.py` — Causal graph definition and backward tracing

**Integration points:**

- The causal graph (Section 6.2) is static knowledge, defined as a data
  structure in `waterfall.py`.
- Anomaly detection runs as a post-analysis pass after all accumulators have
  produced their summaries.
- Output format: structured JSON for programmatic consumption, console table
  for human review.

**Effort:** Medium. PCA is straightforward. The waterfall trace is a simple
graph traversal. The main work is in the causal graph curation and the
recommendation engine.

### Phase 5: Benchmark Validity Scoring

**Goal:** Automated validity checks on every benchmark run.

**New module:** `src/aiperf/analysis/validity.py`

**Integration points:**

- Little's Law check uses `effective_concurrency`, `request_throughput`, and
  `mean(request_latency)` — all already computed.
- Token reconciliation uses `total_output_tokens`, `effective_throughput`, and
  `benchmark_duration` — all already computed.
- Latency reconciliation requires server-side histogram percentiles — available
  from ServerMetricsAccumulator when Prometheus metrics are configured.
- Validity score is included in the JSON export and optionally in the console
  summary.

**Effort:** Low. All input metrics already exist. This is a few dozen lines
of numpy comparisons.

### Phase 6: Comparative Analysis Framework

**Goal:** Cross-configuration meta-analysis.

This phase requires storing and comparing results across multiple benchmark
runs. It is best implemented as a post-hoc analysis tool rather than an
in-pipeline analyzer:

```bash
aiperf compare \
    --results run_a.json run_b.json run_c.json \
    --correlation-stability \
    --capacity-comparison \
    --output meta-analysis.json
```

**Effort:** Medium-High. Requires a new CLI command and result-loading
infrastructure that can ingest completed benchmark results.

### Implementation Priority

| Phase | Name | Effort | Dependency | Value |
|-------|------|--------|-----------|-------|
| 5 | Validity scoring | Low | None | High — catches bad benchmarks |
| 1 | SLO compliance | Medium | None | High — extends existing goodput |
| 2 | Compliance surface | High | Phase 1 | Very high — capacity planning |
| 3 | Cross-correlation | High | None | Medium — diagnostic insight |
| 4 | Anomaly + waterfall | Medium | Phase 3 | Medium — root cause automation |
| 6 | Comparative analysis | Medium-High | Phases 1-3 | Medium — meta-analysis |

**Recommended start:** Phase 5 (validity scoring) is low-effort, high-value,
and has no dependencies. It can ship independently and provides immediate value
by flagging suspect benchmarks. Phase 1 (SLO compliance) follows naturally from
the existing goodput infrastructure.

---

## Appendix A: Mathematical Reference

### A.1 Pearson Cross-Correlation

For discrete time series x[n] and y[n] of length N:

```
R_xy[k] = (1/N) * sum_{n=0}^{N-1-k} (x[n] - mu_x)(y[n+k] - mu_y) / (sigma_x * sigma_y)
```

Where k is the lag (positive k means x leads y), mu is the mean, and sigma is
the standard deviation.

**Computational note:** For large N and many lags, the FFT-based approach is
O(N log N) vs. O(N * K) for the direct method:

```python
def fft_cross_correlation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Full cross-correlation via FFT (O(N log N))."""
    n = len(x)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    fft_size = 2 * n  # Zero-pad to avoid circular correlation
    Fx = np.fft.rfft(x_centered, n=fft_size)
    Fy = np.fft.rfft(y_centered, n=fft_size)
    correlation = np.fft.irfft(Fx * np.conj(Fy), n=fft_size)[:n]
    # Normalize
    norm = n * np.std(x) * np.std(y)
    return correlation / norm if norm > 0 else np.zeros(n)
```

### A.2 PCA for Anomaly Detection

Given a standardized data matrix Z (n_bins x n_metrics):

```
Z = U * S * V^T   (SVD decomposition)
```

The principal components are the rows of V^T. The first k components capture
the "normal" variation. The residual:

```
Z_residual = Z - Z * V_k * V_k^T
```

where V_k is the first k columns of V. The anomaly score for each row:

```
score[i] = ||Z_residual[i]||^2
```

Under normality, score ~ chi^2(n_metrics - k).

### A.3 Segmented Regression

For the piecewise capacity model with unknown breakpoint tau:

```
minimize sum_{t < tau} (y[t] - (a_1 + b_1 * x[t]))^2
       + sum_{t >= tau} (y[t] - (a_2 + b_2 * x[t]))^2
```

Search over candidate breakpoints tau in [x_min + margin, x_max - margin].
For each candidate, fit two OLS regressions and sum the residual variances.
Select the tau that minimizes the total.

The Bayesian Information Criterion (BIC) determines whether the segmented model
is justified over the simple linear model:

```
BIC_segmented = N * log(RSS_segmented / N) + 5 * log(N)   (5 parameters)
BIC_linear    = N * log(RSS_linear / N) + 2 * log(N)      (2 parameters)
```

Use segmented model if BIC_segmented < BIC_linear.

### A.4 Bonferroni and Benjamini-Hochberg Corrections

**Bonferroni (conservative):** Reject H_0 for test i if p_i < alpha / m, where
m is the total number of tests. Controls the Family-Wise Error Rate (FWER).

**Benjamini-Hochberg (less conservative):** Controls the False Discovery Rate
(FDR).

1. Order p-values: p_{(1)} <= p_{(2)} <= ... <= p_{(m)}
2. Find the largest k such that p_{(k)} <= (k/m) * alpha
3. Reject all hypotheses H_{(1)}, ..., H_{(k)}

For the leading indicator analysis with ~800 tests and alpha = 0.05:
- Bonferroni threshold: p < 6.25e-5 (very conservative)
- BH threshold: adaptive, typically accepts 2-5x more discoveries

**Recommendation:** Use Benjamini-Hochberg. In the correlation context, we
expect many true correlations (most server metrics genuinely correlate with
client latency), so the more permissive FDR control is appropriate.

### A.5 Effective Sample Size for Binned Correlations

When computing correlation between binned time series, the effective sample
size depends on the autocorrelation of the binned signals:

```
n_eff = n * (1 - rho_1) / (1 + rho_1)
```

Where rho_1 is the lag-1 autocorrelation of the binned signal. Use n_eff
instead of n in significance testing.

This matters because consecutive time bins are likely correlated (the system
state at time t predicts the state at time t + delta_t). Without this correction,
significance tests are too liberal (too many false positives).

---

## Appendix B: Signal Alignment Details

### B.1 Temporal Resolution by Source

| Source | Resolution | Timestamp Type | Alignment Strategy |
|--------|-----------|---------------|-------------------|
| Client metrics (per-request) | ~10-100ms between requests | `start_ns`, `end_ns` per request | Bin by time window, take median per bin |
| Client sweep metrics | Event-driven (on request start/end) | Nanosecond timestamps on events | Time-weighted average per bin (already implemented in sweep.py) |
| Server metrics (Prometheus) | 1-15s polling interval | Scrape timestamp | Sample-and-hold interpolation to common grid |
| GPU telemetry (DCGM) | 1-5s polling interval | DCGM timestamp | Linear interpolation to common grid |

### B.2 Recommended Common Grid

For cross-correlation analysis, use a common grid with interval:

```
grid_interval = max(server_polling_interval, gpu_polling_interval)
```

Typically 5-15 seconds. This ensures each grid point has at least one
observation from each source.

For the compliance surface (Section 2), a finer grid is possible because only
client metrics are needed:

```
surface_interval = max(1s, 10 / request_rate)
```

### B.3 Handling Missing Data

Server metrics and GPU telemetry may have gaps (missed polls, scrape failures).
For cross-correlation:

- Interpolate short gaps (< 3 intervals) linearly.
- Mark longer gaps as NaN.
- Compute correlation only over non-NaN time steps (pairwise complete
  observations).
- Report the effective number of complete pairs; if less than 30, flag the
  correlation as unreliable.

---

## Appendix C: Causal Graph Extensions

### C.1 Serving-Framework-Specific Edges

The causal graph in Section 6.2 is generic. Framework-specific extensions:

**vLLM-specific edges:**

```python
VLLM_EDGES = [
    CausalEdge(
        cause="kv_cache_usage_perc",
        effect="num_preemptions",  # vLLM-specific metric
        expected_lag_s=1.0,
        mechanism="KV cache exceeds preemption threshold → requests swapped out",
    ),
    CausalEdge(
        cause="num_preemptions",
        effect="time_to_first_token",
        expected_lag_s=2.0,
        mechanism="Preempted requests must re-prefill when rescheduled",
    ),
]
```

**TensorRT-LLM-specific edges:**

```python
TRT_LLM_EDGES = [
    CausalEdge(
        cause="num_requests_running",
        effect="inflight_batcher_queue_latency",
        expected_lag_s=0.5,
        mechanism="In-flight batcher groups requests; larger groups take longer to schedule",
    ),
]
```

### C.2 Edge Weight Calibration

The expected lag values in the causal graph are initial estimates. They should
be calibrated from data:

1. For each edge (A → B), compute the cross-correlation between A and B
   (Section 3.3).
2. The optimal lag from cross-correlation replaces the static `expected_lag_s`.
3. The correlation magnitude becomes the edge weight (strength of the causal
   link).

This turns the static causal graph into a **data-driven causal model** where
edge lags and strengths are estimated from the actual benchmark data.

---

## Appendix D: Comparison with Existing Tools

### D.1 MLPerf Inference

MLPerf Inference defines SLOs but does not perform correlation analysis or
capacity planning. Their approach:

- Fixed SLO thresholds per scenario (single-stream, multi-stream, server, offline)
- Binary pass/fail: either all thresholds are met or the run is invalid
- No leading indicator analysis, no compliance surface, no degradation waterfall

AIPerf's approach extends MLPerf's binary model to a continuous compliance
surface with quantified uncertainty and actionable diagnostics.

### D.2 Locust / wrk2

Load testing tools provide concurrency-vs-latency curves but not:

- Multi-signal correlation across client/server/GPU
- Automated root cause identification
- KV cache-aware capacity planning
- Streaming-specific SLOs (TTFT, ITL decomposition)

### D.3 Prometheus + Grafana

Standard monitoring stack provides real-time dashboards but:

- No automated cross-correlation with time lag
- No compliance surface construction
- No causal graph traversal
- Requires manual investigation for root cause analysis

### D.4 AIPerf's Unique Position

AIPerf is uniquely positioned to implement the full framework described in this
document because it already:

1. Collects all three signal families (client, server, GPU) in a unified pipeline
2. Computes sweep metrics with nanosecond resolution
3. Has a ColumnStore that supports efficient time-range queries with boolean masks
4. Has steady-state detection that identifies the relevant analysis window
5. Has a plugin system (AnalyzerProtocol) that supports modular analysis passes
6. Has bootstrap CI infrastructure for uncertainty quantification
7. Exports structured JSON that can represent nested analysis results

No other benchmarking tool has this combination of capabilities.

---

## References

1. Dean, J. & Barroso, L. A. (2013). "The Tail at Scale." *Communications of
   the ACM*, 56(2), 74-80. — Established p99.9 as the critical metric for
   distributed systems.

2. Zhong, Y., et al. (2024). "DistServe: Disaggregating Prefill and Decoding
   for Goodput-optimized Large Language Model Serving." *OSDI '24*. — Defined
   SLA-based goodput for LLM inference.

3. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language
   Model Serving with PagedAttention." *SOSP '23*. — vLLM's paged attention
   and KV cache management.

4. Little, J. D. C. (1961). "A Proof for the Queuing Formula: L = lambda W."
   *Operations Research*, 9(3), 383-387. — The foundational queuing law used
   for benchmark validity.

5. Tene, G. (2013). "How NOT to Measure Latency." — Coordinated omission and
   latency measurement methodology.

6. Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and
   Practice*, 3rd ed. — Cross-correlation, time series decomposition, and
   stationarity testing.

7. Hotelling, H. (1933). "Analysis of a complex of statistical variables into
   principal components." *Journal of Educational Psychology*, 24(6), 417-441.
   — PCA for multi-dimensional analysis.

8. Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate:
   A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal
   Statistical Society*, Series B, 57(1), 289-300. — FDR correction for
   multiple correlation tests.

9. Killick, R., Fearnhead, P., & Eckley, I. A. (2012). "Optimal detection of
   changepoints with a linear computational cost." *Journal of the American
   Statistical Association*, 107(500), 1590-1598. — PELT changepoint detection,
   relevant for regime identification in the causal graph.

10. Law, A. M. & Kelton, W. D. (2000). *Simulation Modeling and Analysis*, 3rd
    ed. — Statistical methodology for simulation output analysis, including
    effective sample size and batch means.

11. Geyer, C. J. (1992). "Practical Markov chain Monte Carlo." *Statistical
    Science*, 7(4), 473-483. — Effective sample size estimation for correlated
    observations.

12. Bain, L. J. & Engelhardt, M. (1992). *Introduction to Probability and
    Mathematical Statistics*, 2nd ed. — Order-statistic confidence intervals
    for percentiles.

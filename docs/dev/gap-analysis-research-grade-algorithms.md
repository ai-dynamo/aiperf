<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Gap Analysis: Research-Grade Algorithms & Industry Standards

This document compares the AIPerf metrics accumulator's statistical methods and
metrics against published research, MLPerf Inference requirements, and
production SRE best practices. It identifies gaps, assesses severity, and
proposes concrete next steps.

## Current Inventory Summary

### What We Have (Strong)

| Capability | Implementation | Strength |
|---|---|---|
| **SLA-based Goodput** | `--goodput "time_to_first_token:100 inter_token_latency:3.40"` — per-metric thresholds with directionality and unit conversion (DistServe definition) | Research-grade |
| **Sweep-line algorithms** | 10 vectorized sweep functions producing 9 SWEEP_METRIC_SPECS: concurrency, throughput, prefill throughput, ICL-aware, tokens-in-flight, total, per-user throughput, per-user prefill throughput, generation concurrency, prefill concurrency | Ahead of industry |
| **Multi-signal steady-state detection** | 4-signal CUSUM + MSER-5 (concurrency, latency, TTFT, throughput) | Research-grade |
| **Polynomial histogram percentiles** | ~2.5x more accurate than Prometheus linear interpolation for server-side metrics | Novel |
| **Bootstrap confidence intervals** | Circular block resampling preserving temporal correlation | Research-grade |
| **Stationarity validation** | Batch means trend test via Spearman rank correlation (pure numpy) | Solid |
| **Time-weighted statistics** | Duration-weighted percentiles via CDF on step functions | Correct |
| **Columnar storage** | NumPy-backed ColumnStore with O(1) running sums, NaN-sparse | Performant |
| **70+ metrics** | RECORD, AGGREGATE, DERIVED types covering latency decomposition, throughput, tokens, HTTP trace, GPU telemetry, server metrics, usage discrepancy, thinking efficiency | Comprehensive |

### What We're Missing

The gaps below are organized by **severity** — how much their absence undermines
the credibility or utility of benchmark results.

---

## Critical Gaps

### 1. Tail Latency Percentiles (p99.9, p99.99)

**Current state:** Percentiles stop at p99. The `_PERCENTILE_QS` array in
`metric_dicts.py` is `[1, 5, 10, 25, 50, 75, 90, 95, 99]`.

**Why it matters:**
- MLPerf Inference reports p99 but production SRE universally tracks p99.9+
- Google's "Tail at Scale" (Dean & Barroso, 2013) established that p99.9 is
  where systems actually break under fan-out
- With 1,000 requests, p99 is based on ~10 tail observations, p99.9 on ~1.
  With 500 requests, p99.9 is based on ~0.5 observations — we can't
  meaningfully compute it. Users need to know this boundary
- vLLM benchmarks report p99; TensorRT-LLM reports p99 but their internal
  profiling goes to p99.9

**What to add:**
- p99.9, p99.99 fields on MetricResult (nullable — only populated when
  sample size is sufficient)
- Tail latency ratio: p99/p50 and p99.9/p50 (measures tail heaviness)
- Sample size warning when `count < 1000` for p99.9 or `count < 10000` for p99.99

**Effort:** Low — extend `_PERCENTILE_QS`, add fields to MetricResult, add
sample size guards.

---

### 2. Confidence Intervals on Per-Metric Results

**Current state:** Bootstrap CIs exist but only for steady-state boundary
positions and latency. Individual metric results (p50, p99, mean) have no
uncertainty quantification.

**Why it matters:**
- A benchmark reporting "p99 = 142ms" without a CI is a point estimate with
  unknown reliability
- MLPerf requires multiple runs and reports variability across runs
- The simulation output analysis literature (Law & Kelton, "Simulation Modeling
  and Analysis") emphasizes that point estimates without CIs are incomplete
- Users comparing two server configurations need to know if "142ms vs. 148ms"
  is signal or noise

**What to add — two tiers:**

**Tier 1 — Analytical CIs (cheap):**
- For means: CLT-based CI using effective sample size (accounts for
  autocorrelation)
- For percentiles: order-statistic CI using the binomial method
  (Hettmansperger & Sheather, 1986)
- For derived metrics (throughput, goodput): Delta method propagation

**Tier 2 — Bootstrap CIs (expensive, opt-in):**
- Extend existing bootstrap to compute CIs for arbitrary MetricResults
- Block bootstrap on the record array, re-run `_compute_results()` per iteration
- Report CI for every RECORD metric's p50, p95, p99

**Effort:** Tier 1 is medium (need effective sample size estimation via lag-1
autocorrelation). Tier 2 is medium (extend existing bootstrap infrastructure).

---

### 3. Effective Sample Size / Autocorrelation

**Current state:** No autocorrelation analysis. The batch means trend test in
`stationarity.py` detects trends but doesn't estimate serial correlation or
effective sample size.

**Why it matters:**
- Consecutive LLM requests are NOT independent — server-side batching, KV cache
  state, and GPU thermal throttling create temporal correlation
- When lag-1 autocorrelation ρ = 0.5, the effective sample size is
  n_eff ≈ n × (1-ρ)/(1+ρ) = n/3
- Standard CIs assume independence; without adjusting for autocorrelation,
  confidence intervals are too narrow by a factor that scales with ρ
- MLPerf addresses this by requiring multiple independent runs; we can do better
  by properly estimating n_eff within a single run

**What to add:**
- Lag-1 autocorrelation estimate for each RECORD metric (O(n), trivial numpy)
- Effective sample size: `n_eff = n * (1 - ρ) / (1 + ρ)` (Geyer, 1992)
- Report n_eff alongside count in MetricResult metadata
- Use n_eff for CI width calculation

**Effort:** Low — a few lines of numpy per metric. The impact on CI quality
is dramatic.

---

## Important Gaps

### 4. Coordinated Omission Awareness

**Current state:** No explicit handling. The credit-based flow control system
provides natural backpressure, which means AIPerf doesn't suffer from the
*classic* form of coordinated omission (where a benchmark skips sending requests
during a latency spike). However, the subtler form still applies.

**The subtle form:** Under concurrency-limited benchmarking, when the server is
slow, fewer requests complete per unit time, so fewer new requests start. The
latency distribution is biased toward the *response* perspective (how long did
completed requests take?) rather than the *user* perspective (how long did users
who would have submitted requests have to wait?).

**Why it matters:**
- Gil Tene's "How NOT to Measure Latency" (2013) is foundational in
  performance engineering
- wrk2 (the gold standard HTTP benchmark) explicitly corrects for this
- The correction matters most at high load where the server is near saturation
- Without correction, p99 can be understated by 2-10x at high utilization

**What to add:**
- **Intended send time** tracking: record when the credit was *issued* (the
  intended send time) alongside when the request was *actually sent*
- **Corrected latency**: `corrected = actual_end - intended_start` (includes
  queuing delay at the client)
- Report both "response latency" (current) and "service latency" (corrected)
  so users see both perspectives
- The credit_issued_ns field already exists in metadata — we just need to
  use it for a corrected latency metric

**Effort:** Low-Medium — the data is already captured (credit_issued_ns). Need
a new derived metric and documentation explaining the two perspectives.

---

### 5. Latency Decomposition / Queue-Theoretic Metrics

**Current state:** We measure end-to-end request_latency, TTFT, ITL, and
HTTP trace components (dns, connecting, waiting, receiving). But there's no
decomposition into standard queuing theory components.

**Why it matters:**
- Users need to know *why* latency is high — is it queuing delay? Prefill
  compute? Decode compute? Network?
- Little's Law (L = λW) provides a cross-validation: if measured concurrency,
  throughput, and latency don't satisfy L = λW, something is wrong with the
  measurement
- Server utilization = arrival_rate × service_time. Knowing utilization tells
  users how close they are to saturation

**What to add:**

**Latency decomposition:**
- Queue wait time: `credit_issued_ns → request_start_ns` (client-side queuing)
- Prefill + network (TTFT): `request_start_ns → first_response_timestamp`
  (already measured as `time_to_first_token`)
- Decode time: `request_latency - time_to_first_token` (per-record)
- Network overhead: from HTTP trace metrics (dns, connecting, waiting, receiving)

**Cross-validation:**
- Little's Law check: compare `effective_concurrency` (L) vs.
  `request_throughput × avg_latency` (λW). Report discrepancy %.
- This is a free sanity check — we already have all three quantities

**Effort:** Medium — latency decomposition needs new DERIVED metrics. Little's
Law check is a simple post-summarization validation.

---

### 6. Changepoint Detection Alternatives

**Current state:** CUSUM + MSER-5 is solid but represents one family of
approaches. The CUSUM target (time-weighted p95) is a reasonable heuristic but
not parameterized by statistical significance.

**What the literature offers:**
- **PELT** (Pruned Exact Linear Time — Killick et al., 2012): optimal
  multiple changepoint detection with a penalty term. O(n) average case.
  Could detect *multiple* regime changes, not just ramp-up/ramp-down
- **Bayesian Online Changepoint Detection** (Adams & MacKay, 2007): provides
  posterior probability of a changepoint at each observation. More principled
  than CUSUM's argmin heuristic
- **Heidelberger-Welch** (1983): the standard stationarity test in discrete
  event simulation. Tests whether a time series has reached stationarity using
  spectral methods. More rigorous than our batch means approach

**Assessment:** Our CUSUM + MSER-5 approach is well-suited to the two-boundary
problem (find ramp-up end + ramp-down start). PELT and BOCPD would be overkill
for this specific task but could be valuable for detecting mid-run regime changes
(e.g., server scaling events, thermal throttling). Heidelberger-Welch would
strengthen stationarity validation.

**Effort:** High — these are significant implementations. Best approached as
optional analyzers that can be plugged in alongside the existing detection.

---

## Moderate Gaps

### 7. Per-GPU Normalization

**Current state:** No throughput-per-GPU or latency-per-GPU metrics. GPU count
is not tracked in the metrics pipeline (GPU telemetry collects per-device data
but doesn't feed into inference metric normalization).

**Why it matters:**
- Comparing deployments with different GPU counts requires normalization
- MLPerf reports results per-accelerator
- "1000 tokens/sec on 8 GPUs" vs. "800 tokens/sec on 4 GPUs" — the second
  is more efficient but looks worse unnormalized

**What to add:**
- GPU count from telemetry (already collected)
- Derived metrics: `throughput_per_gpu`, `prefill_throughput_per_gpu`
- Only populated when GPU telemetry is enabled

**Effort:** Low — derived metrics from existing data.

---

### 8. Multi-Run Statistical Aggregation

**Current state:** Each benchmark run is independent. No built-in support for
aggregating results across multiple runs.

**Why it matters:**
- MLPerf requires multiple runs and reports mean ± std across runs
- Single-run results are subject to environmental variance (thermal state,
  OS scheduling, network conditions)
- Best practice is 3-5 runs with outlier detection

**What to add:**
- Multi-run results aggregation (mean, std, min, max across runs)
- Coefficient of variation (CV) to flag high-variance metrics
- Optional outlier detection (e.g., modified Z-score on per-run means)
- This is likely a CLI/export feature rather than an accumulator feature

**Effort:** Medium — needs a new post-processing pipeline that reads multiple
result files.

---

### 9. Spectral / Periodicity Analysis

**Current state:** Stationarity is tested via batch means trend (linear trend
only). No detection of periodic patterns.

**Why it matters:**
- GPU boost clock cycling creates periodic latency oscillations
- Kubernetes pod scaling events create step-change + periodic patterns
- Garbage collection pauses create periodic spikes
- Detecting periodicity helps users diagnose the *cause* of variance

**What to add:**
- Welch's method (pure numpy — `np.fft` is included in numpy) on windowed latency time series
- Dominant frequency + amplitude as metadata on steady-state summary
- Optional — this is a diagnostic tool, not a core metric

**Effort:** Medium — Welch's method is well-defined. Uses numpy's built-in FFT
(`np.fft.rfft`), so no additional dependency.

---

### 10. Formal Stationarity Testing

**Current state:** Batch means Spearman correlation detects linear trends.
Thresholds are hardcoded (|ρ| > 0.65, p < 0.05).

**What the literature recommends:**
- **Augmented Dickey-Fuller (ADF)** test: the standard unit root test for
  stationarity. Tests H0: series has a unit root (non-stationary)
- **KPSS test**: tests H0: series is stationary (complementary to ADF)
- **Heidelberger-Welch**: specifically designed for simulation output
- Running both ADF and KPSS gives a 2x2 classification:
  ADF rejects + KPSS doesn't → stationary (good)
  ADF doesn't + KPSS rejects → non-stationary (bad)
  Both reject → trend-stationary (remove trend)
  Neither rejects → inconclusive

**Assessment:** Our batch means approach is reasonable for detecting gross
trends. ADF/KPSS would be more rigorous but require either scipy or a
nontrivial pure-numpy implementation.

**Effort:** Medium-High for pure numpy; Low if scipy is acceptable as optional
dependency.

---

## Low-Priority / Nice-to-Have

### 11. HDR Histogram for Client-Side Percentiles

**Current state:** Raw numpy arrays with linear interpolation percentile
(`metric_result_from_array`). Keeps all observations in memory.

**HDR Histogram advantages:**
- Constant memory regardless of observation count
- Designed for latency measurement (logarithmic bucketing)
- Industry standard in Java perf tools (HdrHistogram by Gil Tene)

**Assessment:** For our use case (bounded run, all data in memory), raw numpy
arrays are actually *more* accurate than HDR Histogram. HDR trades accuracy for
constant memory. Since we already have the full array, we get exact percentiles.
This is NOT a gap — our approach is better for offline analysis.

**Recommendation:** No change for client-side metrics. Document that we use
exact percentiles (not bucketed approximations).

---

### 12. MLPerf Output Format Compliance

**Current state:** No MLPerf-formatted output.

**Why it matters:** MLPerf Inference is the industry standard benchmark for
submission and comparison. Supporting their output format would enable direct
comparability.

**Assessment:** MLPerf compliance is primarily about output format and run
rules (number of runs, warmup, etc.), not about the statistical methods
themselves. Our statistical methods are more sophisticated than what MLPerf
requires. Format compliance would be a thin export layer.

**Effort:** Low — formatting layer only.

---

## Summary Matrix

| Gap | Severity | Effort | Impact | Priority |
|---|---|---|---|---|
| **Tail percentiles (p99.9+)** | Critical | Low | Credibility with SRE audience | P0 |
| **Per-metric confidence intervals** | Critical | Medium | Statistical rigor | P1 |
| **Effective sample size / autocorrelation** | Critical | Low | Underpins CI accuracy | P0 |
| **Coordinated omission awareness** | Important | Low-Med | Accuracy at high load | P1 |
| **Latency decomposition + Little's Law** | Important | Medium | Diagnostic value | P2 |
| **Changepoint alternatives (PELT, BOCPD)** | Important | High | Incremental over CUSUM | P3 |
| **Per-GPU normalization** | Moderate | Low | Cross-deployment comparison | P2 |
| **Multi-run aggregation** | Moderate | Medium | MLPerf parity | P2 |
| **Spectral / periodicity analysis** | Moderate | Medium | Diagnostic value | P3 |
| **Formal stationarity (ADF/KPSS)** | Moderate | Medium-High | Incremental over batch means | P3 |
| **HDR Histogram (client-side)** | Low | N/A | Our approach is already better | Skip |
| **MLPerf output format** | Low | Low | Comparability | P3 |

---

## Recommended Implementation Order

**Phase 1 — Statistical Foundations (P0):**
1. Add p99.9, p99.99 to MetricResult (with sample size guards)
2. Implement lag-1 autocorrelation per RECORD metric
3. Compute effective sample size (n_eff)
4. Add tail latency ratio (p99/p50, p99.9/p50)

**Phase 2 — Uncertainty Quantification (P1):**
5. Analytical CIs on means (CLT with n_eff)
6. Order-statistic CIs on percentiles (binomial method)
7. Coordinated omission corrected latency (credit_issued_ns → request_end_ns)

**Phase 3 — Diagnostic Depth (P2):**
8. Latency decomposition derived metrics (queue, prefill, decode)
9. Little's Law cross-validation
10. Per-GPU throughput normalization
11. Multi-run aggregation (CLI feature)

**Phase 4 — Advanced Analysis (P3):**
12. Heidelberger-Welch stationarity test
13. PELT changepoint detection (optional analyzer)
14. Welch's spectral analysis (optional)
15. MLPerf output format

---

## References

- Dean, J. & Barroso, L.A. (2013). "The Tail at Scale." *Communications of the ACM*.
- Tene, G. (2013). "How NOT to Measure Latency." *Strange Loop Conference*.
- Law, A.M. & Kelton, W.D. (2000). *Simulation Modeling and Analysis*. McGraw-Hill.
- Killick, R. et al. (2012). "Optimal Detection of Changepoints with a Linear
  Computational Cost." *JASA*.
- Adams, R.P. & MacKay, D.J.C. (2007). "Bayesian Online Changepoint Detection."
  *arXiv:0710.3742*.
- Heidelberger, P. & Welch, P.D. (1983). "Simulation Run Length Control in the
  Presence of an Initial Transient." *Operations Research*.
- Hettmansperger, T.P. & Sheather, S.J. (1986). "Confidence Intervals Based on
  Interpolated Order Statistics." *Statistics & Probability Letters*.
- Geyer, C.J. (1992). "Practical Markov Chain Monte Carlo." *Statistical Science*.
- Zhong, Y. et al. (2024). "DistServe: Disaggregating Prefill and Decoding for
  Goodput-optimized Large Language Model Serving." *OSDI 2024*.

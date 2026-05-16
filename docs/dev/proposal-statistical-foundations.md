<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Proposal: Statistical Foundations — Tail Percentiles, Autocorrelation, Effective Sample Size

**Phase:** P0 (Statistical Foundations)
**Depends on:** None
**Enables:** Confidence intervals (P1), all downstream analysis

## Motivation

A benchmark that reports "p99 = 142ms" is making an implicit claim about the
tail of the latency distribution, but without two critical pieces of context:

1. **How deep into the tail can we actually see?** With 500 requests, p99 is
   based on ~5 observations. p99.9 would be based on ~0.5 observations — we
   literally can't compute it. Users need to know this boundary.

2. **How reliable is the estimate?** If consecutive requests are correlated
   (ρ = 0.5), the effective sample size is n/3. The point estimate is fine, but
   any confidence interval based on the nominal sample size would be 73% too
   narrow.

This proposal adds the mathematical foundations that underpin all subsequent
statistical improvements.

## Changes

### 1. Extend Percentile Computation

**File:** `src/aiperf/metrics/metric_dicts.py`

```python
# Current
_PERCENTILE_QS = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)

# Proposed
_PERCENTILE_QS = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 99.99], dtype=np.float64)
```

**File:** `src/aiperf/common/models/record_models.py` (MetricResult)

Add fields:
- `p999: float | None` — 99.9th percentile (None when count < 1000)
- `p9999: float | None` — 99.99th percentile (None when count < 10000)

**File:** `src/aiperf/common/models/export_models.py` (JsonMetricResult)

Add corresponding export fields.

**Guard logic in `metric_result_from_array()`:**

```python
p999 = pcts[9] if n >= 1000 else None
p9999 = pcts[10] if n >= 10000 else None
```

The threshold choices (1000 for p99.9, 10000 for p99.99) ensure at least ~1
observation contributes to the percentile estimate. Below these thresholds, the
value would be dominated by interpolation noise.

### 2. Tail Latency Ratio

Add two new DERIVED metrics:

**`tail_latency_ratio_99_50`** — p99 / p50 for request_latency
- Measures how heavy the tail is relative to the median
- Ratio > 3× typically indicates queuing effects or batch scheduling
- Ratio > 10× indicates severe tail behavior

**`tail_latency_ratio_999_50`** — p99.9 / p50 for request_latency
- Only computed when p99.9 is available (count >= 1000)
- The "tail at scale" metric — what users at the 99.9th percentile experience
  relative to the typical user

These are MetricFlags.NO_CONSOLE (available in exports, not cluttering the
default console table).

### 3. Lag-1 Autocorrelation

**File:** `src/aiperf/analysis/autocorrelation.py` (new)

```python
def lag1_autocorrelation(values: NDArray[np.float64]) -> float:
    """Compute lag-1 autocorrelation coefficient.

    Uses the standard estimator: corr(x[:-1], x[1:]).
    Returns 0.0 for arrays with fewer than 3 observations.
    """
    n = len(values)
    if n < 3:
        return 0.0
    x = values - np.mean(values)
    c0 = np.dot(x, x) / n          # variance (lag-0 autocovariance)
    c1 = np.dot(x[:-1], x[1:]) / n  # lag-1 autocovariance
    if c0 == 0:
        return 0.0
    return c1 / c0
```

This is the standard autocovariance estimator (Box-Jenkins). O(n) single pass.

### 4. Effective Sample Size

```python
def effective_sample_size(n: int, rho: float) -> float:
    """Compute effective sample size adjusting for serial correlation.

    Uses the Geyer (1992) initial positive sequence estimator simplified
    to lag-1: n_eff = n * (1 - ρ) / (1 + ρ).

    Clamped to [1, n] to handle edge cases (negative autocorrelation
    or numerical issues).
    """
    if rho >= 1.0:
        return 1.0
    if rho <= -1.0:
        return float(n)
    n_eff = n * (1 - rho) / (1 + rho)
    return max(1.0, min(float(n), n_eff))
```

**Where to compute it:** In `MetricsAccumulator._compute_results()` Phase 3,
after building the MetricResult from a record array. For each RECORD metric:

1. Compute lag-1 autocorrelation on the time-ordered values
2. Compute n_eff
3. Store as metadata on the MetricResult (not a separate metric — it
   qualifies the existing metric's reliability)

### 5. MetricResult Metadata Extension

**File:** `src/aiperf/common/models/record_models.py`

```python
class MetricResult(AIPerfBaseModel):
    # ... existing fields ...
    p999: float | None = Field(default=None, description="99.9th percentile")
    p9999: float | None = Field(default=None, description="99.99th percentile")
    tail_ratio_99_50: float | None = Field(default=None, description="p99/p50 ratio")
    lag1_autocorrelation: float | None = Field(
        default=None,
        description="Lag-1 autocorrelation of time-ordered values"
    )
    effective_sample_size: float | None = Field(
        default=None,
        description="Sample size adjusted for serial correlation"
    )
```

These fields are nullable to avoid bloating results for AGGREGATE/DERIVED
metrics where they don't apply.

## Impact on Existing Code

### metric_result_from_array()
- Extended percentile array (2 more values)
- New metadata computation (autocorrelation + n_eff)
- Backward compatible — new fields default to None

### MetricResult consumers
- Console exporters: no change (new fields excluded from default table)
- JSON/CSV exporters: automatically pick up new fields
- Realtime messages: no change (use MetricResult as-is)
- Plots: may want to add p99.9 to scatter plots when available

### Sweep-line statistics
- `compute_time_weighted_stats()` already returns p50, p90, p95, p99
- Extend SweepStats to include p999, p9999
- Same duration-weighted CDF approach, just two more searchsorted lookups

## Performance Impact

- Two extra percentile lookups: negligible (same sorted array, two more
  index computations)
- Lag-1 autocorrelation: one O(n) dot product per RECORD metric per
  summarization. With ~20 RECORD metrics and 100K records, this is ~20
  numpy dot products on 100K arrays = ~2ms total
- No impact on hot path (record ingestion)

## Testing

- Unit tests for lag1_autocorrelation with known sequences (AR(1) process)
- Unit tests for effective_sample_size with known ρ values
- Parametrized tests for p99.9/p99.99 sample size guards
- Verify tail_ratio_99_50 computation
- Verify JSON/CSV export includes new fields
- Verify console export excludes new fields by default

## Open Questions

1. **Should autocorrelation be computed on time-ordered or session-num-ordered
   values?** Time-ordered is more meaningful (captures temporal correlation),
   but requires a sort. session_num order is "free" (already indexed) and
   approximately time-ordered. Proposal: use session_num order (it's the
   arrival order, which is the natural ordering for correlation analysis).

2. **Should we offer higher-order autocorrelation (lag-2, lag-5)?** Probably
   not for V1. Lag-1 captures the dominant correlation structure. The initial
   positive sequence estimator (Geyer 1992) could be added later for more
   precise n_eff estimation with long-range dependence.

3. **Should p99.9/p99.99 be opt-in?** No — they're cheap to compute when
   sufficient data exists and provide immediate value. The None guard
   prevents misleading values at small sample sizes.

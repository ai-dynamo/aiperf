<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Proposal: Confidence Intervals on Per-Metric Results

**Phase:** P1 (Uncertainty Quantification)
**Depends on:** P0 (effective sample size, autocorrelation)
**Enables:** Rigorous A/B comparison between configurations

## Motivation

Every metric result in AIPerf is currently a point estimate. A user sees
"p99 latency = 142ms" but has no way to know whether the true p99 is between
140-144ms (reliable) or 120-180ms (noise).

This matters when users are making deployment decisions: "Should I use
configuration A (p99=142ms) or configuration B (p99=148ms)?" Without CIs, this
is a coin flip. With CIs, if A is 142 ± 5ms and B is 148 ± 4ms, the difference
is likely real.

## Approach: Analytical CIs (Tier 1)

Analytical CIs are cheap (O(1) per metric after the sort in
`metric_result_from_array`) and don't require multiple bootstrap iterations.
They should be computed by default for all RECORD metrics.

### CI for Means

The standard CI for a mean with correlated observations:

```
CI(mean) = x̄ ± z_{α/2} × s / √n_eff
```

Where:
- `x̄` = sample mean (already computed)
- `s` = sample standard deviation (already computed)
- `n_eff` = effective sample size from P0 (adjusts for autocorrelation)
- `z_{α/2}` = 1.96 for 95% CI

This is the textbook CLT-based interval, but using n_eff instead of n to
account for serial correlation. When ρ = 0.5, the CI is ~73% wider than the
naive version — which is the correct behavior.

### CI for Percentiles

For the qth percentile of an ordered sample of size n, the binomial method
gives exact coverage:

```
The qth percentile lies between order statistics X_{(j)} and X_{(k)} where:
j = ⌊nq - z_{α/2} × √(nq(1-q))⌋
k = ⌈nq + z_{α/2} × √(nq(1-q))⌉ + 1
```

This is the Hettmansperger & Sheather (1986) approach. Since the array is
already sorted in `metric_result_from_array()`, looking up X_{(j)} and X_{(k)}
is O(1).

For correlated observations, we substitute n_eff for n in the formula. This
widens the CI appropriately.

**Important:** When j < 0 or k > n, the CI cannot be computed at the
requested confidence level — return None. This naturally handles the case
where p99.9 with 500 observations has no meaningful CI.

### CI for Derived Metrics

For derived metrics like throughput = total_tokens / duration, the Delta
method propagates uncertainty:

```
Var(f(X,Y)) ≈ (∂f/∂x)² × Var(X) + (∂f/∂y)² × Var(Y)
```

For throughput = tokens / duration:
- `∂f/∂tokens = 1/duration`
- `∂f/∂duration = -tokens/duration²`

This requires knowing the variance of both numerator and denominator. Since
total_tokens is an AGGREGATE (SUM) metric and duration is derived from
timestamps, both have known variances from the CLT.

**Simplification for V1:** Only provide CIs for RECORD metrics (means and
percentiles). Derived metric CIs via Delta method can be Phase 2.

## Data Model

### ConfidenceInterval

```python
@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Confidence interval for a metric statistic."""
    lower: float
    upper: float
    confidence_level: float  # e.g. 0.95
    method: str  # "clt_neff", "binomial_neff", "bootstrap"
```

### MetricResult Extension

```python
class MetricResult(AIPerfBaseModel):
    # ... existing fields ...

    # Confidence intervals (None when not computable)
    ci_mean: ConfidenceInterval | None = Field(
        default=None,
        description="Confidence interval for the mean"
    )
    ci_p50: ConfidenceInterval | None = Field(
        default=None,
        description="Confidence interval for the median"
    )
    ci_p99: ConfidenceInterval | None = Field(
        default=None,
        description="Confidence interval for p99"
    )
    ci_p999: ConfidenceInterval | None = Field(
        default=None,
        description="Confidence interval for p99.9"
    )
```

### Export Format

JSON export with CIs:

```json
{
  "tag": "request_latency",
  "unit": "ns",
  "avg": 142000000,
  "p99": 312000000,
  "count": 5000,
  "effective_sample_size": 1823,
  "lag1_autocorrelation": 0.47,
  "ci_mean": {
    "lower": 138200000,
    "upper": 145800000,
    "confidence_level": 0.95,
    "method": "clt_neff"
  },
  "ci_p99": {
    "lower": 295000000,
    "upper": 338000000,
    "confidence_level": 0.95,
    "method": "binomial_neff"
  }
}
```

## Implementation

### New file: `src/aiperf/analysis/confidence.py`

```python
def ci_mean(
    mean: float,
    std: float,
    n_eff: float,
    confidence: float = 0.95,
) -> ConfidenceInterval | None:
    """CLT-based CI for the mean using effective sample size."""

def ci_percentile(
    sorted_values: NDArray[np.float64],
    q: float,  # e.g. 0.99
    n_eff: float,
    confidence: float = 0.95,
) -> ConfidenceInterval | None:
    """Binomial method CI for a percentile using effective sample size."""
```

### Integration point: `metric_result_from_array()`

After computing the MetricResult, compute CIs:

```python
rho = lag1_autocorrelation(clean)  # from P0
n_eff = effective_sample_size(n, rho)

result.ci_mean = ci_mean(result.avg, result.std, n_eff)
result.ci_p50 = ci_percentile(clean, 0.50, n_eff)
result.ci_p99 = ci_percentile(clean, 0.99, n_eff)
if result.p999 is not None:
    result.ci_p999 = ci_percentile(clean, 0.999, n_eff)
```

## Performance Impact

- z-score lookup: constant
- CLT CI: O(1) per metric (already have mean, std, n_eff)
- Binomial CI: O(1) per percentile (index into already-sorted array)
- Total: ~0.1ms for all CIs across all RECORD metrics

## Console Display

CIs should NOT clutter the default console table. Options:

1. **Opt-in flag:** `--show-confidence-intervals` adds a CI column
2. **Summary line:** Below the table: "95% CIs available in JSON export"
3. **Steady-state exporter:** Already has a rich metadata section — add CIs
   to the steady-state JSON output

Recommendation: Option 2 (summary line) + always include in JSON/CSV exports.

## Why Not Bootstrap for Everything?

Bootstrap CIs (Tier 2) are more robust but expensive:
- 500 iterations × re-running `_compute_results()` = 500× the computation
- For 100K records, this is ~seconds to minutes
- The analytical CIs above are ~0.1ms and correct for the vast majority of
  cases (Gaussian-ish metrics with moderate autocorrelation)

Reserve bootstrap for:
- Steady-state boundary positions (already implemented)
- Cases where analytical CIs are known to be unreliable (heavily skewed
  distributions, very small samples)
- Future: per-metric bootstrap as opt-in via `--bootstrap-metrics`

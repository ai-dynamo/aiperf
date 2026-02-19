# Multi-Run Confidence Reporting

## Overview

Multi-run confidence reporting allows you to run the same benchmark configuration multiple times to quantify measurement variance, assess repeatability, and compute confidence intervals for key metrics. This helps answer the critical question: **"Is this performance difference real or just noise?"**

## What is Confidence Reporting?

When you run a single benchmark, the results can vary due to:
- System jitter (GPU clocks, background tasks)
- Network variance
- Server internal scheduling and batching dynamics
- Periodic stalls or transient errors

By running multiple trials of the same benchmark, you can:
- **Quantify variance**: Understand how much results vary between runs
- **Assess repeatability**: Determine if your measurements are stable
- **Compute confidence intervals**: Get honest uncertainty estimates
- **Make informed decisions**: Know if performance differences are statistically meaningful

## UI Behavior in Multi-Run Mode

Multi-run mode automatically uses the `simple` UI by default for the best experience. The dashboard UI is not supported due to terminal control limitations.

### Default UI Selection

When using `--num-profile-runs > 1`, AIPerf automatically sets `--ui simple` unless you explicitly specify a different UI:

```bash
# These are equivalent - simple UI is auto-selected
aiperf profile --num-profile-runs 5 ...
aiperf profile --num-profile-runs 5 --ui simple ...
```

You'll see an informational message:
```
Multi-run mode: UI automatically set to 'simple' (use '--ui none' to disable UI output)
```

### Supported UI Options

**Simple UI (Default)**
```bash
aiperf profile \
  --num-profile-runs 5 \
  ...
```
Shows progress bars for each run - works well with multi-run mode.

**No UI**
```bash
aiperf profile \
  --num-profile-runs 5 \
  --ui none \
  ...
```
Minimal output, fastest execution - ideal for automated runs or CI/CD pipelines.

### Dashboard UI Not Supported

The dashboard UI (`--ui dashboard`) is incompatible with multi-run mode due to terminal control constraints. If you explicitly try to use it, you'll get an error:

```bash
aiperf profile --num-profile-runs 5 --ui dashboard ...
```

```
ValueError: Dashboard UI is not supported with multi-run mode (--num-profile-runs > 1)
due to terminal control limitations. Please use '--ui simple' or '--ui none' instead.
```

This is a fundamental architectural limitation - Textual requires exclusive terminal control, which isn't possible when the orchestrator coordinates multiple subprocess runs.

### For Live Dashboard Monitoring

If you need live dashboard updates, run benchmarks individually:
```bash
# Run each benchmark separately with live dashboard
aiperf profile --output-artifact-dir ./run1 --ui dashboard ...
aiperf profile --output-artifact-dir ./run2 --ui dashboard ...
aiperf profile --output-artifact-dir ./run3 --ui dashboard ...
```

## Basic Usage

### Simple Multi-Run Benchmark

Run the same benchmark 5 times:

```bash
aiperf profile \
  --model llama-3-8b \
  --endpoint-type openai_chat \
  --url http://localhost:8000/v1/chat/completions \
  --num-profile-runs 5 \
  --concurrency 10 \
  --num-prompts 1000
```

### With Custom Confidence Level

Use 99% confidence intervals instead of the default 95%:

```bash
aiperf profile \
  --model llama-3-8b \
  --endpoint-type openai_chat \
  --url http://localhost:8000/v1/chat/completions \
  --num-profile-runs 5 \
  --confidence-level 0.99 \
  --concurrency 10 \
  --num-prompts 1000
```

### With Cooldown Between Runs

Add a 10-second cooldown between runs to reduce correlation:

```bash
aiperf profile \
  --model llama-3-8b \
  --endpoint-type openai_chat \
  --url http://localhost:8000/v1/chat/completions \
  --num-profile-runs 5 \
  --profile-run-cooldown-seconds 10.0 \
  --concurrency 10 \
  --num-prompts 1000
```

## Output Structure

When `--num-profile-runs > 1`, AIPerf creates a hierarchical output structure with an auto-generated directory name:

```
artifacts/
  llama-3-8b-openai-chat-concurrency_10/
    profile_runs/
      run_0001/
        profile_export_aiperf.json
        profile_export_aiperf.csv
        profile_export.jsonl
        inputs.json
      run_0002/
        ...
      run_0005/
        ...
    aggregate/
      profile_export_aiperf_aggregate.json
      profile_export_aiperf_aggregate.csv
```

### Auto-Generated Directory Name

The directory name is automatically generated based on your benchmark configuration:
- **Model name**: e.g., `llama-3-8b` (from `--model`)
- **Service kind and endpoint type**: e.g., `openai-chat` (from `--endpoint-type`)
- **Stimulus**: e.g., `concurrency_10` (from `--concurrency`) or `request_rate_100` (from `--request-rate`)

Examples:
- `artifacts/gpt-4-openai-chat-concurrency_50/`
- `artifacts/mistral-7b-openai-completions-request_rate_10/`
- `artifacts/llama-2-13b-nim-embeddings-concurrency_20/`

### Per-Run Artifacts

Each run's artifacts are stored in separate directories (`run_0001`, `run_0002`, etc.) and include:
- `profile_export_aiperf.json` - Complete metrics for that run
- `profile_export_aiperf.csv` - CSV export for that run
- `profile_export.jsonl` - Per-request records
- `inputs.json` - Input prompts used

This allows you to:
- Debug outliers by examining specific runs
- Compare individual runs
- Investigate anomalies

### Aggregate Artifacts

The `aggregate/` directory contains statistics computed across all runs:
- `profile_export_aiperf_aggregate.json` - Aggregated statistics
- `profile_export_aiperf_aggregate.csv` - Tabular view of aggregated metrics

## Understanding Aggregate Statistics

For each metric, the aggregate output includes:

- **mean**: Average value across all runs
- **std**: Standard deviation (measure of spread)
- **min**: Minimum value observed
- **max**: Maximum value observed
- **cv**: Coefficient of Variation (normalized variability)
- **se**: Standard Error (uncertainty in the mean)
- **ci_low, ci_high**: Confidence interval bounds
- **t_critical**: t-distribution critical value used

### Example Aggregate Output

```json
{
  "metadata": {
    "aggregation_type": "confidence",
    "num_profile_runs": 5,
    "num_successful_runs": 5,
    "confidence_level": 0.95,
    "run_labels": ["run_0001", "run_0002", "run_0003", "run_0004", "run_0005"]
  },
  "metrics": {
    "request_throughput_avg": {
      "mean": 255.4,
      "std": 12.3,
      "min": 240.1,
      "max": 270.2,
      "cv": 0.048,
      "se": 5.5,
      "ci_low": 243.2,
      "ci_high": 267.6,
      "t_critical": 2.776,
      "unit": "requests/sec"
    },
    "ttft_p99_ms": {
      "mean": 152.7,
      "std": 12.4,
      "min": 138.2,
      "max": 168.9,
      "cv": 0.081,
      "se": 5.55,
      "ci_low": 140.3,
      "ci_high": 165.1,
      "t_critical": 2.776,
      "unit": "ms"
    }
  }
}
```

## Detailed Metric Definitions

This section provides detailed mathematical definitions for each aggregate statistic computed across multiple runs.

### Mean (μ)

**Type:** Aggregate Statistic

The average value of the metric across all successful runs.

**Formula:**
```python
mean = sum(values) / n
```

**Example:**
If TTFT p99 values across 5 runs are [150ms, 152ms, 148ms, 155ms, 151ms], the mean is 151.2ms.

---

### Standard Deviation (σ)

**Type:** Aggregate Statistic

Measures the spread or dispersion of metric values across runs. Uses sample standard deviation (N-1 degrees of freedom).

**Formula:**
```python
std = sqrt(sum((x - mean)^2) / (n - 1))
```

**Example:**
For the TTFT values above, std ≈ 2.59ms, indicating low variability.

---

### Minimum

**Type:** Aggregate Statistic

The smallest value observed across all runs.

**Example:**
For the TTFT values above, min = 148ms.

---

### Maximum

**Type:** Aggregate Statistic

The largest value observed across all runs.

**Example:**
For the TTFT values above, max = 155ms.

---

### Coefficient of Variation (CV)

**Type:** Aggregate Statistic

A normalized measure of variability, expressed as a ratio (not percentage). Useful for comparing variability across metrics with different scales.

**Formula:**
```python
cv = std / mean
```

**Notes:**
- Returns `inf` when mean is zero (division by zero)
- Lower CV indicates more consistent measurements

**Example:**
For the TTFT values above, CV = 2.59 / 151.2 ≈ 0.017 (1.7%), indicating excellent repeatability.

---

### Standard Error (SE)

**Type:** Aggregate Statistic

Measures the uncertainty in the estimated mean. Decreases as sample size increases.

**Formula:**
```python
se = std / sqrt(n)
```

**Example:**
For the TTFT values above with n=5, SE = 2.59 / sqrt(5) ≈ 1.16ms.

**Notes:**
- Smaller SE indicates more precise estimate of the true mean
- SE decreases proportionally to 1/sqrt(n)

---

### Confidence Interval (CI)

**Type:** Aggregate Statistic

A range that likely contains the true population mean with a specified confidence level (default 95%).

**Formula:**
```python
ci_low = mean - t_critical * se
ci_high = mean + t_critical * se
```

Where `t_critical` is the critical value from the t-distribution with (n-1) degrees of freedom.

**Example:**
For the TTFT values above with 95% confidence:
- t_critical ≈ 2.776 (for n=5, df=4)
- CI = `[151.2 - 2.776 * 1.16, 151.2 + 2.776 * 1.16]` = [148.0ms, 154.4ms]

We're 95% confident the true mean TTFT is between 148.0ms and 154.4ms.

**Notes:**
- Uses t-distribution (not normal) for mathematically precise critical values
- Confidence level configurable via `--confidence-level` (default 0.95)
- CI width decreases with more runs (larger n)

---

### t-Critical Value

**Type:** Aggregate Statistic

The critical value from the t-distribution used to compute confidence intervals. Depends on sample size and confidence level.

**Formula:**
```python
t_critical = t.ppf(1 - alpha/2, df)
```

Where:
- `alpha = 1 - confidence_level`
- `df = n - 1` (degrees of freedom)
- `t.ppf` is the percent point function (inverse CDF) of the t-distribution

**Example:**
- For n=5 runs and 95% confidence: t_critical ≈ 2.776
- For n=10 runs and 95% confidence: t_critical ≈ 2.262
- For n=5 runs and 99% confidence: t_critical ≈ 4.604

**Notes:**
- Computed using scipy.stats.t.ppf() for mathematical precision
- Larger sample sizes have smaller t-critical values (approach normal distribution)
- Higher confidence levels have larger t-critical values (wider intervals)

---

## Interpreting Results

### Coefficient of Variation (CV)

The CV is a normalized measure of variability: `CV = std / mean`

**Interpretation Guidelines:**

- **CV < 0.05 (5%)**: Excellent repeatability, low noise
  - Results are very stable
  - High confidence in measurements
  - Small differences are likely meaningful

- **CV 0.05-0.10 (5-10%)**: Good repeatability, acceptable noise
  - Results are reasonably stable
  - Moderate confidence in measurements
  - Medium-sized differences are likely meaningful

- **CV 0.10-0.20 (10-20%)**: Fair repeatability, moderate variance
  - Results show noticeable variation
  - Consider running more trials
  - Only large differences are clearly meaningful

- **CV > 0.20 (>20%)**: High variability
  - Results are unstable
  - Investigate sources of variance
  - Increase number of runs or use cooldown
  - Be cautious about drawing conclusions

**Example:**
```
ttft_p99_ms: mean=152.7ms, cv=0.081 (8.1%)
```
This indicates good repeatability. The p99 TTFT varies by about 8% between runs, which is acceptable for most use cases.

### Confidence Intervals (CI)

The confidence interval tells you: **"If we repeated this experiment many times, X% of the time the true mean would fall in this range."**

**Interpretation Guidelines:**

- **Narrow CI**: High precision, confident in the estimate
  - The true mean is likely very close to the measured mean
  - Small sample size may still be sufficient

- **Wide CI**: Lower precision, more uncertainty
  - The true mean could be anywhere in a broad range
  - Consider increasing `--num-profile-runs`
  - May need to investigate sources of variance

**Example:**
```
ttft_p99_ms: mean=152.7ms, 95% CI=[140.3, 165.1]
```
We're 95% confident the true mean p99 TTFT is between 140.3ms and 165.1ms. The 24.8ms width suggests moderate uncertainty with 5 runs.

### Comparing Configurations

When comparing two configurations, consider:

1. **Do the confidence intervals overlap?**
   - No overlap → Strong evidence of a real difference
   - Partial overlap → Likely a real difference, but less certain
   - Complete overlap → Difference may not be meaningful

2. **Is the difference larger than the CV?**
   - If Config A has mean=100ms (CV=10%) and Config B has mean=120ms
   - Difference is 20%, which is 2× the CV
   - This suggests a real difference

**Example:**
```
Config A: mean=150ms, CI=[145, 155]
Config B: mean=180ms, CI=[175, 185]
```
No overlap in CIs → Strong evidence that Config B is slower.

## When to Use More Runs

### Recommended Number of Runs

- **Quick check**: 3 runs
  - Minimum for basic statistics
  - Good for initial exploration

- **Standard benchmarking**: 5 runs
  - Good balance of time and precision
  - Recommended for most use cases

- **High-precision**: 10 runs
  - When you need very precise estimates
  - When comparing small differences
  - When variance is high

### Signs You Need More Runs

1. **High CV (>10%)**: More runs will reduce uncertainty
2. **Wide confidence intervals**: More runs will narrow the CI
3. **Overlapping CIs when comparing**: More runs may separate them
4. **Inconsistent results**: More runs will clarify the true mean

## Workload Consistency

**Important**: All runs use the same workload (prompts, ordering, scheduling) to ensure fair comparison.

AIPerf automatically:
- Sets `--random-seed 42` if not specified (for multi-run consistency)
- Uses the same prompts in the same order for all runs
- Uses the same request timing patterns

This ensures that observed variance is due to real system noise, not artificial differences in the workload.

### Manual Seed Control

You can specify your own seed:

```bash
aiperf profile \
  --num-profile-runs 5 \
  --random-seed 123 \
  ...
```

All 5 runs will use seed 123, ensuring identical workloads.

## Warmup Behavior

When using multi-run with warmup:

```bash
aiperf profile \
  --num-profile-runs 5 \
  --warmup-request-count 100 \
  ...
```

- By default, warmup runs **once** before the first profile run only
- Subsequent profile runs (2-5) measure steady-state performance without warmup
- Warmup metrics are automatically excluded from results
- Use `--profile-run-disable-warmup-after-first false` to run warmup before each run (useful for long cooldown periods)

This default behavior is more efficient and provides more accurate aggregate statistics by measuring steady-state performance.

## Troubleshooting

### High Variance (CV > 20%)

**Possible causes:**
- System is under load from other processes
- Network instability
- Server batching/scheduling dynamics
- Insufficient warmup

**Solutions:**
1. Use `--profile-run-cooldown-seconds` to reduce correlation
2. Increase `--warmup-request-count` to stabilize server
3. Run benchmarks during low-load periods
4. Investigate server configuration
5. Increase `--num-profile-runs` to better characterize variance

### Failed Runs

If some runs fail, AIPerf will:
- Continue with remaining runs
- Compute statistics over successful runs only
- Report failed runs in aggregate metadata

**Example output:**
```json
{
  "metadata": {
    "num_profile_runs": 5,
    "num_successful_runs": 4,
    "failed_runs": [
      {"label": "run_0003", "error": "Connection timeout"}
    ]
  }
}
```

### Insufficient Successful Runs

If fewer than 2 runs succeed, you'll get an error:
```
ValueError: Insufficient successful runs for confidence intervals.
Got 1 successful run(s), but need at least 2.
Consider increasing --num-profile-runs or investigating why runs are failing.
```

**Solution**: Increase `--num-profile-runs` or fix the underlying issue causing failures.

### Very Long Benchmark Times

If `--num-profile-runs` is large and each run takes a long time:

1. **Reduce run duration**:
   - Use fewer prompts: `--num-prompts 500` instead of `--num-prompts 5000`
   - Use shorter prompts: `--synthetic-input-tokens-mean 100`

2. **Use cooldown strategically**:
   - Only add cooldown if you see high correlation between runs
   - Start without cooldown and add if needed

3. **Run overnight**:
   - For production validation with many runs

## Best Practices

### 1. Start with 5 Runs

This provides a good balance of precision and time investment.

### 2. Check CV First

After running, look at the CV for your key metrics:
- CV < 10%: Results are trustworthy
- CV > 10%: Consider more runs or investigate variance

### 3. Use Warmup

Always use warmup to eliminate cold-start effects:
```bash
--warmup-request-count 100
```

### 4. Set Random Seed for Reproducibility

For reproducible experiments:
```bash
--random-seed 42
```

### 5. Document Your Configuration

Save your command and results for future reference:
```bash
aiperf profile ... | tee benchmark_log.txt
```

### 6. Compare Apples to Apples

When comparing configurations:
- Use the same `--num-profile-runs`
- Use the same `--random-seed`
- Use the same workload parameters

## Advanced Usage

### Combining with Other Features

Multi-run works with all AIPerf features:

**With GPU telemetry:**
```bash
aiperf profile \
  --num-profile-runs 5 \
  --gpu-telemetry-url http://localhost:9400/metrics \
  ...
```

**With server metrics:**
```bash
aiperf profile \
  --num-profile-runs 5 \
  --server-metrics-url http://localhost:8000/metrics \
  ...
```

**With trace replay:**
```bash
aiperf profile \
  --num-profile-runs 5 \
  --trace-file my_trace.jsonl \
  ...
```

### Analyzing Results Programmatically

Load aggregate results in Python:

```python
import json

with open('artifacts/aggregate/profile_export_aiperf_aggregate.json') as f:
    agg = json.load(f)

# Get throughput statistics
throughput = agg['metrics']['request_throughput_avg']
print(f"Mean: {throughput['mean']:.2f} req/s")
print(f"CV: {throughput['cv']:.1%}")
print(f"95% CI: [{throughput['ci_low']:.2f}, {throughput['ci_high']:.2f}]")
```

## Summary

Multi-run confidence reporting helps you:
- ✅ Quantify measurement variance
- ✅ Assess repeatability with CV
- ✅ Compute confidence intervals
- ✅ Make statistically informed decisions
- ✅ Debug outliers with per-run artifacts

**Quick Start:**
```bash
aiperf profile --num-profile-runs 5 [other options]
```

**Key Metrics:**
- **CV < 10%**: Good repeatability
- **Narrow CI**: High precision
- **No CI overlap**: Strong evidence of difference

For more details, see:
- [CLI Options](../cli_options.md) - Full parameter reference
- [Metrics Reference](../metrics_reference.md) - Detailed metric descriptions
- [Architecture](../architecture.md) - How multi-run orchestration works

# Parameter Sweeping

## Overview

Parameter sweeping allows you to benchmark across multiple parameter values (e.g., concurrency levels) in a single command. This enables systematic performance characterization, identification of optimal configurations, and understanding of how your system scales.

Instead of running separate benchmarks for each concurrency level, parameter sweeping automates the process and provides comprehensive analysis including:
- Pareto optimal configurations (best trade-offs)
- Best configurations for key metrics (throughput, latency)
- Confidence intervals when combined with multi-run mode
- Organized hierarchical output structure

## What is Parameter Sweeping?

When you run a parameter sweep, AIPerf:
1. **Executes benchmarks** at each parameter value sequentially
2. **Organizes results** hierarchically for easy navigation
3. **Computes aggregate statistics** across all values
4. **Identifies optimal configurations** based on your objectives
5. **Analyzes performance** across parameter combinations

This helps answer questions like:
- **"What's the optimal concurrency for my workload?"**
- **"How does throughput scale with concurrency?"**
- **"Where does latency start to degrade?"**
- **"What's the best trade-off between throughput and latency?"**

## UI Behavior in Parameter Sweep Mode

Parameter sweep mode automatically uses the `simple` UI by default for the best experience. The dashboard UI is not supported due to terminal control limitations.

### Default UI Selection

When using `--concurrency` with a list of values, AIPerf automatically sets `--ui simple` unless you explicitly specify a different UI:

```bash
# These are equivalent - simple UI is auto-selected
```bash
aiperf profile --concurrency 10,20,30,40 ...
aiperf profile --concurrency 10,20,30,40 --ui simple ...
```

You'll see an informational message:
```text
Parameter sweep mode: UI automatically set to 'simple' (use '--ui none' to disable UI output)
```

### Supported UI Options

**Simple UI (Default)**
```bash
aiperf profile \
```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --ui simple \
  ...
```
Shows progress bars for each sweep value - works well with parameter sweeps.

**No UI**
```bash
aiperf profile \
  --concurrency 10,20,30,40 \
```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --ui none \
  ...
```
Minimal output, fastest execution - ideal for automated runs or CI/CD pipelines.

### Dashboard UI Not Supported

The dashboard UI (`--ui dashboard`) is incompatible with parameter sweep mode. If you explicitly try to use it, you'll get an error:

```bash
aiperf profile --concurrency 10,20,30,40 --ui dashboard ...
```

```text
ValueError: Dashboard UI is not supported with parameter sweeps
due to terminal control limitations. Please use '--ui simple' or '--ui none' instead.
```

## Basic Usage


### Simple Concurrency Sweep

Sweep across multiple concurrency values:

```bash
aiperf profile \
  --model llama-3-8b \
  --endpoint-type chat \
  --url http://localhost:8000/v1/chat/completions \
  --concurrency 10,20,30,40 \
  --num-prompts 1000
```

This runs 4 separate benchmarks with concurrency values of 10, 20, 30, and 40.

### Output Structure (Single Sweep)

When running a simple sweep without confidence runs:

```text
artifacts/
  llama-3-8b-openai-chat-concurrency_sweep/
    concurrency_10/
      profile_export_aiperf.json
      profile_export_aiperf.csv
      profile_export.jsonl
      inputs.json
    concurrency_20/
      profile_export_aiperf.json
      ...
    concurrency_30/
      ...
    concurrency_40/
      ...
    sweep_aggregate/
      profile_export_aiperf_sweep.json
      profile_export_aiperf_sweep.csv
```

Each concurrency value has its own directory with complete benchmark results. The `sweep_aggregate/` directory contains analysis across all values.

## Combining Sweep with Confidence Reporting

You can combine parameter sweeping with multi-run confidence reporting to quantify variance at each parameter value. This provides the most comprehensive analysis.

```bash
aiperf profile \
  --model llama-3-8b \
  --endpoint-type chat \
  --url http://localhost:8000/v1/chat/completions \
  --concurrency 10,20,30,40 \
  --num-profile-runs 5 \
  --num-prompts 1000
```

### Understanding Sweep Modes

When combining sweeps with confidence runs, you can choose between two execution modes:

#### Repeated Mode (Default)

Executes the full sweep pattern multiple times. This preserves dynamic system behavior as load changes.

**Execution pattern** with `--concurrency 10,20,30 --num-profile-runs 5`:
```text
Trial 1: [10 → 20 → 30]
Trial 2: [10 → 20 → 30]
Trial 3: [10 → 20 → 30]
Trial 4: [10 → 20 → 30]
Trial 5: [10 → 20 → 30]
```

**Use when:**
- You want to capture how the system behaves as load changes
- You're testing dynamic scaling or batching behavior
- You want to measure real-world performance patterns

#### Independent Mode

Executes all trials at each sweep value before moving to the next. This isolates each parameter value for independent measurement.

**Execution pattern** with `--concurrency 10,20,30 --num-profile-runs 5`:
```text
Value 10: [trial1, trial2, trial3, trial4, trial5]
Value 20: [trial1, trial2, trial3, trial4, trial5]
Value 30: [trial1, trial2, trial3, trial4, trial5]
```

**Use when:**
- You want to isolate each concurrency level
- You're measuring steady-state performance at each value
- You want to minimize correlation between different parameter values

**To use independent mode:**
```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --num-profile-runs 5 \
  --parameter-sweep-mode independent \
  ...
```

### Output Structure (Sweep + Confidence)

When combining sweep with confidence runs (repeated mode):

```text
artifacts/
  llama-3-8b-openai-chat-concurrency_sweep/
    profile_runs/
      trial_0001/
        concurrency_10/
          profile_export_aiperf.json
          ...
        concurrency_20/
          ...
        concurrency_30/
          ...
        concurrency_40/
          ...
      trial_0002/
        concurrency_10/
        concurrency_20/
        concurrency_30/
        concurrency_40/
      ...
      trial_0005/
        ...
    aggregate/
      concurrency_10/
        profile_export_aiperf_aggregate.json  # Confidence stats across 5 trials
        profile_export_aiperf_aggregate.csv
      concurrency_20/
        ...
      concurrency_30/
        ...
      concurrency_40/
        ...
    sweep_aggregate/
      profile_export_aiperf_sweep.json      # Comparison across concurrency values
      profile_export_aiperf_sweep.csv
```

**Structure explanation:**
- `profile_runs/trial_NNNN/`: Each trial's raw results for all sweep values
- `aggregate/concurrency_VV/`: Confidence statistics for each concurrency value across all trials
- `sweep_aggregate/`: Cross-value comparison and analysis

## Understanding Sweep Aggregates

The sweep aggregate output provides comprehensive analysis across all parameter values.

### Example Sweep Aggregate

```json
{
  "metadata": {
    "aggregation_type": "sweep",
    "sweep_parameters": [
      {
        "name": "concurrency",
        "values": [10, 20, 30, 40]
      }
    ],
    "num_combinations": 4,
    "num_trials_per_value": 5,
    "sweep_mode": "repeated",
    "confidence_level": 0.95
  },
  "per_combination_metrics": [
    {
      "parameters": {
        "concurrency": 10
      },
      "metrics": {
        "request_throughput_avg": {
          "mean": 95.2,
          "std": 3.1,
          "min": 91.5,
          "max": 99.0,
          "cv": 0.033,
          "ci_low": 91.6,
          "ci_high": 98.8,
          "unit": "requests/sec"
        },
        "ttft_p99_ms": {
          "mean": 125.4,
          "std": 8.2,
          "cv": 0.065,
          "ci_low": 115.7,
          "ci_high": 135.1,
          "unit": "ms"
        }
      }
    },
    {
      "parameters": {
        "concurrency": 20
      },
      "metrics": {
        "request_throughput_avg": {
          "mean": 175.8,
          "std": 5.4,
          "cv": 0.031,
          "unit": "requests/sec"
        },
        "ttft_p99_ms": {
          "mean": 145.2,
          "std": 10.1,
          "cv": 0.070,
          "unit": "ms"
        }
      }
    },
    {
      "parameters": {
        "concurrency": 30
      },
      "metrics": {
        "request_throughput_avg": {
          "mean": 245.3,
          "std": 8.2,
          "cv": 0.033,
          "unit": "requests/sec"
        },
        "ttft_p99_ms": {
          "mean": 180.5,
          "std": 12.4,
          "cv": 0.069,
          "unit": "ms"
        }
      }
    },
    {
      "parameters": {
        "concurrency": 40
      },
      "metrics": {
        "request_throughput_avg": {
          "mean": 255.1,
          "std": 12.3,
          "cv": 0.048,
          "unit": "requests/sec"
        },
        "ttft_p99_ms": {
          "mean": 285.7,
          "std": 18.5,
          "cv": 0.065,
          "unit": "ms"
        }
      }
    }
  ],
  "best_configurations": {
    "best_throughput": {
      "parameters": {
        "concurrency": 40
      },
      "metric": 255.1,
      "unit": "requests/sec"
    },
    "best_latency_p99": {
      "parameters": {
        "concurrency": 10
      },
      "metric": 125.4,
      "unit": "ms"
    }
  },
  "pareto_optimal": [
    {"concurrency": 10},
    {"concurrency": 20},
    {"concurrency": 30},
    {"concurrency": 40}
  ]
}
```

### Interpreting Per-Combination Metrics

For each parameter combination, you get:

- **mean**: Average across all trials (if using confidence runs)
- **std**: Standard deviation (variability between trials)
- **cv**: Coefficient of Variation (normalized variability)
- **ci_low, ci_high**: Confidence interval bounds
- **min, max**: Range of observed values

**What to look for:**
- **Low CV (<10%)**: Consistent performance at this concurrency level
- **High CV (>20%)**: High variability, may need more trials or investigation
- **Narrow CI**: High confidence in the mean estimate
- **Wide CI**: More uncertainty, consider more trials

## Pareto Optimal Configurations

A configuration is **Pareto optimal** if no other configuration is strictly better on ALL objectives simultaneously. For parameter sweeps, AIPerf uses two competing objectives:
- **Throughput** (maximize)
- **Latency** (minimize, using p99 TTFT)

### Understanding Pareto Optimality

In the example above, `"pareto_optimal": [{"concurrency": 10}, {"concurrency": 20}, {"concurrency": 30}, {"concurrency": 40}]` means:

- **Concurrency 10**: Best latency (125.4ms), but lower throughput (95.2 req/s)
  - No other config has both better latency AND better throughput

- **Concurrency 20**: Good latency (145.2ms) with moderate throughput (175.8 req/s)
  - Not dominated by any other configuration
  - Better latency than 30 and 40, though lower throughput

- **Concurrency 30**: Good balance (245.3 req/s, 180.5ms)
  - Better throughput than 10, better latency than 40
  - Represents a middle ground trade-off

- **Concurrency 40**: Best throughput (255.1 req/s), but higher latency (285.7ms)
  - No other config has both better throughput AND better latency

**Concurrency 20 is ALSO Pareto optimal** because:
- While concurrency 30 has higher throughput (245.3 vs 175.8), it has WORSE latency (180.5 vs 145.2)
- For a configuration to dominate another, it must be better or equal on ALL objectives
- Since 30's latency is worse, it does not dominate 20

### Choosing from Pareto Optimal Points

All Pareto optimal points are valid choices depending on your priorities:

- **Latency-sensitive applications** (real-time chat, interactive): Choose concurrency 10
- **Moderate latency with good throughput**: Choose concurrency 20
- **Balanced workloads** (general purpose): Choose concurrency 30
- **Throughput-focused** (batch processing, high load): Choose concurrency 40

There's no single "best" - it depends on your service level objectives (SLOs).

### Visualizing the Pareto Frontier

```text
Throughput (req/s)
    ^
260 |                    ● 40 (Pareto optimal)
240 |              ● 30 (Pareto optimal)
220 |
200 |
180 |        ● 20 (Pareto optimal)
160 |
140 |
120 |
100 | ● 10 (Pareto optimal)
 80 |
    +----------------------------------------> Latency (ms)
      120   140   160   180   200   220   240   260   280
```

All points on the frontier (●) are Pareto optimal. Each represents a different trade-off between throughput and latency.

## Mode Comparison: Repeated vs Independent

### When to Use Repeated Mode (Default)

**Use repeated mode when:**
- You want to capture dynamic system behavior as load changes
- You're testing systems with dynamic batching or scaling
- You want to measure real-world performance patterns
- You care about how the system transitions between load levels

**Example:**
```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --num-profile-runs 5 \
  --parameter-sweep-mode repeated \
  ...
```

**Execution:** Each trial runs the full sweep [10→20→30→40], preserving dynamic behavior.

**Benefits:**
- Captures system warm-up and adaptation effects
- Measures performance as load changes (realistic)
- Identifies if previous load affects current performance

**Drawbacks:**
- Results may show correlation between consecutive values
- Harder to isolate individual parameter effects

### When to Use Independent Mode

**Use independent mode when:**
- You want to isolate each parameter value
- You're measuring steady-state performance
- You want to minimize correlation between values
- You're comparing configurations independently

**Example:**
```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --num-profile-runs 5 \
  --parameter-sweep-mode independent \
  ...
```

**Execution:** All 5 trials at concurrency 10, then all 5 at 20, etc.

**Benefits:**
- Each value measured independently
- No correlation between different parameter values
- Clearer isolation of parameter effects

**Drawbacks:**
- Doesn't capture dynamic behavior
- May miss system adaptation effects
- Longer total runtime (no shared warm-up)

### Comparison Table

| Aspect | Repeated Mode | Independent Mode |
|--------|---------------|------------------|
| **Execution** | [sweep] × N trials | N trials × [sweep] |
| **Dynamic behavior** | ✅ Preserved | ❌ Not captured |
| **Isolation** | ❌ May have correlation | ✅ Fully isolated |
| **Use case** | Real-world patterns | Steady-state comparison |
| **Warm-up** | Shared across sweep | Per value |
| **Default** | ✅ Yes | No |

## Workload Consistency and Random Seeds

### Default Seed Behavior

By default, AIPerf uses **different random seeds** for each sweep value to avoid artificial correlation:

```bash
# Default behavior
aiperf profile --concurrency 10,20,30,40 ...
```

**Seed derivation:**
- Base seed: 42 (auto-set) or user-specified via `--random-seed`
- Per-value seeds: `base_seed + sweep_index`
  - Concurrency 10: seed = 42 + 0 = 42
  - Concurrency 20: seed = 42 + 1 = 43
  - Concurrency 30: seed = 42 + 2 = 44
  - Concurrency 40: seed = 42 + 3 = 45

**Why different seeds?**
- Avoids artificial correlation between sweep values
- Each value gets a different but reproducible workload
- More realistic performance characterization

### Using Same Seed Across Values

If you want to use the **same workload** for all sweep values:

```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --parameter-sweep-same-seed \
  ...
```

**Seed behavior:**
- All sweep values use the same seed (42 or user-specified)
- Identical prompts, ordering, and timing patterns
- Useful for comparing how different concurrency levels handle the exact same workload

**When to use same seed:**
- You want to isolate the effect of the parameter change
- You're debugging specific workload behavior
- You want perfectly correlated comparisons

**When NOT to use same seed:**
- General performance characterization (use default)
- You want to avoid artificial correlation
- You're measuring typical performance

### Custom Base Seed

Specify your own base seed:

```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --random-seed 123 \
  ...
```

Per-value seeds will be: 123, 124, 125, 126 (unless `--parameter-sweep-same-seed` is used).

## Cooldown Between Sweep Values

Use cooldown to allow the system to recover between parameter values:

```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --parameter-sweep-cooldown-seconds 30.0 \
  ...
```

### When to Use Cooldown

**Use cooldown when:**
- System needs time to stabilize between load changes
- You're testing systems with caching or memory effects
- You want to minimize correlation between consecutive values
- You're running on shared infrastructure

**Typical values:**
- **0 seconds** (default): No cooldown, fastest execution
- **10-30 seconds**: Light cooldown for basic stabilization
- **60+ seconds**: Heavy cooldown for systems with long memory effects

### Combining Trial and Sweep Cooldowns

When using both sweep and confidence runs, you can set cooldowns at both levels:

```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --num-profile-runs 5 \
  --profile-run-cooldown-seconds 10.0 \
  --parameter-sweep-cooldown-seconds 30.0 \
  ...
```

**Cooldown application (repeated mode):**
- `--profile-run-cooldown-seconds`: Between trials (between complete sweeps)
- `--parameter-sweep-cooldown-seconds`: Between sweep values within a trial

**Cooldown application (independent mode):**
- `--profile-run-cooldown-seconds`: Between trials within a sweep value
- `--parameter-sweep-cooldown-seconds`: Between sweep values

## Troubleshooting

### High Variance at Some Values

**Symptom:** Some concurrency values show high CV (>20%) while others are stable.

**Possible causes:**
- That concurrency level is near a system threshold
- Resource contention at that load level
- Batching or scheduling effects

**Solutions:**
1. Increase `--num-profile-runs` for that value
2. Add `--parameter-sweep-cooldown-seconds` to reduce correlation
3. Investigate system behavior at that load level
4. Check for resource bottlenecks (CPU, memory, GPU)

### Unexpected Pareto Optimal Points

**Symptom:** A configuration you expected to be dominated is Pareto optimal.

**Possible causes:**
- High variance in measurements
- System has non-linear scaling behavior
- Measurement artifacts

**Solutions:**
1. Increase `--num-profile-runs` to reduce variance
2. Check CV for those values - high CV indicates instability
3. Examine per-trial results for outliers
4. Add cooldown to reduce correlation

### No Clear Inflection Points

**Symptom:** Trend analysis doesn't show clear inflection points.

**Possible causes:**
- Linear scaling across the range tested
- Need wider range of parameter values
- System hasn't reached capacity

**Solutions:**
1. Extend the sweep range (e.g., `--concurrency 10,20,30,40,50,60`)
2. Use finer granularity (e.g., `--concurrency 10,15,20,25,30`)
3. Push the system harder to find limits

### Very Long Benchmark Times

**Symptom:** Sweep takes too long to complete.

**Solutions:**
1. **Reduce prompts per run**: `--num-prompts 500` instead of `--num-prompts 5000`
2. **Reduce trials**: `--num-profile-runs 3` instead of `--num-profile-runs 5`
3. **Remove cooldown**: Set cooldowns to 0 if not needed
4. **Reduce sweep range**: Test fewer values initially
5. **Run overnight**: For comprehensive production validation

### Failed Sweep Values

**Symptom:** Some sweep values fail while others succeed.

**Behavior:**
- AIPerf continues with remaining values
- Failed values excluded from aggregate analysis
- Failure details in sweep aggregate metadata

**Example output:**
```json
{
  "metadata": {
    "num_combinations": 4,
    "num_successful_runs": 3,
    "failed_runs": [
      {
        "run_index": 3,
        "error": "Connection timeout after 60s"
      }
    ]
  }
}
```

**Solutions:**
1. Investigate why that value fails (too high load?)
2. Adjust server configuration for higher load
3. Increase timeout values if needed
4. Check system resources at that load level

## Best Practices

### 1. Start with a Wide Range

Begin with a wide range to understand the full performance envelope:

```bash
aiperf profile --concurrency 5,10,20,40,80 ...
```

Then narrow down based on results.

### 2. Use Confidence Runs for Production

For production validation, always combine sweep with confidence runs:

```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --num-profile-runs 5 \
  ...
```

This quantifies variance and provides confidence intervals.

### 3. Check CV Before Drawing Conclusions

Always check the Coefficient of Variation (CV) for each value:
- CV < 10%: Results are trustworthy
- CV > 20%: Need more trials or investigation

### 4. Use Warmup

Always use warmup to eliminate cold-start effects:

```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --warmup-request-count 100 \
  ...
```

### 5. Document Your Findings

Save your sweep aggregate and document your conclusions:

```bash
# Save command for reproducibility
echo "aiperf profile --concurrency 10,20,30,40 ..." > benchmark_command.txt

# Document findings
cat > findings.md << EOF
## Benchmark Results

- Optimal concurrency: 30 (best balance)
- Pareto optimal points: 10, 30, 40
- Throughput inflection: 30 (scaling slows)
- Latency inflection: 40 (sharp degradation)

Recommendation: Operate at concurrency 30 for production.
EOF
```

### 6. Compare Apples to Apples

When comparing different configurations:
- Use the same sweep values
- Use the same number of trials
- Use the same random seed (or same seed derivation)
- Use the same workload parameters

### 7. Understand Your Objectives

Choose Pareto optimal points based on your SLOs:
- **Latency SLO**: Choose the lowest latency Pareto point
- **Throughput SLO**: Choose the highest throughput Pareto point
- **Balanced**: Choose the middle Pareto point

## Advanced Usage

### Combining with Other Features

Parameter sweeping works with all AIPerf features:

**With GPU telemetry:**
```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --gpu-telemetry http://localhost:9400/metrics \
  ...
```

**With server metrics:**
```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --server-metrics http://localhost:8000/metrics \
  ...
```

**With goodput constraints:**
```bash
aiperf profile \
  --concurrency 10,20,30,40 \
  --goodput "time_to_first_token:100 inter_token_latency:10" \
  ...
```

### Analyzing Results Programmatically

Load sweep aggregate results in Python:

```python
import json
import pandas as pd

# Load sweep aggregate
with open('artifacts/.../sweep_aggregate/profile_export_aiperf_sweep.json') as f:
    sweep = json.load(f)

# Extract throughput and latency for each combination
data = []
for combo in sweep['per_combination_metrics']:
    params = combo['parameters']
    metrics = combo['metrics']
    data.append({
        'concurrency': params['concurrency'],
        'throughput': metrics['request_throughput_avg']['mean'],
        'latency_p99': metrics['ttft_p99_ms']['mean'],
        'throughput_cv': metrics['request_throughput_avg'].get('cv', 0),
        'latency_cv': metrics['ttft_p99_ms'].get('cv', 0),
    })

df = pd.DataFrame(data).sort_values('concurrency')
print(df)

# Identify Pareto optimal points
pareto_optimal = sweep['pareto_optimal']
pareto_concurrency_values = [p['concurrency'] for p in pareto_optimal]
print(f"Pareto optimal concurrency values: {pareto_concurrency_values}")

# Get best configurations
best_configs = sweep['best_configurations']
print(f"Best throughput: {best_configs['best_throughput']['parameters']}")
print(f"Best latency: {best_configs['best_latency_p99']['parameters']}")
```

### Creating Custom Visualizations

```python
import matplotlib.pyplot as plt

# Plot throughput vs latency (Pareto frontier)
fig, ax = plt.subplots(figsize=(10, 6))

for _, row in df.iterrows():
    is_pareto = row['concurrency'] in pareto_concurrency_values
    marker = 'o' if is_pareto else 'x'
    color = 'blue' if is_pareto else 'gray'
    label = f"C={row['concurrency']}" + (" (Pareto)" if is_pareto else "")
    ax.scatter(row['latency_p99'], row['throughput'],
               marker=marker, s=100, color=color,
               label=label)

ax.set_xlabel('Latency P99 (ms)')
ax.set_ylabel('Throughput (req/s)')
ax.set_title('Pareto Frontier: Throughput vs Latency')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('pareto_frontier.png')
```

## Summary

Parameter sweeping helps you:
- ✅ Systematically characterize performance across parameter values
- ✅ Identify optimal configurations with Pareto analysis
- ✅ Compare performance across different parameter combinations
- ✅ Quantify variance with confidence intervals
- ✅ Make data-driven capacity planning decisions

**Quick Start:**
```bash
# Simple sweep
aiperf profile --concurrency 10,20,30,40 [other options]

# Sweep with confidence (recommended)
aiperf profile --concurrency 10,20,30,40 --num-profile-runs 5 [other options]
```

**Key Concepts:**
- **Pareto optimal**: Best trade-off configurations
- **Best configurations**: Highest throughput and lowest latency points
- **Sweep modes**: Repeated (dynamic) vs Independent (isolated)
- **CV < 10%**: Good repeatability

For more details, see:
- [Sweep Aggregates API Reference](../api/sweep-aggregates.md) - Complete data format documentation
- [Multi-Run Confidence](./multi-run-confidence.md) - Understanding confidence intervals
- [CLI Options](../cli_options.md) - Full parameter reference
- [Metrics Reference](../metrics_reference.md) - Detailed metric descriptions

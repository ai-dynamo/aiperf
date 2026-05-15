<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Sweep Aggregate API Reference

Complete API documentation for parameter sweep aggregate outputs, including JSON schema, CSV format, and programmatic analysis examples.

## Overview

When running parameter sweeps with AIPerf (e.g., `--concurrency 10,20,30`), the system generates sweep aggregate files that summarize performance across all parameter combinations. These aggregates enable:

- Comparison of performance across parameter combinations
- Identification of optimal configurations
- Pareto frontier analysis for multi-objective optimization
- Statistical analysis with confidence intervals (when using `--num-profile-runs > 1`)

## Output Files

Sweep aggregates are written to different locations depending on the sweep mode:

**Sweep-only** (no `--num-profile-runs`):
```text
artifacts/
  {benchmark_name}/
    sweep_aggregate/
      profile_export_aiperf_sweep.json    # Structured data for programmatic analysis
      profile_export_aiperf_sweep.csv     # Tabular format for spreadsheet analysis
```

**Independent Mode** (sweep + `--num-profile-runs > 1` + `--parameter-sweep-mode independent`):
```text
artifacts/
  {benchmark_name}/
    concurrency_10/aggregate/             # Per-value confidence aggregates
      profile_export_aiperf_aggregate.json
      profile_export_aiperf_aggregate.csv
    concurrency_20/aggregate/
      ...
    sweep_aggregate/                      # Cross-value sweep analysis
      profile_export_aiperf_sweep.json
      profile_export_aiperf_sweep.csv
```

**Repeated Mode** (sweep + `--num-profile-runs > 1`, default mode):
```text
artifacts/
  {benchmark_name}/
    aggregate/
      concurrency_10/                     # Per-value confidence aggregates
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
      concurrency_20/
        ...
      sweep_aggregate/                    # Cross-value sweep analysis
        profile_export_aiperf_sweep.json
        profile_export_aiperf_sweep.csv
```

See [Artifact Directory Layout Reference](#artifact-directory-layout-reference)
below for the full table of layout cases.

The sweep aggregate files contain cross-value analysis including best configurations and Pareto optimal points.

---

## JSON Schema

### Top-Level Structure

```json
{
  "aggregation_type": "sweep",
  "num_profile_runs": 12,
  "num_successful_runs": 12,
  "failed_runs": [],
  "metadata": { ... },
  "per_combination_metrics": [ ... ],
  "best_configurations": { ... },
  "pareto_optimal": [ ... ]
}
```

**Top-Level Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `aggregation_type` | string | Always `"sweep"` for sweep aggregates |
| `num_profile_runs` | int | Total number of profile runs executed |
| `num_successful_runs` | int | Number of successful profile runs |
| `failed_runs` | array | List of failed runs with error details (empty if all succeeded) |
| `metadata` | object | Sweep configuration and execution metadata |
| `per_combination_metrics` | array | List of metrics for each parameter combination |
| `best_configurations` | object | Best parameter combinations for key metrics |
| `pareto_optimal` | array | List of Pareto optimal parameter combinations |

### Metadata Section

Contains information about the sweep configuration.

```json
{
  "metadata": {
    "sweep_parameters": [
      {
        "name": "concurrency",
        "values": [10, 20, 30, 40]
      }
    ],
    "num_combinations": 4
  }
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `sweep_parameters` | array | List of parameter definitions (name and values) |
| `num_combinations` | int | Total number of parameter combinations tested |
| `aggregation_type` | string | Always `"sweep"` (duplicated from the top-level field so consumers that key off `output["metadata"]["aggregation_type"]` work without first checking the top-level key) |
| `sla_constraints` | object | Present only when `plan.sweep.sla_filters` is non-empty. Contains `active_filters` (list of filter dicts), `feasible_count` (int), and `infeasible_count` (int). See `src/aiperf/orchestrator/aggregation/sweep_sla_filter.py` for the filter shape. |

**Note:** For QMC sweeps, `sampling_design.json` is written to `<base>/sweep_aggregate/sampling_design.json` in single-trial and independent modes. In repeated multi-run mode the sweep aggregate can live under `<base>/aggregate/sweep_aggregate/`, so the sampling design is not necessarily a sibling of the repeated-mode aggregate directory.

**Sweep Parameters Structure:**

Each parameter definition contains:
- `name`: Parameter name (e.g., `"concurrency"`, `"request_rate"`)
- `values`: List of values tested for this parameter

### Per-Combination Metrics Section

Contains aggregated metrics for each parameter combination. This is a list where each entry represents one combination.

```json
{
  "per_combination_metrics": [
    {
      "parameters": {
        "concurrency": 10
      },
      "metrics": {
        "request_throughput_avg": {
          "mean": 100.5,
          "std": 5.2,
          "min": 95.0,
          "max": 108.0,
          "cv": 0.052,
          "ci_low": 94.3,
          "ci_high": 106.7,
          "unit": "requests/sec"
        },
        "time_to_first_token_p99": {
          "mean": 120.5,
          "std": 8.1,
          "min": 110.2,
          "max": 132.8,
          "cv": 0.067,
          "ci_low": 111.5,
          "ci_high": 129.5,
          "unit": "ms"
        }
      }
    },
    {
      "parameters": {
        "concurrency": 20
      },
      "metrics": { ... }
    }
  ]
}
```

**Combination Entry Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `parameters` | object | Dictionary of parameter names to values for this combination |
| `metrics` | object | Dictionary of metric names to statistics |

**Metric Statistics Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mean` | float | Mean value across trials |
| `std` | float | Standard deviation across trials |
| `min` | float | Minimum value observed |
| `max` | float | Maximum value observed |
| `cv` | float | Coefficient of variation (std/mean) |
| `ci_low` | float | Lower bound of confidence interval |
| `ci_high` | float | Upper bound of confidence interval |
| `unit` | string | Unit of measurement |

**Note:** Fields `se` (standard error) and `t_critical` (critical t-value) exist on the underlying `ConfidenceMetric` dataclass and are emitted by the per-variation *confidence aggregate* (`profile_export_aiperf_aggregate.json`), but the sweep aggregate's per-combination block strips them.

**Note:** For single-trial sweeps (`--num-profile-runs 1` or omitted), the per-combination metric block still emits the full field set, but the spread fields collapse to degenerate values: `std=0`, `cv=0`, `ci_low=ci_high=mean`. The single-trial projection also emits an `avg` alias of `mean` and passes through every populated percentile field (`p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99`) directly from the underlying `JsonMetricResult`.

### Best Configurations Section

Identifies the parameter combinations that achieved the best performance for key metrics.

```json
{
  "best_configurations": {
    "best_throughput": {
      "parameters": {
        "concurrency": 40
      },
      "metric": 350.2,
      "unit": "requests/sec"
    },
    "best_latency_p99": {
      "parameters": {
        "concurrency": 10
      },
      "metric": 120.5,
      "unit": "ms"
    }
  }
}
```

**Configuration Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `parameters` | object | Parameter combination that achieved best performance |
| `metric` | float | The metric value achieved |
| `unit` | string | Unit of measurement |

**Available Configurations:**

- `best_throughput`: Highest `request_throughput_avg`
- `best_latency_p99`: Lowest `time_to_first_token_p99` (or `request_latency_p99` as fallback)

### Pareto Optimal Section

Lists parameter combinations that are Pareto optimal - configurations where no other configuration is strictly better on all objectives simultaneously.

```json
{
  "pareto_optimal": [
    {"concurrency": 10},
    {"concurrency": 30},
    {"concurrency": 40}
  ]
}
```

**Default Objectives:**
- Maximize: `request_throughput_avg` (throughput)
- Minimize: `time_to_first_token_p99` (latency)

A configuration is Pareto optimal if:
- No other configuration has both higher throughput AND lower latency
- It represents a valid trade-off point on the efficiency frontier

**Example Interpretation:**
```text
Concurrency 10: Low latency, moderate throughput (latency-optimized)
Concurrency 30: Balanced latency and throughput
Concurrency 40: High throughput, higher latency (throughput-optimized)
```

**Multi-Parameter Sweeps:**

For sweeps with multiple parameters (e.g., `--concurrency 10,20 --request-rate 5,10`), each Pareto optimal entry contains all parameter values:

```json
{
  "pareto_optimal": [
    {"concurrency": 10, "request_rate": 5},
    {"concurrency": 20, "request_rate": 10}
  ]
}
```

---

## CSV Format

The CSV export provides a tabular view optimized for spreadsheet analysis and plotting.

### Structure

The CSV file contains multiple sections separated by blank lines:

1. **Per-Combination Metrics Table** (main data)
2. **Best Configurations**
3. **Pareto Optimal Points**
4. **Metadata**

### Per-Combination Metrics Table

The first section is a wide-format table with one row per parameter combination:

```csv
concurrency,request_throughput_avg_mean,request_throughput_avg_std,request_throughput_avg_min,request_throughput_avg_max,request_throughput_avg_cv,time_to_first_token_p99_mean,time_to_first_token_p99_std,time_to_first_token_p99_min,time_to_first_token_p99_max,time_to_first_token_p99_cv
10,100.50,5.20,95.00,108.00,0.0520,120.50,8.10,110.20,132.80,0.0672
20,180.30,8.50,170.00,195.00,0.0471,135.20,9.30,125.00,148.00,0.0688
30,270.80,12.10,255.00,290.00,0.0447,155.80,11.20,142.00,172.00,0.0719
40,285.50,15.30,265.00,310.00,0.0536,180.30,13.50,165.00,200.00,0.0749
```

**Columns:**
- Parameter columns (e.g., `concurrency`, `request_rate`)
- For each metric: `{metric}_mean`, `{metric}_std`, `{metric}_min`, `{metric}_max`, `{metric}_cv`

**Multi-Parameter Example:**

```csv
concurrency,request_rate,request_throughput_avg_mean,request_throughput_avg_std,...
10,5,50.25,2.10,...
10,10,95.30,4.50,...
20,5,98.40,3.20,...
20,10,185.60,7.80,...
```

### Best Configurations Section

```csv
Best Configurations
Configuration,concurrency,Metric,Unit
Best Throughput,40,285.50,requests/sec
Best Latency P99,10,120.50,ms
```

For multi-parameter sweeps:

```csv
Best Configurations
Configuration,concurrency,request_rate,Metric,Unit
Best Throughput,40,10,350.20,requests/sec
Best Latency P99,10,5,95.30,ms
```

### Pareto Optimal Section

```csv
Pareto Optimal Points
concurrency
10
30
40
```

For multi-parameter sweeps:

```csv
Pareto Optimal Points
concurrency,request_rate
10,5
20,10
40,10
```

**Empty frontier:** When no frontier can be computed (a required objective metric is missing from the per-combination block, or every cell was filtered out by SLA constraints), the section renders a single literal `None` row beneath the `Pareto Optimal Points` header instead of the parameter-name header + rows.

### Metadata Section

```csv
Metadata
Field,Value
Aggregation Type,sweep
Sweep Parameters,concurrency
Number of Combinations,4
Number of Profile Runs,12
Number of Successful Runs,12
```

---

## Artifact Directory Structure

### Artifact Directory Layout Reference

The artifact tree branches on three flags: whether a sweep is configured
(`is_sweep`), whether multiple trials run per cell (`trials > 1`), and
the sweep iteration order (`REPEATED` vs `INDEPENDENT`).

| sweep | trials | order       | layout                                          |
|-------|--------|-------------|-------------------------------------------------|
| no    | 1      | -           | `<base>/`                                       |
| no    | >1     | -           | `<base>/profile_runs/run_NNNN/`                 |
| yes   | 1      | -           | `<base>/<dir_name>/`                            |
| yes   | >1     | REPEATED    | `<base>/profile_runs/trial_NNNN/<dir_name>/`    |
| yes   | >1     | INDEPENDENT | `<base>/<dir_name>/profile_runs/trial_NNNN/`    |
| adaptive | any | -      | `<base>/search_iter_NNNN/profile_runs/run_NNNN/` |

`<dir_name>` is the `{leaf_param_name}_{value}` form (e.g.
`concurrency_10`, `request_rate_5.0`); multi-dim sweep cells join
components with `__` (e.g. `concurrency_10__isl_512`). Inner-dir
naming is asymmetric on purpose: the no-sweep multi-run case uses
`run_NNNN`, the sweep + INDEPENDENT case uses `trial_NNNN`.

The sweep-level aggregate path follows a parallel rule:

- REPEATED + multi-run: `<base>/aggregate/sweep_aggregate/`
- everything else (sweep-only, sweep + INDEPENDENT): `<base>/sweep_aggregate/`

Per-variation aggregates land at `<base>/aggregate/<dir_name>/` in
REPEATED mode and `<base>/<dir_name>/aggregate/` in INDEPENDENT mode.

### Repeated Mode (`--parameter-sweep-mode repeated`)

Default mode where the full sweep is executed N times:

```text
artifacts/
  {benchmark_name}/
    profile_runs/
      trial_0001/
        concurrency_10/
          profile_export_aiperf.json
          profile_export.jsonl
        concurrency_20/
          profile_export_aiperf.json
          profile_export.jsonl
        concurrency_30/
          profile_export_aiperf.json
          profile_export.jsonl
      trial_0002/
        concurrency_10/
        concurrency_20/
        concurrency_30/
      trial_0003/
        concurrency_10/
        concurrency_20/
        concurrency_30/
    aggregate/
      concurrency_10/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
      concurrency_20/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
      concurrency_30/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
      sweep_aggregate/
        profile_export_aiperf_sweep.json
        profile_export_aiperf_sweep.csv
```

**Execution Pattern:**
```text
Trial 1: [10 → 20 → 30]
Trial 2: [10 → 20 → 30]
Trial 3: [10 → 20 → 30]
```

### Independent Mode (`--parameter-sweep-mode independent`)

All trials at each parameter value before moving to the next:

```text
artifacts/
  {benchmark_name}/
    concurrency_10/
      profile_runs/
        trial_0001/
          profile_export_aiperf.json
          profile_export.jsonl
        trial_0002/
        trial_0003/
      aggregate/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
    concurrency_20/
      profile_runs/
        trial_0001/
        trial_0002/
        trial_0003/
      aggregate/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
    concurrency_30/
      profile_runs/
        trial_0001/
        trial_0002/
        trial_0003/
      aggregate/
        profile_export_aiperf_aggregate.json
        profile_export_aiperf_aggregate.csv
    sweep_aggregate/
      profile_export_aiperf_sweep.json
      profile_export_aiperf_sweep.csv
```

**Execution Pattern:**
```text
Concurrency 10: [trial1, trial2, trial3]
Concurrency 20: [trial1, trial2, trial3]
Concurrency 30: [trial1, trial2, trial3]
```

### Single-Trial Sweep

When `--num-profile-runs 1` (or omitted), no trial directories are created:

```text
artifacts/
  {benchmark_name}/
    concurrency_10/
      profile_export_aiperf.json
      profile_export.jsonl
    concurrency_20/
      profile_export_aiperf.json
      profile_export.jsonl
    concurrency_30/
      profile_export_aiperf.json
      profile_export.jsonl
    sweep_aggregate/
      profile_export_aiperf_sweep.json
      profile_export_aiperf_sweep.csv
```

---

## Programmatic Analysis Examples

### Example 1: Load and Inspect Sweep Results

```python
import json
from pathlib import Path

# Load sweep aggregate
sweep_file = Path("artifacts/my_benchmark/sweep_aggregate/profile_export_aiperf_sweep.json")
with open(sweep_file) as f:
    sweep_data = json.load(f)

# Inspect metadata
metadata = sweep_data["metadata"]
sweep_params = metadata["sweep_parameters"]
print(f"Sweep parameters: {[p['name'] for p in sweep_params]}")
print(f"Total combinations: {metadata['num_combinations']}")
print(f"Total runs: {sweep_data['num_profile_runs']}")
```

### Example 2: Find Optimal Configuration

```python
# Get best configurations
best_configs = sweep_data["best_configurations"]

best_throughput = best_configs["best_throughput"]
print(f"Best throughput: {best_throughput['metric']:.2f} {best_throughput['unit']}")
print(f"  Parameters: {best_throughput['parameters']}")

best_latency = best_configs["best_latency_p99"]
print(f"Best latency: {best_latency['metric']:.2f} {best_latency['unit']}")
print(f"  Parameters: {best_latency['parameters']}")
```

### Example 3: Analyze Pareto Frontier

```python
# Get Pareto optimal points
pareto_optimal = sweep_data["pareto_optimal"]
print(f"Found {len(pareto_optimal)} Pareto optimal configurations")

# Extract metrics for Pareto points
per_combination_metrics = sweep_data["per_combination_metrics"]

print("\nPareto Frontier:")
for combo in per_combination_metrics:
    params = combo["parameters"]
    # Check if this combination is Pareto optimal
    if params in pareto_optimal:
        metrics = combo["metrics"]
        throughput = metrics["request_throughput_avg"]["mean"]
        latency = metrics["time_to_first_token_p99"]["mean"]
        print(f"  {params}: {throughput:.1f} req/s, {latency:.1f} ms p99")
```

### Example 4: Compare Confidence Intervals

```python
import matplotlib.pyplot as plt

# Extract data for single-parameter sweep
combinations = sweep_data["per_combination_metrics"]

# Assuming single parameter (concurrency)
param_name = sweep_data["metadata"]["sweep_parameters"][0]["name"]
param_values = []
throughputs = []
ci_lows = []
ci_highs = []

for combo in combinations:
    param_value = combo["parameters"][param_name]
    tp = combo["metrics"]["request_throughput_avg"]

    param_values.append(param_value)
    throughputs.append(tp["mean"])
    ci_lows.append(tp.get("ci_low", tp["mean"]))
    ci_highs.append(tp.get("ci_high", tp["mean"]))

# Plot with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(param_values, throughputs, 'o-', label='Mean Throughput')
plt.fill_between(param_values, ci_lows, ci_highs, alpha=0.3, label='95% CI')
plt.xlabel(param_name.title())
plt.ylabel('Throughput (requests/sec)')
plt.title(f'Throughput vs {param_name.title()}')
plt.legend()
plt.grid(True)
plt.savefig('throughput_sweep.png')
```

### Example 5: Export to Pandas DataFrame

```python
import pandas as pd

# Convert per-combination metrics to DataFrame
rows = []
for combo in sweep_data["per_combination_metrics"]:
    row = combo["parameters"].copy()

    # Add metrics
    for metric_name, metric_data in combo["metrics"].items():
        if isinstance(metric_data, dict):
            row[f"{metric_name}_mean"] = metric_data.get("mean")
            row[f"{metric_name}_std"] = metric_data.get("std")
            row[f"{metric_name}_cv"] = metric_data.get("cv")
        else:
            row[metric_name] = metric_data
    rows.append(row)

df = pd.DataFrame(rows)

# Sort by parameter values
param_names = [p["name"] for p in sweep_data["metadata"]["sweep_parameters"]]
df = df.sort_values(param_names)

# Analyze
print(df[[*param_names, "request_throughput_avg_mean", "time_to_first_token_p99_mean"]])

# Export
df.to_csv("sweep_analysis.csv", index=False)
```

### Example 6: Multi-Parameter Sweep Analysis

```python
# For sweeps with multiple parameters
sweep_params = sweep_data["metadata"]["sweep_parameters"]
param_names = [p["name"] for p in sweep_params]

print(f"Multi-parameter sweep: {', '.join(param_names)}")

# Find best combination for each parameter individually
for param_name in param_names:
    # Group by this parameter
    param_groups = {}
    for combo in sweep_data["per_combination_metrics"]:
        param_value = combo["parameters"][param_name]
        if param_value not in param_groups:
            param_groups[param_value] = []
        param_groups[param_value].append(combo)

    # Find best throughput for each value of this parameter
    print(f"\nBest throughput for each {param_name}:")
    for value, combos in sorted(param_groups.items()):
        best_combo = max(combos,
                        key=lambda c: c["metrics"]["request_throughput_avg"]["mean"])
        throughput = best_combo["metrics"]["request_throughput_avg"]["mean"]
        print(f"  {param_name}={value}: {throughput:.1f} req/s")
        print(f"    Full config: {best_combo['parameters']}")
```

### Example 7: Identify Diminishing Returns

```python
# For single-parameter sweeps, calculate efficiency
combinations = sweep_data["per_combination_metrics"]
param_name = sweep_data["metadata"]["sweep_parameters"][0]["name"]

# Sort by parameter value
combinations_sorted = sorted(combinations,
                            key=lambda c: c["parameters"][param_name])

efficiencies = []
for combo in combinations_sorted:
    param_value = combo["parameters"][param_name]
    throughput = combo["metrics"]["request_throughput_avg"]["mean"]
    efficiency = throughput / param_value
    efficiencies.append((param_value, efficiency))

# Find point of diminishing returns (where efficiency drops significantly)
threshold = 0.8  # 20% drop
for i in range(1, len(efficiencies)):
    if efficiencies[i][1] < threshold * efficiencies[i-1][1]:
        print(f"Diminishing returns detected at {param_name}={efficiencies[i][0]}")
        print(f"  Efficiency dropped from {efficiencies[i-1][1]:.2f} to {efficiencies[i][1]:.2f}")
        break
```

### Example 8: Multi-Objective Decision Making

```python
# Score configurations based on weighted objectives
weights = {
    "throughput": 0.6,  # 60% weight on throughput
    "latency": 0.4,     # 40% weight on latency
}

# Extract all throughputs and latencies
combinations = sweep_data["per_combination_metrics"]
throughputs = [c["metrics"]["request_throughput_avg"]["mean"] for c in combinations]
latencies = [c["metrics"]["time_to_first_token_p99"]["mean"] for c in combinations]

max_tp = max(throughputs)
min_lat = min(latencies)
max_lat = max(latencies)

scores = []
for combo in combinations:
    tp = combo["metrics"]["request_throughput_avg"]["mean"]
    lat = combo["metrics"]["time_to_first_token_p99"]["mean"]

    # Normalize: higher is better for both
    tp_score = tp / max_tp
    lat_score = 1 - (lat - min_lat) / (max_lat - min_lat) if max_lat > min_lat else 1.0

    # Weighted combination
    score = weights["throughput"] * tp_score + weights["latency"] * lat_score
    scores.append((combo["parameters"], score))

# Find best configuration
best_params, best_score = max(scores, key=lambda x: x[1])
print(f"Best configuration for given weights: {best_params}")
print(f"  Score: {best_score:.3f}")
```

---

## See Also

- [Parameter Sweeping Tutorial](../tutorials/sweeps.md) - User guide with examples
- [Multi-Run Confidence Tutorial](../tutorials/multi-run-confidence.md) - Understanding confidence statistics
- [Working with Profile Exports](../tutorials/working-with-profile-exports.md) - General export analysis
- [CLI Options Reference](../cli-options.md) - Complete CLI documentation

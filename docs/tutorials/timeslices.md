<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Time Slicing for Performance Analysis

Time slicing allows you to analyze performance metrics across sequential time windows during a benchmark run. This feature provides visibility into performance trends, degradation patterns, and system behavior over time.

## Overview

Time slicing divides your benchmark into equal duration segments, computing metrics independently for each segment. This enables:

- **Performance Trend Analysis**: Identify if performance degrades, improves, or stabilizes over time
- **Warm-up Detection**: Distinguish initial cold-start behavior from steady-state performance
- **Resource Exhaustion**: Spot gradual performance degradation due to memory leaks or resource pressure
- **Load Pattern Impact**: Understand how different phases of load affect system performance
- **Time-series Visualization**: Export data suitable for plotting performance trends

## Core Parameters

### Slice Duration
- `--slice-duration SECONDS`: Duration of each time slice (accepts integers or floats)
- Recommended to be used with `--benchmark-duration`
- Creates non-overlapping sequential time windows
- Example: 60-second benchmark with 10-second slices creates 6 time windows
  - When using time-based benchmarking, a grace period may add additional time slices

### Benchmark Duration
- `--benchmark-duration SECONDS`: Total benchmark duration
- Must be greater than `--slice-duration`
- Determines how many slices will be created

## Basic Time Slicing

### Setting Up the Server

```bash
# Start vLLM server for time slicing demonstration
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

### Simple Time Slicing Example

Run a 60-second benchmark with 10-second slices to analyze performance trends:

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --benchmark-duration 60 \
  --slice-duration 10
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Timeslice analysis enabled: 10 second intervals
INFO     AIPerf System is PROFILING

Profiling: [01:00] - Running for 60 seconds...

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-duration60/

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-duration60/profile_export_aiperf.json
Timeslices: artifacts/Qwen_Qwen3-0.6B-chat-duration60/profile_export_aiperf_timeslices.json
Timeslices CSV: artifacts/Qwen_Qwen3-0.6B-chat-duration60/profile_export_aiperf_timeslices.csv
```

This creates 6 time slices (0-10s, 10-20s, 20-30s, 30-40s, 40-50s, 50-60s), each with independent metrics.

## Output Files

When time slicing is enabled, AIPerf generates additional output files:

### CSV Timeslice Export

**File**: `artifacts/profile_export_aiperf_timeslices.csv`

The CSV uses a "tidy" (long) format optimized for data analysis:

```csv
Timeslice,Metric,Unit,Stat,Value
0,Time to First Token,ms,avg,45.23
0,Time to First Token,ms,min,32.10
0,Time to First Token,ms,max,78.45
0,Time to First Token,ms,p50,44.20
0,Time to First Token,ms,p90,65.30
0,Time to First Token,ms,p95,70.15
0,Time to First Token,ms,p99,76.80
0,Inter Token Latency,ms,avg,12.34
0,Inter Token Latency,ms,min,8.50
...
1,Time to First Token,ms,avg,46.78
1,Time to First Token,ms,min,33.20
...
```

**Format Details**:
- **Timeslice**: Zero-indexed slice number (0, 1, 2, ...)
- **Metric**: Human-readable metric name (e.g., "Time to First Token")
- **Unit**: Measurement unit (ms, tokens/sec, etc.)
- **Stat**: Statistical measure (avg, min, max, p50, p90, p95, p99)
- **Value**: Numeric value formatted to 2 decimal places

### JSON Timeslice Export

**File**: `artifacts/profile_export_aiperf_timeslices.json`

The JSON provides a hierarchical structure with all timeslices in a single file:

```json
{
  "timeslices": [
    {
      "timeslice_index": 0,
      "time_to_first_token": {
        "unit": "ms",
        "avg": 45.23,
        "min": 32.10,
        "max": 78.45,
        "p50": 44.20,
        "p90": 65.30,
        "p95": 70.15,
        "p99": 76.80
      },
      "inter_token_latency": {
        "unit": "ms",
        "avg": 12.34,
        "min": 8.50,
        "max": 18.90,
        "p50": 12.10,
        "p90": 15.80,
        "p95": 16.50,
        "p99": 17.90
      },
      ...
    },
    {
      "timeslice_index": 1,
      "time_to_first_token": {
        "unit": "ms",
        "avg": 46.78,
        ...
      },
      ...
    }
  ],
  "input_config": {
    "model": "Qwen/Qwen3-0.6B",
    "endpoint": "/v1/chat/completions",
    ...
  }
}
```

**Key Fields**:
- `timeslices`: Array of slice objects, ordered by time
- `timeslice_index`: Zero-indexed slice identifier
- Each metric contains `unit` and available statistics
- `input_config`: Benchmark configuration for reproducibility

## Use Cases

### Detecting Warm-up Effects

* Identify initial cold-start latency vs. steady-state performance.
* Expected pattern: Higher latency in slice 0, stabilizing in later slices.

### Performance Degradation Analysis

* Monitor for memory leaks or resource exhaustion.
* Look for: Increasing latency or decreasing throughput in later slices.

### Load Pattern Impact

* Combine with varying concurrency patterns.
* Compare slice patterns across different load levels.

## Visualizing Timeslice Data

To be announced...

## Best Practices

> [!WARNING]
> **Timeslice Boundaries:**
> - Timeslices are calculated based on absolute wall clock time divisions
> - The first timeslice may be shorter if requests don't start exactly at a timeslice boundary
> - The last timeslice may be shorter if the benchmark ends mid-slice

> [!WARNING]
> **Statistical Considerations:**
> - Very short slices may have high variance and unstable metrics
> - Low-concurrency benchmarks need longer slices for adequate sample size

## Troubleshooting

### No Timeslice Files Generated

**Problem**: Running with `--slice-duration` but no `*_timeslices.*` files appear.

**Solutions**:
- Verify `--slice-duration` (in seconds) is less than the benchmark duration
- Check that benchmark completed successfully (not cancelled/interrupted)
- Confirm output directory is writable

### High Variance Between Slices

**Problem**: Metrics fluctuate wildly between consecutive slices.

**Solutions**:
- Increase `--slice-duration` for more stable statistics
- Increase `--concurrency` to generate more requests per slice
- Check for external factors (other processes, network issues)
- Use longer warmup period (`--warmup-request-count`)

## Related Documentation

- [Time-based Benchmarking](time-based-benchmarking.md) - Understanding `--benchmark-duration`
- [Working with Profile Exports](working-with-profile-exports.md) - General export formats
- [GPU Telemetry](gpu-telemetry.md) - Correlating GPU metrics with performance
- [Request Rate and Concurrency](request-rate-concurrency.md) - Load generation strategies

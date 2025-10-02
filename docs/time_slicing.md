<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Time Slicing for Benchmark Metrics

## Overview

Time slicing allows you to split your benchmark results into multiple time-based segments for detailed analysis of performance over time. This feature is particularly useful for:

- Identifying performance degradation or improvement over the course of a benchmark
- Analyzing warm-up effects and steady-state performance
- Detecting periodic performance variations
- Understanding how metrics evolve during long-running tests

## Usage

To enable time slicing in your benchmark, use the `--slice-duration` parameter:

```bash
aiperf profile \
  --model gpt-3.5-turbo \
  --endpoint-type chat \
  --url http://localhost:8000/v1/chat/completions \
  --request-rate 10 \
  --benchmark-duration 60 \
  --slice-duration 5000 \
  --output-artifact-dir ./results
```

### Parameters

- `--slice-duration MILLISECONDS`: The duration of each time slice in milliseconds. For example, `5000` creates 5-second slices.

## Output Format

When time slicing is enabled, the CSV export will contain:

1. **Overall Benchmark Metrics**: Aggregated statistics for the entire benchmark run (same as without time slicing)

2. **Time Slice N Metrics**: Individual sections for each time slice, showing:
   - Time slice index and time range (e.g., "Time Slice 1 (0.00ms - 5000.00ms)")
   - All standard metrics (latency percentiles, throughput, token statistics, etc.)
   - Statistics calculated only from requests that fall within that time window

### Example CSV Structure

```csv
=== Overall Benchmark Metrics ===

Metric,avg,min,max,p50,p90,p95,p99,std
Request Latency (ms),45.23,12.45,89.67,43.21,67.89,75.43,85.21,15.67
...


=== Time Slice 1 (0.00ms - 5000.00ms) ===

Metric,avg,min,max,p50,p90,p95,p99,std
Request Latency (ms),48.12,15.23,85.34,45.67,70.12,78.45,83.21,14.32
...


=== Time Slice 2 (5000.00ms - 10000.00ms) ===

Metric,avg,min,max,p50,p90,p95,p99,std
Request Latency (ms),43.45,10.12,91.23,41.23,65.78,72.34,87.65,16.21
...
```

## Implementation Details

### How It Works

1. **Data Collection**: During the benchmark, raw metric data (as NumPy arrays) is collected alongside the normal aggregated statistics.

2. **Time Segmentation**: After the benchmark completes, the time range is divided into N segments based on the `--slice-duration` parameter.

3. **Filtering**: For each time slice, requests are filtered based on their timestamp to include only those that started within that slice's time window.

4. **Metric Calculation**: Standard statistics (percentiles, averages, etc.) are recalculated for each time slice using only the filtered subset of requests.

5. **Export**: All metrics (overall + time-sliced) are written to the CSV file in separate sections.

### Timestamp Filtering

Requests are assigned to time slices based on their start timestamp (`min_request_timestamp`). A request belongs to a slice if:

```
slice_start_ns <= request_timestamp_ns < slice_end_ns
```

### Performance Considerations

- Enabling time slicing requires storing raw metric arrays in memory, which increases memory usage proportional to the number of requests
- The time slicing computation happens post-benchmark and does not affect the actual benchmark execution
- For very large benchmarks (millions of requests), consider using larger slice durations to reduce the number of slices generated

## Use Cases

### Warm-up Analysis

```bash
# Use small slices to see how performance stabilizes
--benchmark-duration 60 --slice-duration 1000  # 1-second slices
```

### Long-running Stability

```bash
# Use larger slices for extended tests
--benchmark-duration 3600 --slice-duration 60000  # 1-minute slices for 1-hour test
```

### Performance Trending

```bash
# Moderate slices for general trending
--benchmark-duration 300 --slice-duration 10000  # 10-second slices for 5-minute test
```

## Limitations

- Time slicing is only available for record-type metrics (those with individual data points per request)
- Aggregate metrics (calculated across the entire benchmark) are not re-calculated per slice
- Derived metrics may not be recalculated if they depend on aggregate metrics
- Time slices with zero requests will show "No data in this time slice"

## See Also

- [Time-based Benchmarking](tutorials/time-based-benchmarking.md) - Running benchmarks for a specific duration
- [CLI Options](cli_options.md) - Complete list of CLI parameters



<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Prefix Data Synthesis Tutorial

Learn how to analyze and generate synthetic traces with controlled prefix-sharing patterns for KV cache benchmarking.

## Overview

The prefix synthesis feature enables:
- **Analyze** existing traces to understand prefix patterns and cache characteristics
- **Synthesize** new traces that preserve structural properties while allowing controlled scaling
- **Benchmark** with realistic prefix-sharing patterns from production traces

## Prerequisites

- AIPerf installed and configured
- A mooncake-format trace file (JSONL format)
- Basic understanding of prefix caching and KV cache mechanics

## What is Prefix Synthesis?

In Large Language Models, **prefix caching** allows reusing previously computed KV cache entries when the same text prefix appears in multiple requests. The prefix synthesis feature helps you:

1. Understand prefix-sharing patterns in your workload
2. Generate synthetic traces that maintain these patterns
3. Scale traces (more requests, longer contexts, etc.) while preserving statistical properties

## Step 1: Analyze Your Traces

Analyze an existing trace file to extract statistics:

```bash
aiperf analyze-trace traces/production.jsonl \
    --block-size 512 \
    --output-file analysis.json
```

Output example:
```
Trace Analysis Report
============================================================
Total requests:        10,000
Unique prefixes:       2,543
Cache hit rate:        68.5%
Prefix reuse ratio:    45.2%

ISL (Input Sequence Length):
  Min:     512
  P25:     1,024
  Median:  1,920
  P75:     2,816
  Max:     4,096
  Mean:    1,920.3
  Std Dev: 856.2

OSL (Output Sequence Length):
  Min:     64
  P25:     96
  Median:  144
  P75:     208
  Max:     512
  Mean:    156.8
  Std Dev: 72.4
============================================================
```

### Understanding the Statistics

**Summary metrics:**
- **Total requests**: Number of individual requests in the trace
- **Unique prefixes**: How many distinct prefix patterns were observed
- **Cache hit rate**: Percentage of tokens that could be reused (assuming infinite cache)
- **Prefix reuse ratio**: How many prefixes appear more than once

**Percentile statistics** (computed for ISL, OSL, context length, unique prompt length, and hit rate):

| Statistic | Description |
|-----------|-------------|
| `min` | Minimum value |
| `p25` | 25th percentile (Q1) |
| `median` | 50th percentile (P50) |
| `p75` | 75th percentile (Q3) |
| `max` | Maximum value |
| `mean` | Arithmetic mean |
| `std_dev` | Standard deviation (population) |

Percentiles are calculated using linear interpolation: for percentile `p` with `n` sorted values, compute index `k = (n - 1) * p`, then interpolate between `values[floor(k)]` and `values[ceil(k)]`.

These metrics help you understand how much prefix caching could benefit your workload.

## Step 2: Run Benchmarks with Synthesis Parameters

Synthesis happens automatically when you run `aiperf profile` with mooncake traces and synthesis parameters. The trace is transformed in-memory before benchmarking:

```bash
aiperf profile \
    --input-file traces/production.jsonl \
    --custom-dataset-type mooncake_trace \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --synthesis-speedup-ratio 1.0 \
    --synthesis-prefix-len-multiplier 1.0 \
    --synthesis-prefix-root-multiplier 1 \
    --synthesis-prompt-len-multiplier 1.0
```

This runs a benchmark using the original trace characteristics. Adjust the multipliers to scale different aspects.

### Understanding Synthesis Parameters

#### `--synthesis-speedup-ratio` (default: 1.0)
Scale timestamps to simulate faster or slower request rates:
- `1.0`: No change, request times identical
- `2.0`: 2x faster (timestamps halved)
- `0.5`: 2x slower (timestamps doubled)

Example: Simulate 2x more concurrent load:
```bash
aiperf profile \
    --input-file traces/production.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-speedup-ratio 2.0 \
    ...
```

#### `--synthesis-prefix-len-multiplier` (default: 1.0)
Scale the length of core prefix paths (shared prefixes):
- `1.0`: No change
- `1.5`: Extend shared prefixes by 50%
- `0.5`: Reduce shared prefixes by 50%

Example: Simulate longer context windows:
```bash
aiperf profile \
    --input-file traces/production.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-prefix-len-multiplier 1.5 \
    ...
```

#### `--synthesis-prefix-root-multiplier` (default: 1)
Distribute traces across N independent radix trees:
- `1`: All traces share the same prefix tree (default)
- `2`: Traces randomly assigned to 2 independent trees (50% each)
- `3`: Traces randomly assigned to 3 independent trees (33% each)

Each tree has identical structure but different hash IDs, so traces in different trees cannot share prefixes. This reduces the effective cache hit rate by splitting the workload.

Example: Simulate lower cache hit rates with more diverse prefix roots:
```bash
aiperf profile \
    --input-file traces/production.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-prefix-root-multiplier 3 \
    ...
```

#### `--synthesis-prompt-len-multiplier` (default: 1.0)
Scale the length of unique prompts (non-shared portions):
- `1.0`: No change
- `2.0`: Double unique prompt lengths
- `0.5`: Halve unique prompt lengths

Example: Simulate shorter user prompts:
```bash
aiperf profile \
    --input-file traces/production.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-prompt-len-multiplier 0.7 \
    ...
```

#### `--synthesis-max-isl` (optional)
Cap the maximum input sequence length:
- Not set: No cap
- `4096`: Maximum 4,096 tokens per request

Example: Test with bounded context:
```bash
aiperf profile \
    --input-file traces/production.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-max-isl 4096 \
    ...
```

## Advanced Examples

### Scenario 1: Simulate High Cache Hit Rate

Analyze original traces to understand their cache characteristics, then benchmark with boosted prefix reuse:

```bash
# Analyze original
aiperf analyze-trace prod.jsonl --output-file analysis.json

# Benchmark with more prefix reuse
aiperf profile \
    --input-file prod.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-prefix-root-multiplier 5 \
    --synthesis-prompt-len-multiplier 0.8 \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat
```

### Scenario 2: Load Testing with Scaled Timeline

Compress timestamps to simulate 10x faster request rate:

```bash
aiperf profile \
    --input-file prod.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-speedup-ratio 10.0 \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat
```

### Scenario 3: Stress Testing with Extended Context

Benchmark with longer contexts while maintaining prefix patterns:

```bash
aiperf profile \
    --input-file prod.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-prefix-len-multiplier 2.0 \
    --synthesis-max-isl 8192 \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat
```

### Scenario 4: Controlled Multi-Turn Simulation

Benchmark with more diverse prefix patterns for multi-turn scenarios:

```bash
aiperf profile \
    --input-file prod.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-prefix-root-multiplier 10 \
    --synthesis-prompt-len-multiplier 1.2 \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat
```

## Understanding Trace Format

The mooncake trace format is JSONL (JSON Lines), where each line is a JSON object representing one request:

```json
{
  "input_length": 512,
  "output_length": 128,
  "timestamp": 0,
  "hash_ids": [1, 2, 3],
  "session_id": "session-1",
  "delay": 0
}
```

**Required fields:**
- `input_length`: Number of input tokens

**Optional fields:**
- `output_length`: Expected output tokens
- `timestamp`: Absolute timestamp in milliseconds (for fixed schedules)
- `hash_ids`: List of hash IDs representing prefix blocks
- `session_id`: Conversation/session identifier for multi-turn
- `delay`: Milliseconds to wait before sending (for multi-turn)

## Tips and Best Practices

### 1. Analyze Before Benchmarking
Always run `analyze-trace` first to understand your data:
```bash
aiperf analyze-trace your_trace.jsonl --output-file analysis.json
```

### 2. Start with Small Changes
Test parameters incrementally rather than changing everything at once:
```bash
# Test prefix scaling alone
aiperf profile \
    --input-file traces/production.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-prefix-len-multiplier 1.2 \
    --model Qwen/Qwen3-0.6B --endpoint-type chat

# Test speedup alone
aiperf profile \
    --input-file traces/production.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-speedup-ratio 2.0 \
    --model Qwen/Qwen3-0.6B --endpoint-type chat
```

### 3. Compare Multiple Parameter Sets
Run benchmarks with different synthesis parameters to compare:
```bash
# Conservative: slight increase in cache hits
aiperf profile \
    --input-file prod.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-prefix-len-multiplier 1.1 \
    --synthesis-prefix-root-multiplier 2 \
    --model Qwen/Qwen3-0.6B --endpoint-type chat

# Aggressive: strong cache hit focus
aiperf profile \
    --input-file prod.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-prefix-len-multiplier 2.0 \
    --synthesis-prefix-root-multiplier 5 \
    --model Qwen/Qwen3-0.6B --endpoint-type chat

# Load test: faster request rate
aiperf profile \
    --input-file prod.jsonl \
    --custom-dataset-type mooncake_trace \
    --synthesis-speedup-ratio 5.0 \
    --model Qwen/Qwen3-0.6B --endpoint-type chat
```

### 4. Preserve Real Patterns
The synthesis preserves statistical properties. For best results:
- Use realistic input traces from production
- Avoid extreme multiplier values (typically 0.5-3.0)
- Compare results against baseline (no synthesis parameters)

## Troubleshooting

### Issue: "Input file not found"
```
Error: Input file not found: traces/production.jsonl
```
**Solution:** Verify the file path is correct:
```bash
ls -la traces/production.jsonl
```

### Issue: "No unique prefixes found"
```
Total requests: 1000
Unique prefixes: 0
```
**Solution:** Your trace file doesn't have `hash_ids`. Synthesis will still work with `input_length` and `output_length` fields, but prefix caching information won't be available.

### Issue: Low cache hit rate
```
Cache hit rate: 5.2%
```
**Solution:** Your workload has low prefix reuse. Try:
- Increasing `--synthesis-prefix-len-multiplier` to extend shared prefixes
- Using `--synthesis-prefix-root-multiplier` to create more diverse patterns
- Analyzing a different trace file that has more reuse


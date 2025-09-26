<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Time-based Benchmarking

Time-based benchmarking allows you to run benchmarks for a specific duration rather than a fixed number of requests.
This approach is ideal for measuring sustained performance and testing service stability over time.

## Overview

Time-based benchmarking provides several advantages:

- **Consistent Measurement Window**: Compare performance across different configurations using the same time duration
- **Real-world Simulation**: Mirror production scenarios where load is sustained over time
- **Resource Utilization**: Identify memory leaks, connection pooling issues, and resource exhaustion patterns
- **SLA Validation**: Establish and verify performance guarantees over specific time periods
- **Grace Period Control**: Handle in-flight requests gracefully or force immediate completion as needed

## Core Parameters

### Benchmark Duration
- `--benchmark-duration SECONDS`: Total time to run the benchmark
- Requests are sent continuously until duration expires

### Grace Period
- `--benchmark-grace-period SECONDS`: Time to wait for in-flight requests after duration expires
- Default: 30 seconds
- Set to 0 for immediate completion when duration ends

## Basic Time-based Testing

### Setting Up the Server

```bash
# Start vLLM server for time-based benchmarking
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

### Duration Testing

Run brief performance checks to quickly validate service health:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Run 30-second benchmark with concurrency
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --benchmark-duration 30.0 \
    --benchmark-grace-period 15.0 \
    --synthetic-input-tokens-mean 200 \
    --synthetic-input-tokens-stddev 50 \
    --output-tokens-mean 100 \
    --output-tokens-stddev 20 \
    --concurrency 5 \
    --warmup-request-count 3 \
    --random-seed 33333
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->


## Use Cases

> [!TIP]
> **When to Use Time-based Benchmarking:**
> - **SLA Validation**: Verify performance meets requirements over time
> - **Capacity Planning**: Determine sustainable load levels
> - **Stability Testing**: Identify performance degradation over time
> - **Resource Planning**: Understand resource consumption patterns
> - **Production Readiness**: Validate service stability before deployment

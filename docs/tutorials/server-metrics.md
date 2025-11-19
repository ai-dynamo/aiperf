<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Server Metrics Collection with AIPerf

This guide shows you how to use AIPerf's automatic server metrics collection feature. Server metrics provide insights into LLM inference server performance, including request counts, latencies, cache utilization, and custom application metrics.

## Overview

AIPerf **automatically collects metrics by default** from Prometheus-compatible endpoints exposed by LLM inference servers like vLLM, SGLang, TRT-LLM, and others. These metrics complement AIPerf's client-side measurements with server-side observability data.

**What You'll Learn:**
- How automatic server metrics collection works (enabled by default)
- Configure additional custom Prometheus endpoints
- Understand the output files and data format
- Use server metrics for performance analysis

## Quick Start

### Basic Usage

Server metrics are **automatically collected** - just run AIPerf normally:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url localhost:8000 \
    --concurrency 4 \
    --request-count 100
```

**What happens automatically:**
1. AIPerf discovers and queries Prometheus `/metrics` endpoint on your inference server (checks `--url` port)
2. Collects metrics every 100ms (configurable)
3. Exports time-series data to JSONL file
4. Saves metadata about collected metrics

> [!TIP]
> No flag needed! Server metrics are collected by default. Use `--server-metrics <urls>` to add additional endpoints, or `--no-server-metrics` to disable.

### Automatic Endpoint Discovery

By default, AIPerf automatically:

1. **Extracts the hostname and port** from your inference endpoint URL (`--url`)
2. **Tests the same port** as your inference endpoint on that hostname
3. **Optionally tests additional ports** if configured via `AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS`
4. **Collects from all reachable endpoints that respond to the `/metrics` endpoint**

**Example:**
```bash
# Inference URL: http://gpu-server:8000/v1/chat/completions
# AIPerf automatically tries:
#   - http://gpu-server:8000/metrics  (same port as inference endpoint)
#   - Plus any ports specified in AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS (if set)
```

> [!NOTE]
> By default, `AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS` is empty. AIPerf only checks the inference endpoint's port unless you configure additional ports to check.

### Custom Endpoint URLs

Specify custom Prometheus endpoints explicitly:

```bash
# Single custom endpoint
aiperf profile --model MODEL ... --server-metrics http://localhost:9400/metrics

# Multiple endpoints (multi-node or multiple services)
aiperf profile --model MODEL ... --server-metrics \
    http://node1:8081/metrics \
    http://node2:8081/metrics \
    http://monitoring:9090/metrics
```

> [!NOTE]
> URLs can be specified with or without the `http://` prefix and `/metrics` suffix. AIPerf normalizes them automatically:
> - `localhost:8081` → `http://localhost:8081/metrics`
> - `http://server:9400` → `http://server:9400/metrics`

### Disabling Server Metrics

To disable automatic server metrics collection:

```bash
aiperf profile --model MODEL ... --no-server-metrics
```

This completely disables server metrics collection for the run.

## Understanding Server Metrics

### What Metrics Are Collected?

AIPerf collects **all metrics** exposed by Prometheus-compatible endpoints, with automatic filtering:

- **Collected:** All counter, gauge, histogram, and summary metrics
- **Automatically Filtered:** Metrics ending with `_created` (internal Prometheus timestamps)

Common metrics from LLM inference servers include:

#### vLLM Metrics Examples
- **Request Metrics:** `vllm:request_success_total`, `vllm:request_failure_total`
- **Latency Distributions:** `vllm:time_to_first_token_seconds`, `vllm:e2e_request_latency_seconds`
- **Cache Performance:** `vllm:gpu_cache_usage_perc`, `vllm:cpu_cache_usage_perc`
- **Throughput:** `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`

#### Prometheus Metric Types Supported
- **Counter:** Cumulative values (e.g., total requests, total tokens)
- **Gauge:** Point-in-time values (e.g., cache utilization %)
- **Histogram:** Distribution with buckets (e.g., latency percentiles)
- **Summary:** Pre-computed quantiles (e.g., p50, p90, p99)

### Output Files

AIPerf generates two files per benchmark run:

#### 1. Time-Series Data: `server_metrics_export.jsonl`

Line-delimited JSON with metrics snapshots collected over time:

```json
{"endpoint_url":"http://localhost:8081/metrics","timestamp_ns":1700000000000000000,"endpoint_latency_ns":5234567,"metrics":{"vllm:request_success_total":[{"labels":{"model_name":"llama"},"value":150.0}],"vllm:gpu_cache_usage_perc":[{"labels":{"model_name":"llama"},"value":87.5}],"vllm:time_to_first_token_seconds":[{"labels":{"model_name":"llama"},"histogram":{"0.001":5,"0.005":12,"0.01":25,"0.1":100,"+Inf":100},"sum":45.2,"count":100}]}}
{"endpoint_url":"http://localhost:8081/metrics","timestamp_ns":1700000100000000,"endpoint_latency_ns":5123456,"metrics":{"vllm:request_success_total":[{"labels":{"model_name":"llama"},"value":152.0}],"vllm:gpu_cache_usage_perc":[{"labels":{"model_name":"llama"},"value":88.2}],"vllm:time_to_first_token_seconds":[{"labels":{"model_name":"llama"},"histogram":{"0.001":5,"0.005":12,"0.01":27,"0.1":102,"+Inf":102},"sum":46.8,"count":102}]}}
```

**Each line contains:**
- `endpoint_url`: Source Prometheus endpoint
- `timestamp_ns`: Collection timestamp (nanoseconds since epoch)
- `endpoint_latency_ns`: Time to fetch metrics from endpoint (nanoseconds)
- `metrics`: Dictionary of metric families with samples
  - **Counter/Gauge:** `{"value": 42.0}` or `{"labels": {...}, "value": 42.0}`
  - **Histogram:** `{"histogram": {"le": count, ...}, "sum": X, "count": N}` (le = bucket upper bounds)
  - **Summary:** `{"summary": {"quantile": value, ...}, "sum": X, "count": N}` (quantile = percentile labels)

**Space Optimization with Deduplication:**
The file is automatically **deduplicated** per endpoint to reduce file size while preserving accurate timeline information:

1. **First occurrence** of metrics → always written (marks start of period)
2. **Consecutive identical metrics** → skipped and counted
3. **Change detected** → last duplicate written (marks end of period), then new record written (marks start of new period)

**Example:** Input `A,A,A,B,B,C,D,D,D,D` → Output `A,A,B,B,C,D,D`

This ensures you have actual timestamp observations for when metrics changed, enabling accurate duration calculations and time-series analysis. Deduplication uses equality comparison on the metrics dictionary for each endpoint separately.

#### 2. Metadata: `server_metrics_metadata.json`

Pretty-printed JSON with metric schemas and documentation:

```json
{
  "endpoints": {
    "http://localhost:8081/metrics": {
      "endpoint_url": "http://localhost:8081/metrics",
      "endpoint_display": "http://localhost:8081/metrics",
      "metric_schemas": {
        "vllm:request_success_total": {
          "type": "counter",
          "help": "Total number of successful requests",
          "bucket_labels": null,
          "quantile_labels": null
        },
        "vllm:time_to_first_token_seconds": {
          "type": "histogram",
          "help": "Histogram of time to first token in seconds",
          "bucket_labels": ["0.001", "0.005", "0.01", "0.025", "0.05", "0.075", "0.1", "0.25", "0.5", "0.75", "1.0", "2.5", "5.0", "7.5", "10.0", "+Inf"],
          "quantile_labels": null
        }
      }
    }
  }
}
```

**Contains:**
- Metric types and help text
- Histogram bucket labels (for understanding distributions)
- Summary quantile labels (for pre-computed percentiles)
- Endpoint URLs and display names (currently identical)

> [!TIP]
> Use the metadata file to understand what metrics are available and how to interpret the JSONL data.

> [!NOTE]
> **Output Directory Structure:** Files are created in `artifacts/{run-name}/` where `{run-name}` is automatically generated from your model, endpoint type, schedule, and concurrency (e.g., `Qwen_Qwen3-0.6B-openai-chat-concurrency4`).
>
> **Custom Filenames:** When using `--profile-export-prefix custom_name`, files become:
> - `artifacts/{run-name}/custom_name_server_metrics.jsonl`
> - `artifacts/{run-name}/custom_name_server_metrics_metadata.json`

## Example Workflow: Benchmarking vLLM

### Setup vLLM with Prometheus Metrics

vLLM exposes Prometheus metrics on port 8000 by default (same as the inference endpoint):

```bash
# Start vLLM with Prometheus metrics enabled
docker run -d --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000

# Wait for server to be ready
sleep 30
```

### Verify Prometheus Endpoint

```bash
# Check if metrics are exposed
curl http://localhost:8000/metrics | head -20

# Expected output:
# # HELP vllm:request_success_total Total number of successful requests
# # TYPE vllm:request_success_total counter
# vllm:request_success_total{model_name="Qwen/Qwen3-0.6B"} 0.0
# ...
```

### Run AIPerf with Server Metrics

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url localhost:8000 \
    --streaming \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 100 \
    --warmup-request-count 10 \
    --server-metrics
```

> [!NOTE]
> Since vLLM exposes metrics on the same port as the inference endpoint (8000), AIPerf automatically discovers it at `http://localhost:8000/metrics`. You can also explicitly specify it with `--server-metrics http://localhost:8000/metrics` if preferred.

### Inspect Output Files

```bash
# Check what files were generated (run creates a subdirectory in artifacts/)
ls -lh artifacts/*/server_metrics*

# Output example (in artifacts/{model-endpoint-schedule-concurrency}/):
# server_metrics_export.jsonl       # Time-series data (one JSON object per line)
# server_metrics_metadata.json      # Metric schemas and descriptions

# Count how many metric snapshots were collected
wc -l artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency4/server_metrics_export.jsonl

# View metadata file (pretty-printed JSON)
cat artifacts/Qwen_Qwen3-0.6B-openai-chat-concurrency4/server_metrics_metadata.json
```

## Configuration Options

### Environment Variables

Customize collection behavior with environment variables:

```bash
# Collect metrics every 500ms instead of default 100ms
export AIPERF_SERVER_METRICS_COLLECTION_INTERVAL=0.5

# Configure default ports for automatic discovery
export AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS="8081,6880,9400"

# Increase flush period to 5 seconds (wait longer for final metrics after benchmark ends)
export AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD=5.0

# Run benchmark
aiperf profile --model MODEL ... --server-metrics
```

**Available Settings:**

| Environment Variable | Default | Range | Description |
|---------------------|---------|-------|-------------|
| `AIPERF_SERVER_METRICS_COLLECTION_INTERVAL` | 0.1s | 0.01s - 300s | Metrics collection frequency |
| `AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD` | 2.0s | 0.0s - 30s | Wait time for final metrics after benchmark |
| `AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS` | empty list | comma-separated | Additional ports to check during auto-discovery (beyond inference endpoint port) |
| `AIPERF_SERVER_METRICS_REACHABILITY_TIMEOUT` | 5s | 1s - 300s | Timeout for endpoint reachability tests |

## Multi-Node Server Metrics

For distributed LLM deployments (tensor parallelism, pipeline parallelism), collect metrics from all nodes:

```bash
# Example: 4-node distributed vLLM deployment
aiperf profile \
    --model meta-llama/Llama-2-70b-hf \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url http://head-node:8000 \
    --concurrency 16 \
    --request-count 500 \
    --server-metrics \
        http://node-0:8081/metrics \
        http://node-1:8081/metrics \
        http://node-2:8081/metrics \
        http://node-3:8081/metrics
```

**Output Structure:**
Each endpoint's metrics are stored separately in the JSONL file with its `endpoint_url` field, allowing you to:
- Analyze per-node performance
- Detect load imbalances
- Monitor distributed system health

## Understanding the Data Format

### JSONL Record Structure

Each line in `server_metrics_export.jsonl` is a JSON object containing ALL metrics from one endpoint at one point in time:

```json
{
  "endpoint_url": "http://localhost:8081/metrics",
  "timestamp_ns": 1700000000000000000,
  "endpoint_latency_ns": 5234567,
  "metrics": {
    "dynamo_frontend_requests_total": [
      {
        "labels": {"job": "inference-server", "instance": "localhost:8081", "server_id": "server-0"},
        "value": 15420.0
      }
    ],
    "dynamo_frontend_queued_requests": [
      {
        "labels": {"job": "inference-server", "instance": "localhost:8081", "server_id": "server-0"},
        "value": 23.0
      }
    ],
    "kvstats_gpu_cache_usage_percent": [
      {
        "labels": {"job": "inference-server", "instance": "localhost:8081", "server_id": "server-0"},
        "value": 0.847
      }
    ],
    "dynamo_frontend_time_to_first_token_seconds": [
      {
        "labels": {"job": "inference-server", "instance": "localhost:8081", "server_id": "server-0"},
        "histogram": {
          "0.001": 5,
          "0.005": 42,
          "0.01": 128,
          "0.02": 315,
          "0.04": 412,
          "+Inf": 450
        },
        "sum": 8.234,
        "count": 450
      }
    ],
    "dynamo_frontend_request_duration_seconds": [
      {
        "labels": {"job": "inference-server", "instance": "localhost:8081", "server_id": "server-0"},
        "histogram": {
          "0.01": 12,
          "0.025": 45,
          "0.05": 123,
          "0.1": 302,
          "0.5": 438,
          "+Inf": 450
        },
        "sum": 67.8,
        "count": 450
      }
    ],
    "http_request_latency_seconds": [
      {
        "labels": {"job": "inference-server", "instance": "localhost:8081", "server_id": "server-0"},
        "summary": {
          "0.5": 0.023,
          "0.9": 0.087,
          "0.99": 0.234
        },
        "sum": 145.6,
        "count": 1500
      }
    ]
  }
}
```

**Top-Level Fields:**
- `endpoint_url`: Source Prometheus endpoint URL
- `timestamp_ns`: Unix timestamp in nanoseconds when metrics were collected
- `endpoint_latency_ns`: Time taken to fetch metrics from endpoint (nanoseconds)
- `metrics`: Dictionary containing ALL metrics from this endpoint at this timestamp

**Sample Structure by Type:**
- **Counter/Gauge:** `{"labels": {...}, "value": N}`
- **Histogram:** `{"labels": {...}, "histogram": {"le": count, ...}, "sum": N, "count": N}`
- **Summary:** `{"labels": {...}, "summary": {"quantile": value, ...}, "sum": N, "count": N}`

### Multi-Endpoint Data (Interleaved)

When collecting from multiple endpoints, records are **interleaved by write time** (when deduplication completes), not strictly by collection time:

```jsonl
{"endpoint_url":"http://node1:8081/metrics","timestamp_ns":1700000000000000,"metrics":{...}}
{"endpoint_url":"http://node1:8081/metrics","timestamp_ns":1700000300000000,"metrics":{...}}
{"endpoint_url":"http://node2:8081/metrics","timestamp_ns":1700000000050000,"metrics":{...}}
{"endpoint_url":"http://node2:8081/metrics","timestamp_ns":1700000400050000,"metrics":{...}}
{"endpoint_url":"http://node1:8081/metrics","timestamp_ns":1700000500000000,"metrics":{...}}
{"endpoint_url":"http://node2:8081/metrics","timestamp_ns":1700000500050000,"metrics":{...}}
```

**Key Points:**
- Records are **NOT** strictly alternating between endpoints
- **Deduplication** causes multiple consecutive records from same endpoint (first occurrence + last duplicate before change)
- Use `endpoint_url` field to filter/group by endpoint during analysis
- Each endpoint is collected and deduplicated independently
- Timestamps reflect actual collection times, not write order

**Example with Deduplication:**
```
Collection Cycle    node1 metrics    node2 metrics    Written to File
t=0ms              A (new)          X (new)          node1: A@0ms
t=100ms            A (dup)          X (dup)          (skipped - duplicates)
t=200ms            A (dup)          X (dup)          (skipped - duplicates)
t=300ms            A (dup)          Y (change!)      node1: A@300ms (last dup)
                                                     node2: X@0ms (first)
t=400ms            B (change!)      Y (dup)          node2: X@400ms (last dup)
                                                     node1: B@300ms (first)
                                                     node2: Y@400ms (new)
```

Notice: node1 records appear twice in a row because deduplication wrote the last duplicate (A@300ms) then the new value (B@300ms).

### Metadata File Structure

The `server_metrics_metadata.json` file describes all collected metrics:

```json
{
  "endpoints": {
    "endpoint_url": {
      "endpoint_url": "http://localhost:8081/metrics",
      "endpoint_display": "http://localhost:8081/metrics",
      "metric_schemas": {
        "metric_name": {
          "type": "counter|gauge|histogram|summary",
          "help": "Metric description from Prometheus",
          "bucket_labels": ["0.001", "0.01", ...],  // histogram only
          "quantile_labels": ["0.5", "0.9", "0.99"]  // summary only
        }
      }
    }
  }
}
```

**Use the metadata file to:**
- Discover what metrics are available
- Understand metric types and meanings
- Know histogram bucket boundaries and summary quantiles

## Troubleshooting

### No Metrics Collected

**Symptoms:**
- No `*_server_metrics.jsonl` file generated
- Console shows "0 endpoints reachable"

**Solutions:**

1. **Verify Prometheus endpoint is accessible:**
   ```bash
   curl http://localhost:8081/metrics
   ```
   If this fails, check:
   - Is the inference server running?
   - Is Prometheus exposition enabled?
   - Are you using the correct port?

2. **Check automatic discovery is trying the right port:**
   - By default, AIPerf only checks the same port as your inference endpoint
   - If your Prometheus metrics are on a different port, either:
     ```bash
     # Option 1: Set additional ports for discovery
     export AIPERF_SERVER_METRICS_DEFAULT_BACKEND_PORTS="8081,9400"

     # Option 2: Explicitly specify the endpoint (recommended)
     aiperf profile --model MODEL ... --server-metrics http://localhost:8081/metrics
     ```

3. **Use explicit URLs instead of auto-discovery:**
   ```bash
   aiperf profile --model MODEL ... --server-metrics http://localhost:8000/metrics
   ```

### Empty JSONL File

**Symptoms:**
- File exists but contains no data or only a few lines

**Solutions:**

1. **Check collection interval:**
   ```bash
   # Collect less frequently if metrics aren't changing
   export AIPERF_SERVER_METRICS_COLLECTION_INTERVAL=1.0
   ```

2. **Verify metrics are exposed:**
   ```bash
   # Check if endpoint returns metric data
   curl http://localhost:8081/metrics | grep -c "^vllm:"
   ```

3. **Review AIPerf logs for errors:**
   ```bash
   # Look for ServerMetricsManager messages
   aiperf profile ... --verbose
   ```

### High File Sizes

**Symptoms:**
- JSONL file is very large (>100MB for short benchmarks)

**Solutions:**

1. **Reduce collection frequency:**
   ```bash
   export AIPERF_SERVER_METRICS_COLLECTION_INTERVAL=1.0  # Collect every 1 second
   ```

2. **Limit collection to specific endpoints:**
   ```bash
   # Only collect from primary node instead of all nodes
   aiperf profile ... --server-metrics http://primary-node:8081/metrics
   ```

3. **Check deduplication is working:**
   - The JSONL writer automatically deduplicates unchanged metrics
   - If file is still large, metrics are changing frequently (expected for active workloads)

### Endpoint Timeout Errors

**Symptoms:**
- Console shows "endpoint unreachable" or timeout errors

**Solutions:**

1. **Increase reachability timeout:**
   ```bash
   export AIPERF_SERVER_METRICS_REACHABILITY_TIMEOUT=10
   ```

2. **Check network connectivity:**
   ```bash
   # Test latency to endpoint
   time curl -s http://localhost:8081/metrics > /dev/null
   ```

3. **Verify endpoint health:**
   - Some servers may be slow to respond during high load
   - Check server logs for errors

## Next Steps

- **GPU Telemetry:** Combine server metrics with [GPU telemetry](gpu-telemetry.md) for comprehensive observability<br>

## Summary

Server metrics collection in AIPerf provides:

✅ **Enabled by default** - automatic discovery of Prometheus endpoints (checks inference endpoint port)<br>
✅ **Comprehensive collection** of all exposed metrics (counters, gauges, histograms, summaries)<br>
✅ **Efficient storage** with automatic deduplication (per endpoint)<br>
✅ **Multi-node support** for distributed deployments<br>
✅ **Easy analysis** with JSONL format and metadata schemas<br>

No configuration needed - just run AIPerf and server metrics are collected automatically!<br>

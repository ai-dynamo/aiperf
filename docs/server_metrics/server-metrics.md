<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Server Metrics Collection

AIPerf automatically collects metrics from Prometheus-compatible endpoints exposed by LLM inference servers (vLLM, SGLang, TRT-LLM, Dynamo, etc.).

## Quick Reference

| Feature | Description | Default |
|---------|-------------|---------|
| **Auto-discovery** | Automatically finds `/metrics` endpoint on server URL | Enabled |
| **Collection** | Scrapes metrics every 333ms during benchmark | Enabled |
| **Outputs** | JSON (aggregated), CSV (tabular), JSONL (time-series), Parquet (cumulative deltas) | JSON + CSV |
| **Custom endpoints** | `--server-metrics URL [URL...]` for additional endpoints | None |
| **Disable** | `--no-server-metrics` to turn off collection | Enabled |
| **Windowed stats** | `--slice-duration SECONDS` for time-sliced analysis | Off |

**Key metrics by server:**

<details open>
<summary><b>vLLM</b></summary>

| Metric | Type | What to Watch |
|--------|------|---------------|
| `vllm:num_requests_running` | gauge | Active batch size (`stats.avg`) |
| `vllm:num_requests_waiting` | gauge | Queue depth—growing = saturation (`stats.max`) |
| `vllm:kv_cache_usage_perc` | gauge | >0.9 = capacity limit (`stats.max`) |
| `vllm:num_preemptions` | counter | >0 = memory pressure (`stats.total`) |
| `vllm:e2e_request_latency_seconds` | histogram | E2E latency (`stats.p99_estimate`) |
| `vllm:time_to_first_token_seconds` | histogram | TTFT (`stats.p99_estimate`) |
| `vllm:inter_token_latency_seconds` | histogram | ITL (`stats.p99_estimate`) |
| `vllm:generation_tokens` | counter | Decode throughput (`stats.rate`) |

</details>

<details open>
<summary><b>Dynamo</b></summary>

| Metric | Type | What to Watch |
|--------|------|---------------|
| `dynamo_frontend_inflight_requests` | gauge | Active requests (`stats.avg`) |
| `dynamo_frontend_queued_requests` | gauge | Requests awaiting first token (`stats.avg`) |
| `dynamo_frontend_request_duration_seconds` | histogram | E2E latency (`stats.p99_estimate`) |
| `dynamo_frontend_time_to_first_token_seconds` | histogram | TTFT (`stats.p99_estimate`) |
| `dynamo_frontend_inter_token_latency_seconds` | histogram | ITL (`stats.p99_estimate`) |
| `dynamo_frontend_requests` | counter | Throughput (`stats.rate`) |
| `dynamo_component_kvstats_gpu_cache_usage_percent` | gauge | Backend cache usage (`stats.max`) |

</details>

<details open>
<summary><b>SGLang</b></summary>

| Metric | Type | What to Watch |
|--------|------|---------------|
| `sglang:num_running_reqs` | gauge | Active batch size (`stats.avg`) |
| `sglang:num_queue_reqs` | gauge | Queue depth—growing = saturation (`stats.max`) |
| `sglang:token_usage` | gauge | >0.9 = capacity limit (`stats.max`) |
| `sglang:cache_hit_rate` | gauge | Prefix cache efficiency (`stats.avg`) |
| `sglang:gen_throughput` | gauge | Real-time tokens/s (`stats.avg`) |
| `sglang:queue_time_seconds` | histogram | Queue wait (`stats.p99_estimate`) |

</details>

<details open>
<summary><b>TRT-LLM</b></summary>

| Metric | Type | What to Watch |
|--------|------|---------------|
| `trtllm:e2e_request_latency_seconds` | histogram | E2E latency (`stats.p99_estimate`) |
| `trtllm:time_to_first_token_seconds` | histogram | TTFT (`stats.p99_estimate`) |
| `trtllm:time_per_output_token_seconds` | histogram | ITL (`stats.p99_estimate`) |
| `trtllm:request_queue_time_seconds` | histogram | Queue wait (`stats.p99_estimate`) |
| `trtllm:request_success` | counter | Completed requests (`stats.rate`) |

</details>

## Quick Start

Server metrics are **collected by default** - just run AIPerf normally:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url localhost:8000 \
    --concurrency 4 \
    --request-count 100
```

AIPerf automatically:
1. Discovers the `/metrics` endpoint on your inference server (base URL + `/metrics`)
2. Tests endpoint reachability before profiling starts
3. Captures baseline metrics before warmup period begins (reference point for deltas)
4. Collects metrics at configurable intervals during warmup and profiling
5. Performs final scrape after profiling completes (captures end state)
6. Exports selected formats (default: JSON + CSV):
   - `server_metrics_export.json` - Aggregated statistics (profiling period only)
   - `server_metrics_export.csv` - Tabular format (profiling period only)
   - `server_metrics_export.jsonl` - Time-series data (all scrapes, opt-in only)
   - `server_metrics_export.parquet` - Raw time-series with delta calculations (opt-in only)

> [!NOTE]
> **Custom file naming:** The `--profile-export-prefix` (or `--profile-export-file`) flag changes the prefix for all export files, including server metrics. Any file extension is automatically stripped from the provided value. For example:
> ```bash
> aiperf profile --model MODEL ... --profile-export-prefix my_benchmark
> # Produces: my_benchmark_server_metrics.json, my_benchmark_server_metrics.csv, etc.
>
> # --profile-export-file is an alias for --profile-export-prefix, so this is equivalent:
> aiperf profile --model MODEL ... --profile-export-file my_benchmark.json
> # Produces the same files (the .json extension is stripped automatically)
> ```

**Time filtering:** Statistics in JSON/CSV exports exclude the warmup period, showing only metrics from the profiling phase. The JSONL file contains all scrapes (including warmup) for complete time-series analysis.

**Format selection:** By default, only JSON and CSV formats are generated to avoid large JSONL files. To include JSONL for time-series analysis:
```bash
aiperf profile --model MODEL ... --server-metrics-formats json csv jsonl
```

### Adding Custom Endpoints

```bash
# Single endpoint
aiperf profile --model MODEL ... --server-metrics http://localhost:8081

# Multiple endpoints (distributed deployment)
aiperf profile --model MODEL ... --server-metrics \
    http://node1:8081 \
    http://node2:8081
```

### Disabling Server Metrics

```bash
aiperf profile --model MODEL ... --no-server-metrics
```

### Selecting Output Formats

```bash
# Default: JSON + CSV only
aiperf profile --model MODEL ...

# Add time-series formats as needed
aiperf profile --model MODEL ... --server-metrics-formats json csv parquet
aiperf profile --model MODEL ... --server-metrics-formats json csv jsonl parquet
```

| Format | Use Case | Size |
|--------|----------|------|
| **JSON/CSV** (default) | Summary statistics, CI/CD thresholds | Small |
| **Parquet** | SQL queries, pandas/DuckDB analytics | Compressed |
| **JSONL** | Debugging, raw Prometheus snapshots | 10-100x larger |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AIPERF_SERVER_METRICS_COLLECTION_INTERVAL` | 0.333s | Collection frequency (333ms, ~3Hz) |
| `AIPERF_SERVER_METRICS_COLLECTION_FLUSH_PERIOD` | 2.0s | Wait time for final metrics after benchmark |
| `AIPERF_SERVER_METRICS_REACHABILITY_TIMEOUT` | 10s | Timeout for endpoint reachability tests |
| `AIPERF_SERVER_METRICS_EXPORT_BATCH_SIZE` | 100 | Batch size for JSONL writer |
| `AIPERF_SERVER_METRICS_SHUTDOWN_DELAY` | 5.0s | Shutdown delay for command response transmission |

## Output Files

> [!NOTE]
> The filenames below are defaults. When `--profile-export-prefix <prefix>` is used, server metrics files are named `<prefix>_server_metrics.{json,csv,jsonl,parquet}` (any file extension in the prefix is stripped automatically). All files are written to the artifact directory (`--artifact-directory`, default: `./artifacts/<run_info>`).

### 1. Time-Series: `server_metrics_export.jsonl`

Line-delimited JSON with metrics snapshots over time:

```json
{
  "endpoint_url": "http://localhost:8000/metrics",
  "timestamp_ns": 1763591215220757503,
  "endpoint_latency_ns": 719764167,
  "metrics": {
    "vllm:num_requests_running": [{"value": 12.0}],
    "vllm:kv_cache_usage_perc": [{"value": 0.72}],
    "vllm:request_success": [{"value": 1500.0}],
    "vllm:time_to_first_token_seconds": [{
      "buckets": {"0.01": 145.0, "0.1": 1498.0, "+Inf": 1500.0},
      "sum": 32.456,
      "count": 1500.0
    }]
  },
  "request_sent_ns": 1763591214500993336,
  "first_byte_ns": 1763591215220757503
}
```

**Fields:**
- `endpoint_url`: Source Prometheus endpoint
- `timestamp_ns`: Collection timestamp in nanoseconds
- `endpoint_latency_ns`: HTTP round-trip time in nanoseconds
- `metrics`: All metrics from this endpoint
  - Counter/Gauge: `{"value": N}` or `{"labels": {...}, "value": N}`
  - Histogram: `{"buckets": {"le": count}, "sum": N, "count": N}` with optional labels

### 2. Aggregated Statistics: `server_metrics_export.json`

Aggregated statistics from profiling period. Metrics from all endpoints are merged, each series tagged with `endpoint_url`.

```json
{
  "schema_version": "1.0",
  "aiperf_version": "0.3.0",
  "benchmark_id": "2900a136-3c1a-4520-adaa-5719822b729b",
  "summary": {
    "endpoints_configured": ["http://localhost:8000/metrics"],
    "endpoints_successful": ["http://localhost:8000/metrics"],
    "start_time": "2025-12-15T02:04:23.028529",
    "end_time": "2025-12-15T02:05:15.294690",
    "endpoint_info": {
      "http://localhost:8000/metrics": {
        "total_fetches": 157,
        "first_fetch_ns": 1765793061967310848,
        "last_fetch_ns": 1765793114960054143,
        "avg_fetch_latency_ms": 246.83,
        "unique_updates": 157,
        "first_update_ns": 1765793061967310848,
        "last_update_ns": 1765793114960054143,
        "duration_seconds": 52.99,
        "avg_update_interval_ms": 339.70,
        "median_update_interval_ms": 333.48
      }
    }
  },
  "metrics": {
    "vllm:kv_cache_usage_perc": {
      "type": "gauge",
      "description": "KV-cache usage. 1 means 100 percent usage.",
      "unit": "ratio",
      "series": [{
        "endpoint_url": "http://localhost:8000/metrics",
        "labels": { "engine": "0", "model_name": "Qwen/Qwen3-0.6B" },
        "stats": {
          "avg": 0.191, "min": 0.0, "max": 0.202, "std": 0.038,
          "p1": 0.003, "p5": 0.178, "p10": 0.191, "p25": 0.198,
          "p50": 0.202, "p75": 0.202, "p90": 0.202, "p95": 0.202, "p99": 0.202
        },
        "timeslices": [
          { "start_ns": 1765793063028529452, "end_ns": 1765793068028529452, "avg": 0.107, "min": 0.0, "max": 0.191 },
          { "start_ns": 1765793068028529452, "end_ns": 1765793073028529452, "avg": 0.192, "min": 0.191, "max": 0.194 }
        ]
      }]
    },
    "vllm:request_success": {
      "type": "counter",
      "description": "Count of successfully processed requests.",
      "unit": "requests",
      "series": [{
        "endpoint_url": "http://localhost:8000/metrics",
        "labels": { "engine": "0", "finished_reason": "length", "model_name": "Qwen/Qwen3-0.6B" },
        "stats": {
          "total": 19.0, "rate": 0.359,
          "rate_avg": 0.38, "rate_min": 0.0, "rate_max": 1.8, "rate_std": 0.751
        },
        "timeslices": [
          { "start_ns": 1765793063028529452, "end_ns": 1765793068028529452, "total": 0.0, "rate": 0.0 },
          { "start_ns": 1765793073028529452, "end_ns": 1765793078028529452, "total": 9.0, "rate": 1.8 }
        ]
      }]
    },
    "vllm:e2e_request_latency_seconds": {
      "type": "histogram",
      "description": "Histogram of e2e request latency in seconds.",
      "unit": "seconds",
      "series": [{
        "endpoint_url": "http://localhost:8000/metrics",
        "labels": { "engine": "0", "model_name": "Qwen/Qwen3-0.6B" },
        "stats": {
          "count": 19, "sum": 259.87, "avg": 13.68,
          "count_rate": 0.359, "sum_rate": 4.90,
          "p1_estimate": 2.25, "p5_estimate": 5.77, "p10_estimate": 8.26,
          "p25_estimate": 10.82, "p50_estimate": 13.75, "p75_estimate": 15.35,
          "p90_estimate": 17.24, "p95_estimate": 19.51, "p99_estimate": 31.77
        },
        "buckets": {
          "0.3": 0, "0.5": 0, "1.0": 0, "2.5": 1, "5.0": 1,
          "10.0": 3, "15.0": 11, "20.0": 18, "30.0": 18, "+Inf": 19
        },
        "timeslices": [
          {
            "start_ns": 1765793063028529452, "end_ns": 1765793068028529452,
            "count": 0, "sum": 0.0, "avg": 0.0,
            "buckets": { "0.3": 0, "0.5": 0, "1.0": 0, "2.5": 0, "5.0": 0, "10.0": 0, "15.0": 0, "20.0": 0, "+Inf": 0 }
          }
        ]
      }]
    }
  },
  "input_config": {
    "endpoint": { "model_names": ["Qwen/Qwen3-0.6B"], "streaming": true },
    "loadgen": { "concurrency": 400, "request_rate": 5000.0, "request_count": 30000 },
    "output": { "slice_duration": 5.0 }
  }
}
```

Query with jq:
```bash
jq '.metrics["vllm:e2e_request_latency_seconds"].series[0].stats.p99_estimate' server_metrics_export.json
```

### 3. CSV Export: `server_metrics_export.csv`

Tabular export organized in four sections (separated by blank lines): **gauge**, **counter**, **histogram**, **info**.

- Labels expanded into individual columns for easy filtering/pivoting
- Open directly in Excel/Sheets or load with pandas

```python
from io import StringIO
import pandas as pd

with open("server_metrics_export.csv") as f:
    sections = [pd.read_csv(StringIO(s)) for s in f.read().strip().split('\n\n') if s.strip()]
```

### 4. Parquet Export: `server_metrics_export.parquet`

Raw time-series data with delta calculations applied. Uses a normalized schema (~50% smaller than wide format) where histogram buckets are separate rows. Each label becomes a column for SQL filtering.

**Schema overview:**

| Column | Type | Description |
|--------|------|-------------|
| `endpoint_url` | string | Source Prometheus endpoint |
| `metric_name` | string | Metric name |
| `metric_type` | string | `gauge`, `counter`, or `histogram` |
| `timestamp_ns` | int64 | Collection timestamp (nanoseconds) |
| `value` | float64 | Gauge/counter value (delta for counters) |
| `sum`, `count` | float64 | Histogram sum/count deltas |
| `bucket_le`, `bucket_count` | string, float64 | Histogram bucket bound and delta count |
| *(label columns)* | string | Dynamic columns from Prometheus labels |

See [Parquet Schema Reference](server_metrics_parquet_schema.md) for complete schema, metadata, and query examples.

**Related documentation:**
- [JSON Schema Reference](server_metrics_json_schema.md) - Complete JSON export format specification
- [Server Metrics Reference](server_metrics_reference.md) - Metric definitions by backend (vLLM, SGLang, TRT-LLM, Dynamo)
- [Parquet Schema Reference](server_metrics_parquet_schema.md) - Raw time-series data schema

**Quick examples:**

```bash
# DuckDB queries
duckdb -c "SELECT * FROM 'server_metrics_export.parquet' WHERE metric_name LIKE 'vllm:%' ORDER BY timestamp_ns"
duckdb -c "SELECT metric_name, AVG(value) FROM '*.parquet' WHERE metric_type='gauge' GROUP BY metric_name"

# Combine multiple runs (handles schema differences)
duckdb -c "SELECT * FROM read_parquet('artifacts/*/server_metrics_export.parquet', union_by_name=true)"
```

```python
import pandas as pd
df = pd.read_parquet('server_metrics_export.parquet')
df[df['metric_name'] == 'vllm:kv_cache_usage_perc'].plot(x='timestamp_ns', y='value')
```

---

## Statistics by Metric Type

Now that you understand the output formats, let's examine how statistics are structured within each metric type.

Statistics are nested under a `stats` field within each series item. All metrics use the `stats` format for consistent API access.

### Gauge (point-in-time values)

Statistics: `avg`, `min`, `max`, `std`, `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99`

Gauge percentiles are computed from **actual collected samples** (not estimated from buckets).

### Counter (cumulative totals)

Statistics: `total`, `rate`, and when `--slice-duration` is set: `rate_avg`, `rate_min`, `rate_max`, `rate_std`

- `total`: Change during profiling period (uses last pre-profiling sample as reference)
- `rate`: Increase per second (total/duration)
- Counter resets are detected and handled (negative deltas → total = 0)

### Histogram (distributions)

Statistics (`stats`): `count`, `count_rate`, `sum`, `sum_rate`, `avg`, `p1_estimate`, `p5_estimate`, `p10_estimate`, `p25_estimate`, `p50_estimate`, `p75_estimate`, `p90_estimate`, `p95_estimate`, `p99_estimate`

Series-level field: `buckets` (per-bucket delta counts, not cumulative)

- `avg` (sum/count) is **exact**
- Percentiles are **estimates** from bucket interpolation

> [!NOTE]
> **Prometheus Summary metrics are not supported.** Summary quantiles are computed cumulatively over the entire server lifetime, making them unsuitable for benchmark-specific analysis. Major LLM inference servers (vLLM, SGLang, TRT-LLM, Dynamo) use Histograms instead, which allow period-specific percentile estimation.

## Timesliced Statistics

When configured with `--slice-duration`, AIPerf computes windowed statistics over fixed time intervals. Each series includes a `timeslices` array with per-window statistics:

```json
{
  "stats": { "avg": 25.5, "min": 0.0, "max": 50.0 },
  "timeslices": [
    { "start_ns": 1765615837721140145, "end_ns": 1765615839721140145, "avg": 22.9, "min": 0.0, "max": 42.0 },
    { "start_ns": 1765615839721140145, "end_ns": 1765615841721140145, "avg": 49.8, "min": 49.0, "max": 50.0 }
  ]
}
```

- **Gauges**: Each timeslice contains `avg`, `min`, `max`
- **Counters**: Each timeslice contains `total`, `rate`
- **Histograms**: Each timeslice contains `count`, `sum`, `avg`, `buckets`

Partial timeslices (at the end of the collection period) are marked with `is_complete: false` and excluded from aggregate statistics (e.g., `rate_avg`, `rate_min`) to ensure fair comparison. Individual timeslice data includes both complete and partial slices for data completeness.

---

## Labeled Metrics

Prometheus metrics with labels (e.g., `model`, `status`) are aggregated separately for each unique label combination. When collecting from multiple endpoints, series are merged together with each tagged by `endpoint_url`.

## Unit Inference

AIPerf automatically infers units from metric names and descriptions using standard Prometheus conventions (`_seconds`, `_bytes`, `_requests`, etc.). Units appear in both JSON and CSV exports. The `unit` field is optional—if no unit can be inferred, it's omitted.

## Common Metrics by Server

### vLLM

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:num_requests_running` | gauge | Requests in execution batches |
| `vllm:num_requests_waiting` | gauge | Requests in queue (saturation indicator) |
| `vllm:kv_cache_usage_perc` | gauge | KV-cache usage (0.0-1.0, >0.9 = capacity limit) |
| `vllm:num_preemptions` | counter | Requests preempted due to memory pressure |
| `vllm:prefix_cache_hits` | counter | Tokens served from prefix cache |
| `vllm:prefix_cache_queries` | counter | Tokens queried (hit_rate = hits/queries) |
| `vllm:time_to_first_token_seconds` | histogram | Time to first token (TTFT) |
| `vllm:e2e_request_latency_seconds` | histogram | End-to-end latency |
| `vllm:inter_token_latency_seconds` | histogram | Time between output tokens (ITL) |
| `vllm:request_queue_time_seconds` | histogram | Time spent waiting in queue |
| `vllm:request_prefill_time_seconds` | histogram | Time spent in prefill phase |
| `vllm:request_decode_time_seconds` | histogram | Time spent in decode phase |
| `vllm:request_success` | counter | Completed requests |
| `vllm:prompt_tokens` | counter | Total prompt tokens (rate = prefill throughput) |
| `vllm:generation_tokens` | counter | Total generated tokens (rate = decode throughput) |

### Dynamo

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_frontend_requests` | counter | Requests by endpoint/model/status |
| `dynamo_frontend_inflight_requests` | gauge | Requests currently processing |
| `dynamo_frontend_queued_requests` | gauge | Requests awaiting first token |
| `dynamo_frontend_request_duration_seconds` | histogram | End-to-end HTTP latency |
| `dynamo_frontend_time_to_first_token_seconds` | histogram | TTFT including routing overhead |
| `dynamo_frontend_inter_token_latency_seconds` | histogram | Inter-token latency (ITL) |
| `dynamo_frontend_input_sequence_tokens` | histogram | Prompt token distribution |
| `dynamo_frontend_output_sequence_tokens` | histogram | Response token distribution |
| `dynamo_component_requests` | counter | Per-component (prefill/decode) requests |
| `dynamo_component_request_duration_seconds` | histogram | Per-component processing time |
| `dynamo_component_inflight_requests` | gauge | Active requests per worker |
| `dynamo_component_errors` | counter | Errors by component/type |
| `dynamo_component_kvstats_gpu_cache_usage_percent` | gauge | Backend KV-cache usage |

### SGLang

| Metric | Type | Description |
|--------|------|-------------|
| `sglang:num_running_reqs` | gauge | Running requests |
| `sglang:num_queue_reqs` | gauge | Queued requests (saturation indicator) |
| `sglang:token_usage` | gauge | Memory utilization (>0.9 = capacity limit) |
| `sglang:cache_hit_rate` | gauge | Prefix cache hit rate |
| `sglang:gen_throughput` | gauge | Real-time generation tokens/s |
| `sglang:queue_time_seconds` | histogram | Queue wait time |
| `sglang:per_stage_req_latency_seconds` | histogram | Latency by stage (prefill_*/decode_*) |

### TRT-LLM

| Metric | Type | Description |
|--------|------|-------------|
| `trtllm:time_to_first_token_seconds` | histogram | Time to first token (TTFT) |
| `trtllm:e2e_request_latency_seconds` | histogram | End-to-end latency |
| `trtllm:time_per_output_token_seconds` | histogram | Per-token generation time (ITL) |
| `trtllm:request_queue_time_seconds` | histogram | Time in WAITING phase |
| `trtllm:request_success` | counter | Completed requests |

---

## Troubleshooting

| Problem | Check | Solution |
|---------|-------|----------|
| High p99, good p50 | `vllm:num_requests_waiting` spikes | Queue buildup—reduce concurrency or increase server capacity |
| OOM crashes | `vllm:kv_cache_usage_perc` approaching 1.0 | Reduce `max_model_len` or increase `gpu_memory_utilization` |
| Low throughput | `vllm:num_requests_running` vs `vllm:num_requests_waiting` | Low both = client bottleneck; high waiting = server bottleneck |
| Endpoint unreachable | `curl http://localhost:8000/metrics` | Check server running, network, firewall; use explicit `--server-metrics` URL |

---

## CI/CD Integration

```python
import json

with open('server_metrics_export.json') as f:
    data = json.load(f)

latency = data['metrics']['vllm:e2e_request_latency_seconds']['series'][0]['stats']
assert latency['p99_estimate'] < 5.0, f"P99 latency too high: {latency['p99_estimate']}"
```


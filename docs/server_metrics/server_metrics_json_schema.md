<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Server Metrics JSON Export Schema

This document describes the structure and semantics of every field in the AIPerf server metrics JSON export format.

## Overview

The server metrics JSON export provides aggregated statistics from Prometheus metrics collected during a benchmark run.

### Data Organization

**Metrics are grouped by name across all endpoints.** When scraping multiple servers (e.g., prefill worker at `:10000` and decode worker at `:10001`), metrics with the same name appear under a single key.

**Each unique endpoint + label combination keeps its own separate series.** Within each metric, the `series` array contains one entry for every distinct combination of endpoint URL and Prometheus labels, with independent statistics.

For example, if `vllm:num_requests_running` is scraped from 3 endpoints with 2 label sets each, you get 6 per-endpoint series.

### Example Command

```bash
aiperf profile \
  -m Qwen/Qwen3-0.6B \
  --url localhost:10000 \
  --server-metrics localhost:10001 localhost:10002 \
  --request-count 50 \
  --concurrency 50
```

Note: The `--url` endpoint (`localhost:10000`) is automatically scraped for server metrics.

**Format selection:** By default, AIPerf generates JSON and CSV exports. This document describes the JSON format. To control which formats are generated, use `--server-metrics-formats`:
- Default: `--server-metrics-formats json csv` (JSONL and Parquet excluded to avoid large files)
- Include JSONL: `--server-metrics-formats json csv jsonl`
- Include Parquet: `--server-metrics-formats json csv parquet`
- JSON only: `--server-metrics-formats json`

The Parquet format exports raw time-series data with delta calculations in columnar format, optimized for SQL analytics with DuckDB, pandas, or Polars. See [Parquet Schema Reference](server_metrics_parquet_schema.md) for the complete schema.

**Related documentation:**
- [Server Metrics Tutorial](server-metrics.md) - Quick start guide and usage examples
- [Server Metrics Reference](server_metrics_reference.md) - Metric definitions by backend (vLLM, SGLang, TRT-LLM, Dynamo)
- [Parquet Schema Reference](server_metrics_parquet_schema.md) - Raw time-series data schema

### Data Access

Metrics are organized for O(1) lookup by name with nested stats within each series:

```python
data["metrics"]["metric_name"]["series"][0]["stats"]["p99"]
```

---

## Top-Level Structure

```json
{
  "schema_version": "1.0",
  "aiperf_version": "0.8.0",
  "benchmark_id": "550e8400-e29b-41d4-a716-446655440000",
  "summary": { ... },
  "metrics": { ... },
  "input_config": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version for this export format (e.g., `"1.0"`) |
| `aiperf_version` | string or null | AIPerf version that generated this export (e.g., `"0.8.0"`). `null` if version unavailable. |
| `benchmark_id` | string or null | Unique UUID identifying this benchmark run. `null` if not available. |
| [`summary`](#summary-section) | object | Collection metadata and endpoint information |
| [`metrics`](#metrics-section) | object | Metrics keyed by name, each containing type info and series data |
| `input_config` | object | Serialized user configuration used for this benchmark run |

---

## Summary Section

```json
"summary": {
  "endpoints_configured": [
    "http://localhost:10000/metrics",
    "http://localhost:10001/metrics"
  ],
  "endpoints_successful": [
    "http://localhost:10000/metrics",
    "http://localhost:10001/metrics"
  ],
  "start_time": "2025-12-10T16:07:13.596361",
  "end_time": "2025-12-10T16:07:35.749758",
  "endpoint_info": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `endpoints_configured` | array[string] | Full endpoint URLs that were configured for scraping |
| `endpoints_successful` | array[string] | Full endpoint URLs that returned data |
| `start_time` | datetime | When metrics collection started (ISO 8601) |
| `end_time` | datetime | When metrics collection ended (ISO 8601) |
| [`endpoint_info`](#endpoint-info) | object | Per-endpoint collection metadata |

### Endpoint Info

```json
"endpoint_info": {
  "http://localhost:10000/metrics": {
    "total_fetches": 144,
    "first_fetch_ns": 1765529006843416914,
    "last_fetch_ns": 1765529029508409301,
    "avg_fetch_latency_ms": 296.8633202916667,
    "unique_updates": 72,
    "first_update_ns": 1765529006843416914,
    "last_update_ns": 1765529029508409301,
    "duration_seconds": 22.664992387,
    "avg_update_interval_ms": 319.225244887324,
    "median_update_interval_ms": 334.0127105
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_fetches` | int | Total number of HTTP fetches from this endpoint |
| `first_fetch_ns` | int | Timestamp of first fetch in nanoseconds |
| `last_fetch_ns` | int | Timestamp of last fetch in nanoseconds |
| `avg_fetch_latency_ms` | float | Average time to fetch metrics from this endpoint in milliseconds |
| `unique_updates` | int | Number of fetches that returned changed metrics |
| `first_update_ns` | int | Timestamp of first unique update in nanoseconds |
| `last_update_ns` | int | Timestamp of last unique update in nanoseconds |
| `duration_seconds` | float | Time span from first to last unique update in seconds |
| `avg_update_interval_ms` | float | Average time between unique metric updates in milliseconds |
| `median_update_interval_ms` | float or null | Median time between unique metric updates in milliseconds. More robust to outliers than average. `null` if fewer than 2 intervals. |

---

## Metrics Section

Each metric entry has this structure:

```json
"metrics": {
  "metric_name": {
    "type": "gauge|counter|histogram",
    "description": "Metric description from HELP text",
    "unit": "seconds|tokens|requests|...",
    "series": [ ... ]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Prometheus metric type: [`gauge`](#gauge-metrics), [`counter`](#counter-metrics), or [`histogram`](#histogram-metrics) |
| `description` | string | Human-readable description from Prometheus HELP text |
| `unit` | string or null | Unit inferred from metric name suffix. See [Unit Inference](#unit-inference) for complete mapping of suffixes to unit values. |
| `series` | array | Statistics for each unique endpoint + label combination |

---

## Series Fields (Common)

Every series entry contains these common fields:

```json
{
  "endpoint_url": "http://localhost:10000/metrics",
  "labels": {"model": "Qwen/Qwen3-0.6B", "dynamo_component": "prefill"}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `endpoint_url` | string | Full endpoint URL (e.g., `http://localhost:10000/metrics`) |
| `labels` | object or null | Prometheus labels for this time series. `null` or missing if metric has no labels. |

---

## Gauge Metrics

Gauges represent point-in-time values that can go up or down (e.g., current queue depth, memory usage).

### Gauge Series Fields

| Field | Type | Description |
|-------|------|-------------|
| `endpoint_url` | string | Full endpoint URL |
| `labels` | object/null | Prometheus labels for this series |
| [`stats`](#gauge-stats-fields) | object | Nested statistics object (always present) |
| [`timeslices`](#gauge-timeslices) | array | Optional: Statistics broken down by time window |

### Gauge Stats Fields

| Field | Type | Description |
|-------|------|-------------|
| `avg` | float | Mean of all observed values during collection |
| `min` | float | Minimum observed value |
| `max` | float | Maximum observed value |
| `std` | float | Standard deviation of observed values |
| `p1` | float | 1st percentile |
| `p5` | float | 5th percentile |
| `p10` | float | 10th percentile |
| `p25` | float | 25th percentile |
| `p50` | float | 50th percentile (median) |
| `p75` | float | 75th percentile |
| `p90` | float | 90th percentile |
| `p95` | float | 95th percentile |
| `p99` | float | 99th percentile |

### Gauge with Variation

```json
{
  "endpoint_url": "http://localhost:10002/metrics",
  "labels": {
    "dynamo_component": "backend",
    "dynamo_endpoint": "generate",
    "model": "Qwen/Qwen3-0.6B"
  },
  "stats": {
    "avg": 36.68055555555556,
    "min": 0.0,
    "max": 50.0,
    "std": 16.87887786545273,
    "p1": 0.0,
    "p5": 2.0,
    "p10": 8.0,
    "p25": 25.0,
    "p50": 45.5,
    "p75": 47.0,
    "p90": 48.0,
    "p95": 49.0,
    "p99": 50.0
  },
  "timeslices": [
    {
      "start_ns": 1765411635639590410,
      "end_ns": 1765411637639590410,
      "avg": 5.0,
      "min": 0.0,
      "max": 15.0
    },
    {
      "start_ns": 1765411637639590410,
      "end_ns": 1765411639639590410,
      "avg": 31.67,
      "min": 24.0,
      "max": 35.0
    }
  ]
}
```

**Example interpretation** (`dynamo_component_inflight_requests`):
- "On average, 36.7 requests were in-flight"
- "In-flight requests ranged from 0 to 50"
- "99% of the time, in-flight requests were at or below 50"

### Gauge with No Variation (constant)

When a gauge never changes during collection (standard deviation = 0), stats are still provided for API consistency. All percentiles equal the constant value:

```json
{
  "endpoint_url": "http://localhost:11001/metrics",
  "labels": {
    "dynamo_component": "prefill",
    "dynamo_namespace": "acasagrande_sglang_acasagrande_sglang_disagg"
  },
  "stats": {
    "avg": 1024.0,
    "min": 1024.0,
    "max": 1024.0,
    "std": 0.0,
    "p1": 1024.0,
    "p5": 1024.0,
    "p10": 1024.0,
    "p25": 1024.0,
    "p50": 1024.0,
    "p75": 1024.0,
    "p90": 1024.0,
    "p95": 1024.0,
    "p99": 1024.0
  }
}
```

### Gauge Timeslices

Each gauge timeslice contains statistics for a fixed time window:

| Field | Type | Description |
|-------|------|-------------|
| `start_ns` | int | Timeslice start timestamp in nanoseconds |
| `end_ns` | int | Timeslice end timestamp in nanoseconds |
| `is_complete` | bool | Only present when `false` (partial timeslice, typically the final slice). Omitted for complete timeslices. |
| `avg` | float | Average value during this timeslice |
| `min` | float | Minimum value during this timeslice |
| `max` | float | Maximum value during this timeslice |

```json
{
  "start_ns": 1765411635639590410,
  "end_ns": 1765411637639590410,
  "avg": 5.0,
  "min": 0.0,
  "max": 15.0
}
```

---

## Counter Metrics

Counters are monotonically increasing values (e.g., total requests processed, total bytes transferred).

### Counter Series Fields

| Field | Type | Description |
|-------|------|-------------|
| `endpoint_url` | string | Full endpoint URL |
| `labels` | object/null | Prometheus labels for this series |
| [`stats`](#counter-stats-fields) | object | Nested statistics object (always present) |
| [`timeslices`](#counter-timeslices) | array | Optional: Statistics broken down by time window |

### Counter Stats Fields

| Field | Type | Description |
|-------|------|-------------|
| `total` | float | **Total increase** in counter value during collection period |
| `rate` | float | **Overall rate**: `total / duration_seconds` |
| `rate_avg` | float | Time-weighted average rate between change points |
| `rate_min` | float | Minimum instantaneous rate observed between consecutive scrapes |
| `rate_max` | float | Maximum instantaneous rate observed between consecutive scrapes |
| `rate_std` | float | Standard deviation of point-to-point rates |

### Counter with Activity

```json
{
  "endpoint_url": "http://localhost:10001/metrics",
  "labels": {
    "dynamo_component": "prefill",
    "dynamo_endpoint": "generate",
    "model": "Qwen/Qwen3-0.6B"
  },
  "stats": {
    "total": 318092.0,
    "rate": 14206.446174934012,
    "rate_avg": 14458.727272727272,
    "rate_min": 0.0,
    "rate_max": 69626.0,
    "rate_std": 25812.771107887304
  },
  "timeslices": [
    {
      "start_ns": 1765411635103733481,
      "end_ns": 1765411637103733481,
      "total": 104707.0,
      "rate": 52353.5
    },
    {
      "start_ns": 1765411637103733481,
      "end_ns": 1765411639103733481,
      "total": 74133.0,
      "rate": 37066.5
    }
  ]
}
```

**Example interpretation** (`dynamo_component_request_bytes`):
- `stats.total: 318092` → "318,092 bytes were received during the benchmark"
- `stats.rate: 14206.4` → "Overall throughput was 14,206 bytes/second"
- `stats.rate_avg: 14458.7` → "Average instantaneous rate was 14,459 bytes/second"
- `stats.rate_min: 0.0` → "Slowest period saw 0 bytes/second (idle)"
- `stats.rate_max: 69626.0` → "Fastest burst reached 69,626 bytes/second"

### Counter with No Activity

When a counter doesn't change during the collection period (total = 0), stats are still provided for API consistency:

```json
{
  "endpoint_url": "http://localhost:10001/metrics",
  "labels": {
    "dynamo_component": "prefill",
    "dynamo_endpoint": "clear_kv_blocks",
    "model": "Qwen/Qwen3-0.6B"
  },
  "stats": {
    "total": 0.0,
    "rate": 0.0
  }
}
```

### Counter Timeslices

Each counter timeslice contains the delta and rate for a fixed time window:

| Field | Type | Description |
|-------|------|-------------|
| `start_ns` | int | Timeslice start timestamp in nanoseconds |
| `end_ns` | int | Timeslice end timestamp in nanoseconds |
| `is_complete` | bool | Only present when `false` (partial timeslice, typically the final slice). Omitted for complete timeslices. |
| `total` | float | Total increase in counter value during this timeslice |
| `rate` | float | Rate of counter value increase per second during this timeslice |

```json
{
  "start_ns": 1765411635103733481,
  "end_ns": 1765411637103733481,
  "total": 104707.0,
  "rate": 52353.5
}
```

---

## Histogram Metrics

Histograms track distributions of values (e.g., request latencies, token counts). Prometheus histograms maintain cumulative bucket counts and a running sum.

### Histogram Series Fields

| Field | Type | Description |
|-------|------|-------------|
| `endpoint_url` | string | Full endpoint URL |
| `labels` | object/null | Prometheus labels for this series |
| [`stats`](#histogram-stats-fields) | object | Nested statistics object (always present for histograms) |
| [`buckets`](#bucket-data) | object/null | Map of bucket upper bounds to delta counts. Present when count > 0, may be `null` if counter reset detected. |
| [`timeslices`](#histogram-timeslices) | array | Optional: Statistics broken down by time window |

### Histogram Stats Fields

| Field | Type | Description |
|-------|------|-------------|
| `count` | int | Total count change over collection period (number of observations) |
| `sum` | float | Total sum change over collection period |
| `avg` | float | Overall average value: `sum / count` |
| `count_rate` | float | Average count change per second (observations per second) |
| `sum_rate` | float | Average sum change per second |
| `p1_estimate` | float | Estimated 1st percentile |
| `p5_estimate` | float | Estimated 5th percentile |
| `p10_estimate` | float | Estimated 10th percentile |
| `p25_estimate` | float | Estimated 25th percentile |
| `p50_estimate` | float | Estimated 50th percentile (median) |
| `p75_estimate` | float | Estimated 75th percentile |
| `p90_estimate` | float | Estimated 90th percentile |
| `p95_estimate` | float | Estimated 95th percentile |
| `p99_estimate` | float | Estimated 99th percentile |

Note: Percentiles are *estimates* interpolated from histogram buckets.

### Histogram with Observations

```json
{
  "endpoint_url": "http://localhost:10001/metrics",
  "labels": {
    "dynamo_component": "prefill",
    "dynamo_endpoint": "generate",
    "model": "Qwen/Qwen3-0.6B"
  },
  "stats": {
    "count": 50,
    "sum": 2.2072624189999814,
    "avg": 0.04414524837999963,
    "count_rate": 2.233071906073402,
    "sum_rate": 0.09857951394400953,
    "p1_estimate": 0.025,
    "p5_estimate": 0.028,
    "p10_estimate": 0.030,
    "p25_estimate": 0.033,
    "p50_estimate": 0.038245593313299506,
    "p75_estimate": 0.052658494249919106,
    "p90_estimate": 0.07715849424991911,
    "p95_estimate": 0.08532516091658578,
    "p99_estimate": 0.0918584942499191
  },
  "buckets": {
    "0.005": 0,
    "0.01": 0,
    "0.025": 0,
    "0.05": 35,
    "0.1": 50,
    "0.25": 50,
    "0.5": 50,
    "1": 50,
    "2.5": 50,
    "5": 50,
    "10": 50,
    "+Inf": 50
  },
  "timeslices": [
    {
      "start_ns": 1765411635103733481,
      "end_ns": 1765411637103733481,
      "count": 15,
      "sum": 0.5630153879999966,
      "avg": 0.03753435919999978,
      "buckets": {
        "0.005": 0,
        "0.025": 0,
        "0.05": 10,
        "0.1": 15,
        "0.25": 15,
        "0.5": 15,
        "1": 15,
        "2.5": 15,
        "5": 15,
        "10": 15,
        "+Inf": 15
      }
    },
    {
      "start_ns": 1765411637103733481,
      "end_ns": 1765411639103733481,
      "count": 12,
      "sum": 0.631630536000003,
      "avg": 0.05263587800000025,
      "buckets": {
        "0.005": 0,
        "0.025": 0,
        "0.05": 8,
        "0.1": 12,
        "0.25": 12,
        "0.5": 12,
        "1": 12,
        "2.5": 12,
        "5": 12,
        "10": 12,
        "+Inf": 12
      }
    }
  ]
}
```

### Histogram with No Observations

When a histogram has no observations, `stats` contains only `count: 0`, and `buckets` contains all zeros:

```json
{
  "endpoint_url": "http://localhost:10001/metrics",
  "labels": {
    "dynamo_component": "prefill",
    "dynamo_endpoint": "clear_kv_blocks",
    "model": "Qwen/Qwen3-0.6B"
  },
  "stats": {
    "count": 0
  },
  "buckets": {
    "0.005": 0,
    "0.01": 0,
    "0.025": 0,
    "0.05": 0,
    "0.1": 0,
    "0.25": 0,
    "0.5": 0,
    "1": 0,
    "2.5": 0,
    "5": 0,
    "10": 0,
    "+Inf": 0
  }
}
```

### Bucket Data

Bucket keys are the upper bound (as strings), values are **delta counts** (number of new observations in each bucket during the collection period). The `+Inf` bucket contains the total delta count.

```json
"buckets": {
  "0.005": 0,
  "0.05": 35,
  "0.1": 50,
  "+Inf": 50
}
```

### Histogram Timeslices

Each histogram timeslice contains count, sum, average, and bucket deltas for a fixed time window:

| Field | Type | Description |
|-------|------|-------------|
| `start_ns` | int | Timeslice start timestamp in nanoseconds |
| `end_ns` | int | Timeslice end timestamp in nanoseconds |
| `is_complete` | bool | Only present when `false` (partial timeslice, typically the final slice). Omitted for complete timeslices. |
| `count` | int | Change in count during this timeslice |
| `sum` | float | Change in sum during this timeslice |
| `avg` | float | Average value during this timeslice: `sum / count` |
| `buckets` | object/null | Map of bucket upper bounds to delta counts during this timeslice |

```json
{
  "start_ns": 1765411635103733481,
  "end_ns": 1765411637103733481,
  "count": 15,
  "sum": 0.5630153879999966,
  "avg": 0.03753435919999978,
  "buckets": {
    "0.005": 0,
    "0.05": 10,
    "0.1": 15,
    "+Inf": 15
  }
}
```

### Histogram Field Semantics by Use Case

The meaning of histogram fields depends on what the histogram measures:

#### Request-Level Histograms (e.g., `vllm:e2e_request_latency_seconds`)

| Field | Semantic Meaning | Example |
|-------|------------------|---------|
| `stats.count` | Number of requests | 50 requests |
| `stats.count_rate` | Request throughput | 2.23 requests/second |
| `stats.avg` | Mean request duration | 0.044 seconds |
| `stats.sum` | Total time spent on requests | 2.21 seconds |
| `stats.sum_rate` | **Concurrency metric**: seconds of request time per second of real time | 0.099 (≈0.1 concurrent requests) |
| `stats.p99_estimate` | 99th percentile latency | 0.092 seconds |

#### Token-Level Histograms (e.g., `input_sequence_tokens`)

| Field | Semantic Meaning | Example |
|-------|------------------|---------|
| `stats.count` | Number of requests | 50 requests |
| `stats.count_rate` | Request throughput | 2.29 requests/second |
| `stats.avg` | Mean tokens per request | 986 tokens |
| `stats.sum` | Total tokens processed | 49,311 tokens |
| `stats.sum_rate` | **Token throughput** | 2,264 tokens/second |
| `stats.p99_estimate` | 99th percentile tokens | 2,193 tokens |

---

## Field Presence Rules

Fields are omitted when not applicable to reduce JSON size. All series now use consistent `stats` format.

| Condition | Fields Present |
|-----------|----------------|
| **Gauge (any)** | `endpoint_url`, `labels`, `stats` (with all percentiles), `timeslices` (optional) |
| **Gauge with no variation** (std=0) | Same as above, but all percentiles equal the constant value and std=0 |
| **Counter (any)** | `endpoint_url`, `labels`, `stats` (with total, rate, rate_* fields), `timeslices` (optional) |
| **Counter with no activity** (total=0) | Same as above, but total=0 and all rates=0  |
| **Histogram with no observations** (count=0) | `endpoint_url`, `labels`, `stats` (count=0 only), `buckets` (all zeros) |
| **Histogram with observations** (count>0) | `endpoint_url`, `labels`, `stats` (all fields), `buckets`, `timeslices` (optional) |
| Metric has no labels | `labels` is `null` or omitted |
| Unit cannot be inferred | `unit` is `null` or omitted |
| Timeslices not requested | `timeslices` omitted |

---

## Unit Inference

Units are inferred from metric name suffixes. Longer suffixes are matched first to handle compound suffixes correctly (e.g., `_tokens_total` matches before `_total`).

The "JSON Unit Value" column shows the actual string that appears in the `unit` field of the exported JSON (computed via `enum.name.lower().replace("_per_second", "/s")`).

| Metric Name Suffix | JSON Unit Value |
|-------------------|-----------------|
| **Time** | |
| `_seconds`, `_seconds_total` | `seconds` |
| `_milliseconds`, `_ms`, `_ms_total` | `milliseconds` |
| `_nanoseconds`, `_ns`, `_ns_total` | `nanoseconds` |
| **Size** | |
| `_bytes`, `_bytes_total` | `bytes` |
| `_kilobytes` | `kilobytes` |
| `_megabytes` | `megabytes` |
| `_gigabytes` | `gigabytes` |
| **Counts** | |
| `_total`, `_count` | `count` |
| `_tokens`, `_tokens_total` | `tokens` |
| `_requests`, `_requests_total`, `_reqs` | `requests` |
| `request_success` | `requests` *(special case: no underscore prefix)* |
| `_errors`, `_errors_total`, `_error_count`, `_error_count_total` | `errors` |
| `_blocks`, `_blocks_total`, `_block_count` | `blocks` |
| **Rates** | |
| `_gb_s` | `gb/s` |
| **Ratios** | |
| `_ratio` | `ratio` |
| `_percent`, `_perc` | `percent` |
| **Physical** | |
| `_celsius` | `celsius` |
| `_joules` | `joule` |
| `_watts` | `watt` |

**Note:** Additional units may be inferred from metric description text (e.g., "in milliseconds", "(GB/s)"). Description-based inference takes priority when both suffix and description are present.

---

## Data Normalization

All statistics in the export are computed over the **collection period**, which may exclude warmup time based on configuration. Understanding how each metric type is normalized is critical for correct interpretation.

### Counter Normalization

Counters are cumulative values in Prometheus—they only increase (except on server restart). The export normalizes them to **deltas** (changes) over the collection period:

| Export Field | Calculation | Example |
|--------------|-------------|---------|
| `total` | `final_value - reference_value` | If counter went from 1000 to 1500, `total = 500` |
| `rate` | `total / duration_seconds` | Overall rate: `500 / 22.0 = 22.7/second` |
| `rate_avg` | Mean of per-timeslice rates | Average instantaneous rate across all timeslices |
| `rate_min` | Minimum per-timeslice rate | Slowest period (may be 0 during idle) |
| `rate_max` | Maximum per-timeslice rate | Fastest burst |
| `rate_std` | Standard deviation of rates | Variability of rate over time |

**Counter reset handling:** If a counter decreases (server restart), the delta is clamped to 0 to avoid negative totals.

**Reference point:** The reference value for delta calculation is the last sample **before** the collection period starts (after warmup exclusion), ensuring accurate deltas at the period boundary.

### Gauge Normalization

Gauges are point-in-time values. Statistics are computed from all samples within the collection period:

| Export Field | Calculation | Notes |
|--------------|-------------|-------|
| `avg` | Arithmetic mean of all samples | Simple average, not time-weighted |
| `min`, `max` | Minimum/maximum observed | Extreme values seen |
| `std` | Sample standard deviation (ddof=1) | Unbiased estimate using Bessel's correction |
| `p1` - `p99` | **Exact percentiles** | Computed from raw sample data using NumPy |

**Constant gauge handling:** If standard deviation = 0 (gauge never varied), all percentiles will equal the constant value.

### Histogram Normalization

Histograms are cumulative in Prometheus—both bucket counts and sum only increase. The export normalizes to **deltas**:

| Export Field | Calculation | Notes |
|--------------|-------------|-------|
| `count` | `final_count - reference_count` | Number of observations during period |
| `sum` | `final_sum - reference_sum` | Sum of observed values during period |
| `avg` | `sum / count` | Average value per observation |
| `count_rate` | `count / duration_seconds` | Observations per second |
| `sum_rate` | `sum / duration_seconds` | Sum increase per second |
| `buckets` | Per-bucket deltas | Each bucket shows count increase during period |
| `p*_estimate` | **Estimated percentiles** | See [Histogram Percentile Estimation](#histogram-percentile-estimation) |

### Timeslice Normalization

When `--slice-duration` is configured (default: 2 seconds), the collection period is divided into fixed-duration windows. Each timeslice contains:

- **Gauges**: `avg`, `min`, `max` for that window
- **Counters**: `total` (delta) and `rate` for that window
- **Histograms**: `count`, `sum`, `avg`, and optional `buckets` for that window

**Fallback behavior:** If the configured slice duration is smaller than the actual metric update interval, the system falls back to per-interval mode where each sample interval becomes its own "timeslice".

### Warmup Exclusion

When warmup time is configured, metrics collected during warmup are **excluded** from all statistics. The `reference_value` for delta calculations is taken from the last sample before the warmup period ends.

---

## Histogram Percentile Estimation

Histogram percentiles are **estimates** because Prometheus histograms only store cumulative bucket counts, not individual observations. AIPerf uses a **polynomial histogram algorithm** for significantly improved accuracy over standard linear interpolation.

### Why Standard Interpolation Fails

Standard Prometheus histogram interpolation assumes observations are **uniformly distributed** within each bucket. This assumption fails badly when:

1. **Observations cluster near boundaries**: Real latency distributions often cluster near 0 or near bucket edges
2. **+Inf bucket contains data**: The unbounded bucket makes interpolation impossible
3. **Bucket widths are large**: Wide buckets hide the true distribution shape

Standard interpolation can produce errors of **5-10x** on P99 estimates for typical LLM inference workloads.

### Polynomial Histogram Algorithm

AIPerf implements a four-phase algorithm that provides ~5x reduction in percentile estimation error:

**Phase 1 - Per-bucket mean learning:**
When a scrape interval has all observations in a single bucket, the exact mean for that bucket can be computed: `mean = sum_delta / count_delta`. These learned means are accumulated over time via `accumulate_bucket_statistics()`.

**Phase 2 - Estimate bucket sums:**
For each finite bucket, estimate the sum using learned means (or midpoint fallback). This gives `estimated_finite_sum`.

**Phase 3 - +Inf bucket back-calculation:**
The +Inf bucket sum is calculated as `total_sum - estimated_finite_sum`. Observations are spread around the estimated mean `inf_avg = inf_sum / inf_count` within the +Inf range.

**Phase 4 - Generate finite observations with sum constraint:**
For each bucket, observations are placed using one of several strategies based on learned statistics:
- **F3 two-point mass**: When variance is extremely tight (< 1% of bucket width)
- **Blended distribution**: When variance is tight (< 20%) and mean is near center (< 30% offset)
- **Variance-aware distribution**: When variance is moderate
- **Shifted uniform**: Fallback when only mean is learned (no variance data)
- **Pure uniform**: Final fallback using bucket midpoint

After initial placement, positions are adjusted proportionally across all buckets to match the adjusted target sum (`total_sum - inf_sum_estimate`), with each bucket's adjustment capped at ±40% of bucket width.

### Percentile Field Naming

Histogram percentiles use the `_estimate` suffix to indicate they are approximations:

| Field | Description |
|-------|-------------|
| `p1_estimate` - `p99_estimate` | Estimated percentiles using polynomial algorithm |

Gauge percentiles (computed from raw samples) do not have the `_estimate` suffix because they are exact.

---

## Example Queries

### Find all metrics with p99 > 1 second
```python
for name, metric in data["metrics"].items():
    for series in metric["series"]:
        stats = series.get("stats", {})
        # Gauge percentiles use "p99", histogram uses "p99_estimate"
        p99 = stats.get("p99") or stats.get("p99_estimate")
        if p99 and p99 > 1.0 and metric.get("unit") == "seconds":
            print(f"{name}: p99={p99:.2f}s")
```

### Calculate total bytes transferred across all endpoints
```python
total = sum(
    series.get("stats", {}).get("total", 0)
    for series in data["metrics"]["dynamo_component_request_bytes"]["series"]
)
```

### Find highest throughput endpoint
```python
max_throughput = max(
    (series.get("stats", {}).get("rate", 0), series["endpoint_url"])
    for series in data["metrics"]["dynamo_component_requests"]["series"]
)
```

### Access timeslice data
```python
metric = data["metrics"]["dynamo_component_inflight_requests"]
for series in metric["series"]:
    if series.get("timeslices"):
        for ts in series["timeslices"]:
            duration_ns = ts["end_ns"] - ts["start_ns"]
            duration_s = duration_ns / 1e9
            print(f"  {duration_s:.1f}s window: avg={ts['avg']:.2f}")
```

---

## Minimal Example

```json
{
  "schema_version": "1.0",
  "aiperf_version": "0.8.0",
  "benchmark_id": "550e8400-e29b-41d4-a716-446655440000",
  "summary": {
    "endpoints_configured": [
      "http://localhost:10000/metrics",
      "http://localhost:10001/metrics",
      "http://localhost:10002/metrics"
    ],
    "endpoints_successful": [
      "http://localhost:10000/metrics",
      "http://localhost:10001/metrics",
      "http://localhost:10002/metrics"
    ],
    "start_time": "2025-12-10T16:07:13.596361",
    "end_time": "2025-12-10T16:07:35.749758",
    "endpoint_info": {
      "http://localhost:10000/metrics": {
        "total_fetches": 144,
        "first_fetch_ns": 1765529006843416914,
        "last_fetch_ns": 1765529029508409301,
        "avg_fetch_latency_ms": 296.86,
        "unique_updates": 72,
        "first_update_ns": 1765529006843416914,
        "last_update_ns": 1765529029508409301,
        "duration_seconds": 22.66,
        "avg_update_interval_ms": 319.23,
        "median_update_interval_ms": 334.01
      },
      "http://localhost:10001/metrics": {
        "total_fetches": 140,
        "first_fetch_ns": 1765529007434057293,
        "last_fetch_ns": 1765529029554057293,
        "avg_fetch_latency_ms": 285.42,
        "unique_updates": 70,
        "first_update_ns": 1765529007434057293,
        "last_update_ns": 1765529029554057293,
        "duration_seconds": 22.12,
        "avg_update_interval_ms": 316.00,
        "median_update_interval_ms": 320.50
      },
      "http://localhost:10002/metrics": {
        "total_fetches": 142,
        "first_fetch_ns": 1765529006950000000,
        "last_fetch_ns": 1765529029400000000,
        "avg_fetch_latency_ms": 290.15,
        "unique_updates": 71,
        "first_update_ns": 1765529006950000000,
        "last_update_ns": 1765529029400000000,
        "duration_seconds": 22.45,
        "avg_update_interval_ms": 318.10,
        "median_update_interval_ms": 325.75
      }
    }
  },
  "metrics": {
    "vllm:num_requests_running": {
      "type": "gauge",
      "description": "Number of requests currently in the model execution batch",
      "unit": "requests",
      "series": [
        {
          "endpoint_url": "http://localhost:10000/metrics",
          "labels": {"model": "Qwen/Qwen3-0.6B"},
          "stats": {
            "avg": 36.68,
            "min": 0.0,
            "max": 50.0,
            "std": 16.88,
            "p1": 0.0,
            "p5": 2.0,
            "p10": 8.0,
            "p25": 25.0,
            "p50": 45.5,
            "p75": 47.0,
            "p90": 48.0,
            "p95": 49.0,
            "p99": 50.0
          }
        },
        {
          "endpoint_url": "http://localhost:10001/metrics",
          "labels": {"model": "Qwen/Qwen3-0.6B"},
          "stats": {
            "avg": 12.34,
            "min": 0.0,
            "max": 25.0,
            "std": 8.21,
            "p1": 0.0,
            "p5": 1.0,
            "p10": 3.0,
            "p25": 6.0,
            "p50": 14.0,
            "p75": 18.0,
            "p90": 22.0,
            "p95": 24.0,
            "p99": 25.0
          }
        },
        {
          "endpoint_url": "http://localhost:10002/metrics",
          "labels": {"model": "Qwen/Qwen3-0.6B"},
          "stats": {
            "avg": 8.92,
            "min": 0.0,
            "max": 18.0,
            "std": 5.67,
            "p1": 0.0,
            "p5": 1.0,
            "p10": 2.0,
            "p25": 4.0,
            "p50": 10.0,
            "p75": 13.0,
            "p90": 16.0,
            "p95": 17.0,
            "p99": 18.0
          }
        }
      ]
    },
    "vllm:request_success": {
      "type": "counter",
      "description": "Count of successfully completed requests",
      "unit": "requests",
      "series": [
        {
          "endpoint_url": "http://localhost:10000/metrics",
          "labels": {"model": "Qwen/Qwen3-0.6B"},
          "stats": {
            "total": 50.0,
            "rate": 2.23,
            "rate_avg": 2.27,
            "rate_min": 0.0,
            "rate_max": 11.5,
            "rate_std": 4.09
          }
        },
        {
          "endpoint_url": "http://localhost:10001/metrics",
          "labels": {"model": "Qwen/Qwen3-0.6B"},
          "stats": {
            "total": 50.0,
            "rate": 2.26,
            "rate_avg": 2.30,
            "rate_min": 0.0,
            "rate_max": 10.8,
            "rate_std": 3.95
          }
        },
        {
          "endpoint_url": "http://localhost:10002/metrics",
          "labels": {"model": "Qwen/Qwen3-0.6B"},
          "stats": {
            "total": 50.0,
            "rate": 2.23,
            "rate_avg": 2.25,
            "rate_min": 0.0,
            "rate_max": 11.2,
            "rate_std": 4.01
          }
        }
      ]
    },
    "vllm:e2e_request_latency_seconds": {
      "type": "histogram",
      "description": "End-to-end request latency from arrival to completion",
      "unit": "seconds",
      "series": [
        {
          "endpoint_url": "http://localhost:10000/metrics",
          "labels": {"model": "Qwen/Qwen3-0.6B"},
          "stats": {
            "count": 50,
            "sum": 2.21,
            "avg": 0.044,
            "count_rate": 2.23,
            "sum_rate": 0.099,
            "p1_estimate": 0.025,
            "p5_estimate": 0.028,
            "p10_estimate": 0.030,
            "p25_estimate": 0.033,
            "p50_estimate": 0.038,
            "p75_estimate": 0.052,
            "p90_estimate": 0.077,
            "p95_estimate": 0.085,
            "p99_estimate": 0.092
          },
          "buckets": {"0.005": 0, "0.05": 35, "0.1": 50, "+Inf": 50}
        },
        {
          "endpoint_url": "http://localhost:10001/metrics",
          "labels": {"model": "Qwen/Qwen3-0.6B"},
          "stats": {
            "count": 50,
            "sum": 1.85,
            "avg": 0.037,
            "count_rate": 2.26,
            "sum_rate": 0.084,
            "p1_estimate": 0.020,
            "p5_estimate": 0.023,
            "p10_estimate": 0.025,
            "p25_estimate": 0.028,
            "p50_estimate": 0.032,
            "p75_estimate": 0.045,
            "p90_estimate": 0.065,
            "p95_estimate": 0.072,
            "p99_estimate": 0.078
          },
          "buckets": {"0.005": 0, "0.05": 42, "0.1": 50, "+Inf": 50}
        },
        {
          "endpoint_url": "http://localhost:10002/metrics",
          "labels": {"model": "Qwen/Qwen3-0.6B"},
          "stats": {
            "count": 50,
            "sum": 2.05,
            "avg": 0.041,
            "count_rate": 2.23,
            "sum_rate": 0.091,
            "p1_estimate": 0.022,
            "p5_estimate": 0.025,
            "p10_estimate": 0.027,
            "p25_estimate": 0.030,
            "p50_estimate": 0.035,
            "p75_estimate": 0.048,
            "p90_estimate": 0.072,
            "p95_estimate": 0.080,
            "p99_estimate": 0.086
          },
          "buckets": {"0.005": 0, "0.05": 38, "0.1": 50, "+Inf": 50}
        }
      ]
    }
  },
  "input_config": {
    "model": "Qwen/Qwen3-0.6B",
    "url": "http://localhost:10000",
    "loadgen": {
      "concurrency": 50,
      "request_count": 50
    },
    "cli_command": "aiperf profile --model 'Qwen/Qwen3-0.6B' --url 'http://localhost:10000' --server-metrics 'http://localhost:10001/metrics' 'http://localhost:10002/metrics' --request-count 50 --concurrency 50",
    "benchmark_id": "550e8400-e29b-41d4-a716-446655440000",
    "server_metrics": [
      "http://localhost:10001/metrics",
      "http://localhost:10002/metrics"
    ]
  }
}
```

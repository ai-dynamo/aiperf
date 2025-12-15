<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Server Metrics Parquet Export Schema

Schema reference for the `server_metrics_export.parquet` file. Optimized for SQL analytics with DuckDB, pandas, and Polars.

## Overview

The Parquet export provides raw time-series data with **cumulative delta calculations** applied at each timestamp. Uses a **normalized schema** where histogram buckets are separate rows (not wide columns), producing ~50% smaller files.

### Enable Parquet Export

```bash
aiperf profile --model MODEL ... --server-metrics-formats json csv parquet
```

### Delta Calculations

All values are deltas from a reference point (last sample before profiling period):

| Metric Type | Value Semantics |
|-------------|-----------------|
| **Gauge** | Raw value at timestamp (no delta) |
| **Counter** | Cumulative delta from reference (`value[t] - value[ref]`) |
| **Histogram** | Cumulative deltas for `sum`, `count`, and each `bucket_count` |

Negative deltas (counter resets) are clamped to 0.

---

## Schema Definition

### Fixed Columns

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `endpoint_url` | `string` | No | Prometheus endpoint URL (e.g., `http://localhost:8000/metrics`) |
| `metric_name` | `string` | No | Metric name (e.g., `vllm:kv_cache_usage_perc`) |
| `metric_type` | `string` | No | `gauge`, `counter`, or `histogram` |
| `unit` | `string` | Yes | Inferred unit (`seconds`, `tokens`, `requests`, `ratio`, etc.) |
| `description` | `string` | Yes | Metric HELP text from Prometheus |
| `timestamp_ns` | `int64` | No | Collection timestamp in nanoseconds since epoch |

### Value Columns

| Column | Type | Nullable | Used By | Description |
|--------|------|----------|---------|-------------|
| `value` | `float64` | Yes | Gauge, Counter | Metric value (raw for gauge, delta for counter) |
| `sum` | `float64` | Yes | Histogram | Cumulative sum delta from reference |
| `count` | `float64` | Yes | Histogram | Cumulative count delta from reference |
| `bucket_le` | `string` | Yes | Histogram | Bucket upper bound (e.g., `0.1`, `+Inf`) |
| `bucket_count` | `float64` | Yes | Histogram | Cumulative bucket count delta (observations <= `bucket_le`) |

### Dynamic Label Columns

Prometheus labels become individual columns (alphabetically sorted):

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `engine` | `string` | Yes | vLLM engine ID |
| `engine_type` | `string` | Yes | Engine type (`trtllm`, `unified`) |
| `finished_reason` | `string` | Yes | Request completion reason |
| `model_name` | `string` | Yes | Model identifier |
| `dynamo_component` | `string` | Yes | Dynamo worker component |
| `tp_rank` | `string` | Yes | Tensor parallel rank |
| `pp_rank` | `string` | Yes | Pipeline parallel rank |
| `stage` | `string` | Yes | SGLang processing stage |
| *(others)* | `string` | Yes | Any additional Prometheus labels |

Label columns vary by endpoint/model. Use `union_by_name=true` for cross-file queries.

**Note:** Prometheus labels that conflict with reserved column names (`endpoint_url`, `metric_name`, `metric_type`, `unit`, `description`, `timestamp_ns`, `value`, `sum`, `count`, `bucket_le`, `bucket_count`) are silently excluded.

---

## Row Structure by Metric Type

Column order: fixed columns → label columns (alphabetically) → value columns.

### Gauge/Counter: One Row per Timestamp

```
endpoint_url | metric_name              | metric_type | unit  | description |timestamp_ns        | model_name   | value | sum  | count | bucket_le | bucket_count
-------------|--------------------------|-------------|-------|-------------|---------------------|--------------|-------|------|-------|-----------|-------------
http://...   | vllm:kv_cache_usage_perc | gauge       | ratio | KV-cache... | 1765793061967310848 | Qwen/Qwen3-0.6B | 0.72  | null | null  | null      | null
http://...   | vllm:request_success     | counter     | null  | Count of... | 1765793061967310848 | Qwen/Qwen3-0.6B | 150.0 | null | null  | null      | null
```

### Histogram: N Rows per Timestamp (One per Bucket)

```
endpoint_url | metric_name                      | metric_type | unit    | description  | timestamp_ns        | model_name      | value | sum    | count | bucket_le | bucket_count
-------------|----------------------------------|-------------|---------|--------------|---------------------|-----------------|-------|--------|-------|-----------|-------------
http://...   | vllm:e2e_request_latency_seconds | histogram   | seconds | Histogram... | 1765793061967310848 | Qwen/Qwen3-0.6B | null  | 259.87 | 19.0  | 0.3       | 0.0
http://...   | vllm:e2e_request_latency_seconds | histogram   | seconds | Histogram... | 1765793061967310848 | Qwen/Qwen3-0.6B | null  | 259.87 | 19.0  | 1.0       | 1.0
http://...   | vllm:e2e_request_latency_seconds | histogram   | seconds | Histogram... | 1765793061967310848 | Qwen/Qwen3-0.6B | null  | 259.87 | 19.0  | 5.0       | 3.0
http://...   | vllm:e2e_request_latency_seconds | histogram   | seconds | Histogram... | 1765793061967310848 | Qwen/Qwen3-0.6B | null  | 259.87 | 19.0  | +Inf      | 19.0
```

---

## File Metadata

Parquet file metadata (accessible via `pq.read_metadata()`) includes:

| Key | Description |
|-----|-------------|
| `aiperf.schema_version` | Schema version (`1.0`) |
| `aiperf.version` | AIPerf version |
| `aiperf.benchmark_id` | Unique benchmark UUID |
| `aiperf.exporter` | Exporter class name (`ServerMetricsParquetExporter`) |
| `aiperf.export_timestamp_utc` | Export timestamp (ISO 8601) |
| `aiperf.time_filter_start_ns` | Profiling period start (nanoseconds) |
| `aiperf.time_filter_end_ns` | Profiling period end (nanoseconds) |
| `aiperf.profiling_duration_ns` | Profiling duration (nanoseconds) |
| `aiperf.profiling_duration_seconds` | Profiling duration (seconds) |
| `aiperf.endpoint_urls` | JSON array of endpoint URLs |
| `aiperf.endpoint_count` | Number of endpoints |
| `aiperf.label_columns` | JSON array of label column names |
| `aiperf.label_count` | Number of label columns |
| `aiperf.metric_count` | Total unique metrics |
| `aiperf.metric_type_counts` | JSON object: `{"gauge": N, "counter": N, "histogram": N}` |
| `aiperf.model_names` | JSON array of model names |
| `aiperf.concurrency` | Benchmark concurrency setting |
| `aiperf.request_rate` | Benchmark request rate (if set) |
| `aiperf.input_config` | Full user configuration (JSON) |
| `aiperf.hostname` | Collection host |
| `aiperf.python_version` | Python version |
| `aiperf.pyarrow_version` | PyArrow version |
| `aiperf.schema_note` | Cross-file query hint |

**Compression:** Snappy (good compression ratio with fast decompression)

---

## Example Queries

### DuckDB

```sql
-- Time-series for a specific metric
SELECT timestamp_ns, value
FROM 'server_metrics_export.parquet'
WHERE metric_name = 'vllm:kv_cache_usage_perc'
ORDER BY timestamp_ns;

-- Filter by label
SELECT timestamp_ns, value
FROM 'server_metrics_export.parquet'
WHERE metric_name = 'vllm:request_success'
  AND model_name = 'Qwen/Qwen3-0.6B'
ORDER BY timestamp_ns;

-- Histogram bucket distribution at final timestamp
SELECT bucket_le, bucket_count
FROM 'server_metrics_export.parquet'
WHERE metric_name = 'vllm:e2e_request_latency_seconds'
  AND timestamp_ns = (SELECT MAX(timestamp_ns) FROM 'server_metrics_export.parquet'
                      WHERE metric_name = 'vllm:e2e_request_latency_seconds')
ORDER BY CAST(REPLACE(bucket_le, '+Inf', '999999') AS DOUBLE);

-- Aggregate across multiple runs (handles schema differences)
SELECT metric_name, AVG(value) as avg_value
FROM read_parquet('artifacts/*/server_metrics_export.parquet', union_by_name=true)
WHERE metric_type = 'gauge'
GROUP BY metric_name;

-- Compare endpoints
SELECT endpoint_url, metric_name, AVG(value) as avg_value
FROM 'server_metrics_export.parquet'
WHERE metric_type = 'gauge'
GROUP BY endpoint_url, metric_name;
```

### pandas

```python
import pandas as pd

df = pd.read_parquet('server_metrics_export.parquet')

# Filter to gauge metrics
gauges = df[df['metric_type'] == 'gauge']

# Time-series plot
kv_usage = df[df['metric_name'] == 'vllm:kv_cache_usage_perc']
kv_usage.plot(x='timestamp_ns', y='value', title='KV Cache Usage')

# Pivot histogram buckets
hist = df[df['metric_name'] == 'vllm:e2e_request_latency_seconds']
pivot = hist.pivot(index='timestamp_ns', columns='bucket_le', values='bucket_count')
```

### Polars

```python
import polars as pl

df = pl.read_parquet('server_metrics_export.parquet')

# Filter and aggregate
result = (
    df.filter(pl.col('metric_type') == 'gauge')
    .group_by('metric_name')
    .agg([
        pl.col('value').mean().alias('avg'),
        pl.col('value').max().alias('max'),
    ])
)

# Lazy scan for large files
lazy = pl.scan_parquet('artifacts/*/server_metrics_export.parquet')
result = lazy.filter(pl.col('metric_name') == 'vllm:kv_cache_usage_perc').collect()
```

### Reading Metadata

```python
import pyarrow.parquet as pq
import json

metadata = pq.read_metadata('server_metrics_export.parquet')
schema_metadata = metadata.schema.to_arrow_schema().metadata

# Access specific fields
benchmark_id = schema_metadata[b'aiperf.benchmark_id'].decode()
config = json.loads(schema_metadata[b'aiperf.input_config'])
label_columns = json.loads(schema_metadata[b'aiperf.label_columns'])
```

---

## Best Practices

### Cross-File Analysis

Label columns vary by endpoint and model. Always use `union_by_name`:

```sql
-- DuckDB
SELECT * FROM read_parquet('run_*/server_metrics_export.parquet', union_by_name=true);
```

```python
# pandas
import pandas as pd
from pathlib import Path

dfs = [pd.read_parquet(p) for p in Path('.').glob('run_*/server_metrics_export.parquet')]
combined = pd.concat(dfs, ignore_index=True)
```

### Histogram Percentile Estimation

Reconstruct percentiles from bucket data. Note that `bucket_count` values are **cumulative** (each bucket includes all observations with value <= `bucket_le`), matching Prometheus histogram semantics:

```python
import numpy as np

def estimate_percentile(bucket_les, bucket_counts, percentile):
    """Estimate percentile from histogram buckets using linear interpolation."""
    # Convert bucket_le strings to floats (handle +Inf)
    bounds = [float(b) if b != '+Inf' else np.inf for b in bucket_les]
    counts = np.array(bucket_counts)

    total = counts[-1]  # +Inf bucket has cumulative total
    target = total * (percentile / 100)

    for i, (le, count) in enumerate(zip(bounds, counts)):
        if count >= target:
            if i == 0:
                return le
            prev_le = bounds[i-1] if i > 0 else 0
            prev_count = counts[i-1] if i > 0 else 0
            # Linear interpolation within bucket
            fraction = (target - prev_count) / (count - prev_count) if count > prev_count else 0
            return prev_le + fraction * (le - prev_le)
    return bounds[-2]  # Return last finite bound
```

### Memory-Efficient Processing

For large files, use lazy evaluation:

```python
# Polars lazy scan
import polars as pl
df = pl.scan_parquet('server_metrics_export.parquet') \
    .filter(pl.col('metric_name') == 'vllm:kv_cache_usage_perc') \
    .collect()

# DuckDB direct query (doesn't load entire file)
import duckdb
result = duckdb.query("""
    SELECT AVG(value) FROM 'server_metrics_export.parquet'
    WHERE metric_name = 'vllm:kv_cache_usage_perc'
""").fetchone()
```

---

## Schema Version History

| Version | Changes |
|---------|---------|
| `1.0` | Initial schema with normalized histogram buckets |

---

*For aggregated statistics, see [JSON Schema](server_metrics_json_schema.md). For metric definitions, see [Server Metrics Reference](server_metrics_reference.md). For usage examples, see the [Server Metrics Tutorial](server-metrics.md).*

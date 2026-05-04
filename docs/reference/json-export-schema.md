---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: JSON Export Schema
---

# `profile_export_aiperf.json` Schema

After every `aiperf profile` run, AIPerf writes a summary JSON file (default name `profile_export_aiperf.json`) under the artifact directory. Each top-level metric entry holds a stats block; this page documents which fields appear in that block, when they appear, and how the schema is versioned.

The on-disk shape is produced by `JsonMetricResult` in [`src/aiperf/common/models/export_models.py`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/common/models/export_models.py). Fields that are unset are omitted from the JSON output (`exclude_none=True`), so the field set per metric varies by metric type — this page is the source of truth for which fields to expect where.

## Per-metric stats fields

| Field | Type | Always present? | Notes |
|---|---|---|---|
| `unit` | string | yes | Display unit, e.g. `"ms"`, `"requests/sec"`, `"tokens"`. |
| `avg` | float | record metrics with observations; derived/aggregate metrics | For derived/aggregate scalar metrics, `avg` carries the single computed value. |
| `min` | number | record metrics with a distribution | Smallest observation. |
| `max` | number | record metrics with a distribution | Largest observation. |
| `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99` | float | record metrics with a distribution | Percentiles. Omitted for derived/aggregate metrics that have no distribution. |
| `std` | float | record metrics with a distribution | Sample standard deviation. |
| `count` | int | **record metrics only** | Number of records contributing to the distribution. Intentionally omitted for derived/aggregate scalar metrics where it would trivially be 1 and risks being misread as the request count. |
| `sum` | number | record metrics with a distribution sum | Sum of all observations. Absent for derived metrics whose value is itself a computed rate or total. |

The metric type (`record` / `aggregate` / `derived`) is documented per-metric in [Metrics Reference](../metrics-reference.md). At a glance: latencies and per-request lengths are `record`; counts and timestamps are `aggregate`; throughputs and run-level totals are `derived`.

### Example

A run with 20 requests against a streaming chat endpoint produces entries shaped like this:

```json
{
  "schema_version": "1.1",
  "request_latency": {
    "unit": "ms",
    "avg": 2620.71,
    "min": 2145.06,
    "max": 3411.10,
    "p50": 2568.73,
    "p99": 3371.24,
    "std": 297.93,
    "count": 20,
    "sum": 52414.29
  },
  "request_throughput": {
    "unit": "requests/sec",
    "avg": 1.45
  },
  "request_count": {
    "unit": "requests",
    "avg": 20.0
  }
}
```

Note that `request_throughput` (derived) and `request_count` (aggregate) carry only `unit` + `avg` — no `count`, no `sum`, no percentiles. `request_latency` (record) carries the full set.

## Schema versions

The current schema version is exported as the top-level `schema_version` field on the JSON document. Bump on additive changes; coordinate a major bump for any field rename or removal.

| Version | Change |
|---|---|
| `1.0` | Initial shape: `unit`, `avg`, `min`, `max`, `std`, `p1`–`p99`. |
| `1.1` | Added `count` and `sum` to per-metric stats blocks. Backward-compatible for readers that ignore unknown fields; the new fields are present only on record-type metrics, omitted on derived/aggregate. |

### Other JSON exports use independent schema versions

`aiperf` writes additional JSON files when `--num-profile-runs >= 2`:

- `profile_export_aiperf_aggregate.json` — confidence aggregation across runs. Per-metric blocks have a different shape (`mean`, `std`, `cv`, `se`, `ci_low`, `ci_high`, `t_critical`, `unit`) and own their own `schema_version` (`AggregateConfidenceJsonExporter.SCHEMA_VERSION`, currently `"1.0"`).
- `profile_export_aiperf_collated.json` — pools per-request values from all runs into a single population, then emits combined percentiles (`mean`, `std`, `p50`, `p90`, `p95`, `p99`, `count`) under a `combined` key plus a `per_run` list of run-level summaries. Uses its own `schema_version` (`"1.0.0"`).

The `schema_version` documented on this page applies only to `profile_export_aiperf.json`. The other files evolve on their own cadence.

## For downstream parsers

- **Treat absent fields as "not applicable to this metric type," not "data missing."** A derived-metric block with no `count` is normal; a record-metric block with no `count` indicates a bug.
- **Do not assume the field set is closed.** Future minor schema bumps may add fields. Use `schema_version` to detect compat; ignore unknown fields.
- **`unit` is authoritative for the value's interpretation.** Do not infer units from the metric tag.

---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: List-Metric Aggregation
---

# List-Metric Aggregation

Some record metrics carry a `list[...]` value per request rather than a single scalar — each list element is itself a measurement. Today this is `inter_chunk_latency` only: every request contributes one inter-chunk gap per pair of consecutive streamed chunks.

At the run level the records-manager has to summarize across all per-request lists into a single set of stats (`avg`, `min`, `max`, `std`, `p50`, `p90`, `p99`, …). Naïvely concatenating the lists into a flat array gives exact stats but linear memory: `records × samples_per_record × 8 B`. For a long-context streaming benchmark (1 M requests × ~5 K chunks/request) that reaches **~37 GB** on the records-manager pod alone — the original cause of an OOM at ramp scale.

To bound memory, AIPerf aggregates list-valued record metrics with a **t-digest sketch** + **five running side-channel scalars**.

## What stays exact, what becomes approximate

| Stat | Source | Accuracy |
|---|---|---|
| `count` | running `int` | bit-exact |
| `sum` | running `float64` | bit-exact (within float round-off across summation orders) |
| `min`, `max` | running scalars | bit-exact |
| `avg` | `sum / count` | bit-exact |
| `std` | `sqrt(max(0, sum_sq/count − avg²))` | bit-exact (population std, matches `np.std`) |
| `p1` … `p99` | t-digest sketch | approximate — see empirical band below |

Memory cost of the side-channel scalars is **40 bytes** regardless of sample count. T-digest centroids stay bounded (~4 KB sketch at the default compression) regardless of sample count.

## Empirical accuracy

Backed by [`crick.TDigest`](https://github.com/dask/crick) (Cython/C). Measured against a numpy reference across five trials at three workload sizes, fresh RNG seed per trial, lognormal `mean=ln(5), sigma=0.4` clipped to [0.5, 50] ms (representative ICL distribution at moderate ITL):

| Sample count | Worst-case max %err across percentiles | Throughput |
|---:|---:|---:|
| 200 K | 0.068 % | ~9.6 M updates/s |
| 5 M | 0.021 % | ~12.0 M updates/s |
| 50 M | **0.012 %** | ~12.0 M updates/s |

At the default compression the worst-case relative error is **40× under** the 0.5% band a t-digest typically promises. Mid-range percentiles (p10–p90) are tighter still — usually 5-digit relative agreement with the exact numpy reference.

The compression parameter is exposed as `AIPERF_METRICS_TDIGEST_COMPRESSION` (default 500) for benchmarks that want even tighter percentiles at the cost of a slightly larger sketch.

## Per-record values are unchanged

The aggregation described above is **only** at the run-level. The per-record JSONL (`profile_export.jsonl`) preserves each request's full `list` value verbatim — exact, byte-for-byte, ready for downstream tooling like `aiperf plot` to compute its own per-request stats.

## What this means for benchmark output

For ICL specifically:

- The numbers in `profile_export_aiperf.{json,csv}` come from the t-digest aggregator. Percentile values typically match a direct numpy computation to within 0.05% relative error on benchmark-scale runs; tail percentiles (p1, p99) at small N exhibit slightly more rank-jitter but stay well under the 0.5% band.
- `count`, `sum`, `min`, `max`, `avg`, `std` are computed exactly and match what an exact array would produce.
- Per-request ICL lists in `profile_export.jsonl` are unchanged — anything that needs sample-level precision can read those.

For all other metrics: **no change**. Scalar record metrics still use the exact-storage `MetricArray` path. Aggregate metrics (`inter_token_latency`, `request_latency`, etc.) compute through their own existing aggregator; t-digest is not in their path.

## Where it lives

- Aggregator class: [`src/aiperf/metrics/list_metric_aggregation.py`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/metrics/list_metric_aggregation.py) — `TDigestListMetricAggregator`.
- Selection site: [`src/aiperf/post_processors/metric_results_processor.py`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/post_processors/metric_results_processor.py) — first-touch dispatch by `isinstance(value, list)`.
- Compression knob: `Environment.METRICS.TDIGEST_COMPRESSION` (env: `AIPERF_METRICS_TDIGEST_COMPRESSION`, default 500).
- Dependency: [`crick~=0.0.8`](https://pypi.org/project/crick/) (Cython/C-backed t-digest, BSD-3, dask-org maintained).

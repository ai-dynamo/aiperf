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
| `p1` … `p99` | t-digest sketch | approximate, ≤ ~0.5 % relative error |

Memory cost of the side-channel scalars is **40 bytes** regardless of sample count. T-digest centroids stay bounded (low single-digit KB) regardless of sample count.

## Empirical accuracy

Measured on a 1 470-sample run (30 streaming requests × 49 chunks each) against an exact numpy reference:

| field | exact | t-digest | rel diff |
|---|---|---|---|
| `min` | 3.0497700 | 3.0497700 | 0 |
| `max` | 6.9478710 | 6.9478710 | 0 |
| `avg` | 4.9999602 | 4.9999602 | 0 |
| `std` | 0.7915493 | 0.7915493 | 3.3 × 10⁻¹³ |
| `p50` | 5.2294285 | 5.2289453 | 0.0092 % |
| `p90` | 6.2683405 | 6.2703040 | 0.031 % |
| `p99` | 6.5617756 | 6.5755314 | 0.21 % |

Mid-range percentiles are ~100× tighter than the 0.5 % band; the extreme tails (p99 in particular) are noisiest because t-digest is rank-accurate and the rank jitter at the tail translates to value jitter. At larger N the gap shrinks further.

## Per-record values are unchanged

The aggregation described above is **only** at the run-level. The per-record JSONL (`profile_export.jsonl`) preserves each request's full `list` value verbatim — exact, byte-for-byte, ready for downstream tooling like `aiperf plot` to compute its own per-request stats.

## What this means for benchmark output

For ICL specifically:

- The numbers in `profile_export_aiperf.{json,csv}` come from the t-digest aggregator. Percentile values may differ from a direct numpy computation by up to ~0.5 % at small benchmark sizes; at large sizes the error is well below benchmark noise.
- `count`, `sum`, `min`, `max`, `avg`, `std` are computed exactly and match what an exact array would produce.
- Per-request ICL lists in `profile_export.jsonl` are unchanged — anything that needs sample-level precision can read those.

For all other metrics: **no change**. Scalar record metrics still use the exact-storage `MetricArray` path. Aggregate metrics (`inter_token_latency`, `request_latency`, etc.) compute through their own existing aggregator; t-digest is not in their path.

## Where it lives

- Aggregator class: [`src/aiperf/metrics/list_metric_aggregation.py`](../../src/aiperf/metrics/list_metric_aggregation.py) — `TDigestListMetricAggregator`.
- Selection site: [`src/aiperf/post_processors/metric_results_processor.py`](../../src/aiperf/post_processors/metric_results_processor.py) — first-touch dispatch by `isinstance(value, list)`.
- Dependency: [`tdigest~=0.5.2.2`](https://pypi.org/project/tdigest/) (pure Python, no transitive C/Rust deps).

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native SGLang speculative-decoding console table

## Status

Approved design for the native Rust port of origin/main commit
`34b2be2ee1159cc7e6985e027027791d18dad693`.

## Problem

Native server-metrics collection already parses and exports SGLang
`spec_accept_rate` and `spec_accept_length` gauges. Native SPEED-Bench report
generation also recognizes their exported names. Neither path puts those
statistics into ordinary native profile console output: the fixed-width console
exporter reads request metrics and grouped errors but ignores the separate
`NativeReport.server_metrics` map.

The upstream Python exporter is already merged, but native `aiperf profile`
does not invoke it. Users therefore lose the direct end-of-run indication that
SGLang speculative decoding ran and how many drafts it accepted.

## Goals

1. Render the upstream SGLang speculative-decoding summary in native
   `profile_export_console.txt` and the corresponding non-TTY console surface.
2. Select only series for configured models and leader PP/TP ranks.
3. Preserve distinct matching endpoint/label series instead of averaging them.
4. Match upstream display scale, precision, finite-value, and activity rules.
5. Leave raw server-metrics report and export values unchanged.

## Non-goals

- Changing server-metric scraping, parsing, accumulation, or metric names.
- Adding vLLM or TensorRT-LLM console tables; the upstream commit adds an
  SGLang-specific exporter.
- Changing SPEED-Bench report extraction.
- Adding user-facing flags or a second console exporter registration.
- Changing the public native-v2 report schema or mutating server-metrics
  statistics for display.
- Re-merging upstream commit `34b2be2ee1`; exact ancestry is already present.

## Configuration contract

The resolved internal console policy will carry
`model_names: Vec<String>`, copied from every authored
`models.items[*].name` in stable order. The wire decoder will default the field
to an empty vector so older internal protocol fixtures remain decodable.

This field belongs to console presentation policy, not `NativeReport`: it is
needed only to reproduce upstream configured-model filtering and must not
expand the committed report schema. An empty configured-model set produces no
speculative-decoding table.

## Series selection

The console renderer will inspect only these keys in
`NativeReport.server_metrics`:

- `sglang:spec_accept_rate`
- `sglang:spec_accept_length`

A selected series must satisfy all of the following:

1. Its stats shape is a gauge distribution.
2. Its labels contain `model_name`, whose Unicode-lowercased value equals one
   configured model's Unicode-lowercased value.
3. `pp_rank` is absent or exactly `"0"`.
4. `tp_rank` is absent or exactly `"0"`.

No selected series are folded together. Their existing deterministic report
order is retained.

## Activity gate

Before building rows, collect matching acceptance-length series. If at least
one exists, render only when at least one has a finite `max` greater than zero.
An all-zero matching length set suppresses the whole table, including a
positive rate, because SGLang can register these gauges before speculative
decoding runs.

When no matching acceptance-length series exist, use matching acceptance-rate
series as the fallback activity signal and require at least one finite positive
`max`. Missing, non-finite, or non-positive maxima do not prove activity.

## Row construction and rendering

Each selected series becomes one row only when `avg`, `min`, `max`, `p50`, and
`p90` are all present and finite both before and after display scaling. Invalid
rows are omitted without failing the authoritative profile report.

The table title is exactly
`NVIDIA AIPerf | Server Metrics: Speculative Decoding`. Its columns are
`Metric`, `mean`, `min`, `max`, `p50`, and `p90`. Rows are configured as:

| Source metric | Row name | Display scale | Precision |
| --- | --- | ---: | ---: |
| `sglang:spec_accept_rate` | `Accept Rate (%)` | 100 | 1 |
| `sglang:spec_accept_length` | `Accept Length` | 1 | 2 |

Scaling is presentation-only. `NativeReport.server_metrics` and
`server_metrics_export.*` continue to contain raw ratios and lengths.

When a metric has one selected series, its row uses the base row name. For
multiple selected series, append a parenthesized suffix containing:

1. normalized `endpoint=<value>` when endpoint values differ;
2. `model_name=<value>` when model-label values differ;
3. every other label whose value differs, sorted by label name, excluding
   `model_name`, `pp_rank`, and `tp_rank`;
4. `series=<1-based index>` when the selected series still have no
   distinguishing suffix component.

Endpoint display removes the scheme and a trailing `/metrics`, reusing the
existing native export normalization helper. Native table rendering is plain
text, so label values are not interpreted as markup.

The table is a separate table block after the grouped request-metric tables and
before usage-discrepancy and OSL-mismatch warning panels. Runs without valid
active rows retain their previous bytes.

## Error handling and performance

The renderer performs no I/O and returns absence for unsupported or incomplete
data. It must not log once per metric series, panic on missing labels/stats, or
index unchecked collections. Display multiplication must reject overflow to a
non-finite value.

This work occurs once after execution over two named metric entries. It is not
part of request dispatch, token observation, scheduling, or accumulation hot
paths, and needs no synchronization or new dependency.

## Acceptance tests

1. A mixed-case configured model renders the matching leader rate and length
   with exact one-/two-decimal values; another model and nonzero PP/TP ranks are
   absent.
2. Matching rows from distinct endpoints and differing `dp_rank` labels remain
   separate; common labels do not clutter suffixes; a non-finite required stat
   omits only its row.
3. A positive matching rate renders when length is absent. Adding matching
   all-zero length series suppresses the table.
4. Config export carries all authored model names in order.
5. Focused tests, formatting, runtime Clippy, release CLI build, independent
   task review, and Graham review complete with no unresolved Critical or
   Important finding.

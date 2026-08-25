<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Commit 003 — `34b2be2ee1`

Upstream subject: `feat: support Speculative Decoding metrics in AIPerf (#1153)`.

## Scope decision

**Applicable native profile console-exporter port.** Upstream adds a dedicated
SGLang speculative-decoding table to ordinary profile console output. It is not
only server-metric recognition: it selects configured models, removes duplicate
scheduler-rank series, scales acceptance rate for display, preserves distinct
endpoint/label series, rejects incomplete or non-finite rows, and suppresses
gauges that were registered but never became active.

The exact upstream commit is already present in native branch ancestry through
two-parent merge `4d076c660f31d9a9bf66f839867c3b9737e1a0ba`, whose second parent is
`34b2be2ee1159cc7e6985e027027791d18dad693`. This remediation must not add a
second merge of that commit.

## Code evidence

- Upstream `SGLangSpeculativeDecodingConsoleExporter` consumes
  `sglang:spec_accept_rate` and `sglang:spec_accept_length`, matches each
  series' `model_name` against every configured model case-insensitively, and
  keeps only `pp_rank="0"` / `tp_rank="0"` leaders (with absent rank labels
  treated as zero).
- The upstream table displays mean, min, max, p50, and p90. Acceptance rate is
  multiplied by 100 and shown with one decimal; acceptance length is shown with
  two decimals. The server-metrics artifacts retain their raw values.
- Upstream renders separate rows rather than averaging multiple matching
  series. Row suffixes distinguish endpoints and only labels whose values
  differ among the selected series. Rows with any required non-finite summary
  value are omitted.
- Upstream considers a matching positive acceptance-length maximum the primary
  proof that speculation ran. If matching length series exist but are all zero,
  it suppresses the table even when the rate gauge is present. A positive rate
  maximum is the fallback only when no matching length series exist.
- Native `rust/runtime/src/server_metrics/accumulator.rs` already retains the
  necessary SGLang gauges as labeled `SidecarMetric` distributions, and
  `NativeReport.server_metrics` preserves their endpoints, labels, and summary
  statistics.
- Native `rust/cli/src/speed_bench.rs` recognizes the two SGLang names only
  while reading `server_metrics_export.json` for an explicit SPEED-Bench
  report. That path does not affect ordinary `aiperf profile` console output.
- Native `rust/runtime/src/export/console_txt.rs` currently renders request
  metrics and warnings only. It never reads `NativeReport.server_metrics`, so
  the merged Python console exporter cannot supply native profile parity.

## Port decision

Carry the ordered configured model-name set into the internal
`cfg.export.console_txt` policy. The native console exporter will use that set
to select matching SGLang gauge series and render a dedicated
`NVIDIA AIPerf | Server Metrics: Speculative Decoding` table after the primary
metric tables and before warning panels.

The native table will preserve upstream selection and activity semantics:
case-insensitive configured-model matching; leader PP/TP rank filtering;
per-series endpoint and varying-label disambiguation; finite
avg/min/max/p50/p90 enforcement; percent-only display scaling; and
length-first inactive-gauge suppression. It will consume the finalized report
model only and will not alter collection, server-metrics JSON/CSV values,
SPEED-Bench projection, or the public native-v2 report schema.

## Verification requirements

- A console renderer regression must prove the matching mixed-case leader
  series renders rate and length with exact scale/precision while other models
  and non-leader PP/TP ranks do not appear.
- A multi-series regression must prove endpoint and varying-label distinctions
  are retained without averaging and that a row missing any finite required
  statistic is omitted.
- An activity regression must prove a positive rate renders when no matching
  length gauge exists, while adding an all-zero matching length gauge suppresses
  the entire table.
- A config-export regression must prove every configured model name reaches the
  internal console policy in authored order.
- Focused runtime tests, formatting, Clippy, and a release CLI build must use
  `/usr/bin/sccache` and a Cargo target below `/mnt/4tb`.
- The complete native implementation range must receive independent task review
  and Graham review with no unresolved Critical or Important findings.

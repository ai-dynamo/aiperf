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

## Upstream test scope

The exact upstream commit adds one test path,
`tests/unit/exporters/test_server_speculative_decoding_console_exporter.py`.
It adds no integration or E2E test. The native port therefore strengthens the
upstream coverage with a Rust product test that launches the real
`aiperf profile` binary against the in-process native mock and reads the emitted
console and raw server-metrics artifacts.

## Closure evidence

The existing two-parent merge
`4d076c660f31d9a9bf66f839867c3b9737e1a0ba` remains the only merge of exact
upstream commit `34b2be2ee1159cc7e6985e027027791d18dad693`.
The native remediation landed as:

- `debc81e3ef4f93b037297d05605bc31a6a85d90b` projects every configured model
  name into the compatibility-defaulted internal console policy.
- `67fd3896f963a10d6dfe66666203cdbbb63e802c` renders active, finite,
  leader-selected SGLang speculative series without changing raw report data.
- `91aae2ee0587accd97dc1c640320035cfacb0e23` adds deterministic native mock
  gauges and the real-profile Rust E2E; `ae7c592f99c60645df162912434a932a5e8b89f7`
  strengthens it to bind every exact display cell to its metric row.
- `475352c0d318526408f43fbdbc0deb1e0ee269c9` documents the native mock's raw
  SGLang fixture values and leader labels.

Fresh verification used `/usr/bin/sccache` and
`/mnt/4tb/aiperf-origin-port-003-remediation-target`. The console exporter
module reported 24 passed; config-export tests reported 7 passed; the real
native profile E2E reported 1 passed; mock Prometheus tests reported 3 passed;
the debug CLI/mock build, runtime all-target engine Clippy, and release CLI
build all exited successfully. The release build completed in 3m35s. Direct
Rustfmt checks on all changed Rust files and the complete range diff check
passed. Workspace-wide formatting remains blocked only by the unrelated
pre-existing wrapping difference in
`rust/runtime/src/engine/sidecar_input.rs:787`, which this port leaves intact.

Task reviews are clean after one E2E assertion-strength fix. Whole-range review
is approved with zero Critical, Important, or Minor findings after its mock
catalog follow-up. The final systems review ends `GRAHAM APPROVED` with zero
findings. Detailed reports live in
`.superpowers/sdd/2026-08-25-native-sglang-speculative-console/`.

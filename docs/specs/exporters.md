<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Exporters

## Purpose

`aiperf_runtime::export` is the native output plane: a typed IO-free native-v2
report core plus a static set of `Exporter` implementations behind one trait. It
is the sole emitter on the native path. `AIPERF_RUNTIME_NATIVE_EXPORT` exists
only in the Python package (`src/aiperf/common/environment.py`) and gates the
legacy Python mesh's own emission; the native binary never reads it, so it
cannot redirect native export.

## Built

### Report core and trait

The typed `Reporter` produces one `NativeReport` (see [metrics.md](metrics.md)),
which every exporter renders. The `Exporter` trait declares `enabled(cfg) -> bool`;
`ExporterRegistry` runs each enabled exporter over the finalized report in emit
order. The v2 report uses a proper nested `metrics` map keyed by name and keys off
metric type: distribution metrics get `avg`/percentiles, scalar metrics get a
plain `value`.

### Sinks

Nine sinks behind the one trait:

- aiperf-v1 (genai-perf-v1) JSON + CSV
- timeslice JSON + CSV
- server-metrics JSON + CSV + Parquet
- accuracy CSV
- console `.txt` (with warning/insight detectors)
- OTLP per-record metrics
- MLflow
- W&B

Per-record formats are selected through `artifacts.records`. Parquet requires the
`parquet` Cargo feature; with the feature off the config is still accepted and no
Parquet exporter is registered.

## Source anchors

- `rust/runtime/src/export/` (`mod.rs` `Exporter`/`ExporterRegistry`,
  `genai_perf.rs`, `timeslice.rs`, `server_metrics/`, `accuracy_csv.rs`,
  `console_txt.rs`, `otel.rs`, `mlflow.rs`, `wandb/`, `parquet.rs`,
  `per_record_parquet.rs`).
- `rust/runtime/src/metrics_core/report.rs`, `rust/runtime/src/report.rs`.
- `rust/e2e-tests/tests/{test_exporters.rs,test_records_parquet.rs}`.

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Metrics

## Purpose

`aiperf_runtime::metrics_core` is the IO-free metrics engine: it accumulates
observer events into a column store, computes record, aggregate, derived,
phase-window, and sweep metrics, and produces the typed report the exporter plane
consumes. It links to but does not duplicate transport or telemetry detail.

## Built

### Engine

The accumulator is a NaN-sparse column store with exact ragged replay. It computes
record, aggregate, and derived metrics with authoritative completion-usage
reconciliation (server usage wins when present, absent fields stay absent), SLO
goodput, all effective/active and ICL-aware sweep curves, duration-weighted
statistics, phase windows and timeslices, and per-model/per-endpoint series.
Worker-local stores merge deterministically at a boundary (see
[execution-model.md](execution-model.md)); graph workers merge lean local stores.
Online/scheduled/adaptive/accuracy adapters feed observer timing, classification,
and usage plus real request traces. The typed `Reporter` produces the
`NativeReport` the exporters render (see [exporters.md](exporters.md)).

### Catalog

`metrics_core::catalog` is the metric identity graph: the inherited metric
identities plus the native sweep identities, each with exact metadata and
dependencies and an implementation for every record/aggregate/derived row whose
source data exists. Validation and a deterministic metadata fingerprint pin the
graph. Telemetry-owned injected rows stay absent until their producer supplies
values (see [telemetry.md](telemetry.md)).

### Exact vs sketch

Exact mode retains records as configured artifacts require. Sketch mode
(`--sketch-metrics` or `AIPERF_METRICS_SKETCH=1`) uses mergeable t-digests: counts,
sums, extrema, and rate aggregates remain exact; percentiles and standard
deviation are streaming estimates; per-record outputs are unavailable. Numeric
values stay finite or explicitly absent at serialization boundaries.

## Future requirements

- Genai-perf v1 export exists as one exporter sink; further v1 compatibility
  breadth is scoped by the exporter plane, not the engine.

## Source anchors

- `rust/runtime/src/metrics_core/` (`accumulator.rs`, `store.rs`, `ingest.rs`,
  `derived.rs`, `sweepline/`, `window.rs`, `catalog.rs`, `report.rs`, `value.rs`,
  `counter.rs`, `units.rs`, `accuracy.rs`).
- `rust/runtime/src/metrics.rs`, `rust/runtime/src/report.rs`.
- `rust/cli/tests/{sweep_parity.rs,sweep_aggregate_parity.rs,sketch_env.rs}`.

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Definition registry

## Purpose

This record states the target for a **single shared definition layer**: one place
where every named output — benchmark metrics today, dataset-analysis outputs next,
server and GPU telemetry later — gets a lookup-only `Definition` carrying its
display identity (`header`, units, `larger_is_better`, `value_type`, grouping,
ordering) and nothing about how it is computed. Consumers that compare a value
against an SLA threshold, print a metrics table, order columns, or render a value
resolve those questions through the registry instead of re-deriving them locally.

The goal is to end three current gaps at once:

1. **No shared identity across families.** Benchmark metrics have `MetricTag` +
   `CATALOG`; dataset-analysis outputs have none — their identity is a serde field
   name or an ad-hoc string literal in the CSV writer
   (`export/dataset_analysis.rs`), so a new distribution silently misses the CSV
   until a `rows.push` line is added by hand.
2. **Definition tangled with computation.** `MetricSpec` mixes presentation
   (`header`, `unit`, `display_unit`, `console_group`, `plot_direction`,
   `value_type`) with the dependency graph, `kind`, and `aggregation`. A consumer
   that only wants the display identity must depend on the whole computation
   vocabulary.
3. **Completeness is a convention, not an invariant.** `spec_for` is
   `CATALOG.iter().find(...)` (a linear scan; flagged in review) returning
   `Option`; a `MetricTag` with no catalog row compiles and only fails at runtime
   where callers `.expect(...)`. Completeness is guarded only by
   `assert_eq!(CATALOG.len(), 125)`, which does not catch an added enum variant.

[runner-protocol.md](runner-protocol.md) and the metrics plane record the current
built behavior; this record states how the definition layer is meant to converge.

## Built

The shared definition layer is built for the two static families (benchmark metrics
and analyzer outputs). Server and GPU telemetry remain outside it — only the
namespaced seam is reserved.

### The `Definition` type

A lookup-only value (`runtime/src/metrics_core/definition.rs`); no dependency graph,
aggregation, or accumulation state. `#[non_exhaustive]`, so future families are
additive:

```rust
#[non_exhaustive]
pub struct Definition {
    pub id: DefinitionId,             // namespaced full tag name — the key
    pub header: &'static str,
    pub short_header: Option<&'static str>,
    pub short_header_hide_unit: bool,
    pub unit: Unit,                   // native unit; used for math / SLA compare
    pub display_unit: Option<Unit>,   // None => unit
    pub display_order: Option<u32>,
    pub group: DefinitionGroup,
    pub larger_is_better: bool,       // first-class; subsumes the flag + PlotMetricDirection
    pub value_type: MetricValueType,
    pub aliases: &'static [&'static str],
    pub deprecated_since: Option<&'static str>,
}
```

The three consumer questions are centralized as methods:

- `effective_display_unit(&self) -> Unit` — `display_unit.unwrap_or(unit)`.
- `passes_threshold(&self, value: Native, threshold: Native) -> bool` — the one
  place SLA direction logic lives (`larger_is_better ? value >= threshold : value <=
  threshold`). Both operands are native-unit newtypes, so a mixed-scale comparison is
  a **compile error**.
- `format_value(&self, value: f64) -> String` — (unit + `value_type` + precision
  policy) backs the new `aiperf metrics list/describe` command and provides a shared
  render seam for future sinks; the existing exporters (console, CSV, JSON, OTLP,
  W&B, etc.) intentionally retain their current byte-identical rendering and were not
  migrated in this change.

`MetricValueType` carries an `Int` variant alongside the float/duration cases so
count-typed metrics render without a spurious decimal.

### Typed units for SLA

`Native(f64)` / `Display(f64)` newtypes make unit-scale mismatch unrepresentable —
hand-rolled (not `uom`), implementing only dimensionally valid arithmetic and `From`
in the safe direction; `native + display` does not compile. SLA now goes through
typed `Native` and `Definition::passes_threshold`: `SloThreshold` resolves the
threshold once at construction (`native`/`from_display`, pre-resolved
`larger_is_better`), so `compute_good_request` does no per-record registry lookup.

### `id` as a versioned contract

The `id` is namespaced (`aiperf.<tag>`, `analyzer.*`), making cross-family collisions
structurally impossible. A rename keeps the old id in `aliases`; `resolve` honors it.
An `insta` JSON snapshot of the sorted public id set (134 ids) turns any accidental
rename or removal into a reviewable diff.

### Static core: compile-time complete, O(1)

`MetricTag` is a fieldless, contiguously-discriminated enum
(`runtime/src/metrics_core/catalog.rs`). The catalog is now a `const`-static
`CATALOG: [MetricSpec; MetricTag::COUNT]` array — the runtime post-pass
(`configure_catalog_metadata`) was folded into the const literals, and the 58 rows
that built flags with the `A | B` operator were rewritten to `.union()` (bitflags 2.x
`union` is const). `metric_definition(tag)` is O(1) via discriminant indexing into the
array, guarded by a discriminant-order check.

**Deviation (d):** compile-time completeness comes from the **const-array length**
(`[MetricSpec; MetricTag::COUNT]` cannot be built without one row per variant) plus a
discriminant-order guard, **not** from a 125-arm exhaustive `const fn` match as the
spec originally envisioned. The array length is the completeness invariant; a new
`MetricTag` variant without a row fails to build.

**Deviation (a):** the string-keyed lookup is a `LazyLock<HashMap<&str,
&'static Definition>>`, **not** a compile-time `phf` map. A `const phf` cannot hold
`&CATALOG[..].def` references (the catalog array is itself a `static`, so its element
addresses are not `const`-known), so the id map is built once at first use. The
`definition(id)` / `resolve(name)` API (exact match incl. aliases; `resolve` applies
the `turnN_` base-concept rule) is unchanged from the design.

### Analyzer outputs: base concepts + resolver

Analyzer outputs register a small static set of **base concepts** (`analyzer.isl`,
`analyzer.osl`, `analyzer.per_turn_isl`, …) in the same registry. Parameterized names
(`turn3_isl`) are not stored per name; the producer holds the base id and composes the
concrete label. `export/dataset_analysis.rs` sources its CSV row names from the
registry, producing byte-identical output.

**Deviation (c):** the analyzer CSV registry-sourcing currently materializes only the
**row-name token** from the base definition (the CSV has no unit/header column to
drive), so header/unit fields on the analyzer defs are carried but not yet rendered
into that artifact.

### `MetricSpec` refactor

`MetricSpec` now embeds a `Definition` and holds only computation plus the typed
console group:

```rust
pub struct MetricSpec {
    pub tag: MetricTag,
    pub def: Definition,          // def.id == the metric's namespaced id
    pub flags: MetricFlags,       // minus LARGER_IS_BETTER (now def.larger_is_better)
    pub console_group: MetricConsoleGroup,
    pub required: &'static [MetricTag],
    pub kind: MetricType,
    pub aggregation: Option<AggregationKind>,
}
```

Seven legacy scalar presentation fields (`header`, `short_header`,
`short_header_hide_unit`, `unit`, `display_unit`, `display_order`, `plot_direction`)
were removed; consumer sites read through delegating accessors (`spec.header()` →
`spec.def.header`). `plot_direction` is gone; `def.larger_is_better` is the single
source of direction (a `plot_direction_for(&def)` helper derives the plotting enum).

**Deviation (b):** `console_group` is **retained as a typed `MetricConsoleGroup`
metric-render field** on `MetricSpec` (it drives console table grouping), while
`def.group = DefinitionGroup::Named(console_group)` carries the portable string label
in the `Definition`. The typed field was not collapsed into `def.group`.

### Definition-driven rendering and CLI

`Definition::format_value` renders values and `header`/`short_header`/
`effective_display_unit`/`display_order`/`group` drive layout. `aiperf metrics list`,
`aiperf metrics describe <id>`, and `aiperf metrics list --markdown`
(`cli/src/metrics_list.rs`) dump the registry, so a metrics reference is produced from
the registry and cannot drift.

### Hot-path boundary (invariant, holds)

`definition()`/`resolve()` return `&'static` and are config/render-time only; the
per-record and per-token paths never call them. The one hot-path direction question
(`larger_is_better`) is pre-resolved onto `SloThreshold` at construction. The audit
confirmed zero metric-catalog lookups in transport, dispatch, scheduler, `scheduled`,
`request_rate`, `phase_runtime`, graph, or engine paths.

## Future requirements

### Extensibility seam (server / GPU) — seam only

Server metrics (`runtime/src/server_metrics/`, arbitrary Prometheus name +
`labels: BTreeMap<String,String>`, discovered at runtime) and GPU telemetry
(`runtime/src/gpu_telemetry/`, fixed record struct) are **not merged into the registry
yet** — only the namespaced-id seam (`server.*`, `gpu.*`) is reserved. Whole future
families are intended to register through a `DefinitionSource` trait yielding
`&'static Definition`s, collected at bootstrap into a runtime `OnceLock`/`LazyLock`
view, reusing the existing `AIPerfRegistry`/`AIPerfExtension` transactional pattern
(duplicate ids rejected, fail-closed). Truly dynamic server metrics resolve through a
`server.*`-namespaced escape hatch, kept distinct from the closed static families.

### Non-goals

- No change to the computation graph, `kind`, `aggregation`, or accumulation.
- No wire/artifact format change: console, CSV, and JSON output stay byte-compatible.
- Server and GPU families are not merged now; only the namespaced seam is
  established.

## Source anchors

- `runtime/src/metrics_core/catalog.rs` — `MetricTag` (fieldless, contiguous,
  `COUNT`/`index`), `MetricSpec` (embedded `def` + computation + typed
  `console_group`), `spec!`, the const-static `CATALOG: [MetricSpec; COUNT]`,
  `metric_definition`/`spec_for` (discriminant indexing), `plot_direction_for`,
  `validate_catalog`.
- `runtime/src/metrics_core/definition.rs` — `Definition`,
  `DefinitionId`/namespacing, `DefinitionGroup`, `Native`/`Display`,
  `effective_display_unit`/`passes_threshold`/`format_value`, the
  `LazyLock<HashMap>` id registry, `definition`/`resolve` (+ aliases, `turnN_` rule).
- `cli/src/metrics_list.rs` — `aiperf metrics list`/`describe <id>`/`--markdown`
  registry dump.
- `runtime/src/metrics_core/accumulator.rs` — `SloThreshold`
  (`native`/`from_display`, pre-resolved `larger_is_better`), `compute_good_request`;
  moves to `Definition::passes_threshold`.
- `runtime/src/metrics_core/units.rs` — `Unit`, `MetricValueType`, `convert_value`
  (non-const; config-time conversion).
- `runtime/src/dataset/analysis.rs`, `runtime/src/export/dataset_analysis.rs` —
  analyzer base-concept definitions; CSV/JSON writers move from string literals to
  registry lookups.
- `runtime/src/extensions/` — `AIPerfRegistry`/`AIPerfExtension`, reused for the
  `DefinitionSource` family-registration seam.
- Consumers reading presentation fields: `runtime/src/metrics_core/report.rs`,
  `sweepline/mod.rs`, `sidecar.rs`, `runtime/src/export/console_txt.rs`,
  `export/wandb/mod.rs`, `cli/src/compare.rs`.

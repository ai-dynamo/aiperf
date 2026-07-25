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

The pieces this record unifies exist today, separately.

**Benchmark metrics — registry-driven, but definition and computation are fused.**
`MetricTag` is a fieldless, contiguously-discriminated enum of 125 variants
(`runtime/src/metrics_core/catalog.rs`), with `MetricTag::COUNT` and
`index(self) -> usize { self as usize }` already relying on that contiguity.
`MetricSpec` carries both presentation fields (`header`, `short_header`,
`short_header_hide_unit`, `unit`, `display_unit`, `display_order`, `console_group`,
`plot_direction`, `value_type`) and computation fields (`flags`, `required`,
`kind`, `aggregation`). `CATALOG` is a `LazyLock<Vec<MetricSpec>>`: a `vec![spec!…]`
literal list, then a runtime post-pass, `configure_catalog_metadata`, that *mutates*
`short_header`, `value_type`, and `plot_direction` per tag after construction — this
post-construction mutation, not the graph validation, is what forces `LazyLock`.
`spec_for(tag)` is a linear `CATALOG.iter().find`. `larger_is_better` is encoded as
`MetricFlags::LARGER_IS_BETTER`, and `plot_direction` is derived from it.
`validate_catalog` (uniqueness, dependency resolution, tier rules, acyclicity via a
petgraph toposort) runs at bootstrap and feeds `DERIVED_TOPO_ORDER`; it does **not**
assert every `MetricTag` has a spec.

The SLA/good-request direction was recently pre-resolved off the per-record path:
`SloThreshold` now carries a `larger_is_better: bool` computed once at construction
(`native`/`from_display`), and `compute_good_request` reads it directly. Every
metric-catalog `spec_for` caller lives in `metrics_core/accumulator.rs` and runs at
config-time or end-of-phase summary-time; the transport/dispatch/scheduler/graph/
engine hot paths contain zero metric-catalog lookups.

**Dataset-analysis outputs — a plain typed report, no registry.** `DatasetAnalysis`
(`runtime/src/dataset/analysis.rs`) is a struct tree of typed sections holding
`Option<StatSummary>` distributions. Output identity is the serde field name (JSON)
or a hand-written string literal (`rows.push(("isl", …))`, `format!("turn{ti}_isl")`)
in `export/dataset_analysis.rs`. The only shared schema is `const STAT_KEYS`, reused
to match the genai_perf CSV columns. There is no tag enum, no `spec_for`, no
validation.

**Server and GPU telemetry — separate, partly dynamic.** Server metrics
(`runtime/src/server_metrics/`) are scraped Prometheus samples keyed by an arbitrary
name plus `labels: BTreeMap<String,String>`, discovered at runtime. GPU telemetry
(`runtime/src/gpu_telemetry/`) has its own fixed record struct. Neither is in any
registry.

## Future requirements

### The `Definition` type

A lookup-only value; no dependency graph, aggregation, or accumulation state.

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

Methods centralize the questions consumers ask:

- `effective_display_unit(&self) -> Unit` — `display_unit.unwrap_or(unit)`.
- `passes_threshold(&self, value: Native, threshold: Native) -> bool` — the one
  place SLA direction logic lives: `if larger_is_better { value >= threshold } else
  { value <= threshold }`. Both operands are native-unit newtypes (see below), so a
  mixed-scale comparison is a **compile error**, not a latent bug.
- `format_value(&self, value: f64) -> String` — the single renderer (unit +
  `value_type` + precision policy) every sink uses, so a value renders identically
  in console, CSV, and JSON.

`Definition` and the family/group enums are `#[non_exhaustive]` so future families
are additive, never a breaking change.

### `id` as a versioned contract

The `id` is a CSV header, a JSON key, and an SLA-config token that external tools
parse, so it is treated as a stable contract:

- **Namespaced** — `aiperf.ttft`, `analyzer.isl`, later `server.<name>`,
  `gpu.<field>`. Namespacing makes cross-family collisions structurally impossible
  and keeps origin legible.
- **Aliased** — a rename keeps the old id in `aliases`; `resolve` honors it, so
  renaming a Rust variant never breaks a downstream parser or an existing config.
- **Snapshotted** — an `insta` JSON snapshot of the sorted public id set (with
  `INSTA_UPDATE=no` in CI) turns any accidental rename or removal into a reviewable
  diff rather than a silently broken artifact. Snapshot diffs are reviewed with
  `cargo insta review`, never blanket-accepted.

### Static core: compile-time complete, O(1)

The 125 metric definitions live behind an **exhaustive `const fn` match** on
`MetricTag`, returning `&'static Definition`:

```rust
pub const fn metric_definition(tag: MetricTag) -> &'static Definition {
    match tag {
        MetricTag::RequestCount => &REQUEST_COUNT_DEF,
        // … one arm per variant, no wildcard …
    }
}
```

The exhaustive, wildcard-free match makes a `MetricTag` with no `Definition` a
**compile error** (the same mechanism `as_str` already uses for names), replacing
the runtime `assert_eq!(len, 125)`. It is the exhaustiveness contract; we do **not**
rely on `as usize` indexing into a parallel array for correctness, because a
fieldless enum without an explicit `#[repr]` has an unspecified discriminant layout
and reordering would silently misalign. LLVM lowers the small-integer match to a
jump table, so lookup stays effectively O(1). `spec_for` is reimplemented over this
match (or the const array derived from it), retiring the `CATALOG.iter().find`
linear scan.

The string-keyed lookup for the static core (metric ids + analyzer base ids) is a
compile-time **`phf`** perfect-hash map `&'static str -> &'static Definition` —
collision-free, zero runtime init:

```rust
pub fn definition(id: &str) -> Option<&'static Definition>;  // exact, incl. aliases
pub fn resolve(name: &str) -> Option<&'static Definition>;   // exact, else base-concept
```

Because the whole static core is `const`/`phf`, it carries no `LazyLock`. Runtime
initialization is reserved for the genuinely non-const parts, which stay as separate
`static … : LazyLock<…>` (never `const`): `validate_catalog`/`DERIVED_TOPO_ORDER`
(petgraph toposort) and any future dynamic family (below). The immutable catalog data
and the derived/validated views are split cleanly.

### Analyzer outputs: base concepts + resolver

Analyzer outputs register a small static `&[Definition]` of **base concepts** —
`analyzer.isl`, `analyzer.osl`, `analyzer.total`, `analyzer.isl_osl_ratio`,
`analyzer.turns_per_conversation`, `analyzer.per_turn_isl`, `analyzer.per_turn_osl`,
timeline scalars. Parameterized emitted names (`turn3_isl`) are not stored per name;
the producer holds the base id (`analyzer.per_turn_isl`) and looks *that* up for
unit/header/direction, composing the concrete label itself — exactly the shape of
today's `format!("turn{ti}_isl")` code. `resolve(name)` is a convenience that maps a
concrete name to its base via a small registered prefix rule; no per-row string
parsing is required on any hot path. `export/dataset_analysis.rs` replaces its
string literals with base-def lookups.

### Typed units for SLA

Unit-scale mismatch (comparing a ms threshold against a ns value) is made
unrepresentable with lightweight hand-rolled newtypes `Native(f64)` /
`Display(f64)` — not `uom`, whose dimensional-analysis machinery is overkill and
costs compile time and call-site ceremony here. The newtypes implement only
dimensionally valid arithmetic (add/sub within a scale; scale by bare scalars) and
`From` only for the safe direction; there is **deliberately no cross-scale
arithmetic**, so `native + display` does not compile. `Unit::convert_value` is not
`const`, so display→native conversion stays a config-time step — which is already
how `SloThreshold::from_display` resolves a threshold once at construction. The
`Definition` stores unit *enums* only; numbers never get converted at definition
time.

### Definition-driven rendering and docs

Every sink reads presentation off the looked-up `Definition`:
`Definition::format_value` renders values, and `header`/`short_header`/
`effective_display_unit`/`display_order`/`group` drive table and column layout. An
`aiperf metrics list` / `describe <id>` command dumps the registry, and a generated
markdown reference table is emitted from it, so docs are produced from the registry
and cannot drift.

### Refactor of `MetricSpec`

`MetricSpec` embeds a `Definition` and keeps only computation:

```rust
pub struct MetricSpec {
    pub tag: MetricTag,
    pub def: Definition,          // def.id == the metric's namespaced id
    pub flags: MetricFlags,       // minus LARGER_IS_BETTER (now def.larger_is_better)
    pub required: &'static [MetricTag],
    pub kind: MetricType,
    pub aggregation: Option<AggregationKind>,
}
```

- `configure_catalog_metadata` (the runtime post-pass that mutates `short_header`,
  `value_type`, `plot_direction`) is folded into the `spec!`/const literals or a
  `const fn`-of-tag, so the catalog becomes `const`/`static` rather than a mutated
  `LazyLock<Vec>`.
- The 58 catalog rows that build flags with the `A | B` **operator** are rewritten
  to `A.union(B)` (bitflags 2.x `union` is const; the `|` operator overload is not
  callable in `const`). Any flag test needed *inside* a `const fn` compares raw bits
  (`(bits & OTHER) == OTHER`) rather than relying on `.contains` being const. No
  zero-bit flag is defined.
- `plot_direction` is deleted; `def.larger_is_better` is the single source of
  direction for both SLA and plotting.
- The ~50 consumer sites that read `spec.header`/`.unit`/`.display_unit`/
  `.plot_direction`/`.console_group` (concentrated in `metrics_core`, thin exporter
  tail) are kept stable with delegating accessors on `MetricSpec`
  (`spec.header()` → `spec.def.header`), bounding the diff.

### Extensibility seam (server / GPU)

The static core stays `const`/`phf`. Whole future families register through a
`DefinitionSource` trait yielding `&'static Definition`s, collected at bootstrap into
a runtime `OnceLock`/`LazyLock` view — reusing the existing
`AIPerfRegistry`/`AIPerfExtension` transactional pattern (duplicate ids rejected,
fail-closed) rather than a parallel mechanism, and preferring a trait-object source
list over a global mutable registry. Truly dynamic server metrics (arbitrary
Prometheus name + labels) resolve through a `server.*`-namespaced escape hatch, kept
distinct from the closed static families.

### Hot-path boundary (invariant)

`definition()`/`resolve()` return `&'static` and are **config/render-time only**;
the per-record and per-token paths never call them. Any field the hot path needs
(e.g. `larger_is_better`) is pre-resolved onto the relevant config struct once, as
`SloThreshold` already does. This is documented on the lookup functions and holds by
construction — the audit confirmed zero metric-catalog lookups in the transport,
dispatch, scheduler, `scheduled`, `request_rate`, `phase_runtime`, graph, or engine
paths.

### Non-goals

- No change to the computation graph, `kind`, `aggregation`, or accumulation.
- No wire/artifact format change: definitions carry the same `header`/`unit` values,
  so console, CSV, and JSON output stay byte-compatible.
- Server and GPU families are not merged now; only the namespaced seam is
  established.

### Verification

- **Compile-time completeness** — a new `MetricTag` without a `Definition` fails to
  build (exhaustive match, no wildcard).
- **Lookup** — `definition("aiperf.ttft")` and an analyzer base id resolve to the
  expected `Definition`; `resolve("turn3_isl")` resolves to `analyzer.per_turn_isl`;
  an alias resolves to its canonical def.
- **Id-set snapshot** — `insta` JSON snapshot of the sorted public id set, CI locked
  with `INSTA_UPDATE=no`.
- **SLA direction** — `passes_threshold` holds in both directions; `Native`/`Display`
  mixing fails to compile (trybuild or a documented type-level check).
- **Unit round-trip** — display→native→display is lossless within tolerance.
- **Rendering** — `format_value` output is identical across sinks; an existing
  metrics table and CSV render byte-identically before and after the refactor.

## Source anchors

- `runtime/src/metrics_core/catalog.rs` — `MetricTag` (fieldless, contiguous,
  `COUNT`/`index`), `MetricSpec`, `spec!`, `CATALOG`, `configure_catalog_metadata`,
  `spec_for`, `validate_catalog`; the enum/spec split and const-match lookup land
  here.
- `runtime/src/metrics_core/definition.rs` — new module: `Definition`,
  `DefinitionId`/namespacing, `DefinitionGroup`, `Native`/`Display`, the `const fn`
  metric-definition match, the `phf` id map, `definition`/`resolve`; re-exported as
  `crate::definitions`.
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

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `rust-port-flow` — an interactive, zoomable request-lifecycle deck for the AIPerf Rust port

**Date:** 2026-07-24
**App:** `apps/aiperf-flow` (in `/home/anthony/nvidia/projects/aiperf/ajc/rust`)
**Skill:** `.agents/skills/aiperf-flow-diagrams/SKILL.md` (authoritative authoring rules)
**Status:** design approved (brainstorming) — pending implementation plan

## Purpose

Build a **new, from-scratch** explainer deck, `apps/aiperf-flow/src/decks/rust-port-flow/`,
that tells the story of **one benchmark request's life** through the AIPerf Rust port —
starting from a single big-picture map and letting the reader **drill down, zoom, and play**
through each subsystem. It must be **more interactive than any existing deck** (which are
limited to `PageTabs` swapping static diagrams): semantic-zoom drill-down, an animated
request you can step/play through the pipeline, and live seam toggles.

The deck deliberately does **not** repeat what other decks do. The existing
`rust-aiperf-architecture` deck is organized *by subsystem tab*; the existing subsystem decks
(`aiperf-metrics-accumulator`, `dynosim-offline-flow`, `cellular-architecture`,
`graph-subsystem-overview`, `mock-server-architecture`, …) each cover one area with the
tabbed/static pattern. `rust-port-flow` is organized as **one continuous request-lifecycle
canvas** with a novel interaction model, and it does not reuse the `PageTabs`-swap approach.
It may *link out* to those decks as "go deeper" pointers but does not duplicate their content.

## Narrative spine (the 9 stages, in order)

The reader's mental model, which is the fixed order of the pipeline:

0. **Big Picture** — the whole lifecycle as one connected map (the entry level).
1. **Runtime & self-exec** — `aiperf-cli` → Config v2 → protocol-v2 `EnvelopeV2` stdio →
   re-exec of the same binary in `--execute` mode (the composition root). Introduce the
   **three orthogonal seams** (Time / Transport / Workload) as three colored axes.
2. **Dataset loading** — loaders → `SegmentStore` (six disjoint BLAKE3 content domains,
   prefix-folded hashing so shared prefixes dedup) → dense integer `Handle`s → `Turn`/body
   freeze.
3. **Sharing the dataset with workers** — the frozen `SegmentStore`: **bytes live exactly
   once**, conversations/turns carry `Handle`s not bytes (zero-copy sharing across worker
   threads). Explicitly distinguish the `content_server` as a *separate run-owned media
   delivery sidecar* — it is **not** how dataset text is shared.
4. **Workers synchronize & connect** — coordinator → `run_sharded_scheduled` spawns `W`
   self-contained sub-cell OS threads (thread-per-core, each its own `current_thread`
   runtime + `LocalSet`), a shared `RealClockAnchor` monotonic origin, `GlobalAdmission`
   shared per-cell gate, `merge_shards` finalize (sort, not renumber). `workers == 1` is the
   byte-unchanged single-coordinator path.
5. **The Clock seam (virtual vs real)** — `Clock` trait; `RealClock` (wall time) vs
   `SimClock` (integer-nanosecond virtual time, deterministic `(at_ns, seq_no)` ordering);
   `Clock::is_virtual()` selects real-reactor vs simulation driver. All transport time access
   routes through `Clock` — never `Instant::now`/`SystemTime::now`/raw `tokio::time`.
6. **The Transport seam** — a transport implements exactly **two traits** (`WorkerSink` +
   `ExecutionSinkBuilder`); everything else is shared. The four targets: **HTTP**
   (`TransportSink`, hyper, streaming), **gRPC** (`GrpcTransportSink`, Tonic, non-streaming),
   **dry-run**, **dynosim** (offline co-sim, `SteppableEngine`).
7. **The request hot-path** — `ScheduledRuntime`/`Workload` (`RequestRateWorkload` etc.) →
   `SlotPool` + `StopChecker` admission → `Rc<dyn Dispatcher>` → the chosen sink → shared
   `transport::reduce::reduce_parsed_response` → shared `transport::measure`. TTFT = first
   token observation.
8. **Aggregation → final results** — worker-local `NativeMetricsObserver` accumulation →
   `metrics_core` NaN-sparse column store (exact ragged replay) → **exact folds vs sketch**:
   final reports stay **exact from records**; **t-digest** (`cellular::sketch::TDigest`)
   sketches drive mergeable heartbeats/cellular merge (percentiles + stddev become streaming
   estimates; counts/sums/extrema stay exact) → deterministic boundary merge → `NativeReporter`
   → `NativeReport` → `ExporterRegistry` (nine sinks) → runner emits `RunTerminalV2` with a
   `report_path`.

### Code-vs-spec corrections baked in
- Sketch percentiles use **t-digest** (`cellular::sketch::TDigest`), **not** DDSketch.
- `content_server` is a **media delivery sidecar**, not the dataset-sharing mechanism.
- The reporter type is `NativeReporter` producing `NativeReport` (not a bare `Reporter`).

## Interaction model (the point of this deck)

A single continuous canvas navigated by **zoom, not page-swap**. Three real zoom levels:

1. **Overview (level 0):** the 9-stage pipeline as connected React Flow nodes. Every stage
   box is clickable.
2. **Subsystem (level 1):** clicking a stage **expands it in place** — a `motion`
   `layout`/`layoutId` shared-element transition grows the clicked stage to fill the canvas
   and reveals its internal subgraph; sibling stages slide aside and dim. No route/page swap.
3. **Leaf (level 2), where it earns it:** inside an expanded stage, selected nodes drill once
   more — e.g. Transport → "HTTP" reveals hyper request/SSE-decode/`reduce` internals;
   Aggregation → "sketch" reveals the t-digest-merge vs exact-fold comparison.

Navigation: breadcrumb bar; click backdrop or `Esc` pops up a level; arrow keys move between
sibling stages at the current level. `prefers-reduced-motion` is respected (transitions
degrade to instant).

### The "play" layer — watch a request live
- A **Play / Step / Reset + scrubber** control (built on the app's `useStepSimulator`) sends
  an **animated request particle** through the pipeline: issued at the scheduler → gated at
  `SlotPool` admission → `Dispatcher` → chosen transport sink → server → SSE tokens stream
  back (TTFT highlight on first token) → `reduce`/`measure` fire → record lands in the
  worker-local accumulator → boundary merge → exporter. Each step highlights the active node
  and shows a "what's happening now" caption naming the **real types**.
- **Clock-mode toggle:** flip `RealClock` ↔ `SimClock`; the particle's timing visibly changes
  (wall-paced vs discrete virtual-time hops) — making the clock seam tangible.
- **Transport selector:** HTTP / gRPC / dry-run / dynosim re-routes the *same* animated request
  through a different sink, showing the two-trait seam swap without changing anything upstream.

## Components — build the interaction primitives as SHARED, reusable app components

The novel interaction mechanics are built as **general, reusable components promoted into the
app's shared library** (a new `src/interactive/` module — parallel to `src/nodes`, `src/edges`,
`src/shell`), designed to be domain-agnostic so **any future deck can reuse them**. The
`rust-port-flow` deck is then a thin composition of these shared primitives plus its own
content. Shared primitives are additive-only — they must not change or break existing shared
components (`src/nodes`/`src/edges`/prose/layout) that other decks depend on.

### New shared primitives (`src/interactive/`)
- `ZoomStage` — domain-agnostic semantic-zoom container: manages level state (parametric depth,
  not hard-coded to 3), the active node id, breadcrumb, `motion` `layout`/`layoutId`
  shared-element expand/collapse, backdrop/`Esc`-to-pop, arrow-key sibling nav,
  `prefers-reduced-motion` fallback. Driven by a generic `ZoomTree<T>` data shape (node → its
  subgraph), so it is not AIPerf-specific.
- `PipelineCanvas` — reusable canvas wrapper that renders a level's nodes/edges and owns its
  own `ReactFlowProvider` per `<ReactFlow>` (per the skill's one-provider-per-instance trap).
- `RequestParticle` / `useFlowPlayer` — reusable animated-token-along-a-path primitive + a
  play/step/scrub hook built on `useStepSimulator`, generic over a typed `FlowStep[]` (active
  node id + caption + optional timing), so any deck can animate "a thing moving through a
  graph".
- `SeamToggle` — reusable segmented control for swapping a diagram between named variants
  (built on shared `src/prose/Pill.tsx` — no new chip/badge type), generic over a variant enum;
  used here for Clock-mode and Transport-selector but reusable anywhere.

Each new shared primitive gets its own tests and is documented so the `aiperf-flow-diagrams`
skill's component vocabulary can later reference it. (Adding a short entry to the SKILL.md
vocabulary list for the new `src/interactive/` primitives is part of this work.)

### Deck-local (`src/decks/rust-port-flow/`)
- Per-stage detail modules (one file per stage) supplying that stage's nodes/edges + `Callout`
  cards + `EvidenceRow` source anchors, and each stage's `ZoomTree`/`FlowStep` data.
- `RustPortFlowDeck.tsx` — the deck shell composing the shared `ZoomStage` + `PipelineCanvas` +
  `RequestParticle`/`useFlowPlayer` + `SeamToggle` with the AIPerf pipeline content; registered
  in the deck registry and on Home.

Reused as-is (not rebuilt): `useStepSimulator`, `useReveal`, `Callout`, `Grid`/`Row`/`Stack`,
`Pill`, `Eyebrow`, `Legend`/`Swatch`, `nodeTypes`/`edgeTypes`, `EvidenceRow`, and the token
helpers in `theme/tokens.ts`. All colors from token role helpers; all dynamic class names via
static lookup tables (Tailwind-JIT rule); any hand-drawn SVG uses `categoryFillClassName`/
`categoryStrokeClassName`.

## Source anchors (verified against real code, `rust/…`)

| Concept | Type / fn | File:line |
|---|---|---|
| Clock seam | `trait Clock` | `runtime/src/clock/clock.rs:12` |
| Real clock | `struct RealClock`, `RealClockAnchor` | `runtime/src/clock/real_clock.rs:52`, `:27` |
| Virtual clock | `struct SimClock` | `runtime/src/clock/sim_clock.rs:48` |
| Worker fan-out | `fn run_sharded_scheduled` | `runtime/src/engine/sharded_scheduled.rs:245` |
| Global admission | `struct GlobalAdmission` | `runtime/src/engine/execute/sharding.rs:25` |
| Transport trait 1 | `trait WorkerSink` | `runtime/src/engine/turn_execution.rs:74` |
| Transport trait 2 | `trait ExecutionSinkBuilder` | `runtime/src/engine/turn_execution.rs:136` |
| Dispatch | `trait Dispatcher` | `runtime/src/transport/core/dispatch.rs:332` |
| HTTP sink | `struct TransportSink` | `runtime/src/transport/http/sink.rs:164` |
| gRPC sink | `struct GrpcTransportSink` | `runtime/src/transport/grpc/sink.rs:102` |
| Admission pool | `struct SlotPool` | `runtime/src/timing/slots.rs:105` |
| Stop bounds | `struct StopChecker` | `runtime/src/timing/stop.rs:164` |
| Workload | `struct RequestRateWorkload` | `runtime/src/request_rate.rs:140` |
| Worker-local metrics | `struct NativeMetricsObserver` | `runtime/src/metrics.rs:203` |
| Sketch | `TDigest` (`cellular::sketch`) | `runtime/src/cellular/mod.rs:33` |
| Reporter | `struct NativeReporter`, `NativeReport` | `runtime/src/metrics_core/report.rs:1031`, `:1079` |
| Exporters | `trait Exporter`, `struct ExporterRegistry` | `runtime/src/export/mod.rs:208`, `:258` |

`SegmentStore`/`Handle`/`Turn`, `EnvelopeV2`/`RunTerminalV2`, `content_server`, and
`SteppableEngine` paths to be pinned to exact file:line during implementation (each stage's
`EvidenceRow` must cite real code, not the spec markdown).

## Verification

- **TDD:** one `.test.tsx` per stage/primitive asserting **real rendered content/behavior**
  (named types appear; clicking a stage changes zoom level; play advances the active step;
  clock/transport toggle changes state) — not "renders without crashing".
- Registered in the deck registry + Home; `cd apps/aiperf-flow && npm test && npm run build`
  both clean. If any dynamic class name was touched, grep `dist/assets/*.css` for the literal
  class strings after build (Tailwind-JIT check).
- Each page with multiple `<ReactFlow>` instances confirmed to give each its own
  `ReactFlowProvider`.
- **Live proof:** rendered screenshots of the big-picture map, one expanded subsystem, and the
  play-layer mid-request.

## Out of scope

- No modification of *existing* shared components or existing decks — the new interaction
  primitives are **additive** shared modules (`src/interactive/`); existing
  `src/nodes`/`src/edges`/prose/layout and `rust-aiperf-architecture` etc. stay unchanged.
- No new runtime/Rust code — this is a documentation/explainer deck only.

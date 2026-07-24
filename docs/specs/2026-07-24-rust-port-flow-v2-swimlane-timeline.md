<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `rust-port-flow` v2 — swimlane-timeline redesign

**Date:** 2026-07-24
**App:** `apps/aiperf-flow`
**Supersedes:** the node-graph *rendering* of `2026-07-24-rust-port-flow-deck-design.md`
(the 9-stage content, verified source anchors, and the shared `src/interactive/`
primitives `ZoomStage`/`useFlowPlayer`/`SeamToggle` are KEPT; only the React-Flow
box-and-arrow presentation is replaced).

## Why

The node-boxes-connected-by-arrows-on-a-canvas paradigm reads as a generic flowchart and
does not express what a benchmarking runtime is actually about: **a request moving through
time**. User feedback: the whole layout/concept — not just the arrow styling — needs to
change. Approved direction (synthesis of "timeline/swimlanes" + "nested system map" +
"single track/journey"): show the request's life as **one continuous track riding a time
axis through subsystem swimlanes, grouped inside nested seam frames**.

## The concept

A single **request line** runs left→right along a horizontal **time axis**. The time axis
*is* the Clock seam. The line threads top→bottom through horizontal **swimlanes**, one per
subsystem, in the order the request touches them. Lanes contain labeled **stage regions**
(a block per stage, positioned along the axis). Lane-segments are grouped inside translucent
**seam frames** (nested-composition view). Labeled lane-regions carry the structure; the
bright request line riding through them is the hero.

```
┌─ CLOCK SEAM ─── time →  (RealClock: wall-ms | SimClock: event ticks) ──────────────┐
│ dataset   │●freeze                                                                  │
│ scheduler │   ●issue ─●admit                                                        │
│ ┌WORKLOAD┐        ┌ TRANSPORT SEAM ───────────────────────┐                        │
│ transport │        ╰─●dispatch ═HTTP═▸ ●server ◂tokens (TTFT▲)                      │
│ aggregate │                             ╰──────────●reduce ─●measure ─●merge        │
│ export    │                                                        ●report ─●terminal│
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Lanes (top→bottom, request-touch order)
1. **Dataset** — pre-run origin; `SegmentStore` freeze (event: `freeze`).
2. **Scheduler / Workload** — `ScheduledRuntime`/`RequestRateWorkload` (events: `issue`, `admit` via `SlotPool`/`StopChecker`).
3. **Transport** — `Rc<dyn Dispatcher>` → the chosen sink (event: `dispatch`).
4. **Server** — the target (events: `send`, `TTFT` first-token, `stream`).
5. **Aggregate** — `NativeMetricsObserver` → `metrics_core` (events: `reduce`, `measure`, `merge`).
6. **Export** — `NativeReporter`→`NativeReport`→`ExporterRegistry` (events: `report`, `terminal` → `RunTerminalV2`).

### Seam frames (nested grouping)
- **Clock** = the whole time axis (its unit/scale is clock-mode dependent).
- **Workload** = the scheduler admission segment.
- **Transport** = the dispatch → server → reduce segment.

## Interaction (reuses existing shared primitives)

- **Drill-down** via the existing `ZoomStage`: click a stage region → it zooms into that
  subsystem's own detail (a mini-timeline or a panel with the stage's `Callout` cards +
  `EvidenceRow` — the v1 content, re-presented as lane detail). Breadcrumb + `Esc`/backdrop
  pop; arrow keys move between sibling stages. Parametric depth (overview → lane detail →
  leaf, e.g. Transport → HTTP send internals; Aggregate → t-digest-merge vs exact-fold).
- **Play** via the existing `useFlowPlayer`: the request token rides the line along the
  x-axis; the active stage region highlights; the existing caption panel names the real type.
- **Clock toggle** (`SeamToggle`) rescales the **x-axis**: `RealClock` spaces events by
  wall-milliseconds (realistic latency offsets); `SimClock` collapses them to evenly-spaced
  **virtual event ticks**. Axis labels/units switch — virtual time becomes *visible*.
- **Transport toggle** (`SeamToggle`) reroutes the transport-lane segment through
  HTTP / gRPC / dry-run / dynosim (dry-run has no server round-trip; dynosim rides virtual
  time). The same request re-routes without changing anything upstream.

## New shared primitives (additive, reusable — `src/interactive/`)

A custom swimlane-timeline renderer, domain-agnostic (lanes/regions/events/path are generic
data — any future deck can use it). Replaces React-Flow *for this deck only*; other decks
keep `PipelineCanvas`/`nodeTypes` untouched.

- `TimelineTrack` — top-level SVG renderer: given `lanes`, per-lane `regions` (each a time
  span + label), `events` (points on the axis), `seamFrames` (labeled spans grouping lanes),
  and a `requestPath` (ordered event ids), it lays out the axis, lanes, regions, frames, and
  the weaving request polyline. Props include an `activeEventId` (from `useFlowPlayer`) and a
  `scale: "real" | "virtual"` (from the Clock toggle) controlling x-mapping.
- `TimeAxis`, `Lane`, `StageRegion`, `SeamFrame`, `RequestLine`, `EventMarker` — subcomponents.
- **SVG coloring (SKILL.md rule):** every hand-drawn `<rect>`/`<path>`/`<line>`/`<circle>`
  uses `categoryFillClassName`/`categoryStrokeClassName` (or `inkClassName` + an explicit
  `stroke/fill="currentColor"` for role colors) — never `bg-*`/`border-*` on SVG.
- Reused unchanged: `ZoomStage`, `useFlowPlayer`, `SeamToggle`, all prose/layout primitives.

## Data model

Extend the existing `StageDef` (do not discard it) with lane/timeline metadata:
- `lane: LaneId` — which swimlane the stage sits in.
- `events: { id: string; label: string; atOrder: number; realOffsetMs?: number }[]` — the
  points the request line passes through (order + optional real-latency offset for the
  RealClock scale; SimClock uses evenly-spaced `atOrder`).
- The v1 `subgraph`/`leaves` become the stage's **drill detail** (rendered inside `ZoomStage`
  when zoomed into that lane), not the top-level view.
The v1 stage content (captions, verified `rust/…` source anchors, real type names) is carried
over verbatim — this is a re-presentation, not a re-research.

## Verification

- **TDD** per new primitive + the redesigned deck: tests assert real rendered content
  (lanes/regions/events present with real labels; clicking a region drills; play advances the
  active event; clock toggle changes the x-scale mapping; transport toggle reroutes).
- `cd apps/aiperf-flow && npm test && npm run build` both clean; Tailwind-JIT grep on `dist`
  for any new dynamic class; SVG uses `fill-*/stroke-*` (verified, not `bg-*`).
- **Live proof screenshots** (verified visually): the overview timeline; a drilled lane;
  play mid-request with the token on the axis; the RealClock↔SimClock x-axis rescale.

## Out of scope

- No changes to other decks or to `PipelineCanvas`/`nodeTypes`/`edgeTypes` (kept for decks
  that still use the node-graph). `FlowEdge` is untouched — this deck simply stops using it.
- No new runtime/Rust code. Content is a re-presentation of already-verified v1 material.

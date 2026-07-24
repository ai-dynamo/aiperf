<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `apps/aiperf-flow` — ELK auto-layout engine

**Date:** 2026-07-23
**App:** `apps/aiperf-flow`
**Status:** Approved design, pre-implementation.

## Why

Every React Flow diagram in `apps/aiperf-flow` positions its nodes with **hand-picked
`{x,y}` values** — the boustrophedon `overviewPosition` (`COL=300, ROW=380`) in
`rust-port-flow/stage.ts`, per-stage `position: {x,y}` guesses inside each
`stages/*.tsx` subgraph, `seedSubgraph`'s `{x:0,y:0}`, and the same pattern across
~25 other decks. But the nodes are **variable-sized**: `Card` is
`min-w-[180px] max-w-[260px]` and grows *taller* as its title/detail text wraps
(commit `a22ce2150b`). Authors are therefore guessing coordinates for boxes whose
real rendered footprint they cannot know.

Two failure modes result, matching the reported complaint ("boxes too smooshed
together, arrows that make no sense"):

1. **Smooshing / overlap** — guessed spacing is tighter than a box's true (wrapped)
   height/width, so boxes collide.
2. **Nonsensical arrows** — edges connect a node's right-side `source` handle to the
   next node's left-side `target` handle. When a downstream node's guessed position
   lands left of or above its source, the edge doubles back across the canvas and
   reads as a tangle.

There is currently **no** auto-layout and **no** node-size measurement anywhere in the
app (`grep` for `dagre`/`elkjs`/`useNodesInitialized`/`.measured` → zero hits). The fix
is to stop authoring positions by hand and instead **compute** them from graph
structure and *measured* node sizes, via the Eclipse Layout Kernel (ELK, `elkjs`).

## Approach (decided)

- **Algorithm:** ELK (`elkjs`), layered algorithm with orthogonal edge routing —
  chosen over dagre (weaker routing, unmaintained) and a custom engine (we'd own the
  hard edge-routing math). Best fit for the horizontal request-lifecycle shape.
- **Scope:** an **app-wide shared seam** any diagram opts into. Non-breaking: default
  off, so un-migrated decks keep their manual positions until converted.
- **Default direction:** `RIGHT` (left→right), matching the request-lifecycle decks;
  `DOWN` available per-diagram.
- **Rollout:** engine + seam + full `rust-port-flow` adoption first (proves it), then
  every remaining deck in follow-up waves (one commit per deck).

## The core mechanic — measure → layout → apply

Only React Flow knows a node's rendered size, and only after a first paint. The engine
runs this cycle inside a `ReactFlowProvider`:

1. Nodes render once at placeholder positions; the canvas is held at `opacity-0`.
2. `useNodesInitialized()` fires once React Flow has populated
   `node.measured.{width,height}` for every node.
3. The engine builds an ELK graph from those measured sizes + the edge list and calls
   `elk.layout()` (async; deterministic given identical input + sizes).
4. ELK's returned `x/y` are applied back onto the nodes, the canvas fades in, and
   `fitView()` frames the result.

Because ELK reserves each box's true measured footprint and routes edges layered in the
flow direction with orthogonal bends, boxes cannot overlap and arrows cannot double
back — the two failure modes above are eliminated structurally, not tuned away.

## Module: `src/layout/graph/`

- **`elkEngine.ts`** — pure wrapper `layoutGraph(nodes, edges, opts): Promise<Node[]>`.
  Maps React Flow nodes/edges ↔ the ELK graph JSON, sets `elk.direction`, node/layer
  spacing, and optional **swimlane partitioning** (ELK `partitioning.activate` with a
  per-node `partition`/`layer` from an `opts.laneOf(node)` callback) for lane-aligned
  diagrams. No React imports — unit-testable in isolation.
- **`useElkLayout.ts`** — `useElkLayout(nodes, edges, opts)`, the seam hook. Owns the
  `useNodesInitialized` gate, the async layout call, applying results, a `laidOut`
  boolean for the fade-in, and `fitView()` on completion. Must run inside a
  `ReactFlowProvider` (uses React Flow hooks). Re-runs when the node/edge identity or
  `opts` change.
- **`AutoLayoutFlow.tsx`** — a drop-in `<ReactFlow>` wrapper (its own
  `ReactFlowProvider`, per the one-provider-per-instance trap) that runs the hook
  internally. This is the migration target for the ~25 decks that embed a raw
  `<ReactFlow>` rather than going through `PipelineCanvas`.

### `ElkOptions`

```ts
interface ElkOptions {
  direction?: "RIGHT" | "DOWN";        // default "RIGHT"
  nodeSpacing?: number;                // within a layer
  layerSpacing?: number;               // between layers
  laneOf?: (node: Node) => string;     // opt-in swimlane partitioning
}
```

## Integration points

- **`PipelineCanvas`** gains `layout?: ElkOptions | "off"` (default `"off"`). When set,
  an inner component (child of the existing provider) runs `useElkLayout`. Because
  `ZoomStage` renders through `PipelineCanvas`, every `ZoomStage`-based deck adopts by
  flipping this one prop. `"off"` preserves today's manual-position behavior exactly.
- **Raw-`<ReactFlow>` decks** migrate by swapping their inline `<ReactFlow>` for
  `<AutoLayoutFlow>`, deck by deck.
- **`rust-port-flow` (flagship, this work):** route the stage level-1 subgraphs and the
  `leaves` React Flow diagrams — the ones that smoosh — through the engine, and **delete**
  the hand-computed coordinates: `overviewPosition`/`COL`/`ROW`, the per-stage
  `position` fields in `stages/*.tsx`, and `seedSubgraph`'s coords. Authors henceforth
  declare only graph structure (nodes + edges); the engine places them. Placeholder
  `position: {x:0, y:0}` remains on node objects only to satisfy the React Flow `Node`
  type — never read.

### Scope boundary

This targets **React Flow box-and-arrow diagrams**. `rust-port-flow`'s *overview* is a
custom SVG `TimelineTrack` (swimlane + request line), a separate layout system **not in
scope here**. If its lanes also read as cramped, that is a distinct follow-up.

## Test-env fallback (required)

jsdom reports every element's size as `0`, so `useNodesInitialized` never yields real
measurements and ELK cannot lay out under `npm test`. The engine detects
absent/zero measurements and falls back to a deterministic size estimate (a fixed
per-node width/height) so tests still render and can assert on structure and on
*positions having been assigned*. Real placement quality is verified in `npm run build`
plus a visual pass — unit tests alone do not prove the layout looks right, mirroring the
skill's Tailwind-JIT caveat.

Determinism: given the same nodes/edges/sizes and `opts`, ELK returns identical
coordinates, so snapshot-style position assertions are stable.

## Testing

- `elkEngine.test.ts` — pure `layoutGraph`: a 3-node chain lays out left→right with
  non-overlapping bounding boxes and monotonically increasing `x`; swimlane
  partitioning groups nodes by `laneOf`.
- `useElkLayout.test.tsx` — the fallback path assigns positions to all nodes under
  jsdom; `laidOut` flips true; re-runs when inputs change.
- `rust-port-flow` deck tests continue to assert on rendered stage/leaf **content**
  (not coordinates), so deleting the manual positions cannot regress them.
- Pre-delivery: `cd apps/aiperf-flow && npm test && npm run build` both clean.

## Rollout waves

1. **This plan:** `src/layout/graph/` engine + `PipelineCanvas`/`AutoLayoutFlow` seam +
   full `rust-port-flow` adoption (delete its manual positions).
2. **Follow-up waves:** convert the remaining ~25 decks, one commit each, flipping the
   `layout` prop or swapping to `AutoLayoutFlow`. Each keeps working untouched until its
   wave. A short migration note is added to the `aiperf-flow-diagrams` skill so future
   decks default to the engine instead of hand-picked positions.

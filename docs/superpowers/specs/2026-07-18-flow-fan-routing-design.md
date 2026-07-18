<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow Fan Routing Design

**Date:** 2026-07-18
**Status:** Approved
**Scope:** First-class fan topology and complete route correction across all nine explainer decks

## Goal

Make every explainer route communicate its real topology. Flow authors must be
able to express fan-out and fan-in directly, while the compiler and renderer
produce shared trunks, connected junctions, correctly placed arrowheads, and
topology-aware traveling balls. Migrate all nine decks away from ambiguous or
incorrect hand-authored lines, arrows, and motion paths.

## Locked decisions

| Decision | Choice |
|---|---|
| Fan representation | First-class topology in package IR and `SceneRenderer` |
| Deck coverage | All nine files under `apps/explainers/decks-flow/`, including `tstar-warmup.flow` |
| Arrowheads | Only at semantic destinations |
| Fan-out balls | One incoming traveler duplicates onto all outgoing branches |
| Fan-in balls | Incoming travelers converge, then one traveler leaves the junction |
| Default route | Orthogonal, perimeter-anchored, shared trunk rendered once |
| Verification | No new tests; build, package, IR verifier, and Playwright verifier gates |
| Generated packages | Rebuild from corrected `.flow` sources; never edit as source of truth |

## Baseline inventory (2026-07-18 audit)

Nine decks, 133 scene slides, ~18.5k lines of `.flow` source:

| Stroke / motion | Count | Notes |
|---|---:|---|
| `core.path` (manual `d:`) | 178 | Fan trees, curves, legacy chains |
| `core.connector` | 66 | Node-anchored straight edges |
| `core.line` | 47 | Absolute-coordinate segments |
| `core.route` | 32 | Node-anchored orthogonal auto-route |
| `motion.signal` | 132 | ~40 anchored, ~80 manual `d:` |
| `core.elbow` | 0 | Defined in schema, unused in decks |

Fan-out and fan-in today are **composition only** — hand-drawn branch paths plus
parallel `motion.signal` nodes (for example `slurm-velo.flow` slides 13–14,
`velo-deep-dive.flow` slide 9). There is no topology primitive.

Priority rewrite targets:

1. `slurm-velo.flow` — task fork (~332) and rank-0 fan-out/fan-in (~1717, ~1963)
2. `velo-deep-dive.flow` — star fan-in (~980)
3. `cellular-internals.flow` — 68 manual paths (largest legacy surface)
4. `rust-architecture.flow` / `cellular-algorithms.flow` — 47 absolute `core.line` edges

## Platform gaps to close with this work

| Gap | Location | Fix in this project |
|---|---|---|
| Motion ignores authored elbow/branch geometry | `SceneRenderer.tsx` `motionSignalPathData` | Fan + single-edge motion share resolved segment data |
| Verifier omits `core.route` / `core.elbow` / fans | `flow-verifier/geometry.mjs` | Extend `ARROW_CAPS`, elbow path resolve, fan cardinality |
| Verifier straight-lines elbows | `geometry.mjs` `arrowPathData` | Mirror `elbowPathData` from renderer |
| Play gate missing `tstar-warmup` | `flow-verifier/play.mjs` `DECK_ROUTES` | Add `/#/tstar-warmup` (9/9 decks) |
| Draw vs trace split on motion | `SceneRenderer.tsx` | Fan playback uses `trace`; `draw` reveals strokes only |

## Current problem

The decks mix low-level `core.line`, `core.path`, and `core.arrow` nodes with
anchored connectors, elbows, routes, and `motion.signal`. Branching diagrams
are assembled from independent segments. This permits overlapping trunks,
disconnected junctions, inconsistent arrowheads, routes that pass through
boxes, and traveling balls whose path differs from the painted route.

The renderer also treats node-anchored motion specially by replacing its route
with straight clipped segments. That behavior cannot preserve elbow geometry
or model a split/merge event. Correctness therefore requires a topology model,
not another deck-local coordinate pass.

## Authoring model

Add two package capabilities and matching native scene keywords:

- `core.fan-out` / `fan-out`
- `core.fan-in` / `fan-in`

A fan-out has exactly one source and at least two destinations:

```text
{
  id: "dispatch"
  capability: "core.fan-out"
  from: { nodeId: "controller", anchor: "e" }
  to: [
    { nodeId: "cell-0", anchor: "w" },
    { nodeId: "cell-1", anchor: "w" },
    { nodeId: "cell-n", anchor: "w" }
  ]
  axis: "x"
}
```

A fan-in has at least two sources and exactly one destination:

```text
{
  id: "merge"
  capability: "core.fan-in"
  from: [
    { nodeId: "worker-0", anchor: "e" },
    { nodeId: "worker-1", anchor: "e" }
  ]
  to: { nodeId: "report", anchor: "w" }
  axis: "x"
}
```

Both forms accept:

- `axis: "x" | "y"` to select the trunk direction.
- `junction: { x, y }` to override automatic junction placement.
- Existing stroke, dash, color, and arrow-tip style properties.
- Explicit endpoint anchors; soft or missing anchors resolve to facing
  perimeter edges.

Automatic junction placement uses a stable midpoint in the available corridor
between source and destination bounds. It must not move based on timeline
state. An authored junction wins when automatic placement cannot express the
intended layout.

## IR and compiler

Introduce a first-class fan node rather than desugaring to unrelated
connectors:

```text
FanNodeIr {
  kind: "fan"
  capability: "core.fan-out" | "core.fan-in"
  from: ConnectorEndpointIr | ConnectorEndpointIr[]
  to: ConnectorEndpointIr | ConnectorEndpointIr[]
  axis?: "x" | "y"
  junction?: PointIr
}
```

Schema validation enforces fan cardinality:

- fan-out: one `from`, two or more `to`
- fan-in: two or more `from`, one `to`

Language capture, native parsing, lowering, package validation, and public
exports preserve the topology as one node. The compiler does not emit
overlapping connector nodes for the shared trunk.

## Geometry and arrow contract

The renderer resolves fan geometry in world space, then rebases it into the
current group coordinate space. A resolved fan contains:

- one junction,
- one shared trunk,
- one branch per endpoint on the many side,
- one ordered trajectory per logical source-to-destination flow.

Segments that share endpoints are deduplicated before paint. Orthogonal bends
must be connected exactly; no visual gap or doubled stroke is allowed at the
junction.

Primitive direction defaults become explicit:

- `core.line`, `core.path`, `core.divider`: undirected by default.
- `core.arrow`, `core.connector`, `core.elbow`, `core.route`: directed.
- `core.fan-out`, `core.fan-in`: directed topology.
- `motion.signal`: a trajectory guide; it does not add an independent
  arrowhead unless explicitly requested.

For fan-out, arrowheads appear at every destination box and nowhere on the
incoming trunk or junction. For fan-in, one arrowhead appears at the single
destination box and nowhere on incoming branches or the junction. Arrow tips
use the existing marker resolution and clipping rules and stop outside box
fills.

## Ball and timeline contract

Painted geometry and motion use the same resolved segment data. The renderer
must not reconstruct a different straight-line path for fan motion.

Fan-out playback:

1. One ball travels from the source to the junction.
2. At the junction, it disappears.
3. One ball appears on each outgoing branch at the same frame.
4. Branch balls travel to their respective destinations.

Fan-in playback:

1. One ball travels from each source toward the junction.
2. Branch timing is normalized so all balls reach the junction together.
3. Incoming balls disappear at the junction.
4. One ball leaves the junction and travels to the destination.

`draw` reveals the topology without inventing extra route geometry. `trace`
drives the topology-aware balls. Existing single-edge `motion.signal` remains
valid for non-branching flows. Reduced-motion playback jumps to the final
static route and omits all traveling balls.

## Deck migration

Rewrite:

1. `segment-pools.flow`
2. `dynosim.flow`
3. `tstar-warmup.flow`
4. `velo-deep-dive.flow`
5. `slurm-velo.flow`
6. `rust-architecture.flow`
7. `rust-architecture-atlas.flow`
8. `cellular-internals.flow`
9. `cellular-algorithms.flow`

Classify every route-like node before replacement:

- divider or visual rule,
- freeform illustration path,
- directed point-to-point edge,
- fan-out,
- fan-in,
- motion-only trajectory.

Apply these migration rules:

- Replace semantic splits and merges with fan nodes.
- Replace directed axis-aligned low-level paths with anchored
  `core.route`, `core.elbow`, or `core.connector`.
- Keep `core.line` and `core.path` only for undirected rules or genuine
  freeform drawing.
- Make all exceptional arrow behavior explicit.
- Remove duplicate motion guides and obsolete static companion dots.
- Preserve slide narrative, node placement, theme, and timing unless route
  correctness requires a timing adjustment.

## Verification

### Verifier rules

Extend the IR verifier to reject:

- invalid fan cardinality,
- disconnected trunk/branch junctions,
- duplicate overlapping route segments,
- directed routes that pierce unrelated boxes,
- arrowheads on undirected primitives without an explicit override,
- missing arrowheads at semantic destinations,
- arrowheads at internal fan junctions,
- orphan motion signals or companion dots,
- ball trajectories that differ from painted paths.

### End-to-end gates

1. Build the schema, language, compiler, and explainer applications without
   adding or modifying tests.
2. Rebuild all nine deck packages.
3. Run the IR verifier with zero route errors.
4. Run the Playwright play/visual verifier across every deck.
5. Inspect representative fan-out and fan-in scenes at start, junction, and
   terminal frames, including reduced motion.

## Error handling

Compilation fails with the fan node id and source range when cardinality,
endpoint shape, or axis is invalid. Missing node references fail package
validation. Geometry that cannot produce a connected finite route is reported
by the verifier rather than silently falling back to `(0, 0)` or a straight
line.

## Non-goals

- General graph layout or automatic node placement.
- Obstacle-avoiding maze routing beyond the existing orthogonal corridor
  model.
- Bezier fan routes.
- Restyling slides or changing their narrative content.
- A second renderer or deck-specific fan implementation.

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Advanced Curved Connector Routing Design

**Date:** 2026-07-20
**Status:** Approved
**Scope:** Deterministic obstacle-aware curved routing for the explainers Flow SDK

## Goal

Extend `sdk.Edge(mode = "curve")` from a single cubic Bézier into a deterministic
diagram router. Curved edges must support every source-to-target combination of
the nine box anchors, avoid all intervening node geometry, preserve exact
endpoint attachment, produce stable routes across rerenders, and share their
resolved geometry with motion signals and verifiers.

The work is delivered in this priority order:

1. Obstacle-aware routing.
2. Adaptive tangents, same-side links, overlap handling, and self-loops.
3. Parallel-edge lanes, optional bundling, crossing penalties, and stable route
   selection.

Existing straight and elbow routing remains unchanged. Advanced routing is
opt-in through curve mode.

## Locked decisions

- Use a visibility graph with deterministic A* search.
- Treat every positive-area, non-connector scene node as an obstacle.
- Exclude the source node and target node from the obstacle set.
- Inflate obstacles by configurable clearance before routing.
- Route automatically by default and provide optional author overrides.
- Preserve authored `path` and `points` precedence over automatic routing.
- Use all nine anchors: `center`, `n`, `s`, `e`, `w`, `ne`, `nw`, `se`, and
  `sw`, including existing aliases.
- Feed rendered edges and motion signals from the same resolved path.
- Remove `connector-routing.test.ts`; retain verification through the existing
  build and geometry-verifier infrastructure.

## Public authoring model

`sdk.Edge(mode = "curve")` enables the advanced router:

```flow
sdk.Edge(
  id = "request-to-worker",
  mode = "curve",
  from = { nodeId: "request", anchor: "se" },
  to = { nodeId: "worker", anchor: "nw" },
  style = {
    clearance: 12,
    curvature: 0.45,
    avoidObstacles: true,
    preferredSide: "auto",
    bundle: false,
    parallelGap: 8
  }
)
```

The style controls are:

- `clearance`: finite nonnegative obstacle padding. The default is 12 scene
  units.
- `curvature`: finite rounding strength clamped to `[0.05, 0.95]`. The default
  is `0.45`.
- `avoidObstacles`: boolean, default `true`. When false, the router uses only
  adaptive endpoint geometry.
- `preferredSide`: `"auto"`, `"n"`, `"s"`, `"e"`, or `"w"`. The default is
  `"auto"`.
- `bundle`: boolean, default `false`. Compatible edges may share a corridor
  while retaining separate endpoint branches.
- `parallelGap`: finite nonnegative spacing between parallel routes. The
  default is 8 scene units.

Invalid or unsupported style values fall back to their defaults. They do not
make scene compilation fail because style values remain an open compatibility
surface.

## Routing engine boundary

`connector-routing.ts` becomes a pure geometry module. It must not depend on
React, the DOM, animation state, or scene traversal.

The router accepts:

- source and target endpoint positions;
- source and target anchor names and outward normals;
- source and target node bounds when available;
- obstacle identifiers and world-space bounds;
- optional sibling-edge descriptors for lanes and bundling;
- normalized routing options.

The router returns:

- source and target escape points;
- an ordered waypoint list;
- rounded cubic segments;
- canonical SVG path data;
- route bounds;
- whether a fallback was used;
- obstacle and sibling-edge decisions needed by verification.

The output uses finite world-space coordinates. `SceneRenderer` rebases the
canonical path into the current layout origin only after routing.

## Obstacle selection

`SceneRenderer` derives obstacles from `SceneNodeIndex.worldGeometryById`.
Eligible obstacles have finite positive width and height. The router excludes:

- the connector itself;
- the source node;
- the target node;
- ancestor containers that enclose the source or target node;
- connector, line, path, route, elbow, fan, and motion-only nodes;
- zero-area geometry.

Groups and component containers remain obstacles when they have positive-area
geometry and do not contain either endpoint. Their descendants may also be
obstacles. Exact duplicate bounds collapse to the lexicographically smallest
node ID so they do not multiply graph nodes or costs.

Each remaining obstacle is inflated by `clearance`. Endpoint escape corridors
through the source and target bounds are explicitly permitted; no other route
segment may enter an inflated obstacle.

## Anchor and endpoint geometry

Each anchor resolves to an outward unit normal:

- cardinal anchors use axis-aligned normals;
- corner anchors use normalized diagonals;
- center anchors point toward the peer endpoint;
- unknown anchors retain the existing center fallback.

The source escape point extends outward from the source anchor. The target
escape point extends outward from the target anchor, with the final route
approaching opposite the target's outward normal. Escape length is at least the
configured clearance and grows when necessary to leave overlapping inflated
bounds.

Same-side links generate clockwise and counterclockwise candidates around the
relevant node. Self-links generate four loop candidates, one per cardinal side,
then honor `preferredSide` or choose the lowest-cost candidate. Overlapping
source and target boxes extend escape points until both are outside the overlap
before graph search begins.

## Visibility graph

Candidate vertices include:

- source and target anchors;
- source and target escape points;
- clearance-offset corners of every normalized obstacle;
- deterministic side-midpoint candidates when they shorten same-side routes;
- loop candidates for self-links;
- compatible corridor points supplied by sibling-edge routing.

Two vertices are connected when the segment between them has finite length and
does not enter any inflated obstacle. Boundary-tangent segments are allowed.
Graph construction uses stable sorting by coordinate, role, and obstacle ID.
No insertion-order-dependent iteration may affect the result.

## Deterministic A* scoring

The route score is a lexicographically stable weighted sum of:

- Euclidean segment length;
- bend count and turn angle;
- movement opposite an endpoint's required normal;
- crossings with already resolved sibling routes;
- deviation from `preferredSide`;
- congestion in an occupied parallel lane;
- reward for an explicitly enabled compatible bundle corridor.

All ties are broken by a canonical route key formed from rounded coordinates
and vertex roles. Given identical inputs, the router must emit byte-identical
SVG path data.

## Parallel lanes and bundling

Sibling edges are grouped by stable source ID, target ID, anchor pair, and route
mode. Parallel edges receive symmetric offsets around the canonical route in
stable edge-ID order using `parallelGap`.

When `bundle` is false, shared-corridor occupancy adds cost so independent
routes separate when practical. When `bundle` is true, compatible edges receive
a corridor reward and share the longest collision-free middle section. Their
source and target branches remain separate, and every resulting path is checked
against obstacles after offsets are applied.

Edge routing follows scene document order with edge ID as a tie-breaker.
Previously routed sibling paths are immutable inputs to later routes, preventing
rerender oscillation.

## Smoothing

The selected visibility path is simplified by removing duplicate and collinear
waypoints. Every remaining corner is rounded into cubic Bézier segments.
Rounding distance is bounded by:

- adjacent segment lengths;
- configured curvature;
- obstacle clearance;
- parallel-lane spacing.

The router samples or analytically bounds each rounded segment against inflated
obstacles. If smoothing introduces a collision, it reduces the local rounding
radius. If no positive radius is safe, that corner remains sharp. Endpoint
tangents always follow the authored anchor normals.

## Fallback behavior

The primary result must be collision-free. If the visibility graph has no path:

1. retry with bundling disabled;
2. retry with parallel offsets disabled;
3. retry with zero smoothing;
4. choose the deterministic candidate with the least total obstacle
   penetration.

The final step guarantees finite renderable output while setting
`usedFallback: true` and recording the penetrated obstacle IDs. The verifier
reports fallback use. Rendering remains available rather than silently dropping
the edge.

## Renderer and motion integration

`SceneRenderer` resolves source and target endpoints and gathers world-space
obstacles before invoking the router. It caches route results for the duration
of one scene-index/render pass using a stable key derived from geometry,
options, and sibling-edge state.

`arrowPathData` consumes the returned canonical SVG path. Motion signals use the
same route result, including obstacle detours, smoothing, parallel offsets, and
bundles. Arrowhead shortening occurs after routing and must preserve the final
approach tangent.

Straight connectors, authored paths, point polylines, elbows, and fan geometry
do not enter this router.

## Verifier integration

Delete `apps/explainers/src/core/diagram/connector-routing.test.ts`.

The browser-safe geometry verifier and Node flow verifier retain matching route
behavior. Verifier scenarios cover:

- all 81 source/target anchor pairs;
- an intervening obstacle;
- multiple obstacles requiring more than one turn;
- same-side routes;
- overlapping boxes;
- self-loops;
- parallel edge separation;
- enabled bundling;
- deterministic reruns;
- fallback reporting;
- finite SVG output and exact endpoint attachment.

The verifier checks route segments and rounded curves against inflated
obstacles. A normal curve route that penetrates an obstacle is an error.
Fallback routes are warnings containing edge and obstacle IDs.

Completion gates are:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npm run build
npm run flow-verifier:ir
npm run flow-verifier:extended
```

## Change surface

Primary implementation files:

- `apps/explainers/src/core/diagram/connector-routing.ts`
- `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- `apps/explainers/src/flow/sdk/generic/topology.ts`
- `apps/explainers/src/flow/dev-tools/verify-geometry.ts`
- `apps/explainers/scripts/flow-verifier/geometry.mjs`

The schema does not need a new connector node shape because route controls fit
the existing open style record. Documentation comments in the Flow IR and
embedded-scene authoring helpers remain synchronized with the supported API.

## Non-goals

- Changing straight or elbow route defaults.
- Automatically converting existing deck edges to curve mode.
- Routing fan trunks and branches through the new curve router.
- General polygon obstacles; obstacles remain axis-aligned scene bounds.
- Interactive route editing or author-visible control handles.
- Globally optimal simultaneous routing across every edge in a scene.

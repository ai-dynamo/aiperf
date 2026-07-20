<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Advanced Curved Connector Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic obstacle-aware, adaptive, lane-separated, optionally bundled curved connectors for every source/target anchor pair in the explainers Flow SDK.

**Architecture:** A pure TypeScript routing engine receives world-space endpoint geometry, rectangular obstacles, sibling routes, and normalized options. It builds a stable visibility graph, searches it with deterministic A*, rounds the selected polyline into collision-checked cubic segments, and returns canonical route metadata and SVG. `SceneRenderer` owns scene indexing and obstacle selection; rendered strokes, motion signals, and verifiers consume the same route semantics.

**Tech Stack:** TypeScript 5.8, React 19, SVG paths, existing Flow Scene IR, Node ESM verifier, Playwright full-deck verifier

## Global Constraints

- Advanced routing is opt-in through `sdk.Edge(mode = "curve")`.
- Existing straight, path, point-polyline, elbow, route, and fan behavior must not change.
- Authored `d`, `path`, and `points` retain precedence over automatic routing.
- Support `center`, `n`, `s`, `e`, `w`, `ne`, `nw`, `se`, and `sw`, plus existing aliases.
- Treat every finite positive-area non-connector scene node as an obstacle, except source, target, and endpoint ancestor containers.
- Default options are `clearance: 12`, `curvature: 0.45`, `avoidObstacles: true`, `preferredSide: "auto"`, `bundle: false`, and `parallelGap: 8`.
- Invalid open-style values fall back to defaults; they do not fail scene compilation.
- Given identical inputs, generated SVG path data must be byte-identical.
- Delete `apps/explainers/src/core/diagram/connector-routing.test.ts`; verifier scenarios replace that file's coverage.
- Add no dependencies.
- Do not create git commits unless the user explicitly requests them.

---

## File Map

- Create `apps/explainers/src/core/diagram/connector-routing-types.ts`: immutable public routing DTOs and normalized defaults.
- Create `apps/explainers/src/core/diagram/connector-routing-geometry.ts`: finite geometry, inflation, intersection, segment visibility, path simplification, and cubic collision helpers.
- Create `apps/explainers/src/core/diagram/connector-routing-search.ts`: stable visibility-graph construction and deterministic A*.
- Modify `apps/explainers/src/core/diagram/connector-routing.ts`: public facade, anchor normals, route candidate construction, smoothing, lanes, bundling, and fallback.
- Delete `apps/explainers/src/core/diagram/connector-routing.test.ts`.
- Modify `apps/explainers/src/core/diagram/SceneRenderer.tsx`: obstacle collection, endpoint bounds, scene-level sibling routing, path rebasing, shared motion path.
- Modify `apps/explainers/src/flow/sdk/generic/topology.ts`: document curve options and preserve normalized style controls.
- Modify `apps/explainers/src/flow/schema/ir.ts`: document curved route style vocabulary.
- Modify `apps/explainers/src/flow/language/embedded-scene.ts`: document advanced curve controls.
- Modify `apps/explainers/src/flow/dev-tools/verify-geometry.ts`: mirror route invocation and expose route metadata checks in browser tooling.
- Modify `apps/explainers/scripts/flow-verifier/geometry.mjs`: Node-compatible route mirror and collision checks.
- Modify `apps/explainers/scripts/flow-verifier/ir.mjs`: advanced routing scenario matrix and findings.
- Modify `apps/explainers/scripts/flow-verifier.mjs`: run the synthetic routing matrix once per verifier invocation.

---

### Task 1: Pure Obstacle-Aware Routing Core

**Files:**
- Create: `apps/explainers/src/core/diagram/connector-routing-types.ts`
- Create: `apps/explainers/src/core/diagram/connector-routing-geometry.ts`
- Create: `apps/explainers/src/core/diagram/connector-routing-search.ts`
- Modify: `apps/explainers/src/core/diagram/connector-routing.ts`
- Delete: `apps/explainers/src/core/diagram/connector-routing.test.ts`

**Interfaces:**
- Produces: `routeCurve(input: CurveRouteInput): CurveRouteResult`
- Produces: `normalizeCurveRouteOptions(style): CurveRouteOptions`
- Produces: geometry helpers used by renderer and browser verifier.
- Consumes: no renderer, DOM, or React state.

- [ ] **Step 1: Remove the standalone Vitest file**

Delete:

```text
apps/explainers/src/core/diagram/connector-routing.test.ts
```

Do not add a replacement `*.test.ts` file.

- [ ] **Step 2: Add immutable route DTOs and defaults**

Create `connector-routing-types.ts` with these exported contracts:

```ts
export type Point2 = Readonly<{ x: number; y: number }>;
export type Bounds2 = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;
export type RouteObstacle = Readonly<{ id: string; bounds: Bounds2 }>;
export type PreferredSide = "auto" | "n" | "s" | "e" | "w";
export type CurveRouteOptions = Readonly<{
  clearance: number;
  curvature: number;
  avoidObstacles: boolean;
  preferredSide: PreferredSide;
  bundle: boolean;
  parallelGap: number;
}>;
export type RoutedSibling = Readonly<{
  id: string;
  sourceId?: string;
  targetId?: string;
  fromAnchor?: string;
  toAnchor?: string;
  waypoints: readonly Point2[];
  segments: readonly CubicSegment[];
}>;
export type CurveRouteInput = Readonly<{
  edgeId: string;
  start: Point2;
  end: Point2;
  fromAnchor?: string;
  toAnchor?: string;
  sourceId?: string;
  targetId?: string;
  sourceBounds?: Bounds2;
  targetBounds?: Bounds2;
  obstacles: readonly RouteObstacle[];
  siblings: readonly RoutedSibling[];
  options: CurveRouteOptions;
}>;
export type CubicSegment = Readonly<{
  start: Point2;
  control1: Point2;
  control2: Point2;
  end: Point2;
}>;
export type CurveRouteResult = Readonly<{
  d: string;
  waypoints: readonly Point2[];
  segments: readonly CubicSegment[];
  bounds: Bounds2;
  usedFallback: boolean;
  penetratedObstacleIds: readonly string[];
}>;
export const DEFAULT_CURVE_ROUTE_OPTIONS: CurveRouteOptions = Object.freeze({
  clearance: 12,
  curvature: 0.45,
  avoidObstacles: true,
  preferredSide: "auto",
  bundle: false,
  parallelGap: 8,
});
```

- [ ] **Step 3: Implement finite geometry and visibility helpers**

Create `connector-routing-geometry.ts`. Export:

```ts
export function inflateBounds(bounds: Bounds2, amount: number): Bounds2;
export function pointInBounds(point: Point2, bounds: Bounds2, strict?: boolean): boolean;
export function segmentIntersectsBounds(
  start: Point2,
  end: Point2,
  bounds: Bounds2,
  allowBoundary?: boolean,
): boolean;
export function segmentIsVisible(
  start: Point2,
  end: Point2,
  obstacles: readonly RouteObstacle[],
): boolean;
export function simplifyWaypoints(points: readonly Point2[]): readonly Point2[];
export function cubicPoint(segment: CubicSegment, t: number): Point2;
export function cubicPenetrations(
  segment: CubicSegment,
  obstacles: readonly RouteObstacle[],
): readonly string[];
export function routeBounds(points: readonly Point2[]): Bounds2;
export function canonicalPointKey(point: Point2): string;
```

Use Liang–Barsky segment/rectangle clipping for visibility. Treat exact
boundary tangency as visible, but reject any interval whose midpoint lies in
the strict rectangle interior. Sample cubic segments at a deterministic 33
positions (`t = 0/32` through `32/32`) for verifier-compatible collision
checking. Round canonical coordinates to three decimals.

- [ ] **Step 4: Implement stable visibility graph construction**

Create `connector-routing-search.ts` with:

```ts
export type RouteVertex = Readonly<{
  id: string;
  point: Point2;
  role: "start" | "start-escape" | "corner" | "side" | "end-escape" | "end";
}>;
export type SearchOptions = Readonly<{
  startNormal: Point2;
  endNormal: Point2;
  preferredSide: PreferredSide;
  siblingSegments: readonly Readonly<{ start: Point2; end: Point2 }>[];
  bundle: boolean;
}>;
export function buildVisibilityVertices(
  start: Point2,
  startEscape: Point2,
  endEscape: Point2,
  end: Point2,
  obstacles: readonly RouteObstacle[],
): readonly RouteVertex[];
export function findBestRoute(
  vertices: readonly RouteVertex[],
  obstacles: readonly RouteObstacle[],
  options: SearchOptions,
): readonly Point2[] | undefined;
```

Stable-sort vertices by `x`, `y`, role, then ID. Connect every visible pair.
The A* score is:

```ts
const score =
  length +
  bendRadians * 18 +
  reverseEndpointDistance * 12 +
  crossingCount * 80 +
  preferredSidePenalty * 24 +
  corridorCost;
```

Use `canonicalPointKey` plus vertex IDs as the final tie-break key. Never rely
on `Map` insertion order to choose between equal-cost candidates.

- [ ] **Step 5: Replace the single-cubic facade with obstacle-aware routing**

In `connector-routing.ts`, retain `isCurveRoute`, `isElbowRoute`, and
`elbowPathData`. Import the new DTOs and implement:

```ts
export function normalizeCurveRouteOptions(
  style: Readonly<Record<string, unknown>> | undefined,
): CurveRouteOptions {
  const finite = (value: unknown, fallback: number, min: number, max: number) =>
    typeof value === "number" && Number.isFinite(value)
      ? Math.min(Math.max(value, min), max)
      : fallback;
  const preferredSide =
    style?.preferredSide === "n" ||
    style?.preferredSide === "s" ||
    style?.preferredSide === "e" ||
    style?.preferredSide === "w"
      ? style.preferredSide
      : "auto";
  return {
    clearance: finite(style?.clearance, 12, 0, 200),
    curvature: finite(style?.curvature, 0.45, 0.05, 0.95),
    avoidObstacles:
      typeof style?.avoidObstacles === "boolean" ? style.avoidObstacles : true,
    preferredSide,
    bundle: typeof style?.bundle === "boolean" ? style.bundle : false,
    parallelGap: finite(style?.parallelGap, 8, 0, 100),
  };
}
```

`routeCurve` must:

1. normalize and stable-sort inflated obstacles;
2. calculate anchor normals and escape points;
3. call `findBestRoute`;
4. simplify the returned polyline;
5. smooth corners with cubic segments, reducing local radius until
   `cubicPenetrations` is empty;
6. retry without smoothing when needed;
7. return the least-penetrating finite fallback if search fails;
8. format `d` with three-decimal canonical numbers.

Keep `curvePathData(start, end, fromAnchor, toAnchor, curvature)` as a
compatibility wrapper that calls `routeCurve` with no obstacles or siblings.

- [ ] **Step 6: Type-check the pure core**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npx tsc --noEmit
```

Expected: exit 0 with no TypeScript diagnostics.

---

### Task 2: SceneRenderer Obstacle Collection and Shared Paths

**Files:**
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`

**Interfaces:**
- Consumes: `routeCurve`, `normalizeCurveRouteOptions`, `CurveRouteResult`.
- Produces: one world-space resolved route per curved edge for both stroke and motion consumers.

- [ ] **Step 1: Add ancestor and route metadata to the scene index**

Extend `SceneNodeIndex`:

```ts
type SceneNodeIndex = Readonly<{
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
  ancestorIdsById: ReadonlyMap<string, readonly string[]>;
}>;
```

Pass an immutable ancestor list through `indexSceneNodes.visit`. Record it
before recursing into children.

- [ ] **Step 2: Implement obstacle eligibility and collection**

Add:

```ts
function isRouteObstacleNode(node: SceneNodeLike): boolean {
  const capability = capabilityOf(node);
  return !ARROW_CAPABILITIES.has(capability) &&
    capability !== "motion.signal" &&
    node.kind !== "connector" &&
    node.kind !== "fan";
}

function curveObstacles(
  node: SceneNodeLike,
  from: ScenePointLike,
  to: ScenePointLike,
  index: SceneNodeIndex,
  options: CurveRouteOptions,
): readonly RouteObstacle[] {
  if (!options.avoidObstacles) return [];
  const endpointIds = new Set(
    [from.nodeId, to.nodeId].filter((id): id is string => typeof id === "string"),
  );
  for (const id of [...endpointIds]) {
    for (const ancestor of index.ancestorIdsById.get(id) ?? []) endpointIds.add(ancestor);
  }
  return [...index.worldGeometryById]
    .filter(([id, geometry]) => {
      const candidate = index.nodesById.get(id);
      return !endpointIds.has(id) &&
        candidate !== undefined &&
        isRouteObstacleNode(candidate) &&
        Number.isFinite(geometry.x) &&
        Number.isFinite(geometry.y) &&
        Number.isFinite(geometry.width) &&
        Number.isFinite(geometry.height) &&
        geometry.width > 0 &&
        geometry.height > 0;
    })
    .map(([id, bounds]) => ({ id, bounds }))
    .sort((left, right) => left.id.localeCompare(right.id));
}
```

- [ ] **Step 3: Route in world space and rebase canonical paths**

Replace direct curve calls in `arrowPathData` with a helper that resolves
world-space anchors and source/target bounds:

```ts
function resolvedCurveRoute(
  node: SceneNodeLike,
  from: ScenePointLike,
  to: ScenePointLike,
  index: SceneNodeIndex,
): CurveRouteResult {
  const start = resolveEndpointWorld(from, index);
  const end = resolveEndpointWorld(to, index);
  const options = normalizeCurveRouteOptions(node.style);
  return routeCurve({
    edgeId: node.id,
    start,
    end,
    fromAnchor: from.anchor,
    toAnchor: to.anchor,
    sourceId: from.nodeId,
    targetId: to.nodeId,
    sourceBounds: from.nodeId ? index.worldGeometryById.get(from.nodeId) : undefined,
    targetBounds: to.nodeId ? index.worldGeometryById.get(to.nodeId) : undefined,
    obstacles: curveObstacles(node, from, to, index, options),
    siblings: [],
    options,
  });
}
```

Add `translateRoutePath(route, -layoutOrigin.x, -layoutOrigin.y)` in the pure
router so SVG command coordinates are rebased without regex editing.

- [ ] **Step 4: Share route resolution with motion signals**

Replace the separate curve branch in `motionConnectorPathData` with the same
`resolvedCurveRoute` helper. Apply `boundaryOnlyMotionPath` only after receiving
the canonical curved path, preserving detours.

- [ ] **Step 5: Exclude route-only style properties from React CSS**

Add these keys to the `styleToCss` skip list:

```ts
key === "clearance" ||
key === "curvature" ||
key === "avoidObstacles" ||
key === "preferredSide" ||
key === "bundle" ||
key === "parallelGap"
```

- [ ] **Step 6: Build and compile every deck**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npm run build
```

Expected: TypeScript exits 0 and Vite reports a successful production build.

---

### Task 3: Adaptive Loops, Parallel Lanes, and Bundling

**Files:**
- Modify: `apps/explainers/src/core/diagram/connector-routing.ts`
- Modify: `apps/explainers/src/core/diagram/connector-routing-search.ts`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`

**Interfaces:**
- Extends: `routeCurve` without changing its signature.
- Consumes: stable sibling routes in scene document order.
- Produces: self-loop candidates, same-side candidates, separated lanes, optional bundles.

- [ ] **Step 1: Add same-side and self-loop candidate generation**

In `connector-routing.ts`, implement:

```ts
function selfLoopCandidates(input: CurveRouteInput): readonly (readonly Point2[])[] {
  const bounds = input.sourceBounds ?? input.targetBounds;
  if (bounds === undefined) return [];
  const gap = input.options.clearance + input.options.parallelGap;
  const left = bounds.x - gap;
  const right = bounds.x + bounds.width + gap;
  const top = bounds.y - gap;
  const bottom = bounds.y + bounds.height + gap;
  return [
    [input.start, { x: input.start.x, y: top }, { x: input.end.x, y: top }, input.end],
    [input.start, { x: right, y: input.start.y }, { x: right, y: input.end.y }, input.end],
    [input.start, { x: input.start.x, y: bottom }, { x: input.end.x, y: bottom }, input.end],
    [input.start, { x: left, y: input.start.y }, { x: left, y: input.end.y }, input.end],
  ];
}
```

Generate clockwise and counterclockwise perimeter candidates when source and
target normals point to the same side. Feed all candidates through the same
visibility and score functions.

- [ ] **Step 2: Escape overlapping endpoint bounds**

Detect overlapping inflated source/target bounds. Increase escape distance in
increments of `max(clearance, 1)` until each escape point is outside both
inflated bounds, capped at four combined box diagonals. If the cap is reached,
mark the route for fallback scoring rather than emitting non-finite geometry.

- [ ] **Step 3: Resolve curved edges in stable scene order**

Build a scene-level map before rendering:

```ts
type ResolvedCurveRoutes = ReadonlyMap<string, CurveRouteResult>;

function resolveSceneCurveRoutes(
  roots: readonly SceneNodeLike[],
  index: SceneNodeIndex,
): ResolvedCurveRoutes;
```

Walk roots in document order, collect curve edges, stable-sort ties by edge ID,
and call `routeCurve` with prior compatible routes converted to
`RoutedSibling`. `arrowPathData` and motion look up by `node.id`.

- [ ] **Step 4: Apply deterministic parallel offsets**

Group siblings by source ID, target ID, and normalized anchor pair. Sort by edge
ID. Assign lane indices:

```ts
const lane = index - (count - 1) / 2;
const offset = lane * options.parallelGap;
```

Offset interior waypoints along segment normals while keeping exact endpoints.
Recheck visibility after offsetting; reduce offset toward zero in 25% steps if
needed.

- [ ] **Step 5: Add optional bundle corridor scoring**

When `bundle` is true and siblings share compatible source/target direction,
extract the longest visible sibling middle segment. Add its endpoints as
corridor candidates and apply a negative corridor cost only to the middle
section. Keep endpoint escape branches independent. When `bundle` is false,
apply a positive congestion cost to overlapping middle segments.

- [ ] **Step 6: Rebuild**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npm run build
```

Expected: exit 0.

---

### Task 4: Browser and Node Verifier Coverage

**Files:**
- Modify: `apps/explainers/src/flow/dev-tools/verify-geometry.ts`
- Modify: `apps/explainers/scripts/flow-verifier/geometry.mjs`
- Modify: `apps/explainers/scripts/flow-verifier/ir.mjs`
- Modify: `apps/explainers/scripts/flow-verifier.mjs`

**Interfaces:**
- Browser verifier imports the pure TypeScript router.
- Node verifier mirrors canonical geometry and routing behavior in ESM.
- IR verifier emits errors for ordinary penetrations and warnings for explicit fallback routes.

- [ ] **Step 1: Update browser-safe geometry verification**

Import `routeCurve`, `normalizeCurveRouteOptions`, and collision helpers in
`verify-geometry.ts`. Replace its direct `curvePathData` call with the same
route input shape used by `SceneRenderer`. Add:

```ts
export type CurveRouteFinding = Readonly<{
  severity: "error" | "warn";
  code: "CURVE_OBSTACLE_PENETRATION" | "CURVE_FALLBACK";
  edgeId: string;
  obstacleIds: readonly string[];
}>;

export function verifyCurveRouteResult(
  edgeId: string,
  result: CurveRouteResult,
  obstacles: readonly RouteObstacle[],
): readonly CurveRouteFinding[];
```

- [ ] **Step 2: Mirror the deterministic router in Node ESM**

Port the pure routing types and functions into
`scripts/flow-verifier/geometry.mjs`. Keep constant values, stable sort keys,
33-sample cubic checking, score weights, fallback order, and three-decimal
formatting byte-equivalent to the TypeScript implementation.

Export:

```js
export function normalizeCurveRouteOptions(style) {}
export function routeCurve(input) {}
export function verifyCurveRouteResult(edgeId, result, obstacles) {}
```

- [ ] **Step 3: Add an explicit verifier scenario matrix**

In `ir.mjs`, export `verifyAdvancedCurveRouting()`. Import and call it once in
`flow-verifier.mjs` before iterating through compiled packages. Construct
synthetic route inputs independent of deck contents:

```js
const ANCHORS = ["center", "n", "s", "e", "w", "ne", "nw", "se", "sw"];
for (const fromAnchor of ANCHORS) {
  for (const toAnchor of ANCHORS) {
    scenarios.push({
      id: `anchors-${fromAnchor}-${toAnchor}`,
      start: { x: 40, y: 100 },
      end: { x: 360, y: 100 },
      fromAnchor,
      toAnchor,
      obstacles: [{ id: "middle", bounds: { x: 170, y: 55, width: 60, height: 90 } }],
    });
  }
}
```

Add fixed scenarios for two obstacles, same-side anchors, overlapping endpoint
bounds, a self-loop, three parallel siblings, bundling, and forced fallback.
For every scenario:

- run the router twice and require identical `d`;
- require finite path coordinates;
- require exact authored start and end;
- require no penetration unless `usedFallback`;
- require fallback obstacle IDs when fallback is true.

- [ ] **Step 4: Run the IR verifier**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npm run flow-verifier:ir
```

Expected:

```text
summary: 0 error(s), 0 warn(s)
```

---

### Task 5: SDK Documentation and Full Verification

**Files:**
- Modify: `apps/explainers/src/flow/sdk/generic/topology.ts`
- Modify: `apps/explainers/src/flow/schema/ir.ts`
- Modify: `apps/explainers/src/flow/language/embedded-scene.ts`

**Interfaces:**
- Documents the style controls already accepted by the open style record.
- Does not add or version a new IR field.

- [ ] **Step 1: Document curve options at the SDK factory**

Update `topology.ts` module docs and `EDGE_ENDPOINT_PROPS` comments to name all
six controls and defaults. Preserve `mode: "curve"` as `core.connector` with
`style.route = "curve"`.

- [ ] **Step 2: Document route behavior in IR**

Update `ConnectorEndpointIr` and connector style comments in `ir.ts` to state
that curve mode uses endpoint normals, obstacle avoidance, deterministic
fallback, and exact endpoint attachment.

- [ ] **Step 3: Update embedded-scene authoring guidance**

List:

```text
style.route: "curve"
style.clearance
style.curvature
style.avoidObstacles
style.preferredSide
style.bundle
style.parallelGap
```

- [ ] **Step 4: Run type checking, build, and fast verification**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npx tsc --noEmit
npm run build
npm run flow-verifier:ir
```

Expected: TypeScript, Vite build, and IR verification all exit 0; IR summary
reports zero errors and zero warnings.

- [ ] **Step 5: Run full-deck playback verification**

Before starting, inspect existing terminals for a healthy Vite server and reuse
it when available. Otherwise run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npm run flow-verifier:extended
```

Expected: all registered deck routes complete without geometry, browser,
console, or playback errors.

- [ ] **Step 6: Inspect final scope**

Run:

```bash
git diff --check
git status --short
git diff -- \
  apps/explainers/src/core/diagram \
  apps/explainers/src/flow/dev-tools/verify-geometry.ts \
  apps/explainers/src/flow/sdk/generic/topology.ts \
  apps/explainers/src/flow/schema/ir.ts \
  apps/explainers/src/flow/language/embedded-scene.ts \
  apps/explainers/scripts/flow-verifier \
  docs/superpowers/specs/2026-07-20-advanced-curved-connector-routing-design.md \
  docs/superpowers/plans/2026-07-20-advanced-curved-connector-routing.md
```

Expected: no whitespace errors; the diff contains only curved-routing
implementation, verifier, and documentation changes. Do not alter or discard
the repository's unrelated pre-existing modifications.

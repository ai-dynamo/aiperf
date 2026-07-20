<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Safe Scene Resolution and Managed Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace renderer-time geometry guesses with one deterministic resolved scene, make directed edges and motion safe by default, and add opt-in managed containers that remove routine coordinate tuning.

**Architecture:** A pure resolver consumes semantic Scene IR and produces canonical node bounds, generated chrome, connectors, motion paths, and source-mapped diagnostics. `SceneRenderer`, browser verification, and the Node verifier consume that result rather than maintaining parallel geometry logic. Existing absolute scenes retain authored coordinates; new stack, grid, rail, overlay, and frame containers opt into managed layout.

**Tech Stack:** TypeScript 5.8, React 19, Zod Scene IR schemas, SVG, Vitest, Vite/vite-node, Node ESM Flow verifier, Playwright

## Global Constraints

- Resolution is pure and deterministic; identical semantic IR and viewport inputs produce identical resolved output and diagnostics.
- Safe deterministic omissions are auto-corrected; ambiguous or lossy geometry is preserved and reported.
- `sdk.Edge` is directed with a visible arrowhead unless the author sets `arrowhead = false`.
- Authored `path` and `points` retain shape precedence and are never silently reversed.
- Existing absolute-positioned nodes outside managed containers retain their authored geometry.
- Existing semantic IDs, documented ports, timeline targets, and accessibility labels remain valid.
- Renderer, verifier, and edge-associated motion consume the same resolved connector path.
- Managed width and height are minimum constraints unless `fixedWidth` or `fixedHeight` is true.
- Add no dependencies.
- Every new source file receives the NVIDIA Apache-2.0 SPDX header and module documentation.
- Do not create git commits unless the user explicitly requests them.

## File Map

- Create `apps/explainers/src/core/diagram/scene-types.ts`: shared authored-scene structural types currently declared in `SceneRenderer.tsx`.
- Create `apps/explainers/src/core/diagram/resolution/types.ts`: resolved scene, node, connector, generated-part, snapshot, and diagnostic contracts.
- Create `apps/explainers/src/core/diagram/resolution/resolve-scene.ts`: canonical world layout, semantic chrome indexing, connector resolution orchestration, and diagnostics.
- Create `apps/explainers/src/core/diagram/resolution/resolve-scene.test.ts`: deterministic layout, paint ownership, and compatibility coverage.
- Create `apps/explainers/src/core/diagram/resolution/resolve-connectors.ts`: path precedence, endpoint resolution, direction policy, route metadata, and path-direction validation.
- Create `apps/explainers/src/core/diagram/resolution/resolve-connectors.test.ts`: directed defaults, explicit opt-out, route reuse, and diagnostics.
- Create `apps/explainers/src/core/diagram/resolution/serialize.ts`: JSON-safe resolved snapshots used by vite-node verification.
- Modify `apps/explainers/src/core/diagram/capabilities/types.ts`: return content bounds, generated ports, and layout diagnostics.
- Modify `apps/explainers/src/core/diagram/capabilities/layout.ts`: padding, alignment, fixed dimensions, overlay, and frame layout.
- Modify `apps/explainers/src/core/diagram/capabilities/layout.test.ts`: managed-container contract coverage.
- Modify `apps/explainers/src/core/diagram/capabilities/registry.ts`: register overlay and frame capabilities.
- Modify `apps/explainers/src/core/diagram/capabilities/chrome.ts`: capability-specific generated IDs and one semantic paint description.
- Modify `apps/explainers/src/core/diagram/SceneRenderer.tsx`: consume `ResolvedScene` and remove independent layout, chrome, and connector inference.
- Modify `apps/explainers/src/core/diagram/SceneRenderer.sdk-primitives.test.tsx`: single-paint, resolved connector, and edge-bound motion assertions.
- Modify `apps/explainers/src/flow/schema/ir.ts`: edge-bound motion IR and managed capability IDs.
- Modify `apps/explainers/src/flow/compiler/semantic-scene-node.ts`: retain `edgeRef` during direct semantic lowering.
- Modify `apps/explainers/src/flow/compiler/semantic-scene-node.test.ts`: edge-bound motion round-trip.
- Modify `apps/explainers/src/flow/sdk/generic/chrome.ts`: emit semantic chrome roots without duplicate generated primitive children.
- Modify `apps/explainers/src/flow/sdk/generic/topology.ts`: directed edge defaults.
- Modify `apps/explainers/src/flow/sdk/generic/motion.ts`: add `sdk.Signal(edge = "edge-id")`.
- Modify `apps/explainers/src/flow/sdk/generic/layout.ts`: expose managed layout inputs plus `sdk.Overlay` and `sdk.Frame`.
- Modify `apps/explainers/src/flow/sdk/generic/catalog.test.ts`: semantic shape, edge, signal, and managed-layout factory tests.
- Modify `apps/explainers/src/flow/sdk/registry.ts`: register overlay and frame definitions.
- Modify `apps/explainers/src/flow/dev-tools/verify-geometry.ts`: verify canonical resolved scenes.
- Modify `apps/explainers/scripts/compile-decks.ts`: emit packages plus JSON-safe resolved scene snapshots.
- Modify `apps/explainers/scripts/flow-verifier.mjs`: load resolved snapshots and add `--verbose`.
- Modify `apps/explainers/scripts/flow-verifier/ir.mjs`: verify resolved geometry instead of recomputing it.
- Modify `apps/explainers/scripts/flow-verifier/geometry.mjs`: remove duplicated layout/routing logic after all consumers move.
- Modify `apps/explainers/scripts/screenshot-deck.mjs`: accept an explicit viewport.
- Modify `apps/explainers/decks-flow/flow-sdk-examples.flow`: document frame, overlay, alignment, padding, fixed sizing, and edge-bound signals.
- Modify `apps/explainers/decks-flow/aiperf-vs-locust.flow`: migrate the worker-process slide to managed containers and automatic routing.
- Modify `apps/explainers/src/flow/language/embedded-scene.ts`: document new authoring defaults and managed inputs.

---

### Task 1: Canonical Scene Types and World-Layout Resolver

**Files:**
- Create: `apps/explainers/src/core/diagram/scene-types.ts`
- Create: `apps/explainers/src/core/diagram/resolution/types.ts`
- Create: `apps/explainers/src/core/diagram/resolution/resolve-scene.ts`
- Create: `apps/explainers/src/core/diagram/resolution/resolve-scene.test.ts`
- Modify: `apps/explainers/src/core/diagram/capabilities/types.ts`
- Modify: `apps/explainers/src/core/diagram/capabilities/registry.ts`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Modify: `apps/explainers/src/core/package-adapter.tsx`
- Modify: `apps/explainers/src/core/final-card-from-scene.tsx`

**Interfaces:**
- Produces: `resolveScene(scene: SceneIrLike): ResolvedScene`.
- Produces: `CapabilityLayout` with `bounds`, `contentBounds`, `childGeometries`, and optional generated ports.
- Produces: shared scene structural types with no React dependency.
- Consumes: existing semantic `SceneIrLike`, `resolveCapabilityLayout`, and authored relative positioning.

- [ ] **Step 1: Write failing canonical-resolution tests**

Create fixtures proving local child placement, absolute compatibility, stable
ancestor maps, and no mutation:

```ts
const scene: SceneIrLike = {
  viewport: { width: 700, height: 400 },
  roots: [
    {
      id: "stack",
      kind: "group",
      capabilityId: "layout.stack",
      geometry: { x: 40, y: 60, width: 0, height: 0 },
      style: { direction: "column", gap: 8 },
      children: [
        panel("one", 100, 30),
        panel("two", 100, 30),
      ],
    },
    panel("absolute", 80, 24, { x: 500, y: 300 }),
  ],
  timeline: [],
};
const resolved = resolveScene(scene);
expect(resolved.worldGeometryById.get("one")).toEqual({
  x: 40, y: 60, width: 100, height: 30,
});
expect(resolved.worldGeometryById.get("two")?.y).toBe(98);
expect(resolved.worldGeometryById.get("absolute")?.x).toBe(500);
expect(resolved.ancestorIdsById.get("two")).toEqual(["stack"]);
expect(scene.roots[0]?.geometry).toEqual({ x: 40, y: 60, width: 0, height: 0 });
expect(resolveScene(scene)).toEqual(resolved);
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/explainers
npx vitest run src/core/diagram/resolution/resolve-scene.test.ts
```

Expected: FAIL because `resolveScene` and the resolution contracts do not
exist.

- [ ] **Step 3: Extract renderer-independent scene types**

Move `SceneGeometryLike`, `ScenePointLike`, `SceneNodeLike`, `SceneIrLike`,
viewport, camera, timeline, accessibility, and style types into
`scene-types.ts`. Add structural source-map data:

```ts
export type SceneSourceRangeLike = Readonly<{
  source: string;
  start: Readonly<{ offset: number; line: number; column: number }>;
  end: Readonly<{ offset: number; line: number; column: number }>;
}>;

export type SceneNodeLike = Readonly<{
  id: string;
  kind?: string;
  capabilityId?: string;
  capability?: string;
  geometry?: SceneGeometryLike;
  layout?: SceneGeometryLike;
  relativePosition?: SceneRelativePositionLike;
  style?: Readonly<Record<string, SceneStyleValue>>;
  props?: Readonly<Record<string, unknown>>;
  text?: string;
  accessibility?: SceneNodeAccessibilityLike;
  sourceMap?: SceneSourceRangeLike;
  children?: readonly SceneNodeLike[];
  d?: string;
  path?: string;
  points?: readonly ScenePointLike[];
  from?: ScenePointLike | readonly ScenePointLike[];
  to?: ScenePointLike | readonly ScenePointLike[];
  via?: ScenePointLike;
  axis?: string;
  junction?: ScenePointLike;
  edgeRef?: string;
}>;
```

Re-export these types from `SceneRenderer.tsx` for source compatibility while
changing capability modules and package adapters to import from
`scene-types.ts`.

- [ ] **Step 4: Define the canonical resolved contracts**

Implement these contracts in `resolution/types.ts`:

```ts
export type SceneResolutionDiagnostic = Readonly<{
  code: string;
  severity: "error" | "warning" | "info";
  message: string;
  range: SceneSourceRangeLike;
  nodeIds: readonly string[];
  repair?: string;
}>;

export type ResolvedGeneratedPart = Readonly<{
  id: string;
  ownerId: string;
  role:
    | "chrome"
    | "title"
    | "detail"
    | "subtitle"
    | "caption"
    | "label"
    | "step";
  geometry: SceneGeometryLike;
}>;

export type ResolvedPoint = Readonly<{ x: number; y: number }>;

export type ResolvedConnector = Readonly<{
  id: string;
  source: ResolvedPoint;
  target: ResolvedPoint;
  sourceId?: string;
  targetId?: string;
  d: string;
  directed: boolean;
  showArrowhead: boolean;
  usedFallback: boolean;
  penetratedObstacleIds: readonly string[];
}>;

export type ResolvedScene = Readonly<{
  scene: SceneIrLike;
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
  ancestorIdsById: ReadonlyMap<string, readonly string[]>;
  generatedPartsById: ReadonlyMap<string, ResolvedGeneratedPart>;
  connectorsById: ReadonlyMap<string, ResolvedConnector>;
  diagnostics: readonly SceneResolutionDiagnostic[];
}>;
```

Extend `CapabilityLayout` without adding renderer state:

```ts
export type CapabilityLayout = Readonly<{
  bounds: SceneGeometryLike;
  contentBounds: SceneGeometryLike;
  childGeometries: readonly SceneGeometryLike[];
  generatedPorts?: Readonly<Record<string, SceneGeometryLike>>;
}>;
```

Existing capability resolvers initially return `contentBounds: bounds`.

- [ ] **Step 5: Implement deterministic world-layout traversal**

Move the current `resolveContainerLayout`, local-coordinate detection,
relative-position handling, and recursive indexing from `SceneRenderer.tsx`
into `resolve-scene.ts`. Expose only:

```ts
export function resolveScene(scene: SceneIrLike): ResolvedScene;
```

Traversal requirements:

1. Visit roots and children in document order.
2. Resolve a container once with `resolveCapabilityLayout`.
3. Record final world geometry before visiting children.
4. Resolve relative positions only against previously visited nodes.
5. Preserve authored absolute coordinates outside local managed containers.
6. Freeze returned arrays and expose read-only maps.
7. Leave `connectorsById` empty until Task 3.

Use a deterministic synthetic range for structural test scenes without source
metadata:

```ts
export const UNKNOWN_SCENE_RANGE: SceneSourceRangeLike = {
  source: "<scene>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};
```

Production Flow nodes retain their authored ranges, so every emitted diagnostic
is source-mapped.

Cut `SceneRenderer` over to one `resolveScene(scene)` call for node lookup,
world geometry, ancestors, and managed child placement. Keep its existing
connector and chrome branches temporarily; Tasks 2 and 3 remove those remaining
parallel interpretations.

- [ ] **Step 6: Run focused layout tests and build**

Run:

```bash
cd apps/explainers
npx vitest run \
  src/core/diagram/resolution/resolve-scene.test.ts \
  src/core/diagram/capabilities/layout.test.ts
npm run build
```

Expected: tests pass and the production build exits zero.

- [ ] **Step 7: Review the Task 1 diff**

Run:

```bash
git diff --check
git diff -- \
  apps/explainers/src/core/diagram/scene-types.ts \
  apps/explainers/src/core/diagram/resolution \
  apps/explainers/src/core/diagram/capabilities \
  apps/explainers/src/core/diagram/SceneRenderer.tsx \
  apps/explainers/src/core/package-adapter.tsx \
  apps/explainers/src/core/final-card-from-scene.tsx
```

Expected: no whitespace errors and no connector, SDK-default, or deck behavior
changes yet. If the user explicitly authorized commits, commit this isolated
foundation as `refactor(explainers): centralize scene resolution`.

---

### Task 2: Single Semantic Paint Ownership and Renderer Cutover

**Files:**
- Modify: `apps/explainers/src/core/diagram/resolution/resolve-scene.ts`
- Modify: `apps/explainers/src/core/diagram/resolution/resolve-scene.test.ts`
- Modify: `apps/explainers/src/core/diagram/capabilities/chrome.ts`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.sdk-primitives.test.tsx`
- Modify: `apps/explainers/src/flow/sdk/generic/chrome.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/catalog.test.ts`

**Interfaces:**
- Consumes: `resolveSemanticChrome(node, resolvedBounds)`.
- Produces: generated parts indexed once under their documented IDs.
- Produces: semantic chrome SDK roots containing authored props but no
  compatibility rect/text children.

- [ ] **Step 1: Add failing SDK and renderer ownership tests**

Add one factory test and one rendering test for each native chrome family:
header, panel, chip, note, lane, band, and stepper. The note regression must
assert:

```ts
expect(noteRoot).toMatchObject({
  capabilityId: "core.note",
  props: { text: "The worker only executes" },
  children: [],
});
expect(noteFragment.ports.caption).toEqual({
  nodeId: `${noteRoot.id}__caption`,
});

const { container } = renderScene(noteRoot);
expect(
  [...container.querySelectorAll("text")].filter(
    (node) => node.textContent === "The worker only executes",
  ),
).toHaveLength(1);
```

Add a resolver fixture containing an authored child whose ID collides with
`${panel.id}__title`; expect one `SCENE_DUPLICATE_PAINT_OWNER` error.
Define local `expandComponent(id, props)`, `renderScene(root)`, and semantic
fixture builders in the test files; each delegates directly to the current SDK
registry or `SceneRenderer` and introduces no production helper.

- [ ] **Step 2: Run focused tests and confirm the duplicate-owner cases fail**

Run:

```bash
cd apps/explainers
npx vitest run \
  src/flow/sdk/generic/catalog.test.ts \
  src/core/diagram/resolution/resolve-scene.test.ts \
  src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
```

Expected: new semantic-shape and duplicate-owner assertions fail.

- [ ] **Step 3: Make semantic chrome IDs capability-specific**

Update `resolveSemanticChrome` so generated IDs preserve existing ports:

```ts
function generatedTextId(node: SceneNodeLike, role: SemanticTextRole): string {
  const capability = node.capabilityId ?? node.capability;
  if (capability === "core.chip") return `${node.id}__label`;
  if (capability === "core.note") return `${node.id}__caption`;
  if (role === "detail" && capability === "core.header") {
    return `${node.id}__caption`;
  }
  return `${node.id}__${role}`;
}
```

For stepper entries, reuse the semantic child IDs (`${node.id}-step-${index}`)
for indexed step geometry and use `${stepId}__label` for generated copy.
Return one root box and one text part per semantic role.

- [ ] **Step 4: Convert native chrome factories to semantic roots**

Rewrite header, panel, chip, and note factories to emit one semantic group:

```ts
const root = withOrigin(
  {
    kind: "group",
    id: context.instanceId,
    capabilityId: "core.note",
    geometry,
    style: { fill: surfaceRole, stroke: strokeRole, strokeWidth, radius },
    props: { text, inkRole },
    accessibility: { label: text },
    fallback: text,
    sourceMap: context.sourceMap,
    children: [],
  },
  context,
  "sdk.note",
  "root",
);
```

Apply the same pattern to header, panel, and chip. Preserve existing port IDs,
action targets, SDK origins, styles, relative positions, accessibility, and
fallback strings. Lane, band, and stepper factories already emit semantic
roots; remove any remaining generated visual children from those paths.

- [ ] **Step 5: Index generated parts and reject duplicate owners**

During `resolveScene`, call `resolveSemanticChrome` after final bounds are
known. Add every generated box/text part to `generatedPartsById` and
`worldGeometryById`. Before insertion, reject an ID already owned by an authored
node or another generated part:

```ts
diagnostics.push({
  code: "SCENE_DUPLICATE_PAINT_OWNER",
  severity: "error",
  message: `Generated part "${part.id}" is owned by both "${prior}" and "${node.id}".`,
  range: node.sourceMap,
  nodeIds: [prior, node.id],
  repair: "Remove the compatibility child; semantic chrome owns this role.",
});
```

- [ ] **Step 6: Make SceneRenderer paint only resolved chrome**

Reuse the single `resolveScene(scene)` result introduced in Task 1. Remove the
remaining direct call to `resolveSemanticChrome`; rendering reads bounds and
generated parts from the resolved maps. Keep timeline state, theme paint,
camera, SVG marker definitions, and accessibility behavior in the renderer.

- [ ] **Step 7: Run focused tests and every deck compilation**

Run:

```bash
cd apps/explainers
npx vitest run \
  src/flow/sdk/generic/catalog.test.ts \
  src/core/diagram/resolution/resolve-scene.test.ts \
  src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
npm run build
npm run flow-verifier:ir
```

Expected: all commands pass and IR verification reports zero errors.

- [ ] **Step 8: Review the Task 2 diff**

Run `git diff --check` and inspect only the files listed for Task 2. If commits
were explicitly authorized, commit as
`refactor(explainers): enforce semantic paint ownership`.

---

### Task 3: Directed Defaults and Canonical Connector Resolution

**Files:**
- Create: `apps/explainers/src/core/diagram/resolution/resolve-connectors.ts`
- Create: `apps/explainers/src/core/diagram/resolution/resolve-connectors.test.ts`
- Modify: `apps/explainers/src/core/diagram/resolution/resolve-scene.ts`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.sdk-primitives.test.tsx`
- Modify: `apps/explainers/src/flow/sdk/generic/topology.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/catalog.test.ts`

**Interfaces:**
- Produces: `resolveConnectors(input: ResolveConnectorsInput): ResolveConnectorsResult`.
- Consumes: final world bounds, ancestor IDs, authored connectors, and existing
  `routeCurve`/`elbowPathData`.
- Produces: all `ResolvedConnector` entries and connector diagnostics.

- [ ] **Step 1: Add failing directed-default tests**

Cover SDK and resolver behavior:

```ts
expect(expandEdge({ mode: "path", path: "M0 0 L100 0" }).style).not.toMatchObject({
  markerEnd: "none",
});
expect(resolve(edge({ from: "a", to: "b" }))).toMatchObject({
  directed: true,
  showArrowhead: true,
});
expect(resolve(edge({ from: "a", to: "b", arrowhead: false }))).toMatchObject({
  directed: false,
  showArrowhead: false,
});
expect(resolve(reversedPathEdge).diagnostics).toContainEqual(
  expect.objectContaining({ code: "SCENE_AUTHORED_PATH_REVERSED" }),
);
```

Also assert a curve fallback carries `usedFallback` and
`penetratedObstacleIds` into a `SCENE_ROUTE_FALLBACK` warning.
Define local `edge(props)` and `resolve(node, fixtures)` helpers in
`resolve-connectors.test.ts`; `resolve` creates source/target panel bounds and
calls `resolveConnectors` directly.

- [ ] **Step 2: Run focused tests and confirm old path defaults fail**

Run:

```bash
cd apps/explainers
npx vitest run \
  src/core/diagram/resolution/resolve-connectors.test.ts \
  src/flow/sdk/generic/catalog.test.ts
```

Expected: FAIL because path/line factories stamp `markerEnd = "none"` and no
canonical connector map exists.

- [ ] **Step 3: Remove the unsafe SDK default**

In `buildEdgeFragment`, do not set `markerEnd` for path or line mode:

```ts
let style: Record<string, StyleValueIr> = { fill: "none" };
if (mode === "route") style.route = "elbow";
if (mode === "curve") style.route = "curve";
if (arrowhead === false) {
  style.markerEnd = "none";
  style.arrowhead = false;
} else {
  style.markerEnd = "arrow";
  style.arrowhead = true;
}
style = { ...style, ...styleOverride };
```

Explicit style overrides remain last. Divider, bracket, and guide factories
must author `arrowhead = false`; remove renderer heuristics based on IDs such
as `split`, `rule`, or `guide`.

- [ ] **Step 4: Implement pure path endpoint extraction**

In `resolve-connectors.ts`, implement:

```ts
export function svgPathEndpoints(
  d: string,
): Readonly<{ start: ScenePoint; end: ScenePoint }> | undefined;
```

Support absolute and relative `M/L/H/V/C/S/Q/T/A/Z` endpoint semantics. Return
`undefined` for malformed or non-finite data. Use this only for validation;
never rewrite authored path data.

- [ ] **Step 5: Resolve every connector once**

Implement:

```ts
export type ResolveConnectorsInput = Readonly<{
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
  ancestorIdsById: ReadonlyMap<string, readonly string[]>;
}>;

export type ResolveConnectorsResult = Readonly<{
  connectorsById: ReadonlyMap<string, ResolvedConnector>;
  diagnostics: readonly SceneResolutionDiagnostic[];
}>;

export function resolveConnectors(
  input: ResolveConnectorsInput,
): ResolveConnectorsResult;
```

Use precedence `d -> path -> points -> curve -> elbow -> straight`. Compare
authored path endpoints with declared anchors using a two-scene-unit tolerance.
Preserve reversed path data and emit `SCENE_AUTHORED_PATH_REVERSED`. Store curve
fallback metadata and emit `SCENE_ROUTE_FALLBACK`. Emit:

- `SCENE_DIRECTED_ARROWHEAD_DEFAULTED` info when direction is auto-corrected;
- `SCENE_CONNECTOR_ENDPOINT_DETACHED` error when resolved path endpoints miss
  declared ports;
- `SCENE_CONNECTOR_INTERSECTION` warning for an authored path crossing an
  unrelated resolved node;
- `SCENE_CONNECTOR_VISUALLY_AMBIGUOUS` warning when the first or final path
  segment comes within the configured clearance of a non-endpoint node.

Automatic routes continue to use the existing deterministic obstacle and
crossing penalties. Stable-sort diagnostics by source location, code, then
node ID.

For `mode = "route"`/`style.route = "elbow"`, reuse the visibility search from
`routeCurve`, but serialize its collision-free `waypoints` as orthogonal
`M/L` segments rather than cubic segments. Explicit `via` remains an authored
two-bend elbow and is preserved, with intersection diagnostics when it crosses
unrelated content.

- [ ] **Step 6: Cut SceneRenderer over to resolved connectors**

Delete renderer-owned `arrowPathData`, scene curve route maps, route obstacle
collection, and arrowhead inference after equivalent pure helpers are used by
`resolveConnectors`. Rendering retrieves `resolved.connectorsById.get(node.id)`
for path data and marker policy.

- [ ] **Step 7: Run connector, renderer, and full IR tests**

Run:

```bash
cd apps/explainers
npx vitest run \
  src/core/diagram/resolution/resolve-connectors.test.ts \
  src/core/diagram/SceneRenderer.sdk-primitives.test.tsx \
  src/flow/sdk/generic/catalog.test.ts
npm run build
npm run flow-verifier:ir
```

Expected: all pass; the verifier summary contains zero errors and zero
warnings.

- [ ] **Step 8: Review the Task 3 diff**

Run `git diff --check`. Confirm authored paths are byte-identical and no
absolute node moved. If commits were explicitly authorized, commit as
`feat(explainers): make directed edges safe by default`.

---

### Task 4: Edge-Bound Motion Signals

**Files:**
- Modify: `apps/explainers/src/flow/schema/ir.ts`
- Modify: `apps/explainers/src/flow/compiler/semantic-scene-node.ts`
- Modify: `apps/explainers/src/flow/compiler/semantic-scene-node.test.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/motion.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/catalog.test.ts`
- Modify: `apps/explainers/src/core/diagram/scene-types.ts`
- Modify: `apps/explainers/src/core/diagram/resolution/resolve-connectors.ts`
- Modify: `apps/explainers/src/core/diagram/resolution/resolve-connectors.test.ts`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.sdk-primitives.test.tsx`

**Interfaces:**
- Produces: Flow authoring `sdk.Signal(edge = "edge-id")`.
- Produces: first-class IR field `edgeRef?: string`.
- Consumes: a previously resolved connector ID.
- Preserves: standalone signal modes using `from`/`to`, `path`, or `points`.

- [ ] **Step 1: Add failing schema, SDK, and renderer tests**

Add these cases:

```ts
const signal = expandSignal({ id: "motion", edge: "request-credit" });
expect(signal).toMatchObject({
  capabilityId: "motion.signal",
  edgeRef: "request-credit",
});
expect(signal).not.toHaveProperty("from");
expect(signal).not.toHaveProperty("to");
expect(sceneNodeSchema.parse(signal)).toEqual(signal);

const resolved = resolveScene(sceneWithEdgeBoundSignal);
expect(resolved.connectorsById.get("motion")?.d).toBe(
  resolved.connectorsById.get("request-credit")?.d,
);
expect(resolved.connectorsById.get("motion")?.showArrowhead).toBe(false);
```

Assert `edge` combined with `from`, `to`, `path`, or `points` fails with
`SDK_SIGNAL_MODE_CONFLICT`, and an unknown edge produces
`SCENE_SIGNAL_EDGE_NOT_FOUND`.
Add a standalone signal that exactly duplicates an existing connector's
endpoints and path; expect `SCENE_SIGNAL_DUPLICATES_EDGE` with a repair message
to use `edge`.

- [ ] **Step 2: Run focused tests and confirm edge mode is rejected**

Run:

```bash
cd apps/explainers
npx vitest run \
  src/flow/compiler/semantic-scene-node.test.ts \
  src/flow/sdk/generic/catalog.test.ts \
  src/core/diagram/resolution/resolve-connectors.test.ts \
  src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
```

Expected: FAIL because `edge`/`edgeRef` are unsupported.

- [ ] **Step 3: Add strict edge-bound motion IR**

Extend `ConnectorNodeIr` with optional endpoints and `edgeRef`:

```ts
export type ConnectorNodeIr = RenderNodeBaseIr & Readonly<{
  kind: "connector";
  from?: ConnectorEndpointIr;
  to?: ConnectorEndpointIr;
  edgeRef?: string;
  via?: PointIr;
  axis?: ConnectorAxisIr;
}>;
```

The Zod refinement requires ordinary connectors to have `from` and `to`.
`motion.signal` requires exactly one mode:

1. `edgeRef`;
2. both `from` and `to`; or
3. `path`/`points`.

Reject partial endpoints and mixed modes. Preserve `edgeRef` in
`lowerSemanticSceneNode`.

- [ ] **Step 4: Add the SDK signal edge mode**

Add `edge: { type: "string", required: false }` to `SIGNAL_PROPS`.
`buildSignalNode` maps it to `edgeRef` and returns
`SDK_SIGNAL_MODE_CONFLICT` for mixed geometry:

```ts
const edgeRef = stringProp(props, "edge");
const modes = [
  edgeRef !== undefined,
  from !== undefined || to !== undefined,
  path !== undefined || points !== undefined,
].filter(Boolean).length;
if (modes !== 1 || ((from === undefined) !== (to === undefined))) {
  return failSignalMode(context, componentId);
}
```

- [ ] **Step 5: Resolve signals from canonical edge paths**

Resolve ordinary connectors first, then edge-bound signals in document order.
Copy `d`, source, target, and fallback metadata from the referenced connector;
force `directed: false` and `showArrowhead: false`. A signal may not reference
another signal. Emit an error for unknown, self, or signal targets.
For standalone signals, compare their exact authored path or endpoint pair with
resolved ordinary connectors. Emit `SCENE_SIGNAL_DUPLICATES_EDGE` when one is
identical; do not auto-convert it because that would alter authored IR.

- [ ] **Step 6: Run focused tests and build**

Run:

```bash
cd apps/explainers
npx vitest run \
  src/flow/compiler/semantic-scene-node.test.ts \
  src/flow/sdk/generic/catalog.test.ts \
  src/core/diagram/resolution/resolve-connectors.test.ts \
  src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
npm run build
```

Expected: all tests and the build pass.

- [ ] **Step 7: Review the Task 4 diff**

Run `git diff --check`. Confirm existing standalone signals compile unchanged.
If commits were explicitly authorized, commit as
`feat(explainers): bind motion signals to resolved edges`.

---

### Task 5: Opt-In Managed Containers

**Files:**
- Modify: `apps/explainers/src/core/diagram/capabilities/types.ts`
- Modify: `apps/explainers/src/core/diagram/capabilities/layout.ts`
- Modify: `apps/explainers/src/core/diagram/capabilities/layout.test.ts`
- Modify: `apps/explainers/src/core/diagram/capabilities/registry.ts`
- Modify: `apps/explainers/src/core/diagram/capabilities/chrome.ts`
- Modify: `apps/explainers/src/flow/schema/ir.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/layout.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/catalog.test.ts`
- Modify: `apps/explainers/src/flow/sdk/registry.ts`
- Modify: `apps/explainers/decks-flow/flow-sdk-examples.flow`

**Interfaces:**
- Extends: `sdk.Stack`, `sdk.Grid`, and `sdk.Rail`.
- Produces: `sdk.Overlay` (`layout.overlay`) and `sdk.Frame`
  (`layout.frame`).
- Shared props: `padding`, `align`, `justify`, `fixedWidth`, and
  `fixedHeight`.
- Child override: `style.position = "absolute"` preserves that child's
  authored local `x`/`y` and removes it from normal flow.

- [ ] **Step 1: Add failing managed-layout tests**

Add pure tests for padding, alignment, fixed overflow, overlay, and title-safe
frame placement:

```ts
expect(resolveCapabilityLayout(stackWithPadding, children)).toMatchObject({
  bounds: { x: 20, y: 30, width: 140, height: 96 },
  contentBounds: { x: 12, y: 12, width: 116, height: 72 },
  childGeometries: [
    { x: 12, y: 12, width: 116, height: 30 },
    { x: 12, y: 50, width: 116, height: 30 },
  ],
});
expect(resolveCapabilityLayout(frame, [child]).childGeometries[0]?.y)
  .toBeGreaterThanOrEqual(34);
expect(resolveCapabilityLayout(overlay, [a, b]).childGeometries)
  .toEqual([
    { x: 0, y: 0, width: 80, height: 40 },
    { x: 0, y: 0, width: 80, height: 40 },
  ]);
```

Add a fixed-width overflow fixture expecting
`SCENE_MANAGED_CONTENT_OVERFLOW`. Add an ordinary managed overlap fixture
expecting `SCENE_MANAGED_CHILD_OVERLAP`; the same children in
`layout.overlay` produce no overlap error.

- [ ] **Step 2: Run layout tests and confirm new contracts fail**

Run:

```bash
cd apps/explainers
npx vitest run \
  src/core/diagram/capabilities/layout.test.ts \
  src/core/diagram/resolution/resolve-scene.test.ts
```

Expected: FAIL because shared inputs, frame, overlay, and managed diagnostics
do not exist.

- [ ] **Step 3: Normalize shared managed inputs**

Add:

```ts
export type ManagedAxisAlignment = "start" | "center" | "end" | "stretch";
export type ManagedMainAlignment =
  | "start"
  | "center"
  | "end"
  | "space-between";

export type ManagedLayoutOptions = Readonly<{
  padding: number;
  align: ManagedAxisAlignment;
  justify: ManagedMainAlignment;
  fixedWidth: boolean;
  fixedHeight: boolean;
}>;

export function managedLayoutOptions(node: SceneNodeLike): ManagedLayoutOptions;
```

Defaults are `padding: 0`, `align: "start"`, `justify: "start"`,
`fixedWidth: false`, and `fixedHeight: false`. Reject non-finite numeric inputs
at SDK validation; clamp padding to zero or greater in package-form
compatibility input.

- [ ] **Step 4: Extend stack, grid, and rail layout**

Place normal-flow children inside padded content bounds, apply `align` on the
cross axis and `justify` on the main axis, and treat authored dimensions as
minimums unless fixed. Children with `style.position = "absolute"` retain local
coordinates and do not advance flow cursors, but remain subject to containment
diagnostics.

- [ ] **Step 5: Implement overlay and frame**

Register:

```ts
{ capabilityId: "layout.overlay", resolveLayout: resolveOverlayLayout },
{ capabilityId: "layout.frame", resolveLayout: resolveFrameLayout },
```

`resolveOverlayLayout` aligns every normal child to the same padded content
box. `resolveFrameLayout` reserves 28 units for title-only content and 48 units
for title plus detail, then lays children out as a column in the remaining
content bounds. Both expand around intrinsic content unless fixed.

Add `layout.frame` to native semantic chrome resolution. It paints one frame
box plus generated `__title` and optional `__detail` parts from the final frame
bounds; layout owns the title-band geometry and paint consumes it.

- [ ] **Step 6: Add SDK factories and schemas**

Expose:

```flow
sdk.Frame(
  id = "worker",
  title = "One worker process, one event loop",
  padding = 14,
  gap = 12,
  fixedWidth = true,
  width = 640
) {
  children { ... }
}

sdk.Overlay(id = "overlay", align = "center", padding = 8) {
  children { ... }
}
```

Add `layout.overlay` and `layout.frame` to foundation capability schemas,
registry packs, local-layout handling, SDK catalog tests, and the Flow SDK
example deck. Preserve child ports as `child[index]`.

- [ ] **Step 7: Run managed-layout tests and compile every deck**

Run:

```bash
cd apps/explainers
npx vitest run \
  src/core/diagram/capabilities/layout.test.ts \
  src/core/diagram/resolution/resolve-scene.test.ts \
  src/flow/sdk/generic/catalog.test.ts
npm run build
npm run flow-verifier:ir
```

Expected: all pass; existing absolute decks retain their previous geometry.

- [ ] **Step 8: Review the Task 5 diff**

Run `git diff --check`. Inspect before/after screenshots for
`flow-sdk-examples`; only the new managed-layout example may change. If commits
were explicitly authorized, commit as
`feat(explainers): add managed scene containers`.

---

### Task 6: Resolver-Backed Browser and Node Verification

**Files:**
- Create: `apps/explainers/src/core/diagram/resolution/serialize.ts`
- Modify: `apps/explainers/src/core/diagram/resolution/types.ts`
- Modify: `apps/explainers/src/core/diagram/resolution/resolve-scene.test.ts`
- Modify: `apps/explainers/src/flow/dev-tools/verify-geometry.ts`
- Modify: `apps/explainers/scripts/compile-decks.ts`
- Modify: `apps/explainers/scripts/flow-verifier.mjs`
- Modify: `apps/explainers/scripts/flow-verifier/ir.mjs`
- Modify: `apps/explainers/scripts/flow-verifier/geometry.mjs`

**Interfaces:**
- Produces: `resolvedSceneSnapshot(resolved: ResolvedScene): ResolvedSceneSnapshot`.
- Produces: verifier bundle `{ packages, resolvedScenes }` from vite-node.
- Consumes: the same `resolveScene` used by `SceneRenderer`.

- [ ] **Step 1: Add failing snapshot and verifier tests**

Round-trip a resolved fixture through `JSON.stringify`/`JSON.parse` and assert:

```ts
expect(snapshot.nodes.find(({ id }) => id === "task-1")?.bounds).toEqual(
  resolved.worldGeometryById.get("task-1"),
);
expect(snapshot.connectors.find(({ id }) => id === "credit")?.d).toBe(
  resolved.connectorsById.get("credit")?.d,
);
expect(snapshot.diagnostics).toEqual(resolved.diagnostics);
```

Add IR verifier fixtures for duplicate paint ownership, missing endpoint,
managed overlap, viewport escape, and route fallback. Assert stable finding
codes and source locations.

- [ ] **Step 2: Run focused tests and confirm snapshot support is absent**

Run:

```bash
cd apps/explainers
npx vitest run src/core/diagram/resolution/resolve-scene.test.ts
npm run flow-verifier:ir -- --deck aiperf-vs-locust
```

Expected: the new snapshot test fails before implementation.

- [ ] **Step 3: Implement the JSON-safe snapshot**

Define:

```ts
export type ResolvedSceneSnapshot = Readonly<{
  sceneId?: string;
  viewport: SceneViewportLike;
  nodes: readonly Readonly<{
    id: string;
    capability: string;
    bounds: SceneGeometryLike;
    ancestorIds: readonly string[];
  }>[];
  generatedParts: readonly ResolvedGeneratedPart[];
  connectors: readonly ResolvedConnector[];
  diagnostics: readonly SceneResolutionDiagnostic[];
}>;

export function resolvedSceneSnapshot(
  resolved: ResolvedScene,
): ResolvedSceneSnapshot;
```

Sort map entries by document order with ID as the tie-break. Never serialize
`Map`, functions, React values, or theme paint.

- [ ] **Step 4: Emit verifier packages and resolved snapshots together**

Change `compile-decks.ts` output to:

```ts
type VerifierBundle = Readonly<{
  packages: readonly DeckPackage[];
  resolvedScenes: readonly Readonly<{
    deckId: string;
    slideId: string;
    snapshot: ResolvedSceneSnapshot;
  }>[];
}>;
```

Resolve every `slide.render?.scene` with `resolveScene`. Also include the final
card scene when present, using slide ID `"__final-card"`.

- [ ] **Step 5: Make IR verification consume snapshots**

Update `loadPackages` to parse `VerifierBundle`. Pass each matching snapshot to
`verifyPackageIr`. Replace verifier-owned final bounds, connector path,
arrowhead, overlap, and paint-ownership calculations with snapshot data.

Add `--verbose` to print informational auto-corrections. Default output hides
info, reports warnings, and fails on errors; `--warn` continues to make
warnings fatal.

- [ ] **Step 6: Remove duplicated geometry algorithms**

Delete layout, route, and direction logic from
`flow-verifier/geometry.mjs` once `rg` proves no consumer remains. Retain only
playhead/time helpers and SVG sampling utilities still used by browser
playback:

```bash
rg "geomOf|arrowPathData|normalizeCurveRouteOptions|routeCurve|resolveFanGeometry" \
  scripts/flow-verifier scripts/flow-verifier.mjs
```

For each match, either switch it to snapshot data or document why it is a DOM
playback-only helper.

- [ ] **Step 7: Make browser dev verification call the same resolver**

In `verify-geometry.ts`, call `resolveScene(scene)` once and base all static
geometry findings on its maps and diagnostics. Browser-only checks may inspect
computed SVG path length and visibility, but must compare against the resolved
connector ID and path.

Add a pure final validation pass in `resolveScene` before serialization:

- `SCENE_ABSOLUTE_SIBLING_OVERLAP` warning for intersecting siblings outside
  `layout.overlay`;
- `SCENE_VIEWPORT_ESCAPE` warning when visible node, generated text, or
  arrow-tip bounds exceed the scene viewport;
- `SCENE_FIXED_CONTENT_OVERFLOW` error when fixed managed content exceeds its
  content bounds;
- `SCENE_DUPLICATE_GENERATED_ID` error for any repeated generated part ID.

The pass uses resolved bounds only and therefore feeds renderer, browser
verification, and Node verification identically.

- [ ] **Step 8: Run verifier and production gates**

Run:

```bash
cd apps/explainers
npm test
npm run build
npm run flow-verifier:ir
npm run flow-verifier:extended
```

Expected: all commands exit zero; the IR summary reports zero errors. Warnings
from unconverted absolute-positioned scenes are printed with
source locations but remain non-fatal unless `--warn` is supplied.

- [ ] **Step 9: Review the Task 6 diff**

Run `git diff --check`. Confirm the Node verifier imports resolution results
produced under vite-node and no longer mirrors layout/routing formulas. If
commits were explicitly authorized, commit as
`test(explainers): verify canonical resolved scenes`.

---

### Task 7: Worker-Slide Migration and Responsive Acceptance

**Files:**
- Modify: `apps/explainers/decks-flow/aiperf-vs-locust.flow`
- Modify: `apps/explainers/decks-flow/flow-sdk-examples.flow`
- Modify: `apps/explainers/scripts/screenshot-deck.mjs`
- Modify: `apps/explainers/src/flow/language/embedded-scene.ts`
- Modify: `docs/superpowers/specs/2026-07-20-safe-scene-resolution-and-managed-layout-design.md`

**Interfaces:**
- Consumes: canonical connector resolution, edge-bound signals, frame, stack,
  rail, and overlay.
- Produces: the representative migrated slide and documented authoring examples.

- [ ] **Step 1: Add a worker-slide structural regression**

Extend the IR verifier's deck-specific expectations for slide
`AIPerf: inside one worker process`:

```js
const nodes = new Map(snapshot.nodes.map((node) => [node.id, node]));
const connectors = new Map(
  snapshot.connectors.map((connector) => [connector.id, connector]),
);
expectNode(nodes, "s10-worker", "layout.frame");
expectNode(nodes, "s10-tasks", "layout.stack");
expectNode(nodes, "s10-steps", "layout.rail");
if (connectors.get("s10-e8")?.showArrowhead !== true) {
  findings.push(finding("error", deck.id, slide.id, "worker-credit-direction",
    "s10-e8 must resolve as a directed edge with an arrowhead"));
}
if (sceneNodeById(scene, "s10-motion")?.edgeRef !== "s10-e8") {
  findings.push(finding("error", deck.id, slide.id, "worker-motion-edge",
    "s10-motion must reference s10-e8"));
}
```

Add `expectNode` and `sceneNodeById` as local `ir.mjs` helpers in this step.
`expectNode` appends a finding when an ID or capability differs; it does not
throw, so all deck findings remain aggregated. Also require no
`SCENE_MANAGED_CHILD_OVERLAP` diagnostic and exactly one generated caption or
label owned by `s10-note`.

- [ ] **Step 2: Run the deck verifier and confirm the migration assertions fail**

Run:

```bash
cd apps/explainers
npm run flow-verifier:ir -- --deck aiperf-vs-locust
```

Expected: FAIL because slide 10 still uses absolute panels, an authored `path`,
and endpoint-based motion.

- [ ] **Step 3: Migrate the worker slide**

Replace manual child coordinates with:

- `sdk.Frame(id = "s10-worker")` for the titled worker region;
- a row `sdk.Rail` containing `s10-recv`, a column
  `sdk.Stack(id = "s10-tasks")`, and `s10-pool`;
- a row `sdk.Rail(id = "s10-steps")` containing `s10-step1` and `s10-step2`;
- automatic `sdk.Edge` routing for `s10-e8`, with no authored path and no
  explicit arrowhead;
- `sdk.Signal(id = "s10-motion", edge = "s10-e8")`;
- one semantic `sdk.Note` or `sdk.Label` for the footer copy.

Keep all public timeline target IDs stable. New container IDs are timeline
neutral except where the structural regression names them.

- [ ] **Step 4: Add viewport control to the screenshot tool**

Accept `--viewport WIDTHxHEIGHT`:

```js
function parseViewport(value) {
  const match = /^(\d+)x(\d+)$/.exec(value ?? "");
  if (!match) throw new Error(`invalid --viewport "${value}"; expected WIDTHxHEIGHT`);
  return { width: Number(match[1]), height: Number(match[2]) };
}
```

Default remains the existing viewport. Include the chosen viewport in the
result JSON.

- [ ] **Step 5: Document safe authoring**

Update `embedded-scene.ts`, `flow-sdk-examples.flow`, and the approved design
spec with final implemented names and defaults:

- edges are directed unless `arrowhead = false`;
- edge-associated motion uses `edge`;
- managed containers are opt-in;
- `padding`, `align`, `justify`, `fixedWidth`, and `fixedHeight`;
- `style.position = "absolute"` is the explicit managed-child escape hatch.

- [ ] **Step 6: Run complete verification**

Run:

```bash
cd apps/explainers
npm test
npm run build
npm run assert:sdk-authoring -- --strict
npm run flow-verifier:ir
npm run flow-verifier:ir -- --deck aiperf-vs-locust --warn
npm run flow-verifier:extended
node scripts/screenshot-deck.mjs \
  --deck aiperf-vs-locust \
  --viewport 1440x1100 \
  --out artifacts/screenshots/aiperf-vs-locust-safe-desktop
node scripts/screenshot-deck.mjs \
  --deck aiperf-vs-locust \
  --viewport 1280x720 \
  --out artifacts/screenshots/aiperf-vs-locust-safe-short
node scripts/screenshot-deck.mjs \
  --deck aiperf-vs-locust \
  --viewport 390x844 \
  --out artifacts/screenshots/aiperf-vs-locust-safe-mobile
```

Expected: every command exits zero. Repository-wide IR reports zero errors;
the targeted `aiperf-vs-locust --warn` run also proves the representative deck
has zero warnings.

- [ ] **Step 7: Inspect slide 10 at all three viewport sizes**

Confirm:

- worker title and header caption have clear separation;
- task, pool, and control-flow copy do not overlap;
- `s10-e8` visibly runs from slot granted to credit in;
- motion follows exactly `s10-e8`;
- semantic note copy appears once;
- diagram, subtitles, and footer remain visible.

- [ ] **Step 8: Final scope and formatting review**

Run:

```bash
git diff --check
git status --short
git diff -- \
  apps/explainers/src/core/diagram \
  apps/explainers/src/flow \
  apps/explainers/scripts \
  apps/explainers/decks-flow/aiperf-vs-locust.flow \
  apps/explainers/decks-flow/flow-sdk-examples.flow \
  docs/superpowers/specs/2026-07-20-safe-scene-resolution-and-managed-layout-design.md \
  docs/superpowers/plans/2026-07-20-safe-scene-resolution-and-managed-layout.md
```

Expected: no whitespace errors and no unrelated pre-existing changes are
altered or discarded. If commits were explicitly authorized, commit the
migration and documentation as
`feat(explainers): migrate worker slide to managed layout`.

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Semantic Scene IR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace compile-time scene-primitive desugaring with source-compatible semantic Scene IR rendered and laid out through one native capability registry.

**Architecture:** SDK and package-form lowering emit semantic nodes that retain their canonical capability IDs and authored data. Pure capability layout modules resolve final bounds and child placement once; `SceneRenderer` uses those results for indexing, connector anchors, and rendering. Migration adapters keep each intermediate state buildable, then all production calls to `desugarPackageNode` are removed.

**Tech Stack:** TypeScript 5.8, React 19, Zod Scene IR schemas, SVG, Vitest, Vite, Node ESM Flow verifier, Playwright

## Global Constraints

- Existing `.flow` syntax and canonical SDK names remain unchanged.
- Existing root IDs, authored child IDs, semantic ports, and timeline action targets remain valid.
- Positive authored width and height are minimum dimensions; zero requests intrinsic sizing.
- SDK factories remain deterministic and must not depend on React, the DOM, network access, wall-clock time, or mutable global state.
- Browser rendering and geometry verification consume the same pure layout functions.
- Unknown capabilities fail closed.
- No dependencies are added.
- No feature flag or second production rendering path remains at completion.
- Do not create git commits unless the user explicitly requests them.

## File Map

- Create `apps/explainers/src/core/diagram/capabilities/types.ts`: native capability layout and registry contracts.
- Create `apps/explainers/src/core/diagram/capabilities/layout.ts`: pure identity, stack, grid, rail, pad, lane, band, swimlane, and stepper layout.
- Create `apps/explainers/src/core/diagram/capabilities/registry.ts`: explicit registration, duplicate detection, and fail-closed lookup.
- Create `apps/explainers/src/core/diagram/capabilities/layout.test.ts`: intrinsic/minimum sizing and deterministic child placement.
- Create `apps/explainers/src/flow/compiler/semantic-scene-node.ts`: direct package-record/AST lowering to semantic `RenderNodeIr`.
- Create `apps/explainers/src/flow/compiler/semantic-scene-node.test.ts`: schema and generated-ID compatibility.
- Modify `apps/explainers/src/flow/schema/ir.ts`: retain capability-specific authored props on semantic nodes under strict schemas.
- Modify `apps/explainers/src/core/diagram/SceneRenderer.tsx`: consume registry layout for indexing and rendering.
- Modify `apps/explainers/src/flow/compiler/lower.ts`: direct semantic primitive lowering.
- Modify `apps/explainers/src/flow/compiler/lower-explainer-scene.ts`: direct package-form semantic lowering.
- Modify `apps/explainers/src/flow/compiler/expand-sdk.ts`: direct semantic freeform lowering.
- Modify `apps/explainers/src/flow/sdk/generic/layout.ts`: emit semantic layout nodes without compiler imports.
- Modify `apps/explainers/src/flow/sdk/generic/chrome.ts`: emit semantic chrome nodes without generated visual children.
- Modify `apps/explainers/src/flow/compiler/desugar-scene-primitives.ts`: reduce to direct lowering helpers, then delete desugar-only branches.
- Modify `apps/explainers/src/flow/dev-tools/verify-geometry.ts`: use shared capability layout.
- Modify `apps/explainers/scripts/flow-verifier/geometry.mjs`: consume compiled shared layout rather than maintaining layout formulas.
- Modify `docs/superpowers/specs/2026-07-20-expanded-sdk-component-primitives-design.md`: reference semantic Scene IR as the authoritative expansion contract.

---

### Task 1: Native Capability Layout Registry

**Files:**
- Create: `apps/explainers/src/core/diagram/capabilities/types.ts`
- Create: `apps/explainers/src/core/diagram/capabilities/layout.ts`
- Create: `apps/explainers/src/core/diagram/capabilities/registry.ts`
- Create: `apps/explainers/src/core/diagram/capabilities/layout.test.ts`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`

**Interfaces:**
- Produces: `CapabilityLayout`, `NativeSceneCapability`, `resolveCapabilityLayout(node, children)`.
- Consumes: existing `SceneNodeLike`-compatible geometry, capability ID, style, and children.

- [ ] **Step 1: Write failing pure-layout tests**

Cover:

```ts
expect(resolveCapabilityLayout(stepper, [])).toMatchObject({
  bounds: { x: 500, y: 220, width: 280, height: 90 },
});
expect(resolveCapabilityLayout(lane, [panelA, panelB]).bounds.height).toBeGreaterThanOrEqual(174);
expect(resolveCapabilityLayout(rail, [chipA, chipB, chipC]).bounds.height).toBe(26);
expect(() => createCapabilityRegistry([identity, identity])).toThrow(/duplicate/i);
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
cd apps/explainers
npx vitest run src/core/diagram/capabilities/layout.test.ts
```

Expected: failure because capability modules do not exist.

- [ ] **Step 3: Add contracts and deterministic registry**

Implement:

```ts
export interface CapabilityLayout {
  readonly bounds: SceneGeometryLike;
  readonly childGeometries: readonly SceneGeometryLike[];
}

export interface NativeSceneCapability {
  readonly capabilityId: string;
  resolveLayout(
    node: SceneNodeLike,
    children: readonly SceneNodeLike[],
  ): CapabilityLayout;
}

export function createCapabilityRegistry(
  definitions: readonly NativeSceneCapability[],
): ReadonlyMap<string, NativeSceneCapability>;

export function resolveCapabilityLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout;
```

Register identity layout for leaf/chrome primitives and native layout
definitions for `layout.stack`, `layout.grid`, `layout.rail`, `layout.pad`,
`core.lane`, `core.band`, `core.swimlane`, and `core.stepper`.

- [ ] **Step 4: Move layout formulas out of `SceneRenderer`**

Replace `computeStackLayout`, `computeGridLayout`, `computeRailLayout`, and
`resolveContainerLayout` with the shared registry call. Preserve world
translation and relative-position handling in `SceneRenderer`.

- [ ] **Step 5: Run focused tests and build**

Run:

```bash
cd apps/explainers
npx vitest run src/core/diagram/capabilities/layout.test.ts
npm run build
```

Expected: tests pass; TypeScript and Vite build pass.

### Task 2: Semantic Scene Node Schema and Direct Lowering

**Files:**
- Create: `apps/explainers/src/flow/compiler/semantic-scene-node.ts`
- Create: `apps/explainers/src/flow/compiler/semantic-scene-node.test.ts`
- Modify: `apps/explainers/src/flow/schema/ir.ts`
- Modify: `apps/explainers/src/flow/schema/capability.ts`

**Interfaces:**
- Consumes: normalized package records and `ScenePrimitiveAst` values.
- Produces: `lowerSemanticSceneNode(record, common): RenderNodeIr`.

- [ ] **Step 1: Add failing semantic round-trip tests**

Assert:

```ts
const panel = lowerSemanticSceneNode(
  {
    id: "panel",
    capability: "core.panel",
    geometry: { x: 10, y: 20, width: 160, height: 64 },
    title: "Profile",
    detail: "source",
  },
  common,
);
expect(panel).toMatchObject({
  id: "panel",
  capabilityId: "core.panel",
  title: "Profile",
  detail: "source",
  children: [],
});
expect(panel.children).not.toEqual(
  expect.arrayContaining([expect.objectContaining({ id: "panel-title" })]),
);
expect(sceneIrSchema.parse({ roots: [panel], timeline: [] })).toBeTruthy();
```

Add equivalent cases for header, chip, lane, band, swimlane, stepper, pad,
circle, ellipse, bracket, callout, note, and divider.

- [ ] **Step 2: Run the tests and confirm direct lowering is absent**

Run:

```bash
cd apps/explainers
npx vitest run src/flow/compiler/semantic-scene-node.test.ts
```

Expected: failure because `lowerSemanticSceneNode` is not defined.

- [ ] **Step 3: Add semantic payload fields and strict schema branches**

Retain canonical authored fields on semantic nodes:

```ts
title?: string;
detail?: string;
caption?: string;
text?: string;
steps?: readonly string[];
labels?: readonly string[];
linked?: boolean;
inset?: number;
orientation?: string;
```

Use capability-specific strict refinements so unsupported authored payload
fields are rejected while the shared style vocabulary remains accepted.

- [ ] **Step 4: Implement direct semantic lowering**

`lowerSemanticSceneNode` must:

- normalize `capability`/`capabilityId`;
- choose the existing IR `kind` without generating visual children;
- preserve authored children;
- preserve geometry, style, accessibility, fallback, source map, endpoints,
  paths, fan arrays, and motion fields;
- retain semantic title/detail/steps/labels/linkage fields.

- [ ] **Step 5: Run semantic tests and schema tests**

Run:

```bash
cd apps/explainers
npx vitest run src/flow/compiler/semantic-scene-node.test.ts
npm run build
```

Expected: all pass.

### Task 3: Native Layout and Chrome Rendering

**Files:**
- Create: `apps/explainers/src/core/diagram/capabilities/chrome.tsx`
- Modify: `apps/explainers/src/core/diagram/capabilities/registry.ts`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.sdk-primitives.test.tsx`

**Interfaces:**
- Consumes: semantic node plus `CapabilityLayout`.
- Produces: `renderCapabilityChrome(input): ReactNode`.

- [ ] **Step 1: Add renderer tests for semantic nodes without generated children**

Render semantic panel, header, chip, lane, band, swimlane, and stepper nodes
whose `children` arrays contain only authored children. Assert visible title,
detail, chrome, and step labels plus deterministic internal DOM IDs.

- [ ] **Step 2: Confirm tests fail**

Run:

```bash
cd apps/explainers
npx vitest run src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
```

Expected: semantic text/chrome is absent.

- [ ] **Step 3: Implement native chrome render helpers**

Implement capability-specific SVG fragments in `chrome.tsx` using the current
theme roles and geometry constants. Generated visual IDs use:

```ts
`${node.id}__chrome`
`${node.id}__title`
`${node.id}__detail`
`${node.id}__caption`
`${node.id}__step-${index}`
```

These are DOM IDs only unless already exposed by a documented action or port.

- [ ] **Step 4: Delegate semantic rendering from `SceneRenderer`**

Keep timeline opacity, route rendering, world transforms, markers, and
accessibility in `SceneRenderer`. Delegate panel/header/chip/note/callout/
bracket/divider/lane/band/swimlane/stepper visual fragments to native helpers.

- [ ] **Step 5: Verify renderer and build**

Run:

```bash
cd apps/explainers
npx vitest run src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
npm run build
```

Expected: tests and build pass.

### Task 4: SDK Factories Emit Semantic Nodes

**Files:**
- Modify: `apps/explainers/src/flow/sdk/generic/layout.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/chrome.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/catalog.test.ts`
- Modify: `apps/explainers/src/flow/sdk/diagram/catalog.test.ts`

**Interfaces:**
- Consumes: existing component props, slots, expansion context.
- Produces: semantic `SceneFragment` roots plus unchanged ports/actions.

- [ ] **Step 1: Extend catalog tests with semantic-shape assertions**

Assert `sdk.Panel` returns one `core.panel` root with title/detail fields and
no generated title/detail children. Assert lane/band/swimlane/stepper roots
retain their capability IDs and authored child IDs. Snapshot all ports and
actions before changing factories.

- [ ] **Step 2: Confirm tests fail under primitive-tree factories**

Run:

```bash
cd apps/explainers
npx vitest run src/flow/sdk/generic/catalog.test.ts
```

Expected: semantic-shape assertions fail.

- [ ] **Step 3: Rewrite layout factories**

Remove the compiler import and `desugarOrFail`. Emit semantic group nodes:

```ts
{
  kind: "group",
  id: context.instanceId,
  capabilityId: "core.stepper",
  geometry: geometryFromProps(props),
  style: { gap },
  steps,
  linked,
  children: stepRoots,
}
```

Preserve the exact existing `ports` and `actions` maps.

- [ ] **Step 4: Rewrite chrome factories**

Emit one semantic root for panel/header/chip/note/callout/bracket/divider and
retain only authored slot children. Preserve root IDs, semantic ports, action
targets, SDK provenance, fallback, and accessibility.

- [ ] **Step 5: Run catalog tests and all deck compilation**

Run:

```bash
cd apps/explainers
npx vitest run src/flow/sdk/generic/catalog.test.ts src/flow/sdk/diagram/catalog.test.ts
npm run flow-verifier:ir
```

Expected: tests pass; every `.flow` deck compiles.

### Task 5: Compiler Cutover and Desugar Removal

**Files:**
- Modify: `apps/explainers/src/flow/compiler/lower.ts`
- Modify: `apps/explainers/src/flow/compiler/lower-explainer-scene.ts`
- Modify: `apps/explainers/src/flow/compiler/expand-sdk.ts`
- Modify: `apps/explainers/src/flow/compiler/desugar-scene-primitives.ts`
- Modify: `apps/explainers/src/flow/compiler/validate-sdk-authoring.ts`

**Interfaces:**
- Consumes: `lowerSemanticSceneNode` from Task 2.
- Produces: one direct semantic lowering path with no `desugarPackageNode`.

- [ ] **Step 1: Add compatibility tests for all compiler entry points**

Compile equivalent panel/stepper scenes through:

- package-form scene lowering;
- native scene-primitive lowering;
- SDK scene mixed with freeform package records.

Assert equivalent root capability IDs, authored payload, IDs, ports, and
timeline targets.

- [ ] **Step 2: Route all three call sites through direct semantic lowering**

Replace calls in:

```ts
lowerScenePrimitive(...)
normalizePackageNode(...)
normalizePackageRecord(...)
```

with `lowerSemanticSceneNode`. Preserve diagnostics for unknown capability IDs.

- [ ] **Step 3: Remove macro-only production code**

Delete `DESUGAR_PACKAGE_CAPABILITIES`, `isDesugarCapability`,
`desugarPackageNode`, `desugarOrFail`, and their imports. Keep only direct
first-class normalization helpers still used by semantic lowering, renaming
the file/module to avoid “desugar” terminology if imports remain.

- [ ] **Step 4: Add a source-level regression gate**

Assert no production source imports or invokes `desugarPackageNode`:

```bash
rg "desugarPackageNode|DESUGAR_PACKAGE_CAPABILITIES" \
  apps/explainers/src \
  --glob '!**/*.test.ts'
```

Expected: no matches.

- [ ] **Step 5: Run compiler and package gates**

Run:

```bash
cd apps/explainers
npm test
npm run build
npm run assert:sdk-authoring -- --strict
npm run flow-verifier:ir
```

Expected: all pass.

### Task 6: Shared Verification and Documentation

**Files:**
- Modify: `apps/explainers/src/flow/dev-tools/verify-geometry.ts`
- Modify: `apps/explainers/scripts/flow-verifier/geometry.mjs`
- Modify: `docs/superpowers/specs/2026-07-20-expanded-sdk-component-primitives-design.md`

**Interfaces:**
- Consumes: `resolveCapabilityLayout`.
- Produces: browser/verifier geometry parity and current documentation.

- [ ] **Step 1: Replace verifier layout formulas**

Expose a Node-compatible build of the pure capability layout module and call
it from browser and script verification. Keep connector routing checks
unchanged.

- [ ] **Step 2: Add verifier checks**

Reject:

- non-finite final bounds;
- child geometry outside non-overflowing semantic containers;
- missing action/port target IDs;
- serialized generated chrome/title/detail children under semantic chrome
  roots.

- [ ] **Step 3: Update the expanded SDK design**

Replace the ordinary-primitive expansion claim with a reference to
`2026-07-20-native-semantic-scene-ir-design.md`; retain its component catalog.

- [ ] **Step 4: Run complete non-HMR verification**

Run:

```bash
cd apps/explainers
npm test
npm run build
npm run flow-verifier:extended
node scripts/screenshot-deck.mjs --deck flow-sdk-examples
```

Expected: all commands exit zero and screenshots include ten unique slides
plus the final card.

- [ ] **Step 5: Inspect visual acceptance targets**

Review slides 2, 3, 4, 7, 9, 10, and `final-card.png` for:

- no clipping;
- readable stepper labels;
- lane/band child containment;
- clean pipeline/fan/flow routes;
- no header or final-card bleed.


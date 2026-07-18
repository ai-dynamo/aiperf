# AIPerf Flow Semantic Projection Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the single landed runtime `SemanticProjection` consumed by
Canvas hit metadata, `SemanticTwin`, and `SvgFallback`: remove transitional
`kind`/`role` overlap, align evaluated-scene time naming, and lock
cross-backend conformance without adapters or information loss.

**Architecture:** The canonical TypeScript contract is landed in
`@aiperf/flow-runtime` under `evaluate/types.ts`; the twin-local duplicate has
been removed and runtime consumers import the shared shape. Normalize the
remaining transitional fields, strengthen evaluator/twin/SVG/Canvas
conformance, and keep optional schema promotion in the display-list plan.

**Tech Stack:** TypeScript strict mode, Vitest, Testing Library, `@aiperf/flow-runtime` (and existing `@aiperf/flow-schema` Scene IR only as evaluator input).

## Global Constraints

- One `SemanticProjection` type in the runtime. Re-exports are allowed; a second structural definition is forbidden.
- The projection is backend-neutral: no Canvas, React, or SVG types in the contract.
- Canvas hit metadata, `SemanticTwin`, and `SvgFallback` must consume the same projection object (or the same fields via `EvaluatedScene.semantic`) without renaming adapters.
- Do not modify `apps/aiperf-flow/preview/**`.
- Do not add React/SVG capability renderers, hybrid leaf renderers, or document-specific scene components.
- Do not reinterpret Flow IR inside backends; backends read evaluated products only.
- Activate `.venv` before repo commands: `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Verify with `cd apps/aiperf-flow && npm test -w @aiperf/flow-runtime` (and `npm run flow:check` if present).
- Do not create git commits unless the user explicitly requests them.

## Current implementation baseline

One runtime `SemanticProjection` is defined in `evaluate/types.ts` and consumed
by `FlowApp`, the scene evaluator, merge logic, `SemanticTwin`, fallback table,
focus coordinator, and SVG fallback. It already carries scene identity, reading
order, entities, `fromId`/`toId` relations, evidence, transcript cue, and
captions.

Remaining drift is narrow but normative:

- `SemanticEntityProjection` still has both `role?` and transitional `kind?`;
- twin DOM currently falls back from `kind` to `role`;
- `EvaluatedScene` uses `atMs` while the player and promoted display-list
  contracts use `timeMs`;
- cross-backend tests must prove one projection object drives Canvas hit labels,
  semantic focus, SVG fallback labels, evidence, transcript, and captions.

The tasks below are a hardening/migration checklist. Already-landed unification
steps must be verified and preserved, not rebuilt as a second type.

## Execution status

- Tasks 1 and 2 are landed except for renaming `EvaluatedScene.atMs` to
  `timeMs`.
- Task 3’s shared-type import, relation migration, and focus coordination are
  landed; removing entity `kind` and its DOM fallback remains.
- Tasks 4 and 5 are landed at the unit-contract level; strengthen them with one
  shared cross-backend conformance fixture.
- Task 6 remains the completion gate.

Failing-test expectations in landed steps document the original TDD sequence;
do not recreate removed duplicate types merely to reproduce those failures.

## Canonical contract

Place the single definition in `apps/aiperf-flow/packages/runtime/src/evaluate/types.ts`:

```typescript
import type { SourceReference } from "../display-list.js";

/** One semantic entity exposed consistently by every render backend. */
export type SemanticEntityProjection = Readonly<{
  id: string;
  /** Human-readable label used for aria-label, hit metadata, and twin buttons. */
  label: string;
  /** Accessibility / semantic role (maps to twin `data-role`, not React role props). */
  role?: string;
  description?: string;
  evidenceIds?: readonly string[];
  source?: SourceReference;
}>;

/** One directed semantic relationship between projected entities. */
export type SemanticRelationProjection = Readonly<{
  id: string;
  fromId: string;
  toId: string;
  label?: string;
  role?: string;
  source?: SourceReference;
}>;

/**
 * Backend-neutral accessibility and interaction meaning for a scene.
 * Consumed by Canvas hit metadata, SemanticTwin, and SvgFallback.
 */
export type SemanticProjection = Readonly<{
  sceneId: string;
  readingOrder: readonly string[];
  entities: readonly SemanticEntityProjection[];
  relations: readonly SemanticRelationProjection[];
  transcriptCueId?: string;
  captions?: readonly string[];
}>;

/** One immutable scene snapshot at an integer virtual time. */
export type EvaluatedScene = Readonly<{
  sceneId: string;
  timeMs: number;
  displayList: DisplayList;
  semantic: SemanticProjection;
}>;
```

**Naming rules locked by this plan:**

- Relations use `fromId` / `toId` (not `from` / `to`).
- Classification uses `role` (not `kind`). Twin DOM may expose `data-role={entity.role}`.
- `sceneId` is required on the projection and must equal `EvaluatedScene.sceneId`.
- Entity `label` is the sole focus/selection-friendly display string for hit metadata and twin buttons. Do not invent parallel `focusTarget` strings on the projection; focus targets default to entity `id`.

## File map

| File | Responsibility after this plan |
| --- | --- |
| `packages/runtime/src/evaluate/types.ts` | Canonical `SemanticProjection` (+ entity/relation) |
| `packages/runtime/src/evaluate/scene-evaluator.ts` | Emits full canonical projection |
| `packages/runtime/src/semantic/semantic-twin.tsx` | React twin only; imports types from evaluate |
| `packages/runtime/src/semantic/focus-coordinator.ts` | Uses canonical projection |
| `packages/runtime/src/backends/svg/svg-fallback.tsx` | Reads `scene.semantic` canonical fields |
| `packages/runtime/src/backends/canvas/canvas-renderer.ts` | Resolves hit labels from `SemanticProjection` (or labeled hit regions derived from it) |
| `packages/runtime/test/fixtures/evaluated-scene.ts` | One conformance fixture projection |
| `packages/runtime/test/**` | Updated expectations; no second shape |

## Optional schema promotion

Unifying the runtime TypeScript type is **in scope**. Promoting it to a strict Zod `SemanticProjectionIr` in `@aiperf/flow-schema` is **optional and out of scope here**. When schema promotion lands, follow [`2026-07-17-aiperf-flow-display-list.md`](2026-07-17-aiperf-flow-display-list.md): add `SemanticProjectionIr` beside `EvaluatedSceneIr`, keep field names identical to this canonical contract, and re-export runtime types from schema or make runtime types aliases of the parsed IR. Do not fork a third shape during promotion.

---

### Task 1: Canonical type + contract tests

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/types.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/semantic-projection.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts` (export evaluate types if not already public)

**Interfaces:**
- Consumes: `SourceReference` from `display-list.ts`
- Produces: `SemanticEntityProjection`, `SemanticRelationProjection`, `SemanticProjection` as defined in Canonical contract

- [ ] **Step 1: Write the failing contract test**

```typescript
// apps/aiperf-flow/packages/runtime/test/evaluate/semantic-projection.test.ts
import { describe, expect, test } from "vitest";

import type {
  EvaluatedScene,
  SemanticProjection,
} from "../../src/evaluate/types.js";

function assertProjection(projection: SemanticProjection): void {
  expect(projection.sceneId.length).toBeGreaterThan(0);
  expect(projection.readingOrder.length).toBeGreaterThan(0);
  for (const id of projection.readingOrder) {
    const known =
      projection.entities.some((entity) => entity.id === id) ||
      projection.relations.some((relation) => relation.id === id);
    expect(known).toBe(true);
  }
  for (const relation of projection.relations) {
    expect(relation).toHaveProperty("fromId");
    expect(relation).toHaveProperty("toId");
    expect(relation).not.toHaveProperty("from");
    expect(relation).not.toHaveProperty("to");
  }
  for (const entity of projection.entities) {
    expect(typeof entity.label).toBe("string");
    expect(entity.label.length).toBeGreaterThan(0);
  }
}

describe("SemanticProjection contract", () => {
  test("requires sceneId, readingOrder, entities, relations, and optional transcript fields", () => {
    const projection: SemanticProjection = {
      sceneId: "lifecycle",
      readingOrder: ["arrive", "admit"],
      entities: [
        { id: "arrive", label: "Arrive", role: "phase" },
        {
          id: "admit",
          label: "Admit",
          role: "phase",
          evidenceIds: ["ev-1"],
        },
      ],
      relations: [
        {
          id: "r0",
          fromId: "arrive",
          toId: "admit",
          label: "then admit",
          role: "next",
        },
      ],
      transcriptCueId: "cue-admit",
      captions: ["Admission completes."],
    };

    assertProjection(projection);
    const scene: EvaluatedScene = {
      sceneId: projection.sceneId,
      timeMs: 0,
      displayList: {
        commands: [],
        hitRegions: [],
        paintBounds: { x: 0, y: 0, width: 1, height: 1 },
        damageBounds: { x: 0, y: 0, width: 1, height: 1 },
      },
      semantic: projection,
    };
    expect(scene.semantic.sceneId).toBe(scene.sceneId);
  });
});
```

- [ ] **Step 2: Run test to verify it fails on missing `sceneId` in evaluate types**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- test/evaluate/semantic-projection.test.ts
```

Expected: FAIL (TypeScript compile error and/or missing required `sceneId` on evaluate `SemanticProjection`).

- [ ] **Step 3: Implement the canonical types in `evaluate/types.ts`**

Replace the evaluate-local shapes with the Canonical contract block above. Keep `EvaluatedScene` and require `semantic.sceneId === sceneId` by convention (enforced in evaluator tests in Task 2).

- [ ] **Step 4: Export types from the runtime package index**

```typescript
// apps/aiperf-flow/packages/runtime/src/index.ts
export * from "./evaluate/types.js";
```

- [ ] **Step 5: Run test to verify it passes**

```bash
npm test -w @aiperf/flow-runtime -- test/evaluate/semantic-projection.test.ts
```

Expected: PASS

- [ ] **Step 6: Record the canonical-contract checkpoint**

Record changed files and passing commands in the implementation report. Create
a commit only if the user explicitly requests one.

---

### Task 2: Scene evaluator emits the canonical projection

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/scene-evaluator.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/scene-evaluator.test.ts`

**Interfaces:**
- Consumes: `SceneIr`, canonical `SemanticProjection`
- Produces: `evaluateScene(...).semantic` with `sceneId`, `readingOrder`, entities, relations (`fromId`/`toId`), and optional `transcriptCueId` / `captions` when narration is available

- [ ] **Step 1: Update the failing evaluator expectation**

In `scene-evaluator.test.ts`, change the accessibility projection assertion to:

```typescript
test("projects authored accessibility reading order", () => {
  const evaluated = evaluateScene(scene());

  expect(evaluated.semantic).toEqual({
    sceneId: "foundation",
    readingOrder: ["panel", "label", "route"],
    entities: [
      { id: "panel", label: "Request panel" },
      {
        id: "label",
        label: "Request label",
        description: "Names the request",
      },
    ],
    relations: [
      {
        id: "route",
        fromId: "panel",
        toId: "label",
        label: "Request route",
      },
    ],
  });
  expect(evaluated.semantic.sceneId).toBe(evaluated.sceneId);
});
```

Add a second test for narration → captions when `scene.narration` is a non-empty string:

```typescript
test("projects narration into captions when present", () => {
  const evaluated = evaluateScene({
    ...scene(),
    narration: "Request panel receives traffic.",
  });
  expect(evaluated.semantic.captions).toEqual([
    "Request panel receives traffic.",
  ]);
});
```

- [ ] **Step 2: Run tests to verify failure**

```bash
npm test -w @aiperf/flow-runtime -- test/evaluate/scene-evaluator.test.ts
```

Expected: FAIL — missing `sceneId` (and captions) on evaluator output.

- [ ] **Step 3: Update `semanticProjection` in `scene-evaluator.ts`**

```typescript
function semanticProjection(
  scene: SceneIr,
  index: EvaluationIndex,
): SemanticProjection {
  const entities: SemanticEntityProjection[] = [];
  const relations: SemanticRelationProjection[] = [];

  for (const id of scene.accessibility.readingOrder) {
    const node = index.nodes.get(id);
    if (node === undefined) {
      throw new Error(
        `Accessibility reading order references unknown node "${id}".`,
      );
    }
    const common = {
      id,
      label: node.accessibility.label,
      ...(node.accessibility.description === undefined
        ? {}
        : { description: node.accessibility.description }),
    };
    if (node.kind === "connector") {
      relations.push({
        id,
        fromId: node.from.nodeId,
        toId: node.to.nodeId,
        label: node.accessibility.label,
      });
    } else {
      entities.push(common);
    }
  }

  const narration = scene.narration.trim();
  return {
    sceneId: scene.id,
    entities,
    relations,
    readingOrder: [...scene.accessibility.readingOrder],
    ...(narration.length > 0 ? { captions: [narration] } : {}),
  };
}
```

Do not invent `transcriptCueId` from foundation Scene IR yet unless a stable cue id field exists; leave it omitted until timeline/narration IR provides one.

- [ ] **Step 4: Run tests to verify pass**

```bash
npm test -w @aiperf/flow-runtime -- test/evaluate/scene-evaluator.test.ts
```

Expected: PASS

- [ ] **Step 5: Record the evaluator checkpoint**

Record changed files and passing commands in the implementation report. Create
a commit only if the user explicitly requests one.

---

### Task 3: SemanticTwin and focus coordinator consume evaluate types

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/semantic/semantic-twin.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/src/semantic/focus-coordinator.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/semantic/semantic-twin.test.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/test/semantic/focus-coordinator.test.ts`

**Interfaces:**
- Consumes: `SemanticProjection` from `../evaluate/types.js`
- Produces: twin DOM using `fromId`/`toId` and `data-role`; re-export types for existing importers

- [ ] **Step 1: Rewrite twin tests onto the canonical field names**

Update `semantic-twin.test.tsx` fixture:

```typescript
import type { SemanticProjection } from "../../src/evaluate/types.js";
import { SemanticTwin } from "../../src/semantic/semantic-twin.js";

const projection: SemanticProjection = {
  sceneId: "lifecycle",
  readingOrder: ["observe", "arrive", "admit"],
  entities: [
    {
      id: "arrive",
      label: "Arrive",
      description: "Request enters the runtime",
      role: "phase",
    },
    {
      id: "admit",
      label: "Admit",
      description: "Worker admits the request",
      role: "phase",
    },
    {
      id: "observe",
      label: "Observe",
      description: "Observer records metrics",
      role: "phase",
      evidenceIds: ["ev-1"],
    },
  ],
  relations: [
    {
      id: "r0",
      fromId: "arrive",
      toId: "admit",
      role: "next",
      label: "then admit",
    },
    {
      id: "r1",
      fromId: "admit",
      toId: "observe",
      role: "next",
      label: "then observe",
    },
  ],
  transcriptCueId: "cue-admit",
  captions: ["Admission completes before observation."],
};
```

Update relation queries that previously used `data-from` / `data-to` / `data-kind` to `data-from-id` / `data-to-id` / `data-role` (or keep attribute names but bind the new fields—prefer `data-from-id` / `data-to-id` / `data-role` for honesty).

Apply the same fixture field renames in `focus-coordinator.test.ts`.

- [ ] **Step 2: Run twin + focus tests to verify failure**

```bash
npm test -w @aiperf/flow-runtime -- test/semantic/semantic-twin.test.tsx test/semantic/focus-coordinator.test.ts
```

Expected: FAIL (imports still use twin-local `from`/`to`/`kind`).

- [ ] **Step 3: Delete twin-local type definitions; import and re-export**

```typescript
// semantic-twin.tsx
import type {
  SemanticEntityProjection,
  SemanticProjection,
} from "../evaluate/types.js";

export type {
  SemanticEntityProjection,
  SemanticProjection,
  SemanticRelationProjection,
} from "../evaluate/types.js";
```

Update relation rendering:

```tsx
<li
  data-from-id={relation.fromId}
  data-relation-id={relation.id}
  data-role={relation.role}
  data-to-id={relation.toId}
  key={relation.id}
>
  {relation.label ?? `${relation.fromId} → ${relation.toId}`}
</li>
```

Update entity button `data-kind` → `data-role={entity.role}`.

In `focus-coordinator.ts`, change the import to:

```typescript
import type { SemanticProjection } from "../evaluate/types.js";
```

- [ ] **Step 4: Run twin + focus tests to verify pass**

```bash
npm test -w @aiperf/flow-runtime -- test/semantic/semantic-twin.test.tsx test/semantic/focus-coordinator.test.ts
```

Expected: PASS

- [ ] **Step 5: Record the semantic-twin checkpoint**

Record changed files and passing commands in the implementation report. Create
a commit only if the user explicitly requests one.

---

### Task 4: SvgFallback + display-list consumers

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/backends/svg/svg-fallback.tsx` (imports only if needed)
- Modify: `apps/aiperf-flow/packages/runtime/test/backends/svg-fallback.test.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/test/display-list.test.ts`

**Interfaces:**
- Consumes: `EvaluatedScene.semantic` with required `sceneId`
- Produces: unchanged SVG interaction behavior; labels still from `entity.label`

- [ ] **Step 1: Update SVG fallback and display-list fixtures to include `sceneId` on `semantic`**

```typescript
semantic: {
  sceneId: "request-scene",
  readingOrder: ["request"],
  entities: [{ id: "request", label: "Request" }],
  relations: [],
},
```

In `display-list.test.ts`, replace the evaluate-shaped fixture that lacked `sceneId` and used `role` on entities (keep `role`; add `sceneId`).

- [ ] **Step 2: Run SVG + display-list tests**

```bash
npm test -w @aiperf/flow-runtime -- test/backends/svg-fallback.test.tsx test/display-list.test.ts
```

Expected: FAIL until fixtures and any `scene.semantic` assumptions include `sceneId`.

- [ ] **Step 3: Minimal SvgFallback adjustments**

Keep importing `SemanticEntityProjection` from `../../evaluate/types.js`. Prefer `scene.semantic.sceneId` for aria fallbacks (already uses `scene.sceneId`; assert they match in a small unit test):

```typescript
test("semantic sceneId matches evaluated sceneId", () => {
  expect(scene.semantic.sceneId).toBe(scene.sceneId);
});
```

No new React/SVG capability renderers.

- [ ] **Step 4: Re-run tests — expect PASS**

```bash
npm test -w @aiperf/flow-runtime -- test/backends/svg-fallback.test.tsx test/display-list.test.ts
```

- [ ] **Step 5: Record the SVG-fallback checkpoint**

Record changed files and passing commands in the implementation report. Create
a commit only if the user explicitly requests one.

---

### Task 5: Canvas hit metadata resolves labels from the projection

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/backends/canvas/canvas-renderer.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/backends/canvas-renderer.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/backends/backend-conformance.test.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/test/fixtures/evaluated-scene.ts`

**Interfaces:**
- Consumes: `DisplayList` + `SemanticProjection` (or entities map keyed by id)
- Produces: `CanvasSemanticHitRegion` with `entityId`, `label` from entity.label, `focusTarget` defaulting to entity id

- [ ] **Step 1: Write failing conformance fixture using one projection**

Replace the cast-heavy hit-region extras and twin-local hand assembly with:

```typescript
// test/fixtures/evaluated-scene.ts
import { buildDisplayList, type DisplayList } from "../../src/display-list.js";
import type {
  EvaluatedScene,
  SemanticProjection,
} from "../../src/evaluate/types.js";

export const CONFORMANCE_ENTITY = {
  id: "request-a",
  label: "Request A",
  focusTarget: "request-a",
} as const;

export const CONFORMANCE_SELECTION = {
  focusedEntityId: CONFORMANCE_ENTITY.id,
  selectedEntityIds: [CONFORMANCE_ENTITY.id],
} as const;

export const conformanceSemantic: SemanticProjection = {
  sceneId: "backend-conformance",
  readingOrder: [CONFORMANCE_ENTITY.id],
  entities: [
    {
      id: CONFORMANCE_ENTITY.id,
      label: CONFORMANCE_ENTITY.label,
      role: "button",
    },
  ],
  relations: [],
};

export const evaluatedSceneFixture: EvaluatedScene = {
  sceneId: conformanceSemantic.sceneId,
  timeMs: 0,
  displayList: buildDisplayList({
    commands: [
      {
        kind: "path",
        id: "request-shape",
        order: 0,
        path: "M 16 16 H 112 V 48 H 16 Z",
        fill: "#76b900",
        paintBounds: { x: 16, y: 16, width: 96, height: 32 },
        damageBounds: { x: 16, y: 16, width: 96, height: 32 },
      },
    ],
    hitRegions: [
      {
        id: "request-hit",
        semanticId: CONFORMANCE_ENTITY.id,
        order: 0,
        bounds: { x: 16, y: 16, width: 96, height: 32 },
      },
    ],
    paintBounds: { x: 16, y: 16, width: 96, height: 32 },
    damageBounds: { x: 16, y: 16, width: 96, height: 32 },
  }),
  semantic: conformanceSemantic,
};

export const displayListFixture: DisplayList = evaluatedSceneFixture.displayList;
```

Update `backend-conformance.test.tsx` so `SemanticTwin` receives `evaluatedSceneFixture.semantic` directly (no hand-built twin shape), and Canvas hit normalization receives the projection:

```tsx
<SemanticTwin
  focusedEntityId={CONFORMANCE_SELECTION.focusedEntityId}
  onActivate={() => undefined}
  onFocus={() => undefined}
  projection={evaluatedSceneFixture.semantic}
  selectedEntityId={CONFORMANCE_ENTITY.id}
/>
```

- [ ] **Step 2: Run conformance — expect FAIL on missing hit labels**

```bash
npm test -w @aiperf/flow-runtime -- test/backends/backend-conformance.test.tsx
```

Expected: FAIL because `normalizeHitRegions` currently reads loose `label` off hit regions instead of the projection.

- [ ] **Step 3: Thread `SemanticProjection` into Canvas hit normalization**

```typescript
export function renderCanvasDisplayList(
  displayList: DisplayList,
  context: CanvasRenderContext,
  options: CanvasDisplayListOptions & {
    semantic?: SemanticProjection | undefined;
    selectedEntityIds?: readonly string[] | undefined;
    focusedEntityId?: string | null | undefined;
  } = {},
): CanvasDisplayListOutput {
  renderDisplayList(
    context,
    displayList,
    options.devicePixelRatio === undefined
      ? {}
      : { devicePixelRatio: options.devicePixelRatio },
  );
  return {
    hitRegions: normalizeHitRegions(
      displayList,
      options.semantic,
      options.selectedEntityIds ?? [],
      options.focusedEntityId ?? null,
    ),
  };
}

function normalizeHitRegions(
  displayList: DisplayList,
  semantic: SemanticProjection | undefined,
  selectedEntityIds: readonly string[],
  focusedEntityId: string | null,
): readonly CanvasSemanticHitRegion[] {
  const entities = new Map(
    (semantic?.entities ?? []).map((entity) => [entity.id, entity]),
  );
  const selected = new Set(selectedEntityIds);
  return displayList.hitRegions.map((region) => {
    const entity = entities.get(region.semanticId);
    const entityId = region.semanticId;
    const label = entity?.label ?? entityId;
    return {
      entityId,
      label,
      focusTarget: entityId,
      focusable: true,
      selected: selected.has(entityId),
      bounds: region.bounds,
      // focused is selection-state for conformance; keep selected boolean
      // as today. Do not add React types.
    };
  });
}
```

Wire conformance to pass `semantic: evaluatedSceneFixture.semantic` and selection ids. Prefer entity `label` over any leftover loose hit-region fields; remove the fixture cast once labels resolve from the projection.

If existing canvas-renderer unit tests call `renderCanvasDisplayList` without semantic, keep backward-compatible fallbacks (`label = entityId`) and add one dedicated test that proves projection labels win.

- [ ] **Step 4: Run canvas + conformance tests — expect PASS**

```bash
npm test -w @aiperf/flow-runtime -- \
  test/backends/canvas-renderer.test.ts \
  test/backends/backend-conformance.test.tsx \
  test/backends/hit-test.test.ts
```

Expected: PASS — Canvas, twin, and SVG snapshots share entity ids and labels from one projection.

- [ ] **Step 5: Record the Canvas-conformance checkpoint**

Record changed files and passing commands in the implementation report. Create
a commit only if the user explicitly requests one.

---

### Task 6: Package-wide green + drift guard

**Files:**
- Modify: any remaining runtime tests still constructing twin-local `from`/`to`/`kind` shapes (grep-driven)
- Optional note only: cross-link already present in this plan to display-list schema promotion

**Interfaces:**
- Consumes: entire `@aiperf/flow-runtime` test suite
- Produces: zero duplicate `SemanticProjection` type aliases with divergent fields

- [ ] **Step 1: Grep for divergent shapes**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
rg -n "fromId|toId|transcriptCueId|type SemanticProjection|from:|to:|kind\\?:" packages/runtime
```

Forbidden leftovers:

- A second `export type SemanticProjection =` outside `evaluate/types.ts`
- Relation objects using `from` / `to` instead of `fromId` / `toId`
- Entity `kind` fields (use `role`)
- Imports of projection types from `semantic-twin.js` in new code (re-exports may remain temporarily; prefer `evaluate/types.js`)

- [ ] **Step 2: Fix any remaining fixtures; run full runtime package tests**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime
```

Expected: PASS

- [ ] **Step 3: Confirm forbidden surfaces untouched**

```bash
git status -- apps/aiperf-flow/preview
```

Expected: no changes under `preview/**`.

- [ ] **Step 4: Record the unification completion checkpoint**

Record changed files and passing commands in the implementation report. Create
a commit only if the user explicitly requests one.

---

## Out of scope

- Mounting the unified projection in `FlowApp` / site shell (live-cinematic plan).
- Zod `SemanticProjectionIr` schema promotion (display-list plan).
- Capability hybrid evaluators / leaf React or SVG renderers (hybrid-renderers plan).
- Any edits under `apps/aiperf-flow/preview/**`.
- Inventing full transcript cue timelines beyond simple narration → `captions` projection.

## Completion gate

- [ ] Exactly one structural `SemanticProjection` definition in runtime source.
- [ ] Evaluator, twin, SVG fallback, and Canvas hit metadata share `sceneId`, `readingOrder`, `entities`, `relations`, and optional `transcriptCueId` / `captions`.
- [ ] Relations use `fromId` / `toId`; classification uses `role`; entity `label` drives focus/selection strings.
- [ ] Backend conformance uses one fixture projection for all three surfaces.
- [ ] `npm test -w @aiperf/flow-runtime` is green.
- [ ] `preview/**` unchanged; no new React/SVG capability renderers.

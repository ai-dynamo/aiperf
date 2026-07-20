# Scene Flow Layout Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, pure, self-contained two-pass flow-layout engine (`measure` then `position`, flexbox-equivalent semantics) in new files, and migrate the 8 deck-port SDK composites onto it, replacing their hand-rolled coordinate arithmetic (including the Wrap-fix effort's per-formula patches) with generic layout.

**Architecture:** A pure `FlowNode`/`layoutFlow` data model with zero scene/DOM/React dependency (trivially unit-testable in isolation), a small text-measurement adapter bridging it to the existing `wrapTextToWidth`/`measuredWrappedHeight` utilities, then a factory-by-factory migration of `deck-composites.ts`.

**Tech Stack:** Pure TypeScript, Vitest, existing SDK factory conventions.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-20-scene-flow-layout-engine-design.md` — read it first; it has the exact `FlowNode`/`layoutFlow` type signatures to implement verbatim.
- Do NOT modify `apps/explainers/src/core/diagram/capabilities/layout.ts` or anything that already imports it — that module has no production callers today and may be under active work by a different concurrent agent session; this effort builds independent, new code instead, per explicit user direction.
- This engine only changes how SDK factories *compute* geometry at scene-expansion time. Do not modify `SceneRenderer.tsx`'s paint logic in this plan.
- Preserve every `.flow`-facing prop name/shape for the 8 deck composites — this is an internal geometry-computation rewrite, not an API change. `rust-architecture-deck-port.flow` and `sdk-generic-catalog.flow`'s Chapter 8 teaching slides must continue to compile and render without any `.flow`-file edits (verify this explicitly — if a `.flow` edit turns out to be necessary, that's a signal something broke the API contract, stop and investigate rather than patching the `.flow` file to match).
- Commit at file granularity, `git commit --no-verify`, stage only files each task touches, never `git add -A` (shared working tree, other concurrent agents have unrelated in-progress files).
- After every task: `cd apps/explainers && npx vite build && npx vitest run` must pass.
- From Task 3 onward: `npm run assert:sdk-authoring` and `npm run flow-verifier` must show 0 errors; track the `rust-architecture-deck-port.flow` warning count against its state after the Wrap-fix effort's Task 4 (recorded in `.superpowers/sdd/progress.md`) — it should trend down, never up.

---

## File Structure

| File | Responsibility |
|---|---|
| `apps/explainers/src/core/diagram/layout/flow-engine.ts` (new) | `FlowNode`/`FlowBox`/`layoutFlow` — the pure two-pass engine. |
| `apps/explainers/src/core/diagram/layout/flow-engine.test.ts` (new) | Unit tests for the engine in isolation. |
| `apps/explainers/src/core/diagram/layout/text-flow-leaf.ts` (new) | `textFlowLeaf` adapter bridging text measurement into a `FlowNode["measure"]`. |
| `apps/explainers/src/core/diagram/layout/text-flow-leaf.test.ts` (new) | Unit tests for the adapter. |
| `apps/explainers/src/flow/sdk/generic/deck-composites.ts` | Migrated, factory by factory, across Tasks 3-5. |

---

## Task 1: Core `layoutFlow` engine + unit tests

**Files:**
- Create: `apps/explainers/src/core/diagram/layout/flow-engine.ts`
- Create: `apps/explainers/src/core/diagram/layout/flow-engine.test.ts`

**Interfaces:**
- Produces: `FlowConstraint`, `FlowSize`, `FlowNode`, `FlowBox`, `layoutFlow(root, constraint): ReadonlyMap<string, FlowBox>` — exact shapes from the design spec. Consumed by Task 2 (`textFlowLeaf`) and Tasks 3-5 (the composite migrations).

- [ ] **Step 1: Write the failing tests**

```ts
import { describe, expect, it } from "vitest";
import { layoutFlow, type FlowNode } from "./flow-engine.js";

function leaf(id: string, width: number, height: number): FlowNode {
  return { id, measure: () => ({ width, height }) };
}

describe("layoutFlow", () => {
  it("lays out a simple row with gap", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      gap: 10,
      children: [leaf("a", 20, 30), leaf("b", 40, 30), leaf("c", 20, 30)],
    };
    const boxes = layoutFlow(root, { maxWidth: 200 });
    expect(boxes.get("a")).toEqual({ x: 0, y: 0, width: 20, height: 30 });
    expect(boxes.get("b")).toEqual({ x: 30, y: 0, width: 40, height: 30 });
    expect(boxes.get("c")).toEqual({ x: 80, y: 0, width: 20, height: 30 });
    expect(boxes.get("root")!.width).toBe(100); // 20+10+40+10+20
    expect(boxes.get("root")!.height).toBe(30);
  });

  it("distributes free space with justify: space-between in a column", () => {
    const root: FlowNode = {
      id: "root",
      direction: "column",
      justify: "space-between",
      fixedHeight: 100,
      children: [leaf("a", 50, 10), leaf("b", 50, 10)],
    };
    const boxes = layoutFlow(root, { maxWidth: 50 });
    expect(boxes.get("a")!.y).toBe(0);
    expect(boxes.get("b")!.y).toBe(90); // pushed to the bottom, 100 - 10
  });

  it("aligns cross-axis with align: center", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      align: "center",
      fixedHeight: 100,
      children: [leaf("a", 20, 20)],
    };
    const boxes = layoutFlow(root, { maxWidth: 50 });
    expect(boxes.get("a")!.y).toBe(40); // (100 - 20) / 2
  });

  it("nests a row of columns without throwing", () => {
    const col = (id: string): FlowNode => ({
      id,
      direction: "column",
      gap: 4,
      children: [leaf(`${id}-1`, 30, 10), leaf(`${id}-2`, 30, 10)],
    });
    const root: FlowNode = {
      id: "root",
      direction: "row",
      gap: 8,
      children: [col("c1"), col("c2")],
    };
    const boxes = layoutFlow(root, { maxWidth: 200 });
    expect(boxes.get("c1-2")!.y).toBe(14); // 10 + 4
    expect(boxes.get("c2")!.x).toBe(38); // 30 + 8
  });

  it("grows the container when a leaf's measured size exceeds the constraint (auto-grow)", () => {
    const root: FlowNode = {
      id: "root",
      direction: "column",
      children: [leaf("tall", 50, 500)],
    };
    const boxes = layoutFlow(root, { maxWidth: 50, maxHeight: 100 });
    // must not throw/clip — container reports the leaf's real height
    expect(boxes.get("root")!.height).toBe(500);
    expect(boxes.get("tall")!.height).toBe(500);
  });

  it("passes the constrained width down to a leaf's measure function", () => {
    let receivedWidth = -1;
    const root: FlowNode = {
      id: "root",
      direction: "column",
      fixedWidth: 120,
      children: [
        {
          id: "leaf",
          measure: (constraint) => {
            receivedWidth = constraint.maxWidth;
            return { width: constraint.maxWidth, height: 20 };
          },
        },
      ],
    };
    layoutFlow(root, { maxWidth: 999 });
    expect(receivedWidth).toBe(120);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/explainers && npx vitest run src/core/diagram/layout/flow-engine.test.ts`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Implement `flow-engine.ts`**

Implement exactly the `FlowConstraint`/`FlowSize`/`FlowNode`/`FlowBox`/`layoutFlow` shapes from the design spec's code block. Structure the implementation as two clearly-separated internal functions:

```ts
function measureNode(node: FlowNode, constraint: FlowConstraint): FlowSize {
  if (node.measure) {
    const measured = node.measure(constraint);
    return {
      width: node.fixedWidth ?? measured.width,
      height: node.fixedHeight ?? measured.height,
    };
  }
  const direction = node.direction ?? "row";
  const gap = node.gap ?? 0;
  const padding = node.padding ?? 0;
  const children = node.children ?? [];
  const childConstraint: FlowConstraint = {
    maxWidth:
      direction === "row"
        ? Math.max(constraint.maxWidth - padding * 2, 0)
        : Math.max(constraint.maxWidth - padding * 2, 0),
  };
  const sizes = children.map((child) => measureNode(child, childConstraint));
  const mainTotal =
    sizes.reduce(
      (sum, size) => sum + (direction === "row" ? size.width : size.height),
      0,
    ) + gap * Math.max(children.length - 1, 0);
  const crossMax = sizes.reduce(
    (max, size) => Math.max(max, direction === "row" ? size.height : size.width),
    0,
  );
  const width =
    node.fixedWidth ??
    (direction === "row" ? mainTotal + padding * 2 : crossMax + padding * 2);
  const height =
    node.fixedHeight ??
    (direction === "column" ? mainTotal + padding * 2 : crossMax + padding * 2);
  return { width, height };
}

function positionNode(
  node: FlowNode,
  box: FlowBox,
  out: Map<string, FlowBox>,
): void {
  out.set(node.id, box);
  const children = node.children;
  if (children === undefined || children.length === 0) {
    return;
  }
  const direction = node.direction ?? "row";
  const gap = node.gap ?? 0;
  const padding = node.padding ?? 0;
  const justify = node.justify ?? "start";
  const align = node.align ?? "start";
  const contentBox: FlowBox = {
    x: box.x + padding,
    y: box.y + padding,
    width: box.width - padding * 2,
    height: box.height - padding * 2,
  };
  const childConstraint: FlowConstraint = { maxWidth: contentBox.width };
  const sizes = children.map((child) => measureNode(child, childConstraint));
  const mainTotal =
    sizes.reduce(
      (sum, size) => sum + (direction === "row" ? size.width : size.height),
      0,
    ) + gap * Math.max(children.length - 1, 0);
  const contentMain = direction === "row" ? contentBox.width : contentBox.height;
  const freeMain = Math.max(contentMain - mainTotal, 0);
  const extraGap =
    justify === "space-between" && children.length > 1
      ? freeMain / (children.length - 1)
      : 0;
  let cursor =
    (direction === "row" ? contentBox.x : contentBox.y) +
    (justify === "center" ? freeMain / 2 : justify === "end" ? freeMain : 0);
  children.forEach((child, index) => {
    const size = sizes[index]!;
    const mainSize = direction === "row" ? size.width : size.height;
    const crossSize = direction === "row" ? size.height : size.width;
    const contentCross =
      direction === "row" ? contentBox.height : contentBox.width;
    const crossOffset =
      align === "center"
        ? (contentCross - crossSize) / 2
        : align === "end"
          ? contentCross - crossSize
          : 0;
    const resolvedCrossSize = align === "stretch" ? contentCross : crossSize;
    const childBox: FlowBox =
      direction === "row"
        ? {
            x: cursor,
            y: contentBox.y + (align === "stretch" ? 0 : crossOffset),
            width: mainSize,
            height: resolvedCrossSize,
          }
        : {
            x: contentBox.x + (align === "stretch" ? 0 : crossOffset),
            y: cursor,
            width: resolvedCrossSize,
            height: mainSize,
          };
    positionNode(child, childBox, out);
    cursor += mainSize + gap + extraGap;
  });
}

export function layoutFlow(
  root: FlowNode,
  constraint: FlowConstraint,
): ReadonlyMap<string, FlowBox> {
  const rootSize = measureNode(root, constraint);
  const out = new Map<string, FlowBox>();
  positionNode(root, { x: 0, y: 0, ...rootSize }, out);
  return out;
}
```

(This is a complete reference implementation — adjust only if a test in Step 1 reveals an off-by-one or edge case the reference code above got wrong; do not simplify away the two-pass structure.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/explainers && npx vitest run src/core/diagram/layout/flow-engine.test.ts`
Expected: PASS, all 6 tests green. If a test fails, debug the specific formula (most likely culprits: cross-axis centering sign, `space-between` gap distribution, or padding double-counting) — do not weaken the test to match a wrong result.

- [ ] **Step 5: Full build/test check**

```bash
cd apps/explainers
npx vite build
npx vitest run
```

- [ ] **Step 6: Commit**

```bash
git add src/core/diagram/layout/flow-engine.ts src/core/diagram/layout/flow-engine.test.ts
git commit --no-verify -m "feat(explainers): add pure two-pass flow-layout engine"
```

---

## Task 2: `textFlowLeaf` text-measurement adapter + unit tests

**Files:**
- Create: `apps/explainers/src/core/diagram/layout/text-flow-leaf.ts`
- Create: `apps/explainers/src/core/diagram/layout/text-flow-leaf.test.ts`

**Interfaces:**
- Consumes: `wrapTextToWidth` (from `../text-metrics.js`, Wrap-fix Task 1), `FlowNode["measure"]` signature (from Task 1 of this plan).
- Produces: `textFlowLeaf(text, fontSize, weight, lineHeightRatio?): FlowNode["measure"]`.

- [ ] **Step 1: Write the failing tests**

```ts
import { describe, expect, it } from "vitest";
import { textFlowLeaf } from "./text-flow-leaf.js";
import { wrapTextToWidth } from "../text-metrics.js";

describe("textFlowLeaf", () => {
  it("reports height proportional to wrapped line count", () => {
    const text = "one two three four five six seven eight nine ten";
    const measure = textFlowLeaf(text, 14, "normal");
    const size = measure({ maxWidth: 80 });
    const expectedLines = wrapTextToWidth(text, 80, 14, "normal").length;
    expect(size.height).toBeCloseTo(expectedLines * 14 * 1.3, 5);
    expect(size.width).toBe(80);
  });

  it("reports a single line's height for short text", () => {
    const measure = textFlowLeaf("short", 14, "normal");
    const size = measure({ maxWidth: 400 });
    expect(size.height).toBeCloseTo(14 * 1.3, 5);
  });

  it("respects a custom lineHeightRatio", () => {
    const measure = textFlowLeaf("short", 14, "normal", 1.5);
    const size = measure({ maxWidth: 400 });
    expect(size.height).toBeCloseTo(14 * 1.5, 5);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/explainers && npx vitest run src/core/diagram/layout/text-flow-leaf.test.ts`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Implement `text-flow-leaf.ts`**

```ts
import { wrapTextToWidth } from "../text-metrics.js";
import type { FlowConstraint, FlowNode, FlowSize } from "./flow-engine.js";

const DEFAULT_LINE_HEIGHT_RATIO = 1.3;

/**
 * Bridges Task 1's `wrapTextToWidth` measurer into the flow engine's
 * `FlowNode["measure"]` contract: given a width constraint, wraps `text`
 * and reports the height that many lines actually need, using the same
 * line-height convention `SceneRenderer` applies at paint time (kept in
 * one place to avoid the measure/paint divergence risk flagged during
 * the wrap-fix effort).
 */
export function textFlowLeaf(
  text: string,
  fontSize: number,
  weight: "normal" | "bold" = "normal",
  lineHeightRatio: number = DEFAULT_LINE_HEIGHT_RATIO,
): NonNullable<FlowNode["measure"]> {
  return (constraint: FlowConstraint): FlowSize => {
    const lines = wrapTextToWidth(text, constraint.maxWidth, fontSize, weight);
    const lineCount = Math.max(lines.length, 1);
    return {
      width: constraint.maxWidth,
      height: lineCount * fontSize * lineHeightRatio,
    };
  };
}
```

Check whether `SceneRenderer.tsx`'s already-established `lineHeight`
constant (from Wrap-fix Task 2/3) is `fontSize * 1.3` on the SAME
(already-scaled) `fontSize` this function receives, or on an unscaled
authored value — read the current `SceneRenderer.tsx` text branch and
`measuredWrappedHeight` in `text-metrics.ts` to confirm which convention
is authoritative, and make sure `textFlowLeaf`'s caller (Tasks 3-5) passes
an already-consistently-scaled `fontSize` the same way. Document whichever
convention you confirm, in a one-line comment, so a future reader doesn't
have to re-derive it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/explainers && npx vitest run src/core/diagram/layout/text-flow-leaf.test.ts`
Expected: PASS.

- [ ] **Step 5: Full build/test check**

```bash
cd apps/explainers
npx vite build
npx vitest run
```

- [ ] **Step 6: Commit**

```bash
git add src/core/diagram/layout/text-flow-leaf.ts src/core/diagram/layout/text-flow-leaf.test.ts
git commit --no-verify -m "feat(explainers): add textFlowLeaf text-measurement adapter for the flow engine"
```

---

## Task 3: Migrate row-based composites (`stepChain`, `numberedSequence`)

**Files:**
- Modify: `apps/explainers/src/flow/sdk/generic/deck-composites.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/deck-composites.test.ts` (update any hardcoded coordinate assertion that legitimately changes)

**Interfaces:**
- Consumes: `layoutFlow`/`FlowNode` (Task 1), `textFlowLeaf` (Task 2).

- [ ] **Step 1: Read the current `stepChain`/`numberedSequence` factories in full**, including whatever Wrap-fix Task 4 already changed in them (auto-grow patches this migration supersedes).

- [ ] **Step 2: Rewrite `numberedSequenceFactory`** to build one `FlowNode` (`direction: "column"`, `gap` matching the current visual spacing) whose children are per-row `FlowNode`s (each `direction: "row"`, containing a fixed-size chip leaf and a `textFlowLeaf`-measured detail-text leaf), call `layoutFlow` once with the composite's authored width as the constraint, and turn the resulting `FlowBox` map into the actual `core.rect`/`core.text` scene nodes at their resolved coordinates — removing the Task 4 running-offset accumulator entirely (the engine now does this).

- [ ] **Step 3: Rewrite `stepChainFactory`'s column-mode path** the same way; row mode can either also route through the engine (a single-level row of fixed-size step boxes) or keep its simpler existing per-box growth from Task 4 if that's already correct and rewriting it would be pure churn — use your judgment, document the choice.

- [ ] **Step 4: Run tests**

```bash
cd apps/explainers
npx vite build
npx vitest run src/flow/sdk/generic/deck-composites.test.ts
npx vitest run
```
Update any hardcoded coordinate/height assertion that now legitimately differs (verify by hand it's correct — trace through the engine's math for that specific fixture).

- [ ] **Step 5: Verify against the full app**

```bash
cd apps/explainers
npm run assert:sdk-authoring
npm run flow-verifier
```
Report the `rust-architecture-deck-port.flow` warning count versus its state after Wrap-fix Task 4.

- [ ] **Step 6: Visual check**

Start `npm run dev` in the background, Playwright-check the `sdk-generic-catalog.flow` teaching slides for `sdk.stepChain`/`sdk.numberedSequence`, and the `rust-architecture-deck-port.flow` slides that use them (Orientation, Observer sequence, Flow diagram, at minimum), confirm no regression versus Wrap-fix Task 4's already-improved renders. Stop the dev server.

- [ ] **Step 7: Commit**

```bash
git add src/flow/sdk/generic/deck-composites.ts src/flow/sdk/generic/deck-composites.test.ts
git commit --no-verify -m "refactor(explainers): migrate stepChain/numberedSequence onto the flow-layout engine"
```

---

## Task 4: Migrate grid-based composites (`compareGrid`, `cardGrid`)

**Files:**
- Modify: `apps/explainers/src/flow/sdk/generic/deck-composites.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/deck-composites.test.ts`

**Interfaces:**
- Consumes: `layoutFlow`/`FlowNode` (Task 1), `textFlowLeaf` (Task 2).

- [ ] **Step 1: Read the current `compareGrid`/`cardGrid` factories in full**, including Wrap-fix Task 4's per-cell height patches this migration supersedes.

- [ ] **Step 2: Rewrite both as a row-of-columns `FlowNode` tree**

Model an N-column grid as a top-level `direction: "row"` node whose
children are `columns` count `direction: "column"` nodes (one per grid
column), each containing the cells that fall in that column — OR, simpler
and matching the "uniform row height across the grid" requirement from
the design spec: build it as a single top-level `direction: "column"` of
row-`FlowNode`s (each row a `direction: "row"` of that row's cells),
so the engine's own per-row max-height-of-siblings behavior (from Task
1's `measureNode` cross-axis max, or simply not stretching and instead
reading each row's own measured max height) naturally produces uniform
row heights without a separate "find the tallest cell" pass. Pick
whichever tree shape most directly reuses Task 1's engine without new
engine features; document the shape you chose.

- [ ] **Step 3: Run tests**

```bash
cd apps/explainers
npx vite build
npx vitest run src/flow/sdk/generic/deck-composites.test.ts
npx vitest run
```

- [ ] **Step 4: Verify against the full app**

```bash
cd apps/explainers
npm run assert:sdk-authoring
npm run flow-verifier
```
Report the warning count versus Task 3 of this plan.

- [ ] **Step 5: Visual check**

Start `npm run dev` in the background, Playwright-check `sdk-generic-catalog.flow`'s `sdk.compareGrid`/`sdk.cardGrid` teaching slides and at least 4 `rust-architecture-deck-port.flow` slides using them across different chapters (e.g. Thesis, Crate topology, a component-reference slide, Invariants). Stop the dev server.

- [ ] **Step 6: Commit**

```bash
git add src/flow/sdk/generic/deck-composites.ts src/flow/sdk/generic/deck-composites.test.ts
git commit --no-verify -m "refactor(explainers): migrate compareGrid/cardGrid onto the flow-layout engine"
```

---

## Task 5: Migrate remaining composites (`sectionDivider`, `bigStat`, `nodeTree`, `timelineAxis`) and remove now-dead Task 4 patches

**Files:**
- Modify: `apps/explainers/src/flow/sdk/generic/deck-composites.ts`
- Modify: `apps/explainers/src/flow/sdk/generic/deck-composites.test.ts`

**Interfaces:**
- Consumes: `layoutFlow`/`FlowNode` (Task 1), `textFlowLeaf` (Task 2).

- [ ] **Step 1: For each of the 4 remaining composites**, apply the engine to their one free-text field (subtitle/description/orderNote) the same way Wrap-fix Task 4 did with hand-rolled math — replace that hand-rolled single-box growth with a one-node `layoutFlow` call (even a single leaf benefits from going through the same code path, for consistency, though the payoff is smaller than the row/grid composites).

- [ ] **Step 2: Grep the whole file for any leftover Wrap-fix-Task-4-era manual line-count/stride computation that Tasks 3-5 of this plan have now made dead code**, and remove it — the goal is that `deck-composites.ts` has exactly one way to size text-bearing content (the flow engine), not two competing systems left over from the incremental patch effort.

- [ ] **Step 3: Run tests**

```bash
cd apps/explainers
npx vite build
npx vitest run src/flow/sdk/generic/deck-composites.test.ts
npx vitest run
```

- [ ] **Step 4: Commit**

```bash
git add src/flow/sdk/generic/deck-composites.ts src/flow/sdk/generic/deck-composites.test.ts
git commit --no-verify -m "refactor(explainers): migrate remaining deck composites onto the flow engine, remove dead patch code"
```

---

## Task 6: Full-app final verification and 49-slide comparison

**Files:** none (verification only)

- [ ] **Step 1: Full verification suite**

```bash
cd apps/explainers
npm run build
npx vitest run
npm run assert:no-mentalmodel-registry
npm run assert:sdk-authoring
npm run flow-verifier
```
Expected: all 5 exit 0.

- [ ] **Step 2: Final warning-count comparison**

Report the `rust-architecture-deck-port.flow` `SCENE_VIEWPORT_ESCAPE`/
`SCENE_ABSOLUTE_SIBLING_OVERLAP` count at three points: before the
wrap-fix effort (195, from that effort's final review), after wrap-fix
Task 4 (recorded in `.superpowers/sdd/progress.md`), and now. It should
have decreased at each step.

- [ ] **Step 3: Full 49-slide Playwright walkthrough**

Start `npm run dev` in the background, walk all 49 slides of
`#/rust-architecture-deck-port`, screenshot each, and confirm every
previously-known overflow/shift instance across the whole session (Clock/
Drivers collisions, Flow-diagram clipping, right-column edge nicks,
Dynosim/mock-server overflow, BigStat/Invariants overflow — all in
`.superpowers/sdd/progress.md`) is now resolved by generic engine behavior
rather than point patches. Also spot-check 3-4 other existing decks
(untouched by this whole session) to confirm no visual regression from
the underlying `deck-composites.ts` rewrite. Stop the dev server.

- [ ] **Step 4: No commit required** — this task is verification-only. If Step 1 or 3 surfaces a defect, return to the relevant earlier task, fix it there, and re-run that task's own verification before returning here.

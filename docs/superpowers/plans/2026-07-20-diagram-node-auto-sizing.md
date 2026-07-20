# Diagram Node Auto-Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make shared SDK diagram nodes grow to fit their labels with scale-aware metrics, and reflow children inside layout containers with fixed gaps.

**Architecture:** Introduce one pure `text-metrics` module as the source of truth for `SCENE_TEXT_SCALE` and width estimation. Capability layout resolves intrinsic leaf sizes (chip, panel, note, stepper) treating authored dimensions as minimums, then existing stack/rail/lane layouts place children with unchanged gaps. SceneRenderer and chrome consume the same metrics so painted text and boxes stay paired.

**Tech Stack:** TypeScript, Vitest, existing `capabilities/layout.ts` + `registry.ts`, SceneRenderer

## Global Constraints

- Explicit authored width/height are minimums unless the node clips overflow.
- Preserve authored/default gaps; reflow only inside layout containers.
- Do not move absolute-positioned top-level siblings.
- Deterministic char-width estimation only — no DOM `measureText`.
- Layout estimators must honor `SCENE_TEXT_SCALE = 0.9`.
- No per-deck `.flow` coordinate edits; no shell CSS changes in this plan.

## File Structure

| File | Responsibility |
|---|---|
| `apps/explainers/src/core/diagram/text-metrics.ts` | Shared scale, padding/band constants, `estimateTextWidth` |
| `apps/explainers/src/core/diagram/text-metrics.test.ts` | Unit tests for estimation and scale |
| `apps/explainers/src/core/diagram/capabilities/layout.ts` | Intrinsic leaf layouts + scale-aware stepper |
| `apps/explainers/src/core/diagram/capabilities/chrome.ts` | Chrome placement using shared metrics |
| `apps/explainers/src/core/diagram/SceneRenderer.tsx` | Import scale helper from `text-metrics` |
| `apps/explainers/src/flow/dev-tools/verify-geometry.ts` | Resolve capability layout when collecting bounds |
| Tests under `capabilities/layout.test.ts`, `SceneRenderer.sdk-primitives.test.tsx` | Regression coverage |

---

### Task 1: Shared scale-aware text metrics

**Files:**
- Create: `apps/explainers/src/core/diagram/text-metrics.ts`
- Create: `apps/explainers/src/core/diagram/text-metrics.test.ts`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx` (replace local `SCENE_TEXT_SCALE` / `scaledSceneFontSize`)

**Interfaces:**
- Consumes: nothing new
- Produces:
  - `export const SCENE_TEXT_SCALE = 0.9`
  - `export const DEFAULT_SCENE_FONT_SIZE = 14`
  - `export const CHAR_WIDTH = 6.2`
  - `export const INSET = 8`, `TITLE_HEIGHT = 22`, `DETAIL_HEIGHT = 20`, `CHIP_PAD_X = 24`, `STEPPER_MIN_CHIP_WIDTH = 72`, `STEPPER_CHIP_HEIGHT = 26`
  - `export function estimateTextWidth(text: string, fontSize: number, weight?: "normal" | "bold"): number`
  - `export function scaledSceneFontSize(value: unknown, fallback?: number): number`
  - `export function stepperChipWidth(label: string, index: number): number`

- [ ] **Step 1: Write the failing metrics tests**

```ts
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  SCENE_TEXT_SCALE,
  estimateTextWidth,
  scaledSceneFontSize,
  stepperChipWidth,
} from "./text-metrics.js";

describe("scene text metrics", () => {
  it("exports the shared scene text scale", () => {
    expect(SCENE_TEXT_SCALE).toBe(0.9);
  });

  it("scales authored and default font sizes", () => {
    expect(scaledSceneFontSize(20)).toBe(18);
    expect(scaledSceneFontSize(undefined)).toBe(12.6);
  });

  it("estimates width with the scene text scale", () => {
    expect(estimateTextWidth("authoritative", 11, "bold")).toBe(
      Math.ceil(13 * 6.2 * 0.9),
    );
  });

  it("sizes stepper chips from numbered labels under the text scale", () => {
    expect(stepperChipWidth("layout", 0)).toBe(
      Math.max(72, Math.ceil("1. layout".length * 6.2 * 0.9) + 24),
    );
  });
});
```

- [ ] **Step 2: Run the metrics test and confirm it fails**

Run:

```bash
npm --prefix apps/explainers test -- src/core/diagram/text-metrics.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `text-metrics.ts`**

```ts
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic, scale-aware text metrics for scene layout and rendering.

export const SCENE_TEXT_SCALE = 0.9;
export const DEFAULT_SCENE_FONT_SIZE = 14;
export const CHAR_WIDTH = 6.2;
export const BOLD_CHAR_WIDTH = 6.2;
export const INSET = 8;
export const TITLE_HEIGHT = 22;
export const DETAIL_HEIGHT = 20;
export const CHIP_PAD_X = 24;
export const STEPPER_MIN_CHIP_WIDTH = 72;
export const STEPPER_CHIP_HEIGHT = 26;
export const STEPPER_CHIP_PAD = 24;

export function scaledSceneFontSize(
  value: unknown,
  fallback = DEFAULT_SCENE_FONT_SIZE,
): number {
  const fontSize =
    typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return fontSize * SCENE_TEXT_SCALE;
}

export function estimateTextWidth(
  text: string,
  fontSize: number,
  weight: "normal" | "bold" = "normal",
): number {
  const unit = weight === "bold" ? BOLD_CHAR_WIDTH : CHAR_WIDTH;
  const ratio = fontSize / 11;
  return Math.ceil(text.length * unit * ratio * SCENE_TEXT_SCALE);
}

export function stepperChipWidth(label: string, index: number): number {
  const text = `${index + 1}. ${label}`;
  return Math.max(
    STEPPER_MIN_CHIP_WIDTH,
    estimateTextWidth(text, 11, "bold") + STEPPER_CHIP_PAD,
  );
}
```

- [ ] **Step 4: Point SceneRenderer at the shared helpers**

In `SceneRenderer.tsx`, delete the local `SCENE_TEXT_SCALE` / `DEFAULT_SCENE_FONT_SIZE` / `scaledSceneFontSize` definitions and import `scaledSceneFontSize` from `./text-metrics.js`. Keep call sites unchanged.

- [ ] **Step 5: Run metrics + renderer font tests**

```bash
npm --prefix apps/explainers test -- src/core/diagram/text-metrics.test.ts src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
```

Expected: all pass.

---

### Task 2: Intrinsic chip / panel / note sizing and scale-aware stepper layout

**Files:**
- Modify: `apps/explainers/src/core/diagram/capabilities/layout.ts`
- Modify: `apps/explainers/src/core/diagram/capabilities/chrome.ts`
- Modify: `apps/explainers/src/core/diagram/capabilities/layout.test.ts`
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.sdk-primitives.test.tsx` (stepper width assertion if chrome widths change)

**Interfaces:**
- Consumes: `estimateTextWidth`, `stepperChipWidth`, `INSET`, `TITLE_HEIGHT`, `DETAIL_HEIGHT`, `CHIP_PAD_X`, `STEPPER_*` from `text-metrics.ts`
- Produces:
  - `resolveChipLayout(node, children): CapabilityLayout`
  - `resolvePanelLayout(node, children): CapabilityLayout` (also used for note-like title/detail chrome)
  - `resolveStepperLayout` uses shared `stepperChipWidth`
  - Registry entries for `core.chip`, `core.panel`, `core.note` (and `core.header` if title-driven height needs a minimum)

- [ ] **Step 1: Write failing layout tests**

Add to `layout.test.ts`:

```ts
  it("expands a semantic stepper using scale-aware chip widths", () => {
    const stepper = node("steps", "core.stepper", 160, 90, {
      props: { steps: ["layout", "slots", "timeline"], linked: true },
      style: { gap: 16 },
    });
    const layout = resolveCapabilityLayout(stepper, []);
    const expected =
      stepperChipWidth("layout", 0) +
      stepperChipWidth("slots", 1) +
      stepperChipWidth("timeline", 2) +
      16 * 2;
    expect(layout.bounds).toEqual({ x: 0, y: 0, width: expected, height: 90 });
  });

  it("grows a chip to fit its label while treating authored size as a minimum", () => {
    const short = node("chip", "core.chip", 84, 26, {
      props: { label: "ok" },
    });
    const long = node("chip", "core.chip", 84, 26, {
      props: { label: "authoritative" },
    });
    expect(resolveCapabilityLayout(short, []).bounds.width).toBe(84);
    expect(resolveCapabilityLayout(long, []).bounds.width).toBeGreaterThan(84);
    expect(resolveCapabilityLayout(long, []).bounds.width).toBe(
      Math.max(84, estimateTextWidth("authoritative", 11, "bold") + 24),
    );
  });

  it("grows a panel to fit title and detail bands", () => {
    const panel = node("panel", "core.panel", 100, 40, {
      props: { title: "Profile source panel", detail: "authoritative metrics" },
    });
    const layout = resolveCapabilityLayout(panel, []);
    expect(layout.bounds.width).toBeGreaterThan(100);
    expect(layout.bounds.height).toBeGreaterThanOrEqual(40);
  });

  it("reflows a rail after chips auto-grow, preserving gap", () => {
    const rail = node("rail", "layout.rail", 160, 22, {
      style: { direction: "row", gap: 8 },
    });
    const children = [
      node("a", "core.chip", 84, 26, { props: { label: "authoritative" } }),
      node("b", "core.chip", 84, 26, { props: { label: "ok" } }),
    ];
    // Intrinsic children must be resolved before rail placement in the test
    // by using resolveCapabilityLayout on each child, then placing via rail
    // with those resolved geometries — or by extending registry so rail
    // children are intrinsic-resolved first (preferred implementation).
    const a = resolveCapabilityLayout(children[0]!, []).bounds;
    const b = resolveCapabilityLayout(children[1]!, []).bounds;
    const layout = resolveCapabilityLayout(rail, [
      { ...children[0]!, geometry: a },
      { ...children[1]!, geometry: b },
    ]);
    expect(layout.childGeometries[1]?.x).toBe(a.width + 8);
    expect(layout.bounds.width).toBe(a.width + 8 + b.width);
  });
```

Import `estimateTextWidth` and `stepperChipWidth` from `../text-metrics.js`. Replace the old stepper test that expects `width: 279` with the scale-aware version above.

- [ ] **Step 2: Run layout tests and confirm the new cases fail**

```bash
npm --prefix apps/explainers test -- src/core/diagram/capabilities/layout.test.ts
```

Expected: chip/panel/scale-aware stepper assertions fail (chip still width 84; stepper still 279).

- [ ] **Step 3: Implement intrinsic leaf layouts**

In `layout.ts`:

1. Import shared metrics; delete local `STEPPER_CHAR_WIDTH` / pad duplicates; call shared `stepperChipWidth`.
2. Add:

```ts
export function resolveChipLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const label =
    (typeof node.props?.label === "string" && node.props.label) ||
    (typeof node.props?.text === "string" && node.props.text) ||
    node.accessibility?.label ||
    "";
  const width = Math.max(
    authored.width,
    label.length > 0 ? estimateTextWidth(label, 11, "bold") + CHIP_PAD_X : authored.width,
  );
  const height = Math.max(authored.height, STEPPER_CHIP_HEIGHT);
  return {
    bounds: { ...authored, width, height },
    childGeometries: children.map(geometryOf),
  };
}

export function resolvePanelLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const title =
    (typeof node.props?.title === "string" && node.props.title) ||
    (typeof node.props?.label === "string" && node.props.label) ||
    "";
  const detail =
    (typeof node.props?.detail === "string" && node.props.detail) ||
    (typeof node.props?.caption === "string" && node.props.caption) ||
    "";
  const titleWidth =
    title.length > 0 ? estimateTextWidth(title, 14, "bold") + INSET * 2 : 0;
  const detailWidth =
    detail.length > 0 ? estimateTextWidth(detail, 11.5, "normal") + INSET * 2 : 0;
  const contentHeight =
    INSET * 2 +
    (title.length > 0 ? TITLE_HEIGHT : 0) +
    (detail.length > 0 ? DETAIL_HEIGHT + 4 : 0);
  return {
    bounds: {
      ...authored,
      width: Math.max(authored.width, titleWidth, detailWidth),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries: children.map(geometryOf),
  };
}
```

3. Register `core.chip` → `resolveChipLayout`, `core.panel` → `resolvePanelLayout`, `core.note` → `resolvePanelLayout` (caption via `detail`/`caption` props) in `LAYOUT_CAPABILITIES`.

4. In `indexSceneNodes` / rail layout path: when resolving a container, resolve each child's capability layout before using child widths (if not already). Confirm `SceneRenderer` `indexSceneNodes` already re-resolves children; if rail currently uses authored child geometry only, update container resolvers or the indexer so child intrinsic bounds feed the parent. Prefer fixing the indexer once: for each child, set geometry from `resolveCapabilityLayout(child, grandchild…).bounds` before parent layout.

- [ ] **Step 4: Align chrome.ts with shared metrics**

Replace local `INSET` / `TITLE_HEIGHT` / `STEPPER_*` / `stepWidth` with imports from `text-metrics.ts`. Keep `resolveSemanticChrome` placement logic; use shared `stepperChipWidth` for stepper boxes.

- [ ] **Step 5: Run layout + renderer tests**

```bash
npm --prefix apps/explainers test -- src/core/diagram/capabilities/layout.test.ts src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
```

Expected: pass. Update any stepper width hard-codes in renderer tests to match scale-aware widths (label text assertions stay the same).

---

### Task 3: Verifier layout parity and package verification

**Files:**
- Modify: `apps/explainers/src/flow/dev-tools/verify-geometry.ts`
- Test: add or extend a focused test near existing geometry helpers if one exists; otherwise add assertions in `layout.test.ts` that document the shared contract used by verification
- Optionally modify: `apps/explainers/scripts/flow-verifier/geometry.mjs` only if it can import the same pure logic without a large rewrite — otherwise document that browser `verify-geometry.ts` is the parity path for this task and leave `.mjs` for a follow-up

**Interfaces:**
- Consumes: `resolveCapabilityLayout` from `capabilities/registry.js`
- Produces: collected node bounds that reflect intrinsic expansion and container reflow

- [ ] **Step 1: Locate bound-collection in `verify-geometry.ts`**

Find where authored `geometry`/`layout` is read for each node. Change that path so container and leaf bounds come from:

```ts
import { resolveCapabilityLayout } from "../../core/diagram/capabilities/registry.js";

function resolvedBounds(node: SceneNodeLike, children: readonly SceneNodeLike[]) {
  return resolveCapabilityLayout(node, children).bounds;
}
```

Pass already-resolved child nodes (with updated geometries) into parents, matching SceneRenderer indexing order.

- [ ] **Step 2: Add a small regression assertion**

If no dedicated verify-geometry test file exists, add one layout-contract comment test in `layout.test.ts` that rail+chip intrinsic widths match the formula used above (already covered in Task 2). For verify-geometry, prefer a tiny unit test file only if helpers are exported; do not invent a large harness.

- [ ] **Step 3: Run full explainer tests and build**

```bash
npm --prefix apps/explainers test
npm --prefix apps/explainers run build
```

Expected: exit 0.

- [ ] **Step 4: Diagnostics**

Check edited files for new linter issues; fix any introduced by this work.

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Shared scale-aware metrics module | Task 1 |
| Authored size as minimum; grow to fit | Task 2 |
| Chip / panel / note / stepper intrinsic sizing | Task 2 |
| Container reflow with fixed gaps | Task 2 (rail test) |
| Absolute siblings unchanged | Non-goal; no code path moves them |
| SceneRenderer uses shared scale | Task 1 |
| Verifier uses same layout rules | Task 3 |
| No deck edits / no shell CSS | Global constraints |

## Self-review notes

- Stepper expected width changes from `279` to the scale-aware sum — tests must not keep the old constant.
- Rail reflow depends on resolving child intrinsic layouts before parent placement; fix the indexer or pass pre-resolved child geometries in tests and production the same way.
- Do not attempt full semantic-IR migration of card factories beyond wiring `core.panel` / `core.chip` / `core.note` layout hooks needed for sizing.

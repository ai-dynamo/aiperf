# Scene Flow Layout Engine — Design

## Purpose

`apps/explainers`' SDK component factories (most concretely, the 8
deck-port composites in `deck-composites.ts`) currently compute every
child's `x`/`y`/`width`/`height` by hand, in each factory, as a series of
arithmetic expressions (row stride × index, column width × index, etc.).
Wrap-fix Tasks 1-4 patched the worst symptom (text overflowing/clipping)
by teaching those hand-rolled formulas to consult measured wrapped-line
counts, but the underlying architecture still has no general notion of
"lay out these children along an axis, sized to their own content,
centered/stretched appropriately" — every new composite re-derives this
from scratch, and every fix has been a point patch to one formula.

This spec adds a small, pure, framework-agnostic two-pass flow-layout
engine (`measure` then `position`, the same model real flow/flexbox
layout uses) as new, self-contained code, and migrates the 8 deck
composites onto it. It deliberately does **not** modify
`core/diagram/capabilities/layout.ts` (an existing, apparently-unused
`resolveStackLayout`/`resolveGridLayout` module with no production
callers, and — per a concurrent-agent code trace this session — possibly
under active work by another session right now) to avoid any collision;
this is a fresh, independent implementation, per explicit user direction.

## Built (this spec, not yet implemented — plan at
`docs/superpowers/plans/2026-07-20-scene-flow-layout-engine.md`)

### Core engine — `apps/explainers/src/core/diagram/layout/flow-engine.ts` (new module, new directory)

Pure data model, no React/DOM/SVG dependency:

```ts
export type FlowConstraint = { maxWidth: number; maxHeight?: number };
export type FlowSize = { width: number; height: number };

export type FlowNode = {
  id: string;
  // Leaf nodes measure themselves given a width constraint (e.g. text
  // wrapping to that width); container nodes' measure is derived from
  // children by the engine and this field is omitted for containers.
  measure?: (constraint: FlowConstraint) => FlowSize;
  direction?: "row" | "column"; // containers only; default "row"
  gap?: number;
  justify?: "start" | "center" | "end" | "space-between";
  align?: "start" | "center" | "end" | "stretch";
  padding?: number;
  children?: readonly FlowNode[];
  // A fixed dimension always wins over measured/derived size on that axis.
  fixedWidth?: number;
  fixedHeight?: number;
};

export type FlowBox = { x: number; y: number; width: number; height: number };

/**
 * Two-pass layout: measures every node bottom-up (leaves via their own
 * `measure`, containers by summing/maxing children along their axis),
 * then positions every node top-down from the root's resolved box.
 * Returns every node's box in coordinates relative to the root's origin
 * (caller translates into absolute scene coordinates).
 */
export function layoutFlow(
  root: FlowNode,
  constraint: FlowConstraint,
): ReadonlyMap<string, FlowBox>;
```

`layoutFlow` internally does exactly two recursive passes — `measureNode`
(bottom-up: containers derive size from children per `direction`,
respecting `fixedWidth`/`fixedHeight` overrides) and `positionNode`
(top-down: given a resolved box, place children along the main axis with
`gap`/`justify`, align the cross axis with `align`, recursing into any
child that is itself a container). No node's position or size in either
pass depends on later parts of the same pass — this keeps the algorithm
`O(n)` and side-effect-free, so it's trivially unit-testable without any
scene/DOM/factory scaffolding.

### Text-measure adapter

A small helper, `textFlowLeaf(text, fontSize, weight, lineHeightRatio)`,
in the same `layout/` directory, wraps Task 1's `wrapTextToWidth` and
Task 3's line-height convention into a `FlowNode["measure"]` function:
given a `maxWidth` constraint, wraps the text and returns
`{ width: maxWidth, height: lineCount * fontSize * lineHeightRatio }` —
reusing the exact same measurement Tasks 1-3 already use elsewhere, so
this engine's text sizing stays consistent with what `SceneRenderer`
actually paints (the same consistency risk Task 3's review flagged and
verified once already).

### Migration — the 8 deck composites

Each of `sectionDivider`, `stepChain`, `bigStat`, `compareGrid`,
`numberedSequence`, `timelineAxis`, `nodeTree`, `cardGrid` in
`deck-composites.ts` is rewritten to: build a `FlowNode` tree describing
its intended structure (a row of step boxes, a grid of cards, etc.) using
`textFlowLeaf` for any text-bearing child, call `layoutFlow` once, then
turn the resulting `FlowBox` map into concrete `core.rect`/`core.text`
scene-node geometries — replacing the hand-rolled stride/offset
arithmetic each factory currently has (including the Wrap-fix Task 4
patches, which this migration supersedes/removes since the engine now
does that job generically).

## Explicitly out of scope

- `core/diagram/capabilities/layout.ts` and anything already using it —
  untouched, to avoid the collision risk noted above.
- Migrating the generic (non-deck-port) SDK components
  (`sdk.section`/`sdk.panel`/`sdk.card`/`sdk.toolbar`/`sdk.splitPane`/
  `sdk.mediaObject`, the Task 3 standalone prose primitives) onto this
  engine — they keep their current (now wrap/auto-grow-aware, per Wrap-fix
  Tasks 2-3) behavior. A future spec can migrate them once this engine has
  proven itself on the 8 composites.
- Any change to `SceneRenderer.tsx`'s paint logic — this engine only
  changes how factories *compute* geometry at scene-expansion time; paint
  time is unaffected (Task 2's auto-wrap already handles final-mile text
  rendering within whatever box the factory decides on).

## Verification

- Unit tests for `layoutFlow` covering: a simple row of 3 fixed-size
  leaves with gap, a column with `justify: "space-between"`, a grid-like
  nested row-of-columns, `align: "stretch"` vs `"center"`, and a leaf whose
  `measure` reports a size larger than its container's constraint (must
  not throw or infinite-loop — container grows to accommodate, mirroring
  the "auto-grow" requirement from the wrap-fix effort).
- Unit tests for `textFlowLeaf` confirming it produces the same line count
  as `wrapTextToWidth` directly, for a few representative strings/widths.
- Per-composite tests in `deck-composites.test.ts` continue to pass
  (updated where a hardcoded coordinate legitimately changes because the
  engine computes a more correct value than the old hand-rolled formula —
  verified by hand each time, not rubber-stamped).
- Full app regression: `npm run build`, `npx vitest run`,
  `npm run assert:sdk-authoring`, `npm run flow-verifier` (0 errors; the
  `SCENE_VIEWPORT_ESCAPE`/`SCENE_ABSOLUTE_SIBLING_OVERLAP` warning count on
  `rust-architecture-deck-port.flow` should be lower than after Wrap-fix
  Task 4, since the engine handles auto-grow generically rather than via
  Task 4's per-formula patches).
- Live Playwright re-walk of all 49 `rust-architecture-deck-port` slides,
  confirming no regression versus Wrap-fix Task 4's already-improved
  state, and ideally further improvement on any slide Task 4's patches
  didn't fully resolve.

## Source anchors

- `apps/explainers/src/core/diagram/layout/flow-engine.ts` (new)
- `apps/explainers/src/core/diagram/layout/flow-engine.test.ts` (new)
- `apps/explainers/src/core/diagram/layout/text-flow-leaf.ts` (new)
- `apps/explainers/src/flow/sdk/generic/deck-composites.ts` (migrated)
- `apps/explainers/decks-flow/rust-architecture-deck-port.flow` (re-verified, not necessarily re-authored — the engine changing internal factory math should not require changing the `.flow` DSL calls themselves, since prop names/shapes are unchanged)

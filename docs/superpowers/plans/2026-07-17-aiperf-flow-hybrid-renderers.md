# AIPerf Flow Hybrid Evaluators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the six landed P0 hybrid contributions—glyph-run, span-map,
semantic-morph, segment-strip, waterfall, and queue—inside the shared scene
evaluator. `leaf.correspondence-tween` may later improve semantic-morph motion
but is not required for its backend-neutral contribution.

**Architecture:** Hybrid components are backend-neutral evaluators, not React or SVG renderers. The shared scene evaluator merges their fragments and produces one `DisplayList` and one `SemanticProjection`; Canvas, the always-mounted semantic HTML twin, and the simplified SVG/HTML fallback consume only those two products.

**Depends on:**
- [`2026-07-17-aiperf-flow-live-cinematic-runtime.md`](2026-07-17-aiperf-flow-live-cinematic-runtime.md) for the mounted runtime, integer virtual clock, backend composition, focus synchronization, and pause/resume behavior.
- [`2026-07-17-aiperf-flow-display-list.md`](2026-07-17-aiperf-flow-display-list.md)
  for evaluated-scene, display-list, hit-region, deterministic-ordering, and
  quality-profile contracts.
- [`2026-07-17-aiperf-flow-semantic-projection.md`](2026-07-17-aiperf-flow-semantic-projection.md)
  for the canonical runtime semantic projection.
- [`2026-07-17-aiperf-flow-p0-core-components.md`](2026-07-17-aiperf-flow-p0-core-components.md) for `ComponentNodeIr`, capability descriptors, descriptor-id binding, and the five pure leaves.

**Tech Stack:** TypeScript strict mode, Vitest, `@aiperf/flow-schema`, `@aiperf/flow-runtime`.

## Global Constraints

- The six P0 hybrid contributions are `core.glyph-run`, `core.span-map`,
  `core.semantic-morph`, `core.segment-strip`, `viz.waterfall`, and
  `viz.queue`.
- Hybrids contribute immutable `layout`, `semantic`, and `display` fragments into the existing shared scene evaluator. They do not own a renderer or create a second evaluator.
- Canvas, semantic HTML twin, and SVG/HTML fallback accept `DisplayList` plus `SemanticProjection` only. They must not inspect `ComponentNodeIr`, call leaves, or dispatch by hybrid capability id.
- Do not create React/SVG-primary capability renderers. In particular, do not create `capabilities/*.tsx` implementations for these hybrids and do not add JSX-returning `render` methods as their primary path.
- Do not modify `apps/aiperf-flow/preview/**`.
- Leaves remain pure TypeScript in:
  - `packages/runtime/src/leaves/glyph-measure.ts`
  - `packages/runtime/src/leaves/span-interval.ts`
  - `packages/runtime/src/leaves/segment-strip-layout.ts`
  - `packages/runtime/src/leaves/queue-policy.ts`
  - `packages/runtime/src/leaves/waterfall-nest-layout.ts`
- Leaves return immutable measurement, interval, layout, or policy data. They never import React, return JSX, mutate Canvas/SVG/DOM state, or read wall time.
- `node.semanticModel` owns semantic identity. `node.layoutPlan` and leaf output may resolve geometry but may not mint or rewrite semantic ids.
- Evaluation uses integer virtual time from the shared evaluation context. Direct seek and continuous playback to the same time must produce equal outputs.
- Unknown capabilities and malformed required attachments fail closed with capability and node ids in the diagnostic.
- Every task follows TDD: add a focused failing test, verify the expected failure, implement the minimum backend-neutral behavior, then rerun the focused suite.
- Activate the project environment before every command:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
```

- Do not create git commits unless explicitly requested.

---

## Contract and ownership map

```text
ComponentNodeIr
  + capability descriptor
  + integer EvaluationContext
        ↓
P0 Hybrid Evaluator
  ├─→ LayoutFragment
  ├─→ SemanticProjectionFragment
  └─→ DisplayListFragment
        ↓
Shared SceneEvaluator
  ├─→ DisplayList
  └─→ SemanticProjection
        ↓
  ├─→ Canvas backend
  ├─→ semantic HTML twin
  └─→ SVG/HTML fallback
```

The display-list plan owns evaluated-scene/display-list product types and
deterministic merging. The semantic-projection plan owns the single runtime
projection. The live-cinematic plan owns the three consumers and mounted viewer
behavior. This plan owns only the six hybrid contributions and their
registration with the shared evaluator.

## Current implementation baseline

The contribution seam is landed and integrated:

- `evaluate/contributions/` contains all six component contributions;
- `evaluate/registry.ts` binds capability IDs;
- `evaluate/merge-contributions.ts` deterministically merges products;
- `scene-evaluator.ts` evaluates component nodes through that registry;
- contribution and component-scene-evaluator tests cover the mounted path.

Tasks 1–8 describe landed outcomes. Preserve the existing
`contributions/`/registry/merge structure; do not create parallel
`component-fragment.ts` or `component-evaluators.ts` modules. Remaining work is
hardening diagnostics, deterministic merge invariants, quality behavior,
cross-backend parity, and the verification gate.

---

## Task 1: Harden the landed hybrid contribution seam

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/types.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/registry.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/merge-contributions.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/scene-evaluator.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/types.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/component-scene-evaluator.test.ts`

**Interfaces:**
- Consumes: `ComponentNodeIr`, canonical evaluation context, and capability id resolved through `resolveCapabilityId`.
- Produces: `ComponentEvaluationFragment` containing immutable `layout`, `semantic`, and `display` fragments compatible with the canonical contracts from the display-list plan.
- Produces: `ComponentEvaluator = (node, context) => ComponentEvaluationFragment`.
- Extends the shared `SceneEvaluator`; it does not introduce a parallel component-rendering pipeline.

- [ ] **Step 1: Write failing contract tests**
  - Assert fragment merge order is deterministic by scene source order and stable fragment id.
  - Assert duplicate semantic ids, dangling display hit-region ids, non-finite geometry, and unknown capability ids fail closed.
  - Assert layout overrides preserve ids authored in `semanticModel`.
  - Assert the evaluator receives integer virtual time and has no wall-clock input.

- [ ] **Step 2: Run the focused test and verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- component-fragment
```

Expected: FAIL because the hybrid contribution seam is not registered with the shared evaluator.

- [ ] **Step 3: Implement the minimal contribution and merge contracts**
  - Add no React, JSX, DOM, Canvas, or SVG imports.
  - Reuse canonical `DisplayList`, hit-region, and `SemanticProjection` types; do not duplicate them.
  - Make the shared evaluator the only operation that merges component fragments into frame products.

- [ ] **Step 4: Rerun the focused test**

Expected: PASS.

---

## Task 2: Evaluate `core.glyph-run`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/glyph-run.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/glyph-run-contribution.test.ts`
- Reuse: `apps/aiperf-flow/packages/runtime/src/leaves/glyph-measure.ts`

**Contract:**
- The leaf measures grapheme boundaries and advances.
- The hybrid maps those measurements and authored attachments into stable grapheme layout entries, semantic reading-order entries, text draw commands, and hit regions.

- [ ] **Step 1: Write a failing `"café 🚀"` golden test**
  - Assert grapheme ids and reading order match `semanticModel`.
  - Assert layout-plan geometry overrides measured geometry without changing ids.
  - Assert display commands and hit regions reference the same semantic ids.

- [ ] **Step 2: Run the focused test and verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- glyph-run
```

Expected: PASS for landed behavior; new hardening assertions fail before their
corresponding invariant is implemented.

- [ ] **Step 3: Implement the pure hybrid evaluator**
  - Call `glyph-measure` only for missing derived geometry.
  - Return fragments only; do not return React nodes or SVG elements.

- [ ] **Step 4: Rerun the focused test**

Expected: PASS.

---

## Task 3: Evaluate `core.span-map`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/span-map.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/span-map-contribution.test.ts`
- Reuse: `apps/aiperf-flow/packages/runtime/src/leaves/span-interval.ts`

**Contract:**
- The leaf indexes and validates source/target intervals.
- The hybrid emits span and relation layout, semantic entities/relations, routed display commands, coverage diagnostics, and corresponding hit regions.

- [ ] **Step 1: Write a failing token-span golden test**
  - Assert mapped relation endpoints use authored semantic ids.
  - Assert required-cover gaps appear in both semantic and display fragments.
  - Assert no geometry override changes relation identity.

- [ ] **Step 2: Run the focused test and verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- span-map
```

Expected: PASS for landed behavior; new hardening assertions fail before their
corresponding invariant is implemented.

- [ ] **Step 3: Implement the pure hybrid evaluator**
  - Keep interval analysis in `span-interval.ts`.
  - Emit backend-neutral path and hit-region data only.

- [ ] **Step 4: Rerun the focused test**

Expected: PASS.

---

## Task 4: Evaluate `core.segment-strip`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/segment-strip.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/segment-strip-contribution.test.ts`
- Reuse: `apps/aiperf-flow/packages/runtime/src/leaves/segment-strip-layout.ts`

**Contract:**
- The leaf computes deterministic segment bounds, clipping, and continuation state.
- The hybrid emits segment layout, ordered segment semantics, rectangle/text display commands, and one hit region per inspectable segment.

- [ ] **Step 1: Write a failing seven-segment golden test**
  - Assert deterministic total width and segment order.
  - Assert truncation and continuation are present in both semantic and display fragments.
  - Assert a supplied layout plan overrides bounds but not segment ids or roles.

- [ ] **Step 2: Run the focused test and verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- segment-strip
```

Expected: PASS for landed behavior; new hardening assertions fail before their
corresponding invariant is implemented.

- [ ] **Step 3: Implement the pure hybrid evaluator**
  - Keep packing and clipping calculations in the leaf.
  - Return immutable fragments with shared style tokens only.

- [ ] **Step 4: Rerun the focused test**

Expected: PASS.

---

## Task 5: Evaluate `viz.waterfall`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/waterfall.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/waterfall-contribution.test.ts`
- Reuse: `apps/aiperf-flow/packages/runtime/src/leaves/waterfall-nest-layout.ts`

**Contract:**
- The leaf computes nested interval and lane geometry.
- The hybrid emits interval/point layout, lane-grouped semantics, bar/point/label display commands, and semantic hit regions at the requested integer virtual time.

- [ ] **Step 1: Write failing waterfall beat tests**
  - Assert nested intervals, points, and open spans retain semantic ids.
  - Assert direct seek and stepped playback to the same integer time produce equal fragments.
  - Assert reduced motion changes decorative motion metadata only.

- [ ] **Step 2: Run the focused test and verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- waterfall
```

Expected: PASS for landed behavior; new hardening assertions fail before their
corresponding invariant is implemented.

- [ ] **Step 3: Implement the pure hybrid evaluator**
  - Keep nesting and lane placement in the leaf.
  - Derive visible beat state solely from evaluation context time.

- [ ] **Step 4: Rerun the focused test**

Expected: PASS.

---

## Task 6: Evaluate `viz.queue`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/contributions/queue.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/queue-contribution.test.ts`
- Reuse: `apps/aiperf-flow/packages/runtime/src/leaves/queue-policy.ts`

**Contract:**
- The leaf computes the immutable queue event series for the selected policy.
- The hybrid projects queue state at integer virtual time into waiting/service layout, request/event semantics, lane/chip display commands, and request hit regions.

- [ ] **Step 1: Write failing queue beat tests**
  - Assert waiting, serving, and departed states at named beats.
  - Assert direct seek equals continuous playback at each beat.
  - Assert every visible request command and hit region resolves to a semantic entity.

- [ ] **Step 2: Run the focused test and verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- queue
```

Expected: PASS for landed behavior; new hardening assertions fail before their
corresponding invariant is implemented.

- [ ] **Step 3: Implement the pure hybrid evaluator**
  - Keep scheduling policy and event-series generation in the leaf.
  - Derive frame state solely from the shared evaluation context.

- [ ] **Step 4: Rerun the focused test**

Expected: PASS.

---

## Task 7: Harden the six-entry P0 contribution registry

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/registry.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/component-scene-evaluator.test.ts`

**Interfaces:**
- Preserves registrations for exactly `core.glyph-run`, `core.span-map`,
  `core.semantic-morph`, `core.segment-strip`, `viz.waterfall`, and
  `viz.queue`.
- Preserves descriptor manifest sorting, duplicate-id rejection, and `resolveCapabilityId` dispatch.
- Does not add React `render` methods to P0 entries.

- [ ] **Step 1: Extend registry tests**
  - Assert all six and only the six P0 hybrid ids are registered.
  - Assert duplicate ids and unknown ids fail closed.
  - Assert each registration exposes fragment evaluation and no JSX renderer.

- [ ] **Step 2: Run the focused test and verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- create-p0-evaluators
```

Expected: PASS for the landed set; new hardening assertions fail before their
corresponding diagnostic or invariant is implemented.

- [ ] **Step 3: Preserve all six contributions in the shared scene evaluator**
  - Keep foundation React/SVG fallback compatibility outside this evaluator set.
  - Do not route P0 hybrids through `RuntimeCapability.render`.

- [ ] **Step 4: Rerun the focused test**

Expected: PASS.

---

## Task 8: Prove backend input isolation and parity

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/test/backends/p0-hybrid-conformance.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/backends/backend-conformance.test.tsx`
- Reuse: shared Canvas, semantic-twin, and SVG-fallback test fixtures owned by the live-cinematic plan.

**Interfaces:**
- Consumes: only the `DisplayList` and `SemanticProjection` returned by the shared evaluator.
- Proves: all three consumers preserve semantic ids, focus targets, reading order, and selection without access to source hybrid nodes or leaves.

- [ ] **Step 1: Write a failing conformance matrix**
  - Evaluate one fixture for each of the six P0 hybrids.
  - Pass only `DisplayList` and `SemanticProjection` to Canvas, twin, and SVG/HTML fallback harnesses.
  - Assert matching semantic ids and hit/focus targets across consumers.
  - Assert backend modules do not import `ComponentNodeIr`, P0 leaves, or P0 evaluator modules.

- [ ] **Step 2: Run the conformance test and verify failure**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- p0-hybrid-conformance
```

Expected: FAIL until every backend consumes the shared products without capability-specific access.

- [ ] **Step 3: Complete shared-product wiring only**
  - Adapt backend entry points if required by the live-cinematic contract.
  - Add no capability-specific Canvas, React, SVG, or DOM rendering branches.

- [ ] **Step 4: Rerun conformance tests**

Expected: PASS for all six hybrids and all three consumers.

---

## Task 9: Verification gate

- [ ] **Step 1: Run runtime tests**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime
```

Expected: PASS.

- [ ] **Step 2: Run the Flow workspace check**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm run flow:check
```

Expected: PASS.

- [ ] **Step 3: Verify architectural boundaries**
  - No files under `apps/aiperf-flow/preview/**` changed.
  - No P0 hybrid evaluator is a `.tsx` file or imports React.
  - All five leaves remain pure `.ts` modules and have focused unit tests.
  - Canvas, semantic twin, and SVG/HTML fallback consume only `DisplayList` and `SemanticProjection`.
  - The shared scene evaluator is the sole merger of hybrid layout, semantic, and display fragments.
  - No sixth P0 hybrid, parallel scene evaluator, or capability-specific backend branch was introduced.

- [ ] **Step 4: Run documentation validation**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust
/usr/bin/python3 tools/check_docs_current.py
```

Expected: PASS.

---

## Dependency order

```text
display-list contracts ∥ live-cinematic runtime ∥ five pure leaves
  → Task 1 shared contribution seam
  → Tasks 2–6 five P0 hybrid evaluators
  → Task 7 evaluator registration
  → Task 8 backend input isolation and parity
  → Task 9 verification
```

Tasks 2–6 may proceed independently after Task 1. Task 8 must wait for the shared products and backend contracts owned by the cited plans.

## Out of scope

- React/SVG-primary P0 capability components
- `capabilities/*.tsx` files for the five hybrids
- Capability-specific Canvas, semantic-twin, or SVG/HTML rendering branches
- Changes under `apps/aiperf-flow/preview/**`
- A new scene evaluator, display-list format, semantic-projection shape, clock, or backend stack
- `core.semantic-morph` or any P0 hybrid beyond the five listed in this plan
- Compiler, language, stdlib symbol, browser-shell, or document-specific work
- Code implementation or git commits as part of this rewrite

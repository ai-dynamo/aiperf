# AIPerf Flow P0 Core Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the composable `.flow` stdlib model by implementing P0 substrate (schema, binding, leaves), hand-authored IR fixtures, and three flagship wrappers (`TokenSpanMorph`, `PromptSegmentComposer`, `RequestLifecycleWaterfall`).

**Architecture:** Extend strict Flow IR v1 with optional `capabilityId`,
semantic model, and layout-plan attachments; dispatch runtime by descriptor id;
implement five narrow deterministic leaves; lower components into the shared
evaluated-scene and display-list contracts; project the same semantics through
Canvas 2D, the semantic HTML twin, and simplified SVG/HTML fallback; unblock
`.flow` authoring with symbol grammar + compiler in a later increment while
validating via golden IR first.

**Tech Stack:** TypeScript, Zod 4, Chevrotain, Vitest, Canvas 2D, React/HTML
semantic twin and shell, SVG/HTML fallback, `Intl.Segmenter` for grapheme
measurement.

## Global Constraints

- Authors commit only `.flow` for domain visuals; no document-specific React/TS/CSS.
- Semantic entity/relation ids survive layout overrides (identity before geometry).
- Unknown props/fields fail closed at compile and IR validation boundaries.
- Runtime binds by `capabilityId` / component id, not `core.${kind}` alone.
- Hybrid components use at most one leaf each; leaves are golden-testable pure functions.
- Leaves emit immutable semantic, layout, analysis, or display-plan data. They
  never return React nodes, mutate a Canvas context, or read wall time.
- Every P0 component provides Canvas draw commands and hit regions, semantic
  HTML twin output, and a simplified SVG/HTML fallback from one evaluated
  semantic scene.
- The deterministic virtual clock is the only animation time source. Direct
  seek and continuous playback to the same beat produce equal semantic state.
- Interaction pauses playback by default; resume continues from the exact beat
  and restores the authored camera according to policy.
- Degraded quality may reduce decorative effects but not semantics, captions,
  narration cues, focus, evidence, or interaction.
- Verify flagship fidelity at 3840×2160 as well as responsive desktop, tablet,
  and mobile containers.
- Activate `.venv` before repo commands: `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Verify with `cd apps/aiperf-flow && npm run flow:check`.
- Do not create git commits unless the user explicitly requests them.

---

## Research synthesis (2026-07-17)

The foundation slice, semantic/layout schema, descriptor binding, six P0
component capability descriptors, five deterministic leaves, compiler
pipeline, runtime-local evaluated scene/display list, Canvas backend, semantic
twin, SVG fallback, and three schema-valid flagship IR fixtures are landed and
tested. `FlowApp` mounts the cinematic Canvas/SVG/twin stack. Three flat P0
flagship `.flow` authoring stubs exist under `stdlib/`; the planned namespaced
`stdlib/{core,viz}` tree, barrels, and per-component fixtures do not. Symbol
grammar remains partial. All six hybrid contributions are integrated through
the shared scene evaluator; correspondence-tween remains an optional deferred
motion leaf for `core.semantic-morph`.

Tasks 1–9 describe landed substrate. Task 8 is superseded in detail by the
backend-neutral hybrid-evaluators plan; its contribution seam is integrated and
must be hardened in place. An unwired experimental `glyph-run.tsx` is not the
target architecture. Tasks 10–12 remain delivery work.

**Scope split:** preview UI/fixture work is owned by a separate agent. This plan
covers schema, language, compiler, runtime leaves/capabilities, stdlib, and
IR-backed flagship proofs only—do not make preview chrome the owner of scene,
timeline, camera, or interaction state. P0 must extend the landed
backend-neutral substrate and must not create a competing renderer.

---

## Task 1: Schema substrate — capability id, semantic model, layout plan

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/capability-id.ts`
- Create: `apps/aiperf-flow/packages/schema/src/semantic-model.ts`
- Create: `apps/aiperf-flow/packages/schema/src/layout-plan.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/ir.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Create: `apps/aiperf-flow/packages/schema/vitest.config.ts`
- Create: `apps/aiperf-flow/packages/schema/test/capability-id.test.ts`
- Create: `apps/aiperf-flow/packages/schema/test/semantic-model.test.ts`
- Create: `apps/aiperf-flow/packages/schema/test/layout-plan.test.ts`
- Create: `apps/aiperf-flow/packages/schema/test/ir.test.ts`

**Interfaces:**
- Produces: `resolveCapabilityId(node)`, `SemanticModelIr`, `LayoutPlanIr`, `ComponentNodeIr`, extended `parseFlowIr`.

- [ ] **Step 1:** Failing tests for `resolveCapabilityId`, semantic model strict parse, layout plan strict parse, component node with explicit `capabilityId`.
- [ ] **Step 2:** Implement modules and extend `ir.ts` union with `kind: "component"`.
- [ ] **Step 3:** `npm test -w @aiperf/flow-schema` passes.

---

## Task 2: Runtime descriptor binding

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/renderer.tsx`
- Create: `apps/aiperf-flow/packages/runtime/test/capability-binding.test.tsx`

**Interfaces:**
- Consumes: `resolveCapabilityId` from `@aiperf/flow-schema`.

- [ ] **Step 1:** Test that a `component` node with `capabilityId: "core.span-map"` dispatches to registered capability, not `core.component`.
- [ ] **Step 2:** Replace `registry.require(\`core.${nodeKind(node)}\`)` with `registry.require(resolveCapabilityId(node))`.
- [ ] **Step 3:** Foundation nodes without `capabilityId` still resolve to `core.rect`, etc.

---

## Task 3: P0 capability descriptors

**Files:**
- Modify: `apps/aiperf-flow/packages/schema/src/capability.ts`
- Create: `apps/aiperf-flow/packages/schema/test/p0-capabilities.test.ts`

- [ ] Register descriptors for: `core.glyph-run`, `core.span-map`,
  `core.semantic-morph`, `core.segment-strip`, `viz.queue`, `viz.waterfall`,
  plus five leaf ids.
- [ ] Manifest sort + duplicate-id tests stay green.

---

## Task 4: Pure leaf — `leaf.glyph-measure`

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/leaves/glyph-measure.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/leaves/glyph-measure.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/fixtures/glyph-run-cafe-rocket.json`

- [ ] Golden fixture: `"café 🚀"` grapheme boundaries per flagship agent spec.
- [ ] Use `Intl.Segmenter` with `{ granularity: "grapheme" }`.

---

## Task 5: Pure leaf — `leaf.span-interval`

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/leaves/span-interval.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/leaves/span-interval.test.ts`

- [ ] Overlap index + `requireCover` validation for TokenSpanMorph edges fixture.

---

## Task 6: Pure leaf — `core.segment-strip.layout`

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/leaves/segment-strip-layout.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/leaves/segment-strip-layout.test.ts`

- [ ] Golden layout plan for PromptSegmentComposer seven-segment fixture, seed 42.

---

## Task 7: Pure leaves — `viz.queue.policy` and `viz.waterfall.nest-layout`

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/leaves/queue-policy.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/leaves/waterfall-nest-layout.ts`
- Create: matching tests under `packages/runtime/test/leaves/`

---

## Task 8: Hybrid capability evaluators + P0 registry

The detailed implementation authority for this task is
[`2026-07-17-aiperf-flow-hybrid-renderers.md`](2026-07-17-aiperf-flow-hybrid-renderers.md).
That plan supersedes the earlier idea of per-capability `.tsx` renderers.
Capabilities emit evaluated-scene, display-list, and semantic-projection
fragments; shared backends render them.

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/capabilities/` tree
- Create: `apps/aiperf-flow/packages/runtime/src/create-p0-registry.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/compose-stdlib.test.ts`

- [ ] Evaluate hand-authored IR for each proof wrapper using stdlib capability
  ids into backend-neutral semantic and display plans.
- [ ] Render each plan through Canvas 2D and simplified SVG/HTML; render its
  semantic HTML twin and synchronize hit-region selection with semantic focus.
- [ ] Prove direct-seek parity and pause-to-explore/resume behavior at named
  wrapper beats.

---

## Task 9: Foundation compiler (minimal)

**Files:**
- Create: `apps/aiperf-flow/packages/compiler/src/{link,validate,lower,pack,index}.ts`
- Create: `apps/aiperf-flow/packages/compiler/test/compile.test.ts`
- Create: `apps/aiperf-flow/examples/foundation/request-flow.flow`

- [ ] Lower foundation grammar to canonical IR; then extend for component nodes.

---

## Task 10: Symbol grammar + stdlib `.flow`

**Files:**
- Extend: `packages/language/src/{tokens,ast,parser,formatter}.ts`
- Create: `apps/aiperf-flow/stdlib/core/*.flow`, `stdlib/viz/*.flow`
- Create: `apps/aiperf-flow/examples/p0/*.flow` + `*.expected.json`

---

## Task 11: Flagship integration + preview alignment

**Files:**
- Modify: `apps/aiperf-flow/preview/fixture.ts` (canonical `roots`, timeline, interactions)
- Create: `apps/aiperf-flow/e2e/p0-expressiveness.spec.ts`

- [ ] Three wrappers compile (or load golden IR) and render without
  document-specific runtime code.
- [ ] Each wrapper passes Canvas, semantic-twin, SVG/HTML fallback,
  accessibility, reduced-motion, direct-seek, and pause/resume assertions.
- [ ] `RequestLifecycleWaterfall` passes a 3840×2160 visual-fidelity snapshot
  and desktop/tablet/mobile composition snapshots.

---

## Task 12: Verification gate

- [ ] `npm run flow:check` green
- [ ] Canvas/display-list determinism and semantic-twin parity suites green
- [ ] Reference and degraded frame-time/memory budgets recorded and green
- [ ] Progress ledger updated at `.superpowers/sdd/progress.md`

---

## Dependency order

```text
Task 1 → Task 2 → Task 3 → Tasks 4–7 (leaves, parallel) → Task 8
Task 9 ∥ Tasks 4–8 (compiler can lag IR proofs)
Task 10 → Task 11 → Task 12
```

## Execution options

1. **Subagent-driven (recommended)** — one subagent per task, review between tasks.
2. **Inline** — execute Tasks 1–4 in this session, checkpoint before compiler/stdlib.

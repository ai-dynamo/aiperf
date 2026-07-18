# AIPerf Flow Display-List and Evaluated-Scene Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define and land backend-neutral `EvaluatedSceneIr`, `DisplayListIr`, `HitRegionIr`, and quality-tier contracts in `@aiperf/flow-schema`, plus pure runtime evaluators and seek-parity conformance tests that prove the live-cinematic invariant: integer timeline time is the sole animation source and direct seek equals continuous playback.

**Architecture:** Extend the schema package with strict Zod IR for the evaluated semantic scene and its derived display products. Add a pure evaluation pipeline in `@aiperf/flow-runtime` that consumes validated Flow IR, capability descriptors, layout plans, and viewer inputs and emits immutable evaluated-scene and display-list snapshots. Quality tiers gate decorative draw commands and sampling density without removing semantic entities, relations, captions, narration cues, focus, or interaction. Seek-parity tests golden-hash evaluated outputs at named beats; they do not depend on Canvas, React, or preview shell code.

**Tech Stack:** TypeScript strict mode, Zod 4, Vitest, `@aiperf/flow-schema`, `@aiperf/flow-runtime`.

## Global Constraints

- Ground every contract in the live-cinematic stdlib rules from [`2026-07-17-aiperf-flow-core-components-design.md`](../specs/2026-07-17-aiperf-flow-core-components-design.md): identity before geometry, one integer clock, semantic twin parity, fidelity without semantic loss, and the display contract (draw commands, paint bounds, hit regions, damage bounds, quality tiers, deterministic ordering).
- Align with approved architecture in [`2026-07-17-aiperf-flow-design.md`](../specs/2026-07-17-aiperf-flow-design.md): scene evaluator → display-list builder → backend renderers; wall-clock deltas advance playback but never become scene state.
- Scope is `@aiperf/flow-schema` and `@aiperf/flow-runtime` only. Do not modify `preview/`, browser shell, Canvas renderer, semantic-twin React projection, or SVG fallback in this plan.
- Depends on P0 schema substrate (`SemanticModelIr`, `LayoutPlanIr`, `ComponentNodeIr`, `resolveCapabilityId`) from [`2026-07-17-aiperf-flow-p0-core-components.md`](2026-07-17-aiperf-flow-p0-core-components.md). If those modules are absent, implement only the minimal imports this plan declares and do not fork parallel semantic or layout types.
- Leaves and hybrid capabilities emit display-list fragments and hit regions; they never return React nodes, mutate a rendering context, or read wall time.
- Unknown fields fail closed at parse boundaries. All IR objects carry version metadata and deterministic serialization order.
- Normal and packed Flow IR must produce identical evaluated-scene and display-list semantics.
- Reference quality profile targets 60 evaluated frames per second on the documented reference device; degraded profile may target 30 fps by reducing particles, blur, shadow quality, and sampling density only.
- Reduced-motion, high-contrast, and no-depth variants are orthogonal profile axes that may suppress motion paths and depth cues but must keep correspondence tables, entities, and interaction complete.
- Activate `.venv` before repo commands: `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Verify with `cd apps/aiperf-flow && npm run flow:check`.
- Do not create git commits unless the user explicitly requests them.

---

## Research synthesis (2026-07-17)

The mounted `FlowApp` now composes the runtime-local evaluated scene and display
list with Canvas, SVG fallback, the always-mounted semantic twin, subtitles,
and focus coordination. Those contracts and unit tests live under
`packages/runtime/src/{evaluate,backends,semantic}` and
`packages/runtime/src/display-list.ts`. They are not yet promoted to strict
versioned `*Ir` contracts in `@aiperf/flow-schema` or governed by a complete
shared quality-tier contract.

This plan consolidates and promotes the landed runtime-local evaluated-scene,
display-list, hit-region, and quality types rather than creating a second
pipeline. Runtime `SemanticProjection` unification is owned by
[`2026-07-17-aiperf-flow-semantic-projection.md`](2026-07-17-aiperf-flow-semantic-projection.md);
this plan may later promote that canonical runtime shape to strict schema IR
without renaming fields or creating a third shape.

This plan owns schema promotion for evaluated scenes, display lists, hit
regions, and quality policy plus pure evaluation conformance. It modifies the
existing runtime-local implementation in place.
The live-cinematic and hybrid-renderer plans own mounted-app integration,
capability contributions, and final viewer behavior.

---

## Contract overview

```text
Flow IR + capability descriptors + layout plans
  + integer timeline time + viewport profile + interaction log
        ↓
  SceneEvaluator → EvaluatedSceneIr
        ↓
  DisplayListBuilder → DisplayListIr + HitRegionIr[]
        ↓
  QualityPolicyApplier (reference | degraded × motion/contrast/depth axes)
        ↓
  Backend renderers (out of scope for this plan)
```

**EvaluatedSceneIr** is the canonical backend-neutral snapshot: semantic model, resolved layout geometry, camera, timeline target states, interaction overlay, accessibility outline, and declared quality profile inputs.

**DisplayListIr** is an ordered, backend-neutral command stream with global paint bounds, per-command local bounds, damage regions, draw-order keys, and content hash.

**HitRegionIr** maps pointer and keyboard targets to stable semantic ids, z-order, roles, and semantic-twin focus ids without requiring pixel inspection.

**Quality tiers** are descriptor-driven policies that mark commands as required-semantic or decorative-optional and record which decorative families were suppressed.

---

## Task 1: `EvaluatedSceneIr` schema

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/evaluated-scene.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Create: `apps/aiperf-flow/packages/schema/test/evaluated-scene.test.ts`

**Interfaces:**
- Consumes: `SemanticModelIr`, `LayoutPlanIr`, `SceneIr`, `ComponentNodeIr`, existing strict JSON value helpers.
- Produces: `EvaluatedSceneIr`, `EvaluatedNodeIr`, `CameraStateIr`, `TimelineTargetSnapshotIr`, `InteractionOverlayIr`, `AccessibilityOutlineIr`, `parseEvaluatedSceneIr`, `hashEvaluatedSceneIr`.

- [ ] **Step 1:** Failing tests for strict parse, unknown-field rejection,
  required non-negative safe-integer `timeMs`, semantic entity/relation
  preservation independent of geometry overrides, and deterministic hash
  stability across key ordering permutations in input objects.
- [ ] **Step 2:** Define `EvaluatedSceneIr` with fields: `version`, `timeMs`, `sceneId`, `semanticModel`, `nodes` (resolved bounds + style tokens + capability id + source node id), `camera`, `timelineTargets`, `interaction`, `accessibilityOutline`, `profileInputs` (viewport size, device pixel ratio, responsive variant id).
- [ ] **Step 3:** Export from schema index; `npm test -w @aiperf/flow-schema` passes for evaluated-scene tests.

---

## Task 2: `HitRegionIr` schema

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/hit-region.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Create: `apps/aiperf-flow/packages/schema/test/hit-region.test.ts`

**Interfaces:**
- Consumes: strict bounds primitives shared with layout plan geometry types.
- Produces: `HitRegionIr`, `HitRegionRole`, `HitRegionCollectionIr`, `parseHitRegionIr`, `mergeHitRegionCollections`, `indexHitRegionsAtPoint`.

- [ ] **Step 1:** Failing tests for axis-aligned bounds validation, duplicate semantic-id policy (same id may appear once per z-plane; overlaps resolve by deterministic z-order then source order), and roles enum (`select`, `inspect`, `scrub`, `focus`, `compare`, `navigate`).
- [ ] **Step 2:** Define `HitRegionIr` with fields: `id`, `semanticEntityId`, `semanticTwinTargetId`, `bounds`, `zIndex`, `role`, `keyboardOrder`, `pointerCursor`, optional `timelineAnchorId`, optional `damageBounds`.
- [ ] **Step 3:** Export helpers; schema tests pass.

---

## Task 3: `DisplayListIr` schema

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/display-list.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Create: `apps/aiperf-flow/packages/schema/test/display-list.test.ts`

**Interfaces:**
- Consumes: `HitRegionIr` cross-reference ids, shared color/transform/text attribute primitives.
- Produces: `DisplayListIr`, `DisplayCommandIr` discriminated union, `DamageRegionIr`, `PaintBoundsIr`, `parseDisplayListIr`, `hashDisplayListIr`, `stableSortDisplayCommands`.

- [ ] **Step 1:** Failing tests for command union exhaustiveness, deterministic ordering (`layer`, `zIndex`, `sourceOrder`, `commandIndex`), paint-bounds monotonicity, and hash invariance when non-semantic metadata is reordered.
- [ ] **Step 2:** Define command kinds sufficient for foundation and P0 stdlib proofs: `clear`, `save`, `restore`, `transform`, `clip`, `rect`, `roundedRect`, `path`, `stroke`, `fill`, `text`, `image`, `groupBegin`, `groupEnd`, `composite`, `debugBounds`. Each command carries `sourceNodeId`, optional `semanticEntityId`, `localBounds`, `qualityClass` (`required-semantic` | `decorative`), and optional `hitRegionId`.
- [ ] **Step 3:** Define top-level `DisplayListIr` with `version`, `timeMs`, `commands`, `paintBounds`, `damageRegions`, `hitRegions`, `contentHash`; export and test.

---

## Task 4: Quality tier and profile policy schema

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/quality-tier.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/capability.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Create: `apps/aiperf-flow/packages/schema/test/quality-tier.test.ts`
- Create: `apps/aiperf-flow/packages/schema/test/display-contract-descriptor.test.ts`

**Interfaces:**
- Consumes: `CapabilityDescriptor`, `DisplayCommandIr`.
- Produces: `QualityTierId`, `QualityProfileIr`, `QualityPolicyIr`, `DegradationReportIr`, `applyQualityPolicy`, descriptor field `displayContract`.

- [ ] **Step 1:** Failing tests that reference tier keeps all `required-semantic` commands and hit regions, degraded tier removes or simplifies only `decorative` commands, and reduced-motion suppresses motion-bearing command metadata without dropping entities or correspondence anchors.
- [ ] **Step 2:** Define tiers `reference` and `degraded`, plus profile axes `motion`, `contrast`, `depth`, each with explicit allowed suppressions listed in the schema module doc comment.
- [ ] **Step 3:** Extend capability descriptors with optional `displayContract` containing: supported command kinds, default hit-region role, quality class defaults, budget envelopes (frame-time target, memory ceiling, max decorative commands), and fallback behavior when a command kind is unsupported.
- [ ] **Step 4:** Schema tests pass; capability manifest sort and duplicate-id tests remain green.

---

## Task 5: Evaluation input envelope and hashing utilities

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/evaluation-input.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Create: `apps/aiperf-flow/packages/schema/test/evaluation-input.test.ts`

**Interfaces:**
- Consumes: `SceneIr`, packed-chunk metadata when present.
- Produces: `EvaluationInputIr`, `ViewportProfileIr`, `InteractionLogIr`, `EvaluationContext`, `deterministicStringify`, shared hash helpers used by runtime tests.

- [ ] **Step 1:** Failing tests that evaluation input rejects non-finite or
  fractional timeline values, preserves non-negative safe-integer
  milliseconds, and treats an absent interaction log as an empty immutable log.
- [ ] **Step 2:** Define input envelope fields required by the live-cinematic pure-function rule: validated scene IR reference, capability manifest fingerprint, viewport profile, integer `timeMs`, serializable interaction log snapshot, active quality profile, and optional responsive variant id.
- [ ] **Step 3:** Export helpers; schema tests pass.

---

## Task 6: Pure `SceneEvaluator`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/scene-evaluator.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/types.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/index.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/scene-evaluator.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/fixtures/evaluated-scene-foundation.json`

**Interfaces:**
- Consumes: `EvaluationContext`, `SceneIr`, `TimelinePlayer` snapshot shape, `SemanticModelIr`, `LayoutPlanIr`, capability registry manifest.
- Produces: `evaluateScene(context): EvaluatedSceneIr`.

- [ ] **Step 1:** Failing tests using a minimal foundation scene fixture: entity ids stable when layout bounds override changes, timeline target progress matches `TimelinePlayer` at the same integer `timeMs`, and camera state derives only from IR plus interaction log (never wall clock).
- [ ] **Step 2:** Implement evaluator that walks render tree by `resolveCapabilityId`, merges authored geometry with layout-plan overrides (geometry wins; semantic ids unchanged), and attaches accessibility outline nodes in reading order.
- [ ] **Step 3:** Consume the canonical runtime `SemanticProjection` from the
  semantic-projection plan; if schema promotion is enabled, alias it to the
  strict parsed type without changing field names.
- [ ] **Step 4:** Golden fixture for foundation rect/group/text scene at beats 0 ms, 1500 ms, and 3000 ms; runtime tests pass.

---

## Task 7: Pure `DisplayListBuilder`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/display-list.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/display-list.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/fixtures/display-list-foundation.json`

**Interfaces:**
- Consumes: `EvaluatedSceneIr`, capability `displayContract`, foundation style tokens.
- Produces: the promoted `buildDisplayList(scene): DisplayListIr` without a
  second display-list implementation.

- [ ] **Step 1:** Failing tests for deterministic command ordering, aggregated paint bounds, damage region union across changed nodes between two times, and hit-region attachment on selectable foundation nodes.
- [ ] **Step 2:** Implement builder with capability-specific lowerers for foundation primitives (`core.rect`, `core.text`, `core.group`, `core.connector`) returning backend-neutral commands only.
- [ ] **Step 3:** Golden display-list fixture matches evaluated-scene foundation fixture at the same beats; tests pass.

---

## Task 8: Hit-region index and damage tracking

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/hit-region-index.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/damage-tracker.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/hit-region-index.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/damage-tracker.test.ts`

**Interfaces:**
- Consumes: `HitRegionIr`, `DisplayListIr`, `DamageRegionIr`.
- Produces: `createHitRegionIndex`, `pickHitRegions`, `computeDamageBetween`, `mergeDamageRegions`.

- [ ] **Step 1:** Failing tests for point pick stability, top-most region resolution, keyboard traversal order independent of visual z-fighting tie-breakers, and damage region minimal supersets when decorative commands are removed later by quality policy.
- [ ] **Step 2:** Implement spatial index sufficient for unit tests (sorted array or grid); no Canvas dependency.
- [ ] **Step 3:** Tests pass against foundation and P0 span-map hand-authored evaluated-scene fragments once Task 10 fixtures exist.

---

## Task 9: Quality policy applier

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/quality-policy.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/quality-policy.test.ts`

**Interfaces:**
- Consumes: `DisplayListIr`, `QualityProfileIr`, capability `displayContract`.
- Produces: `applyQualityPolicy(list, profile): { list: DisplayListIr; report: DegradationReportIr }`.

- [ ] **Step 1:** Failing tests that degraded profile removes decorative particle/blur/shadow commands from golden foundation list while preserving semantic text, entity bounds, hit regions, and narration cue markers; reduced-motion zeroes motion metadata on remaining commands.
- [ ] **Step 2:** Implement applier that never drops commands tagged `required-semantic` and never removes hit regions whose roles are `select`, `inspect`, or `focus`.
- [ ] **Step 3:** Tests pass; degradation report lists suppressed command indices and families for diagnostics.

---

## Task 10: P0 hybrid capability display lowerers

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/lowerers/glyph-run.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/lowerers/span-map.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/lowerers/segment-strip.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/lowerers/waterfall.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/lowerers/index.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/p0-lowerers.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/fixtures/evaluated-scene-token-span-morph.json`
- Create: `apps/aiperf-flow/packages/runtime/test/fixtures/evaluated-scene-request-waterfall.json`

**Interfaces:**
- Consumes: leaf outputs (`leaf.glyph-measure`, `leaf.span-interval`, `core.segment-strip.layout`, `viz.waterfall.nest-layout`), `EvaluatedSceneIr` node attachments.
- Produces: capability-specific display-list fragments and hit regions aligned with core-components display contract.

- [ ] **Step 1:** Failing tests that hybrid lowerers emit semantic hit regions for token spans and waterfall intervals, tag decorative lane chrome as `decorative`, and keep morph correspondence ids on commands even when reduced-motion disables tween metadata.
- [ ] **Step 2:** Wire lowerers into `DisplayListBuilder` dispatch by capability id, not node kind alone.
- [ ] **Step 3:** Golden evaluated-scene and display-list fixtures for `TokenSpanMorph` and `RequestLifecycleWaterfall` beats used in core-components verification section; tests pass.

---

## Task 11: Seek-parity and determinism conformance suite

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/seek-parity.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/packed-parity.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/determinism.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/fixtures/seek-parity-beats.json`

**Interfaces:**
- Consumes: `evaluateScene`, `buildDisplayList`, `applyQualityPolicy`, `EvaluationContext`, foundation and P0 fixtures, packed IR equivalents from compiler package tests when available.
- Produces: conformance assertions documented as stable test names for downstream Canvas and semantic-twin plans to reuse.

- [ ] **Step 1:** Failing seek-parity tests that, for each fixture and beat in
  `seek-parity-beats.json`, simulate continuous playback and direct seek to the
  same safe-integer `timeMs`, then assert equal `hashEvaluatedSceneIr`, equal
  `hashDisplayListIr` under the same quality profile, and equal hit-region sets
  keyed by `semanticEntityId`.
- [ ] **Step 2:** Add packed-parity tests asserting normal and packed IR inputs yield identical evaluated-scene and display-list hashes at identical beats when compiler fixtures exist; skip with explicit pending message only if compiler pack tests are not yet landed.
- [ ] **Step 3:** Add determinism tests repeating evaluation one thousand times with identical inputs and asserting stable hashes; add profile-axis matrix tests (`reference`, `degraded`, `reduced-motion`, `high-contrast`, `no-depth`) verifying semantic command and hit-region counts never decrease relative to reference except for explicitly decorative command families.
- [ ] **Step 4:** All evaluate tests pass under `npm test -w @aiperf/flow-runtime`.

---

## Task 12: Runtime evaluation API surface and verification gate

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/registry.ts` (type-only extension for future `evaluate` hook; do not render Canvas here)
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/public-api.test.ts`

**Interfaces:**
- Produces: exported `evaluateFrame(context): { scene: EvaluatedSceneIr; displayList: DisplayListIr; report: DegradationReportIr }` as the single runtime entry for backend renderers and future semantic-twin projection.

- [ ] **Step 1:** Failing test that public API re-exports schema types unchanged and `evaluateFrame` composes evaluator, builder, and quality applier in documented order without wall-clock reads.
- [ ] **Step 2:** Implement `evaluateFrame` wrapper and document contract in module doc comment referencing core-components display contract bullets.
- [ ] **Step 3:** Run full verification:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm run flow:check
npm test -w @aiperf/flow-schema
npm test -w @aiperf/flow-runtime
```

Expected: all tests green; no new dependencies on `preview/`.

- [ ] **Step 4:** Update progress ledger at `.superpowers/sdd/progress.md` with evaluated-scene/display-list contract status and seek-parity gate result.

---

## Dependency order

```text
Task 1 → Task 2 → Task 3 → Task 4 → Task 5
Tasks 1–5 → Task 6 → Task 7 → Task 8 → Task 9
Task 7 → Task 10 → Task 11 → Task 12
```

Tasks 2 and 3 may proceed in parallel after Task 1 begins shared primitive extraction, but Task 3 should consume hit-region cross-reference ids before Task 7 lands.

## Downstream consumers (out of scope)

- Canvas 2D renderer draws `DisplayListIr` commands and uses `HitRegionIr` for pointer routing.
- Semantic HTML twin projects `EvaluatedSceneIr.semanticModel` and `accessibilityOutline` and synchronizes focus using `semanticTwinTargetId`.
- SVG/HTML fallback consumes simplified projections of the same evaluated scene.
- Viewer shell reads `evaluateFrame` outputs; it does not own scene geometry or timeline semantics.

## Execution options

1. **Subagent-driven (recommended)** — one subagent per task, review between tasks, seek-parity gate before Task 12.
2. **Inline** — land Tasks 1–5 schema contracts first, then Tasks 6–9 runtime core, then Task 11 seek-parity before P0 lowerers if fixtures are not ready.

# AIPerf Flow Live Cinematic Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the final runtime architecture for live, narrated,
interactive Flow experiences with professional high-resolution visual fidelity,
deterministic playback, pause-to-explore interaction, and a coequal semantic
accessibility surface.

**Architecture:** Validated Flow IR is evaluated once into a backend-neutral
semantic scene and deterministic display list. Canvas 2D is the preferred
cinematic visual backend; React/HTML owns the shell and always-mounted semantic
twin; SVG/HTML is the simplified fallback. One integer virtual clock drives
visuals, camera, narration, captions, and restoration. A future WebGPU backend
may consume the same contracts without changing authored semantics.

**Tech Stack:** TypeScript strict mode, React 19, Canvas 2D, SVG/HTML, Vitest,
Testing Library, Playwright, and the existing AIPerf Flow schema/runtime
packages.

## Global Constraints

- Authors commit only `.flow` and referenced assets; no document-specific
  React, TypeScript, JavaScript, or CSS.
- Canvas is never the semantic source of truth.
- The semantic HTML twin is mounted whenever the visual renderer is mounted.
- SVG/HTML fallback preserves entities, relations, narration, navigation,
  focus, selection, evidence, and interaction meaning.
- Layout, camera, style, timeline, and interaction evaluate once; renderers do
  not reinterpret Flow IR independently.
- All scene time is integer virtual time. Wall time may advance it but never
  becomes scene state.
- Direct seek and continuous playback to the same time produce equal evaluated
  semantic state.
- Exploration pauses playback by default. Resume continues from the exact
  paused beat and restores the authored camera according to scene policy.
- Quality degradation may reduce blur, glow, particles, shadows, and sampling
  density; it may not remove semantic or narrative content.
- The 3840×2160 profile is a fidelity verification target, not a video-first
  product requirement.
- WebGPU is out of scope for implementation; only backend compatibility of the
  contracts is in scope.
- Activate the repository environment before commands:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Do not create commits unless the user explicitly requests them.

## Current implementation baseline

`FlowApp` now composes `evaluateScene`, the deterministic `DisplayList`, Canvas
rendering, `SvgFallback`, the always-mounted `SemanticTwin`, subtitles, and
focus coordination. The runtime has one shared `SemanticProjection`, though its
entity shape still carries transitional `role`/`kind` overlap. These modules
are unit-tested and mounted; they are not yet promoted to complete strict
versioned schema IR or proven against the full fidelity, exploration,
performance, and cross-backend conformance gates.

This plan hardens the landed stack. It must not create a parallel renderer,
replace `FlowApp` with the foundation `SceneRenderer`, or reintroduce a second
semantic projection. `apps/aiperf-flow/preview/**` remains outside runtime
ownership.

Schema promotion is owned by
[`2026-07-17-aiperf-flow-display-list.md`](2026-07-17-aiperf-flow-display-list.md).
Runtime semantic projection unification is owned by
[`2026-07-17-aiperf-flow-semantic-projection.md`](2026-07-17-aiperf-flow-semantic-projection.md).
Hybrid component contributions are owned by
[`2026-07-17-aiperf-flow-hybrid-renderers.md`](2026-07-17-aiperf-flow-hybrid-renderers.md).

---

### Task 1: Define evaluated-scene and display-list contracts

Detailed implementation authority:
[`2026-07-17-aiperf-flow-display-list.md`](2026-07-17-aiperf-flow-display-list.md).
This task is the parent completion gate and must not create duplicate types.

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/types.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/evaluate/scene-evaluator.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/display-list.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/evaluate/scene-evaluator.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/display-list.test.ts`

**Interfaces:**
- Consumes: validated `SceneIr`, capability registry, viewport profile, integer
  timeline time, and serializable interaction state.
- Produces: `EvaluatedScene`, `DisplayList`, `DrawCommand`, `HitRegion`, and
  `SemanticProjection`.

- [ ] Write tests asserting stable entity IDs, deterministic draw ordering,
  finite geometry, explicit paint/damage bounds, source maps, semantic
  projections, and byte-identical serialization from repeated evaluation.
- [ ] Promote and unify the existing immutable unions through the display-list
  contract plan; do not create competing runtime-local and schema shapes.
- [ ] Make the existing `evaluateScene(input): EvaluatedScene` output directly
  consumable by the existing `SemanticTwin` and display-list backend contracts.
- [ ] Assert render backends receive evaluated contracts only and never import
  parser/compiler packages.
- [ ] Run `npm test -w @aiperf/flow-runtime -- scene-evaluator.test.ts display-list.test.ts`.

---

### Task 2: Harden deterministic time and pause-to-explore state

Detailed implementation authority:
[`2026-07-17-aiperf-flow-virtual-clock.md`](2026-07-17-aiperf-flow-virtual-clock.md).

**Expanded plan:** implement via
[`2026-07-17-aiperf-flow-virtual-clock.md`](2026-07-17-aiperf-flow-virtual-clock.md)
(standalone TDD tasks for integer time, seek≡play, exploration APIs, and store
wiring). Do not edit `preview/**`.

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/player.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/store.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/exploration.ts`
- Test: `apps/aiperf-flow/packages/runtime/test/player-determinism.test.ts`
- Test: `apps/aiperf-flow/packages/runtime/test/exploration.test.ts`

**Interfaces:**
- Produces: `TimelineSnapshot`, `ExplorationSnapshot`,
  `beginExploration()`, `updateExploration()`, and `resumeLesson()`.

- [ ] Test direct seek against continuous playback at every cue boundary and
  representative in-between times.
- [ ] Test that exploration pauses narration, captions, camera, and visual
  tracks at one integer timestamp.
- [ ] Test pan, zoom, selection, focus, compare, and inspector state as
  serializable temporary state.
- [ ] Implement authored-camera restoration and exact-beat resume without
  replaying or skipping narration.
- [ ] Add reduced-motion restoration tests that use cuts or crossfades while
  preserving semantic state.
- [ ] Run `npm test -w @aiperf/flow-runtime -- player-determinism.test.ts exploration.test.ts`.

---

### Task 3: Harden the landed Canvas 2D cinematic backend

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/backends/canvas/canvas-renderer.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/backends/canvas/text-atlas.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/backends/canvas/hit-test.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/backends/canvas/quality.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/backends/canvas-renderer.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/backends/hit-test.test.ts`

**Interfaces:**
- Consumes: `DisplayList`, `CanvasRenderingContext2D`, device-pixel ratio, and
  `QualityProfile`.
- Produces: deterministic draw calls, paint metrics, and semantic hit results.

- [ ] Build a recording Canvas context fixture and test command order,
  transforms, clipping, compositing, text metrics, pixel scaling, and damage
  regions without browser timing.
- [ ] Render backing stores at CSS size × device-pixel ratio while keeping
  logical scene coordinates resolution-independent.
- [ ] Implement deterministic path, text, image, layer, clip, glow, shadow, and
  routed-light-path commands.
- [ ] Implement semantic hit testing from `HitRegion` rather than pixel
  sampling.
- [ ] Implement reference and degraded quality profiles; degradation affects
  decorative cost only.
- [ ] Run the Canvas backend and hit-test unit suites.

---

### Task 4: Harden the landed always-mounted semantic HTML twin

Canonical projection unification is owned by
[`2026-07-17-aiperf-flow-semantic-projection.md`](2026-07-17-aiperf-flow-semantic-projection.md).
This task owns mounting and viewer behavior after that contract lands.

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/semantic/semantic-twin.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/src/semantic/focus-coordinator.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/semantic/fallback-table.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/test/semantic/semantic-twin.test.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/test/semantic/focus-coordinator.test.tsx`

**Interfaces:**
- Consumes: `SemanticProjection`, transcript state, visual selection, and
  runtime actions.
- Produces: landmarks, ordered entities/relations, keyboard controls,
  descriptions, tables, and synchronized focus/selection events.

- [ ] Test reading order independent from visual draw order.
- [ ] Test Canvas selection moving semantic focus and semantic keyboard
  activation selecting the matching visual entity.
- [ ] Test transcript/caption linkage, evidence access, relationship traversal,
  and chart/table alternatives.
- [ ] Keep the twin available in a visually compact mode; do not hide it with
  `display: none` or `aria-hidden`.
- [ ] Run semantic twin and focus coordinator tests plus automated
  accessibility checks.

---

### Task 5: Harden and verify the landed SVG/HTML fallback backend

**Files:**
- Refactor: `apps/aiperf-flow/packages/runtime/src/renderer.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/src/backends/svg/svg-fallback.tsx`
- Create: `apps/aiperf-flow/packages/runtime/test/backends/backend-conformance.test.tsx`

**Interfaces:**
- Consumes: `EvaluatedScene` and `DisplayList`.
- Produces: simplified semantic SVG/HTML with no independent Flow IR
  interpretation.

- [ ] Move foundation SVG creation behind the evaluated-scene contract.
- [ ] Define one conformance fixture and assert equal entity IDs, relations,
  labels, focus targets, selection state, transcript position, and fallback
  meaning across Canvas, semantic twin, and SVG/HTML.
- [ ] Test no-Canvas, print, high-contrast, no-depth, and missing-capability
  behavior.
- [ ] Verify a backend failure cannot remove navigation, transcript, evidence,
  or semantic controls.

---

### Task 6: Complete runtime shell controls and exploration behavior

**Baseline (already landed — do not rebuild):**
- `evaluateScene` (`evaluate/scene-evaluator.ts`), `DisplayList` /
  `buildDisplayList` (`display-list.ts`), Canvas
  (`backends/canvas/canvas-renderer.ts`, `hit-test.ts`), `SvgFallback`
  (`backends/svg/svg-fallback.tsx`), `SemanticTwin` + focus coordinator
  (`semantic/`), and their unit tests exist.
- `FlowApp` and `site.tsx` mount that stack today. This task preserves the
  composition while finishing real controls, exploration, quality management,
  failure isolation, and conformance.
- Schema promotion remains owned by the display-list plan. Runtime projection
  normalization remains owned by the semantic-projection plan.

**Out of scope:**
- Do **not** edit `apps/aiperf-flow/preview/**` (including `preview/App.tsx`).
  Preview is owned by another agent. Integration is proven through
  `packages/runtime` tests and the packed-site mount path in `site.tsx`.

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/app.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/src/site.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/src/theme.css`
- Test: `apps/aiperf-flow/packages/runtime/test/app.test.tsx`
- Do not modify: `apps/aiperf-flow/preview/**`

**Interfaces:**
- Consumes: packed/manifest `FlowIr` (via existing `loadPackedFlow` /
  `FlowApp` props), integer `TimelinePlayer` time, capability registry,
  existing `evaluateScene` → `EvaluatedScene` / `DisplayList`, Canvas or
  `SvgFallback`, always-mounted `SemanticTwin`, and focus coordinator.
- Produces: one mounted stage that evaluates once per integer time, draws
  through a visual backend, and keeps the semantic twin coequal.

**Exact integration — `packages/runtime/src/app.tsx`:**
- [ ] Keep `TimelinePlayer`, scene navigation, play/pause/restart, capability
  gating, and `SceneErrorBoundary` / `SceneFailure`; do not replace the shell
  chrome wholesale.
- [ ] On each valid `scene` + integer `timeMs`, call
  `evaluateScene(scene, timeMs)` (and `buildDisplayList` when the display list
  is not already on the evaluated scene) inside `useMemo` keyed by scene id
  and `timeMs`.
- [ ] Stop mounting foundation `SceneRenderer` for the happy path. Mount a
  cinematic stage that prefers Canvas (`renderDisplayList` /
  `renderCanvasDisplayList` + `hitTest` on pointer events) when a 2D context
  is available.
- [ ] Always mount `SemanticTwin` beside the visual backend (never
  `display: none` / `aria-hidden`), driven by the evaluated
  `SemanticProjection` and `createFocusCoordinator` so Canvas hit selection
  and twin keyboard focus stay synchronized.
- [ ] When Canvas is unavailable or fails inside the error boundary, render
  `SvgFallback` with the same `EvaluatedScene` + `DisplayList` (and shared
  focus/selection props); do not re-interpret raw Flow IR in the fallback.
- [ ] Wire controls to runtime actions already owned by the player/store
  (play, pause, seek/time readout, restart) and extend with select, pan,
  zoom, fit, inspect, exploration pause, and resume once Task 2’s exploration
  API is available — keep inactive placeholders only until that API lands.
- [ ] Keep the cinematic stage dominant; compact title/narrative chrome stays
  in the upper-left safe area and outside the scene’s semantic coordinate
  system. Transcript, captions, and inspector may collapse but must remain
  reachable (including the existing skip link to `#flow-transcript`).

**Exact integration — `packages/runtime/src/site.tsx`:**
- [ ] Keep `loadPackedFlow` → `FlowApp` as the packed-site entry; do not add a
  parallel mount that bypasses `FlowApp`.
- [ ] Ensure `mountFlowSite` still isolates load failures in `SiteLoadFailure`
  and that a successful load mounts the integrated `FlowApp` (evaluated
  scene + backends + twin) with the shared foundation registry.
- [ ] Pass through only shell-level props already supported by `FlowApp`
  (e.g. `reducedMotion` when detectible); do not pull preview-only fixture
  navigation into `site.tsx`.
- [ ] Adjust `theme.css` only as needed for stage dominance, safe-area chrome,
  and twin compact layout — no document-specific styles.

**Verification:**
- [ ] Add/extend `packages/runtime/test/app.test.tsx` for: evaluate-once path
  (no `SceneRenderer` on happy path), Canvas vs `SvgFallback` switching,
  always-mounted twin, keyboard playback/exploration focus restoration, and
  scene-route changes resetting player time.
- [ ] Run `npm test -w @aiperf/flow-runtime -- app.test.tsx`.

---

### Task 7: Add fidelity, responsive, and performance verification

**Files:**
- Create: `apps/aiperf-flow/e2e/live-cinematic-runtime.spec.ts`
- Create: `apps/aiperf-flow/e2e/fixtures/cinematic-foundation.flow`
- Create: `apps/aiperf-flow/scripts/measure-runtime.mjs`
- Modify: `apps/aiperf-flow/package.json`

**Interfaces:**
- Produces: visual snapshots, semantic snapshots, frame metrics, memory metrics,
  and a machine-readable quality report.

- [ ] Add Playwright snapshots at 3840×2160, 1920×1080, tablet, and mobile
  profiles with fixed fonts, assets, device scale, random seed, and timeline
  time.
- [ ] Add dark, light, high-contrast, reduced-motion, reduced-transparency, and
  no-depth snapshots.
- [ ] Measure evaluation, draw, and total frame time separately; record p50,
  p95, and worst frame with the reference environment in the report.
- [ ] Verify reference mode targets 60 frames per second and degraded mode
  targets 30 without semantic loss.
- [ ] Verify text sharpness, caption safe areas, asset resolution, contrast,
  damage bounds, and no unexpected layout overflow.
- [ ] Fail CI on semantic, determinism, accessibility, or visual-snapshot
  regressions. Report performance regressions with the recorded environment so
  hardware differences are explicit.

---

### Task 8: Prove the north star with `RequestLifecycleWaterfall`

**Files:**
- Create: `apps/aiperf-flow/examples/cinematic/request-lifecycle.flow`
- Create: `apps/aiperf-flow/e2e/request-lifecycle-cinematic.spec.ts`
- Modify: `apps/aiperf-flow/README.md`

**Interfaces:**
- Consumes: live cinematic runtime and P0 queue/waterfall components.
- Produces: one `.flow`-only narrated, interactive reference experience.

- [ ] Author topology, queue, transport, model, stream, observer, and synchronized
  waterfall semantics with stable IDs and evidence.
- [ ] Author establish, teach, inspect, and transition beats with captions,
  narration, camera, reduced-motion, responsive, and fallback policies.
- [ ] Verify request chips move through the topology while arrival, admission,
  first-token, terminal, and record times remain distinct in the waterfall.
- [ ] Pause during a named inspect beat, navigate by keyboard through the
  semantic twin, inspect evidence, and resume from the exact beat.
- [ ] Pass Canvas, semantic twin, SVG/HTML fallback, 3840×2160 fidelity,
  responsive, accessibility, determinism, and budget gates.
- [ ] Document the live-interactive north star and state explicitly that video
  export is a possible future consumer of deterministic frames, not the
  primary product.

---

## Completion gate

This plan is complete only when one compiled `.flow` scene:

1. evaluates identically by direct seek and continuous playback;
2. renders cinematically through Canvas at the reference profile;
3. exposes a synchronized, keyboard-operable semantic HTML twin;
4. preserves meaning through SVG/HTML fallback;
5. pauses for exploration and resumes from the exact same beat;
6. remains responsive and accessible under all required variants; and
7. contains no document-specific React, TypeScript, JavaScript, or CSS.

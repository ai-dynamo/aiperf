<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Flow Immersive Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the preview's video-player grammar with a fullscreen Causal
Field and ship Causal Replay, Command Constellation, Context Lens, Focus World,
and the runtime/E2E seams required to support them.

**Architecture:** Add one pure `evaluateFrame` seam over the existing evaluator,
quality policy, hit index, and damage tracker. Build causal beats and commands
as deterministic projections over validated scene/runtime state. Both `FlowApp`
and the preview compose the same focused runtime components; the preview does
not maintain a second evaluator or interaction model.

**Tech Stack:** TypeScript strict mode, React 19, Zod 4, Canvas 2D, SVG/HTML,
Vitest, Testing Library, Playwright, and existing AIPerf Flow packages.

## Global Constraints

- The scene is the application surface; do not wrap it in a glass player card.
- Causal Replay and Command Constellation are the primary interaction spine.
- Context Lens and Focus World are first-class but visually contextual.
- Canvas is never the semantic source of truth.
- The semantic twin remains mounted whenever the visual renderer is mounted.
- All scene time is non-negative safe-integer virtual milliseconds.
- Direct beat seek and continuous playback to the same time produce equal
  evaluated state.
- Exploration pauses at the exact current timestamp and restores authored
  camera and focus policy on resume.
- Quality degradation may remove decorative cost only, never semantics,
  captions, focus, evidence, hit regions, or commands.
- Preview and packed `FlowApp` share interaction vocabulary and Systems Chalk
  chrome roles.
- Scene paints remain evaluated display-list values; renderers do not branch on
  theme IDs.
- Authors commit only `.flow` and referenced assets; no document-specific
  React, TypeScript, JavaScript, or CSS.
- Preserve existing public IR contracts unless a task explicitly adds a
  backward-compatible field.
- Do not modify unrelated architecture-atlas, explainers, Python, or Rust code.
- Before commands run:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Do not create commits unless the user explicitly requests them.

## File Structure

New focused runtime modules:

- `packages/runtime/src/evaluate/frame.ts`: pure evaluated-frame composition.
- `packages/runtime/src/causal-replay.ts`: authored beat projection and
  keyboard traversal.
- `packages/runtime/src/immersive-state.ts`: Context Lens, Focus World, HUD,
  fullscreen, and URL state reducers.
- `packages/runtime/src/commands.ts`: deterministic command catalog and search.
- `packages/runtime/src/immersive/causal-path.tsx`: causal beat navigation.
- `packages/runtime/src/immersive/command-constellation.tsx`: command dialog.
- `packages/runtime/src/immersive/context-lens.tsx`: selected entity inspector.
- `packages/runtime/src/immersive/immersive-controls.tsx`: HUD and fullscreen.

Existing composition files:

- `packages/runtime/src/app.tsx`: shared runtime state and mounted integration.
- `packages/runtime/src/theme.css`: Causal Field chrome and responsive layout.
- `preview/App.tsx`: host/document browser around the shared runtime shell.
- `preview/styles.css`: preview-only outer browser/drawer styles.

Verification:

- focused runtime unit/component tests beside existing test groups;
- root Playwright config and scripts;
- existing cinematic E2E specs plus deterministic screenshot/telemetry outputs.

---

### Task 1: Add the pure `evaluateFrame` seam

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/evaluate/frame.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/evaluate/frame.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/public-api.test.ts`

**Interfaces:**
- Consumes: `SceneIr`, integer `timeMs`, `EvaluateSceneOptions`,
  `QualityPolicyProfile`, optional `DisplayContract`, optional previous
  `DisplayList`.
- Produces:

```ts
export type EvaluateFrameOptions = Readonly<{
  scene?: EvaluateSceneOptions;
  quality?: QualityPolicyProfile;
  displayContract?: DisplayContract;
  previousDisplayList?: DisplayList;
}>;

export type EvaluatedFrame = Readonly<{
  scene: EvaluatedScene;
  displayList: QualityDisplayList;
  report: DegradationReport;
  hitIndex: HitRegionIndex;
  damageRegions: readonly Bounds[];
}>;

export function evaluateFrame(
  scene: SceneIr,
  timeMs: number,
  options?: EvaluateFrameOptions,
): EvaluatedFrame;
```

- [ ] **Step 1: Write failing frame-composition tests**

Test that evaluation happens at the exact integer time, reference quality is the
default, degraded quality preserves semantic hit regions, the hit index uses
the resulting display list, and damage compares the optional previous frame.

```ts
const frame = evaluateFrame(scene, 1500, {
  quality: qualityPolicyProfile("degraded", { motion: "reduced" }),
  previousDisplayList,
});

expect(frame.scene.atMs).toBe(1500);
expect(frame.displayList.hitRegions.map(({ semanticId }) => semanticId))
  .toEqual(frame.scene.displayList.hitRegions.map(({ semanticId }) => semanticId));
expect(frame.report.tier).toBe("degraded");
expect(frame.hitIndex.keyboardTraversal).toEqual(frame.displayList.hitRegions);
expect(frame.damageRegions).toEqual(
  computeDamageBetween(previousDisplayList, frame.displayList),
);
expect(Object.isFrozen(frame)).toBe(true);
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- frame.test.ts
```

Expected: FAIL because `evaluate/frame.ts` and `evaluateFrame` do not exist.

- [ ] **Step 3: Implement minimal pure composition**

```ts
export function evaluateFrame(
  sceneIr: SceneIr,
  timeMs: number,
  options: EvaluateFrameOptions = {},
): EvaluatedFrame {
  const scene = evaluateScene(sceneIr, timeMs, options.scene);
  const quality = applyQualityPolicy(
    scene.displayList,
    options.quality ?? qualityPolicyProfile("reference"),
    options.displayContract,
  );
  const displayList = quality.list;
  return deepFreeze({
    scene: { ...scene, displayList },
    displayList,
    report: quality.report,
    hitIndex: createHitRegionIndex(displayList),
    damageRegions:
      options.previousDisplayList === undefined
        ? [displayList.damageBounds]
        : computeDamageBetween(options.previousDisplayList, displayList),
  });
}
```

Reject non-safe-integer time before evaluation. Do not add schema IR or a
parallel display-list type.

- [ ] **Step 4: Export and verify GREEN**

Export `evaluateFrame`, `EvaluatedFrame`, and `EvaluateFrameOptions` from the
runtime index. Run focused tests and:

```bash
npm run build -w @aiperf/flow-runtime
```

Expected: focused tests and strict TypeScript build pass.

---

### Task 2: Project deterministic causal beats

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/causal-replay.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/causal-replay.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`

**Interfaces:**

```ts
export type CausalBeat = Readonly<{
  id: string;
  label: string;
  description?: string;
  timeMs: number;
  endMs: number;
  targetEntityIds: readonly string[];
  source: "timeline" | "narrative";
}>;

export type CausalBeatState = "complete" | "active" | "future";

export function projectCausalBeats(scene: SceneIr): readonly CausalBeat[];
export function activeCausalBeat(
  beats: readonly CausalBeat[],
  timeMs: number,
): CausalBeat | null;
export function causalBeatState(
  beat: CausalBeat,
  timeMs: number,
): CausalBeatState;
export function adjacentCausalBeat(
  beats: readonly CausalBeat[],
  activeId: string | null,
  direction: "first" | "previous" | "next" | "last",
): CausalBeat | null;
```

- [ ] **Step 1: Write failing projection and traversal tests**

Use real `SceneIr` timeline and narrative cues. Assert deterministic ordering by
`timeMs`, authored order, then ID; duplicate timeline/narrative IDs fail closed;
labels use authored action or subtitle text; targets come only from authored
`target`; all times are safe integers.

```ts
const beats = projectCausalBeats(scene);
expect(beats.map(({ id }) => id)).toEqual([
  "arrival",
  "admission",
  "first-token",
]);
expect(adjacentCausalBeat(beats, "admission", "next")?.id)
  .toBe("first-token");
expect(activeCausalBeat(beats, 1500)?.id).toBe("admission");
```

- [ ] **Step 2: Verify RED**

Run:

```bash
npm test -w @aiperf/flow-runtime -- causal-replay.test.ts
```

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement projection without inferred domain semantics**

Timeline cues map to `id`, `action`, `at`, `at + duration`, and `target`.
Narrative cues are included only when no timeline beat exists at the same ID.
Normalize labels by replacing `-`/`_` with spaces; do not invent names from draw
commands.

- [ ] **Step 4: Verify GREEN and export**

Run focused tests and the runtime build. Expected: all pass.

---

### Task 3: Add serializable immersive interaction state

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/immersive-state.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/immersive-state.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/store.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/exploration.ts`

**Interfaces:**

```ts
export type HudVisibility = "present" | "quiet" | "hidden";
export type FullscreenState = "windowed" | "layout" | "native";

export type ImmersiveState = Readonly<{
  selectedEntityId: string | null;
  contextLensOpen: boolean;
  focusWorldEntityId: string | null;
  comparisonEntityId: string | null;
  commandOpen: boolean;
  hud: HudVisibility;
  fullscreen: FullscreenState;
}>;

export type ImmersiveAction =
  | Readonly<{ type: "select"; entityId: string | null }>
  | Readonly<{ type: "open-context"; entityId: string }>
  | Readonly<{ type: "close-context" }>
  | Readonly<{ type: "enter-focus-world"; entityId: string }>
  | Readonly<{ type: "leave-focus-world" }>
  | Readonly<{ type: "open-command" }>
  | Readonly<{ type: "close-command" }>
  | Readonly<{ type: "set-hud"; visibility: HudVisibility }>
  | Readonly<{ type: "set-fullscreen"; state: FullscreenState }>;

export function createImmersiveState(): ImmersiveState;
export function immersiveReducer(
  state: ImmersiveState,
  action: ImmersiveAction,
): ImmersiveState;
```

- [ ] **Step 1: Write failing reducer and restoration tests**

Assert Context Lens selects its entity, Focus World preserves selection,
leaving Focus World restores selection, command close does not change playback,
and every state is JSON-serializable.

Extend `ExplorationSnapshot` with optional immutable `immersive` state and test
that `resumeLesson` restores the authored timestamp, selected entity, and focus
world state.

- [ ] **Step 2: Verify RED**

Run:

```bash
npm test -w @aiperf/flow-runtime -- immersive-state.test.ts exploration.test.ts
```

Expected: new module imports fail.

- [ ] **Step 3: Implement immutable state transitions**

Use exhaustive discriminated unions. Context Lens and Focus World update stable
semantic IDs only. Do not store DOM nodes, Canvas contexts, wall time, or
backend state.

- [ ] **Step 4: Verify GREEN**

Run focused tests and strict runtime build.

---

### Task 4: Build deterministic Command Constellation

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/commands.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/immersive/command-constellation.tsx`
- Create: `apps/aiperf-flow/packages/runtime/test/commands.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/immersive/command-constellation.test.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`

**Interfaces:**

```ts
export type FlowCommandCategory =
  | "scene"
  | "beat"
  | "entity"
  | "evidence"
  | "action"
  | "accessibility";

export type FlowCommand = Readonly<{
  id: string;
  label: string;
  category: FlowCommandCategory;
  keywords: readonly string[];
  shortcut?: string;
  disabledReason?: string;
  execute(): void;
}>;

export function searchCommands(
  commands: readonly FlowCommand[],
  query: string,
): readonly FlowCommand[];
```

`CommandConstellation` consumes commands, `open`, `onClose`, and an optional
initial query.

- [ ] **Step 1: Write failing search and dialog tests**

Search order is exact label prefix, token prefix, keyword prefix, then authored
order. It is case-insensitive and deterministic. Test arrow navigation, Enter,
Escape, focus trap, disabled reason, and focus restoration to the invoking
button.

```ts
expect(searchCommands(commands, "first token").map(({ id }) => id))
  .toEqual(["beat:first-token", "entity:first-token-marker"]);
```

- [ ] **Step 2: Verify RED**

Run:

```bash
npm test -w @aiperf/flow-runtime -- commands.test.ts command-constellation.test.tsx
```

- [ ] **Step 3: Implement pure search and accessible dialog**

Use native React state and DOM semantics; add no command-menu dependency.
Render `role="dialog"` with a labelled search input and `role="listbox"`.
Store only active command ID, not array index.

- [ ] **Step 4: Verify GREEN and export**

Run focused tests, lints, and runtime build.

---

### Task 5: Build Causal Path, Context Lens, and immersive controls

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/immersive/causal-path.tsx`
- Create: `apps/aiperf-flow/packages/runtime/src/immersive/context-lens.tsx`
- Create: `apps/aiperf-flow/packages/runtime/src/immersive/immersive-controls.tsx`
- Create: `apps/aiperf-flow/packages/runtime/test/immersive/causal-path.test.tsx`
- Create: `apps/aiperf-flow/packages/runtime/test/immersive/context-lens.test.tsx`
- Create: `apps/aiperf-flow/packages/runtime/test/immersive/immersive-controls.test.tsx`

**Interfaces:**

```ts
export type CausalPathProps = Readonly<{
  beats: readonly CausalBeat[];
  timeMs: number;
  onSeek(timeMs: number, beatId: string): void;
}>;

export type ContextLensProps = Readonly<{
  projection: SemanticProjection;
  entityId: string;
  onClose(): void;
  onFocusWorld(entityId: string): void;
  onOpenTwin(entityId: string): void;
}>;

export type ImmersiveControlsProps = Readonly<{
  playing: boolean;
  exploring: boolean;
  hud: HudVisibility;
  fullscreen: FullscreenState;
  onPlayPause(): void;
  onExploreResume(): void;
  onOpenCommands(): void;
  onToggleTwin(): void;
  onToggleFullscreen(): void;
}>;
```

- [ ] **Step 1: Write failing component tests**

Test roving beat focus, Home/End/arrow keys, current beat semantics, lens
relation/evidence projection, missing optional evidence, control names, and HUD
visibility. Test fullscreen actions through an injected adapter rather than the
real browser global.

- [ ] **Step 2: Verify RED**

Run the three focused component tests. Expected: missing modules.

- [ ] **Step 3: Implement minimal accessible components**

`CausalPath` is a labelled navigation control, not `input[type=range]`.
`ContextLens` reads only `SemanticProjection`. Controls remain reachable while
HUD visuals are quiet; hidden decorative chrome must not hide focused controls.

- [ ] **Step 4: Verify GREEN**

Run focused tests and runtime build.

---

### Task 6: Integrate the complete Causal Field into `FlowApp`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/app.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/src/theme.css`
- Modify: `apps/aiperf-flow/packages/runtime/test/app.test.tsx`
- Create: `apps/aiperf-flow/packages/runtime/test/app-immersive.test.tsx`

**Interfaces:**
- Consumes: `evaluateFrame`, causal beats, immersive reducer, command catalog,
  existing `TimelinePlayer`, exploration, Canvas/SVG, semantic twin, narration.
- Produces: one shared Causal Field shell used by packed site and preview.

- [ ] **Step 1: Write failing mounted integration tests**

Assert:

- the stage is the dominant `aria-label="Scene field"` region;
- no range/media scrubber exists;
- causal beats seek exact integer time;
- command palette can jump to scene, beat, entity, and semantic twin;
- Context Lens opens from Canvas and semantic activation;
- Focus World enters/exits without changing beat;
- exploration/resume restores beat, focus, and camera state;
- Canvas failure retains the same HUD, commands, twin, and causal path;
- fullscreen denial announces a recoverable message;
- HUD quiet/hidden policy never hides captions or focused controls.

- [ ] **Step 2: Verify RED**

Run:

```bash
npm test -w @aiperf/flow-runtime -- app-immersive.test.tsx
```

Expected: semantic queries fail against current player-style shell.

- [ ] **Step 3: Replace local evaluation with `evaluateFrame`**

Memoize by scene ID, integer `timeMs`, quality axes, and serializable
interaction state. Keep previous display list in a ref for damage computation.
Do not add a second player or evaluator.

- [ ] **Step 4: Compose all four capabilities**

Build commands from current scenes, projected beats, semantic entities,
evidence IDs, and existing actions. Causal Replay and Command Constellation are
always discoverable. Context Lens and Focus World appear only with a valid
entity.

- [ ] **Step 5: Implement Causal Field CSS**

Map chrome custom properties to Board, Panel, Chalk, Guide, Signal, and Beat.
Remove glass/card/shadow/media-progress styling from mounted shell. Add desktop,
390×844, fullscreen, forced-colors, high-contrast, and reduced-motion rules.

- [ ] **Step 6: Verify GREEN**

Run all app/runtime tests and the runtime build.

---

### Task 7: Converge the preview host without duplicating runtime state

**Files:**
- Modify: `apps/aiperf-flow/preview/App.tsx`
- Modify: `apps/aiperf-flow/preview/styles.css`
- Modify: `apps/aiperf-flow/preview/narrative.test.tsx`
- Create: `apps/aiperf-flow/preview/immersive.test.tsx`

**Interfaces:**
- Consumes: shared Causal Field `FlowApp` or shared runtime immersive
  components/actions.
- Produces: document browser overlay around the same mounted experience.

- [ ] **Step 1: Write failing preview tests**

Assert no `story-stage`, chapter rail, Back/Next pills, or media progress bar.
Assert the document browser opens as a drawer/overlay without reducing the
scene field; audio consent/narrator modes remain; command shortcut opens
Command Constellation; mobile has no horizontal overflow contract class.

- [ ] **Step 2: Verify RED**

Run:

```bash
npm test -w @aiperf/flow-runtime -- ../../preview/immersive.test.tsx
```

If preview tests are not included by the runtime config, add a root preview
Vitest script and run that exact script instead.

- [ ] **Step 3: Remove duplicate evaluator/player ownership**

Prefer rendering shared `FlowApp` with preview host props. If one preview-only
document browser state remains, keep it outside `FlowApp`. Delete preview-local
`evaluateScene`, `TimelinePlayer`, exploration, semantic twin, and renderer
composition after tests prove shared ownership.

- [ ] **Step 4: Convert browser and metadata to overlays**

Keep source/search/scene navigation as an accessible drawer. It must not change
scene coordinate space or own playback state.

- [ ] **Step 5: Verify GREEN**

Run preview tests, runtime app tests, and `npm run flow:check`.

---

### Task 8: Add URL state, fullscreen adapter, and HUD policy

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/immersive-url.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/fullscreen.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/hud-policy.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/immersive-url.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/fullscreen.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/hud-policy.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/app.tsx`

**Interfaces:**

```ts
export type ImmersiveUrlState = Readonly<{
  sceneId: string | null;
  beatId: string | null;
  entityId: string | null;
}>;

export function parseImmersiveUrl(search: string): ImmersiveUrlState;
export function serializeImmersiveUrl(state: ImmersiveUrlState): string;

export interface FullscreenAdapter {
  supported(): boolean;
  active(): boolean;
  enter(element: HTMLElement): Promise<void>;
  exit(): Promise<void>;
}

export function hudVisibilityFor(input: Readonly<{
  playing: boolean;
  exploring: boolean;
  commandOpen: boolean;
  focusedWithinHud: boolean;
  inactive: boolean;
}>): HudVisibility;
```

- [ ] **Step 1: Write failing pure tests**

Test URL round trips and invalid ID fallback, fullscreen success/denial/layout
fallback, and HUD truth table. No arbitrary timer sleeps.

- [ ] **Step 2: Verify RED**

Run the three focused tests.

- [ ] **Step 3: Implement adapters and pure policies**

URL values are decoded strings and validated against current scene/beat/entity
sets by `FlowApp`; parser itself does not access global state. HUD inactivity is
provided as an event/policy input, not `Date.now()` scene state.

- [ ] **Step 4: Integrate and verify GREEN**

Update URL with `history.replaceState` after valid scene/beat/entity changes.
Restore on initial mount. Announce fullscreen denial via existing live region.

---

### Task 9: Add a real Playwright harness and cinematic gates

**Files:**
- Modify: `apps/aiperf-flow/package.json`
- Modify: `apps/aiperf-flow/package-lock.json`
- Create: `apps/aiperf-flow/playwright.config.ts`
- Modify: `apps/aiperf-flow/e2e/live-cinematic-runtime.spec.ts`
- Modify: `apps/aiperf-flow/e2e/request-lifecycle-cinematic.spec.ts`
- Create: `apps/aiperf-flow/e2e/immersive-preview.spec.ts`
- Create: `apps/aiperf-flow/e2e/immersive-preview.spec.ts-snapshots/*`
- Modify: `apps/aiperf-flow/scripts/measure-runtime.mjs`
- Create: `apps/aiperf-flow/e2e/helpers/runtime-metrics.ts`

**Interfaces:**
- Adds latest stable `@playwright/test` through npm.
- Adds scripts:

```json
{
  "e2e": "playwright test",
  "e2e:update": "playwright test --update-snapshots",
  "flow:verify": "npm run flow:check && npm run e2e && npm run measure:runtime"
}
```

- [ ] **Step 1: Install Playwright and establish the harness**

Run:

```bash
cd apps/aiperf-flow
npm install --save-dev @playwright/test
npx playwright install chromium
```

Configure one deterministic Chromium project, a Vite web server, fixed locale,
timezone, color scheme, and reduced animation defaults.

- [ ] **Step 2: Run existing E2E specs and record RED failures**

Run:

```bash
npm run e2e -- e2e/live-cinematic-runtime.spec.ts \
  e2e/request-lifecycle-cinematic.spec.ts
```

Expected: selectors/skips reveal missing seek, telemetry, and immersive shell
behavior. Do not delete assertions to obtain green.

- [ ] **Step 3: Add immersive behavior tests**

Cover Causal Replay, command search/actions, Context Lens, Focus World,
fullscreen fallback, URL restoration, keyboard traversal, captions, semantic
twin, SVG fallback, browser drawer, and mobile overflow.

- [ ] **Step 4: Add deterministic screenshot matrix**

Capture the twelve states in the approved design at fixed fonts, viewport,
device scale, scene, beat, random seed, and theme. Update snapshots only after
manual inspection.

- [ ] **Step 5: Wire live runtime metrics**

Expose test-only performance entries from the real evaluator/draw boundaries,
collect evaluation/draw/total samples in Playwright, and pass the JSON to
`measure-runtime.mjs`. Do not use synthetic numbers.

- [ ] **Step 6: Verify GREEN**

Run all E2E specs in Chromium. Expected: no capability-gap skips remain for
seek, quality variants, frame telemetry, keyboard traversal, or evidence UI.

---

### Task 10: Full verification, documentation sync, and final review

**Files:**
- Modify only if runtime behavior differs from existing claims:
  - `docs/superpowers/plans/2026-07-17-aiperf-flow-live-cinematic-runtime.md`
  - `docs/superpowers/plans/2026-07-17-aiperf-flow-browser-preview.md`
  - `docs/superpowers/plans/2026-07-17-aiperf-flow-display-list.md`
- Update: `.superpowers/sdd/progress.md` (ignored durable ledger)

- [ ] **Step 1: Run full Flow verification**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm run flow:check
npm run e2e
node --test scripts/measure-runtime.test.mjs
```

Expected: all unit tests, strict builds, Chromium E2E, screenshots, and
measurement tests pass.

- [ ] **Step 2: Run repository documentation checks**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/rust
/usr/bin/python3 tools/check_agent_files_sync.py
/usr/bin/python3 tools/check_docs_current.py
```

Expected: both exit zero and framing scan has no matches.

- [ ] **Step 3: Run lint diagnostics and diff checks**

Read lints for every changed source/test file and run:

```bash
git diff --check
```

Expected: no new diagnostics or whitespace errors.

- [ ] **Step 4: Review requirements line by line**

Confirm each acceptance criterion in
`docs/superpowers/specs/2026-07-18-aiperf-flow-immersive-preview-design.md`
has a passing test or inspected screenshot. Record any hardware-sensitive
performance result with environment metadata.

- [ ] **Step 5: Run broad final code review**

Review the complete diff from the merge base, fix every Critical/Important
finding in one wave, rerun covering tests, and re-review until approved.

---

## Dependency Order

```text
Task 1 evaluateFrame
  ├── Task 2 Causal Replay
  ├── Task 3 immersive state
  └── Task 8 URL/fullscreen/HUD pure policies

Task 2 + Task 3 ──> Task 4 Command Constellation
Task 2 + Task 3 ──> Task 5 immersive components
Tasks 1–5 + Task 8 ──> Task 6 FlowApp integration
Task 6 ──> Task 7 preview convergence
Task 7 ──> Task 9 Playwright and quality gates
Task 9 ──> Task 10 final verification
```

Tasks 1–5 and Task 8 may be implemented in isolated worktrees or with strict
file ownership. Tasks 6, 7, 9, and 10 are integration tasks and run
sequentially against the combined state.

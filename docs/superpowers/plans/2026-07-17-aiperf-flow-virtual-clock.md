# AIPerf Flow Virtual Clock and Pause-to-Explore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden `@aiperf/flow-runtime` so all authored lesson time is integer
virtual milliseconds, direct seek equals continuous playback at the same beat,
and pause-to-explore freezes the lesson then resumes from the exact paused beat
via `ExplorationSnapshot` / `resumeLesson()`.

**Architecture:** Keep `TimelinePlayer` as the pure clock + cue evaluator.
Introduce `exploration.ts` as the pause-to-explore orchestrator that freezes one
integer beat, holds serializable temporary viewport/interaction overrides, and
restores authored camera policy on resume. Extend `store.ts` so
`PlaybackStatus` and scene actions can represent exploration without mounting
UI. Wall-clock `Clock.nowNs()` advances virtual time but never becomes scene
state.

**Tech Stack:** TypeScript strict mode, Vitest, existing `@aiperf/flow-schema`
timeline IR, `@aiperf/flow-runtime` player/store.

**Parent plan:** Expands
[live-cinematic-runtime.md Task 2](2026-07-17-aiperf-flow-live-cinematic-runtime.md#task-2-harden-deterministic-time-and-pause-to-explore-state)
into a standalone reviewable vertical slice. Shell mount (`FlowApp` Task 6),
Canvas, semantic twin, and SVG fallback remain owned by the parent plan.

## Global Constraints

- Scope is `packages/runtime` player, store, exploration, and their tests only.
- **Forbidden:** `apps/aiperf-flow/preview/**` — do not create, edit, or import
  preview shell code.
- Do not mount exploration controls in `app.tsx` / `site.tsx` in this plan;
  export APIs for live-cinematic Task 6 to consume.
- All public timeline times are **non-negative safe integers** in milliseconds
  (`timeMs` / `pausedAtMs`). Wall time stays nanoseconds on `Clock` only.
- Direct seek to `T` and continuous play advancing to `T` must produce equal
  `TimelineSnapshot` values (deep equality of `timeMs`, `complete`, `targets`).
- Exploration pauses narration, captions, camera, and visual tracks at one
  integer timestamp; temporary pan/zoom/selection/focus/compare/inspector state
  is serializable JSON and reversible.
- `resumeLesson()` continues from the exact paused beat — no rewind to 0, no
  skipped narration cue, no silent restart.
- Reduced-motion resume uses cut or crossfade camera policy while preserving
  semantic beat state.
- Activate `.venv` before repo commands:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Verify with focused Vitest then `cd apps/aiperf-flow && npm run flow:check`.
- Do not create git commits unless the user explicitly requests them.

## File structure

```text
apps/aiperf-flow/packages/runtime/
├── src/
│   ├── player.ts              # integer virtual time + TimelineSnapshot
│   ├── store.ts               # exploring status + exploration actions
│   ├── exploration.ts         # NEW: pause-to-explore controller
│   └── index.ts               # re-export exploration APIs
└── test/
    ├── player.test.ts         # existing baseline (keep green)
    ├── player-determinism.test.ts   # NEW: seek ≡ play
    ├── exploration.test.ts          # NEW: begin/update/resume
    └── store-exploration.test.ts    # NEW: reducer wiring
```

## Contract overview

```text
Clock.nowNs()  ──advances──▶  TimelinePlayer (integer timeMs)
                                    │
                                    ▼
                            TimelineSnapshot
                                    │
                 beginExploration() │  resumeLesson()
                                    ▼
                         ExplorationSnapshot
                    (frozen beat + authored camera
                     + temporary interaction overrides)
                                    │
                                    ▼
                              store.ts SceneState
                         (playbackStatus: exploring)
```

---

### Task 1: Integer virtual-time contract in `TimelinePlayer`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/player.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/player.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/helpers/manual-clock.ts`

**Interfaces:**
- Consumes: existing `Clock`, `SceneIr["timeline"]`.
- Produces: `normalizeTimeMs(value: number): number`, integer
  `TimelineSnapshot.timeMs`, integer `currentTimeMs()`, integer seek clamp.

- [ ] **Step 1: Write the failing integer-time tests**

Extract `ManualClock` from `player.test.ts` into
`test/helpers/manual-clock.ts` and import it from both player suites.

Add to `player.test.ts`:

```ts
test("emits only non-negative safe-integer timeMs", () => {
  const clock = new ManualClock();
  const player = new TimelinePlayer(timeline, clock);
  player.play();
  clock.advanceMs(250);
  const live = player.currentTimeMs();
  expect(Number.isSafeInteger(live)).toBe(true);
  expect(player.snapshot().timeMs).toBe(live);
  expect(player.seek(250.7).timeMs).toBe(251); // round half-up / Math.round
  expect(player.seek(-3).timeMs).toBe(0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- player.test.ts
```

Expected: FAIL — live time remains float (`250` may pass after 250 ms step, but
`seek(250.7)` still returns a float or unrounded value).

- [ ] **Step 3: Implement integer normalization**

In `player.ts`:

```ts
/** Normalize authored/playback time to a non-negative safe integer ms. */
export function normalizeTimeMs(value: number, durationMs = Number.POSITIVE_INFINITY): number {
  if (!Number.isFinite(value)) {
    return value === Number.POSITIVE_INFINITY
      ? (Number.isFinite(durationMs) ? durationMs : 0)
      : 0;
  }
  const rounded = Math.round(value);
  const clamped = Math.min(
    Number.isFinite(durationMs) ? durationMs : rounded,
    Math.max(0, rounded),
  );
  return clamped;
}
```

Apply `normalizeTimeMs` in `#liveTimeMs()`, `seek()`, `#emit()`/`#compute()`,
and when freezing on `pause()`. Keep wall-clock math in nanoseconds; only the
published virtual timeline is integer ms.

- [ ] **Step 4: Run tests to verify they pass**

Run the same `player.test.ts` command. Expected: PASS. Existing cue-progress
assertions must still hold (250 ms → cli progress 0.375).

- [ ] **Step 5: Record the integer-time checkpoint**

Record changed files and passing commands in the implementation report. Create
a commit only if the user explicitly requests one.

---

### Task 2: Direct seek ≡ continuous play determinism

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/player.ts` (only if parity fails)
- Create: `apps/aiperf-flow/packages/runtime/test/player-determinism.test.ts`

**Interfaces:**
- Consumes: `TimelinePlayer`, `ManualClock`, shared foundation timeline fixture.
- Produces: proven equality of `TimelineSnapshot` for seek vs play at each beat.

- [ ] **Step 1: Write failing seek≡play suite**

```ts
import { describe, expect, test } from "vitest";
import { TimelinePlayer, type TimelineSnapshot } from "../src/player.js";
import { ManualClock } from "./helpers/manual-clock.js";
import type { SceneIr } from "@aiperf/flow-schema";

const timeline = [/* same cues as player.test.ts */] satisfies SceneIr["timeline"];

function playTo(timeMs: number): TimelineSnapshot {
  const clock = new ManualClock();
  const player = new TimelinePlayer(timeline, clock);
  player.play();
  clock.advanceMs(timeMs);
  player.pause();
  return player.snapshot();
}

function seekTo(timeMs: number): TimelineSnapshot {
  return new TimelinePlayer(timeline, new ManualClock()).seek(timeMs);
}

function beats(): number[] {
  const points = new Set<number>([0]);
  for (const cue of timeline) {
    points.add(cue.at);
    points.add(cue.at + cue.duration);
    for (const fraction of [0.25, 0.5, 0.75]) {
      points.add(Math.round(cue.at + cue.duration * fraction));
    }
  }
  points.add(2_000); // authored duration
  return [...points].sort((a, b) => a - b);
}

describe("TimelinePlayer determinism", () => {
  test.each(beats())(
    "direct seek equals continuous play at %i ms",
    (timeMs) => {
      expect(seekTo(timeMs)).toEqual(playTo(timeMs));
    },
  );

  test("repeated seek is idempotent", () => {
    const player = new TimelinePlayer(timeline, new ManualClock());
    expect(player.seek(1_400)).toEqual(player.seek(1_400));
  });

  test("pause freezes snapshot while wall clock advances", () => {
    const clock = new ManualClock();
    const player = new TimelinePlayer(timeline, clock);
    player.play();
    clock.advanceMs(300);
    const paused = player.pause();
    clock.advanceMs(5_000);
    expect(player.snapshot()).toEqual(paused);
  });
});
```

- [ ] **Step 2: Run test to verify failures (if any) are real**

Run:

```bash
npm test -w @aiperf/flow-runtime -- player-determinism.test.ts
```

Expected: either PASS after Task 1, or FAIL only on float drift / end-of-range
mismatch. Fix in `player.ts` until every beat matches — do not weaken
assertions.

- [ ] **Step 3: Harden end-of-timeline play semantics**

If `play()` currently rewinds `#timeMs` to `0` when already complete, change it
so:

- `play()` at `complete` without an explicit `reset()` is a no-op that returns
  the final snapshot (supports exact-beat resume at the last frame);
- `reset()` remains the only path that restarts from 0.

Add:

```ts
test("play at complete does not rewind without reset", () => {
  const player = new TimelinePlayer(timeline, new ManualClock());
  player.seek(2_000);
  expect(player.play().timeMs).toBe(2_000);
  expect(player.play().complete).toBe(true);
});
```

- [ ] **Step 4: Re-run determinism suite — Expected: PASS**

---

### Task 3: `ExplorationSnapshot` types and controller scaffold

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/exploration.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/exploration.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`

**Interfaces:**
- Consumes: `TimelinePlayer`, `TimelineSnapshot`.
- Produces: `ViewportOverride`, `ExplorationInteractionState`,
  `AuthoredCameraSnapshot`, `ExplorationSnapshot`,
  `createExplorationController()`, `beginExploration`, `updateExploration`,
  `resumeLesson`.

- [ ] **Step 1: Write failing API tests**

```ts
import { describe, expect, test } from "vitest";
import { createExplorationController } from "../src/exploration.js";
import { TimelinePlayer } from "../src/player.js";
import { ManualClock } from "./helpers/manual-clock.js";

const timeline = [/* shared fixture */];

test("beginExploration freezes an integer beat and pauses the player", () => {
  const clock = new ManualClock();
  const player = new TimelinePlayer(timeline, clock);
  const exploration = createExplorationController();
  player.play();
  clock.advanceMs(300);
  const snap = exploration.beginExploration(player, {
    authoredCamera: {
      transform: { x: 0, y: 0, scale: 1 },
      policy: "smooth",
    },
    narrationBeatId: "reveal-cli",
    captionIndex: 0,
  });
  expect(snap.pausedAtMs).toBe(300);
  expect(Number.isSafeInteger(snap.pausedAtMs)).toBe(true);
  expect(exploration.isExploring()).toBe(true);
  clock.advanceMs(1_000);
  expect(player.currentTimeMs()).toBe(300);
  expect(exploration.snapshot()?.timeline.timeMs).toBe(300);
});
```

- [ ] **Step 2: Run test — Expected: FAIL (module missing)**

```bash
npm test -w @aiperf/flow-runtime -- exploration.test.ts
```

- [ ] **Step 3: Implement types + controller**

```ts
// exploration.ts
import {
  normalizeTimeMs,
  type TimelinePlayer,
  type TimelineSnapshot,
} from "./player.js";

export type ViewportOverride = Readonly<{
  panX: number;
  panY: number;
  zoom: number;
  fit?: boolean;
}>;

export type ExplorationInteractionState = Readonly<{
  selectedNodeId: string | null;
  focusedEntityId: string | null;
  inspector: Readonly<{ open: boolean; nodeId: string | null }>;
  compareTargetId: string | null;
  viewport: ViewportOverride | null;
}>;

export type AuthoredCameraSnapshot = Readonly<{
  transform: Readonly<{ x: number; y: number; scale: number }>;
  policy: "smooth" | "cut" | "crossfade";
}>;

export type ExplorationSnapshot = Readonly<{
  pausedAtMs: number;
  timeline: TimelineSnapshot;
  authoredCamera: AuthoredCameraSnapshot;
  narrationBeatId: string | null;
  captionIndex: number | null;
  interaction: ExplorationInteractionState;
}>;

const emptyInteraction = (): ExplorationInteractionState =>
  Object.freeze({
    selectedNodeId: null,
    focusedEntityId: null,
    inspector: Object.freeze({ open: false, nodeId: null }),
    compareTargetId: null,
    viewport: null,
  });

export function createExplorationController() {
  let active: ExplorationSnapshot | null = null;

  return {
    isExploring(): boolean {
      return active !== null;
    },
    snapshot(): ExplorationSnapshot | null {
      return active;
    },
    beginExploration(
      player: TimelinePlayer,
      context: Readonly<{
        authoredCamera: AuthoredCameraSnapshot;
        narrationBeatId?: string | null;
        captionIndex?: number | null;
        interaction?: Partial<ExplorationInteractionState>;
      }>,
    ): ExplorationSnapshot {
      const timeline = player.pause();
      const pausedAtMs = normalizeTimeMs(timeline.timeMs);
      active = Object.freeze({
        pausedAtMs,
        timeline: player.seek(pausedAtMs),
        authoredCamera: context.authoredCamera,
        narrationBeatId: context.narrationBeatId ?? null,
        captionIndex: context.captionIndex ?? null,
        interaction: Object.freeze({
          ...emptyInteraction(),
          ...context.interaction,
          inspector: Object.freeze({
            ...emptyInteraction().inspector,
            ...context.interaction?.inspector,
          }),
        }),
      });
      return active;
    },
    updateExploration(
      patch: Partial<ExplorationInteractionState>,
    ): ExplorationSnapshot {
      if (active === null) {
        throw new Error("updateExploration requires an active exploration");
      }
      active = Object.freeze({
        ...active,
        interaction: Object.freeze({
          ...active.interaction,
          ...patch,
          inspector: Object.freeze({
            ...active.interaction.inspector,
            ...patch.inspector,
          }),
        }),
      });
      return active;
    },
    resumeLesson(
      player: TimelinePlayer,
      _options?: Readonly<{ reducedMotion?: boolean }>,
    ): TimelineSnapshot {
      if (active === null) {
        return player.snapshot();
      }
      const beat = active.pausedAtMs;
      active = null;
      player.seek(beat);
      return player.play();
    },
  };
}
```

Export from `index.ts`: `export * from "./exploration.js";`

- [ ] **Step 4: Run exploration tests — Expected: PASS for freeze behavior**

---

### Task 4: Serializable overrides + exact-beat `resumeLesson`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/exploration.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/exploration.test.ts`

**Interfaces:**
- Consumes: Task 3 controller.
- Produces: JSON-round-tripable `ExplorationSnapshot`; resume that neither
  replays nor skips the frozen beat.

- [ ] **Step 1: Write failing serialization and resume tests**

```ts
test("exploration interaction state survives JSON round-trip", () => {
  const clock = new ManualClock();
  const player = new TimelinePlayer(timeline, clock);
  const exploration = createExplorationController();
  player.seek(800);
  exploration.beginExploration(player, {
    authoredCamera: { transform: { x: 10, y: 20, scale: 1 }, policy: "cut" },
  });
  const updated = exploration.updateExploration({
    selectedNodeId: "cli",
    focusedEntityId: "cli",
    compareTargetId: "spawn",
    viewport: { panX: 40, panY: -12, zoom: 1.5, fit: false },
    inspector: { open: true, nodeId: "cli" },
  });
  expect(JSON.parse(JSON.stringify(updated))).toEqual(updated);
  expect(updated.pausedAtMs).toBe(800);
});

test("updateExploration does not advance the paused beat", () => {
  const clock = new ManualClock();
  const player = new TimelinePlayer(timeline, clock);
  const exploration = createExplorationController();
  player.seek(400);
  exploration.beginExploration(player, {
    authoredCamera: { transform: { x: 0, y: 0, scale: 1 }, policy: "smooth" },
  });
  exploration.updateExploration({ viewport: { panX: 1, panY: 2, zoom: 2 } });
  clock.advanceMs(500);
  expect(exploration.snapshot()?.pausedAtMs).toBe(400);
  expect(player.currentTimeMs()).toBe(400);
});

test("resumeLesson continues from the exact paused beat", () => {
  const clock = new ManualClock();
  const player = new TimelinePlayer(timeline, clock);
  const exploration = createExplorationController();
  player.play();
  clock.advanceMs(300);
  const frozen = exploration.beginExploration(player, {
    authoredCamera: { transform: { x: 0, y: 0, scale: 1 }, policy: "smooth" },
    narrationBeatId: "reveal-cli",
  });
  exploration.updateExploration({
    viewport: { panX: 100, panY: 0, zoom: 2 },
  });
  const resumed = exploration.resumeLesson(player);
  expect(exploration.isExploring()).toBe(false);
  expect(resumed.timeMs).toBe(frozen.pausedAtMs);
  clock.advanceMs(100);
  expect(player.currentTimeMs()).toBe(400);
});
```

- [ ] **Step 2: Run tests — Expected: FAIL until resume seeks before play**

- [ ] **Step 3: Implement resume path**

Ensure `resumeLesson`:

1. reads `pausedAtMs` before clearing `active`;
2. clears temporary interaction overrides by dropping `active`;
3. `player.seek(pausedAtMs)` then `player.play()`;
4. returns the post-resume `TimelineSnapshot` at that beat;
5. never calls `reset()`.

Return camera restoration metadata for Task 5 via optional result fields or a
companion helper `cameraRestorePlan(snapshot, options)` if cleaner than
overloading `resumeLesson`.

- [ ] **Step 4: Run exploration tests — Expected: PASS**

---

### Task 5: Authored-camera restoration and reduced-motion policy

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/exploration.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/exploration.test.ts`

**Interfaces:**
- Consumes: `AuthoredCameraSnapshot.policy`, `reducedMotion` flag.
- Produces: `CameraRestorePlan` with `mode: "smooth" | "cut" | "crossfade"` and
  target transform equal to authored camera.

- [ ] **Step 1: Write failing camera-policy tests**

```ts
import {
  createExplorationController,
  cameraRestorePlan,
} from "../src/exploration.js";

test("resume restores authored camera transform", () => {
  const authored = {
    transform: { x: 64, y: 32, scale: 1.25 },
    policy: "smooth" as const,
  };
  const exploration = createExplorationController();
  const player = new TimelinePlayer(timeline, new ManualClock());
  player.seek(500);
  exploration.beginExploration(player, { authoredCamera: authored });
  exploration.updateExploration({
    viewport: { panX: 999, panY: 999, zoom: 4 },
  });
  const snap = exploration.snapshot();
  expect(snap).not.toBeNull();
  const plan = cameraRestorePlan(snap!, { reducedMotion: false });
  expect(plan.target).toEqual(authored.transform);
  expect(plan.mode).toBe("smooth");
});

test("reduced-motion resume uses cut instead of smooth", () => {
  const authored = {
    transform: { x: 0, y: 0, scale: 1 },
    policy: "smooth" as const,
  };
  const player = new TimelinePlayer(timeline, new ManualClock());
  const exploration = createExplorationController();
  player.seek(100);
  const snap = exploration.beginExploration(player, {
    authoredCamera: authored,
  });
  expect(cameraRestorePlan(snap, { reducedMotion: true }).mode).toBe("cut");
});

test("reduced-motion preserves semantic beat and narration marker", () => {
  const player = new TimelinePlayer(timeline, new ManualClock());
  const exploration = createExplorationController();
  player.seek(800);
  const snap = exploration.beginExploration(player, {
    authoredCamera: {
      transform: { x: 1, y: 2, scale: 1 },
      policy: "crossfade",
    },
    narrationBeatId: "trace-spawn",
    captionIndex: 1,
  });
  const resumed = exploration.resumeLesson(player, { reducedMotion: true });
  expect(resumed.timeMs).toBe(800);
  expect(snap.narrationBeatId).toBe("trace-spawn");
  expect(snap.captionIndex).toBe(1);
});
```

- [ ] **Step 2: Run tests — Expected: FAIL (`cameraRestorePlan` missing)**

- [ ] **Step 3: Implement policy helper**

```ts
export type CameraRestorePlan = Readonly<{
  target: AuthoredCameraSnapshot["transform"];
  mode: AuthoredCameraSnapshot["policy"];
}>;

export function cameraRestorePlan(
  snapshot: ExplorationSnapshot,
  options: Readonly<{ reducedMotion?: boolean }> = {},
): CameraRestorePlan {
  const mode =
    options.reducedMotion === true && snapshot.authoredCamera.policy === "smooth"
      ? "cut"
      : snapshot.authoredCamera.policy;
  return Object.freeze({
    target: snapshot.authoredCamera.transform,
    mode,
  });
}
```

Call `cameraRestorePlan` from `resumeLesson` when options are provided so
callers can observe the plan without a separate step; keep the pure helper
exported for unit tests and for live-cinematic Task 6.

- [ ] **Step 4: Run exploration tests — Expected: PASS**

---

### Task 6: Wire exploration into `store.ts`

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/store.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/store-exploration.test.ts`

**Interfaces:**
- Consumes: `ExplorationSnapshot`.
- Produces: `PlaybackStatus` including `"exploring"`; actions
  `begin-exploration`, `update-exploration`, `resume-lesson`.

- [ ] **Step 1: Write failing reducer tests**

```ts
import { describe, expect, test } from "vitest";
import {
  createInitialSceneState,
  sceneReducer,
} from "../src/store.js";
import type { ExplorationSnapshot } from "../src/exploration.js";

const sampleExploration = {
  pausedAtMs: 300,
  timeline: { timeMs: 300, complete: false, targets: {} },
  authoredCamera: {
    transform: { x: 0, y: 0, scale: 1 },
    policy: "smooth",
  },
  narrationBeatId: null,
  captionIndex: null,
  interaction: {
    selectedNodeId: null,
    focusedEntityId: null,
    inspector: { open: false, nodeId: null },
    compareTargetId: null,
    viewport: null,
  },
} satisfies ExplorationSnapshot;

test("begin-exploration records snapshot and exploring status", () => {
  const state = sceneReducer(createInitialSceneState("s1"), {
    type: "begin-exploration",
    exploration: sampleExploration,
  });
  expect(state.playbackStatus).toBe("exploring");
  expect(state.playbackTimeMs).toBe(300);
  expect(state.exploration).toEqual(sampleExploration);
  expect(state.temporaryCameraTakeover).toBe(true);
});

test("update-exploration patches interaction only", () => {
  const exploring = sceneReducer(createInitialSceneState("s1"), {
    type: "begin-exploration",
    exploration: sampleExploration,
  });
  const next = sceneReducer(exploring, {
    type: "update-exploration",
    interaction: { selectedNodeId: "cli" },
  });
  expect(next.exploration?.pausedAtMs).toBe(300);
  expect(next.exploration?.interaction.selectedNodeId).toBe("cli");
});

test("resume-lesson clears exploration and restores paused beat", () => {
  const exploring = sceneReducer(createInitialSceneState("s1"), {
    type: "begin-exploration",
    exploration: sampleExploration,
  });
  const next = sceneReducer(exploring, {
    type: "resume-lesson",
    timeMs: 300,
    status: "playing",
  });
  expect(next.exploration).toBeNull();
  expect(next.playbackTimeMs).toBe(300);
  expect(next.playbackStatus).toBe("playing");
  expect(next.temporaryCameraTakeover).toBe(false);
});

test("change-scene clears exploration", () => {
  const exploring = sceneReducer(createInitialSceneState("s1"), {
    type: "begin-exploration",
    exploration: sampleExploration,
  });
  const next = sceneReducer(exploring, {
    type: "change-scene",
    sceneId: "s2",
  });
  expect(next.exploration).toBeNull();
  expect(next.playbackStatus).toBe("idle");
});
```

- [ ] **Step 2: Run tests — Expected: FAIL on missing fields/actions**

```bash
npm test -w @aiperf/flow-runtime -- store-exploration.test.ts
```

- [ ] **Step 3: Extend store**

```ts
import type { ExplorationInteractionState, ExplorationSnapshot } from "./exploration.js";

export type PlaybackStatus =
  | "idle"
  | "playing"
  | "paused"
  | "exploring"
  | "complete";

export type SceneState = Readonly<{
  currentSceneId: string;
  selectedNodeId: string | null;
  inspector: InspectorState;
  playbackTimeMs: number;
  playbackStatus: PlaybackStatus;
  activeResponsiveVariant: string | null;
  temporaryCameraTakeover: boolean;
  exploration: ExplorationSnapshot | null;
}>;

export type SceneAction =
  | /* existing actions */
  | Readonly<{ type: "begin-exploration"; exploration: ExplorationSnapshot }>
  | Readonly<{
      type: "update-exploration";
      interaction: Partial<ExplorationInteractionState>;
    }>
  | Readonly<{
      type: "resume-lesson";
      timeMs: number;
      status: Exclude<PlaybackStatus, "exploring">;
    }>;
```

Initialize `exploration: null` in `createInitialSceneState`. Implement reducer
cases to match the tests above. Keep `set-playback` working for non-exploration
updates.

- [ ] **Step 4: Run store + exploration + player suites — Expected: PASS**

```bash
npm test -w @aiperf/flow-runtime -- player.test.ts player-determinism.test.ts exploration.test.ts store-exploration.test.ts
```

---

### Task 7: Package export and verification gate

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts` (if needed)
- Verify only: no `preview/**` edits

**Interfaces:**
- Produces: public exports for `normalizeTimeMs`, exploration types/helpers,
  and store exploration actions usable by live-cinematic Task 6.

- [ ] **Step 1: Confirm exports**

`index.ts` must re-export:

- `./player.js` (including `normalizeTimeMs`, `TimelineSnapshot`)
- `./exploration.js`
- `./store.js`

- [ ] **Step 2: Grep guard for forbidden preview edits**

```bash
git status --short apps/aiperf-flow/preview
```

Expected: no changes under `preview/`.

- [ ] **Step 3: Full package verification**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime -- player.test.ts player-determinism.test.ts exploration.test.ts store-exploration.test.ts
npm run flow:check
```

Expected: focused suites green; workspace `flow:check` green.

- [ ] **Step 4: Mark parent Task 2 checkboxes satisfied**

In
[`2026-07-17-aiperf-flow-live-cinematic-runtime.md`](2026-07-17-aiperf-flow-live-cinematic-runtime.md),
Task 2’s checkboxes are considered delivered by this plan’s Tasks 1–6. Do not
duplicate mount work from parent Task 6.

---

## Out of scope (intentionally deferred)

- Wiring `FlowApp` / `site.tsx` explore + resume chrome (parent Task 6).
- Canvas camera interpolation animation frames (parent Task 3 / Plan 6).
- Semantic-twin focus coordinator coupling beyond `focusedEntityId` fields
  (parent Task 4).
- Evaluated-scene / display-list seek-parity golden hashes (owned by
  [`2026-07-17-aiperf-flow-display-list.md`](2026-07-17-aiperf-flow-display-list.md)
  Task 11) — this plan proves `TimelineSnapshot` equality; display hashes reuse
  the same integer `timeMs`.
- Any edits under `apps/aiperf-flow/preview/**`.

## Spec coverage checklist

| Requirement | Task |
|---|---|
| Integer virtual time | Task 1 |
| Direct seek ≡ continuous play | Task 2 |
| `TimelineSnapshot` integer contract | Tasks 1–2 |
| Pause freezes narration/camera/visual beat | Tasks 3–4 |
| Serializable pan/zoom/selection/focus/compare/inspector | Task 4 |
| `beginExploration` / `updateExploration` / `resumeLesson` | Tasks 3–4 |
| Authored camera restoration | Task 5 |
| Reduced-motion cut/crossfade | Task 5 |
| Store exploring status + actions | Task 6 |
| Export + `flow:check` | Task 7 |
| Forbidden `preview/**` | Global + Task 7 |

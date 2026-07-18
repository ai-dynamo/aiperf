# AIPerf Flow Virtual Clock and Pause-to-Explore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the landed integer-time and exploration seams so direct seek
equals continuous playback at the same beat and pause-to-explore resumes the
authored lesson without narration, semantic, or camera drift.

**Architecture:** `TimelinePlayer` is the pure clock and cue evaluator.
`normalizeSceneTimeMs` truncates public scene time to non-negative safe-integer
milliseconds. `exploration.ts` snapshots authored and temporary `SceneState`
through standalone pure functions. `camera-policy.ts` owns camera restoration,
and `FlowApp` owns the active exploration snapshot in React state. Wall-clock
`Clock.nowNs()` advances virtual time but never becomes scene state.

**Tech Stack:** TypeScript strict mode, Vitest, existing Flow timeline IR and
`@aiperf/flow-runtime`.

## Global Constraints

- Public lesson time uses non-negative safe-integer milliseconds (`timeMs`).
- `Clock.nowNs(): bigint` is wall-time input only.
- `normalizeSceneTimeMs` uses truncation: `250.7 → 250`.
- Direct seek and continuous playback to `T` produce deeply equal
  `TimelineSnapshot` values.
- Exploration freezes narration, captions, camera, visual tracks, focus,
  selection, compare state, and inspector state at one beat.
- `beginExploration`, `updateExploration`, and `resumeLesson` remain standalone
  pure functions over `SceneState`.
- `ExplorationSnapshot` remains
  `{ authored: SceneState; exploration: SceneState }`.
- Camera restoration remains in `camera-policy.ts`.
- `FlowApp` owns exploration lifecycle; do not add a competing controller,
  `"exploring"` store status, or exploration reducer actions.
- Reduced motion uses cut or crossfade restoration without changing semantic
  state or the resumed beat.
- Do not modify `apps/aiperf-flow/preview/**`.
- Activate the environment before every command:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Do not create git commits unless the user explicitly requests them.

## Current implementation baseline

Landed APIs:

```ts
normalizeSceneTimeMs(value: number): number
beginExploration(state: SceneState): ExplorationSnapshot
updateExploration(
  snapshot: ExplorationSnapshot,
  state: SceneState,
): ExplorationSnapshot
resumeLesson(snapshot: ExplorationSnapshot): SceneState
```

`FlowApp` uses these functions and camera policy through component state.
This plan adds determinism and integration guarantees around that shape; it
does not replace it.

---

### Task 1: Lock integer-time normalization

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/player.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/player.test.ts`

**Interfaces:**
- Preserves: `normalizeSceneTimeMs`, `TimelinePlayer`, `TimelineSnapshot`.

- [ ] Add parameterized tests for negative, fractional, exact integer, duration
  boundary, non-finite, and unsafe-integer inputs.
- [ ] Assert fractional values truncate and published snapshots contain only
  safe-integer `timeMs`.
- [ ] Assert wall-clock nanoseconds are converted only when advancing virtual
  time and never serialized into scene state.
- [ ] Run `npm test -w @aiperf/flow-runtime -- player.test.ts`.

---

### Task 2: Prove direct seek equals continuous playback

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/test/player-determinism.test.ts`

**Interfaces:**
- Consumes: `TimelinePlayer`, manual `Clock`, representative timeline fixture.

- [ ] Compare direct seek with continuous playback at zero, cue starts,
  cue interiors, cue ends, overlapping cues, scene end, and post-end clamping.
- [ ] Deep-compare `timeMs`, completion, active cues, target state, narration
  cue, caption cue, and camera cue.
- [ ] Repeat seeks in non-monotonic order to prove no hidden history affects
  evaluation.
- [ ] Run the focused player determinism test.

---

### Task 3: Harden the landed exploration snapshot

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/exploration.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/exploration.test.ts`

**Interfaces:**
- Preserves: standalone `beginExploration`, `updateExploration`,
  `resumeLesson`, and `{ authored, exploration }`.

- [ ] Assert begin snapshots the authored state without mutation.
- [ ] Assert updates can change temporary camera, focus, selection, comparison,
  and inspector state while preserving the authored snapshot.
- [ ] Assert resume returns the authored beat and authored semantic state
  exactly, without resetting to zero or retaining temporary overrides.
- [ ] Assert snapshots are finite, serializable JSON data.
- [ ] Run the focused exploration test.

---

### Task 4: Verify camera restoration policy

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/camera-policy.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/camera-policy.test.ts`

**Interfaces:**
- Consumes: authored camera, exploration camera, reduced-motion preference.
- Produces: deterministic cut, crossfade, or guided rejoin policy.

- [ ] Assert normal mode rejoins the authored camera without changing the
  paused beat.
- [ ] Assert reduced motion chooses cut or crossfade and never spatial flight.
- [ ] Assert no-depth mode removes depth interpolation but preserves framing.
- [ ] Assert restoration policy reads no wall clock.

---

### Task 5: Verify mounted `FlowApp` pause-to-explore behavior

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/app.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/test/app.test.tsx`

**Interfaces:**
- Consumes: player, exploration functions, camera policy, Canvas/SVG visual
  backend, semantic twin, transcript, and captions.

- [ ] Start exploration through pointer, keyboard, and semantic-twin activation;
  assert playback, narration, captions, camera, and visuals freeze together.
- [ ] Explore by pan, zoom, focus, selection, comparison, and inspection; assert
  no authored time advances.
- [ ] Resume and assert exact `timeMs`, narration cue, caption cue, semantic
  selection, and authored camera restoration.
- [ ] Repeat with reduced motion and SVG fallback.
- [ ] Verify no exploration state is owned by preview chrome.

---

### Task 6: Complete regression and package gates

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/src/index.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/player-determinism.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/exploration.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/test/app.test.tsx`

- [ ] Export only the landed public names; no compatibility aliases for
  abandoned controller/store APIs.
- [ ] Search runtime source and tests for `createExplorationController`,
  `pausedAtMs`, `cameraRestorePlan`, or an `"exploring"` playback status; expect
  no matches.
- [ ] Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm test -w @aiperf/flow-runtime
npm run flow:check
```

- [ ] Confirm direct-seek, exploration, mounted-app, reduced-motion,
  cross-backend, and accessibility tests pass.

## Completion gate

This plan is complete when public lesson time is deterministic safe-integer
milliseconds; direct seek equals continuous playback; exploration is temporary,
serializable, and reversible; and every mounted backend resumes narration,
semantics, and camera from the exact authored beat.

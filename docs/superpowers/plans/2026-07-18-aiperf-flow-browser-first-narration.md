<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Browser-First Narration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start browser narration immediately while Kokoro prewarms, then route new cues to Kokoro after it becomes ready.

**Architecture:** `KokoroNarratorBackend` remains the single hybrid policy boundary. It sends cues to its browser fallback until the local model is ready, tracks browser ownership for transport controls, and changes routing only at a subsequent cue boundary. `FlowApp` starts the narration timeline immediately after the consent gesture instead of awaiting local-model prewarm.

**Tech Stack:** TypeScript, React, Web Speech API, Kokoro/ONNX worker, Vitest, Testing Library.

## Global Constraints

- Never interrupt an active browser cue solely because Kokoro becomes ready.
- If browser speech is unavailable, preserve the existing wait-for-Kokoro path.
- Kokoro failure must leave browser narration operational.
- Preserve timeline cancellation semantics during seeks and cue changes.

---

### Task 1: Browser-first hybrid routing

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/test/narrative/kokoro-narrator.test.ts`
- Modify: `apps/aiperf-flow/packages/runtime/src/narrative/kokoro-narrator.ts`

**Interfaces:**
- Consumes: existing `NarratorBackend`, `KokoroWorkerPort`, and `prewarm(): Promise<void>`.
- Produces: unchanged public `KokoroNarratorBackend`; `speak()` routes to the browser until `#modelReady`, and transport methods follow the owner of the active cue.

- [ ] **Step 1: Write failing browser-first routing tests**

Add tests which call `speak()` before worker readiness and assert:

```typescript
expect(fallback.spoken).toEqual([utterance]);
expect(worker.sent).toContainEqual(
  expect.objectContaining({ type: "initialize" }),
);
expect(worker.sent).not.toContainEqual(
  expect.objectContaining({ type: "synthesize" }),
);
```

After emitting `ready`, assert that the first cue was not resubmitted and a
second cue is sent to Kokoro:

```typescript
worker.emit({ type: "ready", voices: [] });
backend.cancel();
backend.speak(nextUtterance);
expect(worker.sent).toContainEqual(
  expect.objectContaining({ type: "synthesize", cueId: nextUtterance.cueId }),
);
expect(fallback.spoken).toEqual([utterance]);
```

Also instrument fallback `pause`, `resume`, and `cancel` methods with spies and
assert pause/resume remain browser-owned when readiness arrives during a cue.

- [ ] **Step 2: Run the focused tests and verify the new expectations fail**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
npm test --workspace @aiperf/flow-runtime -- kokoro-narrator.test.ts
```

Expected: FAIL because the pre-ready cue is queued for Kokoro and browser
speech receives nothing.

- [ ] **Step 3: Implement cue-boundary routing**

Add a private browser-cue ownership flag. In `speak()`, when fallback speech is
available and the model is not ready, mark browser ownership, speak
immediately, and start `prewarm()` without queueing the cue. Leave readiness
handling non-interrupting. Route pause and resume to the browser while it owns
the cue. Reset ownership during explicit cancellation; cancellation continues
to stop both backends.

- [ ] **Step 4: Run focused tests**

Run the command from Step 2. Expected: PASS.

### Task 2: Immediate application timeline start

**Files:**
- Modify: `apps/aiperf-flow/packages/runtime/test/app-audio-consent.test.tsx`
- Modify: `apps/aiperf-flow/packages/runtime/src/app.tsx`

**Interfaces:**
- Consumes: existing `unlockSpeechFromGesture()` and narrator timeline.
- Produces: `chooseAudioConsent()` starts narration without waiting for
  `prewarm()` settlement.

- [ ] **Step 1: Write a failing unresolved-prewarm application test**

Make the mocked `prewarm()` return an unresolved promise, spy on backend
`speak()`, select “Play with audio,” and assert the first narration cue is
spoken before that promise resolves.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
npm test --workspace @aiperf/flow-runtime -- app-audio-consent.test.tsx
```

Expected: FAIL because `chooseAudioConsent()` currently starts the narrator in
a continuation attached to `prewarm()`.

- [ ] **Step 3: Start narration immediately**

Replace the conditional prewarm continuation with direct calls to
`narrator.seek(snapshot.timeMs)` and `narrator.play(snapshot.timeMs)`.
`unlockSpeechFromGesture()` continues to launch background prewarming.

- [ ] **Step 4: Run focused application tests**

Run the command from Step 2. Expected: PASS.

### Task 3: Regression verification

**Files:**
- Verify: `apps/aiperf-flow/packages/runtime/src/narrative/kokoro-narrator.ts`
- Verify: `apps/aiperf-flow/packages/runtime/src/app.tsx`

- [ ] **Step 1: Run runtime tests**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
npm test --workspace @aiperf/flow-runtime
```

Expected: all tests pass.

- [ ] **Step 2: Run type checking**

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
npm run typecheck --workspace @aiperf/flow-runtime
```

Expected: exit code 0.

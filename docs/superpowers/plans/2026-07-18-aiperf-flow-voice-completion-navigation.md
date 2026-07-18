<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Voice-Completion Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Advance to the next scene when its final audible narration cue ends normally.

**Architecture:** Add optional per-utterance completion to the transport-neutral narrator backend. The controller validates cue identity and emits scene completion only after its final cue; `FlowApp` owns navigation and stops at the final scene.

**Tech Stack:** TypeScript, React, Web Speech API, Web Audio API, Vitest, Testing Library.

## Global Constraints

- Cancellation must invalidate completion.
- Muted, unavailable, errored, and narration-free scenes must not auto-advance.
- Final-scene completion must stop rather than loop.
- Do not modify `apps/aiperf-flow/preview/`.

---

### Task 1: Backend completion

**Files:**
- Modify: `packages/runtime/src/narrative/narrator.ts`
- Modify: `packages/runtime/src/narrative/kokoro-narrator.ts`
- Test: `packages/runtime/test/narrative/narrator.test.ts`
- Test: `packages/runtime/test/narrative/kokoro-narrator.test.ts`

**Interfaces:**
- Produces: `NarratorBackend.speak(utterance, onComplete?)`.

- [ ] Add failing browser and Kokoro completion/cancellation tests.
- [ ] Run focused tests and confirm failures are caused by the missing callback.
- [ ] Wire browser `onend` and Kokoro source `onended` with cancellation guards.
- [ ] Run focused tests and confirm they pass.

### Task 2: Controller scene completion

**Files:**
- Modify: `packages/runtime/src/narrative/narrator.ts`
- Test: `packages/runtime/test/narrative/narrator.test.ts`

**Interfaces:**
- Produces: optional `onComplete` constructor callback emitted after the final cue.

- [ ] Add failing multi-cue and stale-completion tests.
- [ ] Implement cue-identity and final-cue validation.
- [ ] Run controller tests.

### Task 3: FlowApp navigation

**Files:**
- Modify: `packages/runtime/src/app.tsx`
- Test: `packages/runtime/test/app-narrative.test.tsx`

**Interfaces:**
- Consumes: controller scene-completion callback.
- Produces: next-scene navigation, with final-scene stop.

- [ ] Add failing next-scene and final-scene tests.
- [ ] Connect completion to `navigate(sceneIndex + 1)` when available.
- [ ] Run focused and package runtime tests.

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Browser-First Narration Design

## Goal

Begin audible narration immediately through the browser speech synthesis API
while Kokoro initializes in the background. Once Kokoro is ready, finish the
browser's active cue and use Kokoro for subsequent cues.

## Behavior

- Selecting audible playback activates browser speech and starts the current
  cue without waiting for Kokoro.
- Kokoro prewarms concurrently in its worker.
- A cue already submitted to browser speech is never interrupted merely
  because Kokoro becomes ready.
- The first cue submitted after Kokoro becomes ready uses Kokoro.
- Kokoro loading or inference failure leaves browser speech active.
- If browser speech is unavailable, narration waits for Kokoro as before.
- Pause, resume, cancel, and seek apply to the backend owning the active cue.
- Narrator state identifies browser speech during prewarm and Kokoro after
  handoff, allowing the existing UI to communicate enhanced-voice loading.

## Ownership

The hybrid policy belongs in `KokoroNarratorBackend`. `FlowApp` continues to
construct one narrator backend and the timeline remains unaware of backend
selection. The backend starts Kokoro prewarming independently, routes speech
to browser synthesis until Kokoro reports ready, and then routes new speech to
Kokoro.

The backend tracks whether browser speech owns an active cue. Kokoro readiness
changes routing for future calls only; it does not cancel browser synthesis.
Normal timeline cancellation still cancels both implementations so seeking or
stopping cannot leave stale audio.

## Failure Handling

Kokoro failure preserves browser operation and exposes the fallback state.
Browser speech errors do not prevent Kokoro initialization. If neither backend
is available, the existing unavailable behavior remains unchanged.

## Verification

Unit tests cover immediate browser dispatch, concurrent Kokoro initialization,
cue-boundary handoff, uninterrupted active browser speech, Kokoro failure,
browser-unavailable startup, and transport controls across the transition.
Application tests verify audible playback does not wait for prewarm.

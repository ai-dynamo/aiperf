<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Voice-Completion Navigation Design

## Goal

Advance to the next scene when the final audible narration cue finishes. Stop
on the final scene.

## Design

`NarratorBackend.speak()` accepts an optional completion callback. Browser
speech invokes it from `SpeechSynthesisUtterance.onend`; Kokoro invokes it when
its `AudioBufferSourceNode` ends. Cancellation invalidates pending callbacks so
seek, stop, mute, and navigation cannot trigger stale scene advancement.

`NarratorController` associates completion with a specific cue. It emits scene
completion only when the active cue finished normally and no later narration
cue remains. `FlowApp` handles that signal by navigating to the next scene and
does nothing on the final scene.

Muted playback, unavailable narration, browser speech errors, and scenes
without narration do not auto-advance.

## Verification

Tests cover browser and Kokoro normal completion, cancellation invalidation,
multi-cue scenes advancing only after the final cue, next-scene navigation,
and stopping on the final scene.

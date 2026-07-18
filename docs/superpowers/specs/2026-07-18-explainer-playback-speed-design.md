<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Explainer Playback Speed Controls

**Date:** 2026-07-18  
**Status:** Approved  

## Goal

One pill group controls slideshow wall-clock speed for both Scene IR timelines and Web Speech narration.

## Behavior

- Presets: `0.75×` · `1×` · `1.25×` · `1.5×` · `2×` (default `1×`)
- Persisted per deck as `${storagePrefix}:speed`
- SceneRenderer advances `playbackTimeMs` by wall elapsed × speed (playhead retained on speed change)
- `SpeechSynthesisUtterance.rate = clamp(1.08 × speed, 0.5, 2)`; silent/fallback timers scale by `1/speed`
- Changing speed while playing restarts the current utterance at the new rate; scene continues from the current playhead
- Duration labels use speed-adjusted estimates

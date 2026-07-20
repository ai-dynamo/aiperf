<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Karaoke Subtitles Restoration Design

## Goal

Restore visible karaoke subtitles to explainer playback. The subtitle renderer and
word-timing state still exist, but the renderer is currently reachable only inside
the closed Speaker notes panel.

## Design

`ExplainerShell` renders the existing `Subtitles` component in a full-width row
directly below the cinematic stage. The row appears after the start gate has been
completed and uses the current slide narration and `activeWordIndex` from
`useTimedSlideshow`.

The subtitle component is removed from Speaker notes so opening notes does not
duplicate the narration. Speaker notes retain the slide term, points, caption,
voice controls, timing details, and restart action.

No narration or timing behavior changes. Spoken, active, and pending words keep
their existing styles. Playback speed, pause and resume, silent playback, slide
navigation, and narration-enabled playback continue to use the shared slideshow
timing state.

## Layout

The subtitle row follows the stage in document order and spans the shell's
available content width. It does not overlay or obscure diagrams. Existing
subtitle styling remains the baseline; only the minimum layout adjustment needed
for full-width placement is added.

## Verification

Add a focused shell rendering test that confirms subtitles are visible after
playback starts without opening Speaker notes and are not duplicated in that
panel. Run the explainer app's targeted tests, type checking, and lint checks
available in its package scripts.

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Explainer Scene Visual Polish Design

**Date:** 2026-07-18
**Status:** Approved
**Scope:** Fix visual collisions and tighten depth/emphasis consistency in rendered `.flow` scenes, across `SceneRenderer.tsx`, `chrome.ts`, and `tokens.ts`. No new visual language, node types, or authoring surface.

## Goal

Rendered scenes currently have concrete visual defects visible on real decks:
text/element collisions, an ad-hoc glow rule that fires on color rather than
intent, no enforced gap between the diagram canvas and the headline/body text
block beneath it, and near-identical styling between step chips and label
chips. Fix these four issues without changing the token palette, node
vocabulary, or `.flow` authoring surface.

## Evidence

Screenshots of live decks (rendered via `npm run dev` + Playwright,
2026-07-18) show:

- `segment-pools` scene 1: the header caption "serialize once" wraps to a
  second line and visually overlaps the vertical connector line running
  through the BUILD/FREEZE/DISPATCH diagram.
- `cellular-internals` scene 1: the chapter caption line sits flush against
  the diagram's bottom edge with no vertical breathing room.
- `rust-architecture` scene 1: the "aiperf binary" box has large internal
  dead space with two floating dots beside it that read as unconnected.
- Across scenes: some chips (`one identity`, `authoritative`) render with a
  double drop-shadow glow, others (`autonomous`, step chips) do not, with no
  visible authoring signal explaining the difference.

## Root causes

1. **Caption collision (corrected).** The `segment-pools` scene 1 overlap was
   not text-wrap measurement at all: a separate `sdk.Callout` node
   (`pool-callout`, text "serialize once") was authored in `segment-pools.flow`
   at coordinates that geometrically overlapped the `sdk.Header` box by
   ~20 vertical pixels. `sdk.Callout`'s stem is a straight line from box to an
   explicit target point with no path-avoidance of intervening nodes, so any
   author-supplied position that lands inside another node's box collides.
   No `chrome.ts` measurement logic was implicated.
2. **Glow refined for quality, not "fixed" as inconsistent.** The
   `remappedAccentFill` accent-color detection in `SceneRenderer.tsx`'s box
   render path (glow fires whenever `isAccentThemeRole` matches fill or
   stroke) turned out to be deliberate and consistent: every accent-stroked
   panel glows, every neutral/muted-stroked chip does not. The real defect
   was that the glow itself looked heavy — a stacked double `drop-shadow`
   (14px blur at 70% opacity plus 28px at 40%) read as neon rather than
   premium. The fix reduces it to a single softer glow (8px blur, 45%
   opacity) and deepens the baseline elevation shadow used by non-glowing
   nodes, rather than gating glow behind a new explicit flag.
3. **No enforced diagram-to-text gap.** Scene layout does not reserve a
   minimum vertical gap between the diagram canvas and the headline/body
   text block beneath it, so the two areas can end up touching or nearly
   touching depending on diagram height.
4. **Chip styling collapse.** Step chips (`1. CONTROL`) and label/category
   chips (`one identity`) share the same chip render path and size, with only
   fill color distinguishing them, so at a glance they read as the same kind
   of information.

## Fixes

### 1. Caption/text collision

Reposition the colliding node in `decks-flow/segment-pools.flow`: the
`pool-callout` `sdk.Callout` moved from `x=250,y=40` (overlapping the header)
to `x=300,y=66`, with its target retargeted from `{x:310,y:120}` to
`{x:360,y:95}` — the top of the "FREEZE" step chip, which is also where
serialization actually happens. This is a per-scene content fix, not a
`chrome.ts` change; `sdk.Callout`'s straight-line-stem behavior is unchanged
and authors remain responsible for choosing non-colliding coordinates.
Separately, `decks-flow/rust-architecture.flow` scene 1 dropped an empty
`sdk.Band` background (`zone`) that added dead space around a single small
panel with no grouping value.

### 2. Glow refined for depth, not gated behind a new flag

In `SceneRenderer.tsx`'s box render path, the `remappedAccentFill` branch's
filter changed from a heavy double `drop-shadow` (14px @ 70% + 28px @ 40%) to
a single softer glow (8px blur @ 45% opacity), and the non-glow baseline
shadow deepened from `0 3px 5px rgba(0,0,0,0.28)` to `0 6px 10px
rgba(0,0,0,0.3)` (glow branch baseline: `0 8px 14px rgba(0,0,0,0.4)`). No new
style flag was introduced; the existing accent-color-triggered glow rule is
kept as-is, just visually refined.

### 3. Diagram-to-text minimum gap

Add a `spacing.diagramToText` (or similarly named) value to `tokens.ts` and
apply it as a minimum gap in the scene layout path that positions the
headline/body text block relative to the diagram canvas, so the two areas
never render closer than that gap regardless of diagram height.

### 4. Chip differentiation

In `chrome.ts`'s chip descriptor, give step chips (chips authored as part of
a numbered step sequence) a visually distinct muted/numbered-pill treatment
(e.g. lower-contrast fill, no accent border) versus label/category chips,
which keep their existing colored-outline treatment. This is a style
adjustment in the existing chip render path, not a new node type.

## Non-goals

- No new node types, capabilities, or `.flow` authoring syntax.
- No change to the color token palette (`tokens.ts` `category`/`accent`
  values stay as-is).
- No change to motion/timeline/cue behavior.
- No redesign of node shapes, arrow styling, or composition beyond the four
  fixes above.

## Verification

- Re-rendered the three scenes captured as evidence (`rust-architecture`,
  `cellular-internals`, `segment-pools`, scene 1 of each) via the dev server
  and Playwright screenshot; confirmed no text/element overlap and a visible
  gap between diagram and text block.
- Spot-checked `slurm-velo` and `velo-deep-dive` (step-chip and accent-glow
  scenes not in the original evidence set); chip differentiation and the
  refined glow read correctly with no regressions.
- Ran `npm run flow-verifier`: `IR: verifying 9 package(s)…`, `Play: full-deck
  Playwright walk…`, `summary: 0 error(s), 0 warn(s)`.

## Source anchors

- `apps/explainers/src/core/diagram/SceneRenderer.tsx` — box/panel render
  paths, glow (`remappedAccentFill`), shadow filters.
- `apps/explainers/src/flow/compiler/desugar-scene-primitives.ts` —
  `core.stepper` step-chip chrome.
- `apps/explainers/src/flow/sdk/generic/chrome.ts` — `sdk.Header`,
  `sdk.Callout` descriptors (stem/target geometry).
- `apps/explainers/src/index.css` — `.ex-stage-hero.ex-content-card__diagram`
  bottom inset (diagram-to-text gap).
- `apps/explainers/decks-flow/segment-pools.flow`,
  `apps/explainers/decks-flow/rust-architecture.flow` — per-scene content
  fixes.

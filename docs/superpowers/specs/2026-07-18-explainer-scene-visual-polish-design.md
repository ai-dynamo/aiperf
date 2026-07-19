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

1. **Caption collision.** `sdk.note` / header-caption text nodes in
   `chrome.ts` are placed and sized without measuring wrapped text height
   against sibling node geometry. A caption that wraps to two lines is
   painted at its authored single-line box height, so the second line spills
   into whatever sits below it in local coordinate space.
2. **Glow tied to color, not intent.** In `SceneRenderer.tsx`, the box render
   path applies the full double-glow `drop-shadow` filter whenever
   `isAccentThemeRole(node.style?.fill) || isAccentThemeRole(node.style?.stroke)`
   is true (`remappedAccentFill`, around the `core.box` render branch).
   Category-accent colors (green/blue/orange fills used for semantic
   grouping, not emphasis) trigger the same glow as an intentionally
   highlighted node, so glow ends up wherever accent colors happen to land.
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

In `chrome.ts`, compute the wrapped bounding box of caption/note text at
compile time (measure against the authored max width) and size the caption's
layout box to that measured height rather than a fixed single-line height.
Downstream siblings that are positioned relative to the caption's box
therefore get pushed to account for wrapped lines instead of being painted
underneath them.

### 2. Glow gated by explicit intent

Replace the `remappedAccentFill` accent-color detection in
`SceneRenderer.tsx`'s box render path with an explicit style flag (e.g.
`style.emphasis === true` or `style.glow === true`) that scene authors (or
SDK component defaults) set deliberately. Accent-colored fills that are not
marked emphasized fall through to the existing baseline
`drop-shadow(0 3px 5px rgba(0,0,0,0.28))`. Update any SDK component
(`sdk.chip`, `sdk.card`, etc. in `chrome.ts`) that relied on the old
color-triggered glow to set the new flag explicitly where highlighting was
intended.

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

- Re-render the three scenes captured as evidence (`rust-architecture`,
  `cellular-internals`, `segment-pools`, scene 1 of each) via the dev server
  and Playwright screenshot; confirm no text/element overlap and a visible
  gap between diagram and text block.
- Spot-check 2-3 additional decks with step-chip sequences (`slurm-velo`,
  `velo-deep-dive`) to confirm chip differentiation reads correctly and no
  existing "emphasis" scenes lost their glow after the flag migration.
- Run `npm run flow-verifier` (or the project's existing flow/deck
  verification script) to confirm no `.flow` deck fails to compile after the
  `chrome.ts` changes.

## Source anchors

- `apps/explainers/src/core/diagram/SceneRenderer.tsx` — box/panel render
  paths, glow (`remappedAccentFill`), shadow filters.
- `apps/explainers/src/flow/sdk/generic/chrome.ts` — `sdk.header`,
  `sdk.note`, `sdk.chip`, `sdk.card` descriptors and caption/chip layout.
- `apps/explainers/src/core/tokens.ts` — spacing/radius/color tokens.

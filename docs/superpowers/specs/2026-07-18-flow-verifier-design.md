<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow Verifier Design

**Date:** 2026-07-18  
**Status:** Approved  
**Host:** `apps/explainers`  

## Goal

Gate flow-backed explainer decks so Scene IR and live playback stay visually coherent: no floating boxes, orphan arrows/dots, broken timelines, or arrowheads that lead their strokes. The verifier **plays the entire deck** (Playwright) and also runs a fast IR playhead pass.

## Layers

### A — IR playhead (fast)

Inputs: `apps/explainers/src/decks-generated/*.package.json` (or compile from `decks-flow/*.flow` when `--from-flow`).

For each slide with `render.scene`:

- Non-empty `roots` and `timeline`
- Every timeline `target` resolves to a node id in the scene tree
- Boxes (`core.rect` / text frames with area): finite geometry inside viewport (default 700×400, or authored camera/viewport)
- Arrows/paths: require non-empty `path`/`d`; parse endpoints; warn/error if both ends float far from any box edge/center (orphan connector)
- Dots (`core.dot` / small circles): must sit near some path stroke (within snap tolerance) unless tagged as a static legend chip
- Zero-area rects (non-arrow) are errors
- Simulate `t` across the timeline duration: draw cues must exist for path nodes that expect reveal; document SceneRenderer contract that arrowheads appear only at `drawProgress >= 1`

### B — Playwright full play (source of truth)

1. Serve `apps/explainers` (Vite preview or existing `--base-url`)
2. For each deck route (HashRouter: `/#/<deck-id>`): dismiss StartGate → **Play slideshow** → walk every slide
3. While playing / on each slide, assert against live SVG:
   - `[data-flow-node-id]` present for scene roots
   - No visible path with `data-flow-arrowhead="true"` while `stroke-dashoffset` indicates mid-draw
   - Motion dots (`[data-flow-dot]`, `[data-flow-motion-signal]`) have finite positions inside the SVG viewBox
   - Rects with positive layout are inside the stage viewBox (margin)

## CLI

```bash
node apps/explainers/scripts/flow-verifier.mjs
node apps/explainers/scripts/flow-verifier.mjs --deck segment-pools
node apps/explainers/scripts/flow-verifier.mjs --ir-only
node apps/explainers/scripts/flow-verifier.mjs --play-only --base-url http://127.0.0.1:5173
make flow-verifier
```

Exit `0` when all selected layers pass; non-zero with structured findings on stderr.

## Non-goals

- Pixel visual-regression baselines (optional later)
- LLM “semantic accuracy” judging of diagram meaning beyond structural/animation invariants

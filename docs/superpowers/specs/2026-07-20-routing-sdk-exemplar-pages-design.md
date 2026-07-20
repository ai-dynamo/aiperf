<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Routing SDK Exemplar Pages Design

**Date:** 2026-07-20
**Status:** Approved
**Scope:** Full-page routing cookbook additions to `flow-sdk-examples.flow`

## Goal

Expand the Flow SDK examples deck with complete, copyable routing scenes. The
new sequence must demonstrate every curve anchor pairing, advanced curved-route
behavior, and anchor-safe orthogonal routing without removing the existing
compact topology overview.

The deck remains an executable authoring reference: every exemplar is written
with public `sdk.*` calls, uses a non-empty timeline, labels the behavior being
demonstrated, and passes the existing compiler and geometry gates.

## Locked decisions

- Append dedicated routing exemplars after the existing “Topology patterns”
  slide.
- Keep every existing slide and update deck-wide numbering from 10 to 19.
- Add nine routing slides: the union of the focused, deep, and gallery options.
- Keep all implementation in `flow-sdk-examples.flow`; no new runtime behavior
  or SDK API is part of this work.
- Use actual public authoring syntax rather than raw Scene IR or `freeform`.
- Give every directed edge an explicit draw or trace cue.
- Keep each scene independently understandable and copyable.

## Slide sequence

### 1. Complete 9×9 curve matrix

Demonstrate all 81 source-anchor × target-anchor combinations across:

`center`, `n`, `s`, `e`, `w`, `ne`, `nw`, `se`, and `sw`.

The page uses a compact matrix with one source and one target glyph per cell.
Each cell contains a real `sdk.Edge(mode = "curve")`. Row and column labels
identify source and target anchors. The timeline reveals rows in groups rather
than animating 81 unrelated cues individually.

This page is exhaustive coverage, not the primary copy-paste example. A nearby
caption points authors to the following focused pages for readable patterns.

### 2. Cardinal curves

Four source panels connect through `n`, `s`, `e`, and `w` anchors to four target
panels. Labels make the outward source tangent and inward target approach
obvious. The scene is sparse enough that an author can copy one edge unchanged.

### 3. Corner and center curves

Dedicated examples cover `ne`, `nw`, `se`, `sw`, and `center`. The corner
examples visibly leave along diagonal tangents. The center example demonstrates
the peer-facing fallback rather than implying a fixed direction.

### 4. Same-side links and self-loops

Show `n → n`, `w → w`, and a node-to-itself edge. All paths stay outside their
component bounds. Labels distinguish same-side routing from self-loop routing
and show `preferredSide`.

### 5. Obstacle avoidance

Place source and target panels on opposite sides of two blocking panels. Show:

- the default deterministic route;
- a route with increased `clearance`;
- a route with `preferredSide`;
- a comparison edge with `avoidObstacles = false`.

The comparison uses separate lanes so the paths remain legible.

### 6. Parallel lanes

Render three curved edges with identical endpoint IDs and anchors. The routes
separate symmetrically using `parallelGap`, while all three preserve exact
endpoint attachment.

### 7. Bundling

Render three compatible edges with `bundle = true`. The scene labels the shared
corridor and separate endpoint branches. Motion traces follow the same resolved
geometry as their corresponding strokes.

### 8. Anchor-safe orthogonal routing

Demonstrate cardinal `sdk.Edge(mode = "route")` connections with tall and wide
endpoint displacements:

- west/east targets are approached horizontally;
- north/south targets are approached vertically;
- intermediate legs do not run directly along a connected component side.

A small before/after annotation explains the avoided edge-hugging failure
without rendering intentionally broken production geometry.

### 9. Routing controls reference

Six miniature live examples document:

- `clearance`;
- `curvature`;
- `avoidObstacles`;
- `preferredSide`;
- `bundle`;
- `parallelGap`.

Each mini-example pairs the authored style fragment with its visible effect.
The page also distinguishes `mode = "curve"` from `mode = "route"`.

## Visual system

The new pages reuse the existing deck’s header, panel, chip, note, label, and
theme-role conventions. Curved routes use accent colors consistently:

- primary for baseline/default routes;
- secondary for alternates;
- tertiary for lane and bundle variants;
- warning for disabled-avoidance comparisons;
- green for verified endpoint-safe results.

No page relies on color alone. Anchor names, route labels, and short captions
carry the same distinctions.

## Timeline behavior

Every new page has a non-empty `timeline main` block. Timelines follow the same
order:

1. reveal the page header and explanatory labels;
2. reveal source, target, and obstacle nodes;
3. trace routing edges in semantic groups;
4. reveal the concluding note or legend.

The 9×9 page traces matrix rows or row groups to keep cue count and playback
duration bounded. Every generated directed edge receives a draw, trace, or
reveal-stroke cue through its stable generated ID or an SDK parent action that
expands to those edges.

## Deck metadata updates

Update:

- all eyebrow fractions from `x of 10` to `x of 19`;
- subsequent slide ordinals after “Topology patterns”;
- hub title and description to mention the complete routing cookbook;
- relevant slide comments;
- the final checklist wording to point to the routing exemplars.

Existing slide IDs remain unchanged. New IDs use a dedicated ordinal prefix and
must not collide with existing generated child IDs.

## Verification

Run:

1. `npm run build`;
2. `npm run flow-verifier:ir`;
3. `npm run flow-verifier:extended`.

The new pages must produce:

- no compile or type errors;
- no zero-area, out-of-bounds, missing-target, or missing-path findings;
- no missing draw-cue warnings for their generated edges;
- finite SVG geometry and exact endpoint attachment;
- successful full-deck navigation through all 19 slides.

Failures already present in unrelated decks remain outside this change, but the
new routing pages must add no findings.

## Non-goals

- Changing router algorithms or public SDK behavior.
- Creating another deck or route.
- Replacing the existing topology overview.
- Adding raw SVG, package-form Scene IR, or a new freeform escape hatch.
- Treating the examples deck as a performance benchmark for the router.

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Diagram Node Auto-Sizing and Spacing

## Goal

Give shared SDK diagram primitives deterministic intrinsic sizing that fits
their labels and details, and reflow children inside layout containers with
stable gaps so nodes stop overflowing boxes or crowding siblings.

## Decisions

- Scope is shared across all decks that use SDK primitives, not a
  `cellular-internals`-only patch.
- Explicit authored `width` and `height` are minimums. Content may enlarge a
  node unless the node explicitly clips or hides overflow.
- Layout containers (`layout.stack`, `layout.grid`, `layout.rail`, `core.lane`,
  `core.swimlane`, `core.stepper`, `layout.pad`) keep authored or default gaps
  and reflow later children after a node grows.
- Absolute-positioned top-level siblings are not moved when a neighbor grows.
- Geometry stays deterministic. Use shared scale-aware char-width estimation,
  not browser `measureText`.
- Rendered SVG text already uses `SCENE_TEXT_SCALE = 0.9`. Layout estimators
  must use the same scale so box size matches painted text.

## Design

### Shared text metrics

Add a small pure module (for example
`apps/explainers/src/core/diagram/text-metrics.ts`) that exports:

- `SCENE_TEXT_SCALE` (single source of truth, also consumed by `SceneRenderer`)
- Band and padding constants used by chrome and layout (`INSET`, title/detail
  band heights, chip pad, stepper char width)
- `estimateTextWidth(text, fontSize, weight?)` that multiplies estimated glyph
  width by `SCENE_TEXT_SCALE`

`capabilities/layout.ts`, `capabilities/chrome.ts`, and `SceneRenderer` consume
these helpers. Duplicate literals such as `STEPPER_CHAR_WIDTH = 6.2` are removed
from those call sites.

### Intrinsic leaf sizing

For semantic leaves that carry visible copy, resolve width and height as:

```
width  = max(authoredWidth, contentWidth + horizontalPadding)
height = max(authoredHeight, contentHeight + verticalPadding)
```

Covered capabilities:

| Capability | Content that drives size |
|---|---|
| `core.chip` | Label text |
| `core.panel` / `core.card`-equivalent panel chrome | Title and optional detail |
| `core.note` | Caption text |
| `core.header` | Title and optional caption within header bands |
| `core.label` / `core.text` layout boxes | Authored or default text |
| `core.stepper` | Intrinsic chip widths from step labels (already present; make scale-aware) |

When width or height is omitted, use the same intrinsic formula against the
existing default geometry floors (for example chip 84×26).

Do not change connector routing, theme roles, or authored absolute coordinates
outside layout containers.

### Container reflow

Existing `resolve*Layout` functions already treat container bounds as
minimums and place children with fixed gaps. Keep that contract:

1. Resolve each child's intrinsic geometry first.
2. Place children in order with the authored/default gap.
3. Expand the container to the union of placed children when needed.
4. Leave nodes outside these containers at their authored positions.

### Dual-path cleanup (minimal)

Where panel/chip/note chrome still emit compile-time `core.text` children while
also having native semantic chrome, prefer the semantic chrome path for
intrinsic sizing so text metrics and box metrics stay paired. Do not expand
this into a full IR migration beyond what sizing requires.

### Verifier parity

Node verification (`verify-geometry` / flow-verifier geometry) must resolve the
same capability layout rules used at render time, so expanded rails, steppers,
and chips are visible to checks. Verifier continues to run without DOM
measurement.

## Non-goals

- Per-deck hand-tuned coordinates in `.flow` files
- Adaptive gap inflation or viewport-fitting compression
- Moving absolute-positioned siblings when a neighbor grows
- Canvas/DOM text measurement
- Changing shell stage CSS or subtitle/lede layout

## Verification

- Unit tests for `estimateTextWidth` and scale-aware stepper/chip widths
- Layout tests proving authored dimensions act as minimums and containers keep
  gaps while reflowing later children
- SceneRenderer tests that rendered `font-size` and intrinsic box width stay
  consistent under `SCENE_TEXT_SCALE`
- Explainer package tests and build
- Spot-check crowded slides (for example `cellular-internals` slide 1) for
  label fit inside chips/cards without absolute sibling drift

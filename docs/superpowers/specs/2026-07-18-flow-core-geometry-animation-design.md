<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow Core Geometry & Animation Primitives Design

**Date:** 2026-07-18  
**Status:** Approved  
**Scope:** Generic geometry + animation vocabulary for explainer `@scene` authoring  

## Goal

Authors describe complex diagram concepts with built-in generic utilities — not AIPerf-specific metaphors. Expand `core.*`, `layout.*`, and `motion.*` so decks stop hand-rolling panels, absolute SVG paths, motion-signal pairs, and staggered enter cue lists.

## Locked decisions

| Decision | Choice |
|---|---|
| Vocabulary breadth | Broader geometry + animation catalog (not only today’s deck boilerplate) |
| Dialects | Package-form capabilities **and** native cinematic keywords grow together |
| Migration | Full rewrite of all eight `decks-flow/*.flow` decks onto the new vocabulary |
| Runtime model | **Hybrid** — simple macros desugar; runtime-needed concepts stay first-class |
| Tests | **No new tests** for this work |
| Render path | Extend existing language → compiler → SceneIr → SceneRenderer only |

## Architecture

```text
.flow @scene (package or native)
  → language capture / parse
  → lowerExplainerScene (desugar macros + emit first-class IR)
  → DeckPackage SceneIr
  → SceneRenderer (+ FlowArrow / MotionSignal)
  → flow-verifier (existing gates; no new verifier suites required)
```

### Hybrid lowering

**Desugar** (compiler expands; IR stays familiar `rect` / `text` / `connector` / `group`):

- `core.circle` / `core.ellipse`
- `core.panel` / `core.header`
- `core.bracket` / `core.callout`
- `layout.pad`
- `core.arrow` when endpoints are explicit `d` / `points` / absolute coords
- `enter-children` timeline sugar → expanded or compact stagger

**First-class** (survive into package IR; SceneRenderer understands):

- Relative child layout inside groups/panels/stacks/grids
- `core.connector` / `core.elbow` with `{ nodeId, anchor }` endpoints
- `layout.stack` / `layout.grid`
- `motion.signal` / `motion.pulse`
- Compact `stagger` timeline records
- `fade` / `exit` cue actions
- Per-cue `easing`

Foundation nodes (`core.rect`, `core.text`, `core.path`, `core.line`, `core.dot`) remain valid.

## v1 geometry catalog

| Capability | Native keyword | Mode |
|---|---|---|
| `core.rect` / `core.text` / `core.path` / `core.line` / `core.dot` | existing | foundation |
| `core.circle` / `core.ellipse` | `circle` / `ellipse` | desugar |
| `core.panel` | `panel` | desugar → rect + relative title/detail text |
| `core.header` | `header` | desugar → strip rect + left/right text |
| `core.arrow` | `arrow` | desugar\* |
| `core.connector` | `connector` | first-class |
| `core.elbow` | `elbow` | first-class orthogonal route |
| `core.bracket` | `bracket` | desugar |
| `core.callout` | `callout` | desugar |
| `layout.stack` | `stack` | first-class |
| `layout.grid` | `grid` | first-class |
| `layout.pad` | `pad` | desugar |

\*Node-anchored `core.arrow` uses the first-class connector path generator, then arrowhead defaults.

**Anchors:** `center`, `n`/`s`/`e`/`w`, `ne`/`nw`/`se`/`sw` (aliases `top`/`bottom`/`left`/`right` accepted).

**Relative children:** local `layout` coords relative to parent content box; absolute coords still allowed.

**Out of v1:** flex wrap, auto edge avoidance, bezier helpers beyond elbow, icons, `viz.*` charts.

## v1 animation catalog

| Concept | Mode | Behavior |
|---|---|---|
| Existing `enter`/`reveal`, `draw`, `emphasis`/`emphasize`, `pulse` | keep | as today |
| `motion.signal` | first-class | path + traveling tip; replaces path+dot+heuristic |
| `motion.pulse` | first-class | pulse outline tied to real cues |
| `stagger` timeline group | first-class compact IR | `{ targets[], action, at, duration, step }` |
| `enter-children` | sugar | stagger enter on direct children of a group |
| `fade` / `exit` | first-class | opacity down |
| `easing` on cues | pass-through | `linear` \| `ease-in` \| `ease-out` \| `ease-in-out` |

Every diagram slide still requires a non-empty timeline. Reduced-motion jumps to end state; traveling signals omitted.

**Out of v1:** springs, general property keyframes, camera choreography helpers.

## Migration

Rewrite all eight decks under `apps/explainers/decks-flow/`:

1. `rust-architecture`
2. `rust-architecture-atlas`
3. `segment-pools`
4. `slurm-velo`
5. `velo-deep-dive`
6. `cellular-internals`
7. `cellular-algorithms`
8. `dynosim`

Prefer `core.panel` / `core.header` / `core.elbow`|`connector` / `motion.signal` / `stagger`|`enter-children` over rect+text boilerplate, hand `d:` paths, and manual cue lists. Rebuild `decks-generated/*.package.json` after migration.

## Non-goals

- New test suites or TDD cycles for this change
- AIPerf-domain `viz.*` components
- Splitting decks across files / MentalModel escape hatch
- Second render path

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow Structure Components v3 Design

**Date:** 2026-07-18
**Status:** Approved
**Scope:** Heavy structure catalog + fan-out adoption across all nine explainer decks

## Goal

Ship a structure-oriented geometry pack so authors stop hand-rolling phase rails, swimlanes, background zones, and absolute connector paths. Then adopt v2 annotations + v3 structure across every `apps/explainers/decks-flow/*.flow` deck (including `tstar-warmup`).

## Locked decisions

| Decision | Choice |
|---|---|
| Approach | Heavy structure (B) + full-deck fan-out (prior choice 3) |
| New capabilities | `layout.rail`, `core.lane`, `core.band`, `core.swimlane`, `core.stepper`, `core.route` |
| Lowering | Hybrid — desugar macros; first-class rail + route |
| Decks | All nine `.flow` files under `decks-flow/` |
| Tests | No new tests |
| Commits | Only if explicitly requested |

## Catalog

| Capability | Native keyword | Mode | Behavior |
|---|---|---|---|
| `layout.rail` | `rail` | first-class | Equal slots along row/column; children typically chips/panels; `direction` + `gap` |
| `core.lane` | `lane` | desugar | Titled horizontal strip + optional local children |
| `core.band` | `band` | desugar | Background zone rect; title optional |
| `core.swimlane` | `swimlane` | desugar → group | Stacked `core.lane` (or panel rows) with shared left label column |
| `core.stepper` | `stepper` | desugar | Numbered steps from `steps: [...]` or children; optional connectors between steps |
| `core.route` | `route` | first-class | Auto orthogonal path between `{ nodeId, anchor }` endpoints (elbow generator; no hand `d:`) |
| v2 reuse | existing | desugar | `chip` / `note` / `divider` |
| v1 reuse | existing | mixed | elbows, stack/grid, callout/bracket, `motion.pulse` |

## Architecture

```text
.flow @scene (rail / lane / band / swimlane / stepper / route)
  → language tokens + package capability strings
  → desugar-scene-primitives (lane, band, swimlane, stepper)
  → SceneIr + SceneRenderer (rail layout like stack; route = elbow path)
  → parallel deck migration (9 decks)
  → rebuild decks-generated → flow-verifier:ir
```

### Desugar / first-class notes

- **`layout.rail`:** Mirror `layout.stack` placement but force equal slot widths/heights from parent geometry ÷ child count (minus gaps).
- **`core.lane`:** Group + chrome rect + title text (panel-like, shorter height defaults).
- **`core.band`:** Single rect (or group with rect child) with muted fill; no required label text.
- **`core.swimlane`:** Outer group; each authored child becomes a lane row; optional `labels: [...]` for left column.
- **`core.stepper`:** Expand `steps` strings into `core.chip` children inside a `layout.rail`; optional `core.route`/`elbow` between consecutive chips when `linked: true`.
- **`core.route`:** Same IR as `core.elbow` (connector + anchors); capability retained so authors prefer `route` for “auto path” intent. SceneRenderer elbow path generator applies.

## Fan-out

Migrate all nine decks:

1. `segment-pools` (deepen)
2. `dynosim` (deepen)
3. `tstar-warmup` (deepen)
4. `velo-deep-dive`
5. `slurm-velo`
6. `rust-architecture`
7. `rust-architecture-atlas`
8. `cellular-internals`
9. `cellular-algorithms`

Per deck: introduce at least one v3 structure use where it fits; convert obvious hand `core.path` between axis-aligned boxes to `core.route`/`elbow`; add v2 chips/notes/dividers where bottom bars / phase tags / hairlines exist; keep IR verifier green.

## Non-goals

- Bezier freeform routing
- Collision avoidance beyond elbow midpoints / `via`
- AIPerf-domain `viz.*` components
- New test suites or Playwright play layer
- Pixel visual-regression baselines

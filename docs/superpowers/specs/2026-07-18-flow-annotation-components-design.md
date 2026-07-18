<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow Annotation Components Design

**Date:** 2026-07-18
**Status:** Approved
**Scope:** Small v2 annotation catalog + adopt underused v1 primitives on two decks

## Goal

Authors stop hand-rolling phase tags, bottom caption bars, and hairline splits. Ship three desugar macros (`core.chip`, `core.note`, `core.divider`) and migrate `segment-pools` + `dynosim` onto them plus unused v1 (`layout.stack`, `core.callout` / `core.bracket`, `motion.pulse`, more `core.elbow`).

## Locked decisions

| Decision | Choice |
|---|---|
| Approach | Annotation pack (A) + two-deck adoption |
| New capabilities | `core.chip`, `core.note`, `core.divider` |
| Lowering | Desugar to existing IR (`group`/`rect`/`text`/`connector`) |
| Decks | `segment-pools.flow`, `dynosim.flow` only |
| Tests | No new tests |
| Commits | Only if explicitly requested |

## Catalog

| Capability | Native keyword | Mode | Behavior |
|---|---|---|---|
| `core.chip` | `chip` | desugar | Small rounded rect + centered label (`text` / `title`) |
| `core.note` | `note` | desugar | Wide low strip + caption (`text` / `caption`) |
| `core.divider` | `divider` | desugar | Headless H/V line; `markerEnd: none` |
| `layout.stack` | `stack` | first-class (exists) | Phase / chip rows |
| `core.callout` / `core.bracket` | existing | desugar (exists) | One each per migrated deck |
| `motion.pulse` | existing | first-class (exists) | Replace full-scene pulse overlays |
| `core.elbow` | existing | first-class (exists) | Axis-aligned box-to-box paths |

## Architecture

```text
.flow @scene (chip / note / divider)
  → language tokens + package capability strings
  → desugar-scene-primitives
  → SceneIr (rect/text/group/connector)
  → SceneRenderer (no new paint path required)
  → rebuild decks-generated → flow-verifier:ir
```

## Non-goals

- Migrating the other six decks
- `layout.rail` / swimlanes / badges beyond chip
- New test suites or Playwright play layer
- AIPerf-domain `viz.*` components

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow Structure Components v3 Implementation Plan

> **For agentic workers:** Use subagent-driven-development for parallel deck migration after primitives land.

**Goal:** Ship heavy structure catalog (`layout.rail`, `core.lane`, `core.band`, `core.swimlane`, `core.stepper`, `core.route`), wire schema → language → desugar/renderer, fan-out across all nine decks, rebuild + IR-verify.

**Spec:** `docs/superpowers/specs/2026-07-18-flow-structure-components-v3-design.md`

**Constraints:** No new tests; no commits unless asked; hybrid lowering.

## File map

| Area | Files |
|---|---|
| Schema | `apps/aiperf-flow/packages/schema/src/ir.ts`, `capability.ts` |
| Language | `tokens.ts`, `ast.ts`, `parser.ts` |
| Compiler | `desugar-scene-primitives.ts` |
| Renderer | `apps/explainers/src/core/diagram/SceneRenderer.tsx` (rail layout; route ≡ elbow) |
| Decks | all `apps/explainers/decks-flow/*.flow` |
| Packages | `npm run build:explainer-packages` |
| Gate | `npm run flow-verifier:ir` |

## Tasks

### Task 1: Schema + language
- [ ] Add six capability ids to `FoundationCapabilityId` + Zod + `FOUNDATION_CAPABILITIES`
- [ ] Native keywords `rail` / `lane` / `band` / `swimlane` / `stepper` / `route`
- [ ] Extend `SCENE_PRIMITIVE_CAPABILITIES` + parser OR alts

### Task 2: Desugar
- [ ] `core.lane`, `core.band`, `core.swimlane`, `core.stepper` macros
- [ ] `core.route` → first-class connector (like elbow) via `capabilityKind` / `lowerFirstClassPackageNode`
- [ ] Register desugar flags

### Task 3: SceneRenderer
- [ ] `layout.rail` placement (equal slots)
- [ ] Treat `core.route` as elbow-capable connector
- [ ] Local layout membership for new group-like caps as needed

### Task 4: Fan-out decks (parallel OK)
- [ ] Deepen segment-pools / dynosim / tstar-warmup
- [ ] Migrate velo-deep-dive, slurm-velo, rust-architecture(+atlas), cellular-internals, cellular-algorithms

### Task 5: Verify
- [ ] Build schema/language/compiler
- [ ] `build:explainer-packages` (8 or 9 packages — include tstar-warmup if registered)
- [ ] `flow-verifier:ir` → 0 errors
- [ ] Note in task-6 report or new sdd note

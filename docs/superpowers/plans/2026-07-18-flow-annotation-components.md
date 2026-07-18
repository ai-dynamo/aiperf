<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow Annotation Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Ship `core.chip` / `core.note` / `core.divider`, wire schema → language → desugar, migrate segment-pools + dynosim onto v2 + unused v1, rebuild packages, IR-verify.

**Spec:** `docs/superpowers/specs/2026-07-18-flow-annotation-components-design.md`

**Constraints:** No new tests; no commits unless asked; hybrid desugar only.

## File map

| Area | Files |
|---|---|
| Schema | `apps/aiperf-flow/packages/schema/src/ir.ts`, `capability.ts` |
| Language | `tokens.ts`, grammar / parser / ast as needed |
| Compiler | `desugar-scene-primitives.ts` |
| Decks | `apps/explainers/decks-flow/segment-pools.flow`, `dynosim.flow` |
| Packages | rebuild via `npm run build:explainer-packages` |

## Tasks

### Task 1: Schema
- [ ] Add `core.chip` / `core.note` / `core.divider` to `FoundationCapabilityId` + Zod literals
- [ ] Register in `FOUNDATION_CAPABILITIES`

### Task 2: Language
- [ ] Native keywords `chip` / `note` / `divider`
- [ ] Wire grammar / AST like existing `panel` / `bracket`

### Task 3: Desugar
- [ ] `core.chip` → group/rect + local label text
- [ ] `core.note` → strip rect + caption text
- [ ] `core.divider` → connector path, `markerEnd: none`
- [ ] Register in `isDesugarCapability` / `capabilityKind`

### Task 4: Migrate decks
- [ ] segment-pools: chips, stack, callout/bracket, pulse, elbows
- [ ] dynosim: notes, divider, callout, pulse/elbow where obvious

### Task 5: Verify
- [ ] `flow:build` if needed
- [ ] `build:explainer-packages`
- [ ] `flow-verifier:ir` → 0 errors

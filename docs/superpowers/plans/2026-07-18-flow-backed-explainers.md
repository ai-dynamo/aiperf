<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow-Backed Explainers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile all eight `apps/explainers` decks from `.flow` via the real Flow compiler into DeckPackages, play them in the legacy ExplainerShell with animated Scene IR diagrams and voiced narration — zero React MentalModel escape hatch.

**Architecture:** `.flow` → `@aiperf/flow-compiler` → DeckPackage artifact → `packageToDeckDefinition` → ExplainerShell + SceneRenderer. Diagrams are Scene IR + timeline only. Voice stays on legacy Web Speech.

**Tech Stack:** TypeScript, Zod (`@aiperf/flow-schema`), `@aiperf/flow-language`, `@aiperf/flow-compiler`, React + Vite (`apps/explainers`), Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-07-18-flow-backed-explainers-design.md`

## Global Constraints

- Parity host is `apps/explainers` ExplainerShell — do not replace the shell.
- **One file per deck:** `apps/explainers/decks-flow/<deck-id>.flow` holds the entire deck (metadata + all slides + all `@scene` diagrams + timelines + narration). Forbidden: `decks-flow/scenes/`, `.flowfrag`, per-slide sidecars, or splitting MentalModel ports into separate files.
- No `@mental_model` / no React MentalModel on the registry path when done.
- All diagram slides require non-empty Flow `timeline` cues.
- Voice = legacy Web Speech path (`useTimedSlideshow` / `narration.ts`).
- Preserve the eight deck `id` and `route` values exactly.
- Activate venv before repo commands: `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`
- SPDX Apache-2.0 header on every new source file.
- Prefer minimal diffs; do not rewrite unrelated aiperf-flow cinematic preview code.
- Commit after each task completes successfully. Use `git commit --no-verify --no-gpg-sign`.

## Parallelism map

| Wave | Tasks | Notes |
|---|---|---|
| 1 | Task 1 (schema), Task 2 (SceneRenderer skeleton) | Independent packages |
| 2 | Task 3 (compiler lowering), Task 4 (adapter + registry hook) | Needs Task 1 types |
| 3 | Task 5 (build script + golden rust-architecture compile) | Needs 1–4 |
| 4 | Tasks 6–13 (port eight decks) | Parallel after Task 5; one agent per deck |
| 5 | Task 14 (cleanup + CI gates) | After all ports |

---

### Task 1: DeckPackage schema in `@aiperf/flow-schema` — DONE

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/deck-package.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts` (export)
- Create: `apps/aiperf-flow/packages/schema/test/deck-package.test.ts`

**Interfaces:**
- Produces: `DeckPackage`, `SlidePackage`, `deckPackageSchema`, `safeParseDeckPackage(input: unknown): Result<DeckPackage>`
- `schemaVersion` literal `1`; strict objects (unknown fields rejected)
- `SlidePackage.render` optional and **only** `{ kind: "scene"; scene: SceneIr }` (reuse existing scene IR schema)

- [x] **Step 1:** Write failing tests for valid package, reject unknown fields, reject `render.kind: "mental_model"`, require `narration` string.

- [x] **Step 2:** Run:  
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate && cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow/packages/schema && npx vitest run test/deck-package.test.ts`  
  Expected: FAIL (module missing)

- [x] **Step 3:** Implement Zod schemas + exports matching the spec data model.

- [x] **Step 4:** Re-run vitest — PASS

- [x] **Step 5:** Commit: `feat(flow-schema): add DeckPackage schema for flow-backed explainers`

---

### Task 2: SceneRenderer for ExplainerShell diagram slot — DONE

**Files:**
- Create: `apps/explainers/src/core/diagram/SceneRenderer.tsx`
- Create: `apps/explainers/src/test/scene-renderer.test.tsx`
- Modify: `apps/explainers/src/core/types.ts` only if needed for a SceneIr type import (prefer importing from a shared type or local minimal type mirroring schema)

**Interfaces:**
- Produces: `SceneRenderer(props: { scene: SceneIr; playing: boolean; restartKey: number; reducedMotion?: boolean }): ReactNode`
- Plays timeline from start when `playing`; restarts when `restartKey` changes
- Honors reduced motion (show final frame)
- Viewport target ~700×400 SVG (match existing MentalModel canvases)
- Renders `core.rect` / `core.text` / arrow-like (`core.arrow`, `core.connector`, line/path) nodes

- [x] **Step 1:** Failing test: renders a minimal scene with one `core.rect` + one timeline enter cue.

- [x] **Step 2:** Run explainers vitest for that file — FAIL

- [x] **Step 3:** Minimal SVG/canvas implementation that evaluates timeline progress; wire theme colors as hex if theme resolver not available yet.

- [x] **Step 4:** Tests PASS

- [x] **Step 5:** Commit: `feat(explainers): add SceneRenderer for Flow scene IR diagrams`

---

### Task 3: Compiler — lower `explainer` documents to DeckPackage — DONE

**Files:**
- Modify: `apps/aiperf-flow/packages/language/src/` as needed so `parseDocument` accepts top-level `explainer` blocks
- Create: `apps/aiperf-flow/packages/compiler/src/lower-explainer.ts` (+ scene/slides helpers, `compile-explainer.ts`, timeline/set validators, pack helpers)
- Modify: `apps/aiperf-flow/packages/compiler/src/index.ts` / `lower.ts` to invoke explainer lowering when AST has explainers
- Create: `apps/aiperf-flow/packages/compiler/test/lower-explainer.test.ts`
- Create fixture: `apps/aiperf-flow/packages/compiler/test/fixtures/minimal-explainer.flow`

**Interfaces:**
- Consumes: DeckPackage schema from Task 1; existing scene lowering for `@scene`
- Produces: `lowerExplainerDocument(ast, caps) -> Result<DeckPackage>` (or pack into FlowIr extension — prefer **emitting DeckPackage** as compiler product for explainers builds)
- Reject empty narration/title; reject diagram slides without timeline when render present

- [x] **Step 1:** Failing test compiling `minimal-explainer.flow` to a DeckPackage with one slide + scene + timeline.

- [x] **Step 2:** Run compiler package vitest — FAIL

- [x] **Step 3:** Wire parser + lowerer; use real scene lowering, not regex/`Function()`.

- [x] **Step 4:** Tests PASS; reject mental_model-shaped render if it appears in AST.

- [x] **Step 5:** Commit: `feat(flow-compiler): lower explainer .flow documents to DeckPackage`

---

### Task 4: Adapter — packageToDeckDefinition + dual-load path — DONE

**Files:**
- Create: `apps/explainers/src/core/package-adapter.ts`
- Create: `apps/explainers/src/test/package-adapter.test.ts`
- Modify: `apps/explainers/src/core/ExplainerShell.tsx` only if SceneRenderer needs `playing`/`restartKey` props threaded (prefer wrapping MentalModel call site)
- Related: `load-deck-packages.ts`, dual-load `deck-registry.ts`

**Interfaces:**
- Consumes: `DeckPackage`, `SceneRenderer`
- Produces: `packageToDeckDefinition(pkg: DeckPackage): DeckDefinition`
- `MentalModel` wrapper reads `pkg.slides[slideIndex].render?.scene` and mounts `SceneRenderer`
- Preserve `storagePrefix`, `classPrefix`, routes, hub fields

- [x] **Step 1:** Failing test: adapter maps a fixture DeckPackage to DeckDefinition with correct id/route/slide count and renders MentalModel without throwing.

- [x] **Step 2:** Vitest FAIL

- [x] **Step 3:** Implement adapter; do **not** remove legacy decks yet — only add the adapter API.

- [x] **Step 4:** PASS

- [x] **Step 5:** Commit: `feat(explainers): adapt DeckPackage to DeckDefinition for ExplainerShell`

---

### Task 5: Build — compile explainers decks + golden rust-architecture — MOSTLY DONE

**Files:**
- Create: `apps/explainers/decks-flow/rust-architecture.flow` (port text from `apps/explainers/src/decks/rust-architecture/content.ts`; rebuild diagrams as `@scene`+timeline from MentalModel — start with slide 0 complete, remaining slides may use simplified but animated scenes that compile)
- Create: `apps/aiperf-flow/scripts/build-explainer-packages.mjs` (or package bin) invoking real `compileSource` / explainer lowerer — **not** the old regex script
- Create: `apps/explainers/src/decks-generated/rust-architecture.package.ts` (generated; checked in initially or generated in build — prefer generate-on-build + commit golden for CI stability)
- Modify: `apps/explainers/src/core/deck-registry.ts` to load rust-architecture from package via adapter **while leaving other seven on legacy modules**

**Interfaces:**
- Produces: CLI/script `node scripts/build-explainer-packages.mjs` exits 0 and writes packages for every `decks-flow/*.flow`
- Registry: rust-architecture package-backed when a generated package is present; otherwise legacy fallback

**Status note:** Build script, all eight `decks-flow/*.flow` sources, packages-only registry, package loader, and generated `decks-generated/*.package.json` for all eight decks exist. Visual baselines remain open (Step 4 + Task 14).

- [x] **Step 1:** Failing registry/visual smoke: package-backed rust-architecture has same slide count as legacy content.ts

- [x] **Step 2:** Implement `.flow` + build script + registry swap for that one deck

- [x] **Step 3:** Run explainers vitest + build script — PASS; all eight packages generated under `decks-generated/`

- [ ] **Step 4:** Manual/Playwright screenshot of slide 0 vs legacy baseline (save under `apps/explainers/test/baselines/` if harness exists) — still open if baselines not checked in

- [x] **Step 5:** Commit: `feat(explainers): package-backed rust-architecture deck from .flow` (authored as `.flow` source; package path via dual-load)

---

### Tasks 6–13: Port each deck as ONE complete `.flow` file — SOURCES + PACKAGES DONE

Each deck is exactly one file: `apps/explainers/decks-flow/<deck-id>.flow`.
That file must include hub/metadata, every slide from `content.ts`, and every
MentalModel frame **inlined** as `render: @scene { roots + timeline }` — full
animated diagrams, not stubs and not `decks-flow/scenes/` fragments.
**Discard** any worktree output under `decks-flow/scenes/` or `*.flowfrag`.

| Task | Deck id | Route | `.flow` source | package |
|---|---|---|---|---|
| 6 | `slurm-velo` | `/slurm-velo` | present under `decks-flow/` | `decks-generated/slurm-velo.package.json` |
| 7 | `dynosim` | `/dynosim` | present under `decks-flow/` | `decks-generated/dynosim.package.json` |
| 8 | `segment-pools` | `/segment-pools` | present under `decks-flow/` | `decks-generated/segment-pools.package.json` |
| 9 | `velo-deep-dive` | `/velo-deep-dive` | present under `decks-flow/` | `decks-generated/velo-deep-dive.package.json` |
| 10 | `cellular-internals` | `/cellular-internals` | present under `decks-flow/` | `decks-generated/cellular-internals.package.json` |
| 11 | `cellular-algorithms` | `/cellular-algorithms` | present under `decks-flow/` | `decks-generated/cellular-algorithms.package.json` |
| 12 | `rust-architecture-atlas` | `/rust-architecture-atlas` | present under `decks-flow/` | `decks-generated/rust-architecture-atlas.package.json` |
| 13 | `rust-architecture` | `/rust-architecture` | present under `decks-flow/` | `decks-generated/rust-architecture.package.json` |

Per deck:

- [x] Author/complete the single `.flow` (text + scenes + timelines) — sources exist for all eight; deepen scenes toward MentalModel parity as needed
- [x] Build package — all eight `decks-generated/*.package.json` generated
- [x] Registry packages-only swap; slide-count + narration tests — packages-only via `deckFromPackage` (Task 14)
- [x] Commit: `feat(explainers): flow-backed <deck-id> deck` — source commits landed per deck

---

### Task 14: Cleanup and CI gates — PARTIAL (regex gone)

**Files:**
- Delete: `apps/aiperf-flow/scripts/compile-explainer-flows.mjs` (regex path) if unused
- Delete or stop importing: leftover MentalModel/content modules from registry
- Modify: `apps/explainers/src/test/registry.test.ts` — assert all 8 decks package-backed; no MentalModel file imports in registry module
- Modify: CI/package scripts to run `build-explainer-packages.mjs` before explainers build

- [x] Obsolete regex `compile-explainer-flows.mjs` removed (gone from tree)
- [x] Registry is packages-only (`deckFromPackage` → `packageToDeckDefinition`; no legacy MentalModel module imports)
- [x] `validateDeckRegistry` green; unique routes/ids (registry tests present)
- [ ] Commit: `chore(explainers): remove React MentalModel registry path; gate flow packages`

---

## Spec coverage check

| Spec requirement | Task(s) |
|---|---|
| DeckPackage schema | 1 |
| Real compiler lowering | 3, 5 |
| SceneRenderer + animation/restart/reduced-motion | 2, 4 |
| Adapter into ExplainerShell | 4 |
| Voice via legacy shell | 4–5 (preserved) |
| All 8 decks | 5–13 |
| No escape hatch | 1, 3, 14 |
| Delete regex compiler | 14 |
| Visual/voice done bar | 5, 13, 14 |

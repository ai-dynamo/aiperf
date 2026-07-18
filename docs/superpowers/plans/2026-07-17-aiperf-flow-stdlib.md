# AIPerf Flow P0 Stdlib Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the P0 standard-library tree at `apps/aiperf-flow/stdlib/{core,viz}/*.flow` with barrel index exports, authored symbol stubs, and hand-validated expected IR fixtures — proving the composable `.flow` stdlib model before the symbol grammar fully parses every construct.

**Architecture:** Stdlib components are typed, namespaced `.flow` symbol definitions that lower to `kind: "component"` Flow IR nodes with explicit `capabilityId`, optional `semanticModel`, optional `layoutPlan`, props, slots, and timeline anchors. Flow-only components compose foundation nodes and other stdlib symbols; hybrid components declare one leaf binding each. Until the language parser accepts symbol grammar, stubs are committed as the authoring contract and validated through golden IR fixtures consumed by schema and compiler tests. Index files re-export public PascalCase symbols for import by domain wrappers in `examples/p0/`.

**Tech Stack:** `.flow` source, Flow IR v1 (Zod), Vitest, `@aiperf/flow-schema`, `@aiperf/flow-compiler` (when symbol lowering lands).

**Out of scope:** Preview UI, bespoke domain wrappers (`TokenSpanMorph`, `PromptSegmentComposer`, `RequestLifecycleWaterfall`), runtime capability renderers, and leaf implementations — those belong to sibling plans.

## Global Constraints

- Authors commit only `.flow` under `stdlib/`; no stdlib React, TypeScript scene code, or CSS.
- Every P0 stdlib component maps 1:1 to a registered P0 capability id in `@aiperf/flow-schema`.
- Every public symbol preserves stable semantic IDs and declares timeline
  anchors, semantic-twin meaning, simplified fallback meaning, and
  pause-to-explore behavior. Symbols never encode Canvas, DOM, SVG, or WebGPU
  implementation details.
- Unknown props, slots, and fields fail closed at compile and IR validation boundaries.
- Semantic entity and relation ids in stubs and fixtures survive layout overrides.
- Hybrid stdlib components bind at most one leaf id; leaves are never exported from public index files.
- Index exports surface PascalCase symbol names only; internal leaf ids stay packer-internal.
- Once Phase 2 fixtures land, every expected IR fixture must parse through
  `parseFlowIr` / `safeParseFlowIr` before the phase is marked complete.
- Authored `.flow` stubs may not parse until symbol grammar lands; track them with `@pending-parse` doc comments and a compile-skip manifest until Phase 3.
- Activate `.venv` before repo commands: `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Verify with `cd apps/aiperf-flow && npm run flow:check` once Phase 3 parity tests are wired.
- Do not create commits unless the user explicitly requests them.

## P0 stdlib inventory

Eleven components across layers 1–2 plus `viz.queue` and `viz.waterfall` (per [`2026-07-17-aiperf-flow-core-components-design.md`](../specs/2026-07-17-aiperf-flow-core-components-design.md)).

| File | Component id | Symbol export | Class | Leaf |
|---|---|---|---|---|
| `core/semantic-entity.flow` | `core.semantic-entity` | `SemanticEntity` | flow-only | — |
| `core/semantic-relation.flow` | `core.semantic-relation` | `SemanticRelation` | flow-only | — |
| `core/semantic-morph.flow` | `core.semantic-morph` | `SemanticMorph` | hybrid | `leaf.correspondence-tween` (stub binding; leaf deferred) |
| `core/glyph-run.flow` | `core.glyph-run` | `GlyphRun` | hybrid | `leaf.glyph-measure` |
| `core/span-map.flow` | `core.span-map` | `SpanMap` | hybrid | `leaf.span-interval` |
| `core/segment-strip.flow` | `core.segment-strip` | `SegmentStrip` | hybrid | `core.segment-strip.layout` |
| `core/focus-context.flow` | `core.focus-context` | `FocusContext` | flow-only | — |
| `core/compare.flow` | `core.compare` | `Compare` | flow-only* | `core.compare.sync` deferred to P1 |
| `core/structured-payload.flow` | `core.structured-payload` | `StructuredPayload` | flow-only* | virtual-tree leaf deferred to P1 |
| `viz/queue.flow` | `viz.queue` | `Queue` | hybrid | `viz.queue.policy` |
| `viz/waterfall.flow` | `viz.waterfall` | `Waterfall` | hybrid | `viz.waterfall.nest-layout` |

\*P0 ships as flow-only composition stubs; leaf promotion does not block stdlib authorship or IR fixtures.

## File structure

```text
apps/aiperf-flow/stdlib/
├── index.flow                         # barrel: re-exports core + viz public symbols
├── pending-parse.json                 # stub paths skipped by check until Phase 3
├── core/
│   ├── index.flow                     # export SemanticEntity, SemanticRelation, …
│   ├── semantic-entity.flow
│   ├── semantic-relation.flow
│   ├── semantic-morph.flow
│   ├── glyph-run.flow
│   ├── span-map.flow
│   ├── segment-strip.flow
│   ├── focus-context.flow
│   ├── compare.flow
│   └── structured-payload.flow
├── viz/
│   ├── index.flow                     # export Queue, Waterfall
│   ├── queue.flow
│   └── waterfall.flow
└── fixtures/
    ├── core/
    │   ├── semantic-entity.expected.json
    │   ├── semantic-relation.expected.json
    │   ├── semantic-morph.expected.json
    │   ├── glyph-run.expected.json
    │   ├── span-map.expected.json
    │   ├── segment-strip.expected.json
    │   ├── focus-context.expected.json
    │   ├── compare.expected.json
    │   └── structured-payload.expected.json
    └── viz/
        ├── queue.expected.json
        └── waterfall.expected.json
```

## Phased delivery

Symbol grammar may not parse stdlib constructs yet. Delivery proceeds in four phases; later phases must not rewrite semantic ids or capability bindings established in earlier phases.

| Phase | What lands | Validates how |
|---|---|---|
| **0 — Scaffold** | Directory tree, index barrels, `pending-parse.json` | Tree exists; index files list expected exports |
| **1 — Authored stubs** | Full `.flow` symbol definitions with props, slots, events, timeline anchors | Human review + formatter (when symbol-aware) |
| **2 — Expected IR** | `fixtures/**/*.expected.json` per component | `parseFlowIr` strict + capability id presence |
| **3 — Compile parity** | Symbol parser + compiler lower stdlib → IR | Compiled IR matches fixtures; stubs removed from pending manifest |

Phase 2 may start immediately and run in parallel with runtime leaf work. Phase 3 depends on Task 10 (symbol grammar) from the P0 core-components plan.

## Research synthesis (2026-07-17)

- P0 capability descriptors and five deterministic leaves are defined in `@aiperf/flow-schema` and `@aiperf/flow-runtime`.
- Foundation compiler links and lowers scene grammar; component nodes and symbol definitions are not yet fully parsed.
- Golden IR is the interim proof path. Three flat flagship authoring stubs
  (`TokenSpanMorph.flow`, `PromptSegmentComposer.flow`, and
  `RequestLifecycleWaterfall.flow`) exist under `stdlib/` and are parser-tested.
  The namespaced `stdlib/{core,viz}` tree, barrels, and per-component fixtures
  defined by Phases 0–2 do not exist yet; those phases create them without
  discarding the flagship stubs.
- Domain wrappers in `examples/p0/` import from `@aiperf/flow-stdlib` paths once index exports and compile parity exist; they are not part of this plan.

---

## Task 1: Stdlib scaffold and index export contract

**Files:**
- Create: `apps/aiperf-flow/stdlib/index.flow`
- Create: `apps/aiperf-flow/stdlib/core/index.flow`
- Create: `apps/aiperf-flow/stdlib/viz/index.flow`
- Create: `apps/aiperf-flow/stdlib/pending-parse.json`
- Modify: `apps/aiperf-flow/package.json` (add `flow:stdlib:fixtures` script if needed)

**Interfaces:**
- Produces: barrel export lists for 11 public symbols; pending-parse manifest schema.

- [ ] **Step 1:** Create `stdlib/`, `stdlib/core/`, `stdlib/viz/`, `stdlib/fixtures/{core,viz}/` directories.
- [ ] **Step 2:** Author `core/index.flow` exporting all nine core PascalCase symbols.
- [ ] **Step 3:** Author `viz/index.flow` exporting `Queue`, `Waterfall`.
- [ ] **Step 4:** Author root `index.flow` re-exporting `@stdlib/core` and `@stdlib/viz` (or relative import paths per linker convention).
- [ ] **Step 5:** Initialize `pending-parse.json` as an empty list with documented schema `{ "paths": string[], "reason": string }`.

---

## Task 2: Layer 1 flow-only stubs — identity primitives

**Files:**
- Create: `apps/aiperf-flow/stdlib/core/semantic-entity.flow`
- Create: `apps/aiperf-flow/stdlib/core/semantic-relation.flow`
- Create: `apps/aiperf-flow/stdlib/fixtures/core/semantic-entity.expected.json`
- Create: `apps/aiperf-flow/stdlib/fixtures/core/semantic-relation.expected.json`

**Interfaces:**
- `SemanticEntity`: props `id`, `label`, optional `description`, `state`; slot `chrome`; composes `group`, `text`, `inspect`.
- `SemanticRelation`: props `id`, `from`, `to`, `kind`; composes `semantic-entity`, `connector`.
- Expected IR: minimal `component` nodes with `capabilityId` matching component id, `semanticModel` entities/relations, foundation children.

- [ ] **Step 1:** Author stub symbols with typed props, default slot chrome, `on-inspect` action, and `@pending-parse` header.
- [ ] **Step 2:** Hand-author expected IR fixtures with stable semantic ids, one entity and one relation exemplar each.
- [ ] **Step 3:** Add fixtures to `pending-parse.json` for both stub paths.
- [ ] **Step 4:** Schema test loads fixtures and asserts `capabilityId`, `semanticModel`, accessibility labels.

---

## Task 3: Layer 1 hybrid stubs — glyph-run and span-map

**Files:**
- Create: `apps/aiperf-flow/stdlib/core/glyph-run.flow`
- Create: `apps/aiperf-flow/stdlib/core/span-map.flow`
- Create: `apps/aiperf-flow/stdlib/fixtures/core/glyph-run.expected.json`
- Create: `apps/aiperf-flow/stdlib/fixtures/core/span-map.expected.json`

**Interfaces:**
- `GlyphRun`: props `text`, `font`, optional `direction`; leaf binding `leaf.glyph-measure`; emits grapheme span ids.
- `SpanMap`: props `source`, `target`, `edges`, `requireCover`; slots `target-view`, `edge-chrome`; leaf binding `leaf.span-interval`.
- Expected IR: `props` include run ref and edge table; `semanticModel` preserves span ids; `layoutPlan` absent or empty for glyph-run.

- [ ] **Step 1:** Author stubs mirroring the `TokenSpanMorph` composition pattern from the core-components design (without domain naming).
- [ ] **Step 2:** Golden IR for `"café 🚀"` grapheme fixture aligned with `leaf.glyph-measure` test data.
- [ ] **Step 3:** Golden IR for span-map overlap/coverage fixture aligned with `leaf.span-interval` test data.
- [ ] **Step 4:** Register stub paths in `pending-parse.json`.

---

## Task 4: Layer 1 hybrid stub — semantic-morph

**Files:**
- Create: `apps/aiperf-flow/stdlib/core/semantic-morph.flow`
- Create: `apps/aiperf-flow/stdlib/fixtures/core/semantic-morph.expected.json`

**Interfaces:**
- `SemanticMorph`: props `beats`, `correspondences`; timeline anchors per beat; optional leaf binding `leaf.correspondence-tween` (comment-only until leaf lands).
- Expected IR: correspondence table in `semanticModel`; timeline cue attachments; reduced-motion policy noted in props.

- [ ] **Step 1:** Author stub with beat-indexed correspondence props and `target-view` / `source-view` slots.
- [ ] **Step 2:** Hand-author expected IR with at least two beats and entity id continuity across morph.
- [ ] **Step 3:** Schema test asserts correspondence ids survive unchanged when `layoutPlan` bounds differ from defaults.

---

## Task 5: Layer 2 hybrid stub — segment-strip

**Files:**
- Create: `apps/aiperf-flow/stdlib/core/segment-strip.flow`
- Create: `apps/aiperf-flow/stdlib/fixtures/core/segment-strip.expected.json`

**Interfaces:**
- `SegmentStrip`: props `segments`, `orientation`, optional `continuation`; slot `segment-chrome`; leaf binding `core.segment-strip.layout`.
- Expected IR: seven-segment layout plan from PromptSegmentComposer fixture (seed 42), segment semantic ids stable.

- [ ] **Step 1:** Author stub composing `semantic-entity`, optional `glyph-run` / `span-map` references in segment props.
- [ ] **Step 2:** Golden IR fixture matching `segment-strip-layout` leaf test output embedded in `layoutPlan`.
- [ ] **Step 3:** Assert layout override test: edited bounds in fixture do not rewrite segment entity ids.

---

## Task 6: Layer 2 flow-only stubs — focus, compare, structured-payload

**Files:**
- Create: `apps/aiperf-flow/stdlib/core/focus-context.flow`
- Create: `apps/aiperf-flow/stdlib/core/compare.flow`
- Create: `apps/aiperf-flow/stdlib/core/structured-payload.flow`
- Create: matching `fixtures/core/*.expected.json` (three files)

**Interfaces:**
- `FocusContext`: props `focusId`, optional `outline`; composes `camera`, entity state chrome.
- `Compare`: props `panes`, `syncMode`; composes `semantic-morph`, pane groups (flow-only P0; sync leaf deferred).
- `StructuredPayload`: props `root`, optional `window`; composes entities, optional `segment-strip` (virtual-tree leaf deferred).

- [ ] **Step 1:** Author three stubs with typed props, slots, and exploration/pause-safe beat annotations.
- [ ] **Step 2:** Hand-author expected IR for each; compare fixture includes two panes with shared morph correspondence ids.
- [ ] **Step 3:** Add all three stubs to `pending-parse.json`.

---

## Task 7: Viz hybrid stubs — queue and waterfall

**Files:**
- Create: `apps/aiperf-flow/stdlib/viz/queue.flow`
- Create: `apps/aiperf-flow/stdlib/viz/waterfall.flow`
- Create: `apps/aiperf-flow/stdlib/fixtures/viz/queue.expected.json`
- Create: `apps/aiperf-flow/stdlib/fixtures/viz/waterfall.expected.json`

**Interfaces:**
- `Queue`: props `policy`, `capacity`, `entries`; leaf binding `viz.queue.policy`; timeline anchors for enqueue/dequeue beats.
- `Waterfall`: props `lanes`, `intervals`, optional `openSpans`; leaf binding `viz.waterfall.nest-layout`; composes entities, relations, optional event-lane refs.

- [ ] **Step 1:** Author stubs with FIFO/priority policy enum props and lane-ordered interval tables.
- [ ] **Step 2:** Golden IR for queue policy simulation output embedded in `layoutPlan` / analysis props.
- [ ] **Step 3:** Golden IR for nested waterfall intervals aligned with `waterfall-nest-layout` leaf fixture (RequestLifecycleWaterfall lane vocabulary as neutral `lane-*` ids).
- [ ] **Step 4:** Register viz stub paths in `pending-parse.json`.

---

## Task 8: Fixture harness and capability cross-check

**Files:**
- Create: `apps/aiperf-flow/packages/schema/test/stdlib-fixtures.test.ts`
- Create: `apps/aiperf-flow/packages/compiler/test/stdlib-fixtures.test.ts`
- Modify: `apps/aiperf-flow/package.json` (wire `flow:stdlib:fixtures` into `flow:check` when ready)

**Interfaces:**
- Consumes: all 11 `fixtures/**/*.expected.json`, `P0_CAPABILITIES`, `parseFlowIr`.
- Produces: deterministic fixture manifest (sorted paths, content hashes).

- [ ] **Step 1:** Failing test enumerates fixture paths and asserts count === 11.
- [ ] **Step 2:** Each fixture parses strictly; every `component` node `capabilityId` is in `P0_CAPABILITY_IDS`.
- [ ] **Step 3:** Manifest hash test: fixture set hash stable across runs.
- [ ] **Step 4:** Compiler test loads fixture IR through pack/unpack round-trip without semantic drift.

---

## Task 9: Compiler stdlib resolution (Phase 3 gate)

**Files:**
- Modify: `apps/aiperf-flow/packages/compiler/src/link.ts` (stdlib import roots)
- Modify: `apps/aiperf-flow/packages/compiler/src/index.ts`
- Create: `apps/aiperf-flow/packages/compiler/test/compile-stdlib.test.ts`
- Modify: `apps/aiperf-flow/stdlib/pending-parse.json` (drain entries as stubs compile)

**Interfaces:**
- Consumes: authored `.flow` stubs, `pending-parse.json`.
- Produces: compiled IR matching co-located or fixtures-path expected JSON.

- [ ] **Step 1:** Configure linker to resolve `@aiperf/flow-stdlib/*` and relative `stdlib/` imports.
- [ ] **Step 2:** `aiperf-flow check` skips paths listed in `pending-parse.json`; all others must parse.
- [ ] **Step 3:** Per-component compile test: when symbol grammar supports a stub, remove from pending manifest and assert IR equals fixture (normalized ordering).
- [ ] **Step 4:** Index export test: importing `@aiperf/flow-stdlib` surfaces all 11 symbols in linked document export table.

---

## Task 10: Stdlib compose smoke (IR-only, no preview)

**Files:**
- Create: `apps/aiperf-flow/packages/compiler/test/compose-stdlib-ir.test.ts`

**Interfaces:**
- Consumes: fixture IR fragments composed into a single scene (not full domain wrappers).
- Produces: merged scene with reading order, timeline, and capability requirement list.

- [ ] **Step 1:** Compose `GlyphRun` + `SpanMap` fixture nodes into one scene; assert shared span ids.
- [ ] **Step 2:** Compose `SegmentStrip` + `FocusContext` fixture nodes; assert focus id references strip segment entity.
- [ ] **Step 3:** Compose `Queue` + `Waterfall` fixture nodes; assert disjoint semantic namespaces and merged capability list sorted deterministically.
- [ ] **Step 4:** No preview, renderer, or browser imports in this test file.

---

## Task 11: Verification gate

- [ ] All 11 `.flow` stubs committed with `@pending-parse` headers where grammar is incomplete.
- [ ] All 11 expected IR fixtures parse strictly and cross-check against `P0_CAPABILITIES`.
- [ ] `pending-parse.json` lists exactly the stubs not yet compilable; list shrinks monotonically in Phase 3.
- [ ] Index exports: `stdlib/index.flow` re-exports 11 public symbols; no leaf ids exported.
- [ ] `npm test -w @aiperf/flow-schema` and `npm test -w @aiperf/flow-compiler` green for fixture harness.
- [ ] When Phase 3 complete: `npm run flow:check` green with empty `pending-parse.json`.
- [ ] Progress ledger updated at `.superpowers/sdd/progress.md`.

---

## Dependency order

```text
Task 1 → Tasks 2–7 (stubs + fixtures, parallel by layer) → Task 8
Task 9 depends on P0 core-components Task 10 (symbol grammar)
Task 10 depends on Task 8
Task 11 last
```

Tasks 2–7 may run in parallel once Task 1 completes. Task 8 validates fixtures immediately without waiting for parser support.

## Execution options

1. **Subagent-driven (recommended)** — one subagent per task (or per layer batch), review between tasks.
2. **Fixture-first inline** — Task 1 + Task 8 + Tasks 2–7 fixtures before stub prose polish; Phase 3 deferred.

## Relationship to sibling plans

| Sibling plan | Boundary |
|---|---|
| [`2026-07-17-aiperf-flow-p0-core-components.md`](2026-07-17-aiperf-flow-p0-core-components.md) | Owns schema, leaves, runtime capabilities, symbol grammar, domain wrappers in `examples/p0/` |
| [`2026-07-17-aiperf-flow-browser-preview.md`](2026-07-17-aiperf-flow-browser-preview.md) | Owns preview shell only; no stdlib imports |
| This plan | Owns `stdlib/{core,viz}/*.flow`, index exports, fixtures, compile parity for stdlib paths |

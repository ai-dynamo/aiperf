# AIPerf Flow Flagship IR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maintain three hand-authored, schema-valid Flow IR v1 JSON documents
for the P0 flagship wrappers (`TokenSpanMorph`, `PromptSegmentComposer`,
`RequestLifecycleWaterfall`) under `apps/aiperf-flow/examples/p0/`, then deepen
their conformance tests against the deterministic leaf fixtures.

**Architecture:** Golden IR precedes `.flow` authoring and compiler lowering. Each example is a standalone `FlowIr` document whose root scene contains one flagship `component` node with explicit `capabilityId`, typed `props`, embedded `semanticModel`, and (where applicable) a frozen `layoutPlan` matching leaf golden output. Shared fixture fragments live beside the IR files so tests and future compiler parity checks import one source of truth. No `preview/` changes — these fixtures are consumed by schema and runtime test suites only until stdlib `.flow` symbols land.

**Tech Stack:** TypeScript, Zod 4 (`parseFlowIr`, `parseSemanticModel`, `parseLayoutPlan`), Vitest, `@aiperf/flow-schema`, existing P0 leaf functions.

## Global Constraints

- Hand-author JSON only; do not depend on compiler output for these three proofs.
- Do not modify anything under `apps/aiperf-flow/preview/`.
- Unknown props/fields fail closed — every IR object must pass `parseFlowIr` with no extra keys.
- Fixture data must match the landed leaf tests exactly (café 🚀 graphemes; seven-segment seed 42 strip; four lifecycle lanes).
- Semantic entity/relation ids in `semanticModel` are stable; layout plans override geometry only.
- Each scene declares `accessibility.readingOrder`, per-node `accessibility.label`, and a non-empty `fallback`.
- Every node carries a deterministic `sourceMap` pointing at its JSON file path.
- Capability requirements in `capabilities[]` must cover every `capabilityId` referenced in the document.
- These fixtures prove semantic and layout contracts, not final visual fidelity.
  North-star conformance additionally requires evaluated-scene/display-list
  output, Canvas rendering, semantic-twin projection, SVG/HTML fallback,
  deterministic narration/camera beats, and pause-to-explore/exact-beat resume.
- Every fixture must preserve enough semantic identity and reading order for the
  runtime to produce the same entities, relations, focus targets, and evidence
  across all render backends.
- Activate `.venv` before repo commands: `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Verify with `cd apps/aiperf-flow && npm run flow:check`.
- Do not create git commits unless the user explicitly requests them.

## Research synthesis (2026-07-17)

Leaf tests already freeze the canonical fixture values this plan must embed:

| Wrapper | Leaf source | Fixture anchor |
|---|---|---|
| `TokenSpanMorph` | `glyph-measure.test.ts`, `span-interval.test.ts` | Text `"café 🚀"`, graphemes `g0`–`g5`, edges `e0`–`e3`, tokens `t0`–`t4` |
| `PromptSegmentComposer` | `segment-strip-layout.test.ts` | Seven segments, layout seed `42`, total strip width `278` |
| `RequestLifecycleWaterfall` | `waterfall-nest-layout.test.ts` | Lanes `arrival → admission → connect → first-token`, four point/interval events |

P0 capability ids are registered in `packages/schema/src/capability.ts`. Wrapper IR binds to stdlib capabilities (`core.glyph-run`, `core.span-map`, `core.segment-strip`, `viz.waterfall`), not bespoke runtime renderers.

The three schema-valid IR documents and their schema tests are landed. They are
P0 semantic fixtures, not proof that the mounted app already uses Canvas or the
semantic twin. The live-cinematic and hybrid-renderer plans extend these
fixtures with deterministic timeline, narration, interaction, and cross-backend
conformance.

---

## File structure

```text
apps/aiperf-flow/examples/p0/
├── README.md
├── token-span-morph.ir.json
├── prompt-segment-composer.ir.json
└── request-lifecycle-waterfall.ir.json
```

Test files (outside examples):

```text
apps/aiperf-flow/packages/schema/test/flagship-ir-fixtures.test.ts
```

This is the landed file structure. The current test proves strict
`parseFlowIr` acceptance and `irVersion === 1`. Tasks below deepen that test in
place with leaf-parity, semantic identity, and later cross-backend assertions;
they do not create a second fixture tree or manifest.

---

## Deterministic fixture reference (copy verbatim)

### TokenSpanMorph — café 🚀

```json
{
  "runId": "glyph-run-cafe-rocket",
  "text": "café 🚀",
  "graphemes": [
    { "id": "g0", "text": "c", "byteStart": 0, "byteEnd": 1 },
    { "id": "g1", "text": "a", "byteStart": 1, "byteEnd": 2 },
    { "id": "g2", "text": "f", "byteStart": 2, "byteEnd": 3 },
    { "id": "g3", "text": "é", "byteStart": 3, "byteEnd": 5 },
    { "id": "g4", "text": " ", "byteStart": 5, "byteEnd": 6 },
    { "id": "g5", "text": "🚀", "byteStart": 6, "byteEnd": 10 }
  ],
  "tokens": [
    { "id": "t0", "label": "caf" },
    { "id": "t1", "label": "é" },
    { "id": "t2", "label": "🚀" },
    { "id": "t3", "label": "↑" },
    { "id": "t4", "label": "<special>" }
  ],
  "edges": [
    { "id": "e0", "sourceSpanIds": ["g0", "g1", "g2"], "targetSpanIds": ["t0"], "kind": "map" },
    { "id": "e1", "sourceSpanIds": ["g3"], "targetSpanIds": ["t1"], "kind": "map" },
    { "id": "e2", "sourceSpanIds": ["g5"], "targetSpanIds": ["t2", "t3"], "kind": "map" },
    { "id": "e3", "sourceSpanIds": [], "targetSpanIds": ["t4"], "kind": "special-insert" }
  ],
  "requireCover": "source",
  "uncoveredGraphemeWhenAllPresent": ["g4"]
}
```

Morph kinds in `semanticModel.morphs` mirror edge kinds: `e0` → `many-to-one`, `e1` → `one-to-one`, `e2` → `one-to-many`, `e3` → `special-insert`.

### PromptSegmentComposer — seven segments, seed 42

```json
{
  "segments": [
    { "id": "seg-system", "tokens": 12, "role": "system" },
    { "id": "seg-prefix", "tokens": 48, "role": "prefix", "reused": true },
    { "id": "seg-user", "tokens": 24, "role": "user" },
    { "id": "seg-image", "tokens": 0, "role": "image" },
    { "id": "seg-tool", "tokens": 18, "role": "tool" },
    { "id": "seg-assistant", "tokens": 16, "role": "assistant" },
    { "id": "seg-tail", "tokens": 8, "role": "tail", "truncated": true }
  ],
  "layoutOptions": {
    "originX": 0,
    "originY": 0,
    "rowHeight": 24,
    "gap": 4,
    "unitWidth": 2,
    "seed": 42
  }
}
```

Expected frozen `layoutPlan.nodes` (computed from `layoutSegmentStrip`):

| nodeId | x | y | width | height | flags |
|---|---|---|---|---|---|
| seg-system | 0 | 0 | 24 | 24 | — |
| seg-prefix | 28 | 0 | 96 | 24 | continuation |
| seg-user | 128 | 0 | 48 | 24 | — |
| seg-image | 180 | 0 | 2 | 24 | — |
| seg-tool | 186 | 0 | 36 | 24 | — |
| seg-assistant | 226 | 0 | 32 | 24 | — |
| seg-tail | 262 | 0 | 16 | 24 | clip |

Total width: **278**. Routes: `[]`.

### RequestLifecycleWaterfall — lifecycle lanes

```json
{
  "events": [
    { "id": "ev-arrival", "lane": "arrival", "start": 0, "end": 0 },
    { "id": "ev-admission", "lane": "admission", "start": 2, "end": 2 },
    { "id": "ev-connect", "lane": "connect", "start": 2, "end": 18 },
    { "id": "ev-first-token", "lane": "first-token", "start": 120, "end": 120 }
  ],
  "layoutOptions": {
    "laneOrder": ["arrival", "admission", "connect", "first-token"],
    "originX": 0,
    "originY": 0,
    "laneHeight": 16,
    "laneGap": 4,
    "pxPerMs": 1
  }
}
```

Expected frozen `layoutPlan.nodes`:

| nodeId | x | y | width | height |
|---|---|---|---|---|
| ev-arrival | 0 | 0 | 1 | 16 |
| ev-admission | 2 | 20 | 1 | 16 |
| ev-connect | 2 | 40 | 16 | 16 |
| ev-first-token | 120 | 60 | 1 | 16 |

Admission `y` ≠ arrival `y`. Connect width = 16. First-token `x` = 120.

---

### Task 1: Consolidate the landed self-contained fixtures

**Files:**
- Modify: `apps/aiperf-flow/examples/p0/README.md`
- Modify: `apps/aiperf-flow/examples/p0/token-span-morph.ir.json`
- Modify: `apps/aiperf-flow/examples/p0/prompt-segment-composer.ir.json`
- Modify: `apps/aiperf-flow/examples/p0/request-lifecycle-waterfall.ir.json`

**Interfaces:**
- Produces: three self-contained IR documents whose props, semantic models,
  layout plans, and source maps contain every value needed for parity tests.

- [ ] **Step 1:** Compare landed documents to the deterministic reference blocks
  above and correct drift in place.
- [ ] **Step 2:** Keep every `sourceMap.source` inside each IR document and point
  it at that document; do not add a TypeScript source-map helper.
- [ ] **Step 3:** Document all three files in `README.md`; do not add a parallel
  manifest when the directory inventory is sufficient.
- [ ] **Step 4:** Verify each JSON file parses with Node `JSON.parse` and strict
  `parseFlowIr` (no trailing commas; UTF-8 café 🚀 preserved).

---

### Task 2: Hand-author `token-span-morph.ir.json`

**Files:**
- Modify: `apps/aiperf-flow/examples/p0/token-span-morph.ir.json`

**Interfaces:**
- Consumes: embedded deterministic token fixture data and P0 capabilities
  `core.glyph-run`, `core.span-map`, `core.semantic-morph`.
- Produces: `FlowIr` with `id: "token-span-morph"`, one scene `tokenizer-morph`, root component `tok-map`.

**IR shape requirements:**

- `irVersion`: `1`
- `capabilities`: require `core.glyph-run`, `core.span-map`, `core.semantic-morph` at `^1.0.0`
- Scene `tokenizer-morph`:
  - `summary`: describes grapheme-to-token morph for `"café 🚀"`
  - `roots`: single `component` node:
    - `id`: `"tok-map"`
    - `capabilityId`: `"core.span-map"`
    - `geometry`: `{ x: 0, y: 0, width: 640, height: 120 }`
    - `props`: `{ runId, text, requireCover: "source", edges }` sourced from fixture
    - `semanticModel`:
      - `entities`: graphemes `g0`–`g5` (label = grapheme text) plus tokens `t0`–`t4`
      - `relations`: empty array (edges live in morphs for span-map)
      - `morphs`: four entries `e0`–`e3` with `sourceIds`/`targetIds`/`kind` from fixture
    - `children`: two child `component` nodes:
      - `glyph-run` (`capabilityId: "core.glyph-run"`) with `props: { runId, text }`
      - `token-rail` group of five `text` nodes for token labels (fallback path)
  - `accessibility.readingOrder`: `["glyph-run", "tok-map", "token-rail"]`
  - `timeline`: one cue `{ id: "morph-reveal", at: 0, duration: 800, target: "tok-map", action: "reveal-edges" }`
  - `camera`: `[{ id: "main-0", at: 0, x: 0, y: 0, zoom: 1 }]`
  - `interactions`: `[{ id: "inspect-tok-map", event: "select", target: "tok-map", action: "inspect" }]`
  - `narration`: ≥ 40 chars describing the café 🚀 tokenizer morph
  - `fallback`: plain-text table listing grapheme → token correspondences

- [ ] **Step 1:** Write the full JSON document following the shape above; embed fixture data, do not reference external files inside IR (self-contained parse target).
- [ ] **Step 2:** Run `node -e "JSON.parse(require('fs').readFileSync('apps/aiperf-flow/examples/p0/token-span-morph.ir.json','utf8'))"` — must succeed.
- [ ] **Step 3:** Confirm every `sourceMap.source` ends with `token-span-morph.ir.json`.

---

### Task 3: Hand-author `prompt-segment-composer.ir.json`

**Files:**
- Modify: `apps/aiperf-flow/examples/p0/prompt-segment-composer.ir.json`

**Interfaces:**
- Consumes: embedded deterministic segment data, frozen layout table, and
  capability `core.segment-strip`.
- Produces: `FlowIr` with `id: "prompt-segment-composer"`, scene `prompt-strip`.

**IR shape requirements:**

- `capabilities`: require `core.segment-strip` at `^1.0.0`
- Scene `prompt-strip`:
  - `roots`: single `component` node:
    - `id`: `"prompt-strip"`
    - `capabilityId`: `"core.segment-strip"`
    - `geometry`: `{ x: 0, y: 0, width: 320, height: 48 }`
    - `props`: `{ segments, layoutOptions }` from fixture (include `seed: 42`)
    - `layoutPlan`: frozen node bounds table from **Deterministic fixture reference** (`version: 1`, `routes: []`)
    - `semanticModel`:
      - `entities`: one per segment id with `label` = role and `kind: "segment"`
      - `relations`: `[{ id: "prefix-reuse", from: "seg-prefix", to: "seg-user", kind: "reuse" }]`
      - `morphs`: `[]`
    - `children`: seven `rect` child nodes (`seg-system` … `seg-tail`) with geometry copied from `layoutPlan` and `accessibility.label` = role
  - `accessibility.readingOrder`: segment ids in fixture order
  - `timeline`: cue highlighting `seg-prefix` reuse at `at: 400`
  - `fallback`: pipe-separated role/token-count table

- [ ] **Step 1:** Write the JSON; `layoutPlan.nodes` must byte-match the computed table (total width 278).
- [ ] **Step 2:** Verify `seg-prefix` node has `continuation: true` and `seg-tail` has `clip: true` inside `layoutPlan`.
- [ ] **Step 3:** JSON parse smoke test as in Task 2.

---

### Task 4: Hand-author `request-lifecycle-waterfall.ir.json`

**Files:**
- Modify: `apps/aiperf-flow/examples/p0/request-lifecycle-waterfall.ir.json`

**Interfaces:**
- Consumes: embedded deterministic lifecycle data, frozen layout table, and
  capability `viz.waterfall`.
- Produces: `FlowIr` with `id: "request-lifecycle-waterfall"`, scene `lifecycle`.

**IR shape requirements:**

- `capabilities`: require `viz.waterfall` at `^1.0.0`
- Scene `lifecycle`:
  - `roots`: single `component` node:
    - `id`: `"lifecycle-waterfall"`
    - `capabilityId`: `"viz.waterfall"`
    - `geometry`: `{ x: 0, y: 0, width: 160, height: 96 }`
    - `props`: `{ events, layoutOptions }` from fixture
    - `layoutPlan`: frozen bounds from **Deterministic fixture reference**
    - `semanticModel`:
      - `entities`: four events + four lane entities (`lane-arrival`, …)
      - `relations`: each event `from` its lane entity
      - `morphs`: `[]`
    - `children`: four `rect` nodes mirroring event bounds; lane labels as sibling `text` nodes at `x: -40`
  - `accessibility.readingOrder`: lane labels then events in time order
  - `timeline`: cue `{ id: "first-token-pulse", at: 1200, duration: 400, target: "ev-first-token", action: "emphasize" }`
  - `fallback`: multi-line text timeline (`arrival @0`, `admission @2`, `connect 2–18`, `first-token @120`)

- [ ] **Step 1:** Write the JSON; confirm connect span width 16 and first-token x 120 in both `layoutPlan` and child geometries.
- [ ] **Step 2:** Confirm admission and arrival occupy different lane `y` values (20 vs 0).
- [ ] **Step 3:** JSON parse smoke test as in Task 2.

---

### Task 5: Schema validation and leaf golden parity tests

**Files:**
- Modify: `apps/aiperf-flow/packages/schema/test/flagship-ir-fixtures.test.ts`
- Create: `apps/aiperf-flow/packages/runtime/test/flagship-ir-parity.test.ts`

**Interfaces:**
- Consumes: schema parsers in the schema test; landed leaf functions in the
  runtime parity test. The schema package must not import runtime code.

- [ ] **Step 1:** Extend the landed schema test to load each `.ir.json`, call
  `parseFlowIr`, and assert strict success plus `irVersion === 1`.
- [ ] **Step 2:** Assert `capabilities[].id` is a superset of every `capabilityId` found by walking `roots` recursively.
- [ ] **Step 3:** In the runtime parity test, recompute prompt layout from
  embedded props with `layoutSegmentStrip` and deep-compare the embedded plan.
- [ ] **Step 4:** Recompute waterfall layout from embedded props with
  `layoutWaterfallNest` and deep-compare the embedded plan.
- [ ] **Step 5:** Run `projectCoverage` on embedded graphemes and edges and
  assert uncovered/covered behavior for `g4`.
- [ ] **Step 6:** Run the focused schema fixture and runtime parity tests; both
  pass without a schema-to-runtime dependency.

---

### Task 6: Wire into `flow:check` and document dependency

**Files:**
- Modify: `apps/aiperf-flow/package.json` (ensure `flow:check` runs schema tests including new file)
- Modify: `docs/superpowers/plans/2026-07-17-aiperf-flow-p0-core-components.md` — add cross-link under Task 8/10 noting golden IR lives in `examples/p0/` (one line only)

**Interfaces:**
- Produces: green `npm run flow:check` with flagship IR validation included.

- [ ] **Step 1:** Confirm root `flow:check` script executes `@aiperf/flow-schema` tests; adjust if the new test file is excluded.
- [ ] **Step 2:** Run full gate:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd /home/anthony/nvidia/projects/aiperf/ajc/rust/apps/aiperf-flow
npm run flow:check
```

Expected: all workspace packages pass; new parity tests green.

- [ ] **Step 3:** Record in `.superpowers/sdd/progress.md` that flagship IR JSON is landed and preview remains untouched.

---

## Dependency order

```text
Task 1 → Tasks 2–4 (IR authoring, parallel) → Task 5 → Task 6
```

Tasks 2, 3, and 4 may run concurrently once Task 1 fixtures exist.

## Out of scope (explicit)

- `apps/aiperf-flow/preview/**` — no fixture ownership transfer, no visual snapshots.
- `.flow` source files for the three wrappers — deferred to stdlib/compiler increment.
- Hybrid capability renderers evaluating these IR files — consumed later by P0 Task 8 in the core-components plan.
- E2E Playwright specs — separate plan after runtime capabilities land.

## Execution options

1. **Subagent-driven (recommended)** — one subagent per task, review between tasks.
2. **Inline** — author all three IR JSON files in one session after Task 1, then run Task 5 parity suite.

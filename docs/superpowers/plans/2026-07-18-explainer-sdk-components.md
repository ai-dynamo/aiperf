<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Explainer SDK Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated bespoke scene composition across all nine explainer decks with typed `sdk.*` and `aiperf.*` compile-time components that expand to ordinary SceneIr and pass strict authoring gates.

**Architecture:** Add a browser-safe TypeScript SDK registry under `apps/explainers/src/flow/sdk/` with deterministic factories producing `SceneFragment` (roots + ports + semantic action bindings + provenance). Extend native scene parsing and symbol expansion for arrays, slots, bounded `for`, semantic `ref()`, component-instance timeline targets, and explicit `freeform` blocks. Wire SDK expansion before existing lowering/validation; keep `SceneRenderer` generic. Migrate all 133 scenes from package-form `@scene { roots }` to native SDK calls.

**Tech Stack:** TypeScript (browser Flow toolchain in `apps/explainers/src/flow/`), Chevrotain parser, Zod IR, React SVG `SceneRenderer`, `.flow` decks, flow-verifier (IR + optional Playwright).

**Spec:** `docs/superpowers/specs/2026-07-18-explainer-sdk-components-design.md`

## Global Constraints

- SDK architecture: typed TypeScript compile-time registry
- Authoring dialect: convert every deck scene from package-form roots to native component calls
- Vocabulary: layered generic `sdk.*` and AIPerf-specific `aiperf.*` packs
- Migration fidelity: normalize to consistent SDK layouts while preserving meaning and timing
- Renderer: keep `SceneRenderer` generic; SDK expands to ordinary Scene IR
- Enforcement: strict gate rejects prohibited repeated raw compositions
- Tests: do not add, modify, delete, or run tests
- Deck coverage: all nine decks and all 133 scene slides
- Work from repo root: `/home/anthony/nvidia/projects/aiperf/ajc/rust`
- Preserve NVIDIA SPDX headers on new/edited source files
- Do not create git commits unless explicitly requested

---

## File map

| Area | Primary files |
|---|---|
| SDK core | `apps/explainers/src/flow/sdk/types.ts`, `registry.ts`, `expand.ts`, `provenance.ts`, `index.ts` |
| Generic pack | `apps/explainers/src/flow/sdk/generic/chrome.ts`, `layout.ts`, `topology.ts`, `motion.ts` |
| AIPerf pack | `apps/explainers/src/flow/sdk/aiperf/architecture.ts`, `execution.ts`, `metrics.ts` |
| Language | `apps/explainers/src/flow/language/ast.ts`, `parser.ts`, `grammar/explainer.ts`, `embedded-scene.ts` |
| Compiler | `apps/explainers/src/flow/compiler/expand-symbols.ts`, `expand-sdk.ts` (new), `validate-sdk-authoring.ts` (new), `lower-explainer-scene.ts`, `lower-explainer.ts`, `compile-explainer.ts`, `link.ts` |
| Barrel | `apps/explainers/src/flow/index.ts`, `apps/explainers/src/flow/compiler/browser.ts` |
| Decks | `apps/explainers/decks-flow/*.flow` (9 files, 133 scenes) |
| Verifier / gates | `apps/explainers/scripts/flow-verifier.mjs`, `scripts/assert-sdk-authoring.mjs` (new) |
| Renderer | `apps/explainers/src/core/diagram/SceneRenderer.tsx` (no SDK host; geometry only if factories need shared helpers) |

---

### Task 1: SDK core types and registry

**Files:**
- Create: `apps/explainers/src/flow/sdk/types.ts`
- Create: `apps/explainers/src/flow/sdk/provenance.ts`
- Create: `apps/explainers/src/flow/sdk/registry.ts`
- Create: `apps/explainers/src/flow/sdk/index.ts`
- Modify: `apps/explainers/src/flow/index.ts`

**Interfaces:**
- Produces: `SdkActionName`, `SceneFragment`, `SdkExpansionContext`, `SdkComponentDefinition`, `SdkComponentFactory`, `createSdkRegistry()`, `lookupSdkComponent(id)`

- [ ] **Step 1: Define SDK types**

```ts
// apps/explainers/src/flow/sdk/types.ts
export const SDK_ACTION_NAMES = [
  "enter", "draw", "trace", "emphasis", "pulse", "stagger", "exit", "fade",
] as const;
export type SdkActionName = (typeof SDK_ACTION_NAMES)[number];

export type SceneFragment = Readonly<{
  roots: readonly RenderNodeIr[];
  ports: Readonly<Record<string, ConnectorEndpointIr>>;
  actions: Readonly<Partial<Record<SdkActionName, readonly string[]>>>;
}>;

export type SdkExpansionContext = Readonly<{
  instanceId: string;
  sourceMap: SourceRange;
  themeTokens: ReadonlyMap<string, JsonValue>;
}>;
```

- [ ] **Step 2: Implement provenance attachment**

Add `attachSdkOrigin(node, origin: SdkOrigin)` in `provenance.ts`. Store origin on IR nodes via a compiler-only optional field (same pattern as existing `sourceMap`); strip before DeckPackage serialization if needed.

- [ ] **Step 3: Implement registry**

`registry.ts` exports `GENERIC_SDK_COMPONENTS` and `AIPERF_SDK_COMPONENTS` arrays of `SdkComponentDefinition`. Each entry pairs a `ComponentDescriptor` (from `schema/component-descriptor.ts`) with a factory stub returning `{ ok: false }` until Task 2 fills implementations.

- [ ] **Step 4: Export from flow barrel**

Re-export `createSdkRegistry`, `SceneFragment`, and registry lookup helpers from `apps/explainers/src/flow/index.ts`.

**Gate:**
```bash
cd apps/explainers && npm run build
```
Expected: PASS (types compile; factories may be stubs).

---

### Task 2: Generic SDK factories — chrome and content

**Files:**
- Create: `apps/explainers/src/flow/sdk/generic/chrome.ts`
- Modify: `apps/explainers/src/flow/sdk/registry.ts`

**Interfaces:**
- Consumes: Task 1 types, existing desugar helpers in `compiler/desugar-scene-primitives.ts` as reference geometry
- Produces: working factories for `sdk.header`, `sdk.panel`, `sdk.card`, `sdk.chip`, `sdk.note`, `sdk.label`, `sdk.callout`, `sdk.divider`, `sdk.bracket`

- [ ] **Step 1: Implement `sdk.header`**

Default layout `(x:18,y:16,width:664,height:44)` with theme surface/ink roles. Expose ports `title`, `caption`. Bind `enter` → header group id.

- [ ] **Step 2: Implement `sdk.panel` and `sdk.card`**

`sdk.panel`: title + detail (maps to current `core.panel` desugar output).
`sdk.card`: title + detail + subtitle (absorbs 145 bespoke rect+text signatures). Accept preset sizes (`compact`, `standard`, `wide`) instead of raw pixel clusters.

- [ ] **Step 3: Implement annotation chrome**

`sdk.note`, `sdk.chip`, `sdk.label`, `sdk.callout`, `sdk.divider`, `sdk.bracket` — each returns a `SceneFragment` matching current desugar macro geometry with normalized spacing.

- [ ] **Step 4: Register descriptors**

Wire strict prop schemas (required `id`, typed strings/numbers, optional theme roles). Mark all factories `deterministic: true`.

**Gate:**
```bash
cd apps/explainers && npm run build
```

---

### Task 3: Generic SDK factories — layout, topology, motion

**Files:**
- Create: `apps/explainers/src/flow/sdk/generic/layout.ts`
- Create: `apps/explainers/src/flow/sdk/generic/topology.ts`
- Create: `apps/explainers/src/flow/sdk/generic/motion.ts`
- Modify: `apps/explainers/src/flow/sdk/registry.ts`

**Interfaces:**
- Produces: factories for layout (`sdk.stack`, `sdk.grid`, `sdk.rail`, `sdk.lane`, `sdk.swimlane`, `sdk.band`, `sdk.stepper`), topology (`sdk.edge`, `sdk.route`, `sdk.pipeline`, `sdk.fanOut`, `sdk.fanIn`), motion (`sdk.signal`, `sdk.pulse`, `sdk.flow`)

- [ ] **Step 1: Layout factories**

Implement row/column/stack/grid/rail/lane/swimlane/band/stepper as pure layout composers over child slot fragments. Each exposes semantic ports for child attachment.

- [ ] **Step 2: `sdk.edge` unification**

Single component with `mode: "connector" | "route" | "path" | "line"`. Internally emit the correct IR node kind (`core.connector`, `core.route`, `core.path`, `core.line`) so existing `SceneRenderer` handles rendering. Accept `from`/`to` as semantic refs or literal endpoints.

- [ ] **Step 3: Topology composites**

`sdk.pipeline`: ordered nodes + auto edges between consecutive ports.
`sdk.fanOut` / `sdk.fanIn`: wrap existing fan IR with trunk/junction defaults and semantic actions `draw`, `trace`, `emphasis`.

- [ ] **Step 4: Motion factories**

`sdk.signal`: node-anchored or path mode; default motion stroke style (opacity 0.55, strokeWidth 2.4).
`sdk.pulse`: replaces `pulse: true` style hacks and hollow rect overlays.
`sdk.flow`: alias composing signal + optional edge draw binding.

**Gate:**
```bash
cd apps/explainers && npm run build
```

---

### Task 4: AIPerf domain SDK pack

**Files:**
- Create: `apps/explainers/src/flow/sdk/aiperf/architecture.ts`
- Create: `apps/explainers/src/flow/sdk/aiperf/execution.ts`
- Create: `apps/explainers/src/flow/sdk/aiperf/metrics.ts`
- Modify: `apps/explainers/src/flow/sdk/registry.ts`

**Interfaces:**
- Produces: `aiperf.controllerCells`, `aiperf.workerMerge`, `aiperf.requestPipeline`, `aiperf.segmentPool`, `aiperf.warmupHandoff`, `aiperf.veloEnvelope`, `aiperf.phaseLifecycle`, `aiperf.registryBootstrap`, `aiperf.metricsExport`

- [ ] **Step 1: Implement architecture composites**

Each domain component composes generic factories only (no deck copy, no slide ids). Props carry labels, stage names, endpoint lists, and theme roles.

- [ ] **Step 2: Register and validate descriptors**

Ensure every `aiperf.*` id is namespaced and appears in `createSdkRegistry()` lookup.

**Gate:**
```bash
cd apps/explainers && npm run build
```

---

### Task 5: SDK expansion engine

**Files:**
- Create: `apps/explainers/src/flow/sdk/expand.ts`
- Create: `apps/explainers/src/flow/compiler/expand-sdk.ts`
- Modify: `apps/explainers/src/flow/compiler/lower-explainer.ts`
- Modify: `apps/explainers/src/flow/compiler/compile-explainer.ts`

**Interfaces:**
- Consumes: `SdkComponentDefinition`, native `component-invocation` AST nodes
- Produces: `expandSdkInvocations(document, registry) → Result<{ document, instanceIndex, actionIndex }>`

- [ ] **Step 1: Expand single invocation**

`expandSdkInvocation(name, props, slots, context)` validates props via existing `validateProps`, calls factory, merges child fragments, assigns stable generated ids `${instanceId}__${role}`.

- [ ] **Step 2: Expand scene render declarations**

Walk native scene bodies; for invocations where registry contains `sdk.*` or `aiperf.*`, replace invocation with expanded roots. Preserve non-SDK declarations for `freeform` pass-through.

- [ ] **Step 3: Wire into compile pipeline**

Insert after `expandSymbolInvocations` and before scene lowering in `lower-explainer.ts`:

```text
parse → expandSymbolInvocations → expandSdkInvocations → lowerExplainerScene → validateExplainerTimelines → safeParseDeckPackage
```

- [ ] **Step 4: Build instance/action indexes**

Record `instanceId → { actions, ports }` for timeline and ref resolution in Task 6.

**Gate:**
```bash
cd apps/explainers && npm run build
```

---

### Task 6: Native language growth

**Files:**
- Modify: `apps/explainers/src/flow/language/ast.ts`
- Modify: `apps/explainers/src/flow/language/parser.ts`
- Modify: `apps/explainers/src/flow/language/grammar/explainer.ts`
- Modify: `apps/explainers/src/flow/compiler/expand-symbols.ts`
- Modify: `apps/explainers/src/flow/compiler/link.ts`

**Interfaces:**
- Produces: AST nodes for array literals, semantic `ref("instance.port")`, named slots, bounded `for` over arrays, `freeform { ... }` blocks, component-instance timeline targets

- [ ] **Step 1: Array and object prop literals**

Extend parser/value resolution so component props accept JSON arrays and nested objects (compile-time constants only).

- [ ] **Step 2: Semantic references**

Add `ref("controller.output")` value kind. Resolve after all SDK factories expand using the instance/port index from Task 5.

- [ ] **Step 3: Slots and bounded `for`**

Allow component invocations with named slots containing nested SDK calls. Allow `for item in items { sdk.panel(...) }` with finite authored arrays; reject unbounded or dynamic expressions.

- [ ] **Step 4: `freeform` blocks**

Explicit block syntax for unique illustration geometry using raw `core.path`, `core.line`, `core.text`, `core.rect` only inside the block boundary.

- [ ] **Step 5: Symbol expansion parity**

Update `expand-symbols.ts` to stop rejecting slots/loops when they appear inside SDK-native scenes (keep rejecting them inside legacy symbol macros if still needed).

**Gate:**
```bash
cd apps/explainers && npm run build
```

---

### Task 7: Semantic timeline expansion

**Files:**
- Modify: `apps/explainers/src/flow/compiler/lower-explainer-scene.ts`
- Modify: `apps/explainers/src/flow/compiler/validate-explainer-timelines.ts`
- Modify: `apps/explainers/src/flow/language/embedded-scene.ts` (native timeline form)

**Interfaces:**
- Consumes: action index from Task 5
- Produces: lowered cues targeting generated node ids from public actions

- [ ] **Step 1: Accept component-instance targets**

Parse cues like `at 900 trace "dispatch" for 1000` where target is a component instance id, not a generated node id.

- [ ] **Step 2: Expand to internal cues**

Map public action → bound node ids; fan out to multiple cues when an action binds multiple targets. Fail closed on unknown instance or unsupported action with available-action hint.

- [ ] **Step 3: Optional timeline template helper**

Implement compiler-side desugar for `sdk.timeline.standardReveal(header, nodes, edges, motion)` that emits semantic action bindings authors can reference with one template call (not 8–12 hand cues per scene).

**Gate:**
```bash
cd apps/explainers && npm run build
```

---

### Task 8: Strict SDK authoring gate

**Files:**
- Create: `apps/explainers/src/flow/compiler/validate-sdk-authoring.ts`
- Create: `apps/explainers/scripts/assert-sdk-authoring.mjs`
- Modify: `apps/explainers/src/flow/compiler/compile-explainer.ts`
- Modify: `apps/explainers/package.json` (add script)

**Interfaces:**
- Produces: `validateSdkAuthoring(sceneAst | SceneIr, provenance) → Result<void>`

- [ ] **Step 1: Signature detectors**

Detect prohibited patterns using provenance + structural heuristics:
- raw rect/text panel signatures outside `freeform`
- duplicated header geometry outside `sdk.header`
- manual connector chains matching `sdk.pipeline`
- manual fan path trees matching `sdk.fanOut` / `sdk.fanIn`
- painted-path + motion-signal pairs matching `sdk.flow`
- pulse overlays outside `sdk.pulse`
- package-form `@scene { roots:` after migration flag enabled

- [ ] **Step 2: Wire gate into compile**

Run when `strict: true` (already used by `load-deck-flows.ts`) after SDK expansion, before DeckPackage emit.

- [ ] **Step 3: CLI scan script**

`npm run assert:sdk-authoring` walks `decks-flow/*.flow`, compiles each deck, prints detected signatures with source ranges.

**Gate:**
```bash
cd apps/explainers && npm run build
node apps/explainers/scripts/assert-sdk-authoring.mjs --help
```

---

### Task 9: Migrate panel-native decks (5 decks, ~65 scenes)

**Files:**
- Modify: `apps/explainers/decks-flow/segment-pools.flow`
- Modify: `apps/explainers/decks-flow/dynosim.flow`
- Modify: `apps/explainers/decks-flow/tstar-warmup.flow`
- Modify: `apps/explainers/decks-flow/velo-deep-dive.flow`
- Modify: `apps/explainers/decks-flow/rust-architecture-atlas.flow`

- [ ] **Step 1: Convert scenes to native syntax**

Replace `render: @scene { roots: [...] timeline: [...] }` with native scene blocks using SDK calls.

- [ ] **Step 2: Apply chrome + node + edge SDK**

Use `sdk.header`, `sdk.panel`/`sdk.card`, `sdk.edge`, `sdk.signal`, `sdk.pulse`, layout rails/steppers where present.

- [ ] **Step 3: Retarget timelines**

Point cues at component instances and public actions; remove references to generated child ids.

**Gate (per deck while migrating):**
```bash
cd apps/explainers && npm run build
cd apps/explainers && npm run flow-verifier:ir -- --from-flow --deck <deck-id> --warn
```

---

### Task 10: Migrate hybrid decks (2 decks, ~32 scenes)

**Files:**
- Modify: `apps/explainers/decks-flow/cellular-algorithms.flow`
- Modify: `apps/explainers/decks-flow/rust-architecture.flow`

- [ ] **Step 1: Replace `core.line` pulses and bespoke paths with `sdk.edge` / `sdk.signal` / `sdk.pulse`**

- [ ] **Step 2: Promote chapter-map pattern to `sdk.swimlane` + `sdk.grid` + `sdk.band` composition**

- [ ] **Step 3: Retarget timelines to semantic actions**

**Gate:**
```bash
cd apps/explainers && npm run flow-verifier:ir -- --from-flow --deck cellular-algorithms --warn
cd apps/explainers && npm run flow-verifier:ir -- --from-flow --deck rust-architecture --warn
```

---

### Task 11: Migrate bespoke-heavy decks (2 decks, ~36 scenes)

**Files:**
- Modify: `apps/explainers/decks-flow/cellular-internals.flow`
- Modify: `apps/explainers/decks-flow/slurm-velo.flow`

- [ ] **Step 1: Replace 145-class bespoke rect+text with `sdk.card`**

Focus on title+detail+subtitle boxes (`cellular-internals` has ~88 alone).

- [ ] **Step 2: Unify absolute routing**

Convert `core.path` / manual coordinates to `sdk.edge` with `mode: "route"` or anchored connector mode.

- [ ] **Step 3: Nested containers**

Use `sdk.stack` / domain composites for container rects nesting child panels (`slurm-velo` allocation boxes).

- [ ] **Step 4: Preserve unique art in `freeform` only where structurally irreducible**

**Gate:**
```bash
cd apps/explainers && npm run flow-verifier:ir -- --from-flow --deck cellular-internals --warn
cd apps/explainers && npm run flow-verifier:ir -- --from-flow --deck slurm-velo --warn
```

---

### Task 12: Final verification and reporting

**Files:**
- Modify: `.superpowers/sdd/progress.md`
- Optional: `apps/explainers/scripts/report-sdk-usage.mjs` (new)

- [ ] **Step 1: Full compile + IR verifier**

```bash
cd apps/explainers && npm run build
cd apps/explainers && npm run flow-verifier:ir -- --from-flow --warn
```

Expected: **0 errors, 0 warnings** across all 9 decks / 133 scenes.

- [ ] **Step 2: Strict authoring gate**

```bash
cd apps/explainers && npm run assert:sdk-authoring
```

Expected: zero prohibited signatures; zero package-form scenes.

- [ ] **Step 3: Static scan for migration completeness**

Scan `decks-flow/*.flow` for forbidden patterns:
- `roots:` package-form openings
- `capability: "core.panel"` etc. outside `freeform`
- timeline targets matching `__` generated id suffixes

- [ ] **Step 4: SDK usage report**

Emit counts of `sdk.*` / `aiperf.*` invocations per deck and remaining `freeform` blocks.

- [ ] **Step 5: Update progress ledger**

Record completion in `.superpowers/sdd/progress.md` under `explainers-sdk-components`.

---

## Self-review checklist

| Spec requirement | Task |
|---|---|
| Typed TS registry + factories | 1–4 |
| Native component authoring | 6, 9–11 |
| Semantic ports + refs | 5–6 |
| Semantic timeline actions | 7 |
| Generic + AIPerf packs | 2–4 |
| Provenance | 1, 8 |
| Strict authoring gate | 8 |
| All 9 decks / 133 scenes | 9–11 |
| SceneRenderer stays generic | (no renderer SDK host tasks) |
| No tests added/modified/run | all gates use build + verifier + scans only |

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-18-explainer-sdk-components.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task with review between tasks
2. **Inline Execution** — implement tasks sequentially in this session with checkpoints after Tasks 5, 8, and 11

Which approach?

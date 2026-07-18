# AIPerf Flow ComponentDescriptor Schema Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define strict `ComponentDescriptor` schema for P0 hybrid stdlib components — typed props, slots, events, and classification — and wire it to existing `CapabilityDescriptor` records so compiler validation, runtime binding, and generated reference docs share one source of truth.

**Architecture:** `@aiperf/flow-schema` owns both descriptor families.
`CapabilityDescriptor` remains the runtime execution contract: dispatch,
evaluated-scene/display contributions, Canvas eligibility, semantic-twin and
SVG/HTML fallback projections, exploration restoration, quality tiers, and
cost. `ComponentDescriptor` is the stdlib authoring contract: symbol surface,
prop/slot/event schemas, classification, semantic identities, timeline anchors,
and primary-leaf linkage. P0 hybrids share the same stable id in both manifests;
schema helpers prove the linkage at build time. Prop validation runs against
component descriptors before runtime sees IR.

**Tech Stack:** TypeScript strict mode, Zod 4, Vitest.

**Scope boundary:** Files under `apps/aiperf-flow/packages/schema` only. No preview, runtime, compiler, language, or stdlib `.flow` changes in this plan.

## Global Constraints

- Unknown props, slot fields, and event payload fields fail closed (strict Zod objects).
- P0 scope is the six hybrid stdlib components already registered in `P0_CAPABILITIES`: `core.glyph-run`, `core.span-map`, `core.semantic-morph`, `core.segment-strip`, `viz.queue`, `viz.waterfall`.
- Each P0 hybrid declares exactly one `leafId` that must exist in
  `P0_CAPABILITIES` by completion of this plan.
- `ComponentDescriptor.id` equals `ComponentDescriptor.capabilityId` for P0 hybrids (1:1 runtime binding per core-components design rule 5).
- Classification enum is exactly `flow-only`, `hybrid`, `leaf` — no synonyms.
- Leaf kernels remain capability-only at runtime; component descriptors for leaves are optional and out of P0 hybrid scope unless needed for linkage tests.
- Descriptor metadata must be backend-neutral. It may declare Canvas or future
  accelerator eligibility, but authored semantics and events cannot depend on a
  rendering backend.
- Every hybrid capability declares an always-available semantic-twin projection,
  simplified SVG/HTML fallback, deterministic hit-region policy, and
  pause-to-explore/exact-beat-resume policy.
- Quality tiers identify decorative effects that may degrade. They may not
  suppress semantic entities, relations, captions, evidence, focus, or actions.
- Activate `.venv` before repo commands: `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Verify with `cd apps/aiperf-flow && npm test -w @aiperf/flow-schema && npm run build -w @aiperf/flow-schema`.
- Do not create commits unless the user explicitly requests them.

## Descriptor relationship

| Concern | `CapabilityDescriptor` | `ComponentDescriptor` |
|---|---|---|
| Primary consumer | Runtime registry, packer require checks | Compiler/linker, symbol grammar, reference docs |
| Identity | `id`, `version` | `id`, `symbolExport`, `capabilityId`, `version` |
| Execution | `kind`, `nodeKinds`, `deterministic`, display/hit contract, semantic twin, fallback, exploration, quality, cost | — |
| Authoring | — | `classification`, `props`, `slots`, `events`, semantic identities, timeline anchors |
| Hybrid linkage | Leaf ids registered as separate capabilities | `leafId` points at leaf capability id |
| IR surface | `nodeKinds` includes `"component"` or `"leaf"` | Prop keys must match `ComponentNodeIr.props` keys |

Linkage invariants enforced by schema helpers:

1. Every `ComponentDescriptor.capabilityId` resolves to a `CapabilityDescriptor.id` in the supplied capability manifest.
2. P0 hybrid: `classification === "hybrid"` implies exactly one non-empty
   `leafId` registered as a leaf capability (`nodeKinds` contains `"leaf"`).
3. P0 hybrid: `ComponentDescriptor.id === capabilityId ===` the public component id (`core.glyph-run`, not `core.${kind}`).
4. `flow-only` components omit `leafId`; `hybrid` components require it; `leaf`
   classification is reserved for internal kernels and must not appear in the
   P0 stdlib component catalog.
5. Event names use the `on-*` surface convention; action hooks (`inspect`, `focus`, `scrub`, `compare`) are enumerated action ids, not free strings.

## P0 hybrid inventory (descriptor content)

Authoritative prop/slot/event shapes derive from [`2026-07-17-aiperf-flow-core-components-design.md`](../specs/2026-07-17-aiperf-flow-core-components-design.md) and flagship wrapper examples in the same record. Register these six descriptors in Task 4.

| Component id | Symbol | Primary leaf | Key props (required called out) | Slots | Events |
|---|---|---|---|---|---|
| `core.glyph-run` | `GlyphRun` | `leaf.glyph-measure` | `text` (required), `fontFamily`, `fontSize`, `writingMode` | — | `on-select-span` |
| `core.span-map` | `SpanMap` | `leaf.span-interval` | `source`, `target`, `edges`, `requireCover` | `target-view`, `edge-chrome` | `on-select-edge`, `on-inspect-span` |
| `core.semantic-morph` | `SemanticMorph` | `leaf.correspondence-tween` (descriptor added by this plan; runtime implementation deferred post-P0) | `entities`, `relations`, `beats` | `source-chrome`, `target-chrome` | `on-scrub-beat` |
| `core.segment-strip` | `SegmentStrip` | `core.segment-strip.layout` | `segments`, `orientation`, `gap` | `segment-chrome`, `continuation` | `on-select-segment` |
| `viz.queue` | `Queue` | `viz.queue.policy` | `policy`, `capacity`, `items` | `item-chrome`, `waiter-chrome` | `on-select-item`, `on-inspect-dequeue` |
| `viz.waterfall` | `Waterfall` | `viz.waterfall.nest-layout` | `lanes`, `intervals`, `openSpanPolicy` | `lane-chrome`, `interval-chrome` | `on-select-interval`, `on-focus-lane` |

Notes for implementers:

- Prop types reuse shared field primitives (scalar, enum, list, object, ref) backed by existing `JsonValue` compatibility for IR literals.
- Slot input contracts name accepted child component ids or `"render-tree"` for arbitrary subtree fills.
- `core.semantic-morph` keeps hybrid classification and leaf linkage even though
  the tween implementation is deferred. The current capability manifest does
  not register `leaf.correspondence-tween`; this plan adds its deterministic
  leaf descriptor before linkage tests require it.

## Current implementation baseline

`packages/schema/src/component-descriptor.ts` and
`test/component-descriptor.test.ts` are landed. The current contract uses
`symbolExport`, `leafId`, record-shaped `props`/`slots`, string-array `events`,
and `createComponentCatalog`. Tasks below evolve that contract in place; they
do not create a second descriptor module or silently rename the public surface.
- Default values are descriptor metadata only; IR validation rejects missing required props regardless of runtime defaults.

---

## Task 1: Shared field primitives for props, slots, and events

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/component-field.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Create: `apps/aiperf-flow/packages/schema/test/component-field.test.ts`

**Interfaces:**
- Produces: enriched `ComponentPropDescriptor`, `ComponentSlotDescriptor`,
  `ComponentEventDescriptor`, `ActionHook`, strict Zod schemas, and parse
  helpers returning `Result<T>` with `COMPONENT_FIELD_INVALID` diagnostics.

- [ ] **Step 1:** Failing tests for classification enum exhaustiveness, prop required/default mutual exclusion, slot multiplicity (`single` | `list`), event `on-*` name pattern, and unknown-field rejection on strict parse.
- [ ] **Step 2:** Implement field primitives; reuse `jsonValueSchema` only as an IR-value compatibility check, not as the descriptor shape (descriptor meta-types are stricter than instance values).
- [ ] **Step 3:** Export from `index.ts`; `npm test -w @aiperf/flow-schema` passes for new tests.

---

## Task 2: ComponentDescriptor core type and manifest

**Files:**
- Modify: `apps/aiperf-flow/packages/schema/src/component-descriptor.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Modify: `apps/aiperf-flow/packages/schema/test/component-descriptor.test.ts`

**Interfaces:**
- Preserves: `ComponentDescriptor`, `ComponentCatalog`,
  `createComponentCatalog(descriptors)`, `parseComponentDescriptor(input)`, and
  `safeParseComponentDescriptor(input)`.

Required `ComponentDescriptor` fields:

- `id` — namespaced component id matching capability id for hybrids.
- `symbolExport` — PascalCase export (`SpanMap`).
- `version` — semantic version aligned with paired capability.
- `capabilityId` — runtime dispatch target.
- `classification` — `flow-only` | `hybrid` | `leaf`.
- `description` — human summary for docs and diagnostics.
- `props` — readonly record of `ComponentPropDescriptor`.
- `slots` — readonly record of `ComponentSlotDescriptor` (empty allowed).
- `events` — readonly record of `ComponentEventDescriptor` (migrated from the
  landed string array so payload/action metadata remains typed).
- `leafId` — optional; required iff `classification === "hybrid"`.
- `semanticContract`, `timelineAnchors`, `displayContract`,
  `semanticTwinContract`, `fallbackContract`, `explorationContract`, and
  `qualityContract` — backend-neutral north-star contracts.

Catalog behavior mirrors `createCapabilityManifest`: sort by `id`, reject
duplicates with `COMPONENT_DUPLICATE` diagnostic.

- [ ] **Step 1:** Extend landed tests for hybrid-without-leaf,
  flow-only-with-leaf, typed-event migration, duplicate ids, north-star
  contracts, and strict unknown top-level fields.
- [ ] **Step 2:** Evolve the landed descriptor type, Zod schema, catalog builder,
  and parse wrappers in place; preserve public names above.
- [ ] **Step 3:** Tests green.

---

## Task 3: Capability linkage and cross-manifest validation

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/component-capability-link.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Create: `apps/aiperf-flow/packages/schema/test/component-capability-link.test.ts`

**Interfaces:**
- Consumes: `ComponentCatalog`, `CapabilityRegistryManifest` (from
  `P0_CAPABILITIES` / `FOUNDATION_CAPABILITIES`).
- Produces: `validateComponentCapabilityLinkage(components, capabilities)`, `resolveComponentCapabilityId(descriptor)`, `findComponentDescriptor(id, manifest)`.

Validation rules:

- Missing capability target → `COMPONENT_CAPABILITY_MISSING`.
- Hybrid `leafId` not found or not a leaf capability →
  `COMPONENT_LEAF_MISSING`.
- Component id ≠ capabilityId for entries in the P0 hybrid registry → `COMPONENT_CAPABILITY_ID_MISMATCH`.
- Prop name collisions and duplicate slot/event names → `COMPONENT_SURFACE_DUPLICATE`.

- [ ] **Step 1:** Failing tests using synthetic manifests for each error code and one happy-path P0-sized fixture.
- [ ] **Step 2:** Implement linkage validator returning `Result<void>` (value unit) with actionable repair text.
- [ ] **Step 3:** Tests green.

---

## Task 4: P0 hybrid component descriptor registry

**Files:**
- Modify: `apps/aiperf-flow/packages/schema/src/component-descriptor.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/capability.ts`
- Create: `apps/aiperf-flow/packages/schema/test/p0-component-descriptors.test.ts`

**Interfaces:**
- Produces: `P0_COMPONENTS`, `P0_COMPONENT_IDS`, `P0_COMPONENT_SYMBOLS` (id → symbol map).

- [ ] **Step 1:** Failing tests asserting all six P0 hybrid ids, symbols,
  classifications, leaf ids, required props, named slots, and events match the
  inventory table above.
- [ ] **Step 2:** Add the missing deterministic
  `leaf.correspondence-tween` capability descriptor, register the six component
  descriptors, build the catalog via `createComponentCatalog`, and export the
  compile-time id list.
- [ ] **Step 3:** `validateComponentCapabilityLinkage(P0_COMPONENTS, P0_CAPABILITIES)` passes in test setup (single authoritative linkage assertion for P0).

---

## Task 5: Component prop validation against IR nodes

**Files:**
- Modify: `apps/aiperf-flow/packages/schema/src/component-descriptor.ts`
- Modify: `apps/aiperf-flow/packages/schema/test/ir.test.ts`
- Create: `apps/aiperf-flow/packages/schema/test/component-props-validation.test.ts`

**Interfaces:**
- Consumes: `ComponentDescriptor`, `ComponentNodeIr` props bag from `ir.ts`.
- Produces: `validateComponentProps(descriptor, props)`, `validateComponentNode(descriptor, node)` (props only in this task; slot fill validation deferred to compiler plan).

Validation behavior:

- Unknown prop keys → `COMPONENT_PROP_UNKNOWN`.
- Missing required props → `COMPONENT_PROP_REQUIRED`.
- Wrong scalar kind / enum member → `COMPONENT_PROP_TYPE`.
- Extra props fail even when values are well-typed (strict surface).

- [ ] **Step 1:** Failing tests for `core.span-map` happy path, unknown prop, missing `requireCover`, and enum violation fixtures.
- [ ] **Step 2:** Implement validators without modifying `ComponentNodeIr` shape.
- [ ] **Step 3:** Extend `ir.test.ts` with one component node case proving validated props round-trip through `parseFlowIr` then descriptor validation.

---

## Task 6: Verification gate

- [ ] `npm test -w @aiperf/flow-schema` green (all six test files).
- [ ] `npm run build -w @aiperf/flow-schema` green.
- [ ] No new exports outside `@aiperf/flow-schema`; runtime/compiler consumers remain future plans.
- [ ] Update progress ledger at `.superpowers/sdd/progress.md` with ComponentDescriptor schema completion note.

---

## Dependency order

```text
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6
```

Task 3 depends on Task 2 manifest shape and existing `P0_CAPABILITIES`. Task 4 depends on Task 3 linkage helper. Task 5 depends on Task 4 registry content.

## Relationship to sibling plans

- [`2026-07-17-aiperf-flow-p0-core-components.md`](2026-07-17-aiperf-flow-p0-core-components.md) Task 3 registered runtime capabilities; this plan adds the authoring half without changing capability ids.
- Compiler symbol grammar and slot fill validation consume `ComponentDescriptor` in a later increment; this plan only ships schema types, registries, prop validation, and tests.
- Runtime `resolveCapabilityId` is unchanged; component descriptors do not alter dispatch.

## Execution options

1. **Subagent-driven (recommended)** — one subagent per task, review between tasks.
2. **Inline** — Tasks 1–3 in one session, checkpoint before P0 registry authoring (Task 4).

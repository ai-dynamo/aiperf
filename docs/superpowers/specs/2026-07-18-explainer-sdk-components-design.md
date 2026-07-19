<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Explainer SDK Components Design

**Date:** 2026-07-18
**Status:** Approved
**Scope:** Replace repeated bespoke scene composition with typed SDK components across all nine explainer decks

## Goal

Deck authors should describe diagram meaning and data, not assemble recurring
rectangles, labels, routes, branches, motion guides, and cue choreography by
hand. Replace repeated bespoke composition in all explainer decks with a typed,
browser-safe SDK. Add SDK capabilities whenever an existing scene pattern
cannot be expressed cleanly.

## Locked decisions

| Decision | Choice |
|---|---|
| SDK architecture | Typed TypeScript compile-time registry |
| Authoring dialect | Convert every deck scene from package-form roots to native component calls |
| Vocabulary | Layered generic `sdk.*` and AIPerf-specific `aiperf.*` packs |
| Migration fidelity | Normalize to consistent SDK layouts while preserving meaning and timing |
| Renderer | Keep `SceneRenderer` generic; SDK expands to ordinary Scene IR |
| Enforcement | Strict gate rejects prohibited repeated raw compositions |
| SDK implementation | TypeScript descriptors and factories |
| Tests | Do not add, modify, delete, or run tests |
| Deck coverage | All nine decks and all 133 scene slides |

## Current state

Every explainer scene is currently authored in package form:

```text
render: @scene {
  roots: [
    { capability: "core.panel", ... }
    { capability: "core.connector", ... }
  ]
  timeline: [...]
}
```

The local browser compiler already contains:

- component descriptors and catalogs,
- strict prop validation,
- component invocation AST nodes,
- symbol collection and flat invocation expansion,
- native scene parsing and lowering,
- capability validation,
- first-class fan routing,
- in-browser deck compilation and developer verification.

However, current decks do not invoke components. Component expansion supports
only flat calls, rejects slots and loops, and does not provide a standard
registry of factories that can generate Scene IR fragments. As a result, the
decks repeatedly author component internals directly.

## Corpus baseline

Audit of all nine decks under `apps/explainers/decks-flow/` (133 scene slides,
~1,730 capability nodes, ~1,760 timeline cues):

| Metric | Count | Notes |
|---|---:|---|
| Scene slides | 133 | One embedded `@scene` per slide |
| `core.panel` | 471 | Desugar macro; primary node box |
| Bespoke `core.rect` + text | 145 | Top migration target (~40 scenes) |
| `core.header` | 106 | 95 use identical chrome geometry `(18,16,664,44)` |
| `core.connector` | 114 | Node-anchored edges |
| `core.path` | 104 | Absolute SVG routing (bespoke) |
| `core.route` | 33 | Orthogonal auto-routes |
| `core.line` | 25 | Coordinate-pair edges (bespoke) |
| `motion.signal` | 112 | Flow overlays; style duplicated ~84× |
| Fan nodes (`core.fan-out` / `core.fan-in`) | 18 | 13 scenes across 7 decks |
| Pulse workarounds | ~79 | `motion.pulse` (1) + `pulse: true` (46) + hollow rects (32) |
| Timeline `enter` cues | 1,179 | Dominant choreography pattern |
| Timeline `draw` cues | 411 | Often on connectors after node reveal |

Deck archetypes:

| Archetype | Decks | Primary migration focus |
|---|---|---|
| Panel-native | `dynosim`, `tstar-warmup`, `rust-architecture-atlas`, `segment-pools`, `velo-deep-dive` | Chrome presets, motion defaults, timeline templates |
| Bespoke-heavy | `cellular-internals`, `slurm-velo` | `sdk.card` (3-line boxes), `sdk.edge`, nested groups |
| Hybrid | `cellular-algorithms`, `rust-architecture` | Chapter maps, line→edge unification, pulse consolidation |

## Toolchain gaps

The browser-owned compiler under `apps/explainers/src/flow/` already ships
component descriptors, strict prop validation, symbol expansion, embedded-scene
lowering, fan routing, and in-browser deck compilation. Gaps blocking SDK
migration:

- No populated `sdk.*` / `aiperf.*` factory registry wired into compile.
- Symbol expansion supports flat invocations only; slots, arrays, bounded
  `for`, semantic `ref()`, and component-instance timeline targets are rejected.
- Decks still use package-form `@scene { roots: [...] }` with raw
  `capability: "core.*"` nodes instead of native component calls.
- No post-expansion strict authoring gate for prohibited bespoke signatures.
- `SceneRenderer` remains foundation-only (correct); SDK must expand to SceneIr,
  not introduce runtime React component hosts.
- P0 hybrid capabilities (`viz.queue`, `viz.waterfall`, etc.) are defined but
  unused by any deck; domain SDK components compose generic factories instead.

## Migration priority

Apply SDK components in this order so each capability lands across all
applicable decks before the next family is added:

1. **`sdk.card` / `sdk.panel`** — eliminate 145 bespoke rect+text compositions.
2. **`sdk.header` (+ optional `sdk.stepper` / `sdk.rail`)** — dedupe 95
   identical header geometries and standardize phase/tag chrome.
3. **`sdk.edge`** — unify 129 `core.path` / `core.line` with connector/route.
4. **`sdk.pulse`** — replace ~79 pulse workarounds with one motion component.
5. **`sdk.timeline.standardReveal`** — collapse dominant enter→draw→trace
   choreography into semantic component actions.
6. **Topology composites** — `sdk.pipeline`, `sdk.fanOut`, `sdk.fanIn`, and
   AIPerf domain packs for repeated multi-node archetypes.

## Architecture

```text
native @scene component calls
  → parse arrays / objects / slots / bounded iteration
  → resolve sdk.* or aiperf.* descriptor
  → validate props and slots
  → invoke deterministic TypeScript factory
  → SceneFragment (roots + ports + action bindings + provenance)
  → resolve semantic references and timeline targets
  → ordinary SceneIr
  → existing schema / verifier / SceneRenderer
```

SDK expansion is a compiler concern. The production renderer receives the same
generic groups, text, connectors, fans, paths, and timeline cues it renders
today. No SDK React component host or second renderer is introduced.

## SDK registry

Create `apps/explainers/src/flow/sdk/`:

```text
sdk/
  types.ts
  registry.ts
  expand.ts
  provenance.ts
  generic/
    chrome.ts
    layout.ts
    topology.ts
    motion.ts
  aiperf/
    architecture.ts
    execution.ts
    metrics.ts
```

Each entry contains:

```ts
type SdkComponentDefinition = Readonly<{
  descriptor: ComponentDescriptor;
  factory: SdkComponentFactory;
  actions: readonly SdkActionName[];
}>;

type SdkComponentFactory = (
  props: Readonly<Record<string, JsonValue>>,
  slots: Readonly<Record<string, readonly SceneFragment[]>>,
  context: SdkExpansionContext,
) => Result<SceneFragment>;

type SceneFragment = Readonly<{
  roots: readonly RenderNodeIr[];
  ports: Readonly<Record<string, ConnectorEndpointIr>>;
  actions: Readonly<Partial<Record<SdkActionName, readonly string[]>>>;
}>;
```

Factories are pure and deterministic. They receive no DOM, React, filesystem,
network, wall clock, or global mutable state. Instance ids seed every generated
node id, so expansion is stable.

## Semantic ports and references

Components expose named ports such as `input`, `output`, `control`, `result`,
or an indexed family such as `worker[0]`. Native component props may reference
ports by component instance and port name:

```text
sdk.route(
  id: "dispatch-route"
  from: ref("controller.output")
  to: ref("workers.input")
)
```

The compiler resolves semantic ports after all component factories have
expanded. Missing components, missing ports, duplicate ids, and ambiguous
indexed references fail with source-oriented diagnostics. Deck source never
references generated child ids.

## Semantic timeline actions

Each fragment maps public actions to generated targets:

```text
actions: {
  enter: ["dispatch"]
  draw: ["dispatch__trunk", "dispatch__branches"]
  trace: ["dispatch__fan"]
  emphasis: ["dispatch__source", "dispatch__targets"]
}
```

An authored cue targets the component instance:

```text
at 900 trace "dispatch" for 1000
```

The compiler expands it to internal cues using the action binding. An
unsupported action fails with a diagnostic naming the component and supported
actions. This keeps internal ids private and prevents SDK refactors from
breaking deck timelines.

## Native language growth

Native scene authoring gains:

- JSON arrays and nested objects as component props,
- component references and semantic port references,
- named slots,
- bounded `for` expansion over authored arrays,
- component-instance timeline targets,
- explicit `freeform` blocks for unique illustration primitives.

Iteration is compile-time only, bounded by authored finite arrays, and cannot
execute arbitrary expressions. Slots accept component invocations, not raw
source text.

## Generic SDK pack

### Chrome and content

- `sdk.header`
- `sdk.panel`
- `sdk.card`
- `sdk.chip`
- `sdk.note`
- `sdk.label`
- `sdk.legend`
- `sdk.callout`
- `sdk.divider`
- `sdk.bracket`

### Layout

- `sdk.stack`
- `sdk.grid`
- `sdk.rail`
- `sdk.lane`
- `sdk.swimlane`
- `sdk.band`
- `sdk.stepper`
- `sdk.matrix`
- `sdk.layerStack`

### Topology

- `sdk.edge`
- `sdk.route`
- `sdk.pipeline`
- `sdk.fanOut`
- `sdk.fanIn`
- `sdk.hubSpoke`
- `sdk.tree`
- `sdk.bidirectionalLink`

### Motion and state

- `sdk.signal`
- `sdk.flow`
- `sdk.pulse`
- `sdk.stateTransition`

Generic components accept theme roles and semantic data. They do not contain
AIPerf terminology or deck copy.

## AIPerf SDK pack

- `aiperf.controllerCells`
- `aiperf.workerMerge`
- `aiperf.metricsExport`
- `aiperf.requestPipeline`
- `aiperf.segmentPool`
- `aiperf.warmupHandoff`
- `aiperf.veloEnvelope`
- `aiperf.phaseLifecycle`
- `aiperf.registryBootstrap`

Domain components compose generic SDK factories and expose domain-relevant
ports and actions. Their props provide labels, stages, endpoint lists, policy
states, and theme roles. No deck-specific sentence or fixed slide id is stored
in the SDK.

## Provenance

Every generated node carries compiler-only provenance while being assembled:

```ts
type SdkOrigin = Readonly<{
  componentId: string;
  instanceId: string;
  sourceMap: SourceRange;
  generatedRole: string;
}>;
```

Provenance supports diagnostics and the strict authoring gate. It may be
removed from serialized DeckPackage output after validation; it is not a
renderer contract.

## Strict SDK-authoring gate

The gate runs after expansion, while provenance is available. It rejects:

- raw rect/text groups matching panel, card, header, note, chip, or label
  signatures,
- manually repeated row/column/grid placement,
- chains of raw connectors representing pipelines,
- manually assembled split/merge path trees,
- duplicated painted-path and motion-signal pairs,
- repeated pulse overlays,
- raw groups matching an AIPerf SDK component contract,
- package-form `roots` scenes after migration.

Unique illustration geometry is allowed only inside an explicit native
`freeform` block. Freeform content may use path, line, text, and shape
primitives, but cannot contain repeated component signatures or semantic
connectivity that an SDK topology component already represents.

The gate reports the source range, detected signature, and replacement SDK
component. It fails compilation; it does not silently rewrite authored source.

## Deck migration

Rewrite all scene slides in:

1. `segment-pools.flow`
2. `dynosim.flow`
3. `tstar-warmup.flow`
4. `velo-deep-dive.flow`
5. `slurm-velo.flow`
6. `rust-architecture.flow`
7. `rust-architecture-atlas.flow`
8. `cellular-internals.flow`
9. `cellular-algorithms.flow`

Migration rules:

- Convert every package-form scene to native syntax.
- Replace repeated raw composition with SDK calls.
- Normalize spacing, panel sizing, routes, and branch geometry to SDK layout
  rules.
- Preserve slide meaning, narration, labels, ordering, and approximate timing.
- Target SDK component instances in timelines, never generated ids.
- Retain unique freeform artwork only when it has no reusable structural
  meaning.
- Add a generic or AIPerf SDK component when a repeated pattern has no clean
  representation.

Migration proceeds by component family rather than deck-specific one-off
factories so each new SDK capability is adopted across all applicable decks
before the next family is added.

## Error handling

- Unknown SDK component: fail with available namespace/component suggestions.
- Invalid prop or slot: fail with descriptor type, requiredness, and repair.
- Duplicate instance id: fail before factory expansion.
- Factory failure: include component id, instance id, source range, and cause.
- Unknown semantic port: list the component's available ports.
- Unsupported timeline action: list supported public actions.
- Expansion cycle: report the component composition stack.
- Iteration limit exceeded: report authored collection and maximum.
- Prohibited bespoke composition: report detected signature and SDK
  replacement.
- Non-finite or invalid generated geometry: attribute failure to SDK component
  and generated role.

## Verification

Per explicit user direction, do not add, modify, delete, or run tests.

Allowed gates:

1. TypeScript no-emit check.
2. Production Vite build.
3. Compile all nine decks in browser-toolchain mode.
4. Verify all 133 scenes and timelines.
5. Static scan: zero package-form scenes.
6. Strict SDK-authoring gate: zero prohibited bespoke compositions.
7. IR verifier: zero errors and zero warnings.
8. Static import-boundary and browser-safety scans.
9. Report SDK component usage and remaining explicit freeform blocks.

## Success criteria

- Every explainer scene uses native authoring.
- Repeated visual structures are SDK component calls, not raw composition.
- Generic and AIPerf component packs are typed, deterministic, and
  browser-safe.
- Timelines target public component actions.
- Semantic connectivity uses component ports rather than generated ids.
- `SceneRenderer` remains generic.
- The strict authoring gate prevents bespoke patterns from returning.
- All nine decks compile and all 133 scenes verify with zero warnings.
- No tests are added, changed, deleted, or run.

## Non-goals

- A visual scene editor.
- Runtime React SDK components.
- Arbitrary user code execution in `.flow`.
- General-purpose constraint solving or automatic graph layout.
- Pixel parity with the pre-migration bespoke layouts.
- Moving the locally owned browser compiler out of `apps/explainers` into a
  separate Flow workspace (the standalone `apps/aiperf-flow` workspace has been
  removed).

# AIPerf Flow Core Components Design

## Status

Proposed design for the **top 25 shareable core/viz standard-library components**.
This record defines reusable typed `.flow` components that compose other
components, plus the minimal runtime leaf kernel required when declarative
composition is insufficient.

Companion records:

- [`2026-07-17-aiperf-flow-design.md`](2026-07-17-aiperf-flow-design.md) —
  approved product architecture;
- [`2026-07-17-aiperf-flow-component-catalog.md`](2026-07-17-aiperf-flow-component-catalog.md) —
  full capability and symbol vocabulary derived from the Rust codebase.

## Design goal

Authors commit only `.flow` files. Bespoke visualizations must be lightweight
wrappers over a reusable standard library, not document-specific React,
TypeScript, or CSS.

Every component participates in AIPerf Flow’s live-cinematic contract. A
component is not merely a React, SVG, or Canvas renderer: it contributes
backend-neutral semantic state, layout-plan data, deterministic display-list
commands, hit regions, timeline anchors, and semantic-twin output. The runtime
renders those products through a Canvas 2D visual backend, an always-mounted
semantic HTML twin, and a simplified SVG/HTML fallback.

The visual target is the fidelity of a professionally produced high-resolution
explainer rendered live. High-resolution video is the quality metaphor, not the
primary output. Components must remain interactive, responsive, inspectable,
and accessible while the deterministic narrative is playing.

The authoring burden for bespoke components stays low when:

1. **Runtime leaves are small** — measurement, layout plans, policy simulation,
   and analysis only where `.flow` cannot express the behavior honestly.
2. **Stdlib components are composable** — typed props, slots, events, theme
   tokens, and extension points.
3. **Domain symbols wrap stdlib** — `TokenSpanMorph`, `PromptSegmentComposer`,
   `RequestLifecycleWaterfall`, and similar names are `.flow` compositions, not
   parallel renderers.

## Three-tier model (unchanged)

| Tier | What it is | Author commits |
|---|---|---|
| Runtime leaf | Narrow capability: deterministic layout/analysis/measurement | Never (stdlib / packer only) |
| Stdlib component | Typed reusable `.flow` definition with slots | Yes, when extending the library |
| Bespoke component | Project- or scene-specific wrapper over stdlib | Yes — primary author surface |

Promotion rule: repetition promotes composition to a stdlib symbol; algorithm,
cardinality, or specialized accessibility promotes a **leaf**, not a whole
visual package.

## Stdlib component contract

Every stdlib component exposes the same contract shape so compiler validation,
runtime binding, authoring skills, and generated reference docs stay aligned.

### Required surface

- **Component id** — stable namespaced id (`core.semantic-entity`, `viz.queue`).
- **Symbol export** — PascalCase Flow symbol (`SemanticEntity`, `Queue`).
- **Typed props** — required/optional/default; strict schema; unknown fields fail.
- **Typed slots** — named regions with input contracts; default chrome provided.
- **Events/actions** — `on-*` for interaction; declarative action hooks for
  inspect, focus, scrub, compare.
- **Semantic identities** — stable entity/relationship ids that survive layout.
- **Timeline anchors** — named beats for narration and morph correspondence.
- **Display contract** — backend-neutral draw commands, paint bounds, hit
  regions, damage bounds, quality tiers, and deterministic ordering.
- **Theme tokens** — safe visual overrides; unsafe semantic overrides rejected.
- **Accessibility twin** — semantic HTML structure, reading order, keyboard
  path, focus/selection synchronization, transcript linkage, and textual or
  tabular fallback.
- **Exploration contract** — safe pause points, temporary camera/selection
  behavior, and deterministic restoration when playback resumes from the same
  beat.
- **Backend contract** — Canvas 2D support, simplified SVG/HTML fallback, and
  optional accelerator eligibility without backend-specific authored meaning.
- **Classification** — `flow-only`, `hybrid`, or `leaf` (see below).

### Classification

- **`flow-only`** — no dedicated runtime leaf; composes foundation + other stdlib.
- **`hybrid`** — stdlib component + one narrow leaf for layout/analysis.
- **`leaf`** — runtime capability only; not authored directly in scenes except
  via stdlib internals or advanced packs.

### Bespoke authoring pattern

```text
symbol TokenSpanMorph(run: GlyphRunRef, tokens: TokenRef[], edges: MapEdge[]) {
  SpanMap(
    id = "tok-map",
    source = { run: run },
    target = { spans: tokens },
    edges = edges,
    requireCover = source
  ) {
    target-view {
      for t in tokens { SemanticEntity id = t label = t }
    }
    edge-chrome(e) { TokenRibbon(e) }   // bespoke slot fill
  }
}
```

Wrappers add domain props, slot chrome, evidence, and timeline — they do not
fork identity, span, or morph semantics.

## Runtime leaf kernel

These are the **only** justified runtime leaves for the top 25 components.
Everything else is `.flow` composition. Leaves return immutable semantic,
layout, analysis, or display-plan data; they do not return React nodes, mutate
Canvas state, or own the timeline clock. Visual backends consume leaf output
through the scene evaluator and display-list builder.

### Identity, text, and morph

| Leaf | Used by | Justification |
|---|---|---|
| `leaf.glyph-measure` | `core.glyph-run` | Font metrics, grapheme breaks, shaping |
| `leaf.span-interval` | `core.span-map` | Overlap index, coverage, projection |
| `leaf.correspondence-tween` | `core.semantic-morph` | Optional FLIP-style motion (skipped under reduced-motion) |

### Segment, compare, payload, queue

| Leaf | Used by | Justification |
|---|---|---|
| `core.segment-strip.layout` | `core.segment-strip` | Nested strip packing, clip, continuation |
| `core.compare.sync` | `core.compare` | Cross-pane selection/hover/inspect pairing |
| `core.structured-payload.virtual-tree` | `core.structured-payload` | Windowed tree rows, byte-range slice |
| `viz.queue.policy` | `viz.queue` | FIFO/priority/continuation/bounded simulation |

### Resources and events

| Leaf | Used by | Justification |
|---|---|---|
| `viz.partition-grid.analyze` | `viz.partition-grid`, `viz.ownership-map` | Modulo/range/hash assignment, coverage/overlap proofs |
| `viz.event-lane.dual-layout` | `viz.event-lane` | Dual causal/wall packing and counterpart links |
| `viz.slot-pool.virtualize` | `viz.slot-pool` | Optional high-cardinality slot window |
| `viz.barrier.derive-state` | `viz.barrier` | Optional quorum/timeout derivation (prefer explicit state) |

### Time and graphs

| Leaf | Used by | Justification |
|---|---|---|
| `viz.waterfall.nest-layout` | `viz.waterfall` | Nested interval packing, open/derived spans |
| `viz.phase-lifecycle.fsm` | `viz.phase-lifecycle` | Legal transition validation, escalation path |
| `viz.dual-clock.convert` | `viz.dual-clock` | Conversion DAG evaluation, fail-closed mapping |
| `viz.compound-graph.layout` | `viz.compound-graph`, `viz.metric-dag`, `viz.execution-graph` | Nested layout, routing, bundling, collision |
| `viz.execution-graph.fire` | `viz.execution-graph` | Readiness closure, enabled edges, fire frontier |

### Tries, partitions, metrics

| Leaf | Used by | Justification |
|---|---|---|
| `viz.prefix-trie.layout` | `viz.prefix-trie` | Trie build, LCP/split flags, tree positions |
| `viz.reduction-tree.layout` | `viz.reduction-tree` | Balanced fanout tier topology |
| `viz.metric-dag.propagate` | `viz.metric-dag` | Missing-value closure, cycle detection |
| `viz.sweep-line.compute` | `viz.sweep-line` | Active series, threshold crossings, steady-state window |

**Leaf budget:** ~20 narrow capabilities for 25 stdlib components. Several
components share `viz.compound-graph.layout` and `viz.partition-grid.analyze`.

Layout leaves emit **overridable layout-plan IR**. Authored geometry overrides
win; semantic ids and relations are never rewritten by layout.

Visual leaves, when justified by measured fidelity or cardinality requirements,
emit backend-neutral display-list fragments and semantic hit regions. They must
also define an SVG/HTML simplification and semantic-twin projection. WebGPU
acceleration may consume the same fragments in a later backend but cannot alter
component semantics or timing.

## Top 25 stdlib inventory

### Semantic continuity and spans

| Component | Class | Primary leaf | Composes |
|---|---|---|---|
| `core.semantic-entity` | flow-only | — | `group`, `text`, `inspect` |
| `core.semantic-relation` | flow-only | — | `semantic-entity`, `connector` |
| `core.semantic-morph` | hybrid | `leaf.correspondence-tween` | entities, relations, timeline |
| `core.glyph-run` | hybrid | `leaf.glyph-measure` | text clusters, span ids |
| `core.span-map` | hybrid | `leaf.span-interval` | `glyph-run`, `semantic-morph` |

### Segment, focus, compare, payload

| Component | Class | Primary leaf | Composes |
|---|---|---|---|
| `core.segment-strip` | hybrid | `core.segment-strip.layout` | entities, morph, `glyph-run`, `span-map` |
| `core.focus-context` | flow-only | — | `camera`, entity state, outline |
| `core.compare` | hybrid | `core.compare.sync` | `semantic-morph`, panes, optional `focus-context` |
| `core.structured-payload` | hybrid | `core.structured-payload.virtual-tree` | entities, optional `segment-strip` |

### Queues, resources, ownership, barriers, events

| Component | Class | Primary leaf | Composes |
|---|---|---|---|
| `viz.queue` | hybrid | `viz.queue.policy` | entities, morph, timeline |
| `viz.slot-pool` | flow-only* | optional `viz.slot-pool.virtualize` | `segment-strip`, `queue`, entities |
| `viz.resource-ledger` | flow-only | — | strip matrix, optional `event-lane` |
| `viz.ownership-map` | hybrid | `viz.partition-grid.analyze` | entities, morph, partition chrome |
| `viz.barrier` | flow-only* | optional `viz.barrier.derive-state` | entities, morph, connectors |

### Time, phases, clocks, graphs

| Component | Class | Primary leaf | Composes |
|---|---|---|---|
| `viz.waterfall` | hybrid | `viz.waterfall.nest-layout` | entities, relations, optional `event-lane` |
| `viz.phase-lifecycle` | hybrid | `viz.phase-lifecycle.fsm` | `barrier`, `queue`, entities |
| `viz.dual-clock` | hybrid | `viz.dual-clock.convert` | `event-lane` per rail |
| `viz.compound-graph` | hybrid | `viz.compound-graph.layout` | entities, relations, `focus-context` |
| `viz.execution-graph` | hybrid | `viz.execution-graph.fire` | `compound-graph`, optional waterfall |

### Tries, partitions, reduction, metrics, sweep

| Component | Class | Primary leaf | Composes |
|---|---|---|---|
| `viz.prefix-trie` | hybrid | `viz.prefix-trie.layout` | entities, morph, optional `compound-graph` |
| `viz.partition-grid` | hybrid | `viz.partition-grid.analyze` | entities, diagnostics chrome |
| `viz.reduction-tree` | hybrid | `viz.reduction-tree.layout` | `barrier`, value chips, morph |
| `viz.metric-dag` | hybrid | `viz.metric-dag.propagate` + graph layout | `compound-graph`, `focus-context` |
| `viz.sweep-line` | hybrid | `viz.sweep-line.compute` | chart primitives, optional `event-lane` |

\*Optional leaf only above cardinality thresholds.

## Composition layers

Dependency flows upward; bespoke wrappers sit at the top.

```text
Foundation: group, rect, text, connector, camera, timeline, inspect

Layer 1 — identity & spans:
  semantic-entity, semantic-relation, semantic-morph
  glyph-run, span-map

Layer 2 — structure & inspection:
  segment-strip, focus-context, compare, structured-payload

Layer 3 — dynamics:
  queue, slot-pool, resource-ledger, ownership-map, barrier, event-lane

Layer 4 — time & topology:
  waterfall, phase-lifecycle, dual-clock
  compound-graph, execution-graph

Layer 5 — analysis views:
  prefix-trie, partition-grid, reduction-tree, metric-dag, sweep-line

Layer 6 — domain symbols (catalog):
  TokenSpanMorph, PromptSegmentComposer, RequestLifecycleWaterfall, …
```

## Domain wrapper examples

Stdlib components are intentionally domain-neutral. AIPerf symbols from the
catalog wrap them with fixed vocabularies and evidence:

| Bespoke symbol | Wraps | Adds |
|---|---|---|
| `TokenSpanMorph` | `span-map`, `semantic-morph` | tokenizer edge table, ribbon chrome |
| `PromptSegmentComposer` | `segment-strip` | role tags, prefix reuse overlay |
| `ContinuationPriorityQueue` | `viz.queue` | continuation policy, request chips |
| `RequestLifecycleWaterfall` | `viz.waterfall` | fixed lifecycle lanes |
| `ObserverEventRail` | `viz.event-lane` | observer event kinds |
| `InflightSweepLine` | `viz.sweep-line` | steady-state fraction, ramp/drain mute |
| `CellPartitionGrid` | `viz.partition-grid` | two-level cell×worker tiling |
| `AsyncDataflowScene` | `viz.execution-graph` | graph fire log playback |

## Cross-cutting rules

1. **Identity before geometry** — layout plans set bounds; they do not mint new
   semantic ids.
2. **Fail closed** — illegal phase transitions, missing clock conversions, port
   type mismatches, partition gaps, and metric DAG cycles are compiler errors.
3. **Reduced motion** — path tweens, lane slides, and wipe animations degrade
   to cut, crossfade, or static emphasis; correspondence tables stay complete.
4. **Hashing** — content hash includes props, semantic model, leaf version, and
   pack-time frozen outputs (metrics tables, layout plans). Theme tokens and
   slot chrome do not change semantic hashes unless they alter entities or
   relations.
5. **Descriptor binding** — public component ids bind by capability/component
   id, not synthesized `core.${node.kind}` alone.
6. **Promote by evidence** — new runtime leaves require a conformance fixture
   that cannot pass with declarative composition alone.
7. **One clock** — component animation, camera cues, narration cues, and
   interaction restoration derive only from deterministic integer timeline
   time. A component never reads wall time.
8. **Pause-to-explore** — interaction pauses the authored lesson by default;
   inspect, pan, zoom, and compare state is temporary, serializable, and
   reversible. Resume continues from the exact paused beat.
9. **Semantic twin parity** — every meaningful Canvas entity and action has a
   corresponding persistent HTML semantic entity and keyboard operation.
10. **Fidelity without semantic loss** — degraded quality may reduce particles,
    blur, glow, and sampling density, but never removes entities, relations,
    captions, narration cues, focus, or evidence.

## Delivery priority

Aligns with catalog P0/P1:

### P0 — expressiveness proof (stdlib subset)

Implement layers 1–2 plus `viz.queue`, `viz.waterfall`, and leaves:
`glyph-measure`, `span-interval`, `segment-strip.layout`, `queue.policy`,
`waterfall.nest-layout`.

Prove with bespoke wrappers: `TokenSpanMorph`, `PromptSegmentComposer`,
`RequestLifecycleWaterfall`. Each proof must render through Canvas 2D, expose
its semantic HTML twin, degrade through the SVG/HTML fallback, and support
pause-to-explore/resume at a named beat.

### P1 — architecture flagship

Implement layers 3–5 and shared graph/partition/sweep leaves.

Prove with catalog scenes 4–6 wrappers.

### P2 — optional virtualization leaves

`slot-pool.virtualize`, Canvas mark paths, worker-offloaded layout above
documented cardinality thresholds — only after DOM/table fallbacks exist.

## Verification

Each stdlib component requires:

- prop/slot schema tests;
- default vs bespoke slot rendering;
- layout-plan override tests (geometry changes, semantics unchanged);
- accessibility outline and table fallback assertions;
- Canvas hit-region to semantic-twin focus/selection parity;
- reduced-motion and high-contrast snapshots;
- 3840×2160 reference-fidelity snapshots for the flagship wrappers;
- direct-seek versus continuous-playback equality;
- pause-to-explore and exact-beat resume assertions;
- cardinality fixtures at low/medium/high counts;
- normal vs packed IR parity.

Hybrid components additionally require leaf unit tests with golden layout-plan
or analysis outputs.

## Architectural conclusion

The top 25 shareable components are **not** 25 runtime capabilities. They are
a typed `.flow` standard library with ~20 narrow leaves and a clear wrapper path
for bespoke authoring. Domain richness lives in composition and slots; runtime
code stays algorithmic, deterministic, and replaceable.

# AIPerf Flow Delivery Roadmap

## Goal

Deliver the approved **AIPerf Flow** `.flow` language and runtime for interpreted
normal and packed Flow IR as independently reviewable vertical slices. Each
plan must leave working, testable software and preserve the canonical
architecture:

```text
source.flow
  → parser
  → typed AST
  → linker and type checker
  → Flow IR
  → optimizer and packer
  → static chunks
  → deterministic semantic scene graph and virtual clock
  ├─→ Canvas 2D cinematic visual renderer
  ├─→ semantic HTML accessibility twin
  └─→ simplified SVG/HTML fallback
```

The roadmap expands expressive coverage without replacing declarative authoring
with generated React or TypeScript scenes.

The primary product is a live, interactive, narrated explanation with the
visual fidelity of a professionally produced high-resolution tutorial.
High-resolution video is a quality metaphor, not the primary output. Static
hosting packages the live runtime; it does not reduce Flow to static diagrams
or prerecorded video.

The
[`2026-07-17-aiperf-flow-component-catalog.md`](../specs/2026-07-17-aiperf-flow-component-catalog.md)
prioritizes the reusable substrate, AIPerf domain symbols, capability packages,
and flagship scenes delivered across these plans.

## Implementation artifact map

The numbered roadmap plans are product phases. Current implementation work is
split into narrower reviewable artifacts:

- foundation pipeline:
  [`2026-07-17-aiperf-flow-foundation.md`](2026-07-17-aiperf-flow-foundation.md);
- P0 substrate umbrella:
  [`2026-07-17-aiperf-flow-p0-core-components.md`](2026-07-17-aiperf-flow-p0-core-components.md);
- component descriptor schema:
  [`2026-07-17-aiperf-flow-component-descriptors.md`](2026-07-17-aiperf-flow-component-descriptors.md);
- evaluated-scene and display-list schema promotion:
  [`2026-07-17-aiperf-flow-display-list.md`](2026-07-17-aiperf-flow-display-list.md);
- backend-neutral SemanticProjection unification:
  [`2026-07-17-aiperf-flow-semantic-projection.md`](2026-07-17-aiperf-flow-semantic-projection.md);
- live runtime and shell integration:
  [`2026-07-17-aiperf-flow-live-cinematic-runtime.md`](2026-07-17-aiperf-flow-live-cinematic-runtime.md);
- integer virtual clock and pause-to-explore:
  [`2026-07-17-aiperf-flow-virtual-clock.md`](2026-07-17-aiperf-flow-virtual-clock.md);
- hybrid capability evaluators:
  [`2026-07-17-aiperf-flow-hybrid-renderers.md`](2026-07-17-aiperf-flow-hybrid-renderers.md);
- symbol grammar:
  [`2026-07-17-aiperf-flow-symbol-grammar.md`](2026-07-17-aiperf-flow-symbol-grammar.md);
- Plan 3 language and module-system design:
  [`2026-07-17-aiperf-flow-language-module-system.md`](../specs/2026-07-17-aiperf-flow-language-module-system.md);
- P0 standard library:
  [`2026-07-17-aiperf-flow-stdlib.md`](2026-07-17-aiperf-flow-stdlib.md);
- flagship semantic IR:
  [`2026-07-17-aiperf-flow-flagship-ir.md`](2026-07-17-aiperf-flow-flagship-ir.md);
- CLI and deterministic static packaging:
  [`2026-07-17-aiperf-flow-cli.md`](2026-07-17-aiperf-flow-cli.md);
- temporary browser-shell prototype:
  [`2026-07-17-aiperf-flow-browser-preview.md`](2026-07-17-aiperf-flow-browser-preview.md).

Where artifacts overlap, the display-list plan owns shared schema contracts,
the semantic-projection plan owns the single runtime `SemanticProjection`
consumed by Canvas hit metadata, SemanticTwin, and SvgFallback, the
live-cinematic plan owns backend and mounted-app integration, the hybrid
renderer plan owns component evaluators, and the browser-preview plan owns only
temporary shell presentation.

## Planning principles

- The `.flow` source remains the only authored scene format.
- Content authors commit only `.flow` files and referenced assets; no plan may
  require or generate document-specific React, TypeScript, JavaScript, or CSS.
- The runtime interprets Flow IR; it does not evaluate arbitrary source.
- Normal and packed Flow IR have identical semantics and are both directly
  interpretable by the generic browser runtime.
- Layout, camera, timeline, style, and interaction are evaluated once into a
  backend-neutral scene and display list.
- Canvas 2D is the preferred cinematic visual backend. React/HTML owns shell
  chrome and the always-mounted semantic accessibility twin. SVG/HTML is the
  required simplified fallback.
- Future WebGPU acceleration may consume the same evaluated scene; it may not
  define semantics, timing, interaction, or accessibility behavior.
- Viewer interaction pauses narration and authored playback by default, permits
  temporary exploration, and resumes from the exact same beat with authored
  camera continuity.
- Capability descriptors drive schema, compiler validation, runtime binding,
  generated reference material, and agent-skill guidance.
- Every inferred value remains explicitly overridable as its capability lands.
- Each phase includes formatter, diagnostics, runtime, accessibility, and
  fallback behavior for the features it introduces.
- Existing `apps/explainers` decks remain operational until a later migration
  plan explicitly replaces them.
- Dependencies are installed through npm at implementation time so the lockfile
  captures current stable releases.

## Plan sequence

### Plan 1: Foundation vertical slice

**Artifact:** `2026-07-17-aiperf-flow-foundation.md`

Prove the complete architecture with:

- a new npm workspace app at `apps/aiperf-flow/` with packages under `packages/`;
- capability descriptors and versioned Flow IR;
- a block-language parser with source spans and structured diagnostics;
- linking, type checking, and AST-to-Flow-IR lowering;
- an interpreted semantic SVG/HTML foundation fallback with a closed registry;
- deterministic scene packing and a standalone static-site builder;
- CLI commands for `format`, `check`, `build`, `inspect`, and `capabilities`;
- one representative `.flow` fixture with explicit geometry, style,
  timeline, narration, interaction, accessibility, and fallback data.

The first grammar is deliberately narrow in vocabulary but complete through the
pipeline. No throwaway parser, alternate IR, or generated scene component is
allowed. Its React/SVG renderer is explicitly a foundation pipeline proof and
future fallback; it is not the final fidelity architecture.

### Plan 2: Live cinematic runtime substrate

**Artifact:** `2026-07-17-aiperf-flow-live-cinematic-runtime.md`

Establish the final rendering architecture before expanding the vocabulary:

- backend-neutral evaluated-scene and deterministic display-list IR;
- integer virtual time with direct-seek and continuous-playback equality;
- Canvas 2D renderer with resolution-independent camera, text, compositing,
  damage regions, hit regions, and quality profiles;
- React/HTML viewer shell and an always-mounted semantic HTML twin generated
  from the same evaluated scene;
- simplified SVG/HTML fallback and print/no-Canvas operation;
- bidirectional focus and selection synchronization across visual and semantic
  surfaces;
- pause-to-explore, temporary camera takeover, and exact-beat resume;
- 3840×2160 reference-fidelity verification plus desktop, tablet, and mobile
  reframing;
- frame-time, memory, asset-resolution, caption-safe-area, and degradation
  budgets;
- backend conformance fixtures proving semantic parity across Canvas, semantic
  HTML, and SVG/HTML fallback.

WebGPU is not required by this plan. The display contract must permit a future
accelerated backend without changing Flow IR or component semantics.

### Plan 3: Complete language and module system

Expand the foundation into the full authoring language:

- lossless comments and trivia;
- imports, namespaces, exports, aliases, integrity-pinned remote imports, and
  cycle diagnostics;
- constants, variables, typed parameters, expressions, objects, maps, unions,
  references, and parameterized declarative definitions;
- canonical formatter and migration framework;
- stable diagnostic catalog and machine-readable output;
- generated grammar, schema, and capability reference;
- language-server protocol support for completion, hover, go-to-definition,
  rename, and diagnostics.

### Plan 4: Composition, layout, and cardinality

Implement the production spatial model:

- coordinate spaces, units, stages, regions, layers, groups, portals, and HUDs;
- complete geometry, constraints, anchors, guides, clipping, masks, collision,
  hit regions, and exact overrides;
- manual, stack, flex, grid, matrix, layered, compound, radial, tree, force,
  sequence, timeline, state, and chart layout capabilities;
- a common layout-plan IR;
- ELK-backed compound graph layout behind the layout registry;
- edge routing, ports, labels, bundling, clustering, virtualization, and
  deterministic seeds;
- explicit policies and fixtures for 1, 10, 100, and 1,000 entities.

### Plan 5: Visual system and render capability catalog

Implement the broad 2.5D rendering vocabulary:

- typed cascading tokens and theme variants;
- typography, color spaces, gradients, patterns, fills, strokes, markers,
  blending, materials, lighting, shadows, glows, filters, and backdrops;
- text, rich text, code, equations, cards, panels, tables, shapes, paths,
  connectors, images, SVG, video, graph, sequence, state, metric, and particle
  primitives;
- state-dependent styles and exact geometry overrides;
- Canvas display-list implementations for the complete primitive vocabulary;
- semantic-twin and SVG/HTML-fallback projections for every meaningful
  primitive;
- optional accelerator eligibility only where measured Canvas budgets require
  it;
- Semantic Depthfield as the reference theme;
- primitive-level quality tiers, accessibility contracts, and fallbacks.

### Plan 6: Camera, motion, narrative, and playback

Build the cinematic guided experience:

- acts, chapters, scenes, beats, branches, continuity, and concept persistence;
- camera projections, shots, framing, tracking, paths, cuts, morphs, safe
  areas, and user takeover;
- nested timelines, keyframes, springs, easing, staggering, barriers, path
  motion, morphing, and finite loops;
- synchronized narration, subtitles, audio, camera, and visual tracks;
- deterministic virtual time, seek, pause, resume, replay, and restoration;
- interaction-triggered pause, temporary exploration, authored-camera rejoin,
  and exact-beat narration resume;
- explicit reduced-motion and no-motion alternatives;
- establish, teach, inspect, and transition viewer phases.

### Plan 7: Data, interaction, simulation, and responsive variants

Implement dynamic declarative behavior:

- typed records, tables, series, trees, graphs, and bounded streams;
- scales, domains, formatting, transforms, joins, aggregation, windows, bins,
  and derived values;
- repeated render trees and property bindings;
- deterministic pure expressions;
- typed events, actions, conditions, guards, state machines, cancellation,
  undo, and reset;
- drill-down, compare, branch, simulation, scrub, history, and persisted state;
- browser compilation from `.flow` to normal Flow IR for local authoring;
- hybrid sharing through compressed embedded IR for small documents and
  content-addressed IR plus assets for larger documents;
- viewport, container, capability, density, cardinality, and power variants;
- explicit precedence and deterministic degradation.

### Plan 8: Assets, accessibility, quality, and extension SDK

Complete production hardening:

- image, SVG, icon, font, audio, video, texture, and data asset pipelines;
- fingerprinting, deduplication, preload policy, licensing, and fallbacks;
- semantic-twin outlines, independent reading order, synchronized focus and
  selection, keyboard maps, transcripts, data tables, long descriptions, and
  nonvisual depth summaries;
- light, dark, high-contrast, reduced-transparency, no-depth, and print modes;
- frame-time, memory, layout, asset, and interaction budgets;
- typed extension SDK for primitives, layouts, effects, transforms, actions,
  asset loaders, and exporters;
- allowlisted, versioned, integrity-pinned capability packages with declared
  accessibility, fallback, compatibility, and cost contracts;
- a promotion path from generic primitives to reusable `.flow` symbols and
  then to domain capability packages;
- compatibility checks, fallback enforcement, cost models, and conformance
  suites.

### Plan 9: Authoring skills, flagship Flow document, and migration

Make AI-first authoring the supported workflow:

- one canonical generated language and capability guide;
- first-class Cursor and Claude Code flow-author skills;
- source-evidence collection, stable-ID planning, explicit production
  authoring, compilation, preview, and visual-verification workflow;
- one exhaustive flagship AIPerf architecture `.flow` file;
- `.flow`-only expressiveness fixtures for tokenization morphs, multi-segment
  prompts, and token flow through a model-serving system;
- desktop, mobile, high-contrast, reduced-motion, transcript, keyboard, and
  cardinality verification;
- 3840×2160 reference-fidelity, pause-to-explore, exact-beat resume, and
  cross-backend semantic parity verification;
- migration adapters and plans for existing `DeckDefinition` content;
- staged replacement of bespoke `MentalModel.tsx` scenes only after parity is
  demonstrated.

## Cross-plan completion gates

Every plan must:

- activate `/home/anthony/nvidia/projects/aiperf/ajc/rust/.venv` before commands;
- use TDD for each public contract;
- keep source files focused and package boundaries acyclic;
- emit stable human-readable and JSON diagnostics;
- preserve deterministic output and content hashes;
- prove direct-seek and continuous-playback equality for every timeline feature;
- verify semantic parity between normal IR, packed IR, embedded-URL shares, and
  content-addressed shares when those representations are implemented;
- test Canvas, semantic-twin, and SVG/HTML-fallback behavior for every new
  visual capability;
- measure declared frame-time, memory, and degradation budgets;
- update the design record when implementation reveals a changed contract;
- run `npm test`, `npm run build`, and relevant end-to-end checks;
- run `/usr/bin/python3 tools/check_docs_current.py` for documentation changes;
- avoid commits unless the user explicitly requests them.


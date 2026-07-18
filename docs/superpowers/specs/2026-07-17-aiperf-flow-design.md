# AIPerf Flow Design

## Product name

- Product: **AIPerf Flow**
- App: `apps/aiperf-flow`
- Source format: `.flow`
- Packages: `@aiperf/flow-*`
- CLI: `aiperf-flow`
- Intermediate representation: **Flow IR**

## Status

Approved product design for **AIPerf Flow**, a generic, deterministic,
interactive, narrated scene compiler and animation engine. Its primary output
is a live browser experience with the visual fidelity and narrative coherence
of a professionally produced high-resolution explainer, while every object
remains semantic, inspectable, and interactive. AIPerf content is the reference
implementation, not a product-specific boundary.

The companion
[`2026-07-17-aiperf-flow-component-catalog.md`](2026-07-17-aiperf-flow-component-catalog.md)
maps reusable visual capabilities and `.flow` symbols to the executable AIPerf
Rust architecture.

## North star

A Flow experience should feel like watching a meticulously produced 4K
explanation rendered live rather than playing a prerecorded video. “4K” is a
quality bar for composition, typography, motion, lighting, and detail—not a
requirement that the primary output be a video file. The same scene remains
responsive, searchable, keyboard-operable, screen-reader accessible, and
available for pause-to-explore interaction.

The canonical execution path is:

```text
.flow source
  → typed and versioned Flow IR
  → deterministic semantic scene graph
  → shared layout, camera, and timeline evaluation
  ├─→ Canvas 2D cinematic visual renderer
  ├─→ semantic HTML accessibility twin
  └─→ simplified SVG/HTML fallback renderer
```

React owns application chrome, navigation, transcript, captions, inspectors,
and the semantic accessibility surface. Canvas 2D is the preferred core visual
renderer because cinematic motion, dense routed paths, particles, compositing,
and high-resolution effects must not be constrained by DOM cardinality.
WebGPU may later accelerate rendering behind the same evaluated scene and draw
contracts; it is never the semantic source of truth. SVG and HTML remain
required fallback and inspection backends, not the long-term fidelity ceiling.

## Purpose

Build visually ambitious, cinematic, narrated, accessible, and interactive
Flow experiences from declarative source files. AI coding agents author the
source language; deterministic tools validate, pack, evaluate, and render it.
The browser contains no model integration, provider credentials, arbitrary
generated code, or visual editor.

The system favors expressive power over compactness. Explicit and duplicative
source is acceptable because an AI author can follow a large schema reliably.
Every inferred production decision can be overridden.

## Product principles

1. **The source file is the product contract.** Authored Flow documents are
   data, not React components.
2. **Power outranks brevity.** The language exposes the complete visual,
   narrative, responsive, interactive, and accessibility model.
3. **Determinism outranks magic.** Parsing, validation, layout, animation, and
   rendering produce reproducible results from source and registered
   capabilities.
4. **Inference is optional.** Defaults accelerate simple work, but strict mode
   can require explicit values and alternatives.
5. **Extensions make “anything” credible.** No fixed language can predict every
   future visual form, so typed extension capabilities are part of the core
   architecture.
6. **AI authors, tools verify.** Cursor and Claude Code skills generate and
   revise the language, but compiler diagnostics and visual verification decide
   whether the result is valid.
7. **One canonical runtime model.** The runtime interprets validated normal or
   packed Flow IR. Build tools do not generate TypeScript scene files.
8. **The live experience is the product.** Static hosting is a deployment
   mechanism for the interactive runtime, not a reduction to static diagrams or
   prerecorded video.
9. **Flow source is sufficient.** Content authors commit only `.flow` files and
   referenced assets. Per-document React, TypeScript, JavaScript, and CSS are
   neither authored nor generated. Product runtime code and trusted capability
   packages remain independently versioned infrastructure.
10. **One evaluation, multiple renderers.** Layout, camera, timeline, style,
    interaction, and semantic state are evaluated once. Visual, semantic, and
    fallback renderers consume the same evaluated scene without independently
    interpreting authored meaning.
11. **Accessibility is a coequal renderer.** Every meaningful visual entity,
    relation, state, and interaction has an always-available semantic HTML twin
    with authored reading order, keyboard behavior, transcript linkage, and
    textual or tabular alternatives.
12. **Interaction preserves the lesson.** Viewer interaction pauses the
    deterministic timeline by default. Exploration state is temporary and
    reversible; resume restores the authored camera and continues from the same
    beat without replaying or skipping narration.

## Non-goals

- An in-browser visual editor or split-pane authoring studio.
- Runtime AI, model-provider integration, or browser-held model credentials.
- True WebGL 3D as the default rendering model.
- Canvas pixels as the only representation of scene meaning.
- A video-first renderer or prerecorded video as the primary experience.
- Independent visual and accessibility implementations that can drift.
- AI-generated React or TypeScript scene components.
- Arbitrary JavaScript, CSS, or JSX embedded in Flow source.
- A hosted collaboration or publishing service.
- Compact syntax at the expense of production control.

## Authoring workflow

1. A developer supplies a rough outline, audience, source locations, and desired
   experience to an AI coding agent.
2. The agent invokes the flow-author skill and inspects executable source,
   manifests, and other authoritative evidence.
3. The skill creates or revises a `.flow` source file.
4. `aiperf-flow format` canonicalizes source formatting.
5. `aiperf-flow check` parses, links, type-checks, validates, and reports source
   diagnostics.
6. `aiperf-flow build <source> --out <directory>` packs a standalone static site.
7. `aiperf-flow preview <directory>` serves the built site for review; it does not
   edit source.
8. The skill checks desktop, mobile, high-contrast, reduced-motion, transcript,
   and keyboard behavior, then revises the source until verification passes.
9. The generated static directory is published without a server dependency.

## Language overview

The canonical source syntax is a purpose-built block language stored in
`.flow` files. It supports:

- comments and documentation;
- imports and namespaces;
- typed scalar, object, list, map, enum, union, reference, and expression
  values;
- named definitions and parameterized declarative symbols;
- stable IDs and cross-scene references;
- string, rich-text, code, and raw asset literals;
- semantic arrow syntax for graph relationships;
- explicit render-tree syntax for low-level production control;
- source spans for every token and lowered IR object;
- schema and capability version declarations.

Mermaid-like relationships are a convenience inside the larger language. They
do not constrain the language to graph diagrams.

### Illustrative structure

```text
flow "How a request flows" {
  language 1
  capabilities {
    require core.graph "^1"
    require core.camera "^1"
    require core.motion "^1"
  }

  use theme orbital

  chapter "Enter the engine" {
    scene system "The execution boundary" as request-path {
      story {
        summary "The CLI starts an isolated runtime."
        evidence "rust/cli/src/execute_mode.rs"
      }

      world {
        coordinate-space logical
        depth-axis semantic-layer
      }

      diagram {
        cli = service("CLI", process)
        runtime = service("Runtime", engine)
        sink = service("Worker sink", worker)
        cli -> runtime : "spawn --execute"
        runtime -> sink : "dispatch"
      }

      composition {
        place cli at (2u, 4u, 0u)
        constrain runtime right-of cli gap 3u
        constrain sink below runtime gap 2u
        route all-edges orthogonal
      }

      camera main {
        keyframe 0s frame all
        keyframe 2s frame cli, runtime zoom 1.4 depth 2u
        keyframe 5s track runtime -> sink
      }

      timeline primary {
        at 0s reveal cli duration 400ms
        at 800ms trace cli -> runtime duration 1200ms
        at 2.2s reveal runtime with rise
        at 4s trace runtime -> sink duration 900ms
      }

      responsive {
        variant compact when width < 720px {
          override composition flow vertical
        }
        variant dense when node-count > 24 {
          cluster by semantic-layer
          labels focus-only
          bundle edges
        }
      }

      interaction inspection {
        on select node run inspect(selected)
        on select group run dive(selected)
        on key "Escape" run reset-view
      }

      narration {
        transcript """
        The CLI resolves configuration and starts a fresh execution child.
        The runtime dispatches each turn to a worker-local sink.
        """
        synchronize words
      }

      accessibility {
        reading-order cli, runtime, sink
        summary "The CLI starts the runtime, which dispatches to a worker."
      }
    }
  }
}
```

This example is not the complete grammar. It demonstrates that high-level
diagram semantics and explicit production controls coexist in one language.

## Schema domains

### Document and module system

The document model includes:

- language and IR versions;
- capability requirements and compatibility ranges;
- imports with explicit aliases;
- namespaces and exported definitions;
- constants, variables, typed parameters, and immutable bindings;
- reusable themes, symbols, layouts, timelines, interactions, and data
  transforms;
- stable local and qualified IDs;
- metadata, ownership, licensing, and provenance;
- deterministic import resolution and cycle diagnostics.

Imports resolve at build time. Remote imports require integrity hashes and an
explicit allow policy. Builds record the complete dependency graph.

### Assets

Assets include:

- local and integrity-pinned remote images;
- SVG, icons, sprites, and symbol libraries;
- fonts and font fallback stacks;
- audio, narration recordings, ambient sound, and cues;
- video and poster frames;
- textures and data files;
- preload, lazy-load, decode, and cache policies;
- dimensions, MIME type, accessibility alternatives, and licensing metadata;
- quality-tier and unsupported-format fallbacks.

The packer fingerprints, deduplicates, optimizes, and copies assets into the
static output. Broken required assets fail the build.

### Narrative

Narrative objects include:

- acts, chapters, scenes, beats, branches, and endings;
- scene purpose, audience assumptions, learning objectives, and summaries;
- visible explanation, callouts, annotations, definitions, and conclusions;
- narration transcripts, recorded narration, subtitles, word timing, and
  pronunciation hints;
- citations, evidence, source links, and evidence drawers;
- previous/next continuity and concept persistence;
- estimated duration and explicit duration policy;
- linear, branching, and optional-detail paths.

Narrative order is independent from render-tree order and accessibility reading
order. All three are explicit and separately validated.

### World, coordinates, and composition

The world model includes:

- logical, screen, chart, geographic, temporal, and custom coordinate spaces;
- absolute, relative, viewport, font-relative, and semantic units;
- stages, viewports, regions, layers, groups, portals, overlays, and HUDs;
- 2D and 2.5D transforms, origins, perspective, depth, and z-order;
- grids, stacks, flex flows, tracks, alignment, distribution, and gaps;
- anchors, guides, constraints, intrinsic sizing, min/max sizing, and aspect
  ratios;
- clipping, masks, overflow, collision, avoidance, and hit regions;
- manual positions and exact geometry overrides;
- named composition variants.

Constraint solving must be deterministic. Ambiguous or unsatisfiable required
constraints are build errors; optional constraints report what was dropped.

### Render primitives

Built-in primitives include:

- groups, layers, regions, and portals;
- plain and rich text, code, equations, labels, badges, and captions;
- cards, panels, callouts, tables, lists, and legends;
- rectangles, rounded rectangles, circles, ellipses, polygons, lines, arcs,
  arbitrary paths, and compound shapes;
- connectors, routed edges, markers, ports, buses, bands, and edge labels;
- images, SVG, video, audio controls, and asset frames;
- graph nodes, compound nodes, sequence participants, messages, activations,
  state nodes, and transitions;
- bars, lines, areas, scatter marks, heatmaps, distributions, gauges, counters,
  traces, spans, and metric annotations;
- particles, trails, pulses, cursors, highlights, and focus rings;
- reusable symbols with typed parameters and slots.

Every primitive defines geometry, style, states, hit testing, accessibility,
performance cost, serialization, and fallback contracts.

### Expressive composition and domain capabilities

Bespoke visualizations use a three-level authoring model:

1. generic primitives compose text, glyphs, groups, paths, particles,
   connectors, masks, layouts, cameras, and timelines;
2. reusable `.flow` symbols compose those primitives behind typed parameters
   and slots without introducing runtime code;
3. trusted capability packages implement specialized algorithms, rendering, or
   high-cardinality behavior that declarative composition cannot express
   efficiently.

Authors begin with generic composition, extract recurring visual metaphors into
reusable `.flow` symbols, and promote only algorithmic or performance-sensitive
behavior into capabilities. Repeated concepts such as tokenization, prompt
segments, and token flow may therefore acquire semantic vocabulary without
becoming hard-coded language syntax.

Semantic IR retains relationships such as a text span mapping to a token ID or
a prompt containing ordered segments. Lowering must not prematurely erase
those relationships into unrelated geometry. Capabilities can use the semantic
data to produce glyph alignment, splitting, morph paths, staggering, camera
framing, inspection, reduced-motion behavior, and textual fallbacks. Authors
can override the resulting layout, geometry, style, and timing explicitly.

### Visual system

Visual definitions include:

- cascading tokens with typed values and scoped overrides;
- color spaces, palettes, gradients, patterns, and contrast pairs;
- font families, weights, widths, optical sizes, features, leading, tracking,
  and responsive type scales;
- fills, strokes, markers, opacity, blend modes, and compositing;
- materials, surfaces, lighting, shadows, glows, backdrops, and elevation;
- SVG filters and registered typed effects;
- visual states such as default, hover, focus, selected, active, muted,
  disabled, error, and compared;
- theme variants for light, dark, high contrast, print, and reduced effects.

Raw CSS is not embedded in source. Typed style properties cover built-ins;
extension capabilities add schema-defined style properties.

### Layout

Layout capabilities include:

- manual and constraint layout;
- stack, flex, grid, matrix, and table layout;
- layered, compound, radial, force, tree, cluster, and flow layouts;
- sequence, timeline, state, chart, geographic, and custom registered layouts;
- edge routing, port assignment, crossings, bundling, and label placement;
- collision handling and whitespace policies;
- deterministic seeds and iteration budgets;
- pinned nodes and partial manual overrides;
- precomputed, runtime, and hybrid layout policy;
- cardinality-specific strategies and fallback layouts.

Each layout emits a common layout-plan IR containing bounds, transforms, routes,
ports, labels, groups, and diagnostics. Renderers never call layout libraries
directly.

### Camera

Camera definitions include:

- orthographic and perspective-like 2.5D projections;
- viewport, position, target, zoom, rotation, depth, and focus plane;
- frame, fit, follow, track, orbit-like arc, dolly, pan, zoom, cut, and morph
  operations;
- named shots and reusable shot sequences;
- keyframes, paths, constraints, safe areas, and overscan;
- continuity rules between scenes;
- interaction takeover and resume behavior;
- breakpoint, reduced-motion, and low-power alternatives.

Camera behavior is represented in the same timeline system as visual motion and
narration synchronization.

### Motion and timelines

Motion includes:

- named timelines and nested timeline groups;
- absolute, relative, labeled, and cue-based timing;
- tracks, keyframes, holds, delays, loops, and finite repeats;
- easing curves, springs, inertia, and registered interpolators;
- enter, exit, reveal, hide, emphasize, compare, and morph behaviors;
- path motion, path drawing, transforms, style interpolation, and content
  transitions;
- staggering, synchronization, dependencies, and barriers;
- narration, subtitle, audio, camera, and interaction cues;
- seek, pause, resume, replay, and deterministic virtual time;
- explicit reduced-motion and no-motion timelines.

Production builds reject unbounded animation unless a capability explicitly
declares it safe and decorative.

### Interaction and state

Interaction includes:

- click, pointer, hover, focus, keyboard, wheel, drag, pinch, and custom typed
  events;
- selections, hotspots, tooltips, popovers, evidence panels, and inspectors;
- pan, zoom, drill-down, compare, branch, simulate, scrub, and reset actions;
- typed variables, conditions, guards, transitions, and finite state machines;
- action sequences, parallel actions, cancellation, undo, and reset;
- URL, history, and persisted-state policies;
- focus management and keyboard maps;
- narration and playback coordination;
- permission and capability checks.

The expression language is deterministic and side-effect free. Effects occur
only through registered actions with declared inputs and outputs.

### Data and expressions

Data capabilities include:

- typed literals, records, tables, series, trees, graphs, and streams;
- local data assets and build-time data imports;
- fields, scales, domains, ranges, formatting, and interpolation;
- filter, map, group, aggregate, window, join, sort, bin, and derive transforms;
- bindings from data to primitive properties and repeated render trees;
- deterministic pure expressions and registered transforms;
- validation, null policy, finite-number policy, and missing-data fallbacks;
- static, build-time, and bounded runtime evaluation.

Network data fetching is not enabled by default in standalone output. A
capability must explicitly provide and secure it.

### Responsiveness, density, and cardinality

Responsive policy includes:

- viewport and container queries;
- pointer, hover, color, contrast, motion, power, and feature queries;
- named variants with explicit precedence;
- composition, typography, camera, motion, interaction, and content overrides;
- node, edge, group, label, depth, and data cardinality queries;
- clustering, aggregation, sampling, pagination, virtualization, and edge
  bundling;
- label visibility, detail-on-demand, and semantic level-of-detail policies;
- explicit behavior at representative cardinalities;
- render, layout, memory, and interaction budgets;
- deterministic degradation and fallback rules.

The compiler can warn about unhandled cardinality ranges. Strict mode requires
declared policies for every range exercised by fixtures.

### Accessibility

Accessibility definitions include:

- semantic roles and landmarks;
- accessible names, descriptions, summaries, and long descriptions;
- reading order independent from visual order;
- focus order, focus traps, restoration, and skip links;
- keyboard commands and alternatives for pointer gestures;
- transcript, captions, narration controls, and pronunciation;
- high-contrast, reduced-motion, reduced-transparency, and no-depth variants;
- data table and textual equivalents for charts and complex diagrams;
- nonvisual descriptions for depth, spatial movement, and sonification;
- color-independent encodings and contrast contracts.

Required accessibility fields may be generated by authoring skills, but they
are validated like all other source.

### Runtime and quality policy

Runtime policy includes:

- quality tiers and feature-detection gates;
- preload, lazy-load, suspend, and disposal rules;
- chapter and scene chunk boundaries;
- memory, frame-time, layout-time, and asset budgets;
- static, dynamic, and hybrid layout choices;
- worker eligibility and deterministic worker messages;
- graceful degradation and primitive-specific fallbacks;
- telemetry hooks that are disabled by default;
- error-boundary and transcript fallback behavior.

## Extension model

The fixed schema is intentionally extensible. An extension package registers
one or more capabilities:

- primitive;
- layout strategy;
- effect or material;
- data transform;
- expression function;
- interaction action;
- asset loader;
- exporter.

Each capability provides:

- globally unique name and semantic version;
- schema fragment and generated language descriptors;
- AST-to-Flow-IR lowering contract when needed;
- runtime implementation;
- accessibility and fallback contracts;
- serialization and deterministic hashing behavior;
- compatibility range;
- performance-cost model;
- tests and reference examples.

Extensions are ordinary trusted TypeScript packages selected at build time.
`.flow` files cannot install packages or execute arbitrary code. The packer
records required capabilities and fails if they are unavailable or
incompatible.

## Compiler architecture

### Parsing

The parser produces a lossless typed AST with comments, trivia, stable node
identities, and exact source spans. Error recovery reports multiple useful
diagnostics in one run without accepting an invalid document.

### Linking and type checking

The linker resolves imports, namespaces, references, symbols, assets, data
fields, timeline labels, interaction targets, and capability names. The type
checker validates values, expressions, overrides, variants, and extension
schema fragments.

### Semantic validation

Validation checks:

- graph and reference integrity;
- duplicate and unstable identities;
- impossible composition constraints;
- invalid timing and synchronization;
- unreachable narrative or interaction states;
- missing responsive and cardinality policies;
- absent accessibility alternatives;
- unsupported or incompatible capabilities;
- asset integrity;
- non-finite values;
- budget violations;
- evidence and narration policy.

Diagnostics include severity, stable code, source range, explanation, and
actionable repair text. Strict profiles promote selected warnings to errors.

### Lowering to Flow IR

Lowering resolves language sugar and reusable definitions into explicit,
versioned IR. Flow IR contains no unresolved imports, implicit inheritance,
or parser-specific syntax. It preserves source-map links for every object.

The IR is stable enough for runtime interpretation and explicit migrations, but
it is not the human authoring format.

### Optimization and packing

The packer:

- resolves and fingerprints dependencies;
- deduplicates definitions and assets;
- precomputes eligible layouts, routes, text measurements, and timelines;
- retains declarative runtime programs for dynamic behavior;
- partitions by act, chapter, and scene;
- tree-shakes unused capabilities and definitions;
- computes deterministic content hashes;
- emits source maps, manifests, transcripts, and fallback documents;
- copies the generic runtime and required extension bundles;
- verifies that output works from a static origin.

The packed output remains data-driven. It does not contain generated
TypeScript scene source.

### Normal and packed Flow IR

Flow IR has normal and packed representations with identical semantics:

- normal IR is versioned, inspectable, and directly interpretable in the
  browser;
- packed IR may deduplicate values, precompute layouts and timelines, partition
  chunks, and optimize assets;
- packing cannot alter observable scene, interaction, accessibility, or
  fallback behavior;
- source maps preserve `.flow` locations through both representations.

The same interpreter consumes either representation. A browser compiler may
compile `.flow` into normal IR for local authoring and sharing, while production
builds use the CLI optimizer and packer. Neither path generates per-scene React
or TypeScript.

### Content-addressed sharing

The viewer supports two share forms:

- small documents embed compressed normal or packed IR in the URL fragment;
- larger documents use a URL containing a content hash that resolves immutable
  IR and fingerprinted assets from a static host or content-addressed store.

Both forms open in the same generic viewer. Capability manifests record exact
versions, integrity hashes, compatibility ranges, and fallbacks. Embedded
documents may use only capabilities trusted by the host; shared content cannot
load arbitrary executable code from a URL. A share can be inspected and forked,
and can recover canonical `.flow` when source is included.

## Runtime architecture

The runtime consumes only validated normal or packed IR and capability
manifests.

Core runtime units are:

- **loader:** resolves manifests and lazy chunks;
- **capability registry:** binds IR capability names to implementations;
- **scene store:** owns immutable definitions and serializable viewer state;
- **scene evaluator:** resolves semantic state, style, layout plans, camera,
  timelines, and interaction state into one backend-neutral evaluated scene;
- **data evaluator:** executes bounded pure expressions and transforms;
- **layout coordinator:** loads precomputed plans or runs eligible dynamic
  planners;
- **display-list builder:** converts the evaluated scene into deterministic,
  backend-neutral draw commands, hit regions, and damage regions;
- **Canvas visual renderer:** draws the cinematic scene, including compositing,
  depth planes, routed light paths, particles, text, and high-cardinality marks;
- **semantic twin renderer:** maps the evaluated semantic scene to persistent
  HTML landmarks, entities, relations, controls, descriptions, tables, and
  transcript links;
- **fallback renderer:** maps the same evaluated scene to simplified SVG and
  HTML without removing meaning or navigation;
- **timeline engine:** advances deterministic virtual time and coordinates
  camera, motion, audio, narration, and subtitles;
- **interaction engine:** processes typed events and state machines;
- **camera controller:** applies guided shots, pauses for temporary user
  exploration, snapshots takeover state, and restores authored state on resume;
- **accessibility coordinator:** keeps visual hit targets, semantic focus,
  selection, transcript position, and inspector state synchronized;
- **budget manager:** applies quality tiers and declared degradation;
- **error isolation:** falls back per capability or scene without blanking the
  complete Flow document.

The preferred visual implementation is Canvas 2D driven by a deterministic
display list. React and HTML provide the viewer shell and semantic twin. SVG and
HTML provide simplified fallback rendering, debugging, print, and no-Canvas
operation. CSS transforms may position shell overlays but do not own canonical
camera or scene state.

Canvas is not an accessibility boundary. The semantic twin is mounted whenever
the visual renderer is mounted, mirrors focus and selection bidirectionally,
and remains usable when the visual surface is hidden or fails. A capability is
not conformant unless it emits both visual draw data and semantic output, or
declares a validated fallback that does.

WebGPU is an optional future visual backend for effects or cardinalities that
exceed Canvas budgets. It consumes the same evaluated scene and semantic twin
contracts. It cannot introduce authored semantics, interaction behavior, or
timing that other backends cannot represent.

### Deterministic frame and interaction model

All visible and audible state is a pure function of validated IR, immutable
assets, capability versions, viewport/profile inputs, interaction log, and
integer timeline time. Wall-clock deltas advance the timeline but never become
scene state. Seeking to a time produces the same evaluated scene as continuous
playback at that time.

The default interaction lifecycle is:

1. an authored lesson advances under the deterministic clock;
2. pointer, keyboard, or assistive-technology exploration pauses at the current
   beat and records authored camera, focus, selection, and narration state;
3. the user may pan, zoom, select, inspect, compare, or traverse semantic
   relationships without advancing narration;
4. resume restores or smoothly rejoins the authored camera according to the
   scene policy and continues from the exact paused beat.

Scenes may author explicit branches later, but continuing narration underneath
unbounded exploration is not the default.

### Fidelity and performance profiles

The runtime is resolution-independent. A 3840×2160 reference viewport is a
required visual-verification profile because it exposes weak typography,
composition, effects, and asset handling; it does not turn video export into
the primary output. The same source must define deterministic reframing for
desktop, tablet, and mobile containers.

The default desktop quality profile targets 60 evaluated frames per second on
the documented reference device. A declared degraded profile may target 30
frames per second by reducing decorative particles, blur, shadow quality, and
sampling density, but it may not remove semantic entities, narration cues,
captions, focus, or interaction. Frame-time, memory, text sharpness, asset
resolution, and visual damage budgets are descriptor-driven and measurable.

## Visual direction

The default theme is **Semantic Depthfield**, a precise technical instrument:

- translucent semantic planes encode scope and abstraction;
- routed light paths encode flow and causality;
- stable entities persist and morph across scenes;
- typography remains quiet, technical, and highly legible;
- continuous semantic zoom replaces unrelated slide cuts;
- camera depth always communicates meaning;
- motion teaches order, ownership, state, or causality rather than decorating.

Themes can replace the complete visual system through typed tokens, materials,
type, lighting, edge language, and motion character. They cannot remove
required semantics or accessibility.

The guided viewer follows establish, teach, inspect, and transition phases.
Users may pause at any safe beat and at declared exploration points, pan, zoom,
select, inspect evidence, drill into groups, compare states, and then resume
from the same beat. The authored camera remains recoverable throughout
exploration. Viewer chrome stays visually subordinate to the scene and may
collapse, but navigation, captions, transcript, and accessibility controls
remain available.

## AI authoring skills

Cursor and Claude Code receive first-class platform-specific skills backed by
one canonical authoring guide and generated language reference.

The skill workflow is:

1. establish audience, objective, scope, and desired runtime;
2. inspect source-of-truth code and manifests;
3. create a claim and evidence inventory;
4. create a concept graph with stable IDs;
5. design acts, chapters, scenes, beats, and continuity;
6. select built-in and extension capabilities;
7. author complete semantic, render, responsive, interaction, narration, and
   accessibility definitions;
8. run formatter and compiler diagnostics;
9. build and inspect representative output modes;
10. revise source until all required checks pass.

The skills must:

- prefer explicit schema fields over undocumented convention;
- preserve stable IDs during revisions;
- avoid architecture claims without evidence;
- author representative cardinality fixtures;
- provide reduced-motion, compact, high-contrast, transcript, and fallback
  behavior;
- use extensions only when built-ins cannot express the requirement;
- never generate runtime TypeScript as a shortcut;
- keep the `.flow` source comprehensible and formatter-clean.

The language schema, CLI help, generated reference, skill guidance, and
exemplars derive from the same capability descriptors to prevent drift.

## CLI

The initial command surface is:

- `aiperf-flow format <sources...> [--check]`
- `aiperf-flow check <sources...> [--profile <name>]`
- `aiperf-flow build <source> --out <directory> [--profile <name>]`
- `aiperf-flow preview <directory>`
- `aiperf-flow inspect <source> [--ast|--ir|--layout|--manifest]`
- `aiperf-flow migrate <source> --to <language-version>`
- `aiperf-flow capabilities [--json]`
- `aiperf-flow schema [--capability <name>] [--json]`

Commands produce human-readable diagnostics and stable machine-readable JSON
diagnostics for AI skills and CI.

## Failure behavior

- Syntax, linking, type, required accessibility, incompatible capability, and
  impossible required-constraint failures stop the build.
- Density, duration, evidence quality, performance, and optional-constraint
  concerns are warnings unless promoted by the active profile.
- Runtime capability failures activate the capability's declared fallback.
- Scene failures activate a readable 2D scene summary and transcript.
- Unsupported visual effects fall back according to authored quality policy.
- Missing required fallbacks are build failures.
- No runtime failure may remove access to navigation, transcript, or evidence.

## Verification

### Language and compiler

- golden parser and formatter fixtures for every syntax construct;
- malformed-source fixtures for every diagnostic code;
- AST and IR round-trip tests;
- import, namespace, reference, and capability-resolution tests;
- schema compatibility and migration tests;
- property-based tests for parser recovery, formatter idempotence, and reference
  resolution;
- deterministic build and content-hash tests.

### Layout and cardinality

- invariant tests for every built-in layout strategy;
- fixtures at 1, 10, 100, and 1,000 entities;
- edge routing, collision, label, and bounds checks;
- responsive variant precedence tests;
- budget and degradation tests;
- precomputed and runtime layout parity tests.

### Runtime

- virtual-clock tests for timelines, camera, narration, pause, seek, replay, and
  resume;
- direct-seek versus continuous-playback equality at representative beats;
- pause-to-explore tests that preserve the paused beat, synchronize semantic
  focus and visual selection, and restore the authored camera on resume;
- evaluated-scene and display-list determinism tests independent of renderer;
- Canvas, semantic-twin, and SVG/HTML-fallback conformance tests against the
  same semantic fixture;
- interaction state-machine and keyboard tests;
- capability loading, compatibility, and fallback tests;
- lazy loading and disposal tests;
- deterministic state serialization tests;
- scene and capability error-isolation tests.

### Accessibility and visuals

- semantic outline, labels, reading order, focus order, and transcript tests;
- bidirectional focus and selection synchronization between Canvas hit regions
  and the semantic HTML twin;
- keyboard-only end-to-end journeys;
- automated accessibility scans;
- visual snapshots for desktop, mobile, light, dark, high contrast, reduced
  motion, reduced transparency, and no-depth modes, including a 3840×2160
  reference-fidelity profile;
- representative cardinality visual snapshots;
- frame-time and memory budget measurements for reference and degraded quality
  profiles;
- text sharpness, asset-resolution, caption-safe-area, and color-contrast
  assertions at the reference viewport;
- static-host and offline smoke tests.

### Reference content

The first reference Flow document is a flagship AIPerf architecture story that
exercises every schema domain. Existing AIPerf decks then migrate to
`.flow` sources without custom React scene components. Their architecture
claims remain grounded in executable Rust and manifests.

### Expressiveness proofs

The capability catalog is not considered sufficient for bespoke technical
visualization until `.flow`-only fixtures demonstrate:

1. a tokenization morph in which characters and words split into token spans,
   transform into token IDs, and remain traceable backward;
2. prompt composition in which system, user, tool, image, generated, reused,
   and truncated segments retain visible boundaries and lengths;
3. token IDs moving through tokenizer, scheduler, queues, model stages, KV
   cache, and output decoding;
4. a combined cinematic scene coordinating camera, narration, interaction,
   inspection, responsive layout, and semantic continuity;
5. equivalent behavior from normal IR, packed IR, embedded-URL shares, and
   content-addressed shares;
6. semantic parity in reduced-motion, high-contrast, keyboard, transcript, and
   missing-capability modes;
7. equivalent semantic entities, relations, focus, selection, and transcript
   position across Canvas, semantic HTML twin, and SVG/HTML fallback;
8. pause-to-explore and resume from the same beat without narration drift,
   camera discontinuity, or lost interaction state;
9. professional composition, typography, motion, and effects at the
   3840×2160 reference profile while remaining responsive at smaller
   containers.

These fixtures must contain no document-specific React, TypeScript, JavaScript,
or CSS. Visual snapshots, semantic assertions, and deterministic rebuild checks
must cover geometry, timelines, hashes, and packed output.

## Delivery boundaries

The first implementation must establish the complete extensibility and
execution architecture without implementing every imaginable capability at
once. It must include:

- the block language and capability descriptor mechanism;
- parser, formatter, linker, type checker, Flow IR, packer, and CLI;
- a backend-neutral scene evaluator and deterministic display-list contract;
- the interpreted packed runtime with Canvas visual output, an always-mounted
  semantic HTML twin, and a simplified SVG/HTML fallback;
- representative built-in primitives, layouts, camera, motion, interactions,
  data, responsiveness, accessibility, and fallbacks;
- pause-to-explore and exact-beat resume behavior;
- the Semantic Depthfield default theme;
- Cursor and Claude Code authoring skills;
- one exhaustive flagship `.flow` reference that demonstrates extension
  points and production controls;
- deterministic verification and static deployment.

Subsequent capability packages expand the vocabulary without changing the
authoring or runtime architecture.


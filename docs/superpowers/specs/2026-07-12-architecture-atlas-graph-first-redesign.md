# Architecture Atlas Graph-First Redesign

## Status

Approved on 2026-07-12.

This design replaces the current card- and rail-led Architecture Atlas experience. The existing typed content and source-integrity foundation remain useful, but the product presentation and interaction model are rebuilt around true interactive architecture graphs.

## Goal

Create a static React application that explains the true AIPerf Rust architecture from the Python product boundary through `aiperf-runner`, every major Rust subsystem, transport, observation path, and report boundary.

The graph is the product. Text supports the graph instead of competing with it.

The application must:

- tell the full lifecycle from Python Config v2 loading through Rust execution and results;
- show how control, request data, tokens, telemetry, and reports flow through the system;
- support tiered drill-down from systems to source-grounded Rust symbols;
- represent native HTTP, native gRPC, online mock, Dynamo offline, and planned Dynamo online as first-class execution flavors;
- allow nodes and visual seam routing to be rearranged, persisted, and shared;
- preserve the three audience levels through progressive topology disclosure;
- keep migration and parity information subordinate to the architecture.

## Product Information Architecture

The opening route renders the complete Rust architecture and its Python boundary. It does not begin with cards, ownership bands, parity ledgers, or prose-heavy guided content.

Primary scenes follow the actual implementation:

1. Runtime composition
2. Runner protocol and registries
3. Scheduling and phase lifecycle
4. Dataset and segment pipeline
5. Endpoint bindings and HTTP/gRPC transports
6. Graph-IR execution
7. Metrics and telemetry
8. Accuracy and evaluator hosting
9. Crate dependency topology

The opening scene supports a playable end-to-end journey:

```text
Python config load
→ Config-v2 resolution
→ authored request projection
→ aiperf-runner spawn
→ strict JSONL validation
→ frozen RunnerApplication
→ workload preparation
→ scheduling or Graph-IR
→ dataset materialization
→ endpoint binding
→ HTTP, gRPC, or Dynamo dispatch
→ observer callbacks
→ metrics and reporting
→ result returned to Python
```

Python is shown only where the architecture crosses the Python/Rust boundary. Legacy implementation details, migration state, and parity gaps appear as collapsible footnotes attached to relevant nodes or seams. There is no parity-led route or parity-led visual hierarchy.

## Audience Topology

Audience selection changes graph topology and depth rather than merely changing labels:

- **Executive / cross-org:** major systems, responsibilities, and the product execution story.
- **Developer:** Rust subsystems, runtime branches, protocols, and primary extension seams.
- **Core maintainer:** crates, modules, traits, concrete symbols, exact contracts, source evidence, and byte-sensitive logic.

All audiences see the same canonical architecture. Higher levels progressively reveal more of that architecture. Audience selection determines the default topology and maximum automatic expansion depth; deliberate drill-down remains available where appropriate.

## Tiered Drill-Down

Every scene uses the same hierarchy:

- **Tier 0:** complete product journey
- **Tier 1:** Rust systems and external boundaries
- **Tier 2:** internal subsystems, registries, traits, and execution stages
- **Tier 3:** crates, modules, concrete symbols, source evidence, and exact contracts

Double-clicking or activating a node's expand control replaces that node with its children in place. Neighboring nodes reflow around the expanded subgraph while preserving user-authored positions outside the affected region.

The interaction model includes:

- breadcrumbs for the active expansion path;
- collapse controls at every expanded tier;
- edge selection for protocol, payload, lifecycle, and seam details;
- upstream and downstream path highlighting;
- search that can reveal and expand hidden descendants;
- a “follow the pulse” action that traces one request through every boundary;
- keyboard-equivalent controls for all pointer interactions.

## Execution Flavors

The flavor selector contains:

- Native HTTP
- Native gRPC
- Online mock
- Dynamo offline
- Dynamo online — planned

Flavor selection morphs one shared graph. Shared systems remain stable while transport-, clock-, and backend-specific branches change. Users can select two flavors for an overlay comparison: shared paths render once and diverging paths fork only where behavior differs.

### Dynamo Offline

Dynamo offline receives complete source-grounded drill-down:

```text
Python Config v2
→ feature-bearing aiperf-runner
→ offline execution factory
→ shared workload and scheduler
→ SimClock
→ Dynamo SteppableReplay
→ engine and router topology
→ observer and metrics
→ byte-equality report gate
```

The scene exposes the built single, aggregate, and disaggregated topologies; virtual-time event pumping; routing; cancellation; raw-token bypass; worker and KV artifacts; applicable adaptive controls; and Cargo feature gates.

### Dynamo Online

Dynamo online is a first-class selectable flavor, not a hidden roadmap note. Its planned-only nodes and seams remain fully interactive but use an unmistakable planned visual treatment.

The current `aiperf dynosim run --replay-mode online` surface delegates to Dynamo's canonical online replay path. It is not represented as an already-built `aiperf-runner` backend pair. Implemented shared AIPerf/Dynamo pieces use source evidence; future runner integration requires explicit design evidence and planned status. The atlas must not infer unapproved internals or present roadmap architecture as shipped code.

## Graph Interaction

The application uses route-specific graph scenes backed by one shared graph engine and one canonical content catalog.

Nodes:

- are draggable;
- expose named seam ports;
- can be selected, expanded, collapsed, searched, and focused;
- show compact identity and status directly;
- open contextual details without covering the graph unnecessarily.

Edges:

- are selectable;
- animate according to flow type;
- expose protocol and payload contracts;
- support draggable visual waypoints;
- retain fixed semantic endpoints unless explicit topology-experiment mode is enabled.

Topology experiments never overwrite or masquerade as the canonical architecture. Reset restores the source-grounded scene.

Pan, zoom, minimap, fit, isolate, reset, and share controls remain available without dominating the canvas.

## Flow Semantics

Animated paths use distinct channels:

- control and lifecycle;
- requests and payloads;
- output and reasoning tokens;
- telemetry and observations;
- reports and results.

Play, pause, and scrub controls drive a deterministic narrative timeline. The selected execution flavor determines the active branch. Selecting a pulse or edge reveals the owning subsystem, relevant seam, and current payload.

Motion must explain causality. Ambient or decorative animation is excluded.

## Visual Direction

The visual language is “Flight Deck”: dense and operational, but disciplined rather than ornamental.

### Palette

- Canvas graphite: `#070B10`
- Raised steel: `#111A22`
- Active/shared Rust path: `#76B900`
- Request/data flow: `#45C7F4`
- Control/lifecycle flow: `#F5B942`
- Simulated/Dynamo flow: `#A78BFA`
- Planned-only architecture: `#FF7A7A` with dashed geometry

NVIDIA green identifies active or shared Rust architecture. It is not used as generic decoration.

### Typography

- Saira Condensed for system headings
- Manrope for explanatory text
- IBM Plex Mono for crates, modules, symbols, protocols, and metrics

Fonts ship with the static artifact.

### Layout

Nearly the entire viewport belongs to the graph. The shell contains:

- one compact command bar;
- an optional collapsed scene rail;
- the graph canvas;
- a contextual evidence drawer;
- compact timeline controls when a narrative is active.

The design removes large introductory headings, stacked card grids, persistent prose rails, and secondary inventories from the primary visual plane.

The signature interaction is “follow the pulse”: the graph expands around one request as it travels from Python configuration to Rust execution and back.

## Content Model

The source catalog is hierarchical and Zod-validated.

Each node records:

- stable identifier;
- tier and parent;
- audience visibility;
- applicable execution flavors;
- ownership and implementation status;
- labels and descriptions by audience;
- source or design evidence;
- child identifiers;
- named seam ports;
- searchable crate, module, trait, and symbol metadata.

Each edge records:

- stable identifier;
- semantic source and target ports;
- flow type;
- protocol or payload;
- applicable execution flavors;
- lifecycle phase;
- implementation status;
- source or design evidence.

Scene definitions select and arrange catalog entities without duplicating architecture facts.

Implemented claims require code evidence. Planned claims require explicit design evidence. Legacy and parity notes remain optional subordinate annotations.

## State and Sharing

Canonical layouts are computed with ELK in a worker. User movement produces layout overrides:

- node coordinates;
- edge waypoints;
- expanded hierarchy;
- selected flavors;
- comparison state;
- audience;
- focused entity;
- timeline position when useful.

State is Zod-validated, compressed into URL-safe data, and mirrored to local storage. A shared URL is authoritative when present. Invalid, stale, or incompatible state falls back to the canonical scene and explains that recovery non-disruptively.

Semantic graph content is never accepted from the URL.

## Accessibility

The graph has a complete keyboard and screen-reader model:

- every visible node and edge is reachable;
- graph relationships have directed accessible descriptions;
- expand, collapse, move, isolate, and inspect operations have keyboard equivalents;
- focus is restored after drawers and collapses;
- reduced motion disables particles and animated transitions without removing flow meaning;
- color is never the only status or flow indicator;
- a synchronized structured outline is available as an accessibility representation, not as the primary visual experience.

## Failure Handling

- ELK worker failure switches to a deterministic grouped fallback.
- Invalid shared state restores the canonical scene.
- Missing optional source evidence is visible as a content-integrity failure during development and CI.
- Unsupported flavor combinations are disabled with an explanation.
- Planned Dynamo-online entities cannot silently acquire built styling.
- Animation failure never blocks graph navigation or content access.

## Testing and Quality Gates

Unit tests cover:

- hierarchical graph derivation;
- tier expansion and collapse;
- audience topology;
- flavor selection and overlays;
- edge contracts;
- canonical and manual layout merging;
- URL serialization and recovery;
- planned-versus-built status enforcement.

Component and accessibility tests cover:

- keyboard graph operations;
- evidence drawers;
- path highlighting;
- focus restoration;
- reduced motion;
- structured accessibility representation.

Playwright covers:

- the complete follow-the-pulse journey;
- node dragging and persistence;
- edge waypoint movement;
- tiered drill-down;
- audience changes;
- Dynamo-offline traversal;
- Dynamo-online planned traversal;
- flavor overlay comparison;
- shareable layout URLs;
- desktop and mobile layouts;
- Axe audits for every scene and expanded tier.

Visual regression screenshots protect the graph-first composition and status language.

Content integrity validates source paths, symbols, graph references, Cargo dependency claims, execution flavor coverage, and evidence requirements.

CI produces one static deployment artifact.

## Acceptance Criteria

The redesign is complete when:

1. The default screen is a true interactive Rust architecture graph.
2. A user can play the complete Python-to-Rust-to-result story.
3. Every major Rust subsystem supports tiered source-grounded drill-down.
4. Native HTTP, native gRPC, online mock, Dynamo offline, and planned Dynamo online are first-class graph flavors.
5. Two flavors can be overlaid with shared paths rendered once.
6. Nodes and edge waypoints are movable, persistent, and shareable.
7. Audience selection changes topology and technical depth.
8. Parity and legacy information remain subordinate footnotes.
9. Planned architecture cannot be mistaken for built code.
10. Accessibility, source integrity, interaction, visual regression, and production-build gates pass in CI.

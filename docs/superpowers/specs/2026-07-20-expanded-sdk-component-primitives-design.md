<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Expanded SDK Component Primitives

## Summary

Expand the explainer Flow SDK in two ordered phases:

1. Add an exhaustive generic UI and content primitive catalog.
2. Add an exhaustive systems-diagram primitive catalog.

The expansion is capability-led. A small set of reusable semantic Scene IR and
renderer foundations lands before the component factories that need them.
Components remain deterministic compile-time factories: they validate authored
props and slots, produce native semantic Scene IR, expose semantic ports and
timeline actions, and carry source-mapped SDK provenance. The authoritative
runtime contract is
`2026-07-20-native-semantic-scene-ir-design.md`; component factories do not
expand renderer-owned chrome into visual-only primitive children.

“Exhaustive” means the complete catalog listed in this design. Data
visualization is limited to lightweight indicators: progress, meter, gauge,
sparkline, rating, and semaphore. Full charting primitives are outside scope.
AIPerf-specific components are also outside scope.

## Goals

- Make common explainer visuals authorable without bespoke rect, text, path, or
  coordinate clusters.
- Cover generic presentation needs before systems-diagram needs.
- Preserve existing `.flow` decks and SDK component behavior.
- Keep new rendering capabilities reusable and independent of individual SDK
  components.
- Give every component stable generated IDs, semantic ports, timeline actions,
  accessibility metadata, source-mapped diagnostics, and SDK provenance.
- Keep `SceneRenderer.tsx` from growing into a component-specific switchboard.

## Non-goals

- Full bar, line, area, pie, scatter, or heatmap charting.
- Interactive application widgets or application state management.
- AIPerf-domain components.
- Replacing existing canonical components when a new component can compose or
  specialize them.
- Introducing a third-party icon or chart dependency.

## Architecture

### Expansion pipeline

The existing authoring pipeline remains:

1. The parser produces an SDK component invocation.
2. The registry resolves its canonical component descriptor.
3. Descriptor and factory validation produce source-mapped diagnostics.
4. The factory emits a deterministic `SceneFragment`.
5. Semantic references resolve after all component ports are known.
6. SDK provenance is stripped before package serialization.
7. The native capability registry resolves layout and `SceneRenderer` renders
   the resulting semantic Scene IR.

No component factory may depend on React, the DOM, network access, wall-clock
time, or mutable global state.

### Renderer foundations

Add only reusable foundations required by multiple catalog entries while
preserving the existing strict Scene IR union:

- Named icons lower to existing `core.path` connector nodes using an
  in-repository SVG path registry.
- Images lower to existing `ComponentNodeIr` with `core.image`; the renderer
  supports package-resolvable sources plus contain, cover, and fill behavior.
- Existing text nodes render multiline and preformatted code with explicit
  line-height.
- Groups can clip children to local bounds.
- Compact numeric series lower to existing `core.path` children used by
  sparklines, meters, progress indicators, and gauges.

This avoids adding parallel node kinds when the existing IR carries the
required data strictly. Renderer behavior remains capability-level rather than
component-specific.

### Factory organization

Generic factories live in a declarative catalog under
`src/flow/sdk/generic/catalog.ts`. Systems-diagram factories use the same
table-driven pattern under `src/flow/sdk/diagram/catalog.ts`. Family-specific
factory functions inside each module preserve focused boundaries without
duplicating descriptor and registration plumbing across many tiny files.

Shared prop readers, descriptor builders, geometry helpers, node builders,
slot flattening, semantic-port forwarding, and diagnostic helpers move into
small internal utility modules. Existing implementations adopt a shared helper
only when touched by this work; broad unrelated refactoring is excluded.

## Generic Component Catalog

### Foundations

- `sdk.shape`
- `sdk.text`
- `sdk.richText`
- `sdk.icon`
- `sdk.image`
- `sdk.line`
- `sdk.arrow`
- `sdk.spacer`
- `sdk.inset`

`shape` supports named rect, rounded-rect, circle, ellipse, and path variants.
`line` and `arrow` expose `start` and `end` ports. `inset` accepts one child and
adds local padding; `spacer` contributes geometry without visible paint.

### Content

- `sdk.title`
- `sdk.paragraph`
- `sdk.caption`
- `sdk.codeBlock`
- `sdk.quote`
- `sdk.list`
- `sdk.keyValue`
- `sdk.propertyList`

Text components use `TextBlockNodeIr`. `codeBlock` supports a language label,
line numbers, highlighted line indices, and preformatted wrapping policy; it
does not perform syntax parsing. `list` supports ordered and unordered variants.
`propertyList` accepts structured entries and exposes indexed entry ports.

### Status

- `sdk.badge`
- `sdk.statusDot`
- `sdk.avatar`
- `sdk.iconLabel`
- `sdk.alert`
- `sdk.statusCard`
- `sdk.emptyState`

Status variants use semantic names such as `neutral`, `info`, `success`,
`warning`, and `danger`, resolved through theme roles rather than literal
colors. `avatar` supports initials, an icon, or an image source.

### Data display

- `sdk.stat`
- `sdk.metric`
- `sdk.table`
- `sdk.tableRow`
- `sdk.tableCell`
- `sdk.tagList`

`table` composes row and cell slots, computes deterministic column widths, clips
overflow, and exposes `row[i]`, `cell[r][c]`, and forwarded child ports.
`stat` is a compact label/value unit. `metric` adds optional unit, trend, and
lightweight indicator content.

### Navigation and progression

- `sdk.breadcrumb`
- `sdk.tabs`
- `sdk.pagination`
- `sdk.timeline`
- `sdk.timelineItem`

These are presentation components, not interactive controls. Active/current
props affect appearance and semantic actions only. Indexed item ports allow
timeline cues to emphasize individual entries.

### Indicators

- `sdk.progress`
- `sdk.meter`
- `sdk.gauge`
- `sdk.sparkline`
- `sdk.rating`
- `sdk.semaphore`

All numeric indicators accept finite values only. Ranges are errors unless
`clamp = true` is authored. `sparkline` accepts a finite numeric series and
lowers it to a `core.path` series child; it does not add axes, legends, or
chart interaction.

### Containers

- `sdk.section`
- `sdk.toolbar`
- `sdk.splitPane`
- `sdk.mediaObject`

Containers accept SDK child slots and forward semantic child ports. `splitPane`
supports horizontal and vertical arrangements. `mediaObject` composes a leading
visual with title, body, and optional trailing content.

## Diagram Component Catalog

### Actors and compute

- `sdk.user`
- `sdk.client`
- `sdk.service`
- `sdk.server`
- `sdk.process`
- `sdk.worker`
- `sdk.function`
- `sdk.container`
- `sdk.cloud`

Each exposes `self`, `input`, and `output` ports where meaningful. Components
share a common labeled-node contract and differ through iconography, chrome,
and optional detail fields.

### Storage

- `sdk.database`
- `sdk.dataStore`
- `sdk.cache`
- `sdk.file`
- `sdk.objectStore`
- `sdk.volume`

Storage nodes expose read/write ports in addition to input/output aliases.

### Messaging and network

- `sdk.queue`
- `sdk.topic`
- `sdk.stream`
- `sdk.eventBus`
- `sdk.gateway`
- `sdk.endpoint`
- `sdk.loadBalancer`
- `sdk.firewall`

Messaging nodes expose producer and consumer ports. Network nodes expose
inbound and outbound ports. These ports are aliases to stable generated nodes,
not renderer-only coordinates.

### Control flow

- `sdk.start`
- `sdk.end`
- `sdk.processStep`
- `sdk.decision`
- `sdk.merge`
- `sdk.delay`
- `sdk.retry`
- `sdk.loop`

`decision` exposes named branch ports plus indexed branches. `retry` and `loop`
are structural composites with explicit back-edge ports; they do not execute
runtime logic.

### Grouping and security

- `sdk.boundary`
- `sdk.zone`
- `sdk.cluster`
- `sdk.trustBoundary`

These are clipped or outlined containers with child slots, labels, semantic
entry/exit ports, and forwarded child ports.

### Supporting symbols

- `sdk.document`
- `sdk.terminal`
- `sdk.clock`
- `sdk.lock`
- `sdk.key`
- `sdk.warning`

Symbols use the shared icon registry and labeled-node contract. They remain
connectable and animatable like other SDK components.

## Common Component Contract

Every component requires `id`. Where meaningful, components accept:

- `x`, `y`, `width`, and `height`
- `position` for relative placement
- semantic theme-role overrides
- `variant`
- `label` or component-specific visible text
- an accessibility `description`

Factories provide useful default dimensions while preserving authored geometry.
Generated IDs derive only from `context.instanceId` and stable role names.

Every visible component exposes `self`. Components with semantic internals
expose named and indexed ports. Container components forward child ports using
the existing `role[i].port` convention.

Action families remain consistent:

- Content and chrome: `enter`, `emphasis`, `exit`
- Collections and containers: `enter`, `stagger`
- Connectable topology: `enter`, `draw`, `trace`
- Status and indicators: `enter`, `emphasis`, `pulse`, `exit`

## Validation and Diagnostics

Factories reject:

- Missing required props or slots
- Empty required strings or collections
- Non-finite numbers
- Unknown variants or icon names
- Malformed table rows, property entries, or numeric series
- Invalid ranges and values outside a range when `clamp` is not true
- Invalid slot cardinality

Diagnostics identify the component and instance, use the invocation source
range, and provide a concrete repair. Components do not silently select a
different semantic variant. Optional absent content is omitted cleanly.

## Testing

### Unit and schema tests

Each component family receives:

- Descriptor registration and canonical-name tests
- Successful factory expansion tests
- Stable ID and provenance tests
- Semantic port and action-target tests
- Default and authored geometry tests
- Required-prop, invalid-type, invalid-enum, range, and slot-cardinality tests

Each renderer foundation receives focused rendering and fallback tests while
remaining within the existing strict Scene IR node union.

### Integration decks

Add two generated catalog decks:

1. A generic-primitives deck covering every generic component.
2. A diagram-primitives deck covering every diagram component and its semantic
   connection ports.

Both decks compile through the production Flow pipeline, pass geometry
verification, have non-empty timelines, and are captured by the repository
screenshot tooling. Existing SDK examples and production decks remain part of
the regression suite.

### Verification

Run the explainer unit tests, type checking, Flow verification, geometry
verification, catalog-deck compilation, screenshot capture, and existing
package assertions. Screenshot review checks clipping, text overflow, icon
alignment, theme contrast, connector anchoring, and component spacing.

## Delivery Order

1. Shared SDK helpers and reusable Scene IR/renderer foundations.
2. Generic foundations and content.
3. Generic status, data display, navigation, indicators, and containers.
4. Generic catalog deck and regression verification.
5. Diagram actors/compute, storage, and supporting symbols.
6. Diagram messaging/network, control flow, and grouping/security.
7. Diagram catalog deck and full regression verification.

Each step leaves the registry free of stubs for components introduced in that
step. The generic catalog is complete and verified before diagram components
are registered.

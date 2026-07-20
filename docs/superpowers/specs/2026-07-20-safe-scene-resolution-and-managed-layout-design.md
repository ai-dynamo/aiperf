<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Safe Scene Resolution and Managed Layout Design

**Date:** 2026-07-20
**Status:** Approved
**Scope:** Flow-backed explainer scene resolution, routing safety, diagnostics, and opt-in managed layout

## Summary

Add one pure, deterministic scene-resolution stage between semantic Scene IR
and every geometry consumer. The resolved scene becomes the sole source of
truth for final bounds, generated semantic chrome, connector paths,
directionality, arrowheads, motion paths, and geometry diagnostics.

The system automatically corrects deterministic omissions, such as a missing
arrowhead on a directed edge, and reports source-mapped warnings when intent is
ambiguous. Existing absolute-positioned scenes retain their authored geometry.
New opt-in layout containers place and size their children automatically, so
decks can migrate incrementally.

## Problem

Flow scene authors currently make low-level choices that can silently produce
misleading or broken diagrams:

- Path and line edge modes suppress arrowheads unless authors opt back in.
- Independently resolved routes and motion signals can communicate different
  directions.
- Absolute child coordinates can overlap container titles, sibling nodes,
  connector corridors, or viewport edges.
- Semantic components can have more than one visual owner, causing duplicate
  text or chrome.
- Renderer and verifier geometry can diverge, allowing verification to pass
  while browser output remains broken.
- Fixes often require manually tuned coordinates and paths that are fragile
  under later content changes.

These are framework responsibilities. Authors should declare semantic intent
and relationships, not recreate layout, routing, and paint invariants in each
deck.

## Goals

- Establish one canonical resolved-scene representation consumed by rendering,
  verification, and motion.
- Make directed edges visibly directed by default.
- Preserve authored paths and points without losing declared directionality.
- Automatically fix deterministic, intent-preserving omissions.
- Produce actionable, source-mapped diagnostics for ambiguous geometry.
- Guarantee one visual owner for generated semantic chrome and text.
- Add opt-in content-aware containers that eliminate routine coordinate
  arithmetic.
- Preserve existing absolute scene geometry until a deck explicitly migrates.
- Verify semantic relationships rather than relying on brittle screenshot
  pixel equality.

## Non-goals

- Automatically rearranging existing absolute-positioned scenes.
- Introducing a general-purpose constraint solver.
- Removing explicit coordinates, paths, points, anchors, or layout overrides.
- Making browser DOM measurement part of compilation or scene resolution.
- Promoting every new diagnostic to a CI error immediately.
- Converting every existing deck in the first implementation.

## Canonical resolution pipeline

The pipeline becomes:

```text
Flow source
  -> semantic Scene IR
  -> resolved scene
  -> renderer
  -> verifier
  -> motion playback
```

The arrow above describes shared data ownership, not sequential runtime work:
the renderer, verifier, and motion playback each consume the same resolved
scene. They do not independently infer final geometry.

Resolution is pure and deterministic. Given the same semantic IR, viewport,
theme-independent geometry inputs, and resolver version, it returns identical
resolved geometry and diagnostics.

### Resolved scene contract

Each resolved node contains:

- Stable semantic node identity and source map.
- Final absolute bounds.
- Final child bounds for managed containers.
- Named ports and anchor points.
- Exactly one resolved set of generated visual parts.
- Content bounds used for clipping and overflow checks.

Each resolved connector contains:

- Stable edge identity and source map.
- Resolved source and target endpoints.
- Declared direction.
- Resolved path.
- Arrowhead visibility and tip geometry.
- Any fallback or ambiguity metadata.

The resolved scene also contains ordered diagnostics and a mapping from
semantic node IDs to their resolved geometry.

### Ownership boundaries

- Semantic IR preserves author intent and authored values.
- The resolver computes final geometry and safe defaults.
- Capability modules own semantic component layout and generated visual parts.
- The renderer paints resolved output without reconstructing semantic layout.
- The verifier inspects resolved output without maintaining parallel geometry
  rules.
- Motion references resolved connector geometry rather than resolving a second
  path.

## Safe correction policy

The resolver uses **auto-correct plus warn**.

It applies a correction only when the result is deterministic and preserves
declared intent. Examples include:

- Adding an arrowhead to a directed edge whose author did not explicitly
  disable it.
- Expanding a managed container to satisfy content minimums.
- Selecting a deterministic obstacle-free route when no path is authored.
- Reusing a connector's resolved path for its motion signal.

It emits a warning and preserves authored intent when multiple reasonable
interpretations exist. Examples include:

- Overlapping absolute-positioned siblings.
- An authored connector path that passes close enough to another node to make
  its source visually ambiguous.
- Content overflow in a fixed-size absolute node.
- A penetrating route fallback when no obstacle-free path exists.

Explicit author choices take precedence:

- `arrowhead = false` makes an edge intentionally undirected.
- Authored `path` or `points` controls connector shape.
- Explicit coordinates outside a managed container are not moved.
- Explicit managed-child position overrides are honored and diagnosed if they
  conflict with container invariants.

## Directed edge semantics

`sdk.Edge` represents a directed relationship by default.

| Authored form | Resolved behavior |
| --- | --- |
| `sdk.Edge(from, to)` | Directed with an arrowhead |
| `mode = "path"` with `from` and `to` | Authored shape, directed with an arrowhead |
| `mode = "line"` with `from` and `to` | Straight authored shape, directed with an arrowhead |
| `mode = "route"` | Automatically routed and directed |
| `mode = "curve"` | Automatically routed and directed |
| `arrowhead = false` | Explicitly undirected |

An authored path is interpreted from its first point to its last point. The
resolver compares that order with the declared source and target endpoints. A
reversed authored path produces a source-mapped warning; the resolver does not
silently reverse authored path data.

Undirected `sdk.Edge` guides use `arrowhead = false`. Dedicated guide and
divider capabilities remain undirected by definition.

## Routing

Connector resolution proceeds in this order:

1. Resolve source and target nodes, ports, and anchors.
2. Determine direction and arrowhead policy.
3. Use authored path or points when present.
4. Otherwise choose the requested straight, elbow, route, or curve strategy.
5. Route against final resolved node bounds.
6. Validate endpoint attachment, direction, obstacle interaction, and visual
   ambiguity.
7. Store the final path once for rendering, verification, and motion.

Automatic routing prefers paths that:

- Leave the source in its anchor direction.
- Enter the target in its anchor direction.
- Avoid unrelated resolved node bounds with configured clearance.
- Avoid skimming a non-endpoint node in a way that suggests a false source or
  target.
- Minimize bends, crossings, and unnecessary path length using deterministic
  tie-breaking.

The existing advanced curved router remains the curve-mode implementation.
The canonical resolver supplies it with final bounds and stores its result.

### Motion

An edge-associated signal uses `sdk.Signal(edge = "edge-id")` and consumes that
edge's resolved path. The `edge` property is mutually exclusive with `from`,
`to`, `path`, and `points`; conflicting inputs are compile errors.

Standalone motion paths remain supported for animations that do not represent
a connector.

## Semantic paint ownership

Every semantic capability has one owner for generated chrome and text.
Capability resolution returns generated visual parts; those parts do not also
remain as independently rendered compatibility children.

Duplicate generated IDs and duplicate paint roles are resolver errors. For
example, a semantic note resolves to one background and one caption, not a
semantic caption plus a compatibility caption with the same text.

Authored semantic children remain distinct from generated chrome. Their IDs and
timeline targets remain stable.

## Managed layout

Managed layout is opt-in. Existing scenes and nodes outside managed containers
retain their authored coordinates.

The first container capabilities are:

- `sdk.Stack`: horizontal or vertical ordered children.
- `sdk.Grid`: row-and-column placement.
- `sdk.Rail`: source, ordered stages, and destination along one axis.
- `sdk.Overlay`: intentional overlap with explicit alignment.
- `sdk.Frame`: titled content region with content-safe insets.

### Shared container inputs

Containers support:

- Direction or axis where applicable.
- Gap between children.
- Padding and title inset.
- Main-axis and cross-axis alignment.
- Minimum width and height.
- Optional fixed width and height.
- Child order.
- Optional per-child minimum size.
- Explicit per-child position override.

Authored width and height are minimum constraints by default. Managed
containers expand to fit content unless fixed sizing is explicitly requested.
Fixed-size overflow produces a diagnostic.

### Container outputs

Container resolution returns:

- Final container bounds.
- Final child bounds.
- Content bounds.
- Named container and child ports.
- Overflow and overlap diagnostics.

Managed containers reserve their title and detail bands before placing
children. Connector routing uses the resulting content and child bounds, so
authors do not manually create title clearance or connector corridors.

`sdk.Overlay` marks child overlap as intentional. Overlap among ordinary stack,
grid, rail, or frame children is a resolver error because it violates the
container contract.

## Diagnostics

Diagnostics are stable, source-mapped, and ordered by source location and code.
Each diagnostic includes:

- Severity.
- Stable code.
- Source range.
- Related node or edge IDs.
- Concise explanation.
- Suggested author action when automatic correction is unsafe.

Initial diagnostic families include:

- Directed edge missing a visible arrowhead.
- Authored path direction disagrees with `from` and `to`.
- Connector endpoint does not attach to its declared port.
- Connector intersects unrelated content.
- Connector route is visually ambiguous near a non-endpoint node.
- Router used a penetrating fallback.
- Node content exceeds fixed bounds.
- Node or label escapes the viewport.
- Unexpected sibling overlap.
- Duplicate semantic paint ownership.
- Duplicate generated visual ID.
- Motion duplicates an edge path instead of referencing the edge.

Auto-corrected conditions emit informational diagnostics, hidden by default and
shown by the verifier's verbose output. Ambiguous or lossy conditions are
warnings. Managed-container contract violations are errors.

## Compatibility and migration

Migration is incremental:

1. Add the resolved-scene contract and compatibility tests while preserving
   rendered output.
2. Move semantic paint ownership into resolution and remove duplicate paint
   paths.
3. Apply directed-edge defaults and add source-mapped diagnostics.
4. Make edge-associated motion consume resolved edge paths.
5. Add managed containers.
6. Convert one representative deck and tune the authoring APIs.
7. Convert remaining decks opportunistically.
8. Promote high-confidence repository-wide warnings to CI errors only after
   existing decks are clean.

The first representative deck contains panels, repeated tasks, directed edges,
shared connection targets, external control flow, and motion. The
`aiperf-vs-locust` worker-process slide is that representative.

No migration phase may silently reposition absolute scenes.

## Verification

### Unit tests

Unit tests cover:

- Deterministic resolved output.
- Directed-edge defaults and explicit opt-out.
- Authored path preservation and reversed-path diagnostics.
- Single semantic paint ownership.
- Shared resolved paths for connectors and motion.
- Stack, grid, rail, overlay, and frame layout.
- Content-aware sizing and fixed-size overflow.
- Stable diagnostics and source maps.

### IR verification

The IR verifier consumes resolved scenes and checks:

- All values are finite.
- Resolved IDs are unique.
- Connectors attach to declared endpoints.
- Directed connectors have visible arrowheads.
- Connector paths do not unexpectedly penetrate node bounds.
- Managed children remain inside content bounds.
- Ordinary siblings do not overlap.
- Generated semantic visuals have one owner.
- Content and arrow tips remain within the viewport allowance.

### Browser verification

Representative screenshots run at desktop, short-laptop, and mobile viewport
sizes. Browser checks validate:

- No visible clipping or text overlap.
- One visible copy of semantic text.
- Arrowheads appear only at the completed end of traced paths.
- Motion follows the rendered connector.
- Shell content and diagram content remain within their respective regions.

Golden assertions focus on relationships instead of whole-image equality:
non-overlap, endpoint attachment, containment, unique visible ownership, and
matching path geometry.

## Success criteria

The design is successful when:

- The worker-process slide can be expressed without an authored connector path
  or manually tuned child coordinates inside its managed containers.
- Its slot-granted edge visibly points to `Credit in` by default.
- Its motion follows the same resolved edge path.
- Its semantic note or label is painted exactly once.
- The verifier reports the same geometry used by the browser renderer.
- Existing unconverted absolute scenes retain their authored geometry.
- Clean representative decks produce no geometry warnings at supported
  viewport sizes.

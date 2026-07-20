<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Semantic Scene IR

## Summary

Retire compile-time scene-primitive desugaring. Every supported Flow capability
survives compilation as a semantic Scene IR node and is rendered through a
capability registry. Existing `.flow` syntax, component names, generated IDs,
semantic ports, timeline action targets, accessibility labels, and intended
visual output remain compatible.

This design supersedes the expansion-pipeline claim in
`2026-07-20-expanded-sdk-component-primitives-design.md` that SDK components
must emit only ordinary primitive Scene IR. SDK factories remain deterministic
and React-free, but emit semantic nodes instead of pre-rendered rect/text/group
trees.

## Goals

- Make semantic capability identity observable from parser output through
  rendering, verification, and diagnostics.
- Give one owner responsibility for each capability's layout, bounds, ports,
  and visual rendering.
- Eliminate geometry drift between compile-time desugaring and runtime layout.
- Keep `SceneRenderer.tsx` as scene orchestration rather than a capability
  switchboard.
- Preserve all existing `.flow` authoring syntax and stable externally
  referenced IDs.
- Make authored width and height minimum constraints by default; content may
  enlarge a component unless it explicitly supports clipping or overflow.
- Let browser rendering and Node verification consume the same pure geometry
  rules.

## Non-goals

- Changing `.flow` syntax or canonical SDK component names.
- Making SDK factories depend on React, DOM measurement, mutable global state,
  network access, or wall-clock time.
- Replacing semantic SDK components with dedicated React components embedded
  directly in `SceneRenderer.tsx`.
- Retaining a second hidden primitive expansion path as a fallback.
- Redesigning component appearance, theme roles, timeline semantics, or
  connector routing.

## Semantic IR contract

Each semantic Scene IR node contains:

- Stable `kind`, `capabilityId`, `id`, geometry, style, accessibility, fallback,
  source map, and optional children.
- Capability-specific authored data such as panel title/detail, stepper steps,
  lane title, or bracket orientation.
- Semantic children rather than generated visual-only descendants where the
  children are part of author intent.
- No React elements, DOM handles, measured browser values, or render cache.

Generated visual parts such as panel chrome and title text do not become
standalone serialized Scene IR nodes. They are produced by the capability
renderer and receive deterministic internal DOM IDs derived from the semantic
node ID. Authored child nodes and documented timeline targets remain Scene IR
nodes with stable IDs.

Strict Zod schemas reject unknown fields for every semantic capability. Package
serialization and round-trip tests preserve capability-specific data.

## Capability registry

Create focused modules under `src/core/diagram/capabilities/`. A registry maps
each `capabilityId` to a definition with the hooks it needs:

```ts
interface NativeSceneCapability<N extends SceneNodeLike = SceneNodeLike> {
  capabilityId: SceneCapabilityId;
  resolveLayout(input: CapabilityLayoutInput<N>): CapabilityLayout;
  render(input: CapabilityRenderInput<N>): ReactNode;
  resolvePorts?(input: CapabilityLayoutInput<N>): Readonly<Record<string, ConnectorEndpointIr>>;
}
```

`resolveLayout` is pure and deterministic. It returns:

- Final local bounds.
- Child local geometries.
- Optional semantic anchors or auxiliary geometry needed by rendering.

Leaf capabilities use a shared identity layout helper. Container capabilities
compute content-aware layout. Both rendering and scene indexing consume the
same resolved layout object.

The first module families are:

- `primitives.ts`: rect, text, line, path, circle, ellipse, group.
- `chrome.ts`: panel, header, card, chip, note, label, divider, bracket,
  callout, and legend.
- `layout.ts`: stack, grid, rail, pad, lane, band, swimlane, and stepper.
- `topology.ts`: connector, route, fan, pipeline-supporting groups.
- `motion.ts`: signal, pulse, and flow.

Modules can share pure geometry and visual helpers, but capability registration
is explicit and duplicate IDs fail during bootstrap.

## Scene orchestration

`SceneRenderer` retains:

- Recursive traversal and world transforms.
- Timeline appearance state.
- Scene indexing.
- Relative-position resolution.
- Connector and motion-path routing.
- Theme resolution, marker definitions, and accessibility policy.

For each semantic node it:

1. Resolves the capability definition.
2. Calls `resolveLayout` once for the current node and children.
3. Stores final world bounds in the scene index.
4. Passes child geometries to recursive traversal.
5. Delegates visual output to the capability's `render` hook.

Unknown capabilities fail closed with a diagnostic fallback; they are not
silently rendered as generic groups.

## Geometry rules

- Explicit `x` and `y` place the final component origin.
- Positive authored width and height are minimum dimensions.
- Zero or absent dimensions request intrinsic sizing.
- Containers expand to fit all children, gaps, title bands, padding, and
  borders unless an explicit clipping/overflow property says otherwise.
- Text-bearing fixed chrome uses deterministic metric constants; browser text
  measurement does not affect layout.
- Child layout is local to the semantic parent.
- Scene indexing, connector anchors, relative positioning, and rendering use
  the same resolved bounds.
- Connectors never anchor to generated visual-only chrome nodes.

## Compiler and SDK migration

The parser and authored syntax remain unchanged. Package-form aliases and SDK
component calls lower directly to semantic Scene IR:

1. SDK descriptor validation and deterministic factories remain.
2. Factories emit semantic nodes carrying capability-specific data.
3. Semantic port references resolve after all component instances register.
4. SDK provenance is stripped at the existing package boundary.
5. No call to `desugarPackageNode` occurs.

Delete `DESUGAR_PACKAGE_CAPABILITIES` and the corresponding switch cases after
all capabilities have native registry definitions. Rename the remaining
first-class lowering helper to reflect that all package nodes are lowered
directly.

Compatibility aliases accepted by the parser continue to map to canonical
capability IDs. Existing raw package-form scenes and SDK-authored scenes must
produce equivalent semantic nodes.

## IDs, ports, and timelines

- Invocation root IDs do not change.
- Authored child IDs do not change.
- Existing documented generated action IDs remain stable where timelines can
  target them. If a generated visual-only ID was never part of the action/port
  contract, it may become a DOM-only ID.
- Semantic ports resolve to the semantic root or authored child nodes, never
  to ephemeral renderer internals.
- Existing `enter`, `stagger`, `draw`, `trace`, `emphasis`, `pulse`, `fade`,
  and `exit` action targets remain valid.

Before migrating a capability, tests inventory its current ports and actions.
Any unavoidable contract change is treated as a migration error rather than
silently accepted.

## Verifier architecture

Extract capability layout into environment-neutral pure modules. The browser
renderer and Node verifier invoke equivalent exported geometry functions; the
verifier does not maintain an independent handwritten layout mirror.

Verification checks:

- Final bounds are finite and positive where required.
- Children remain within non-overflowing containers.
- Text-bearing chrome meets deterministic minimum-width rules.
- Connector endpoints resolve against final semantic bounds.
- Semantic IDs, ports, and timeline targets exist.
- No desugar-only generated Scene IR nodes remain.

## Migration order

1. Add the capability registry and pure layout result contract while retaining
   current rendering as an adapter.
2. Migrate native primitives and existing stack/grid/rail layout.
3. Migrate lane, band, swimlane, stepper, and pad.
4. Migrate chrome and shape macros.
5. Migrate connector-adjacent and motion capabilities still using macro
   lowering.
6. Switch SDK factories and package-form lowering to semantic nodes.
7. Remove `desugarPackageNode`, macro capability sets, and adapter code.
8. Update the expanded-primitives design to reference this semantic IR
   contract.

Each step must compile every deck before the next begins. The final state has
one production rendering path and no feature flag.

## Testing and acceptance

- Schema and round-trip tests for every semantic node family.
- Registry duplicate/missing capability tests.
- Pure layout tests for intrinsic size, authored minimum size, local child
  placement, and deterministic output.
- Compatibility tests comparing existing deck IDs, ports, actions, and
  timelines before and after migration.
- Build and compile all Flow decks.
- Run strict SDK-authoring and geometry verification.
- Capture all pages of the SDK examples and catalog decks through the
  non-HMR preview server.
- Visually inspect clipping, text padding, connector anchoring, motion paths,
  and final-card isolation.

Acceptance requires:

- No production import or invocation of `desugarPackageNode`.
- No macro-only capability set.
- Existing `.flow` files compile without syntax changes.
- Existing documented IDs, ports, and action targets remain valid.
- Browser and verifier geometry agree.
- All tests, package gates, and screenshot checks pass.


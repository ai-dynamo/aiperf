<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Explainers Browser Flow Toolchain Design

**Date:** 2026-07-18
**Status:** Approved
**Host:** `apps/explainers`

## Goal

Bring the useful browser-safe Flow compiler, authoring, evaluation, and
verification capabilities into `apps/explainers` as locally owned source.
Deck compilation stays synchronous and reliable. Heavy developer-only
evaluation and verification load lazily and are not awaited by application
startup/render initiation. This makes no claim about React commit order. The
eager compiler remains on the production path and contributes to the main
bundle.

## Locked decisions

- All first-party browser toolchain source lives under
  `apps/explainers/src/flow`.
- `apps/explainers` solely owns this browser toolchain source. The standalone
  `apps/aiperf-flow` workspace has been removed, so no separate Flow workspace
  remains to import, alias, or depend on.
- The full browser-safe compiler and authoring surface is owned locally.
- Pure runtime evaluation and IR/geometry verification are copied or ported
  locally.
- Heavy evaluator/verifier code is dynamically imported only for developer
  diagnostics.
- No authoring or inspection UI is added.
- Node-only filesystem, CLI, process, Playwright, deploy, React app shell,
  Canvas backend, and narrative/WASM modules are excluded.
- Do not create or modify tests and do not run test commands.
- Existing deck ids, routes, narration, scenes, and rendering behavior remain
  stable.

## Architecture

```text
decks-flow/*.flow
    │
    ▼
src/flow/compiler/compile-explainer
    │
    ├── parse / symbols / components / link / themes / capabilities
    ├── strict validation / lower / schema validation
    ├── explainer-set id + route validation
    └── actionable diagnostics
    │
    ▼
DeckPackage → packageToDeckDefinition → DECK_REGISTRY

import.meta.env.DEV
    │
    └── dynamic import src/flow/dev-tools
            ├── pure runtime evaluation
            ├── IR/timeline/geometry verification
            └── console diagnostics
```

The eager path contains only code required to compile and validate decks before
registry construction. Runtime evaluation, display-list construction, layout
inspection, and geometry verification live behind a dynamic development-only
import.

## Current integration boundary

`src/flow/index.ts` is the public browser-safe barrel. It exports the browser
compiler, language formatter, schema, and local SDK surface; it does not
eagerly export `runtime` or `dev-tools`. It is not the application production
entry. Live deck loading imports compiler and schema modules directly from
`src/core/load-deck-flows.ts` (`compile-explainer`, `validate-explainer-set`,
`diagnostics`, `schema/index`). Deck sources compile synchronously into
`DeckPackage` values before `DECK_REGISTRY` construction. That eager compiler
path is included in the main production bundle.

After initiating application render, development builds dynamically import
`src/flow/dev-tools/index.ts` to evaluate and verify those already-compiled
packages. React commit is not guaranteed at the moment that import starts.
Lazy developer diagnostics are not awaited by application startup/render
initiation, report to the console, do not mutate package IR, and do not run in
production. Production playback continues through `packageToDeckDefinition`
and `SceneRenderer`.

`apps/explainers` owns this implementation outright. The standalone
`apps/aiperf-flow` workspace has been deleted, so there is no upstream Flow
workspace to import, alias, or synchronize with; this toolchain is now the
canonical first-party Flow source. Node filesystem/CLI/process APIs, implicit
network module loading, Playwright, alternate Flow React/Canvas applications,
and narrative/WASM/worker or inspection-UI subsystems remain outside this
integration boundary.

## Local source layout

```text
apps/explainers/src/flow/
  index.ts
  compiler/
    browser.ts
    compile-explainer.ts
    compile-source.ts
    components.ts
    expand-symbols.ts
    link.ts
    module-resolution.ts
    symbols.ts
    themes.ts
    validate.ts
    validate-explainer-set.ts
    ...
  language/
    index.ts
    parser.ts
    formatter.ts
    ...
  schema/
    index.ts
    component-descriptor.ts
    ...
  runtime/
    evaluate/
    leaves/
    display-list.ts
    damage-tracker.ts
    hit-region-index.ts
    quality-policy.ts
  dev-tools/
    diagnostics.ts
    verify-deck.ts
    verify-geometry.ts
    index.ts
```

File names may follow upstream names where that keeps copied modules easy to
audit. Barrels remain narrow and browser-safe.

## Compiler and language capabilities

The local browser compiler includes:

- Explainer compilation into `DeckPackage`.
- General Flow compilation into `FlowIr`.
- Symbol collection and invocation expansion.
- Component descriptor/catalog validation and prop binding.
- Linking and reference diagnostics.
- Theme collection and validation.
- Capability manifest and strict-mode validation.
- Multi-deck id and route uniqueness validation.
- Module resolution over host-injected source strings.
- Timeline validation.
- Deterministic browser-safe serialization.
- Full document parsing and formatting.

`compileExplainerSource` must stop discarding `capabilities` and `strict`.
Native embedded scenes pass through applicable symbols, linking, themes,
capability, component, and strict validation before lowering. Package-style
embedded scenes continue through their dedicated package-scene normalization,
schema, primitive, and timeline validation path.

The live deck loader compiles all sources, then calls
`validateExplainerSet`. Duplicate ids or routes fail registry construction with
source-oriented diagnostics.

## Module resolution

Browser module resolution cannot use filesystem APIs. The local module resolver
accepts an injected map of canonical module URI to source text. Vite raw imports
supply available `.flow` modules.

Only exact relative `.flow` imports and explicitly injected package/HTTPS
modules resolve. The resolver performs no network requests. Integrity and
canonical identity functions may use the browser-safe hashing dependency
required by the upstream resolver.

Current one-file decks require no imports, so this capability does not alter
their behavior.

## Formatting and serialization

The copied formatter exposes the upstream formatting behavior it actually
supports. It must not claim full explainer-deck formatting until explainer AST
formatting exists.

Deterministic serialization includes only pure browser functions. Filesystem
writers are excluded. Developer tools may expose serialized IR/package text to
the console or programmatic callers; no download/copy UI is added.

## Runtime evaluation

Copy only the pure TypeScript evaluation closure needed to:

- Evaluate scene/timeline state at a requested time.
- Build display-list and semantic projections.
- Apply contribution merging and layout leaves.
- Inspect damage/hit/quality data when those are dependencies of evaluation.

Do not copy:

- The Flow React application or renderer.
- Canvas rendering backends.
- Explainer UI components.
- Kokoro, ONNX Runtime, WASM assets, workers, or narration.
- Fullscreen, immersive HUD, or site navigation.

`SceneRenderer` remains the production renderer. The copied evaluator is a
developer diagnostic oracle, not a second production rendering path.

## Developer verifier

Port the pure logic from the existing IR and geometry verifier into TypeScript:

- Missing or empty scene roots/timelines.
- Unknown timeline targets.
- Unresolvable connector/fan endpoints.
- Orphaned geometry.
- Endpoint anchoring.
- Mid-draw and arrowhead contracts.
- Non-finite geometry.

The verifier consumes in-memory compiled `DeckPackage` values. It never reads
generated JSON.

After registry compilation, development builds dynamically import the verifier
and run it asynchronously. Findings are grouped by deck and printed with stable
codes and source/scene identifiers. Verification findings do not silently
mutate IR or rendering.

Production builds must not execute developer verification. The source remains
locally owned, but Vite may remove the development-only import from production
output.

## Diagnostics

Add shared diagnostic formatting that includes:

```text
source:line:column: severity code: message (repair)
```

The live loader, compiler, and developer verifier use one formatter. Compiler
errors fail closed. Developer evaluator/verifier findings are reported to the
console and do not add a user-facing error panel.

## Dependency policy

Allowed direct browser dependencies in `apps/explainers`:

- `chevrotain`
- `zod`
- `js-sha256` when required by deterministic serialization/module identity

Do not add Node polyfills. Any copied module that imports `node:*`, references
`process`, or assumes filesystem/network access is outside the browser boundary
and must be excluded or split into a pure helper.

## Ownership

The Flow toolchain source is owned by `apps/explainers`. The standalone
`apps/aiperf-flow` workspace has been removed, so there is no separate Flow
workspace to resolve into at build time and no upstream project to synchronize
with.

Some files retain the names and structure from their original migration so the
history stays easy to audit. Explainer-specific integration and developer-tool
adaptations are local. `apps/explainers` is now the source of truth for all
future changes.

Fan endpoint validation rejects invalid fan topology instead of lowering
unresolved endpoints to `{ x: 0, y: 0 }`.

## Error handling

- Parse, link, component, theme, capability, strict, lower, and schema errors
  return structured diagnostics.
- Duplicate deck ids/routes fail registry construction.
- Unsupported module imports return diagnostics; the browser never fetches
  them implicitly.
- Developer verification failures are isolated from registry construction
  unless they duplicate an existing compiler/schema invariant.
- A failure to dynamically load developer tooling logs one clear warning and
  does not break production deck playback.

## Verification constraint

Per explicit user direction:

- Do not add, modify, or delete tests for this feature.
- Do not run Vitest, Playwright, Cargo test, or other test commands.

Allowed verification:

- TypeScript no-emit checks.
- Production Vite build.
- Static import-boundary scans.
- Bundle inspection.
- Documentation consistency checks.

## Success criteria

- `apps/explainers` locally owns the full useful browser-safe Flow compiler,
  formatter, evaluator, and verifier source.
- No runtime import or Vite alias points to a separate Flow workspace; the
  toolchain resolves entirely within `apps/explainers`.
- Live deck compilation honors strict/capability validation and set uniqueness.
- Invalid references, component props, themes, fan endpoints, ids, and routes
  produce actionable diagnostics.
- Heavy runtime evaluation and geometry verification load only in development.
- Production playback continues through `DeckPackage`,
  `packageToDeckDefinition`, and `SceneRenderer`.
- No Node-only or alternate app/runtime subsystems enter the browser bundle.

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Live Browser Flow Compilation for Explainers

**Date:** 2026-07-18
**Status:** Approved
**Host:** `apps/explainers`

## Goal

`apps/explainers` compiles every `decks-flow/*.flow` source directly in the
browser. The application does not load generated `DeckPackage` JSON and does
not require an explainer package pre-build.

The runtime uses the real Flow explainer compiler. It does not use the
incomplete regex-based legacy loader.

## Locked decisions

- Compilation always runs in the browser in development and production.
- The browser imports `.flow` files as raw source through Vite.
- `apps/explainers` owns the browser-safe compiler, language, and schema source
  needed by explainer compilation.
- The standalone `apps/aiperf-flow` workspace has been removed; the explainers
  runtime has no imports, aliases, or workspace dependency on any separate Flow
  workspace.
- Generated `decks-generated/*.package.json` artifacts are retained as
  deterministic build-time mirrors for package assertions and deployment
  checks, but the browser registry neither imports nor requires them.
- Legacy and incomplete Flow loaders are deleted.
- `packageToDeckDefinition`, `ExplainerShell`, and `SceneRenderer` remain the
  runtime adaptation and rendering path.
- Existing deck ids and routes remain stable.
- Do not add tests or run test commands for this change. Verification is
  limited to type checks, production builds, and relevant static checks.

## Runtime architecture

```text
apps/explainers/decks-flow/<id>.flow
    │
    │ Vite raw-source import
    ▼
compileExplainerSource
    │
    ▼
DeckPackage in browser memory
    │
    ▼
packageToDeckDefinition
    │
    ▼
DECK_REGISTRY → ExplainerShell → SceneRenderer
```

The live loader eagerly imports all deck sources so the existing synchronous
registry API can remain synchronous. Compilation occurs during application
module initialization. No generated package fallback exists.

## Locally owned browser compiler

The Flow implementation is owned under `apps/explainers/src/flow/`. It is
organized into focused `compiler`, `language`, `schema`, `sdk`, `runtime`, and
`dev-tools` directories, with browser-safe entrypoints that expose the compiler,
schema, formatter, SDK, and verifier surfaces used by the app and local tools.

All toolchain modules use relative imports within `apps/explainers`. They do
not import files outside the explainers app; the former `@aiperf/flow-compiler`,
`@aiperf/flow-language`, and `@aiperf/flow-schema` packages no longer exist.

The browser compile closure includes explainer parsing and lowering:

- Compiler: explainer compile, lower, scene lower/desugaring, timeline
  validation, and the type-only linked-document vocabulary required by native
  scene lowering.
- Language: tokens, AST, parser, explainer grammar, and embedded-scene parser.
- Schema: diagnostics, source ranges, deck package, scene IR, JSON values,
  layout plans, semantic model, theme, and the capability manifest/constants
  required by the compile request.

Node-only filesystem work remains outside the browser closure. The same local
toolchain also provides deterministic package serialization, SDK expansion,
formatting, and runtime evaluation for build-time tools; the React renderer
remains under `apps/explainers/src/core/diagram`.

The explainers package declares the third-party browser dependencies used by
the copied source directly: `chevrotain` and `zod`. `js-sha256` is not needed.
Browser resolution requires no pre-built `dist/` directory and no Vite alias
outside `apps/explainers`.

## Live deck loader

Replace `src/core/load-deck-packages.ts` with a live Flow source loader.

The loader:

1. Eagerly imports `../../decks-flow/*.flow` as raw strings.
2. Derives each source name from its import path.
3. Calls `compileExplainerSource` with `FOUNDATION_CAPABILITIES`, strict
   validation, and strict SDK-authoring enforcement.
4. Formats compiler diagnostics with source, line, column, severity, code,
   message, and repair guidance.
5. Throws when compilation fails or emits error diagnostics.
6. Validates that the compiled id agrees with the source filename.
7. Converts the in-memory package through `packageToDeckDefinition`.

`deck-registry.ts` retains `EXPECTED_DECK_ROUTES` and validates each compiled
deck against that map. Missing, duplicate, or mismatched decks fail closed.

## Runtime boundary and retained package tooling

The production registry imports only raw `decks-flow/*.flow` sources through
`load-deck-flows.ts`. It does not import `load-deck-packages.ts` or generated
JSON. The latter module and `src/decks-generated/*.package.json` remain as an
auxiliary package-artifact seam for assertions and compatibility checks.

`apps/explainers/scripts/build-explainer-packages.ts` mirrors the live compile
policy with the same local compiler, capability manifest, set validation, and
strict SDK-authoring gate. It writes deterministic package JSON for static
verification and removes stale artifacts. This command is not a prerequisite
for `vite dev`, the production build, or runtime registry construction.

The obsolete regex loader, external `@aiperf/flow-*` packages, and standalone
`apps/aiperf-flow` workspace are removed. `apps/explainers` is the source of
truth for future Flow language changes.

## Tooling

The Flow verifier should consume `.flow` source directly and invoke the same
real explainer compiler in Node. Its default path no longer reads generated
JSON, and `--from-flow` is removed because source compilation is now inherent.

The aggregate explainer gate intentionally rebuilds package mirrors before
running package, strict-authoring, and IR assertions. These artifacts verify
serialization parity; they are not runtime inputs.

Documentation describing the packages-only runtime is updated to describe the
live browser compile path.

## Error behavior

Compilation is deterministic and synchronous. A broken source prevents registry
construction and reports actionable compiler diagnostics. There is no silent
fallback to stale JSON or a partial regex parse.

The initial implementation may surface errors through the existing Vite/runtime
error boundary. A dedicated in-app compiler error screen is out of scope.

## Verification

Per the explicit scope constraint, do not create tests and do not run test
commands.

Allowed verification:

- TypeScript no-emit checks.
- Production Vite build.
- Existing non-test static assertion scripts that remain applicable.
- Repository documentation consistency checks when documentation is changed.

## Success criteria

- Production and development browser builds compile all explainer `.flow`
  sources in memory with the real compiler.
- Browser compilation resolves entirely from source and dependencies owned by
  `apps/explainers`; no separate Flow workspace is imported or aliased.
- Editing a `.flow` source updates the Vite application without regenerating
  JSON.
- No generated DeckPackage JSON is required or loaded by the browser registry.
- No pre-build command is required before starting or building explainers.
- No legacy regex Flow loader remains.
- Existing deck ids, routes, narration, scenes, timelines, and rendering
  adaptation continue through the compiled `DeckPackage`.

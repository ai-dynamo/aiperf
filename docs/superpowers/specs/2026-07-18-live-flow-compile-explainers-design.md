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
- `apps/explainers` owns a local browser-safe copy of the exact compiler,
  language, and schema source needed by explainer compilation.
- The explainers runtime has no imports, aliases, or workspace dependency on
  `apps/aiperf-flow`.
- Generated `decks-generated/*.package.json` artifacts and their pre-build
  pipeline are deleted.
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

The exact transitive source needed by `compileExplainerSource` is copied under
`apps/explainers/src/flow/`. It is organized into focused `compiler`,
`language`, and `schema` directories, with a browser entrypoint that exports
`compileExplainerSource`, `FOUNDATION_CAPABILITIES`, `hasErrors`, and the types
needed by the live loader.

All copied modules use relative imports within `apps/explainers`. They do not
import `@aiperf/flow-compiler`, `@aiperf/flow-language`,
`@aiperf/flow-schema`, or files outside the explainers app.

The local closure includes only explainer parsing and lowering:

- Compiler: explainer compile, lower, scene lower/desugaring, timeline
  validation, and the type-only linked-document vocabulary required by native
  scene lowering.
- Language: tokens, AST, parser, explainer grammar, and embedded-scene parser.
- Schema: diagnostics, source ranges, deck package, scene IR, JSON values,
  layout plans, semantic model, theme, and the capability manifest/constants
  required by the compile request.

Node-only and unrelated modules are not copied. In particular, there is no
package writer, canonical packer, module resolver, filesystem import, general
Flow compile barrel, formatter, CLI, or runtime renderer in the local compiler.

The explainers package declares the third-party browser dependencies used by
the copied source directly: `chevrotain` and `zod`. `js-sha256` is not needed.
Browser resolution requires no pre-built `dist/` directory and no Vite alias
outside `apps/explainers`.

## Live deck loader

Replace `src/core/load-deck-packages.ts` with a live Flow source loader.

The loader:

1. Eagerly imports `../../decks-flow/*.flow` as raw strings.
2. Derives each source name from its import path.
3. Calls `compileExplainerSource` with `FOUNDATION_CAPABILITIES` and strict
   validation.
4. Formats compiler diagnostics with source, line, column, severity, code,
   message, and repair guidance.
5. Throws when compilation fails or emits error diagnostics.
6. Validates that the compiled id agrees with the source filename.
7. Converts the in-memory package through `packageToDeckDefinition`.

`deck-registry.ts` retains `EXPECTED_DECK_ROUTES` and validates each compiled
deck against that map. Missing, duplicate, or mismatched decks fail closed.

## Removed paths

Delete:

- `apps/explainers/src/decks-generated/` package artifacts.
- `apps/explainers/src/core/load-deck-packages.ts`.
- `apps/aiperf-flow/scripts/build-explainer-packages.mjs`.
- `apps/explainers/scripts/assert-deck-packages.mjs`.
- The incomplete regex loader at
  `apps/aiperf-flow/packages/runtime/src/explainer/flow-loader.ts`.
- Package scripts and Makefile targets whose purpose is generating or
  asserting `decks-generated` artifacts.

Remove stale imports and configuration that grant access to or consume
`decks-generated`, including the aiperf-flow preview package registry and
runtime tests that import package JSON. Because this change explicitly excludes
tests, obsolete tests tied only to generated package artifacts are deleted
rather than rewritten.

The canonical Flow workspace sources remain otherwise untouched. The copied
browser compiler is intentionally owned by `apps/explainers`; keeping it in
sync with future Flow language changes is a manual maintenance responsibility.

## Tooling

The Flow verifier should consume `.flow` source directly and invoke the same
real explainer compiler in Node. Its default path no longer reads generated
JSON, and `--from-flow` is removed because source compilation is now inherent.

The aggregate explainer gate may retain static and IR verification, but it must
not generate or require committed package artifacts.

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
  `apps/explainers`; it does not import or alias into `apps/aiperf-flow`.
- Editing a `.flow` source updates the Vite application without regenerating
  JSON.
- No generated DeckPackage JSON is required or loaded.
- No pre-build command is required before starting or building explainers.
- No legacy regex Flow loader remains.
- Existing deck ids, routes, narration, scenes, timelines, and rendering
  adaptation continue through the compiled `DeckPackage`.

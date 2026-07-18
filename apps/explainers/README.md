<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Explainers

Unified narrated slideshow SPA for AIPerf architecture topics.

## Develop

```bash
cd apps/explainers
npm install
npm run dev
```

Routes (hash router):

- `/#/` — hub
- `/#/rust-architecture`
- `/#/slurm-velo`
- `/#/dynosim`

## Build

```bash
npm run build
npm run preview
```

## Browser Flow toolchain

`apps/explainers` owns its browser-safe Flow toolchain under `src/flow`; it
does not import, alias, or build against first-party modules from
`apps/aiperf-flow`.

`src/flow/index.ts` is the public browser-safe barrel (compiler, formatter,
schema, SDK). It is not the application production entry. Live deck loading
uses direct imports from `src/core/load-deck-flows.ts` into compiler and schema
modules (`compile-explainer`, `validate-explainer-set`, `diagnostics`,
`schema/index`) to compile `.flow` sources into `DeckPackage` values and
validate the deck set before registry construction. That eager compiler path
is part of the main production bundle.

The pure runtime evaluator and IR/geometry verifier are local developer tools.
They load only through a development-only dynamic import after initiating
application render (React commit is not guaranteed at that point). Lazy
developer diagnostics are not awaited by application startup/render initiation
and do not run in production; this makes no claim about React commit order. The
eager compiler still contributes to main-bundle size. Diagnostic findings are
written to the console and never mutate compiled packages or the production
renderer. `SceneRenderer` remains the production rendering path.

The browser boundary excludes Node filesystem, CLI, and process APIs; network
module fetching; Playwright; alternate React applications and Canvas backends;
and narration, WASM, worker, and inspection UI subsystems. Future changes in
`apps/aiperf-flow` are synchronized manually: retain comparable local file
names where practical, port only browser-safe behavior, and re-run the build,
fail-closed import-boundary scan, and documentation consistency check after
each sync.

## Deploy to GitHub Pages

```bash
npm run deploy:pages
```

Legacy subfolder URLs redirect into hash routes:

- `rust-architecture-explainer/` → `#/rust-architecture`
- `slurm-explainer/` → `#/slurm-velo`

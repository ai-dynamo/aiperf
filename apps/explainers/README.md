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
- `/#/rust-architecture-atlas`
- `/#/segment-pools`
- `/#/slurm-velo`
- `/#/velo-deep-dive`
- `/#/cellular-internals`
- `/#/cellular-algorithms`
- `/#/dynosim`
- `/#/steppable-replay-engine`
- `/#/tstar-warmup`
- `/#/synthetic-dataset-generator`
- `/#/aiperf-vs-locust`
- `/#/flow-sdk-examples`
- `/#/sdk-generic-catalog`
- `/#/sdk-diagram-catalog`

## Build

```bash
npm run build
npm run preview
```

## Browser Flow toolchain

`apps/explainers` owns the Flow language, compiler, SDK, runtime evaluator,
developer verifier, and production renderer under `src/flow`.

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
and narration, WASM, worker, and inspection UI subsystems. Build-time scripts
under `scripts/` may use Node APIs while importing the same local compiler used
by the browser. Re-run the build, fail-closed import-boundary scan, and
documentation consistency check after toolchain changes.

## Deploy to GitHub Pages

```bash
npm run deploy:pages
```

Legacy subfolder URLs redirect into hash routes:

- `rust-architecture-explainer/` → `#/rust-architecture`
- `slurm-explainer/` → `#/slurm-velo`

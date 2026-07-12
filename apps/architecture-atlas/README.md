<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Architecture Atlas

The Architecture Atlas is an internal, source-grounded SPA for understanding
AIPerf's Python orchestration, Rust execution architecture, implementation
status, and crate relationships. It includes six guided views, a unified
interactive graph, and crate reference routes with executive, developer, and
maintainer audience lenses.

## Requirements

- Node.js 22.12 or newer
- npm 10.9 or newer
- Chromium installed through Playwright for browser tests

## Local commands

Run commands from this directory after activating the repository virtual
environment.

```bash
npm ci
npm run dev
```

`npm run check` is the complete non-browser gate: content validation,
typechecking, linting, unit/accessibility tests, and the production build.
`npm run check:all` adds the production-preview Playwright suite.

```bash
npm run check
npx playwright install chromium
npm run check:all
```

Other focused commands are `npm run validate:content`, `npm run typecheck`,
`npm run lint`, `npm test`, `npm run build`, and `npm run preview`.

## Architecture and content updates

Typed content lives in `src/content/`; schemas and integrity rules live in
`src/domain/`. Guided, unified-atlas, and crate views consume the same catalog.
Update source-grounded records and their tests together, then run
`npm run check` and `npm run e2e`.

Implementation claims must be verified against current source. Specs describe
intent and are not accepted as implementation evidence. Maintainer citations
must identify repository-relative source paths; the app converts valid
citations to absolute GitHub URLs and rejects malformed or app-local evidence
through validation and tests.

The detailed design record is in `docs/design.md`.

## Static artifact

`npm run build` writes the deployable SPA to `dist/`. CI uploads that directory
as the `aiperf-architecture-atlas` artifact. Deploy its contents to a static
host configured to return `index.html` for unknown paths so TanStack Router
deep links such as `/atlas` and `/crates/aiperf-clock` load correctly.

The artifact is self-contained and requires no application server or runtime
data service.

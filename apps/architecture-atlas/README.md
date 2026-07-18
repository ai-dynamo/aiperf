<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Architecture Atlas

The Architecture Atlas is an internal, source-grounded, graph-first SPA for
understanding AIPerf's Python-to-Rust execution story. The graph is the product:
users navigate real architecture topology, execution flavors, and code evidence
without switching to separate guided-card views.

## Requirements

- Node.js 22.12 or newer
- npm 10.9 or newer
- Chromium installed through Playwright for browser tests

## Product scope

The app renders the graph-first architecture journey from Python Config v2
authoring through strict `aiperf-runner` validation, workload execution,
transport dispatch, observation, and report return.

### Scene routes

The canonical scene routes are:

- `/` (Runtime composition)
- `/scenes/runner-protocol-registries`
- `/scenes/scheduling-phase-lifecycle`
- `/scenes/dataset-segment-pipeline`
- `/scenes/endpoint-bindings-transports`
- `/scenes/graph-ir-execution`
- `/scenes/metrics-telemetry`
- `/scenes/accuracy-evaluator-hosting`
- `/scenes/crate-dependency-topology`

Legacy guided paths such as `/journey`, `/execution`, `/data-plane`,
`/observability`, `/parity`, and `/atlas` redirect into these graph-first
routes while preserving supported search state.

### Audiences and execution flavors

Audience controls change topology depth for:

- `executive` (high-level product architecture)
- `developer` (subsystems and runtime seams)
- `maintainer` (crate/module/symbol-level depth)

Execution flavor controls morph one shared graph across:

- `native_http`
- `native_grpc`
- `online_mock`
- `dynamo_offline`
- `dynamo_online` (built, feature-gated in-process replay through the registered
  `dynosim` backend)

Primary and comparison flavor overlays are URL-backed graph state, not static
route variants. Only a distinct dedicated Dynamo-online runner pair remains
planned; the current product path selects online replay through `dynosim`.

### Interaction model

Graph interactions include node drag persistence, tier expansion and collapse,
edge selection, upstream/downstream/isolate tracing, pulse playback controls,
keyboard-first graph operations, evidence drawer focus restoration, and a
reduced-motion equivalent flow path.

## State and sharing

Graph state is versioned and schema-validated. The app persists audience and
graph layout preferences to local storage, mirrors shareable graph state in the
URL, and recovers to canonical scene defaults when URL or stored state is
invalid, stale, or incompatible.

Semantic architecture content is never accepted from URL payloads.

## Source grounding and validation

Typed content lives in `src/content/`; schemas and integrity rules live in
`src/domain/`. Scene routes, graph topology, and crate routes consume one
canonical catalog.

Implemented claims must be backed by repository source evidence. Planned claims
must be backed by explicit design evidence. The content validator rejects
malformed paths, invalid references, and app-local evidence links.

## Local commands

Run commands from this directory after activating the repository virtual
environment.

```bash
npm ci
npm run dev
```

`npm run check` is the complete non-browser gate: content validation,
typechecking, linting, unit/accessibility tests, and production build.
`npm run check:all` adds the Playwright production-preview suites.

```bash
npm run check
npx playwright install chromium
npm run check:all
```

Focused commands:

- `npm run validate:content`
- `npm run typecheck`
- `npm run lint`
- `npm test`
- `npm run build`
- `npm run e2e`
- `npm run preview`

## Static artifact

`npm run build` writes the deployable SPA to `dist/`. CI uploads that directory
as the `aiperf-architecture-atlas-dist` artifact, along with Playwright report
and diagnostic artifacts (screenshots/diffs/traces) from test runs.

Deploy `dist/` to a static host configured to return `index.html` for unknown
paths so deep links such as `/scenes/graph-ir-execution` and
`/crates/clock` resolve correctly.

The artifact is self-contained and requires no application server or runtime
data service.

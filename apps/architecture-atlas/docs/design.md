<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Architecture Atlas design

## Purpose

The Architecture Atlas is an internal, static single-page application that
explains the implemented AIPerf Rust architecture and its Python integration.
It provides three complementary experiences over one source-grounded content
model:

1. six guided routes that progressively disclose ownership, lifecycle,
   execution, data shaping, observability, and parity;
2. a unified interactive architecture graph;
3. a crate-by-crate maintainer reference.

The app describes implementation state, not design intent. Claims distinguish
built, feature-gated, runtime-conditional, compatibility-only, legacy-parallel,
and unbuilt surfaces.

## Application architecture

React and Vite produce a client-only static artifact. TanStack Router owns typed
routes and URL state for audience, filters, graph perspective, search,
selection, and presentation mode. Zod schemas validate authored content and
route state at runtime and in the build-time content validation command.

React Flow renders the unified graph. ELK computes graph layout in a dedicated
worker; a deterministic grouped fallback keeps the graph usable if worker
layout fails. Text inventories, summaries, and semantic grouping remain
available independently of the visual canvas.

The application has no service dependency and no runtime data fetch. Typed
content under `src/content/` is the single input to guided views, the atlas,
crate references, search, and integrity validation.

## Audience lenses

One persistent selector applies an audience lens across every route:

- **Executive** emphasizes ownership, product value, supported modes, migration
  state, and major risks without internal type or file-path detail.
- **Developer** explains contracts, lifecycle, communication, extension points,
  and failure behavior.
- **Maintainer** exposes exact crates, types, protocols, feature gates, parity
  scars, and external repository source evidence.

The lens changes titles, summaries, evidence density, and terminology. It is
stored in the URL and local storage so deep links remain deterministic while
the preference persists across navigation.

## Guided experience

The guided routes form a presentation sequence:

1. system ownership;
2. one-run lifecycle;
3. execution modes;
4. data and request shaping;
5. measurement and evaluation;
6. parity and migration.

Applicable routes expose mode and status filters with URL-backed state and live
result announcements. Presentation mode removes surrounding navigation,
focuses the main region, supports previous/next keyboard navigation, exits with
Escape, and restores focus to its entry control.

## Unified atlas

The atlas combines search, ownership/mode/status filters, ownership and
lifecycle layouts, a graph canvas, a semantic band inventory, and a complete
text inventory. Selecting a component writes a deep-linkable identifier to the
URL, highlights its dependency neighborhood, and opens a non-modal evidence
drawer. The drawer receives focus on entry, closes with its button or Escape,
and restores focus to the initiating control or a visible search fallback.

Source evidence is emitted only as an absolute repository URL. It never resolves
as an application-local route.

## Crate reference

Crate routes provide searchable package navigation, responsibilities,
contracts, supported modes, Cargo dependencies and reverse dependents grouped
by dependency kind, related atlas components, source paths, and parity scars.
Unknown package identifiers render a typed recovery view with links to the
directory and unified atlas.

## Content integrity

`npm run validate:content` parses the catalog and enforces referential integrity,
route coverage, source references, graph endpoints, crate relationships, and
the absence of unsupported claims. Content changes update typed records and
their tests together; specs can inform wording but never prove implementation.

## Accessibility and responsive behavior

Semantic landmarks, labelled controls, visible focus, keyboard-operable
interactions, live result/status announcements, and text alternatives to the
graph are first-class requirements. Narrow layouts convert wide flows into
ordered vertical structures. The stylesheet disables nonessential motion when
`prefers-reduced-motion: reduce` is active.

Vitest and Testing Library cover component semantics and focus behavior.
`jest-axe` audits the shell, representative guided routes, audience variants,
atlas controls/inventory/drawer, and known and unknown crate views. Playwright
runs Chromium desktop and mobile projects against the production preview and
adds browser-level axe checks.

## CI and artifact

The path-scoped Architecture Atlas workflow uses Node 22 and `npm ci`, validates
content, typechecks, lints, runs unit/accessibility tests, builds the production
SPA, and runs Playwright against `vite preview`. The workflow uploads `dist/` as
the deployable static artifact. A static host must serve `index.html` as the
fallback for client-side deep links.

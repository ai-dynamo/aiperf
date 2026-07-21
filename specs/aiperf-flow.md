<!--
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
-->

# `aiperf-flow` — component-native explainer decks

## Purpose

Replace `apps/explainers`'s bespoke `.flow` DSL, compiler pipeline (parse →
SDK-expand → lower → scene-IR → resolve → SVG scene-graph render), and
hand-rolled layout/connector/animation engines with a new app,
`apps/aiperf-flow`, where decks are authored as plain `.tsx` files using a
small set of well-known, actively-maintained React libraries. The goal is an
authoring surface an LLM already deeply understands from training data —
ordinary JSX component composition — rather than a repo-specific mini-language
that has to be learned from scratch, while off-loading the genuinely hard,
error-prone problems (arrow/connector routing between elements, layout, and
synchronized move animations) onto libraries that already solve them well,
instead of re-deriving them per-diagram or maintaining a bespoke engine for
them.

This does not extend or wrap `apps/explainers`. It is a new, separate app.
`apps/explainers` keeps working unmodified until every deck has been ported,
at which point `apps/explainers` is retired and removed.

## Built

`apps/aiperf-flow` exists: Vite + React 19 + TypeScript (strict) + Tailwind
CSS 4 (`@tailwindcss/vite`, CSS-first `@theme` config), `@xyflow/react` 12,
`motion` 12, `react-router-dom` 7, Vitest + Testing Library. Design/authoring
guidance for building on this app lives in the
`.claude/skills/aiperf-flow-diagrams/SKILL.md` skill — read it before adding
components or decks.

- **Design tokens** — `src/theme/tokens.ts` + `src/index.css`'s `@theme`
  block, ported from `apps/explainers/src/core/tokens.ts`'s NVIDIA-deck color
  roles (surface/ink/stroke/accent/category), exposed as
  `surfaceClassName`/`inkClassName`/`strokeClassName`/`accentClassName`/
  `categoryClassName` (plus `categoryBgClassName`/`categoryBgTintClassName`)
  role-to-Tailwind-class helpers. All helpers use static `Record` lookup
  tables, not runtime string interpolation — see the SKILL.md's "Tailwind
  JIT trap" section for why that distinction is load-bearing.
- **Diagram nodes** (`src/nodes/`) — `Header`, `Panel`, `Card`, `Chip` as
  React Flow custom node types (`nodeTypes.ts`), each accepting a typed
  `data` object with an optional `className` merged via `clsx`.
- **Edges** (`src/edges/`) — `FlowEdge` (`edgeTypes.ts`, `type: "flow"`), an
  animated dashed-flow custom edge respecting `prefers-reduced-motion`.
- **Layout primitives** (`src/layout/`) — `Stack`, `Row`, `Grid` (real CSS
  flex/grid, not diagram nodes) for prose/UI content outside a React Flow
  canvas.
- **Prose primitives** (`src/prose/`) — `Callout`, `Table`, `Stat`,
  `Legend`/`Swatch`.
- **State/animation** — `useReveal` (`src/reveal/`, staggered slide-entry
  reveal, restarts correctly when its input order changes mid-lifetime) and
  `useStepSimulator` (`src/state/`, Play/Pause/Next/Back/Reset over a step
  array, clamped, auto-stops at the end — the primitive for interactive
  step-through walkthroughs).
- **Deck/slide model** (`src/deck/`) — `SlideDefinition`/`DeckDefinition`
  types, a `registerDeck`/`getDeck` registry (throws on duplicate id),
  `Slide` (renders one slide's React Flow canvas with reveal-gated node
  visibility), `DeckRoute` (route-level resolution of `/:deckId`).
- **Presentation shell** (`src/shell/`) — `PresentationShell` (chapter-dot
  nav, back/next, progress label, subtitles panel, speaker-notes toggle) for
  the sequential-slide model; `PageTabs` (tab row for switching between named
  pages *within* one deck) for tabbed single-view decks.
- **Worked example deck** (`src/decks/segment-pools/`) — a full, reviewed
  port of `docs/canvases/segment-pools-and-body-plans.canvas.tsx` (a real,
  hand-authored Cursor Canvas), proving the stack end-to-end: three pages as
  real React Flow node/edge graphs (`OverviewPage`, `PrefixPage`,
  `DispatchPage`), a live interning step-simulator (`PoolPage`), a stateful
  domain selector (`PayloadsPage`), and a multi-toggle materializer form
  (`BodyPlanPage`), composed via `SegmentPoolsDeck` + `PageTabs`, routed at
  `/segment-pools`.

The following `apps/explainers` assets remain reference material for the
still-pending 15-deck `.flow` migration (see "Migration" below), not
prerequisites `apps/aiperf-flow` depends on at runtime:

- `src/core/ExplainerShell.tsx` — the original presentation-chrome
  implementation `PresentationShell` was behaviorally ported from.
- `decks-flow/*.flow` (15 decks, ~340 scenes) — the content still to port.
  Each deck's narration/eyebrow/title/lede/points/caption text and visual
  composition is the source of truth for what the ported `.tsx` deck must
  reproduce; the `.flow` syntax itself is discarded.

## Future requirements

### Additional component gaps

A survey of 28 of the user's real, hand-authored Cursor Canvas files
(`.canvas.tsx`) identified further gaps beyond what's built, ranked by
frequency of use in that corpus: a page/tab-level "open source file" host
action (may not apply — `aiperf-flow` isn't IDE-embedded, unlike Cursor
Canvas), auto-DAG-layout helper (lower priority — hand-placed node positions
are at least as common in real usage even where auto-layout is available),
`Divider`/`Spacer`, collapsible `Card` sections, and a syntax-highlighted
`Code` block (currently substituted with a plain monospace `<pre>`, adequate
fidelity so far). None of these block current work; add as needed.

### Migration

All 15 `apps/explainers/decks-flow/*.flow` decks (~340 scenes) are rewritten
as `.tsx` decks under `apps/aiperf-flow`. This is a content-preserving
rewrite: same narration/copy, same visual composition and information density
per slide, same reveal choreography — expressed in the new component
vocabulary instead of `.flow` syntax. Given the scale, this is planned and
executed as its own implementation plan via subagent-driven-development, one
subagent per deck (or per few slides for the two ~50–63-slide decks),
verified by visual comparison against `apps/explainers`'s current rendering of
each slide (screenshot diff) rather than a numeric/geometric equivalence
check, since the new engine computes layout differently by design.

`apps/explainers` is retired (directory removed, build/deploy wiring updated
to point at `apps/aiperf-flow`) once every deck is ported and visually
verified equivalent. Until then, both apps coexist and `apps/explainers`
keeps shipping unchanged.

### What is explicitly dropped

- The `.flow` grammar/parser/compiler (`src/flow/compiler/`,
  `src/flow/language/`, `scripts/compile-decks.ts`).
- The custom SVG scene-graph renderer (`SceneRenderer.tsx`) and scene
  resolution pipeline (`src/core/diagram/resolution/`).
- The hand-rolled flow-engine (`src/core/diagram/layout/flow-engine.ts`) and
  the older `managedBounds`-based layout system in
  `src/core/diagram/capabilities/layout.ts` — both superseded by real CSS
  (Tailwind) plus React Flow's own node positioning.
- `scripts/flow-verifier.mjs`'s IR-based overflow/overlap checks — replaced by
  whatever equivalent visual-regression tooling `apps/aiperf-flow` adopts
  (out of scope for this record; a follow-on decision once the new app's
  screenshot/export tooling is built).

## Source anchors

- `.claude/skills/aiperf-flow-diagrams/SKILL.md` — authoring guidance,
  component vocabulary reference, design rules.
- `apps/aiperf-flow/src/theme/tokens.ts`, `src/index.css` (design tokens).
- `apps/aiperf-flow/src/nodes/`, `src/edges/` (diagram node/edge types).
- `apps/aiperf-flow/src/layout/`, `src/prose/` (layout/prose primitives).
- `apps/aiperf-flow/src/reveal/`, `src/state/` (animation/state primitives).
- `apps/aiperf-flow/src/deck/`, `src/shell/` (deck model, presentation
  shell, page-tab navigation).
- `apps/aiperf-flow/src/decks/segment-pools/` (worked example: a full port
  of `docs/canvases/segment-pools-and-body-plans.canvas.tsx`).
- `apps/explainers/src/core/ExplainerShell.tsx` (presentation-shell port
  source).
- `apps/explainers/decks-flow/*.flow` (remaining content-migration source,
  15 decks).

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

Nothing under `apps/aiperf-flow` exists yet. The following current
`apps/explainers` assets are reference material / migration sources, not
prerequisites the new app depends on at runtime:

- `src/core/tokens.ts` — the NVIDIA-deck visual design tokens (flat white
  slides, NVIDIA green accent, boxy corners, text/surface/accent color roles).
  These become a Tailwind theme config, not TypeScript constants.
- `src/core/ExplainerShell.tsx` and friends — the presentation chrome (chapter
  nav, play/pause, speed control, progress bar, subtitles panel, speaker
  notes, "Present" fullscreen mode). The chrome's *behavior* is worth keeping;
  its implementation (bound to the current scene-IR/timeline model) is not.
- `decks-flow/*.flow` (15 decks, ~340 scenes) — the content to port, not code
  to reuse. Each deck's narration/eyebrow/title/lede/points/caption text and
  its visual composition are the source of truth for what the ported `.tsx`
  deck must reproduce; the `.flow` syntax itself is discarded.

## Future requirements

### Stack

- **Vite** — build tool (already the toolchain `apps/explainers` uses; no
  change here).
- **Tailwind CSS** — all styling. The current `tokens.ts` color roles become a
  Tailwind theme extension (`tailwind.config.ts`) so `bg-surface-elevated`,
  `text-ink-secondary`, `border-accent-primary`, etc. carry the same meanings
  decks already rely on today.
- **[React Flow / `@xyflow/react`](https://reactflow.dev/)** — node/edge
  diagrams. Every box a deck places (panel, chip, card, header, etc.) is a
  React Flow **node** (a real React component, Tailwind-styled); every
  arrow/connector between boxes is a React Flow **edge**. React Flow owns
  measuring node positions and keeping edges visually attached as nodes move
  or the canvas pans/zooms — the exact problem that caused this session's
  layout bugs when solved by hand.
- **[Motion for React](https://motion.dev/docs/react-layout-animations)**
  (formerly Framer Motion, same API) — all animation. `layout` on a `motion`
  component auto-animates any position/size change from *any* cause (reflow,
  conditional render, prop change) using FLIP under the hood. `layoutId`
  gives shared-element transitions: a node that exists in two different
  arrangements within one scene (the "move between locations" requirement)
  animates automatically between them when the arrangement changes — no
  hand-authored before/after snapshot bookkeeping, no custom interpolation
  code in the render path.
- Real CSS (via Tailwind utility classes, plus React Flow's own internal
  positioning) does all box layout. No custom flex/grid-clone layout engine.

### Authoring model

Decks are `.tsx` modules under `apps/aiperf-flow/src/decks/`, one file (or one
small folder) per deck, default-exporting a deck definition consumed by the
app shell. There is no compiler, no IR, no `.flow` grammar, no SDK-expansion
step — a deck *is* its rendered component tree.

A small, reusable component/hook vocabulary carries the conventions today's
`.flow` decks rely on (eyebrow/title/lede/points/caption authoring, reveal
timing, narration/subtitles, chapter structure) as ordinary composable React,
not a parsed language:

- `<Deck>` / `<Slide>` — top-level structure; a `<Slide>` carries the
  eyebrow/title/lede/narration/caption text props the presentation chrome
  reads (chapter nav, subtitles panel, speaker notes).
- `<Panel>`, `<Header>`, `<Chip>`, `<Card>`, `<BigStat>`, etc. — Tailwind-styled
  React Flow custom node components, one per visual primitive the current SDK
  catalog exposes today. These are a direct port of `chrome.ts`/`catalog.ts`'s
  visual vocabulary, re-implemented as React Flow nodes instead of scene-IR
  factories.
- Edges are authored via React Flow's own `edges` array / `<ReactFlow edges=.../>`
  prop, referencing node `id`s — the same anchor-relative addressing
  (`source`, `target`, `sourceHandle`, `targetHandle`) `.flow` connectors
  already use today, just expressed as React Flow's native vocabulary instead
  of a custom `from = { nodeId, anchor }` shape.
- `useReveal(order)` / `<Reveal at={0}>` — a thin hook/component pair over
  Motion's `AnimatePresence`/`variants` that staggers a slide's content in on
  entry, replacing `.flow`'s `timeline { reveal ... }` blocks. Still plain
  React — a deck author (human or AI) composes these directly in JSX, there is
  nothing to parse.
- Moves between two arrangements of the same node set use `layoutId` directly;
  no dedicated `<Move>` abstraction is needed since Motion already owns this.

### Presentation shell

Port `ExplainerShell`'s behavior (chapter navigation, play/pause, playback
speed, progress bar, subtitles, speaker notes, present/fullscreen mode) as new
`apps/aiperf-flow` components reading the `<Deck>`/`<Slide>` prop data instead
of resolved scene IR. This is a straightforward port, not a redesign — the UX
is already correct.

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

- `apps/explainers/src/core/tokens.ts` (design-token migration source).
- `apps/explainers/src/core/ExplainerShell.tsx` (presentation-shell port
  source).
- `apps/explainers/decks-flow/*.flow` (content-migration source, 15 decks).
- `apps/aiperf-flow/` (new app root, not yet created).

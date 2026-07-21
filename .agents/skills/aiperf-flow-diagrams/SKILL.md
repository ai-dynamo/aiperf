<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
---
name: aiperf-flow-diagrams
description: >-
  Author interactive explainer decks/diagrams in `apps/aiperf-flow` — plain
  `.tsx`, no custom DSL. Use whenever a task asks for an "explainer", "deck",
  "diagram", "canvas-style page", "interactive walkthrough", or "visual
  explanation" of an AIPerf subsystem, or mentions `apps/aiperf-flow`,
  `docs/canvases/*.canvas.tsx`, React Flow nodes/edges, or porting a Cursor
  canvas. Read this BEFORE writing any file under `apps/aiperf-flow/src/`.
  Covers the full component vocabulary (diagram nodes, prose primitives,
  layout, animation, state), design rules, and a worked example.
---

# Authoring `apps/aiperf-flow` decks

`apps/aiperf-flow` is a from-scratch explainer/diagram app: Vite + React 19 +
TypeScript + Tailwind CSS 4 + `@xyflow/react` (React Flow) + `motion`. A deck
*is* its rendered component tree — there is no `.flow` DSL, no compiler, no
scene-graph IR. If you find yourself computing pixel positions by hand for
anything other than a React Flow node's *initial* declared `position` hint
(which React Flow may then reflow), stop — that's the exact pattern this app
exists to avoid.

## Before you write anything

1. Read `apps/aiperf-flow/src/theme/tokens.ts` for the full token surface
   (`SurfaceRole`, `InkRole`, `StrokeRole`, `AccentRole`, `CategoryRole` and
   their `*ClassName()` helpers) — this is the only source of color/style.
2. Read the `.tsx` source of whichever components you're about to use (listed
   below) for their exact prop types — don't guess a prop shape, the compiler
   will catch it but reading first is faster.
3. If porting a real Cursor canvas (`docs/canvases/*.canvas.tsx`), read the
   source canvas in full first. It's the content ground truth — labels,
   copy, data, structure all come from there, not invented.

## Component vocabulary

### Diagram nodes (React Flow custom node types, `src/nodes/`)

Boxes in a diagram are React Flow nodes. Register via
`import { nodeTypes } from "../../nodes/nodeTypes.js"` and pass to
`<ReactFlow nodeTypes={nodeTypes} ...>`.

- `Header` (`type: "header"`) — `{ title, caption?, surfaceRole?, className? }`
- `Panel` (`type: "panel"`) — `{ title, detail?, surfaceRole?, strokeRole?, className? }`, has `Handle`s (target left, source right)
- `Card` (`type: "card"`) — `{ title, subtitle?, detail?, strokeRole?, className? }`, has `Handle`s
- `Chip` (`type: "chip"`) — `{ label, strokeRole?, className? }`, no handles (not connectable)

Every node's `data` accepts an optional `className`, merged via `clsx` onto
the component's own classes (yours appended last, never silently dropped).

### The one-`ReactFlowProvider`-per-`<ReactFlow>` trap

Every `<ReactFlow>` instance needs its **own** ancestor `<ReactFlowProvider>`.
If a page renders more than one `<ReactFlow>` (e.g. two diagrams side by side
in a `Grid`) and they share a single outer `<ReactFlowProvider>`, they
silently collide onto the same internal store — only the **last-mounted**
diagram's nodes actually render; the others render an empty canvas with no
error. This hit two independent deck ports in the same session before being
fixed. Wrap each diagram's own `<ReactFlow>` in its own local
`<ReactFlowProvider>` inside whatever shared "diagram frame" component you
build (see `src/decks/rust-aiperf-architecture/shared.tsx`'s `DeckDiagram` for
the reference pattern) — never rely on one page-level provider covering
multiple `<ReactFlow>` instances. A test rendering only one page's diagram in
isolation will not catch this; it only surfaces when a page composes more
than one diagram together, so check pages with multiple `DeckDiagram`s (or
equivalent) specifically.

### Edges (`src/edges/`)

`import { edgeTypes } from "../../edges/edgeTypes.js"`, pass to
`<ReactFlow edgeTypes={edgeTypes} ...>`. Use `type: "flow"` for an edge that
should show data/request movement — it renders as an animated dashed line
(`FlowEdge`, `data?: { color?: string; speed?: "slow"|"normal"|"fast" }`,
respects `prefers-reduced-motion`). Use React Flow's default `type` (omit
`type`) for a plain static connector.

### Prose/layout primitives (`src/layout/`, `src/prose/`)

For content *outside* a React Flow canvas — intro text, forms, stat rows,
comparison tables. Real CSS flex/grid, not diagram nodes.

- `Stack` (`src/layout/Stack.tsx`) — vertical flex column, `{ children, gap?, className? }`
- `Row` (`src/layout/Row.tsx`) — horizontal flex row, `{ children, gap?, align?, justify?, wrap?, className? }`
- `Grid` (`src/layout/Grid.tsx`) — CSS grid, `{ children, columns: number(1-12) | string, gap?, align?, className? }` — a static lookup table maps 1-12 to real `grid-cols-N` classes; a string falls through to inline `gridTemplateColumns`. **Never** build `` `grid-cols-${n}` `` yourself — see "The Tailwind JIT trap" below.
- `Callout` (`src/prose/Callout.tsx`) — `{ tone?: "info"|"warning"|"danger"|"success"|"neutral", title?, children, className? }`
- `Table` (`src/prose/Table.tsx`) — real `<table>`, `{ columns: {key,label,align?}[], rows: (Record<string,ReactNode> & {tone?: "neutral"|"success"|"warning"|"danger"})[], className? }`
- `Stat` (`src/prose/Stat.tsx`) — KPI tile, `{ label, value, trend?, tone?: "neutral"|"positive"|"negative", className? }`
- `Legend` / `Swatch` (`src/prose/Legend.tsx`, `Swatch.tsx`) — color-key rows, `{ entries: { color: CategoryRole; label: string }[] }`
- `Pill` (`src/prose/Pill.tsx`) — compact tag/status chip, `{ children, active?: boolean, tone?: CategoryRole, onClick?: () => void, ariaLabel?: string, className? }`. Renders a `<span>`, or a `<button>` with `aria-pressed={active}` when `onClick` is given. `tone` colors it by `CategoryRole` (ports a source canvas's colored status tag); omit `tone` for the plain neutral/active-accent chip (a selected filter, a status tag, a source-file badge). Pass `ariaLabel` when the visible text alone doesn't convey the chip's meaning (e.g. a red "Rejected" chip whose accessible name should say what was rejected). **Use this instead of writing a local `Pill`/`Badge`/`Tag`/`TonePill` component** — four independent deck ports each built the identical shape from scratch before this was consolidated; check here first.
- `Eyebrow` (`src/prose/Eyebrow.tsx`) — small uppercase, letter-spaced label, `{ children, tone?: CategoryRole, className? }`. Defaults to tertiary ink; pass `tone` for a colored status word (a "Built"/"Rejected" kicker, a category tag). This is the `text-xs font-semibold uppercase tracking-wide` span pattern — it was independently hand-rolled 17+ times across ported decks before being consolidated here. **Reach for this before writing that span by hand** — it's for section kickers, field labels ("Symbol", "Source"), and short status words, not for anything clickable or chip-shaped (that's `Pill`).
- `Framed` (`src/prose/Framed.tsx`) — soft-bordered content panel for grouping prose without the weight of a `Callout` or diagram `Card`, `{ children, surfaceRole?: SurfaceRole, className? }` (defaults to the `"page"` surface role).

Note `Callout`/`Table`/`Stat` use three *different* tone vocabularies
(`Callout`: info/warning/danger/success/neutral; `Table`: neutral/success/warning/danger;
`Stat`: neutral/positive/negative) — each is individually correct for what it
expresses (severity vs. direction), but don't assume they're interchangeable.
`Callout`'s `neutral` tone (gray category) exists for an admonition that isn't
severity-graded — don't approximate it with `info`/`warning`/`success` anymore.

### State and animation

- `useReveal(order: string[], opts?: {stepMs?})` (`src/reveal/useReveal.ts`) — reveals node ids one at a time on a timer, for staggered slide-entry. Restarts correctly if `order` changes mid-lifetime.
- `useStepSimulator<T>(steps: T[], opts?: {autoPlayMs?})` (`src/state/useStepSimulator.ts`) — Play/Pause/Next/Back/Reset over a step array, clamped, auto-stops at the end. **This is the primitive for any "click through to see X happen" interactive walkthrough.** Don't write `while (!sim.isLast) sim.next()` — `next()` schedules a state update, it doesn't mutate synchronously, so that spins forever. Loop a bounded number of times instead (see `src/decks/segment-pools/PoolPage.tsx`'s `simulatePoolInterning`/"Run all" for the reference pattern).
- Motion's `layout`/`layoutId` (from `motion/react`, not yet used anywhere in this codebase but available) — for animating a node's position/size change or a shared-element transition between two arrangements. Reach for this before writing any custom interpolation code.

### Multi-page decks (`src/shell/`)

- `PageTabs<T extends string>` (`src/shell/PageTabs.tsx`) — tab row for switching between named pages *within* one deck (distinct from slide-to-slide navigation). `{ pages: {id:T,label:string}[], current: T, onChange: (id:T)=>void, className? }`. See `src/decks/segment-pools/SegmentPoolsDeck.tsx` for the composition pattern: `useState` for the current page id, conditional render of one page component per tab.
- `PresentationShell` (`src/shell/PresentationShell.tsx`) — slide-to-slide chrome (chapter dots, back/next, progress, subtitles, speaker notes) for the sequential-slide deck model (`src/deck/{types,Slide,registry,DeckRoute}.ts`). Use this for a conventional slide deck; use `PageTabs` for a tabbed single-view deck like a ported Cursor canvas.

## Design rules

Soft and elevated, purposeful — a modern SaaS-dashboard feel on the app's dark
charcoal-grey/green palette. (Earlier revisions of this app used a strictly flat/boxy
`rounded-none`, no-shadow language; that rule was deliberately retired in favor of the
one below — don't reintroduce `rounded-none` on new work.)

- **Radius scale**: `rounded-md` for compact elements (`Pill`, `Chip`, `Button`, form
  controls like `Toggle`/`Select`). `rounded-lg` for standard containers (`Panel`, `Card`,
  `Callout`, `Table`'s wrapper, `Framed`, deck content cards). `rounded-xl` for large
  top-level surfaces (e.g. Home's deck cards). `PageTabs`' pill row stays `rounded-full`.
  Never `rounded-none` — use the smallest of these scales that isn't obviously wrong
  before inventing a new radius value.
- **Shadow scale**: `shadow-sm` as the default resting elevation on any `"elevated"`/
  `"panel"` surface (`Card`, `Panel`, `Callout`, `Table`, `Framed`) — Tailwind's default
  black-based shadow reads as a soft depth cue on this dark palette without a custom
  shadow color. `shadow-md` on hover/focus or otherwise emphasized states (an existing
  `hover:` class list, a selected tab, a clicked `Pill`). Add the shadow alongside the
  existing border, don't replace one with the other — depth comes from both together.
- **Still no gradients, no emojis as decoration.** Rounded corners and shadows are the
  modernization; gradients weren't asked for and would be scope creep.
- **Colors only from `theme/tokens.ts` role helpers.** Never a raw hex value, never an undefined Tailwind class.
- **Don't wrap everything in a bordered box.** Mix open sections with `Card`/`Panel`-bordered ones.
- **Real content only.** Never render a placeholder ("TODO", "Add content here", an empty table). If you don't have the real data/copy, say so and ask, don't fabricate it.

### The Tailwind JIT trap (read this before writing any dynamic class name)

Tailwind's compiler only emits CSS for class names that appear as **complete
literal strings** in source. A runtime-interpolated template string like
`` `bg-category-${color}` `` is invisible to it — the class silently never
makes it into the compiled CSS, and the bug only shows up in a *production
build*, not in tests (which don't run the real Tailwind scanner) and not even
reliably in dev (classes can coincidentally appear elsewhere in source, e.g.
a test file, and "work" by accident until that test is edited). This bug hit
this codebase three times in one session (`Swatch.tsx`, `Table.tsx`,
`Callout.tsx`, `BodyPlanPage.tsx`) before being fixed with static lookup
tables. If you need a class name that depends on a runtime value:

```ts
// WRONG — invisible to Tailwind's scanner, silently drops colors in prod
className={`bg-category-${color}`}

// RIGHT — every possible value is a literal string in source
const CATEGORY_BG_CLASSES: Record<CategoryRole, string> = {
  green: "bg-category-green", yellow: "bg-category-yellow", /* ...all 8 */
};
className={CATEGORY_BG_CLASSES[color]}
```

`theme/tokens.ts` already has this pattern built for category colors —
`categoryClassName()`, `categoryBgClassName()`, `categoryBgTintClassName()`.
Use those instead of writing a new lookup table when a category color is
what you need. `layout/Grid.tsx`'s `GRID_COLS_CLASSES` is the same pattern
for grid columns.

### SVG shapes need `fill-*`/`stroke-*` classes, not `bg-*`/`border-*`

Several decks hand-draw custom SVG charts (`aiperf-metrics-accumulator`'s
sweep-line Gantt, `weka-timing-transforms-interactive`'s swimlane views,
`prose/Chart.tsx`). A `<rect>`/`<path>`/`<circle>`/`<polygon>`/`<ellipse>`/
`<polyline>` is painted by the CSS `fill`/`stroke` properties — `background-
color`/`border-color` (what `categoryBgClassName()`/`surfaceClassName()`/
`strokeClassName()` emit) have **no effect** on it. Two ways this shipped as
a real bug (not just a Tailwind-JIT purge) before being caught:

1. Applying `categoryBgClassName(color)` (or `surfaceClassName(...)`) as an
   SVG shape's `className` with no `fill`/`stroke` attribute at all — the
   shape falls back to SVG's initial `fill: black`, rendering solid black
   regardless of the intended color.
2. Setting `fill="currentColor"` (or `stroke="currentColor"`) explicitly,
   but pairing it with a `bg-*`/`border-*` class instead of a `text-*`
   class — `currentColor` resolves the CSS `color` property, which
   `background-color`/`border-color` classes never set, so the shape
   inherits whatever `color` its ancestor happens to have (often the wrong
   value, or a washed-out default) instead of the intended category color.

Use `categoryFillClassName(role)`/`categoryStrokeClassName(role)` from
`theme/tokens.ts` instead — these emit real `fill-category-*`/`stroke-
category-*` utility classes that set `fill`/`stroke` directly, so no
`fill="currentColor"`/`stroke="currentColor"` attribute is needed at all:

```tsx
// WRONG — background-color has no effect on SVG; renders solid black
<rect className={categoryBgClassName("blue")} />

// WRONG — sets `color`, but fill="currentColor" needs a text-* class here,
// and even then it's an indirection categoryFillClassName avoids entirely
<rect fill="currentColor" className={categoryClassName("blue")} />

// RIGHT — fill-category-blue sets the `fill` CSS property directly
<rect className={categoryFillClassName("blue")} />
```

For a non-category (role-based) stroke on a plain SVG line/rect divider,
`strokeClassName()`/`surfaceClassName()` are equally inert — use
`inkClassName(role)` with an explicit `stroke="currentColor"`/
`fill="currentColor"` attribute instead (there's no ink-based `fill-*`/
`stroke-*` helper, since `InkRole`/`StrokeRole` aren't `CategoryRole`).
This bug class shipped three times before being traced to this root cause
(`Chart.tsx`, `AiperfMetricsAccumulatorDeck.tsx`'s sweep-line chart,
`TStarChop.tsx`/`Timeline.tsx`/`CombinedTimeline.tsx`'s lane boxes) —
whenever you touch a hand-drawn SVG element's color, check this section.

## Pre-delivery self-check

Before considering a deck/component done:

1. Does it use real React Flow (`nodes`/`edges` with `position` hints) for
   diagram content, and real `Stack`/`Row`/`Grid` for prose layout — no hand
   computed pixel positions outside React Flow's node `position` field?
2. Any dynamic class name? Check it against a static lookup table, not a
   template-string interpolation (see above).
3. Any color, spacing, or typography value not traceable to `theme/tokens.ts`
   or an existing component's established scale?
4. Any `rounded-none`, or a bordered box with no `shadow-*`? Apply the
   radius/shadow scale above instead — see "Design rules".
5. Run `cd apps/aiperf-flow && npm test && npm run build` — both must be
   clean. `npm test` alone does not prove the Tailwind-JIT bug is absent; if
   you touched any dynamic class name, also grep `dist/assets/*.css` after
   `npm run build` for the literal class strings you expect to see.
6. TDD: was the test written before the implementation, and does it assert
   real rendered content/behavior (not just "renders without crashing")?
7. Does any page render more than one `<ReactFlow>`? If so, confirm each one
   has its own `<ReactFlowProvider>` (see the trap above), and that the
   page's test asserts on content from *every* diagram on the page, not just
   the first — a shared-provider collision only shows up when you check the
   diagrams other than the last-mounted one.
8. Wrote a new `Pill`/`Badge`/`Tag`/status-chip? Check `src/prose/Pill.tsx`
   first — it already covers plain, active/toggle, clickable, and
   category-tone-colored variants. Wrote an uppercase/tracking-wide label
   span (a section kicker, a field label, a status word)? Check
   `src/prose/Eyebrow.tsx` first, likewise.
9. Colored a hand-drawn `<rect>`/`<path>`/`<circle>`/etc.? Used
   `categoryFillClassName()`/`categoryStrokeClassName()` (or `inkClassName()`
   with an explicit `fill`/`stroke="currentColor"` attribute), never
   `categoryBgClassName()`/`surfaceClassName()`/`strokeClassName()` — see
   "SVG shapes need fill-/stroke- classes" above.

## Worked example

`apps/aiperf-flow/src/decks/segment-pools/` is a complete, reviewed reference
example — six pages ported from a real, hand-authored Cursor canvas
(`docs/canvases/segment-pools-and-body-plans.canvas.tsx`), composed via
`SegmentPoolsDeck.tsx` + `PageTabs`:

- `OverviewPage.tsx`, `PrefixPage.tsx`, `DispatchPage.tsx` — real React Flow
  node/edge diagrams (the canvas's hand-drawn SVG pages, re-authored as
  actual node/edge graphs).
- `PoolPage.tsx` — a live step-through simulator (`useStepSimulator` +
  `Table`/`Stat`/`Callout`), including the correct bounded-loop pattern for
  a "Run all" button.
- `PayloadsPage.tsx` — a stateful selector (plain `useState`, `Row`/`Grid` +
  `Swatch`) swapping displayed content on click.
- `BodyPlanPage.tsx` — a multi-toggle form recomputing derived output live.

Read any of these before building something similar — they're the highest-
fidelity example of "how this vocabulary composes" in the codebase.

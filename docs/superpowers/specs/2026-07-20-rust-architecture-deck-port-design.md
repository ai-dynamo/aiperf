# Rust Architecture Deck Port — Design

## Purpose

Port the reference NVIDIA-branded HTML deck (mirrored at
`~/projects/dl/site-mirror`, extracted to `/tmp/deck.html`, template name
"NVIDIA Deck", 49 unique slides describing the AIPerf Rust runtime
architecture) into a fully native `apps/explainers` `.flow` deck with ~95%
pixel parity: matching text, numbers, colors, and layout geometry.

Every visual in the ported deck — including the deck's several bespoke
illustrations (a real-clock timeline with target/actual markers, a
BinaryHeap pop-order tree, a numbered request-lifecycle callback sequence,
a loop-state diagram, chapter dividers, N-column comparison grids, card
grids) — must be expressed through composable, reusable `sdk.*` components
authored declaratively in the `.flow` file. No literal inline SVG `<path>`
data, no one-off hand-positioned freeform drawing: every shape is a named
component instance with props, so the same visual concept is trivially
reusable in future decks.

## Source of truth

- Reference markup (read-only, not committed to this repo): `/tmp/deck.html`
  (regenerable from `~/projects/dl/site-mirror/claude.ai/design/anthropic.omelette.api.v1alpha.OmeletteService/GetFile`,
  the `<!DOCTYPE html>` payload after its 4-byte BOM prefix).
- Existing SDK conventions: `apps/explainers/src/flow/sdk/generic/{chrome,composites,layout,topology,motion,catalog}.ts`,
  `apps/explainers/src/flow/sdk/diagram/catalog.ts`, `apps/explainers/src/flow/sdk/registry.ts`.
- Existing worked examples of every current primitive:
  `apps/explainers/decks-flow/sdk-generic-catalog.flow`,
  `apps/explainers/decks-flow/sdk-diagram-catalog.flow`.

## Target deck

New file `apps/explainers/decks-flow/rust-architecture-deck-port.flow`:
`id: "rust-architecture-deck-port"`, route `/rust-architecture-deck-port`,
standalone from the existing `rust-architecture`/`rust-architecture-atlas`
explainers (different pacing: a presentation port, not a narrated
walkthrough — still uses the shared `explainer { slide { ... } }` DSL and
narration/lede/points convention every other deck in this repo uses, since
the app has no non-narrated slide mode).

49 slides, grouped into 7 chapters matching the HTML deck's own dividers:

1. **Open** — Cover, Thesis, Orientation (3)
2. **Two Seams** — Divider 01, Seams overview, Clock, Drivers, Transport seam, Observer sequence, Three modes (7)
3. **Crate Topology & Flow** — Divider 02, Crate topology, Module universe, Flow diagram, Failure funnel (5)
4. **Component Reference A–H** — Divider 03, A·Process boundary, A·Coordinator pipeline, B·Input resolution, C·Execution paths, D·HTTP transport, D·HTTP internals, E·gRPC transport, F·Endpoints, G·Dataset pipeline, G·Segment store, G·Pre-serialization (11)
5. **Component Reference H–M** — G·Dispatch materialization, H·RNG substrate, I·Timing and scheduling, I·Phase lifecycle, J·Graph-IR engine, J·Lowering and execution, J·Agentic replay, K·Metrics core (8)
6. **Component Reference K–R** — K·RaggedSeries, L·Reporting and export, M·Adaptive scale, N·Accuracy, O·Side-channel telemetry, P·Cellular, Q·Dynosim, R·Mock server (8)
7. **Closing** — Divider 04, Config catalog, Everything is a trait, Invariants, System map, Closing (6)

(3 + 7 + 5 + 11 + 8 + 8 + 6 = 49.)

## New native SDK components

Every recurring visual pattern in the HTML deck that has no existing
`sdk.*` equivalent becomes one new generic-family composite, added to
`apps/explainers/src/flow/sdk/generic/composites.ts` (or a new
`generic/deck-composites.ts` module if `composites.ts` would grow past
~1500 lines — follow the existing file-size judgment call already made in
this codebase) with full descriptor + factory + unit tests, registered in
`src/flow/sdk/registry.ts`, and demonstrated with a new chapter appended to
`decks-flow/sdk-generic-catalog.flow` (keeping that catalog's own
"every registered primitive gets a teaching slide" invariant intact,
enforced by `npm run assert:sdk-authoring` / the flow-verifier).

1. **`sdk.sectionDivider`** — chapter-break slide body: big mono chapter
   number (`number` prop), `title`, `subtitle`, optional `eyebrow`.
   Covers the deck's 4 "Divider NN" slides.
2. **`sdk.stepChain`** — a `direction: "row" | "column"` chain of numbered
   steps (`steps: [{ number, label, detail }]`), each step a bordered box
   with a mono numeral kicker, connected by arrows. Covers Orientation's
   6-step pipeline and Flow diagram's vertical Python→stdout chain.
3. **`sdk.bigStat`** — a giant centered number (`value`) with wrapped
   supporting `title`/`description` text. Covers "Three modes"' huge `3`.
4. **`sdk.compareGrid`** — an `N`-column grid of top-accent-bordered items
   (`items: [{ label, detail }]`), each with a colored top border. Covers
   Thesis's 3-column takeaways, Failure funnel's 3-column stat row, and the
   many "grid of short facts" panels across the component-reference
   chapters.
5. **`sdk.numberedSequence`** — a vertical list of numbered rows
   (`items: [{ number, title, detail, emphasis? }]`), each with a colored
   square index chip (alternating fill per `emphasis`). Covers Observer
   sequence's 6-callback list.
6. **`sdk.timelineAxis`** — a horizontal axis (`start`, `end`, `unit`,
   `ticks: [{ at, label }]`, `markers: [{ at, label, style }]`, optional
   `target: { at, label }` dashed reference line). Covers the Clock
   slide's RealClock timerfd-vs-wheel diagram.
7. **`sdk.nodeTree`** — a small root/children box-and-line diagram
   (`root: { label, detail }`, `children: [{ label, detail, emphasis? }]`,
   optional `order` annotation text). Covers the Clock slide's SimClock
   BinaryHeap pop-order diagram, and is generically reusable for any small
   hierarchical illustration future decks need.
8. **`sdk.cardGrid`** — a responsive grid of titled bordered mini-cards
   (`columns`, `cards: [{ title, detail, accent? }]`). Covers Crate
   topology's 4-card grid and most component-reference "N boxes side by
   side" panels.

The deck's loop/state-cycle diagram (Drivers slide's `drive_sim` idle-pump
loop) and any directed-graph illustrations reuse the **existing**
`diagram` SDK family (`sdk.ProcessStep`, `sdk.Decision`, `sdk.Merge`,
`sdk.Retry`, `sdk.Loop`, `sdk.Edge`) already registered in
`src/flow/sdk/diagram/catalog.ts` — these are already declarative,
non-hand-drawn primitives, so no new component is needed there.

Every other visual element in the deck (headers, titles, paragraphs, key
values, property lists, badges, tags, tables, breadcrumbs, code blocks) has
a direct existing `sdk.*` equivalent already proven in
`sdk-generic-catalog.flow` and is reused as-is.

## Parity method

"~95% pixel parity" means: for each slide, the ported `.flow` scene's
authored text content, numeric values, and color roles match the HTML
source slide's rendered content and NVIDIA-green/black/white palette
exactly; layout *position* and *size* match within visual tolerance (same
relative geometry — header top, hero mid-stage, footer bottom — not
pixel-identical `x`/`y` coordinates, since the `.flow` 1280×720 canvas and
component padding/line-height differ slightly from the HTML deck's literal
`px` values). Verification per slide: a Playwright screenshot of the
rendered `.flow` slide compared side-by-side against the corresponding
`/tmp/deck.html` slide (manual visual diff per chapter, described in each
sub-plan's verification task — no automated pixel-diff tool is introduced
by this effort).

## Built

Not yet — this spec accompanies the implementation plan at
`docs/superpowers/plans/2026-07-20-rust-architecture-deck-port.md`.

## Source anchors

- `apps/explainers/decks-flow/rust-architecture-deck-port.flow` (new)
- `apps/explainers/src/flow/sdk/generic/composites.ts` (extended) or a new
  `apps/explainers/src/flow/sdk/generic/deck-composites.ts`
- `apps/explainers/src/flow/sdk/registry.ts` (registration)
- `apps/explainers/decks-flow/sdk-generic-catalog.flow` (new teaching chapter)

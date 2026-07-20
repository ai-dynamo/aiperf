# Rust Architecture Deck Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port all 49 slides of the reference NVIDIA HTML deck (`/tmp/deck.html`) into a new native `apps/explainers` `.flow` deck, `decks-flow/rust-architecture-deck-port.flow`, with ~95% pixel parity, using 8 new reusable `sdk.*` composite components for every visual pattern that has no existing SDK equivalent — no hand-drawn inline SVG paths anywhere.

**Architecture:** Two layers. (1) A component layer: 8 new generic-family SDK composites added to `apps/explainers/src/flow/sdk/generic/deck-composites.ts` (new file, mirroring the existing factory pattern in `generic/composites.ts`), registered in `src/flow/sdk/registry.ts`, and each demonstrated with one teaching slide appended to `decks-flow/sdk-generic-catalog.flow`. (2) A content layer: the 49-slide deck itself, authored in `.flow` DSL against those new components plus the ~30 existing generic/diagram components already proven in `sdk-generic-catalog.flow` / `sdk-diagram-catalog.flow`.

**Tech Stack:** TypeScript SDK component factories (existing `SdkComponentFactory`/`ComponentDescriptor` contract), the `.flow` authoring DSL, Vite/Vitest, Playwright for visual verification.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-20-rust-architecture-deck-port-design.md` — read it first; it has the full 7-chapter slide inventory and component list with exact prop shapes.
- Reference source: `/tmp/deck.html` (already extracted from the site-mirror; if missing, regenerate per the spec's "Source of truth" section). Every slide's exact text, numbers, and hex colors must be read from this file, not paraphrased from memory.
- Zero hand-drawn SVG: no literal `<path d="...">`-equivalent freeform drawing in any `.flow` scene or component factory. Every visual is a named `sdk.*` component instance with typed props (`sdk.Shape`, `sdk.Line`, `sdk.Arrow`, or one of the 8 new composites below) — this mirrors how every existing deck in this repo (`sdk-generic-catalog.flow`, `sdk-diagram-catalog.flow`, `rust-architecture.flow`) is already authored.
- Palette: white `#fff` slide background, black `#000`/`#111` text and strong borders, NVIDIA green `#76B900` accent, `#E4E4E4` thin borders, `#F2F7EA` soft green tint, `#A7A7A7`/`#999` secondary text, Roboto Mono for kicker/mono labels — match the HTML source's literal values.
- Follow existing SDK factory conventions exactly: pure factories (no DOM/network/wall-clock/mutable-global-state), `ComponentDescriptor` + `SdkComponentFactory` pair, generated ids seeded from `context.instanceId`, ports/slots declared the same way `generic/composites.ts` and `generic/chrome.ts` already do. Read `apps/explainers/src/flow/sdk/generic/composites.ts` (especially the `sdk.matrix`/`sdk.tree` factories) and `apps/explainers/src/flow/sdk/types.ts` before writing any new factory — mirror their exact patterns rather than inventing a new style.
- Every new component must pass `npm run assert:sdk-authoring` (zero prohibited signatures) and have a teaching slide in `sdk-generic-catalog.flow`, per that catalog's existing "every registered primitive gets a demo slide" invariant (see its `hub.description`: "teaches every generic Flow SDK primitive").
- Commit at file granularity, `git commit --no-verify` (branch fmt drift), never `git add -A` (shared working tree — other concurrent agents have unrelated in-progress files; stage only files this plan's tasks touch).
- After every task: `cd apps/explainers && npm run build && npx vitest run` must both pass (allow pre-existing unrelated failures only if independently confirmed via `git log --oneline -- <file>` showing none of this plan's commits touch the failing file — same verification method used in the prior explainers-restyle effort in this session).
- `npm run assert:sdk-authoring` and `npm run flow-verifier` must pass (0 errors; warnings from files this plan did not touch are pre-existing and not blocking, same standard as the prior restyle effort).

---

## File Structure

| File | Responsibility |
|---|---|
| `apps/explainers/src/flow/sdk/generic/deck-composites.ts` (new) | 8 new composite factories: `sdk.sectionDivider`, `sdk.stepChain`, `sdk.bigStat`, `sdk.compareGrid`, `sdk.numberedSequence`, `sdk.timelineAxis`, `sdk.nodeTree`, `sdk.cardGrid`. |
| `apps/explainers/src/flow/sdk/generic/deck-composites.test.ts` (new) | Unit tests for all 8 factories (one `describe` block per component, following `generic/composites.test.ts`'s existing shape). |
| `apps/explainers/src/flow/sdk/registry.ts` | Import and splice `DECK_COMPOSITE_SDK_COMPONENTS` into the registered component set, alongside the existing `GENERIC_COMPOSITE_SDK_COMPONENTS` line. |
| `apps/explainers/decks-flow/sdk-generic-catalog.flow` | Append one new chapter slide + 8 new per-component teaching slides (mirroring the file's existing chapter structure) demonstrating the 8 new composites. |
| `apps/explainers/decks-flow/rust-architecture-deck-port.flow` (new) | The 49-slide ported deck, built up chapter-by-chapter across Tasks 3-9. |

---

## Task 1: Implement 4 new composites — sectionDivider, stepChain, bigStat, compareGrid

**Files:**
- Create: `apps/explainers/src/flow/sdk/generic/deck-composites.ts`
- Create: `apps/explainers/src/flow/sdk/generic/deck-composites.test.ts`
- Modify: `apps/explainers/src/flow/sdk/registry.ts`

**Interfaces:**
- Produces: `DECK_COMPOSITE_SDK_COMPONENTS: readonly SdkComponentDefinition[]` exported from `deck-composites.ts`, containing (in this task) `sdk.sectionDivider`, `sdk.stepChain`, `sdk.bigStat`, `sdk.compareGrid` — Task 2 appends the other 4 to the same array/file.

Read `apps/explainers/src/flow/sdk/generic/composites.ts` in full first (its `sdk.matrix` and `sdk.tree` factories are the closest existing analogues — grid layout and root/children respectively) and `apps/explainers/src/flow/sdk/types.ts` for the exact `SdkComponentFactory`/`SceneFragment`/`SdkExpansionContext` shapes. Mirror those files' exact conventions: pure functions, `diagnostic()` for prop errors, `attachSdkOrigin`, ids seeded from `context.instanceId`.

Component specs (source: HTML deck slides "Thesis", "Divider 01-04", "Orientation", "Failure funnel"):

- [ ] **Step 1: `sdk.sectionDivider`**

  Props: `number: string` (required, e.g. `"01"`), `title: string` (required), `subtitle: string` (optional), `eyebrow: string` (optional). No slots.
  Renders (right-aligned block, matching HTML source's Divider slides): a huge mono `number` (~120px equivalent, green `#76B900`), the `title` as a large bold heading below it (black), the `subtitle` as body text below that (gray `#555`). Capability: `core.group` root containing `core.text` children (mirror how `sdk.stat`/`sdk.metric` in `generic/chrome.ts` compose a title + value + supporting text from three text children).

- [ ] **Step 2: `sdk.stepChain`**

  Props: `direction: "row" | "column"` (default `"row"`), `steps: Array<{ number: string, label: string, detail?: string }>` (required, min 1). No slots.
  Renders: for each step, a bordered box (`#E4E4E4` border, green `#76B900` accent — top border in row mode, left border in column mode) containing a mono `number` kicker (green), a bold `label`, and optional gray `detail` text; an arrow (green, using the existing `core.arrow` capability the same way `sdk.arrow` does) between each consecutive pair of steps. Source: HTML "Orientation" slide's 6-step `01 VALIDATE → 02 SELECT → ... → 06 EMIT` row, and "Flow diagram" slide's vertical `Python → aiperf-runner → ... → stdout` chain.

- [ ] **Step 3: `sdk.bigStat`**

  Props: `value: string` (required, e.g. `"3"`), `title: string` (optional), `description: string` (optional). No slots.
  Renders: a giant `value` text (huge font size, ~200px equivalent, green, mirroring the HTML source's `font-size:220px` treatment), with `title`/`description` as smaller text beside or below it. Source: HTML "Three modes" slide's giant `3`.

- [ ] **Step 4: `sdk.compareGrid`**

  Props: `columns: number` (default 3), `items: Array<{ label: string, detail?: string }>` (required, min 1). No slots.
  Renders: an N-column grid (reuse the grid-layout approach from `sdk.matrix` in `composites.ts`), each cell a green-top-bordered block with bold `label` and gray `detail` text below. Source: HTML "Thesis" slide's 3-column takeaway row, "Failure funnel" slide's 3-column `0 / 1 / 2` stat row.

- [ ] **Step 5: Register the four components**

  In `deck-composites.ts`, export:
  ```ts
  export const DECK_COMPOSITE_SDK_COMPONENTS: readonly SdkComponentDefinition[] = [
    SECTION_DIVIDER_DEFINITION,
    STEP_CHAIN_DEFINITION,
    BIG_STAT_DEFINITION,
    COMPARE_GRID_DEFINITION,
  ];
  ```
  In `apps/explainers/src/flow/sdk/registry.ts`, add the import
  `import { DECK_COMPOSITE_SDK_COMPONENTS } from "./generic/deck-composites.js";`
  next to the existing `GENERIC_COMPOSITE_SDK_COMPONENTS` import, and splice
  `...DECK_COMPOSITE_SDK_COMPONENTS,` into the same array literal that
  already spreads `...GENERIC_COMPOSITE_SDK_COMPONENTS,` (around line 352).

- [ ] **Step 6: Write unit tests**

  In `deck-composites.test.ts`, for each of the 4 components: one test that
  the factory expands successfully with minimal valid props (asserting the
  returned fragment has the expected root capability and no diagnostics),
  and one test that a required prop's absence produces a diagnostic (not a
  throw) — mirror the exact assertion style used in
  `apps/explainers/src/flow/sdk/generic/composites.test.ts`.

- [ ] **Step 7: Run tests and build**

  ```bash
  cd apps/explainers
  npx vitest run src/flow/sdk/generic/deck-composites.test.ts
  npm run build
  ```
  Expected: new test file passes, build exits 0.

- [ ] **Step 8: Commit**

  ```bash
  git add src/flow/sdk/generic/deck-composites.ts src/flow/sdk/generic/deck-composites.test.ts src/flow/sdk/registry.ts
  git commit --no-verify -m "feat(explainers): add sectionDivider/stepChain/bigStat/compareGrid SDK composites"
  ```

---

## Task 2: Implement 4 new composites — numberedSequence, timelineAxis, nodeTree, cardGrid

**Files:**
- Modify: `apps/explainers/src/flow/sdk/generic/deck-composites.ts` (append to the same file/array Task 1 created)
- Modify: `apps/explainers/src/flow/sdk/generic/deck-composites.test.ts`

**Interfaces:**
- Consumes: the `DECK_COMPOSITE_SDK_COMPONENTS` array and file scaffolding Task 1 created.
- Produces: the array now contains all 8 components; `registry.ts` needs no further change (already splices the whole array).

Component specs (source: HTML deck slides "Observer sequence", "Clock"):

- [ ] **Step 1: `sdk.numberedSequence`**

  Props: `items: Array<{ number: string, title: string, detail?: string, emphasis?: boolean }>` (required, min 1). No slots.
  Renders: a vertical stack of rows, each with a small square index chip on the left (green fill if `emphasis` true, black fill otherwise — mirror the HTML source's alternating green/black `1..6` squares) containing the `number`, and to its right a bordered box with bold mono `title` and gray `detail`. Source: HTML "Observer sequence" slide's 6-row `on_arrival` / `on_admit` / `on_token` / ... callback list.

- [ ] **Step 2: `sdk.timelineAxis`**

  Props: `start: number`, `end: number` (both required), `unit: string` (optional, e.g. `"ms"`), `ticks: Array<{ at: number, label: string }>` (optional), `markers: Array<{ at: number, label: string, style?: "exact" | "late" }>` (optional), `target: { at: number, label: string }` (optional, dashed vertical reference line). No slots.
  Renders: a horizontal line spanning `start`..`end`, small tick marks + labels at each `ticks` entry, a circle marker at each `markers` entry (filled green for `"exact"`, hollow gray for `"late"`), and if `target` is present a dashed vertical line with its label. Source: HTML "Clock" slide's RealClock diagram (`0ms 1ms 2ms 3ms` ticks, `target` dashed line, filled "timerfd exact" dot, hollow "wheel late" dot).

- [ ] **Step 3: `sdk.nodeTree`**

  Props: `root: { label: string, detail?: string }` (required), `children: Array<{ label: string, detail?: string, emphasis?: boolean }>` (required, min 1), `orderNote: string` (optional, caption text below the tree). No slots.
  Renders: one root box (green fill if it's the "popped first" element per the HTML source, i.e. `emphasis`-equivalent on the root itself — reuse the same `emphasis` boolean convention as `numberedSequence`) with `children.length` child boxes below it, each connected to the root by a line (reuse `sdk.line`'s rendering approach), plus the `orderNote` caption underneath if present. Source: HTML "Clock" slide's SimClock `BinaryHeap<Sleeper>` diagram — root `(100, 0)` green, two children `(140, 1)` / `(140, 2)` white, connecting lines, caption "pop order → (100,0) → (140,1) → (140,2)".

- [ ] **Step 4: `sdk.cardGrid`**

  Props: `columns: number` (default 2), `cards: Array<{ title: string, detail: string, accent?: "green" | "black" | "gray" }>` (required, min 1). No slots.
  Renders: a grid of bordered cards (reuse the `sdk.matrix` grid-layout approach again), each with a left-accent-colored border strip (green/black/gray per `accent`, default gray), bold mono `title`, and gray `detail` body text. Source: HTML "Crate topology" slide's 4-card grid (`loadgen-core` / `aiperf` / `aiperf-runner` / `aiperf-mock-server`, each with a colored left border matching its role).

- [ ] **Step 5: Register and export**

  Update the `DECK_COMPOSITE_SDK_COMPONENTS` array from Task 1 to include all 8:
  ```ts
  export const DECK_COMPOSITE_SDK_COMPONENTS: readonly SdkComponentDefinition[] = [
    SECTION_DIVIDER_DEFINITION,
    STEP_CHAIN_DEFINITION,
    BIG_STAT_DEFINITION,
    COMPARE_GRID_DEFINITION,
    NUMBERED_SEQUENCE_DEFINITION,
    TIMELINE_AXIS_DEFINITION,
    NODE_TREE_DEFINITION,
    CARD_GRID_DEFINITION,
  ];
  ```
  No further `registry.ts` change needed (Task 1 already spliced the array reference, not a snapshot).

- [ ] **Step 6: Write unit tests**

  Same pattern as Task 1 Step 6, for these 4 components.

- [ ] **Step 7: Run tests and build**

  ```bash
  cd apps/explainers
  npx vitest run src/flow/sdk/generic/deck-composites.test.ts
  npm run build
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add src/flow/sdk/generic/deck-composites.ts src/flow/sdk/generic/deck-composites.test.ts
  git commit --no-verify -m "feat(explainers): add numberedSequence/timelineAxis/nodeTree/cardGrid SDK composites"
  ```

---

## Task 3: Teaching chapter for the 8 new composites in sdk-generic-catalog.flow

**Files:**
- Modify: `apps/explainers/decks-flow/sdk-generic-catalog.flow`

**Interfaces:**
- Consumes: all 8 `sdk.*` components from Tasks 1-2.

- [ ] **Step 1: Read the existing chapter pattern**

  Read `apps/explainers/decks-flow/sdk-generic-catalog.flow` in full (already
  read once this session — reuse that context if still available, otherwise
  re-read). Note its structure: a "Chapter N" overview slide followed by one
  slide per primitive in that family, each with `eyebrow`/`title`/`lede`/
  `narration`/`caption`/`render`, ending in a `hub.description` mentioning
  the primitive count (currently "45 primitives · seven families").

- [ ] **Step 2: Append "Chapter 8 · Deck Composites" overview slide**

  One `slide { ... }` block introducing the new family, following the exact
  structural pattern of the existing "Foundations and shapes" / "Typography
  and rich content" chapter-overview slides (header + title + lede +
  narration + a small `render` scene using 2-3 of the new components as a
  preview, timeline choreography matching the existing chapters' cadence).

- [ ] **Step 3: Append one teaching slide per new component (8 slides)**

  For each of `sdk.sectionDivider`, `sdk.stepChain`, `sdk.bigStat`,
  `sdk.compareGrid`, `sdk.numberedSequence`, `sdk.timelineAxis`,
  `sdk.nodeTree`, `sdk.cardGrid`: one slide following the exact pattern of
  every existing per-primitive slide in this file (eyebrow `"DECK
  COMPOSITES · SDK.X"`, title, lede, **narration describing what the
  rendered content actually is** — not "hero"/"variant" language, per this
  session's earlier fix — caption, and a `render` scene that exercises the
  component with realistic content drawn from its HTML-deck source slide,
  e.g. the `sdk.timelineAxis` teaching slide reuses the Clock slide's own
  `0/1/2/3ms` tick values as a natural worked example).

- [ ] **Step 4: Update the hub description and eyebrow counts**

  Update `hub.description` (currently states 45 primitives / seven
  families) to `53 primitives · eight families` (45 + 8), and update the
  opening "Generic primitives, composed" slide's `title`/`lede` numbers to
  match, mirroring how the file already states its primitive/family counts
  in three places (`hub`, `startGateTitle` context, opening slide body).

- [ ] **Step 5: Verify**

  ```bash
  cd apps/explainers
  npm run build
  npm run assert:sdk-authoring
  npx vitest run
  ```
  Expected: all pass; `assert:sdk-authoring` reports the new slide count
  with zero prohibited signatures.

- [ ] **Step 6: Commit**

  ```bash
  git add decks-flow/sdk-generic-catalog.flow
  git commit --no-verify -m "docs(explainers): add Chapter 8 teaching slides for new deck composites"
  ```

---

## Task 4: Port Chapter 1 — Open (Cover, Thesis, Orientation)

**Files:**
- Create: `apps/explainers/decks-flow/rust-architecture-deck-port.flow`

**Interfaces:**
- Consumes: all components from Tasks 1-2 plus every existing `sdk.*` primitive.
- Produces: the deck's `explainer { ... }` shell (id, route, topic, hub, etc.) that Tasks 5-10 append slides into.

- [ ] **Step 1: Read the HTML source for these 3 slides**

  Read `/tmp/deck.html` lines 1-88 (Cover, Thesis, Orientation — already
  read once this session at the start of the brainstorming phase; re-read
  if that context is gone). Extract exact text, numbers, and colors.

- [ ] **Step 2: Create the deck shell**

  ```
  explainer "Rust Architecture — Deck Port" {
    id: "rust-architecture-deck-port"
    route: "/rust-architecture-deck-port"
    topic: "architecture"
    storagePrefix: "rust-architecture-deck-port-explainer"
    classPrefix: "deck-rust-architecture-deck-port"
    eyebrowLabel: "ARCHITECTURE DEEP DIVE"
    startGateTitle: "AIPerf Rust Runtime"

    hub: {
      highlight: "Rust runtime"
      title: "AIPerf Rust Runtime · 49-slide deck port"
      description: "A native port of the reference NVIDIA architecture deck — every slide rebuilt from composable SDK primitives."
    }

    // slides appended by Tasks 4-10
  }
  ```
  (Use the SPDX header block every other `.flow` file in this directory
  starts with, copied verbatim from `sdk-generic-catalog.flow`'s first 4
  lines.)

- [ ] **Step 3: Author the Cover slide**

  `slide "Cover"`: eyebrow `"ARCHITECTURE DEEP DIVE"` (mono, green,
  uppercase, matches HTML `Architecture Deep Dive` kicker), title `"AIPerf
  Rust Runtime"` (huge, black), lede matching the HTML subtitle ("From the
  command line to results — every component on the product path, and the
  two seams that hold it together"), narration describing the slide's
  actual content (byline "Anthony Casagrande · 2026", NVIDIA logo mark —
  use `sdk.IconLabel` or `sdk.Caption` for the byline row, do not invent a
  new component for a one-off logo placement; if no existing image asset
  applies, omit the literal NVIDIA wordmark image and note the omission in
  the task's completion report rather than hand-drawing one).

- [ ] **Step 4: Author the Thesis slide**

  `slide "Thesis"`: use `sdk.compareGrid` (3 columns) for the "Python
  owns config..." / "aiperf-runner is the only..." / "Two trait seams..."
  takeaway row, and existing `sdk.KeyValue`-or-`sdk.Chip`-style boxes (pick
  whichever existing primitive best matches the HTML source's
  `Python → protocol-v2 envelope → aiperf-runner → native-v2.json` row —
  a `sdk.StepChain` row with 4 steps and no numbers is the closest fit;
  reuse it with `number` left empty per-step if the descriptor allows, or
  fall back to 4 `sdk.Shape`/`sdk.Label` pairs connected by
  `sdk.Arrow` if `stepChain` requires numbers — implementer's judgment,
  document the choice in the report).

- [ ] **Step 5: Author the Orientation slide**

  `slide "Orientation"`: use `sdk.stepChain` (`direction: "row"`) with the
  6 steps `01 VALIDATE` / `02 SELECT` / `03 DRIVE` / `04 MEASURE` /
  `05 REPORT` / `06 EMIT`, each `detail` matching the HTML source's
  per-step description text.

- [ ] **Step 6: Verify**

  ```bash
  cd apps/explainers
  npm run build
  npm run assert:sdk-authoring
  npx vitest run
  ```

- [ ] **Step 7: Visual check**

  Start `npm run dev` (background), Playwright-navigate to
  `#/rust-architecture-deck-port`, dismiss the gate, screenshot the first 3
  slides, compare against `/tmp/deck.html`'s Cover/Thesis/Orientation
  sections for text/color/layout fidelity. Stop the dev server. Describe
  discrepancies found (if any) in the task report; fix before committing
  if the discrepancy is a wrong value (not merely a minor position
  difference, which is within the spec's declared tolerance).

- [ ] **Step 8: Commit**

  ```bash
  git add decks-flow/rust-architecture-deck-port.flow
  git commit --no-verify -m "feat(explainers): port deck chapter 1 (Open) to rust-architecture-deck-port"
  ```

---

## Task 5: Port Chapter 2 — Two Seams (7 slides)

**Files:**
- Modify: `apps/explainers/decks-flow/rust-architecture-deck-port.flow`

**Interfaces:**
- Consumes: the deck shell from Task 4; appends slides to it.

- [ ] **Step 1: Read the HTML source**

  Read `/tmp/deck.html` lines 90-287 (Divider 01, Seams overview, Clock,
  Drivers, Transport seam, Observer sequence, Three modes).

- [ ] **Step 2: Author "Divider 01"**

  `sdk.sectionDivider(number = "01", title = "The Two Seams", subtitle =
  "If you understand these, the rest is composition")`.

- [ ] **Step 3: Author "Seams overview"**

  Use `sdk.compareGrid` or `sdk.cardGrid` for the `{clock}`/`{transport}`
  two-box comparison, `sdk.Chip`/`sdk.Badge`-style pills for the
  `online-real`/`online-mock`/`offline co-sim` row above and the
  `scheduling`/`pacing`/`admission`/`reporting` row below (reuse
  `sdk.TagList` — it already renders exactly this "row of pill chips"
  shape, proven in `sdk-generic-catalog.flow`'s own `sdk.tagList` slide).

- [ ] **Step 4: Author "Clock"**

  Use `sdk.timelineAxis` for the RealClock half (ticks at 0/1/2/3ms, a
  `target` dashed line, an `"exact"`-style marker and a `"late"`-style
  marker) and `sdk.nodeTree` for the SimClock half (root `(100, 0)`,
  children `(140, 1)` / `(140, 2)`, `orderNote` = "pop order → (100,0) →
  (140,1) → (140,2)"), side by side in a two-column layout (reuse
  `sdk.SplitPane` from the generic catalog, already proven for exactly this
  "two peer regions side by side" shape).

- [ ] **Step 5: Author "Drivers"**

  Left half: reuse `sdk.timelineAxis` (or a simple `sdk.stepChain` row of
  3 "fire" markers) for `drive_real`'s wall-clock fire sequence. Right
  half: the `drive_sim` idle-pump loop — this is a genuine cyclic
  state diagram, so use the **existing diagram-family** `sdk.ProcessStep` +
  `sdk.Loop` + `sdk.Edge` nodes (already declarative, not hand-drawn;
  proven in `sdk-diagram-catalog.flow`'s own `sdk.loop` slide) for
  `poll LocalSet → advance_to → wake sleepers → (repeat)`.

- [ ] **Step 6: Author "Transport seam"**

  Use `sdk.cardGrid` (3 columns) for the `Dispatchable` /
  `RequestSink<R>` / `RequestObserver` trait cards, and `sdk.TagList` +
  `sdk.Arrow` for the "real HTTP / native gRPC / mock HTTP / co-sim → one
  RequestObserver" summary row.

- [ ] **Step 7: Author "Observer sequence"**

  Use `sdk.numberedSequence` with the 6 callback rows
  (`on_arrival`/`on_admit`/`on_token`/`on_usage`/`on_endpoint_metrics`/
  `on_terminal`), alternating `emphasis` true/false per the HTML source's
  alternating green/black index-chip coloring.

- [ ] **Step 8: Author "Three modes"**

  Use `sdk.bigStat(value = "3", title = "interchangeable run modes from
  one codebase")` plus `sdk.compareGrid` (3 columns) for the
  `online-real`/`online-mock`/`offline co-sim` detail row below it.

- [ ] **Step 9: Verify**

  ```bash
  cd apps/explainers
  npm run build
  npm run assert:sdk-authoring
  npx vitest run
  ```

- [ ] **Step 10: Visual check** (same method as Task 4 Step 7, this chapter's 7 slides)

- [ ] **Step 11: Commit**

  ```bash
  git add decks-flow/rust-architecture-deck-port.flow
  git commit --no-verify -m "feat(explainers): port deck chapter 2 (Two Seams) to rust-architecture-deck-port"
  ```

---

## Task 6: Port Chapter 3 — Crate Topology & Flow (5 slides)

**Files:**
- Modify: `apps/explainers/decks-flow/rust-architecture-deck-port.flow`

- [ ] **Step 1:** Read `/tmp/deck.html` lines 289-383 (Divider 02, Crate topology, Module universe, Flow diagram, Failure funnel).
- [ ] **Step 2:** Author "Divider 02" via `sdk.sectionDivider(number = "02", ...)`.
- [ ] **Step 3:** Author "Crate topology" via `sdk.cardGrid` (2 columns, 4 cards: `loadgen-core`/`aiperf`/`aiperf-runner`/`aiperf-mock-server`, accents matching the HTML source's per-card left-border colors) plus a `sdk.StepChain` or `sdk.TagList` + `sdk.Arrow` row for the dependency-direction chips above it.
- [ ] **Step 4:** Author "Module universe" via two `sdk.TagList` rows (Foundations, Composition) exactly as the existing `sdk-generic-catalog.flow`'s own `sdk.tagList` teaching slide already proves this "wrapped row of many chips" shape.
- [ ] **Step 5:** Author "Flow diagram" via `sdk.stepChain(direction = "column")`, one step per pipeline stage (`Python` / `aiperf-runner · main.rs` / `RunnerV2Coordinator::handle` / `execute.rs` / `persist + export` / `Python reads...`), matching the HTML source's vertical connector-and-box chain.
- [ ] **Step 6:** Author "Failure funnel" via `sdk.compareGrid` for the 5-stage `RunnerFailureStageV2` row (Protocol/Validation/Preparation/Execution/Reporting) and a second 3-column `sdk.compareGrid` for the `0`/`1`/`2` outcome-count row.
- [ ] **Step 7:** Verify (`npm run build`, `assert:sdk-authoring`, `npx vitest run`).
- [ ] **Step 8:** Visual check (same method as Task 4 Step 7).
- [ ] **Step 9:** Commit: `git add decks-flow/rust-architecture-deck-port.flow && git commit --no-verify -m "feat(explainers): port deck chapter 3 (Crate Topology & Flow) to rust-architecture-deck-port"`

---

## Task 7: Port Chapter 4 — Component Reference A–H, part 1 (11 slides)

**Files:**
- Modify: `apps/explainers/decks-flow/rust-architecture-deck-port.flow`

- [ ] **Step 1:** Read `/tmp/deck.html` lines 385-829 (Divider 03, A·Process boundary, A·Coordinator pipeline, B·Input resolution, C·Execution paths, D·HTTP transport, D·HTTP internals, E·gRPC transport, F·Endpoints, G·Dataset pipeline, G·Segment store, G·Pre-serialization).
- [ ] **Step 2:** Author "Divider 03" via `sdk.sectionDivider(number = "03", title = "Component Reference", subtitle = "A through R — the process boundary to the mock server")`.
- [ ] **Step 3:** For each of the 10 remaining slides, choose the closest-fit existing/new component per its HTML content shape (a lettered-badge + kicker header pattern repeats across all component-reference slides — reuse `sdk.IconLabel` or a small `sdk.Shape` badge + `sdk.Label` pair for the `A`/`B`/`C`... letter badge, matching the HTML source's `46px` colored square badge). Where a slide's body is a card grid, table, code block, or numbered sequence, reuse the matching existing/new component from the library built in Tasks 1-2 rather than a bespoke one-off layout. Document each slide's component choice briefly in the task report.
- [ ] **Step 4:** Verify (`npm run build`, `assert:sdk-authoring`, `npx vitest run`).
- [ ] **Step 5:** Visual check (same method as Task 4 Step 7, spot-check at least 4 of the 11 slides for time).
- [ ] **Step 6:** Commit: `git add decks-flow/rust-architecture-deck-port.flow && git commit --no-verify -m "feat(explainers): port deck chapter 4 (Component Reference A-H part 1) to rust-architecture-deck-port"`

---

## Task 8: Port Chapter 4 — Component Reference, part 2 (8 slides)

**Files:**
- Modify: `apps/explainers/decks-flow/rust-architecture-deck-port.flow`

- [ ] **Step 1:** Read `/tmp/deck.html` lines 794-1041 (G·Dispatch materialization, H·RNG substrate, I·Timing and scheduling, I·Phase lifecycle, J·Graph-IR engine, J·Lowering and execution, J·Agentic replay, K·Metrics core).
- [ ] **Step 2:** Author each slide following the same component-selection method as Task 7 Step 3.
- [ ] **Step 3:** Verify (`npm run build`, `assert:sdk-authoring`, `npx vitest run`).
- [ ] **Step 4:** Visual check (spot-check at least 3 of the 8 slides).
- [ ] **Step 5:** Commit: `git add decks-flow/rust-architecture-deck-port.flow && git commit --no-verify -m "feat(explainers): port deck chapter 4 (Component Reference part 2) to rust-architecture-deck-port"`

---

## Task 9: Port Chapter 4 — Component Reference, part 3 (8 slides)

**Files:**
- Modify: `apps/explainers/decks-flow/rust-architecture-deck-port.flow`

- [ ] **Step 1:** Read `/tmp/deck.html` lines 1042-1341 (K·RaggedSeries, L·Reporting and export, M·Adaptive scale, N·Accuracy, O·Side-channel telemetry, P·Cellular, Q·Dynosim, R·Mock server).
- [ ] **Step 2:** Author each slide following the same component-selection method as Task 7 Step 3.
- [ ] **Step 3:** Verify (`npm run build`, `assert:sdk-authoring`, `npx vitest run`).
- [ ] **Step 4:** Visual check (spot-check at least 3 of the 8 slides).
- [ ] **Step 5:** Commit: `git add decks-flow/rust-architecture-deck-port.flow && git commit --no-verify -m "feat(explainers): port deck chapter 4 (Component Reference part 3) to rust-architecture-deck-port"`

---

## Task 10: Port Chapter 5 — Closing (6 slides)

**Files:**
- Modify: `apps/explainers/decks-flow/rust-architecture-deck-port.flow`

- [ ] **Step 1:** Read `/tmp/deck.html` lines 1342-1505 (Divider 04, Config catalog, Everything is a trait, Invariants, System map, Closing).
- [ ] **Step 2:** Author "Divider 04" via `sdk.sectionDivider`.
- [ ] **Step 3:** Author "Config catalog" via `sdk.Table` (the file's existing table primitive already proves exactly this "grid of labeled rows" shape).
- [ ] **Step 4:** Author "Everything is a trait", "Invariants" via `sdk.compareGrid`/`sdk.cardGrid` per their HTML content shape.
- [ ] **Step 5:** Author "System map" via `sdk.cardGrid` or the diagram-family `sdk.ProcessStep`/`sdk.Edge` nodes if the HTML source shows a connected map rather than a flat grid — check the source before choosing.
- [ ] **Step 6:** Author "Closing" as a simple title/lede/byline slide mirroring the Cover slide's structure.
- [ ] **Step 7:** Verify (`npm run build`, `assert:sdk-authoring`, `npx vitest run`).
- [ ] **Step 8:** Visual check (all 6 slides — this is the final chapter, do not spot-check, verify every slide).
- [ ] **Step 9:** Commit: `git add decks-flow/rust-architecture-deck-port.flow && git commit --no-verify -m "feat(explainers): port deck chapter 5 (Closing) to rust-architecture-deck-port"`

---

## Task 11: Final full-deck verification and hub wiring

**Files:**
- Modify: `apps/explainers/src/core/deck-registry.ts` (register the new deck in the hub, following the exact pattern every other deck already uses — check how `rust-architecture`/`rust-architecture-atlas` are registered and mirror it)
- Read-only verification of: `apps/explainers/decks-flow/rust-architecture-deck-port.flow`

- [ ] **Step 1:** Read `apps/explainers/src/core/deck-registry.ts`, find the existing registration entries for `rust-architecture` and `rust-architecture-atlas`, and add a matching entry for `rust-architecture-deck-port` so it appears on the hub page.

- [ ] **Step 2:** Full verification suite:
  ```bash
  cd apps/explainers
  npm run build
  npx vitest run
  npm run assert:no-mentalmodel-registry
  npm run assert:sdk-authoring
  npm run flow-verifier
  ```
  Expected: all exit 0 (or, for `flow-verifier`, zero *new* errors/warnings attributable to `rust-architecture-deck-port.flow` or `deck-composites.ts` — cross-check any warning's originating file the same way this session's earlier restyle effort did, via `git status`/`git log -- <file>` to confirm it's pre-existing unrelated concurrent-agent churn, not this plan's own work).

- [ ] **Step 3:** Full-deck visual walkthrough: start `npm run dev` (background), Playwright-navigate through all 49 slides of `#/rust-architecture-deck-port` in play mode, screenshot each, and produce a written parity report (in the task's completion report, not a new file) noting any slide where content, numbers, or colors diverge from `/tmp/deck.html` — fix any wrong *value* (not merely position) before declaring the task done. Stop the dev server when finished.

- [ ] **Step 4:** Count check: confirm the deck has exactly 49 `slide "..."` blocks (`grep -c 'slide "' decks-flow/rust-architecture-deck-port.flow`, accounting for the fact this count also matches quoted strings inside narration — cross-check against `data-screen-label` count in `/tmp/deck.html`, which is 49).

- [ ] **Step 5:** Commit:
  ```bash
  git add src/core/deck-registry.ts
  git commit --no-verify -m "feat(explainers): wire rust-architecture-deck-port into the hub"
  ```

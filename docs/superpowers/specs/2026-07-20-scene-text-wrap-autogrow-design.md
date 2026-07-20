# Scene Text Wrap & Auto-Grow — Design

## Purpose

`apps/explainers`' `SceneRenderer` has no automatic text wrapping: text nodes
render via raw SVG `<text>`/`<tspan>`, splitting onto multiple lines only on
an explicit `\n` character already present in the authored string
(`SceneRenderer.tsx` ~line 3340). Every component author must hand-guess
where to insert `\n` to fit a box at a given width; guessing wrong — or a
future content edit changing string length — causes text to overflow its
box horizontally, or clip vertically when authored box heights assumed a
single line. This was the direct cause of the overflow/clipping bugs found
repeatedly while porting the 49-slide `rust-architecture-deck-port` deck
(Tasks 5, 6, 9, 10 in that effort each hand-fixed individual overflow
instances by manually re-authoring coordinates).

This spec adds real automatic word-wrap plus box auto-grow, so authors stop
needing to hand-place line breaks and hand-guess box heights.

## Built (this spec, not yet implemented — plan at
`docs/superpowers/plans/2026-07-20-scene-text-wrap-autogrow.md`)

### Layer 1 — render-time word-wrap (global, all decks)

A new `wrapTextToWidth(text, maxWidth, fontSize, weight)` function in
`apps/explainers/src/core/diagram/text-metrics.ts`, using the existing
`estimateTextWidth` measurer to greedily pack words onto lines that fit
`maxWidth`, breaking a single word only if it alone exceeds `maxWidth`
(rare; long unbroken tokens like URLs). `SceneRenderer.tsx`'s core text
render path (~line 3340) calls this whenever `node.text` contains **no**
manual `\n` and `node.style?.whiteSpace !== "nowrap"`, using `geom.width`
as `maxWidth`; the resulting lines render as `<tspan>` rows exactly the way
manually-`\n`-separated content already does today (no new rendering
branch, same `lineHeight` logic). Text that already contains manual `\n` is
left as author-authoritative (existing behavior, unchanged) — this lets an
author still force a specific break where wrapping alone wouldn't look
right. `style.whiteSpace: "nowrap"` is the opt-out for content that must
stay on one line (e.g. a numeric stat, a short kicker) even if it would
technically overflow.

This layer applies immediately, with no per-component change required,
to every existing deck in the app — since it only activates when a text
node's content would otherwise have overflowed its width, it can only ever
reduce overflow, never introduce a new regression in content that already
fit.

### Layer 2 — expand-time box auto-grow (scoped set of components)

`SceneRenderer` has no reflow/flexbox layout: every node's `x`/`y`/`width`/
`height` is computed once, in TypeScript, by the SDK component factory at
scene-*expansion* time (not recomputed at paint time). So "grow the box
and push siblings down" must happen in the factories themselves, using the
same `wrapTextToWidth` measurer to count how many lines a `detail`/
`description`/body text prop will actually need at its authored width, and
sizing the generated child node's height (and any subsequent sibling's `y`
offset within the same factory) from that line count instead of a fixed
default.

In scope for this pass — components whose entire purpose is to lay out one
block of body prose, where "how tall should this be" is a direct function
of wrapped line count with no independent children to reflow around:

- `apps/explainers/src/flow/sdk/generic/deck-composites.ts`: all 8
  components built in the deck-port effort (`sectionDivider`, `stepChain`,
  `bigStat`, `compareGrid`, `numberedSequence`, `timelineAxis`, `nodeTree`,
  `cardGrid`) — their `detail`/`description`-shaped props grow their own
  cell/row height, and for row-based composites (`stepChain`,
  `numberedSequence`), growing one row's height shifts every later row's
  `y` down by the delta.
- `apps/explainers/src/flow/sdk/generic/catalog.ts`'s generic
  `sdk.paragraph`/`sdk.richText`/`sdk.quote`/`sdk.text`/`sdk.title` family
  (one shared table-driven factory function producing `core.text`/
  `core.group` nodes) — height grows from the table's default width (or an
  authored `width` override) and the measured line count.
- `apps/explainers/src/flow/sdk/generic/chrome.ts`'s `sdk.note` factory —
  same treatment for its `text` prop.

Explicitly **out of scope** for this pass (documented here so it isn't
mistaken for an oversight): true nested *layout containers* that host
independent child components and must reflow their own bounding box
around however tall those children turn out to be —
`sdk.section`/`sdk.panel`/`sdk.card`/`sdk.toolbar`/`sdk.splitPane`/
`sdk.mediaObject`. Making those auto-size around arbitrarily-grown children
is a genuine flow-layout engine (measure children bottom-up, then
position top-down), a materially larger effort than word-wrap + one
level of row-height growth, and is not needed to fix the overflow bugs
this spec targets (none of those containers were the source of the
reported clipping — the leaf text nodes were). A future spec can pick
this up if a real need arises.

## Verification

- Unit tests for `wrapTextToWidth` covering: short text (no wrap needed),
  text needing exactly 2/3 lines, a single word longer than `maxWidth`
  (must not infinite-loop; emits on its own line), and empty string.
- Full app regression: `npm run build`, `npx vitest run`,
  `npm run assert:no-mentalmodel-registry`, `npm run assert:sdk-authoring`,
  `npm run flow-verifier` (the full Playwright check, across **every**
  deck in `decks-flow/`, not just the port) — must show 0 errors, and the
  `SCENE_VIEWPORT_ESCAPE`/`SCENE_ABSOLUTE_SIBLING_OVERLAP` warning count
  across the whole app should decrease or stay flat, never increase (a new
  warning appearing on a previously-clean deck would indicate a
  regression from the wrap/auto-grow change, not an acceptable tradeoff).
- Re-walk `rust-architecture-deck-port.flow`'s 49 slides visually
  (Playwright) to confirm the specific overflow/shift instances the user
  reported are now resolved, and — as a cleanup, not a correctness
  requirement — simplify any of that deck's manual multi-`Paragraph`-row
  workarounds (from Tasks 9-10 of the prior effort) back to natural
  composite usage now that height auto-grows, where doing so is a clean
  net simplification.

## Source anchors

- `apps/explainers/src/core/diagram/text-metrics.ts` (new `wrapTextToWidth`)
- `apps/explainers/src/core/diagram/SceneRenderer.tsx` (~line 3340, text render path)
- `apps/explainers/src/flow/sdk/generic/deck-composites.ts` (8 factories)
- `apps/explainers/src/flow/sdk/generic/catalog.ts` (text-family generic factory)
- `apps/explainers/src/flow/sdk/generic/chrome.ts` (`sdk.note` factory)
- `apps/explainers/decks-flow/rust-architecture-deck-port.flow` (cleanup pass)

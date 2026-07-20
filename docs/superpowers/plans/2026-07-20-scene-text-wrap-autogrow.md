# Scene Text Wrap & Auto-Grow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic word-wrap to `apps/explainers`' scene text rendering (global, all decks) and expand-time box auto-grow to the standalone prose components (the 8 deck-port composites plus `sdk.paragraph`/`sdk.richText`/`sdk.quote`/`sdk.text`/`sdk.title`/`sdk.note`), eliminating the class of overflow/clip/shift bugs found while porting the 49-slide reference deck.

**Architecture:** Two layers. (1) A pure text-measurement utility (`wrapTextToWidth`) plus one call-site change in `SceneRenderer.tsx`'s existing text-painting path — this is the render-time safety net, applies globally, no per-component changes needed. (2) Factory-level height computation in the scoped component set, reusing the same measurer at scene-expansion time to size generated boxes (and, for row-based composites, shift subsequent row `y` offsets) from actual wrapped line count instead of a fixed default.

**Tech Stack:** TypeScript, existing `SceneRenderer`/SDK factory conventions, Vitest, Playwright.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-20-scene-text-wrap-autogrow-design.md` — read it first.
- Layer 1 (wrap) applies to ALL text nodes across ALL decks by default; only opt-out is `style.whiteSpace: "nowrap"` or the text already containing a manual `\n` (author-authoritative, unchanged behavior).
- Layer 2 (auto-grow) is scoped to exactly: the 8 components in `apps/explainers/src/flow/sdk/generic/deck-composites.ts`, the `sdk.paragraph`/`sdk.richText`/`sdk.quote`/`sdk.text`/`sdk.title` family in `apps/explainers/src/flow/sdk/generic/catalog.ts`, and `sdk.note` in `apps/explainers/src/flow/sdk/generic/chrome.ts`. Do NOT extend auto-grow to nested layout containers (`sdk.section`/`sdk.panel`/`sdk.card`/`sdk.toolbar`/`sdk.splitPane`/`sdk.mediaObject`) — explicitly out of scope per the spec.
- Preserve all existing prop names, component APIs, and capability ids — this is a rendering/sizing behavior change, not an API change.
- Commit at file granularity, `git commit --no-verify` (branch fmt drift), stage only files each task touches, never `git add -A` (shared working tree with other concurrent agents' unrelated in-progress files).
- After every task: `cd apps/explainers && npx vite build && npx vitest run` must pass (or, if a pre-existing unrelated dirty file from another concurrent agent blocks `tsc`, confirm via `git log --oneline -- <file>` that none of this plan's own commits touch it before treating it as pre-existing noise, same standard used throughout the prior deck-port effort).
- `npm run assert:sdk-authoring` and `npm run flow-verifier` (the full Playwright check, across every deck in `decks-flow/`, not just the port) must show 0 errors after every task from Task 2 onward. Track the total `SCENE_VIEWPORT_ESCAPE`/`SCENE_ABSOLUTE_SIBLING_OVERLAP` warning count before/after each task — it must not increase on any deck that didn't have that warning before.

---

## File Structure

| File | Responsibility |
|---|---|
| `apps/explainers/src/core/diagram/text-metrics.ts` | New `wrapTextToWidth` pure function + its unit tests. |
| `apps/explainers/src/core/diagram/SceneRenderer.tsx` | One call-site change in the core text-render path (~line 3340) to invoke `wrapTextToWidth` when appropriate. |
| `apps/explainers/src/flow/sdk/generic/deck-composites.ts` | 8 factories updated to measure wrapped line count and grow generated box/row heights + shift subsequent siblings. |
| `apps/explainers/src/flow/sdk/generic/catalog.ts` | The shared text-family factory grows `sdk.paragraph`/`richText`/`quote`/`text`/`title` node height from measured content. |
| `apps/explainers/src/flow/sdk/generic/chrome.ts` | `sdk.note` factory grows its node height from measured content. |
| `apps/explainers/decks-flow/rust-architecture-deck-port.flow` | Cleanup pass: simplify manual multi-row workarounds now unnecessary. |

---

## Task 1: `wrapTextToWidth` utility + unit tests

**Files:**
- Modify: `apps/explainers/src/core/diagram/text-metrics.ts`
- Create: `apps/explainers/src/core/diagram/text-metrics.test.ts` (if it doesn't already exist — check first; if it exists, extend it)

**Interfaces:**
- Produces: `export function wrapTextToWidth(text: string, maxWidth: number, fontSize: number, weight?: "normal" | "bold"): string[]` — returns an array of line strings. Consumed by Task 2 (`SceneRenderer.tsx`) and Tasks 3-4 (the component factories, to *measure* line count without needing the actual line strings — they can call `wrapTextToWidth(...).length`).

- [ ] **Step 1: Write the failing tests**

```ts
import { describe, expect, it } from "vitest";
import { wrapTextToWidth } from "./text-metrics.js";

describe("wrapTextToWidth", () => {
  it("returns the whole string as one line when it already fits", () => {
    const lines = wrapTextToWidth("short text", 400, 14);
    expect(lines).toEqual(["short text"]);
  });

  it("wraps onto multiple lines when content exceeds maxWidth", () => {
    const long = "one two three four five six seven eight nine ten";
    const lines = wrapTextToWidth(long, 80, 14);
    expect(lines.length).toBeGreaterThan(1);
    // every produced line must individually fit (or be a single
    // unbreakable word longer than maxWidth, per the next test)
    for (const line of lines) {
      expect(line.length).toBeGreaterThan(0);
    }
    // re-joining with spaces must reconstruct the original words in order
    expect(lines.join(" ")).toBe(long);
  });

  it("does not infinite-loop on a single word longer than maxWidth", () => {
    const lines = wrapTextToWidth("supercalifragilisticexpialidocious", 20, 14);
    expect(lines).toEqual(["supercalifragilisticexpialidocious"]);
  });

  it("returns an empty array for empty input", () => {
    expect(wrapTextToWidth("", 400, 14)).toEqual([]);
  });

  it("respects bold vs normal weight when measuring", () => {
    const text = "aaaa bbbb cccc dddd";
    const normalLines = wrapTextToWidth(text, 100, 14, "normal");
    const boldLines = wrapTextToWidth(text, 100, 14, "bold");
    // bold chars measure the same width unit as normal in this measurer
    // today (BOLD_CHAR_WIDTH === CHAR_WIDTH) — assert the function at
    // least accepts and threads the parameter without throwing, and
    // produces the same line count as normal (documents current
    // equal-width assumption; update this assertion if text-metrics.ts's
    // width constants ever diverge for bold).
    expect(boldLines.length).toBe(normalLines.length);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/explainers && npx vitest run src/core/diagram/text-metrics.test.ts`
Expected: FAIL — `wrapTextToWidth` is not exported yet.

- [ ] **Step 3: Implement `wrapTextToWidth`**

Append to `apps/explainers/src/core/diagram/text-metrics.ts`:

```ts
/**
 * Greedy word-wrap: packs whitespace-separated words onto lines that fit
 * `maxWidth` per `estimateTextWidth`, breaking only between words (never
 * mid-word). A single word wider than `maxWidth` on its own still occupies
 * its own line rather than being split or dropped.
 */
export function wrapTextToWidth(
  text: string,
  maxWidth: number,
  fontSize: number,
  weight: "normal" | "bold" = "normal",
): string[] {
  const words = text.split(/\s+/).filter((word) => word.length > 0);
  if (words.length === 0) {
    return [];
  }

  const lines: string[] = [];
  let current = words[0];

  for (let i = 1; i < words.length; i += 1) {
    const word = words[i];
    const candidate = `${current} ${word}`;
    if (estimateTextWidth(candidate, fontSize, weight) <= maxWidth) {
      current = candidate;
    } else {
      lines.push(current);
      current = word;
    }
  }
  lines.push(current);
  return lines;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/explainers && npx vitest run src/core/diagram/text-metrics.test.ts`
Expected: PASS, all 5 tests green.

- [ ] **Step 5: Full build/test check**

```bash
cd apps/explainers
npx vite build
npx vitest run
```
Expected: both succeed.

- [ ] **Step 6: Commit**

```bash
git add src/core/diagram/text-metrics.ts src/core/diagram/text-metrics.test.ts
git commit --no-verify -m "feat(explainers): add wrapTextToWidth greedy word-wrap utility"
```

---

## Task 2: Wire auto-wrap into SceneRenderer's text render path (Layer 1, global)

**Files:**
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx` (~line 3320-3365, the core text-render branch)

**Interfaces:**
- Consumes: `wrapTextToWidth` from Task 1.

- [ ] **Step 1: Read the current text-render branch in full**

```bash
cd apps/explainers
sed -n '3300,3370p' src/core/diagram/SceneRenderer.tsx
```
Confirm the exact current logic: `content.includes("\n") || node.style?.whiteSpace === "pre"` decides whether to split into `<tspan>` lines; otherwise the whole `content` string renders as one `<text>` child with no splitting.

- [ ] **Step 2: Add the auto-wrap branch**

Change the `textLines` computation so content WITHOUT a manual `\n` and WITHOUT `whiteSpace: "nowrap"` is wrapped to `geom.width` via `wrapTextToWidth`, using the already-computed `fontSize` and the node's font weight (read from `node.style?.fontWeight`, defaulting to `"normal"`; treat any value other than `"normal"` as `"bold"` for measurement purposes, matching how `estimateTextWidth` only distinguishes those two). Preserve the existing manual-`\n` and `whiteSpace: "pre"` behavior exactly as today (those users get their literal lines, no re-wrapping) and add a `whiteSpace: "nowrap"` opt-out that renders the content as a single line even if it would overflow (existing single-`<text>`-child behavior, unchanged) — this is for numeric stats/short kickers where wrapping would look wrong regardless of width.

```tsx
const whiteSpaceStyle = node.style?.whiteSpace;
const hasManualBreaks = content.includes("\n") || whiteSpaceStyle === "pre";
const fontWeight =
  node.style?.fontWeight === "bold" || node.style?.fontWeight === 700
    ? "bold"
    : "normal";
const textLines = hasManualBreaks
  ? content.split("\n")
  : whiteSpaceStyle === "nowrap"
    ? undefined
    : geom.width > 0
      ? wrapTextToWidth(content, geom.width, fontSize, fontWeight)
      : undefined;
```

Import `wrapTextToWidth` at the top of the file alongside the existing `estimateTextWidth`/`scaledSceneFontSize` imports from `./text-metrics.js`.

Note: `wrapTextToWidth` can return an array with a single entry (content
already fit on one line) — this is fine, the existing `<tspan>`-mapping
render code below already handles a 1-element `textLines` array
identically to the previous single-`<text>` path, just wrapped in one
`<tspan>` instead. Verify this visually in Step 4 (a single short label
should render pixel-identical to before, not shifted or double-spaced).

- [ ] **Step 3: Run full test suite**

```bash
cd apps/explainers
npx vite build
npx vitest run
```
Expected: both succeed. Some existing test snapshots/assertions that check exact rendered text-node structure MAY need updating if they assert a bare `content` child rather than a single-line `<tspan>` — if any fail here, inspect whether the failure is a real behavior regression (wrong text, wrong position) or purely the wrapper-tspan structural change is legitimate, expected from this task, and update the test's assertion to match the new (equivalent) structure.

- [ ] **Step 4: Visual regression spot-check across existing decks**

Start `npm run dev` in the background. Using Playwright, spot-check at least 3 EXISTING decks (not the new port) that have short single-line labels/titles/captions across a range of sizes — e.g. `sdk-generic-catalog.flow`'s badge/chip slide, a `rust-architecture.flow` slide, and one more — confirm short text renders identically to before (no unexpected shift, no double line-height gap). Then check one deck with genuinely long paragraph text and confirm it now wraps instead of overflowing. Stop the dev server.

- [ ] **Step 5: Full verification**

```bash
cd apps/explainers
npm run assert:sdk-authoring
npm run flow-verifier
```
Expected: 0 errors. Note the total warning count (compare to the pre-task baseses recorded in `.superpowers/sdd/progress.md`'s deck-port-effort entries) — it should not have increased on any deck.

- [ ] **Step 6: Commit**

```bash
git add src/core/diagram/SceneRenderer.tsx
git commit --no-verify -m "feat(explainers): auto-wrap scene text to its box width by default"
```

---

## Task 3: Auto-grow height for standalone text primitives (paragraph/richText/quote/text/title/note)

**Files:**
- Modify: `apps/explainers/src/flow/sdk/generic/catalog.ts` (the shared text-family factory around line 109-121, 534-539 — read the surrounding function in full first)
- Modify: `apps/explainers/src/flow/sdk/generic/chrome.ts` (`sdk.note`'s `noteFactory`, ~line 739-800)

**Interfaces:**
- Consumes: `wrapTextToWidth` from Task 1.

- [ ] **Step 1: Read the full text-family factory function in `catalog.ts`**

Find the function that consumes the table entries at lines 109-121
(`sdk.text`/`sdk.richText`/`sdk.title`/`sdk.paragraph`/`sdk.quote`) and
produces their `core.text`/`core.group` nodes — read it end to end,
including how it currently sets node height (from the table's fixed
`height` value or an authored `height` prop override).

- [ ] **Step 2: Compute grown height from measured line count**

For each of these 5 components, when the author did NOT explicitly pass a
`height` prop (i.e. the factory would otherwise fall back to the table
default), replace that fallback with a computed height: measure
`wrapTextToWidth(text, width, fontSize, weight).length` (using whatever
`width`/`fontSize` the factory already resolves for that node), multiply
by the same `lineHeight` convention `SceneRenderer` uses
(`fontSize * 1.3`, matching Task 2's `lineHeight` fallback — import or
inline this constant so both stay in sync, do not hardcode a second magic
number), and use `max(computedHeight, table-default-minimum)` so a
single-line specimen never shrinks below what it looked like before this
change. If the author DID pass an explicit `height`, keep respecting it
exactly as today (explicit author intent always wins — this is additive,
not a behavior change for callers who already size things correctly).

- [ ] **Step 3: Apply the same treatment to `sdk.note` in `chrome.ts`**

`noteFactory`'s `geometry.height` currently falls back to
`NOTE_DEFAULT_GEOMETRY.height` when no `height` prop is authored — apply
the identical measured-height-when-unauthored logic from Step 2, using
`geometry.width` and the note's own font size.

- [ ] **Step 4: Run tests**

```bash
cd apps/explainers
npx vite build
npx vitest run
```
Expected: both succeed. If any existing test asserts an exact fixed height
for one of these 5 components or `sdk.note` with NO explicit `height` prop
authored in that test's fixture, and the assertion breaks because the
computed height differs from the old fixed default, update the assertion
to the new correct computed value (verify by hand: word-count the test's
fixture text, divide by how many words fit per line at the fixture's
width, confirm the expected line count matches) rather than reverting the
behavior.

- [ ] **Step 5: Visual check**

Start `npm run dev` in the background. Playwright-check `sdk-generic-catalog.flow`'s own `sdk.paragraph`/`sdk.richText`/`sdk.quote`/`sdk.note` teaching slides (these already have realistic body text) — confirm text no longer clips and box heights look proportionate (not oversized for short text, not clipped for long text). Stop the dev server.

- [ ] **Step 6: Verification**

```bash
cd apps/explainers
npm run assert:sdk-authoring
npm run flow-verifier
```
Expected: 0 errors, warning count not increased anywhere.

- [ ] **Step 7: Commit**

```bash
git add src/flow/sdk/generic/catalog.ts src/flow/sdk/generic/chrome.ts
git commit --no-verify -m "feat(explainers): auto-grow paragraph/richText/quote/text/title/note height to wrapped content"
```

---

## Task 4: Auto-grow + sibling-shift for the 8 deck composites

**Files:**
- Modify: `apps/explainers/src/flow/sdk/generic/deck-composites.ts`

**Interfaces:**
- Consumes: `wrapTextToWidth` from Task 1.

- [ ] **Step 1: Read the full file**

Re-read `deck-composites.ts` in full (it has grown across 2 prior tasks in
the deck-port effort) to refresh the exact geometry math each of the 8
factories uses today for `detail`/`description` text and, for row-based
ones, how subsequent row `y` offsets are computed from a fixed row height
constant.

- [ ] **Step 2: `sdk.compareGrid` / `sdk.cardGrid`**

For each grid cell, measure `wrapTextToWidth(item.detail, cellWidth, detailFontSize).length` and grow that cell's generated box height (and, since the grid is a uniform `layout.grid`, use the TALLEST cell's required height as the shared row height for that grid instance — cells stay uniform within one grid, matching the HTML source's own uniform-height card rows) instead of the current fixed cell height.

- [ ] **Step 3: `sdk.numberedSequence`**

For each row, measure its `detail` text's wrapped line count at that row's
box width, compute that row's needed height, and accumulate a running `y`
offset so each subsequent row starts below the actual (possibly grown)
bottom of the previous row — replacing the current fixed
`ROW_HEIGHT * index` stride math with a running-offset accumulator.

- [ ] **Step 4: `sdk.stepChain`**

Same running-offset treatment as Step 3, but only for `direction: "column"`
mode (row mode's steps sit side-by-side, not stacked, so a taller step's
detail text growing vertically doesn't need to shift a sibling's `y` — it
only needs its own box to grow, which cell-level height growth in Step 2's
pattern already covers; apply that same per-box height growth to row-mode
steps' own boxes without touching their `x` positions).

- [ ] **Step 5: `sdk.sectionDivider`, `sdk.bigStat`, `sdk.nodeTree`, `sdk.timelineAxis`**

These either have no long free-form detail-prose field (`sectionDivider`'s
`subtitle`, `bigStat`'s `description`, `nodeTree`'s `orderNote`,
`timelineAxis`'s tick/marker labels are all short by design/HTML-source
convention) or their layout is graph-shaped rather than stacked-row-shaped.
Apply the Step 2-style "grow this one box's own height" treatment to any
single free-text field each has (subtitle/description/orderNote), but do
NOT attempt sibling-shifting for these four — document in your report
whether you found this unnecessary (no long text exists on these fields in
current usage) or applied a minimal single-box growth fix.

- [ ] **Step 6: Run tests**

```bash
cd apps/explainers
npx vite build
npx vitest run src/flow/sdk/generic/deck-composites.test.ts
npx vitest run
```
Expected: all succeed. Update any existing `deck-composites.test.ts`
assertion that hardcodes an exact fixed height/y-offset for a component
under test, the same way Task 3 handled it — verify by hand the new
computed value is correct, don't just make the assertion match whatever
the code produces.

- [ ] **Step 7: Verification against the full app**

```bash
cd apps/explainers
npm run assert:sdk-authoring
npm run flow-verifier
```
Expected: 0 errors. This is the key check — the `SCENE_VIEWPORT_ESCAPE`/
`SCENE_ABSOLUTE_SIBLING_OVERLAP` warning count on `rust-architecture-deck-port.flow`
specifically (which was 195, entirely attributed to this exact class of
bug per the prior effort's final review) should now be substantially
lower. Report the before/after count.

- [ ] **Step 8: Live visual re-walk of the 49-slide deck**

Start `npm run dev` in the background, Playwright-walk all 49 slides of
the (already hub-wired, per the prior effort) `#/rust-architecture-deck-port`
route, screenshot each, and confirm the previously-known overflow/clip
spots (Task 5's Clock/Drivers label collisions, Task 6's Flow-diagram
final-row clip, Task 7's right-column edge nicks, Task 9's Dynosim/mock-server
overflow, Task 10's BigStat/Invariants overflow — all documented in
`.superpowers/sdd/progress.md`'s deck-port-effort entries) are now resolved
by the auto-grow behavior rather than by the hand-authored workarounds
already in place. Stop the dev server.

- [ ] **Step 9: Commit**

```bash
git add src/flow/sdk/generic/deck-composites.ts
git commit --no-verify -m "feat(explainers): auto-grow deck-composite row/cell heights to wrapped detail text"
```

---

## Task 5: Cleanup pass on rust-architecture-deck-port.flow + final full-app verification

**Files:**
- Modify: `apps/explainers/decks-flow/rust-architecture-deck-port.flow`

**Interfaces:**
- Consumes: Tasks 1-4's wrap/auto-grow behavior.

- [ ] **Step 1: Identify manual workarounds that are now unnecessary**

Search the deck for the hand-rolled patterns the prior effort's reports
documented as workarounds for lack of auto-wrap/auto-grow: manually
split multi-row `Paragraph` stacks (Task 9's Dynosim/mock-server fix,
Task 10's Reporting/Adaptive fix), the two-row `stepChain` split for
Orientation's 6 steps (Task 4), and any other hand-placed coordinate
math whose stated purpose (in a code comment, if one exists, or
inferable from the surrounding structure) was "avoid overflow" rather
than "match the HTML source's actual layout intent."

- [ ] **Step 2: Simplify where it's a clean net win**

For each identified workaround, try reverting to the more natural single-
composite-call authoring (e.g. one `sdk.CompareGrid`/`sdk.NumberedSequence`
call instead of several manually-stacked `Paragraph` rows) now that the
composite auto-grows. Only apply a simplification if the resulting render
is equal-or-better fidelity to `/tmp/deck.html` than the current
workaround — if a simplification looks worse, leave the existing
workaround in place and note why in this task's report. This step is a
quality improvement, not a strict requirement — do not force a
simplification that doesn't actually improve anything.

- [ ] **Step 3: Full final verification (whole app, not just this deck)**

```bash
cd apps/explainers
npm run build
npx vitest run
npm run assert:no-mentalmodel-registry
npm run assert:sdk-authoring
npm run flow-verifier
```
Expected: all 5 exit 0.

- [ ] **Step 4: Final 49-slide + whole-app visual confirmation**

Start `npm run dev` in the background. Playwright-walk all 49 slides of
`rust-architecture-deck-port` once more (post-cleanup) confirming no
regression from Step 2's simplifications, then spot-check 3-4 OTHER
existing decks in the app (any pre-existing deck untouched by this whole
plan) to confirm the Task 2 global wrap change didn't alter their
appearance in any visible way. Stop the dev server.

- [ ] **Step 5: Commit** (only if Step 2 made changes to the `.flow` file)

```bash
git add decks-flow/rust-architecture-deck-port.flow
git commit --no-verify -m "refactor(explainers): simplify deck-port workarounds now that text auto-wraps and boxes auto-grow"
```

If Step 2 found nothing worth simplifying, skip this commit — Task 5 is
then verification-only and the plan is complete as of Task 4's commit.

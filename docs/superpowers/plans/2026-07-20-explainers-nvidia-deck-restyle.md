# Explainers NVIDIA-Deck Restyle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle `apps/explainers` from its current dark, NVIDIA-green-on-graphite diagram-tool look to a flat, boxy, light editorial look matching the reference NVIDIA deck (`/tmp/deck.html`, mirrored from `~/projects/dl/site-mirror`'s captured `NvidiaDeck.dc.html` design canvas file) — white backgrounds, black text, `#76B900` NVIDIA green accents, thin gray borders, zero border-radius everywhere, Manrope for headings/body, Roboto Mono for kickers/labels/code.

**Architecture:** This is a token- and CSS-class-level restyle, not a structural rewrite. `src/core/tokens.ts` is the single source of color/radius truth consumed both by CSS custom properties in `src/index.css` (mirrored by hand per the file's own header comment) and directly by TypeScript diagram-rendering code (`SceneRenderer.tsx`, `FlowArrow.tsx`, `SceneBox.tsx`) via the `tokens` object import. Swapping `tokens.ts` + the `:root` block in `index.css` propagates through nearly everything; the remaining work is (a) loading the new fonts, (b) fixing hardcoded dark-theme rgba literals that don't derive from tokens (chrome overlays, gate backdrop, cinematic stage gradients, shadow-on-dark assumptions), and (c) flattening every `border-radius` declaration to `0`.

**Tech Stack:** React 19 + TypeScript + Vite, plain CSS custom properties (no Tailwind/CSS-in-JS), SVG diagram rendering driven by the `tokens` TS object.

## Global Constraints

- No dark/light toggle — this is a full, one-way replacement of the dark theme. Delete dark-specific values; do not keep them behind a flag.
- Every `border-radius` in `src/index.css` (including the `--ex-radius-*` custom properties) becomes `0` — fully flat, boxy corners, matching the deck reference (user explicitly confirmed this over the initially-proposed "keep small radii" compromise).
- Do not use the `AnthropicSans`/`AnthropicSerif` font files found in `~/projects/dl/site-mirror` (Anthropic proprietary assets, wrong brand anyway). Use Manrope (400/500/700/800) + Roboto Mono (400/500) loaded via Google Fonts `@import`, matching the deck's own fallback stack, in place of Inter + JetBrains Mono.
- Keep `--ex-accent: #76b900` (NVIDIA green) — it already matches the deck's accent color; do not change its hue.
- Preserve all existing class names, component structure, and React component APIs. This is a visual reskin of `src/index.css` + `src/core/tokens.ts` only — no JSX/TSX structural changes, except where a hardcoded dark-theme color literal must become a token reference or a new literal light-theme value.
- SPDX header (`Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES`, `Apache-2.0`) stays unchanged at the top of every touched file — do not remove or alter it.
- Run `npm run build` (`tsc --noEmit && vite build`) and `npm test` (`vitest run`) from `apps/explainers/` after each task; both must stay green. `npm run assert:no-mentalmodel-registry`, `npm run assert:sdk-authoring`, and `npm run flow-verifier` are unaffected by a pure CSS/token restyle and do not need to be run per-task, but run them once at the end (Task 6) as a final gate.
- Reference deck markup (for exact colors/values to match) is at `/tmp/deck.html` — treat this as read-only reference material, never copy its markup into the React app.

---

## File Structure

| File | Responsibility |
|---|---|
| `apps/explainers/index.html` | Add the Google Fonts `@import` preconnect/link tags (or leave the `@import` in CSS — see Task 1) for Manrope + Roboto Mono. |
| `apps/explainers/src/core/tokens.ts` | Single source of truth for the new light color palette, radius values (all `0`), and any new tokens (`accent.tint`, `text.onAccent` adjustment). |
| `apps/explainers/src/index.css` | `:root` custom properties mirroring `tokens.ts`; every component class (`.ex-card`, `.ex-btn`, `.ex-panel`, `.ex-pill`, `.ex-stepper`, `.ex-chrome`, `.ex-gate`, `.ex-speaker-notes`, `.explainer-final-card`, etc.) gets its colors/radii/shadows updated to the light/flat language; hardcoded dark rgba literals not derived from a custom property get replaced with light equivalents. |
| `apps/explainers/src/core/diagram/SceneRenderer.tsx` | Consumes `tokens` for fills/strokes/text — verify post-swap rendering; adjust the three hardcoded `rgba(0, 0, 0, ...)` drop-shadow literals (lines ~3612-3660) so shadows still read correctly on a white stage instead of a dark one. |

No new files are created; no files are deleted.

---

## Task 1: Swap color/radius tokens and load new fonts

**Files:**
- Modify: `apps/explainers/src/core/tokens.ts`
- Modify: `apps/explainers/src/index.css:1-73` (the `:root` block + top-of-file font-family declaration)
- Test: `apps/explainers/src/core/*.test.ts` / `*.test.tsx` (existing suite — none should reference specific hex values, but run full suite to confirm)

**Interfaces:**
- Produces: the `tokens` object shape is unchanged (same keys: `text`, `bg`, `fill`, `stroke`, `accent`, `category`, `radius`, `diagram`), only values change, plus one new key `accent.tint`. Every downstream consumer (`SceneRenderer.tsx`, `FlowArrow.tsx`, `SceneBox.tsx`, CSS) keeps working against the same key names.

- [ ] **Step 1: Rewrite `tokens.ts` with the light palette**

Replace the full `tokens` object body in `apps/explainers/src/core/tokens.ts`:

```ts
export const tokens = {
  text: {
    primary: "#000000",
    secondary: "#555555",
    tertiary: "#A7A7A7",
    quaternary: "#C4C4C4",
    link: "#3D6B00",
    onAccent: "#000000",
  },
  bg: {
    page: "#FFFFFF",
    chrome: "#FAFAFA",
    elevated: "#FFFFFF",
    panel: "#F7F7F7",
  },
  fill: {
    primary: "rgba(15, 12, 8, 0.08)",
    secondary: "rgba(15, 12, 8, 0.05)",
    tertiary: "rgba(15, 12, 8, 0.03)",
    quaternary: "rgba(15, 12, 8, 0.015)",
  },
  stroke: {
    primary: "#111111",
    secondary: "#E4E4E4",
    tertiary: "#EFEFEF",
  },
  accent: {
    primary: "#76B900",
    tint: "#F2F7EA",
    control: "#3987A6",
  },
  category: {
    green: "#5E8A1F",
    yellow: "#B08A1E",
    purple: "#6E4FA6",
    blue: "#2A78D6",
    red: "#A63244",
    orange: "#B05E2A",
    cyan: "#3987A6",
    gray: "#7E838E",
  },
  radius: {
    control: 0,
    card: 0,
    stage: 0,
    pill: 0,
    box: 0,
  },
  diagram: {
    strokeWidth: 1.6,
    dashed: "6 5",
  },
} as const;

export type Tokens = typeof tokens;
```

Also update the module doc comment above `export const tokens` to describe the new skin:

```ts
/**
 * Visual tokens for explainers chrome and diagram defaults.
 * Keep hex values in sync with `:root` custom properties in `index.css`.
 *
 * NVIDIA-deck skin: flat white slides, NVIDIA green accent, boxy corners.
 * Matches the reference deck's editorial print language, not a dark app chrome.
 */
```

- [ ] **Step 2: Rewrite the `:root` block in `index.css`**

Replace lines 10-73 of `apps/explainers/src/index.css`:

```css
:root {
  color-scheme: light;

  --ex-text-primary: #000000;
  --ex-text-secondary: #555555;
  --ex-text-tertiary: #a7a7a7;
  --ex-text-quaternary: #c4c4c4;
  --ex-text-link: #3d6b00;
  --ex-text-on-accent: #000000;

  --ex-bg-page: #ffffff;
  --ex-bg-chrome: #fafafa;
  --ex-bg-elevated: #ffffff;
  --ex-bg-panel: #f7f7f7;

  --ex-fill-primary: rgba(15, 12, 8, 0.08);
  --ex-fill-secondary: rgba(15, 12, 8, 0.05);
  --ex-fill-tertiary: rgba(15, 12, 8, 0.03);
  --ex-fill-quaternary: rgba(15, 12, 8, 0.015);

  --ex-stroke-primary: #111111;
  --ex-stroke-secondary: #e4e4e4;
  --ex-stroke-tertiary: #efefef;

  --ex-accent: #76b900;
  --ex-accent-soft: #5e9400;
  --ex-accent-control: #3987a6;
  --ex-accent-done: #5e8a1f;
  --ex-accent-tint: #f2f7ea;
  --ex-category-green: #5e8a1f;
  --ex-category-yellow: #b08a1e;
  --ex-category-purple: #6e4fa6;
  --ex-category-blue: #2a78d6;
  --ex-category-red: #a63244;
  --ex-category-orange: #b05e2a;
  --ex-category-cyan: #3987a6;
  --ex-category-gray: #7e838e;

  --ex-radius-control: 0px;
  --ex-radius-card: 0px;
  --ex-radius-stage: 0px;
  --ex-radius-pill: 0px;

  --ex-grid-line: rgba(15, 12, 8, 0);
  --ex-grid-size: 52px;
  --ex-mono: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --ex-chrome-fade-ms: 320ms;

  font-family:
    Manrope,
    ui-sans-serif,
    system-ui,
    -apple-system,
    "Segoe UI",
    Roboto,
    Helvetica,
    Arial,
    sans-serif;
  line-height: 1.5;
  font-weight: 400;
  font-size: 15px;
  background: var(--ex-bg-page);
  color: var(--ex-text-primary);
  -webkit-font-smoothing: antialiased;
}
```

Note `--ex-grid-line` is set to fully transparent (`rgba(15, 12, 8, 0)`) rather than deleted — the `.ex-page` and `.ex-gate` background-image rules reference it and are addressed in Task 3; setting it transparent here is a safe intermediate state so Task 1 alone doesn't break rendering.

- [ ] **Step 3: Add the Google Fonts import**

At the very top of `apps/explainers/src/index.css`, immediately after the SPDX header block (before the `/* CSS custom properties mirror... */` comment), add:

```css
@import url("https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Roboto+Mono:wght@400;500&display=swap");
```

(A CSS `@import` must precede all other rules except `@charset`; placing it right after the file's leading comment block satisfies this — CSS comments don't count as rules.)

- [ ] **Step 4: Verify build and tests**

Run:
```bash
cd apps/explainers
npm run build
npm test
```
Expected: both succeed. `tsc --noEmit` catches any TS error from the `tokens.ts` change; `vitest run` catches any snapshot/assertion tied to old hex values (none expected, since the codebase reads through the `tokens` object rather than hardcoding).

- [ ] **Step 5: Commit**

```bash
git add apps/explainers/src/core/tokens.ts apps/explainers/src/index.css
git commit -m "style(explainers): swap dark NVIDIA-green tokens for light NVIDIA-deck palette"
```

---

## Task 2: Flatten every remaining `border-radius` and fix radius-adjacent literals

**Files:**
- Modify: `apps/explainers/src/index.css` (search for every literal `border-radius` not already driven by an `--ex-radius-*` var, plus the `.ex-hub-mark__bar` inline `border-radius: 1px`, `.ex-progress__fill` `border-radius: 999px`, `.ex-subtitles__word` `border-radius: 3px`, `.ex-term` `border-radius: 4px`, `.ex-more` / `.ex-subtitles` `border-radius: 4px`)

**Interfaces:**
- Consumes: `--ex-radius-control`, `--ex-radius-card`, `--ex-radius-stage`, `--ex-radius-pill` (already `0px` from Task 1).

- [ ] **Step 1: Grep for every remaining hardcoded `border-radius` value**

```bash
cd apps/explainers
grep -n "border-radius" src/index.css
```

Expected output includes these literal (non-var) declarations that Task 1 did not touch:
- `.ex-hub-mark__bar { border-radius: 1px; }` (line ~197)
- `.ex-term { border-radius: 4px; }` (line ~681)
- `.ex-subtitles { border-radius: 4px; }` (line ~750)
- `.ex-subtitles__word { border-radius: 3px; }` (line ~783)
- `.ex-more { border-radius: 4px; }` (line ~805)
- `.ex-progress__fill { border-radius: 999px; }` (line ~1353)
- `.ex-gate__card`, `.ex-panel`, `.ex-card`, `.ex-alert`, `.ex-code`, `.ex-btn`, `.ex-pill`, `.ex-pill-group`, `.ex-stepper__dot`, `.ex-speaker-notes` already use `var(--ex-radius-*)` and are already `0` from Task 1 — confirm with the grep, no edit needed for these.

- [ ] **Step 2: Zero out the remaining literal radii**

Edit each of the six literal declarations found in Step 1 to `border-radius: 0;`. For example:

```css
.ex-hub-mark__bar {
  display: block;
  width: 3px;
  border-radius: 0;
  background: var(--ex-accent);
}
```

Apply the equivalent change to `.ex-term`, `.ex-subtitles`, `.ex-subtitles__word`, `.ex-more`, and `.ex-progress__fill`.

- [ ] **Step 3: Confirm no radius survives**

```bash
grep -n "border-radius" src/index.css | grep -v "border-radius: 0"
```

Expected: no output (empty).

- [ ] **Step 4: Verify build**

```bash
npm run build
```
Expected: success, no new errors.

- [ ] **Step 5: Commit**

```bash
git add apps/explainers/src/index.css
git commit -m "style(explainers): flatten every remaining border-radius to 0"
```

---

## Task 3: Fix hardcoded dark-theme rgba literals (page grid, gate backdrop, cinematic stage, chrome overlays, shadows)

**Files:**
- Modify: `apps/explainers/src/index.css` (`.ex-page`, `.ex-hub::before`, `.ex-shell` radial gradient, `.ex-cinematic-stage.ex-content-card` gradients, `.ex-stage-copy.ex-content-card__copy` gradient, `.ex-gate` background, `.ex-chrome--top` gradient, `.ex-chrome--bottom` backdrop, `.ex-notes-toggle` backdrop, `.ex-speaker-notes` shadow/backdrop, `.explainer-final-card` background, `.ex-content-card` box-shadow/inset, `.ex-stage-hero.ex-content-card__diagram svg` filter)

**Interfaces:**
- Consumes: `--ex-bg-page` (`#ffffff`), `--ex-accent` (`#76b900`), `--ex-stroke-secondary` (`#e4e4e4`) from Task 1.

This task hand-fixes every place in `index.css` where a color was written as a raw `rgba(8, 9, 11, ...)` / `rgba(255, 255, 255, ...)` / `rgba(0, 0, 0, ...)` literal tuned for a *dark* backdrop, rather than through a token. These don't auto-correct from the Task 1 token swap and will look wrong (near-invisible overlays, inverted gradients, shadows with no contrast) on a white background if left as-is.

- [ ] **Step 1: Remove the page/gate grid background** (the deck reference has a flat white page, no grid)

In `.ex-page` (around line 98), remove the `background-image`/`background-size`/`background-position` grid declarations, leaving only the flat background:

```css
.ex-page {
  min-height: 100%;
  min-height: 100dvh;
  padding: 10px 18px 14px;
  color: var(--ex-text-primary);
  background-color: var(--ex-bg-page);
}
```

Apply the same removal to `.ex-gate` (around line 1026):

```css
.ex-gate {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  padding: 24px;
  background-color: rgba(255, 255, 255, 0.97);
}
```

- [ ] **Step 2: Fix `.ex-hub::before` sweep line and `.ex-shell` radial glow**

`.ex-hub::before` already uses `color-mix(in srgb, var(--ex-accent) ...)` — no change needed there.

`.ex-shell` (around line 1183) has a hardcoded radial gradient tuned for dark:
```css
background:
  radial-gradient(circle at 62% 44%, rgba(118, 185, 0, 0.045), transparent 36%),
  var(--ex-bg-page);
```
Change to a lighter-touch version appropriate for white (keep the green glow subtle, don't remove the effect):
```css
background:
  radial-gradient(circle at 62% 44%, rgba(118, 185, 0, 0.06), transparent 36%),
  var(--ex-bg-page);
```

- [ ] **Step 3: Fix the cinematic stage gradients**

`.ex-cinematic-stage.ex-content-card` (around line 1209) has:
```css
background:
  linear-gradient(180deg, rgba(8, 9, 11, 0.2), rgba(8, 9, 11, 0) 24%),
  radial-gradient(circle at 52% 42%, rgba(241, 242, 244, 0.03), transparent 54%);
```
This darkens the top of the stage and adds a light haze — both were tuned for a dark base. Replace with a version that darkens the top slightly *less* (avoid muddying white) and drops the light-haze (a white haze on white does nothing useful):
```css
background: linear-gradient(180deg, rgba(15, 12, 8, 0.04), rgba(15, 12, 8, 0) 24%);
```

`.ex-stage-copy.ex-content-card__copy` (around line 1282) has a fade-to-opaque-dark scrim behind the caption text:
```css
background: linear-gradient(90deg, rgba(8, 9, 11, 0.96), rgba(8, 9, 11, 0.7) 66%, transparent);
```
Change the base color to white so the same fade behavior (opaque near the text, transparent further out) works on the light stage:
```css
background: linear-gradient(90deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.7) 66%, transparent);
```

- [ ] **Step 4: Fix chrome bar gradients and blurred panel backdrops**

`.ex-chrome--top` (around line 1319):
```css
background: linear-gradient(180deg, rgba(8, 9, 11, 0.96), rgba(8, 9, 11, 0.64), transparent);
```
becomes:
```css
background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 255, 255, 0.64), transparent);
```

`.ex-chrome--bottom` (around line 1376) uses `color-mix(in srgb, var(--ex-bg-chrome) 88%, transparent)` — already token-driven, no change needed, but add a visible border since a near-white translucent bar needs one to read against a white page (it currently has `border: 1px solid var(--ex-stroke-tertiary)` already — confirm it's present and leave as-is).

`.ex-notes-toggle` (around line 1404) uses `color-mix(in srgb, var(--ex-bg-chrome) 92%, transparent)` — already token-driven, no change needed.

`.ex-speaker-notes` (around line 1414):
```css
background: color-mix(in srgb, var(--ex-bg-panel) 94%, transparent);
box-shadow: -24px 0 80px rgba(0, 0, 0, 0.42);
```
The background is already token-driven. Soften the shadow for a light panel (a shadow this heavy reads as a dark vignette on white):
```css
box-shadow: -24px 0 48px rgba(15, 12, 8, 0.14);
```

- [ ] **Step 5: Fix the final-card background and content-card shadow**

`.explainer-final-card` (around line 1465) already uses `var(--ex-bg-page)` — no change needed, comment stays accurate.

`.ex-content-card` (around line 597) box-shadow:
```css
box-shadow:
  inset 0 1px 0 rgba(255, 255, 255, 0.018),
  0 18px 60px rgba(0, 0, 0, 0.16);
```
The inset highlight (`rgba(255,255,255,...)`) simulated a top-edge highlight on a dark card; on a white card it's invisible and can be dropped. Soften the drop shadow to match the deck's own card shadow language (thin border + very soft shadow):
```css
box-shadow: 0 1px 3px rgba(15, 12, 8, 0.08);
```
Also remove the now-redundant highlight gradient in the `background` property of the same rule:
```css
background: var(--ex-bg-elevated);
```
(replacing `linear-gradient(135deg, rgba(255, 255, 255, 0.018), transparent 42%), var(--ex-bg-elevated)`).

`.ex-gate__card` (around line 1040) box-shadow:
```css
box-shadow: 0 24px 90px rgba(0, 0, 0, 0.34);
```
becomes:
```css
box-shadow: 0 12px 40px rgba(15, 12, 8, 0.12);
```

`.ex-stage-hero.ex-content-card__diagram svg` (around line 1269) filter:
```css
filter: drop-shadow(0 24px 54px rgba(0, 0, 0, 0.32));
```
Soften for a light stage:
```css
filter: drop-shadow(0 12px 28px rgba(15, 12, 8, 0.14));
```

- [ ] **Step 6: Verify build**

```bash
cd apps/explainers
npm run build
```
Expected: success.

- [ ] **Step 7: Commit**

```bash
git add apps/explainers/src/index.css
git commit -m "style(explainers): retune dark-theme rgba overlays and shadows for the white stage"
```

---

## Task 4: Fix `SceneRenderer.tsx` shadow literals for the light diagram stage

**Files:**
- Modify: `apps/explainers/src/core/diagram/SceneRenderer.tsx:3612-3613,3660`

**Interfaces:**
- Consumes: nothing new — these are standalone string literals inside existing filter-string-building expressions.

- [ ] **Step 1: Read the current context around each literal**

```bash
cd apps/explainers
sed -n '3600,3665p' src/core/diagram/SceneRenderer.tsx
```

Confirm the three drop-shadow filter strings:
- Line ~3612: `` `drop-shadow(0 8px 14px rgba(0, 0, 0, 0.4)) drop-shadow(0 0 8px color-mix(in srgb, ${strokePaint} 45%, transparent))` `` (used when a node/box is in an elevated/active state)
- Line ~3613: `"drop-shadow(0 6px 10px rgba(0, 0, 0, 0.3))"` (default state)
- Line ~3660: `"drop-shadow(0 5px 7px rgba(0, 0, 0, 0.32))"` (a different element's shadow)

- [ ] **Step 2: Soften the three literals for a white background**

Black shadows at 0.3-0.4 alpha read as heavy smudges on white (they were tuned for a near-black `#08090b` page where a 30-40% black shadow blends into the surrounding darkness). Replace with values roughly a third as strong, using the same warm-black `rgba(15, 12, 8, ...)` used elsewhere in this plan for consistency:

```ts
            ? `drop-shadow(0 8px 14px rgba(15, 12, 8, 0.16)) drop-shadow(0 0 8px color-mix(in srgb, ${strokePaint} 45%, transparent))`
            : "drop-shadow(0 6px 10px rgba(15, 12, 8, 0.12))",
```

and:
```ts
                  "drop-shadow(0 5px 7px rgba(15, 12, 8, 0.13))",
```

Edit these three lines in place, preserving the exact surrounding ternary/ object-literal structure — only the rgba color/alpha values inside each string change.

- [ ] **Step 3: Verify build and tests**

```bash
npm run build
npm test
```
Expected: both succeed. `SceneRenderer.sdk-primitives.test.tsx` exercises this file; confirm it still passes (it does not assert on exact filter strings today, so no test edit is expected — if it does fail on a literal string match, update the expected string in the test to the new value).

- [ ] **Step 4: Commit**

```bash
git add apps/explainers/src/core/diagram/SceneRenderer.tsx
git commit -m "style(explainers): soften SceneRenderer drop-shadows for the light stage"
```

---

## Task 5: Visual verification pass with the running dev server

**Files:** none (verification only, using the `run` skill / manual browser check)

**Interfaces:** none.

- [ ] **Step 1: Start the dev server**

```bash
cd apps/explainers
npm run dev
```
Expected: Vite prints a local URL (typically `http://localhost:5173`).

- [ ] **Step 2: Load the hub page and at least one explainer deck in a browser**

Navigate to the printed local URL. Confirm:
- Page background is flat white, no grid pattern.
- Body/heading text renders in Manrope (check via devtools computed font-family, or visually confirm a geometric sans rather than Inter).
- Kicker/mono labels (eyebrow text, stepper labels, code) render in Roboto Mono.
- `.ex-card` hub tiles have square (non-rounded) corners, thin `#E4E4E4`-ish border, and a green top accent line on hover.
- Buttons/pills/panels all have square corners.
- Open at least one explainer flow and confirm the diagram stage (SVG) renders legibly with dark strokes/text on the white stage — no washed-out or invisible elements from leftover dark-tuned opacities.
- Toggle presentation mode (if available in the UI) and confirm the chrome bars (top bar, progress bar, bottom control bar) are legible against white.

- [ ] **Step 3: Stop the dev server**

Terminate the `npm run dev` process (Ctrl-C or kill the background task).

- [ ] **Step 4: No commit** — this task is verification-only, no files change. If Step 2 surfaces a defect, return to the relevant earlier task, fix it there, and re-run that task's build/test/commit steps.

---

## Task 6: Final full verification gate

**Files:** none (verification only)

**Interfaces:** none.

- [ ] **Step 1: Run the full explainers check suite**

```bash
cd apps/explainers
npm run build
npm test
npm run assert:no-mentalmodel-registry
npm run assert:sdk-authoring
npm run flow-verifier
```
Expected: all five commands exit 0.

- [ ] **Step 2: Confirm no dark-theme residue remains**

```bash
grep -n "#08090b\|#0d0e12\|#14151a\|#20242a\|#f1f2f4\|#a7aab4\|Inter\|JetBrains" src/index.css src/core/tokens.ts
```
Expected: no output (empty) — every old dark hex and old font name has been removed. If anything remains, fix it in the file it's found in and re-run Step 1.

- [ ] **Step 3: Final commit (only if Step 2 required fixes)**

```bash
git add apps/explainers/src/index.css apps/explainers/src/core/tokens.ts
git commit -m "style(explainers): remove residual dark-theme color/font literals"
```

If Step 2 found nothing, no commit is needed for this task — the plan is complete as of Task 4's commit plus Task 5's verification.

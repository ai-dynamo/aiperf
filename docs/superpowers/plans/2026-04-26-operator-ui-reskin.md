# Operator UI Re-skin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-skin the operator UI (`src/aiperf/operator/ui/`) to match dashboard-v2's vocabulary — rounded dark cards, IBM Plex Mono + JetBrains Mono, drop the WORKBENCH/FAULT/CALLSIGN flight-deck theme — while preserving every route, every `data-testid`, every keymap, and every fetch.

**Architecture:** Replace the operator UI's `style.css` foundation with dashboard-v2's tokens; re-skin all chrome (TopRail, LogStrip, CommandPalette, bench-error-flash) and views (`home`, `archive`, `analysis`, `compare`, `log`, `launch`) by class swaps + drop-the-shouting; structurally rewrite `views/run.js`'s markup blocks (header → hero, meter bay → KPI tiles, phase swimlane → phase cards, sparklines, fault callout, etc.) to mirror v2's component patterns inline. Re-tune `chart-theme.js` to v2's palette and tooltip style.

**Tech Stack:** Preact + htm + @preact/signals (pinned via importmap), Chart.js 4, IBM Plex Mono + JetBrains Mono (Google Fonts), Phosphor icon font.

**Reference:**
- Spec: [`docs/superpowers/specs/2026-04-25-operator-ui-reskin-design.md`](../specs/2026-04-25-operator-ui-reskin-design.md)
- Source of truth for the look: `src/aiperf/api/static-v2/`

**Conventions every task obeys:**
- Edit only the files listed in that task. Don't pre-touch later tasks' files.
- Preserve **every existing `data-testid` attribute** verbatim. The set is enumerated in the per-task "Test ID inventory" lists.
- Preserve every existing keyboard shortcut and route handler (no JS behavior changes outside the explicit Run-view rewrite).
- Drop ALL-CAPS shoutmarks ("FAULT", "CALLSIGN", "WORKBENCH", "FLIGHT DECK", "MCC", etc.) in user-facing copy. Tiny labels (KPI tile titles, table headers, eyebrow text) stay uppercase as visual tokens via CSS `text-transform`, not via raw uppercase strings.
- After each task: `pre-commit run --all-files` (no `--no-verify` here — formatting matters), then `uv run pytest -n auto tests/unit/`. Commit with `git commit -s --no-verify` (project HEAD has fmt drift outside our diff per `feedback_commit_with_no_verify.md`).
- Commit on the **current branch** — do not spin off a new branch.

---

## File structure overview

| File | Role after this plan |
| --- | --- |
| `src/aiperf/operator/ui/index.html` | Title + fonts + importmap. Same DOM hooks. |
| `src/aiperf/operator/ui/style.css` | Single stylesheet. New v2-foundation tokens, no grid-paper substrate, all radii 4–8px. Target ~1.6–2k lines. |
| `src/aiperf/operator/ui/lib/theme.js` | JS-emitted color palette consumed by Chart.js callsites. Tokens align with new CSS. |
| `src/aiperf/operator/ui/lib/chart-theme.js` | Idempotent Chart.js defaults — v2 tooltip + grid + palette. |
| `src/aiperf/operator/ui/components/top-rail.js` | Re-skinned topbar + breadcrumb + ⌘K pill + Launch CTA. Same data hooks. |
| `src/aiperf/operator/ui/components/log-strip.js` | LogPane look (filter pills + category chips). Same source. |
| `src/aiperf/operator/ui/components/command-palette.js` | Re-skinned modal. |
| `src/aiperf/operator/ui/components/chart-wrapper.js` | Class swaps only. |
| `src/aiperf/operator/ui/views/home.js` | Re-skinned. |
| `src/aiperf/operator/ui/views/archive.js` | Re-skinned. "FAULT" → "Failed". |
| `src/aiperf/operator/ui/views/analysis.js` | Re-skinned. |
| `src/aiperf/operator/ui/views/compare.js` | Re-skinned. |
| `src/aiperf/operator/ui/views/log.js` | Re-skinned + filter pills. |
| `src/aiperf/operator/ui/views/launch.js` | Re-skinned + primary/ghost button styles. |
| `src/aiperf/operator/ui/views/run.js` | **Structural rewrite** of presentational markup. Same data flow, same testids. |
| `src/aiperf/operator/ui/app.js` | "Error" copy in `bench-error-flash` (one-line edit). |
| `src/aiperf/operator/ui/ui-v1/**` | **Untouched.** |
| `src/aiperf/api/static-v2/**` | **Untouched.** |
| `tests/**` | **Untouched.** All existing tests must continue to pass. |

**Test surface to keep green:**
- `uv run pytest -n auto tests/unit/` — must pass after every task.
- `pre-commit run --all-files` — must pass after every task.
- The e2e suite under `tests/e2e/operator_ui/` is **not** run automatically here (it requires a live cluster), but every selector it relies on is preserved by the testid-preservation rule.

---

## Task 1: Foundation — `style.css`, `index.html`, `lib/theme.js`

**Goal:** Establish the v2 token system and base shell rules. After this task the operator UI will look broken (most class-name styles are still old), but the body / topbar / cards / scrollbar / fonts / buttons / chips primitives all render in the new vocabulary.

**Files:**
- Modify: `src/aiperf/operator/ui/index.html`
- Modify: `src/aiperf/operator/ui/style.css` (full rewrite of the foundation block; keep the legacy class-name rules below the foundation for now — Tasks 3–10 retire them as they're replaced by re-skinned markup)
- Modify: `src/aiperf/operator/ui/lib/theme.js`

- [ ] **Step 1: Replace `index.html` head**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="theme-color" content="#0c0c0c">
  <title>AIPerf Operator</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="./style.css">

  <script type="importmap">
  {
    "imports": {
      "preact": "https://esm.sh/preact@10",
      "preact/hooks": "https://esm.sh/preact@10/hooks",
      "htm/preact": "https://esm.sh/htm@3/preact",
      "@preact/signals": "https://esm.sh/@preact/signals@1?deps=preact@10"
    }
  }
  </script>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <script src="https://unpkg.com/@phosphor-icons/web@2"></script>
</head>
<body>
  <div id="app"></div>
  <script type="module" src="./app.js"></script>
</body>
</html>
```

(Drops `IBM Plex Sans` and `Host Grotesk`. Keeps Phosphor for the icon glyphs the existing views use.)

- [ ] **Step 2: Rewrite the `:root` block + base shell rules at the top of `style.css`**

Replace **everything from the start of the file up through (and including) the `#app::after` vignette rule** with the block below. Leave everything after the `Reset` section's `::after` vignette in place for now — it gets retired by later tasks as their markup is re-skinned.

```css
/* SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0 */

/* AIPerf Operator — design tokens lifted from src/aiperf/api/static-v2/style.css
   so both dashboards share a single visual language. */
:root {
  --bg: #0c0c0c;
  --bg-card: #161616;
  --bg-raised: #222222;
  --bg-tile: #0f0f0f;
  --bg-mid: #1a1a1a;
  --border: #313131;
  --border-hover: #4b4b4b;
  --border-subtle: #1a1a1a;

  --dim: #4b4b4b;
  --muted: #757575;
  --sub: #a7a7a7;
  --text: #eeeeee;
  --white: #ffffff;

  --accent: #76b900;
  --accent-hot: #8ce200;
  --accent-deep: #5a8e00;
  --accent-dim: rgba(118, 185, 0, 0.15);

  --blue:  #3b82f6;  --blue-dim:  rgba(59, 130, 246, 0.15);
  --cyan:  #26c6da;  --cyan-dim:  rgba(38, 198, 218, 0.15);
  --green: #76b900;  --green-dim: rgba(118, 185, 0, 0.15);
  --amber: #ffc107;  --amber-dim: rgba(255, 193, 7, 0.15);
  --red:   #ef5350;  --red-dim:   rgba(239, 83, 80, 0.15);
  --pink:  #ab47bc;  --pink-dim:  rgba(171, 71, 188, 0.15);

  --font-sans: 'IBM Plex Mono', ui-monospace, monospace;
  --font-mono: 'JetBrains Mono', monospace;

  --font-xs:   0.75rem;
  --font-sm:   0.8125rem;
  --font-base: 0.875rem;
  --font-md:   1rem;
  --font-lg:   1.125rem;
  --font-xl:   1.25rem;
  --font-2xl:  1.5rem;

  --top-bar-height: 48px;
  --log-strip-height: 180px;

  --radius:    6px;
  --radius-sm: 4px;
  --radius-lg: 8px;

  --space-1: 0.25rem;  --space-2: 0.5rem;
  --space-3: 0.75rem;  --space-4: 1rem;
  --space-5: 1.25rem;  --space-6: 1.5rem;
  --space-7: 1.75rem;  --space-8: 2rem;

  --transition-fast: 120ms ease;
  --transition:      200ms ease;
  --transition-slow: 320ms ease;

  --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.45);
  --shadow-md: 0 6px 18px rgba(0, 0, 0, 0.55);
  --shadow-lg: 0 14px 36px rgba(0, 0, 0, 0.65);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
img, svg { display: block; max-width: 100%; }
button { font: inherit; color: inherit; background: none; border: none; cursor: pointer; text-align: left; }
a { color: inherit; text-decoration: none; }
a:hover { text-decoration: underline; }
input, select, textarea { font: inherit; color: inherit; }
ol, ul { list-style: none; }

body {
  font-family: var(--font-sans);
  font-size: var(--font-base);
  line-height: 1.5;
  color: var(--text);
  background: var(--bg);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

#app {
  min-height: 100vh;
  isolation: isolate;
  position: relative;
}

h1, h2, h3, h4 {
  font-weight: 600;
  line-height: 1.3;
  color: var(--white);
}

.text-dim    { color: var(--dim); }
.text-muted  { color: var(--muted); }
.text-sub    { color: var(--sub); }
.text-accent { color: var(--accent); }

/* ───── App frame (bench-* shell) ───── */
.bench {
  display: grid;
  grid-template-rows: var(--top-bar-height) 1fr var(--log-strip-height);
  grid-template-areas:
    "rail"
    "main"
    "log";
  min-height: 100vh;
}
.bench[data-route] .bench-main { grid-area: main; min-height: 0; overflow: auto; padding: 16px; }
.bench-error-flash {
  background: rgba(239, 83, 80, 0.08);
  border: 1px solid rgba(239, 83, 80, 0.45);
  border-radius: var(--radius-lg);
  padding: 12px 16px;
  margin-bottom: 16px;
  color: var(--text);
  font-size: var(--font-sm);
}
.bench-error-flash strong { color: var(--red); margin-right: 8px; text-transform: uppercase; letter-spacing: 0.06em; font-size: var(--font-xs); }

/* ───── Generic primitives — reused everywhere ───── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: var(--space-4);
}
.card-title {
  font-size: var(--font-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--dim);
  margin-bottom: var(--space-3);
}

.btn {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--font-sans);
  font-size: var(--font-xs);
  font-weight: 600;
  letter-spacing: 0.04em;
  padding: 6px 12px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: transparent;
  color: var(--sub);
  cursor: pointer;
  transition: border-color var(--transition-fast), background var(--transition-fast), color var(--transition-fast);
}
.btn:hover { border-color: var(--border-hover); color: var(--text); }
.btn--primary {
  background: var(--accent);
  color: var(--bg);
  border-color: var(--accent);
}
.btn--primary:hover { background: var(--accent-hot); border-color: var(--accent-hot); color: var(--bg); }
.btn--ghost { background: transparent; border-color: var(--border); }
.btn--danger { color: var(--red); border-color: var(--red); }

.kbd {
  display: inline-flex; align-items: center; gap: 4px;
  font-family: var(--font-mono);
  font-size: var(--font-xs);
  color: var(--muted);
  background: rgba(75, 75, 75, 0.2);
  border: 1px solid var(--border);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
}

.chip {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: var(--font-xs);
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  letter-spacing: 0.04em;
}
.chip--good { background: var(--green-dim); color: var(--green); }
.chip--warn { background: var(--amber-dim); color: var(--amber); }
.chip--bad  { background: var(--red-dim);   color: var(--red); }
.chip--info { background: var(--blue-dim);  color: var(--blue); }
.chip--neutral { background: rgba(75, 75, 75, 0.18); color: var(--muted); }

.status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  flex-shrink: 0;
}
.status-dot--ok    { background: var(--green); box-shadow: 0 0 8px rgba(118, 185, 0, 0.5); animation: pulse 2s infinite; }
.status-dot--warn  { background: var(--amber); }
.status-dot--bad   { background: var(--red); }
.status-dot--idle  { background: var(--muted); }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.5; }
}

/* ───── Scrollbar (replaces the legacy edge-amber one) ───── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-hover); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--dim); }

/* ───── Empty state ───── */
.empty {
  padding: var(--space-6);
  text-align: center;
  color: var(--muted);
  font-style: italic;
}
```

Below this block, the **rest of the existing `style.css` stays in place for now** so the legacy class names (`.rail-*`, `.run-*`, `.hm-*`, `.arch-*`, `.v-launch`, …) still render something acceptable while later tasks replace them. After the final run-view task, do a sweep to delete any rules that are no longer referenced (Task 10 includes that step).

- [ ] **Step 3: Replace `lib/theme.js` exports**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// AIPerf Operator — JS-side palette aligned to the dashboard-v2 token system
// in style.css. Used by Chart.js callsites (see lib/chart-theme.js) and any
// inline-style consumer that needs a JS literal for a CSS var.

export const palette = {
  bg: '#0c0c0c',
  bgCard: '#161616',
  bgRaised: '#222222',
  bgTile: '#0f0f0f',
  bgMid: '#1a1a1a',

  border: '#313131',
  borderHover: '#4b4b4b',
  borderSubtle: '#1a1a1a',

  dim: '#4b4b4b',
  muted: '#757575',
  sub: '#a7a7a7',
  text: '#eeeeee',
  white: '#ffffff',

  accent: '#76b900',
  accentHot: '#8ce200',
  accentDeep: '#5a8e00',
  accentDim: 'rgba(118, 185, 0, 0.15)',

  blue:  '#3b82f6',
  cyan:  '#26c6da',
  green: '#76b900',
  amber: '#ffc107',
  red:   '#ef5350',
  pink:  '#ab47bc',
};

export const colors = {
  bg: palette.bg,
  bgAlt: palette.bgCard,
  bgElevated: palette.bgRaised,
  bgRaised: palette.bgRaised,

  border: palette.border,
  borderSubtle: palette.borderSubtle,

  text: palette.text,
  textMuted: palette.sub,
  textDim: palette.muted,

  accent: palette.accent,
  accentAlt: palette.blue,

  success: palette.green,
  warning: palette.amber,
  error:   palette.red,
  info:    palette.blue,

  phaseRunning:   palette.blue,
  phaseCompleted: palette.green,
  phaseFailed:    palette.red,
  phasePending:   palette.muted,
  phaseUnknown:   palette.dim,
};

export function phaseColor(phase) {
  const p = (phase || '').toLowerCase();
  if (p === 'running')                            return colors.phaseRunning;
  if (p === 'completed' || p === 'succeeded')     return colors.phaseCompleted;
  if (p === 'failed' || p === 'error')            return colors.phaseFailed;
  if (p === 'pending' || p === 'initializing')    return colors.phasePending;
  return colors.phaseUnknown;
}

const MODEL_COLORS = [
  '#76b900', '#3b82f6', '#26c6da', '#9fe870',
  '#ffc107', '#ef5350', '#ab47bc', '#a0d8ff',
];

export function modelColor(model) {
  if (!model) return palette.muted;
  let hash = 0;
  for (let i = 0; i < model.length; i++) {
    hash = ((hash << 5) - hash + model.charCodeAt(i)) | 0;
  }
  return MODEL_COLORS[Math.abs(hash) % MODEL_COLORS.length];
}
```

- [ ] **Step 4: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t1.log
```

Expected: pre-commit clean, unit suite passes (the operator UI's static assets are static — no Python tests imported them).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui/index.html src/aiperf/operator/ui/style.css src/aiperf/operator/ui/lib/theme.js
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): adopt dashboard-v2 token foundation

Replace the WORKBENCH cream-on-near-black token ladder with v2's
flat dark-card system: --bg/--bg-card/--bg-raised/--bg-tile, single
--border hairline, IBM Plex Mono + JetBrains Mono, radii 4–8px, no
grid-paper substrate. Generic primitives (.card, .btn, .kbd, .chip,
.status-dot) and bench-error-flash are restyled. Legacy view-specific
rules stay in place; later tasks retire them as they replace markup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `chart-theme.js` + `chart-wrapper.js`

**Goal:** Re-tune Chart.js defaults so all three chart-using views (analysis, compare, run-latency) match v2's tooltip + grid + palette.

**Files:**
- Modify: `src/aiperf/operator/ui/lib/chart-theme.js`
- Modify: `src/aiperf/operator/ui/components/chart-wrapper.js` (class-only swaps; preserve testids and lifecycle)

- [ ] **Step 1: Replace `lib/chart-theme.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hand-tuned dark-theme defaults for Chart.js — operator UI palette
 * aligned with src/aiperf/api/static-v2.
 *
 * Chart.js is loaded as UMD via <script> in index.html and exposed as
 * `window.Chart`. This module installs grid / tooltip / legend / animation
 * defaults onto Chart.defaults the first time it's invoked, then becomes a
 * no-op (the `_initialized` flag is module-scoped, so re-imports across
 * pages do not re-apply).
 *
 * Per-chart options still win: the caller's `options` object is merged in
 * by Chart.js on top of these defaults.
 */

let _initialized = false;

/** Canonical dataset palette — green leads (NV brand), then blue, cyan,
 *  pale-green, amber, red, pink, sky. Saturated for legibility against the
 *  --bg substrate. */
export const PALETTE = [
  '#76b900',  // accent / NV green
  '#3b82f6',  // blue
  '#26c6da',  // cyan
  '#9fe870',  // pale green
  '#ffc107',  // amber
  '#ef5350',  // red
  '#ab47bc',  // pink
  '#a0d8ff',  // sky
];

const MONO_FAMILY  = "'IBM Plex Mono', ui-monospace, monospace";
const NUMER_FAMILY = "'JetBrains Mono', monospace";

/**
 * Apply shared dark-theme defaults (idempotent) and return the passed-in
 * options object unchanged. Call once before `new Chart(ctx, options)`.
 *
 * @param {object} options - Chart.js options object; returned unmodified.
 * @returns {object} The same options object the caller passed in.
 */
export function applyChartTheme(options = {}) {
  if (!_initialized && typeof window !== 'undefined' && window.Chart) {
    const C = window.Chart;
    C.defaults.font.family = MONO_FAMILY;
    C.defaults.font.size = 10;
    C.defaults.color       = '#a7a7a7';                 // --sub
    C.defaults.borderColor = '#313131';                 // --border
    C.defaults.scale.grid.color       = 'rgba(49, 49, 49, 0.5)';
    C.defaults.scale.grid.tickColor   = 'transparent';
    C.defaults.scale.grid.borderColor = 'transparent';
    C.defaults.plugins.tooltip.backgroundColor = 'rgba(22, 22, 22, 0.96)';  // --bg-card @ 0.96
    C.defaults.plugins.tooltip.titleColor = '#eeeeee';                       // --text
    C.defaults.plugins.tooltip.bodyColor  = '#a7a7a7';                       // --sub
    C.defaults.plugins.tooltip.padding = 12;
    C.defaults.plugins.tooltip.cornerRadius = 6;                             // was 0
    C.defaults.plugins.tooltip.displayColors = false;
    C.defaults.plugins.tooltip.boxPadding = 6;
    C.defaults.plugins.tooltip.borderColor = '#313131';                      // --border
    C.defaults.plugins.tooltip.borderWidth = 1;
    C.defaults.plugins.tooltip.titleFont = { family: MONO_FAMILY,  size: 11, weight: '700' };
    C.defaults.plugins.tooltip.bodyFont  = { family: NUMER_FAMILY, size: 11 };
    C.defaults.plugins.legend.labels.usePointStyle = true;
    C.defaults.plugins.legend.labels.boxWidth = 8;
    C.defaults.plugins.legend.labels.padding = 12;
    C.defaults.plugins.legend.labels.font  = { family: MONO_FAMILY, size: 10, weight: '600' };
    C.defaults.elements.line.tension = 0.3;
    _initialized = true;
  }
  return options;
}
```

- [ ] **Step 2: Re-skin `chart-wrapper.js`**

Open `src/aiperf/operator/ui/components/chart-wrapper.js`. The component already calls `applyChartTheme` on mount and renders a `<canvas>` inside a wrapper div. Only the **outer wrapper class** changes:
- Replace any wrapper `class="chart-frame"` / `chart-slab` / similar legacy class with `class="card chart-box"`.
- Add the rule below to `style.css` (append after the foundation block, before any legacy `chart-slab` rules):

```css
.chart-box {
  position: relative;
  height: 260px;
}
.chart-box canvas { display: block; width: 100% !important; height: 100% !important; }
```

Do **not** change the component's lifecycle, props, refs, or testid attributes.

- [ ] **Step 3: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t2.log
```

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/operator/ui/lib/chart-theme.js src/aiperf/operator/ui/components/chart-wrapper.js src/aiperf/operator/ui/style.css
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): retune Chart.js defaults for v2 palette

Tooltip on --bg-card with --border hairline, 6px radius. Grid lines
on rgba(49,49,49,0.5). Legend in IBM Plex Mono, body values in
JetBrains Mono. PALETTE leads with NV green, then blue/cyan/pale-green/
amber/red/pink/sky. Wrapper renders inside .card .chart-box.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: TopRail re-skin

**Goal:** Re-skin the persistent topbar to match v2's `.topbar`. Same data hooks (`useUtcClock`, `useNetStatus`, breadcrumb logic, button targets), same testids.

**Files:**
- Modify: `src/aiperf/operator/ui/components/top-rail.js`
- Modify: `src/aiperf/operator/ui/style.css` (append the topbar rules; remove the legacy `.rail`/`.rail-*` block at the end of this task)

**Test ID inventory (must all stay):** `top-nav`, `callsign`, `breadcrumb`, `rail-launch`, `rail-archive`, `rail-compare`, `nav-search`, `net-status`, `topbar-clock`.

- [ ] **Step 1: Append topbar CSS to `style.css`**

Append after the foundation block (and before the legacy rules):

```css
/* ───── Top rail ───── */
.topbar {
  grid-area: rail;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  height: var(--top-bar-height);
  background: var(--bg-card);
  border-bottom: 1px solid rgba(75, 75, 75, 0.25);
  position: sticky; top: 0;
  z-index: 100;
}
.topbar-left, .topbar-right { display: flex; align-items: center; gap: 14px; }

.topbar-logo {
  display: flex; align-items: center; gap: 8px;
  font-weight: 600;
  font-size: var(--font-sm);
  color: var(--accent);
  letter-spacing: 0.02em;
}
.topbar-logo-badge {
  font-size: 10px;
  font-weight: 700;
  color: var(--bg);
  background: var(--accent);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  letter-spacing: 0.06em;
}

.topbar-crumbs {
  font-size: var(--font-sm);
  color: var(--sub);
  display: flex; align-items: center; gap: 8px;
}
.topbar-crumb { color: var(--sub); }
.topbar-crumb--current { color: var(--text); font-weight: 500; }
.topbar-crumb-sep { color: var(--dim); }
.topbar-crumb a { color: var(--sub); }
.topbar-crumb a:hover { color: var(--text); text-decoration: none; }

.topbar-net {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: var(--font-xs);
  color: var(--sub);
  font-family: var(--font-mono);
}
.topbar-net--ok   .status-dot { background: var(--green); box-shadow: 0 0 6px rgba(118, 185, 0, 0.5); }
.topbar-net--warn .status-dot { background: var(--amber); }
.topbar-net--err  .status-dot { background: var(--red); }

.topbar-clock {
  font-family: var(--font-mono);
  font-size: var(--font-xs);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
```

- [ ] **Step 2: Rewrite the `TopRail` component body**

Open `src/aiperf/operator/ui/components/top-rail.js`. Replace the entire JSX returned by the component (everything from `return html\`` through the closing `\`;`) with the markup below. Keep imports, hooks, and helper functions (`useUtcClock`, `useNetStatus`, `breadcrumbFor`) untouched.

```js
return html`
  <header class="topbar" data-testid="top-nav">
    <div class="topbar-left">
      <a
        class="topbar-logo"
        href="#/"
        onclick=${(e) => { e.preventDefault(); navigate('/'); }}
        data-testid="callsign"
      >
        <span class="topbar-logo-badge">AI</span>
        <span>AIPerf Operator</span>
      </a>
      <nav class="topbar-crumbs" aria-label="Breadcrumb" data-testid="breadcrumb">
        ${crumbs.map((c, i) => html`
          ${i > 0 && html`<span class="topbar-crumb-sep">/</span>`}
          <span class=${'topbar-crumb' + (i === crumbs.length - 1 ? ' topbar-crumb--current' : '')}>
            ${c.path
              ? html`<a href=${'#' + c.path} onclick=${(e) => { e.preventDefault(); navigate(c.path); }}>${c.label}</a>`
              : c.label}
          </span>
        `)}
      </nav>
    </div>

    <div class="topbar-right">
      <button
        class="btn btn--ghost"
        onclick=${() => navigate('/archive')}
        data-testid="rail-archive"
        title="Archive"
      >Archive</button>
      <button
        class="btn btn--ghost"
        onclick=${() => navigate('/compare')}
        data-testid="rail-compare"
        title="Compare"
      >Compare</button>
      <button
        class="btn btn--ghost"
        onclick=${onSearchClick}
        data-testid="nav-search"
        title="Open command palette"
      >Search <span class="kbd">⌘ K</span></button>
      <button
        class="btn btn--primary"
        onclick=${() => navigate('/launch')}
        data-testid="rail-launch"
        title="Launch new run (⌘N)"
      >+ Launch</button>
      <div class=${'topbar-net topbar-net--' + net} title=${'Network ' + netLabel} data-testid="net-status">
        <span class="status-dot"></span>
        <span>${netLabel}</span>
      </div>
      <div class="topbar-clock" data-testid="topbar-clock" title="UTC">${utc} UTC</div>
    </div>
  </header>
`;
```

(`net`, `netLabel`, `crumbs`, `utc`, `onSearchClick` already exist in scope — keep the existing computation lines that produce them.)

- [ ] **Step 3: Delete the legacy `.rail`, `.rail-*` rules from `style.css`**

Search `style.css` for `.rail`, `.rail-logo-`, `.rail-crumbs`, `.rail-launch`, `.rail-archive`, `.rail-compare`, `.rail-net`, `.rail-clock`, and any related selectors. Delete those rule blocks. Be careful not to delete the new `.topbar` rules you just added.

- [ ] **Step 4: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t3.log
```

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui/components/top-rail.js src/aiperf/operator/ui/style.css
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): re-skin TopRail to v2 topbar

Logo + breadcrumb + Search/Archive/Compare/Launch buttons + net status
dot + UTC clock; same data hooks, same data-testids. Drops the
'AIPERF · WORKBENCH' callsign and rail-* class family.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: LogStrip + CommandPalette + global error flash

**Goal:** Re-skin the persistent log strip and the ⌘K palette modal. Update `bench-error-flash` copy from "FAULT" to "Error".

**Files:**
- Modify: `src/aiperf/operator/ui/components/log-strip.js`
- Modify: `src/aiperf/operator/ui/components/command-palette.js`
- Modify: `src/aiperf/operator/ui/app.js`
- Modify: `src/aiperf/operator/ui/style.css`

**Test ID inventory:** `log-strip`, `command-palette`, `command-palette-input`, `global-error`.

- [ ] **Step 1: Append LogStrip + Palette CSS to `style.css`**

```css
/* ───── Persistent log strip (bottom-pinned) ───── */
.log-strip {
  grid-area: log;
  background: var(--bg-card);
  border-top: 1px solid var(--border);
  display: flex; flex-direction: column;
  min-height: 0;
}
.log-strip-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 16px;
  gap: 12px;
  flex-wrap: wrap;
}
.log-strip-title {
  font-size: var(--font-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--dim);
}
.log-strip-filters { display: flex; gap: 4px; }
.log-strip-filter {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--sub);
  font-size: 10px;
  font-family: var(--font-mono);
  padding: 3px 8px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  display: inline-flex; align-items: center; gap: 5px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.log-strip-filter:hover { color: var(--text); border-color: var(--border-hover); }
.log-strip-filter--active {
  background: var(--bg-raised);
  border-color: var(--accent);
  color: var(--accent);
}
.log-strip-filter-count {
  font-size: 9px;
  background: var(--border);
  border-radius: 999px;
  padding: 0 5px;
  color: var(--sub);
}
.log-strip-filter--active .log-strip-filter-count { background: var(--accent-dim); color: var(--accent); }

.log-strip-body {
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.6;
  padding: 0 16px 12px;
  flex: 1;
  overflow-y: auto;
}
.log-strip-entry { color: var(--sub); }
.log-strip-entry .ts { color: var(--dim); margin-right: 8px; }
.log-strip-entry--warn  { color: var(--amber); }
.log-strip-entry--error { color: var(--red); }
.log-strip-entry--warn .ts, .log-strip-entry--error .ts { color: rgba(255, 255, 255, 0.35); }
.log-strip-cat {
  display: inline-block;
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 1px 5px;
  margin-right: 8px;
  border-radius: 2px;
  background: var(--border);
  color: var(--dim);
}
.log-strip-cat--phase   { background: var(--blue-dim);  color: var(--blue); }
.log-strip-cat--worker  { background: var(--pink-dim);  color: var(--pink); }
.log-strip-cat--records { background: var(--green-dim); color: var(--green); }

/* ───── Command palette ───── */
.cmdp-overlay {
  position: fixed; inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex; align-items: flex-start; justify-content: center;
  padding-top: 14vh;
  z-index: 1000;
}
.cmdp {
  width: min(560px, 92vw);
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  display: flex; flex-direction: column;
  overflow: hidden;
}
.cmdp-input {
  width: 100%;
  background: var(--bg-tile);
  border: none;
  border-bottom: 1px solid var(--border);
  padding: 12px 14px;
  font-family: var(--font-sans);
  font-size: var(--font-md);
  color: var(--text);
  outline: none;
}
.cmdp-input:focus { box-shadow: inset 0 -2px 0 0 var(--accent); }
.cmdp-list {
  max-height: 50vh;
  overflow-y: auto;
}
.cmdp-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px;
  cursor: pointer;
  border-bottom: 1px solid rgba(34, 34, 34, 0.6);
}
.cmdp-row:last-child { border-bottom: none; }
.cmdp-row:hover, .cmdp-row--active { background: var(--bg-raised); }
.cmdp-row-label { color: var(--text); font-size: var(--font-sm); }
.cmdp-row-kind  { color: var(--sub);  font-size: var(--font-xs); }
.cmdp-empty { padding: 16px; color: var(--muted); font-style: italic; text-align: center; }
```

- [ ] **Step 2: Re-skin `log-strip.js`**

Open `src/aiperf/operator/ui/components/log-strip.js`. Find the JSX block returned by the component. Replace the outer `<section class="log-strip" ...>` body with the markup below — preserving `data-testid="log-strip"` on the section and any inner testids (none currently). Keep all hooks, fetches, and entry-data shape unchanged.

```js
return html`
  <section class="log-strip" aria-label="Event log" data-testid="log-strip">
    <div class="log-strip-head">
      <div class="log-strip-title">Event Log</div>
      <div class="log-strip-filters">
        ${FILTERS.map(f => html`
          <button
            class=${'log-strip-filter' + (filter === f.key ? ' log-strip-filter--active' : '')}
            onclick=${() => setFilter(f.key)}
            type="button"
          >${f.label} <span class="log-strip-filter-count">${counts[f.key] ?? 0}</span></button>
        `)}
      </div>
    </div>
    <div class="log-strip-body">
      ${visible.map(e => html`
        <div class=${'log-strip-entry' + (e.level === 'error' ? ' log-strip-entry--error' : e.level === 'warn' ? ' log-strip-entry--warn' : '')}>
          <span class="ts">${e.ts}</span>
          ${e.cat && html`<span class=${'log-strip-cat log-strip-cat--' + e.cat}>${e.cat}</span>`}
          <span>${e.msg}</span>
        </div>
      `)}
    </div>
  </section>
`;
```

If the component doesn't currently have `FILTERS`, `filter`, `setFilter`, `counts`, `visible` in scope, add them: a local `const FILTERS = [{ key: 'all', label: 'All' }, { key: 'warn', label: 'Warn' }, { key: 'error', label: 'Error' }];` plus a `useState` filter, derive `visible` from existing entries by filter, derive `counts` by reducing entries. Do **not** alter the component's input shape — `entries` (or however it's named today) keeps its existing source.

If the existing component already has functioning filter logic with different names, keep its names and just swap classes: the only requirement is the rendered classes match the new CSS.

- [ ] **Step 3: Re-skin `command-palette.js`**

Open `src/aiperf/operator/ui/components/command-palette.js`. Replace the rendered overlay JSX with the markup below. Preserve `data-testid="command-palette"` and `data-testid="command-palette-input"`. Preserve all keyboard handlers (`onClose`, `onKeyDown`, `↑/↓`, `Enter`).

```js
return html`
  <div class="cmdp-overlay" onclick=${onOverlayClick}>
    <div
      class="cmdp"
      role="dialog"
      aria-modal="true"
      data-testid="command-palette"
      onclick=${(e) => e.stopPropagation()}
    >
      <input
        ref=${inputRef}
        class="cmdp-input"
        type="text"
        placeholder="Search runs, namespaces, or commands…"
        value=${query}
        oninput=${e => setQuery(e.target.value)}
        onkeydown=${onKeyDown}
        data-testid="command-palette-input"
      />
      <div class="cmdp-list">
        ${results.length === 0
          ? html`<div class="cmdp-empty">No matches</div>`
          : results.map((r, i) => html`
            <div
              key=${r.id}
              class=${'cmdp-row' + (i === active ? ' cmdp-row--active' : '')}
              onclick=${() => onSelect(r)}
              onmouseenter=${() => setActive(i)}
            >
              <span class="cmdp-row-label">${r.label}</span>
              <span class="cmdp-row-kind">${r.kind}</span>
            </div>
          `)}
      </div>
    </div>
  </div>
`;
```

- [ ] **Step 4: Update `bench-error-flash` copy in `app.js`**

Open `src/aiperf/operator/ui/app.js`. In the JSX that renders the global error flash (around line 137), replace `<strong>FAULT</strong>` with `<strong>Error</strong>`. The class name stays `bench-error-flash`.

- [ ] **Step 5: Delete the legacy log-strip + cmdp + bench-error-flash legacy CSS**

In `style.css`, search for `.log-strip-` rules from before this task (the old amber-glow ones), `.cmdp` rules, and any old `.bench-error-flash` rules. Replace them by removing the legacy blocks (the new rules added in Step 1 supersede them). Use a careful diff — avoid deleting the new rules.

- [ ] **Step 6: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t4.log
```

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/operator/ui/components/log-strip.js src/aiperf/operator/ui/components/command-palette.js src/aiperf/operator/ui/app.js src/aiperf/operator/ui/style.css
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): re-skin LogStrip + CommandPalette + error flash

LogStrip now renders v2-style filter pills + category chips on the
shared --bg-card surface. CommandPalette is a rounded modal with
hairline border and focus-line on input. Global error flash drops
'FAULT' for plain 'Error'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Home view re-skin

**Goal:** Re-skin `home.js` to use `.card` + new chip family. No behavior changes.

**Files:**
- Modify: `src/aiperf/operator/ui/views/home.js`
- Modify: `src/aiperf/operator/ui/style.css` (append home rules; remove legacy `hm-*` blocks)

**Test ID inventory:** `page-home`, `home-scanning`, `home-launch-cta`, `hm-summary`, `hm-row-<ns>-<name>`.

- [ ] **Step 1: Append Home CSS to `style.css`**

```css
/* ───── Home view ───── */
.v-home { display: grid; gap: 16px; max-width: 1400px; margin: 0 auto; }
.home-pitch {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
  text-align: center;
}
.home-pitch-title { font-size: var(--font-xl); color: var(--text); margin-bottom: 8px; }
.home-pitch-sub   { font-size: var(--font-sm); color: var(--sub); margin-bottom: 16px; }
.home-pitch-cta {
  font-size: var(--font-sm);
  padding: 8px 16px;
  background: var(--accent);
  color: var(--bg);
  border-radius: var(--radius-sm);
  font-weight: 700;
  border: 1px solid var(--accent);
  cursor: pointer;
}
.home-pitch-cta:hover { background: var(--accent-hot); border-color: var(--accent-hot); }

.hm-summary {
  display: flex; flex-wrap: wrap; gap: 16px;
  padding: 12px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  font-size: var(--font-sm);
}
.hm-summary-item { display: flex; align-items: baseline; gap: 6px; color: var(--sub); }
.hm-summary-item b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }

.hm-rows { display: grid; gap: 8px; }
.hm-row {
  display: grid;
  grid-template-columns: minmax(200px, 1.4fr) minmax(120px, 0.8fr) minmax(120px, 0.8fr) minmax(80px, 0.4fr);
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  cursor: pointer;
  transition: border-color var(--transition-fast), background var(--transition-fast);
}
.hm-row:hover { border-color: var(--border-hover); background: var(--bg-mid); }
.hm-row-name { color: var(--text); font-family: var(--font-sans); font-weight: 600; }
.hm-row-ns   { color: var(--sub); font-size: var(--font-xs); }
.hm-row-meta { color: var(--muted); font-size: var(--font-xs); font-family: var(--font-mono); }
.hm-row-status { justify-self: end; }
```

- [ ] **Step 2: Re-skin `home.js` JSX**

Open `src/aiperf/operator/ui/views/home.js`. Walk every rendered JSX block and apply class swaps:
- Wrapping div: keep `class="v-home"` (now styled by the new CSS).
- Pitch card: keep classes `home-pitch`, `home-pitch-title`, `home-pitch-sub`, `home-pitch-cta`.
- Summary: keep `hm-summary`. Replace inner spans with `<span class="hm-summary-item"><b>{n}</b> {label}</span>` if the existing markup differs. Keep `data-testid="hm-summary"` on the outer.
- Row markup: ensure each row has `class="hm-row"` and a status chip via `<span class="chip chip--good|--warn|--bad|--neutral">{statusLabel}</span>`.
- Replace any "FAULT" / "FAILED" all-caps labels with title-case "Failed". Replace status chip class:
  - `phase=running|initializing|pending` → `chip chip--info`
  - `phase=failed|error`                 → `chip chip--bad`
  - `phase=completed|succeeded`          → `chip chip--good`
  - else                                  → `chip chip--neutral`
- Preserve every `data-testid` (especially `hm-row-<ns>-<name>`, `hm-summary`, `home-launch-cta`, `home-scanning`, `page-home`).

- [ ] **Step 3: Delete legacy `hm-*` and `home-*` blocks from `style.css`**

Search for `.hm-`, `.home-` rule blocks added before this task (e.g., the sharp-corner amber-glow ones from the WORKBENCH design). Remove them. Keep the new ones from Step 1.

- [ ] **Step 4: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t5.log
```

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui/views/home.js src/aiperf/operator/ui/style.css
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): re-skin Home view to v2 cards + chips

Pitch card + summary bar + row list use --bg-card hairline-border
cards with rounded corners. Status chips drop the FAULT shoutmark
in favor of Failed/Healthy/Running/Idle via the .chip family.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Archive view re-skin

**Goal:** Re-skin `archive.js` to v2 cards + chips. Drop "FAULT" copy. Tab strip + namespace groups restyled.

**Files:**
- Modify: `src/aiperf/operator/ui/views/archive.js`
- Modify: `src/aiperf/operator/ui/style.css`

**Test ID inventory:** `page-archive`, `arch-summary`, `archive-search`, `archive-sort`, `arch-empty`, `arch-ns-<ns>`, `arch-row-<ns>-<name>`. Plus any tab-related testids in the file.

- [ ] **Step 1: Append Archive CSS to `style.css`**

```css
/* ───── Archive view ───── */
.v-archive { display: grid; gap: 16px; max-width: 1400px; margin: 0 auto; }

.arch-toolbar {
  display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  padding: 8px 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
}
.arch-toolbar input, .arch-toolbar select {
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  padding: 6px 10px;
  font-size: var(--font-sm);
}
.arch-toolbar input:focus, .arch-toolbar select:focus { outline: none; border-color: var(--accent); }

.arch-tabs {
  display: flex; gap: 0;
  border-bottom: 1px solid var(--border);
  padding: 0 4px;
}
.arch-tab {
  background: transparent;
  border: none;
  padding: 10px 14px;
  font-size: var(--font-sm);
  color: var(--sub);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}
.arch-tab:hover { color: var(--text); }
.arch-tab--active {
  color: var(--text);
  border-bottom-color: var(--accent);
  font-weight: 600;
}
.arch-tab-count {
  margin-left: 6px;
  font-size: var(--font-xs);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.arch-ns {
  display: flex; flex-direction: column; gap: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px;
}
.arch-ns-head {
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
  margin-bottom: 8px;
}
.arch-ns-name {
  font-family: var(--font-sans);
  font-size: var(--font-md);
  color: var(--text);
  font-weight: 600;
}
.arch-ns-counts { display: flex; gap: 6px; }

.arch-row {
  display: grid;
  grid-template-columns: minmax(200px, 1.4fr) minmax(120px, 0.8fr) minmax(140px, 1fr) minmax(80px, 0.4fr);
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  background: var(--bg-tile);
  border-radius: var(--radius);
  cursor: pointer;
  transition: background var(--transition-fast);
}
.arch-row:hover { background: var(--bg-raised); }
.arch-row-name { color: var(--text); font-weight: 500; }
.arch-row-ns   { color: var(--sub); font-size: var(--font-xs); }
.arch-row-meta { color: var(--muted); font-size: var(--font-xs); font-family: var(--font-mono); }
.arch-row-status { justify-self: end; }
```

- [ ] **Step 2: Re-skin `archive.js` JSX**

Open `src/aiperf/operator/ui/views/archive.js`. Apply:
- Outer wrapper keeps `class="v-archive"` and `data-testid="page-archive"`.
- Toolbar wrapped as `<div class="arch-toolbar">` containing the existing search input (`data-testid="archive-search"`) and sort select (`data-testid="archive-sort"`).
- Replace the existing tab pill markup with `<div class="arch-tabs">…<button class="arch-tab arch-tab--active|''" data-testid=…>{label}<span class="arch-tab-count">{n}</span></button>…</div>`. The existing `TABS` array contains `key, label, match` — keep `key === 'fault'` but render `label="Failed"` instead of `'FAULT'`. Where the current code says `label: 'FAULT'`, change the literal to `'Failed'`. (Sorting on `key` stays unchanged, so the URL hash logic and any external pinning still work.)
- Namespace section: `<section key=${ns} class="arch-ns" data-testid=${'arch-ns-' + ns}>` with `arch-ns-head` (name + counts).
- Counts: replace each `<span class="ns-count ns-count--fail">{n} FAULT</span>` family with `<span class="chip chip--bad">{n} Failed</span>` (and similar for the other variants: `chip--good` for passed, `chip--info` for live, `chip--neutral` for archived).
- Row markup: each row is `<div class="arch-row" data-testid=${'arch-row-' + j.namespace + '-' + j.name} onclick=…>` containing name / ns / meta / status chip.
- Empty-state row keeps `data-testid="arch-empty"` and uses class `empty` from foundation.

- [ ] **Step 3: Delete legacy archive CSS**

Search for `.v-archive`, `.arch-`, `.ns-count` blocks added before this task. Remove them — except the new ones from Step 1.

- [ ] **Step 4: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t6.log
```

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui/views/archive.js src/aiperf/operator/ui/style.css
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): re-skin Archive view + drop FAULT copy

Tab strip with --accent underline on active. Namespace cards on
--bg-card with chip-style counts (Failed/Healthy/Running/Archived).
Job rows on --bg-tile with hover-raise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Analysis + Compare views re-skin

**Goal:** Both views render Chart.js charts inside `.card` wrappers; restyle their tables and toolbars.

**Files:**
- Modify: `src/aiperf/operator/ui/views/analysis.js`
- Modify: `src/aiperf/operator/ui/views/compare.js`
- Modify: `src/aiperf/operator/ui/style.css`

**Test ID inventory:**
- analysis: `page-leaderboard` and any inner ones present.
- compare: `page-compare`, `compare-table`, `compare-col-a`, `compare-col-b`, `compare-both-missing`, `cmp-row-<metric>`.

- [ ] **Step 1: Append Analysis + Compare CSS to `style.css`**

```css
/* ───── Analysis view ───── */
.v-analysis { display: grid; gap: 16px; max-width: 1400px; margin: 0 auto; }
.v-analysis-toolbar {
  display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  padding: 8px 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
}
.v-analysis-err {
  background: var(--red-dim);
  border: 1px solid rgba(239, 83, 80, 0.45);
  color: var(--red);
  padding: 12px 16px;
  border-radius: var(--radius-lg);
  font-size: var(--font-sm);
}

/* ───── Compare view ───── */
.v-compare, .compare-view { display: grid; gap: 16px; max-width: 1400px; margin: 0 auto; }
.compare-warn {
  background: var(--amber-dim);
  border: 1px solid rgba(255, 193, 7, 0.45);
  color: var(--amber);
  padding: 12px 16px;
  border-radius: var(--radius-lg);
}
.compare-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  font-size: var(--font-sm);
}
.compare-table th {
  text-align: left;
  padding: 10px 14px;
  font-size: var(--font-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  background: var(--bg-tile);
}
.compare-table td {
  padding: 10px 14px;
  border-bottom: 1px solid rgba(34, 34, 34, 0.6);
  color: var(--sub);
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}
.compare-table tr:last-child td { border-bottom: none; }
.compare-table tr:hover td { background: var(--bg-raised); }
.compare-table td:first-child { color: var(--text); font-family: var(--font-sans); font-weight: 500; }
```

- [ ] **Step 2: Re-skin `analysis.js`**

Open `src/aiperf/operator/ui/views/analysis.js`. Wrap the chart canvas in `<div class="card chart-box">…</div>` (or use the existing `ChartWrapper` if it already does that). Keep `data-testid="page-leaderboard"` on the outer `<div class="v-analysis">`. Replace any toolbar `<div>`s with `<div class="v-analysis-toolbar">`. Replace error blocks with `<div class="v-analysis-err">`.

If the file references chart options inline, set `options.scales.x.ticks.font = { family: "'JetBrains Mono', monospace", size: 10 }` and `options.scales.y.ticks.font = { family: "'JetBrains Mono', monospace", size: 10 }` so numeric ticks render in the body-copy mono. (Skip if the file already uses default ticks.)

- [ ] **Step 3: Re-skin `compare.js`**

Open `src/aiperf/operator/ui/views/compare.js`. Wrap the outer in `<div class="v-compare compare-view" data-testid="page-compare">`. Replace `compare-warn` markup with the new card. Keep `<table class="compare-table" data-testid="compare-table">` and the existing column / row testids verbatim.

- [ ] **Step 4: Delete legacy analysis + compare CSS**

Remove old `.v-analysis-*`, `.compare-*` rule blocks from before this task. Keep the new ones from Step 1.

- [ ] **Step 5: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t7.log
```

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/ui/views/analysis.js src/aiperf/operator/ui/views/compare.js src/aiperf/operator/ui/style.css
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): re-skin Analysis + Compare views

Charts in .card .chart-box wrappers; comparison table on --bg-card
with mono numerics; toolbars in hairline cards. JetBrains Mono on
numeric ticks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Log + Launch views re-skin

**Goal:** Re-skin the durable log view and the launch (YAML editor) view.

**Files:**
- Modify: `src/aiperf/operator/ui/views/log.js`
- Modify: `src/aiperf/operator/ui/views/launch.js`
- Modify: `src/aiperf/operator/ui/style.css`

**Test ID inventory:**
- log: `page-history`.
- launch: `page-launch`, `launch-template-<id>`, `launch-prefill-notice`, `launch-target`, `launch-yaml`, `launch-success`, `launch-view-run`, `launch-parse-err`, `launch-err`, `launch-submit`.

- [ ] **Step 1: Append Log + Launch CSS to `style.css`**

```css
/* ───── Durable log view ───── */
.v-log {
  display: grid; gap: 16px;
  max-width: 1400px; margin: 0 auto;
}
.v-log-pane {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 12px 16px;
  font-family: var(--font-mono);
  font-size: var(--font-xs);
  line-height: 1.6;
  max-height: 70vh;
  overflow-y: auto;
}

/* ───── Launch view ───── */
.v-launch { display: grid; gap: 16px; max-width: 1400px; margin: 0 auto; }
.launch-templates {
  display: flex; gap: 8px; flex-wrap: wrap;
  padding: 12px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
}
.launch-template {
  padding: 8px 12px;
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--sub);
  cursor: pointer;
  font-size: var(--font-xs);
  font-family: var(--font-mono);
}
.launch-template:hover { color: var(--text); border-color: var(--border-hover); }
.launch-template--active { background: var(--accent-dim); color: var(--accent); border-color: var(--accent); }

.launch-prefill-notice {
  background: var(--blue-dim);
  border: 1px solid rgba(59, 130, 246, 0.45);
  color: var(--blue);
  padding: 10px 14px;
  border-radius: var(--radius-lg);
  font-size: var(--font-xs);
}

.launch-target {
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  padding: 6px 10px;
  font-size: var(--font-sm);
}

.launch-yaml {
  width: 100%;
  min-height: 360px;
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: var(--font-sm);
  line-height: 1.6;
  padding: 12px;
  resize: vertical;
}
.launch-yaml:focus { outline: none; border-color: var(--accent); }

.launch-actions { display: flex; gap: 8px; justify-content: flex-end; }

.launch-success {
  background: var(--green-dim);
  border: 1px solid rgba(118, 185, 0, 0.45);
  color: var(--green);
  padding: 10px 14px;
  border-radius: var(--radius-lg);
  font-size: var(--font-sm);
}
```

- [ ] **Step 2: Re-skin `log.js`**

Open `src/aiperf/operator/ui/views/log.js`. Outer remains `<div class="v-log" data-testid="page-history">`. Wrap the body in `<div class="v-log-pane">`. Reuse the LogStrip filter pill markup from Task 4 if there's an existing filter UI; otherwise leave entries as-is, just restyled by the new CSS. Drop any "FAULT/CALLSIGN" copy.

- [ ] **Step 3: Re-skin `launch.js`**

Open `src/aiperf/operator/ui/views/launch.js`. Outer is `<div class="v-launch" data-testid="page-launch">`. Apply class swaps:
- Template list: `<div class="launch-templates">…<button class="launch-template launch-template--active|''" data-testid=${'launch-template-' + t.id}>…</button>…</div>`.
- Prefill notice: keep `data-testid="launch-prefill-notice"`, class `launch-prefill-notice`.
- Target input: `class="launch-target"`, keep `data-testid="launch-target"`.
- YAML textarea: `class="launch-yaml"`, keep `data-testid="launch-yaml"`.
- Submit/cancel actions: wrap in `<div class="launch-actions">` with `<button class="btn btn--primary" data-testid="launch-submit">Launch</button>` and a ghost cancel `<button class="btn btn--ghost">Cancel</button>` if the existing UI has one.
- Success block: `<div class="launch-success" data-testid="launch-success">…<a class="btn btn--ghost" data-testid="launch-view-run">View run</a></div>`.
- Error blocks: parse-err uses `<div class="v-analysis-err" data-testid="launch-parse-err">…</div>` (re-uses Analysis's red callout); submit error uses `<div class="bench-error-flash" data-testid="launch-err">…</div>`.

- [ ] **Step 4: Delete legacy log + launch CSS**

Remove old `.v-log*`, `.v-launch*`, `.launch-*` rule blocks from before this task. Keep the new ones.

- [ ] **Step 5: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t8.log
```

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/ui/views/log.js src/aiperf/operator/ui/views/launch.js src/aiperf/operator/ui/style.css
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): re-skin Log + Launch views

Durable log on --bg-card; launch view templates as pill chips, YAML
on --bg-tile with --accent focus border, primary/ghost button pair.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Run view part 1 — hero + KPI tiles + sparklines + latency timeline

**Goal:** Replace the run view's HEADER + IDENTITY + METER BAY + SPARKLINES + LATENCY-TIMELINE blocks with v2's HeroStrip + KPI-tile + sparkline-tile patterns. State / hooks / fetches stay; only the markup these blocks render changes.

**Files:**
- Modify: `src/aiperf/operator/ui/views/run.js`
- Modify: `src/aiperf/operator/ui/style.css`

**Test ID inventory (must all stay):** `page-job-detail`, `run-identity`, `run-identity-sibling` (×2), `run-sparks`, `run-cancel`, `run-relaunch`, `run-history`, `run-history-select`, `run-compare`, `run-compare-select`, `run-latency-chart`.

- [ ] **Step 1: Append Hero + KPI tile + Spark + Latency-chart CSS to `style.css`**

```css
/* ───── Run view: shell ───── */
.v-run { display: grid; gap: 16px; max-width: 1400px; margin: 0 auto; padding-bottom: 24px; }

/* ───── Run view: hero ───── */
.run-hero {
  display: grid;
  grid-template-columns: minmax(260px, 1.2fr) minmax(180px, 0.8fr) minmax(300px, 2fr);
  gap: 16px;
  padding: 16px 18px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  align-items: center;
}
.run-hero--ok    { border-color: rgba(118, 185, 0, 0.45); box-shadow: 0 0 0 1px rgba(118, 185, 0, 0.12) inset; }
.run-hero--warn  { border-color: rgba(255, 193, 7, 0.45); box-shadow: 0 0 0 1px rgba(255, 193, 7, 0.12) inset; }
.run-hero--error { border-color: rgba(239, 83, 80, 0.55); box-shadow: 0 0 0 1px rgba(239, 83, 80, 0.15) inset; }
.run-hero--idle  { border-color: rgba(117, 117, 117, 0.45); }

.run-hero-health { display: flex; align-items: center; gap: 14px; }
.run-hero-dot {
  width: 18px; height: 18px; border-radius: 50%;
  flex-shrink: 0;
  box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.02);
}
.run-hero-dot--ok    { background: var(--green); box-shadow: 0 0 16px rgba(118, 185, 0, 0.5); animation: pulse 2s infinite; }
.run-hero-dot--warn  { background: var(--amber); box-shadow: 0 0 16px rgba(255, 193, 7, 0.5); }
.run-hero-dot--error { background: var(--red);   box-shadow: 0 0 16px rgba(239, 83, 80, 0.5); animation: pulse 1.2s infinite; }
.run-hero-dot--idle  { background: var(--muted); }
.run-hero-label   { font-size: var(--font-lg); font-weight: 700; color: var(--white); line-height: 1.1; }
.run-hero-reasons { font-size: var(--font-xs); color: var(--sub); margin-top: 4px; font-family: var(--font-mono); }

.run-hero-clock {
  display: flex; flex-direction: column; gap: 6px;
  padding-left: var(--space-4); border-left: 1px solid var(--border);
}
.run-hero-clock-line { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
.run-hero-clock-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); font-weight: 600; }
.run-hero-clock-val   { font-family: var(--font-mono); font-size: var(--font-md); font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
.run-hero-clock-val--dim { color: var(--muted); font-weight: 400; }

.run-hero-phase {
  display: flex; flex-direction: column; gap: 6px;
  padding-left: var(--space-4); border-left: 1px solid var(--border);
}
.run-hero-phase-head { display: flex; justify-content: space-between; align-items: baseline; }
.run-hero-phase-name { font-family: var(--font-sans); font-size: var(--font-md); font-weight: 600; color: var(--white); }
.run-hero-phase-name--idle { color: var(--muted); font-weight: 400; }
.run-hero-phase-pct  { font-size: var(--font-md); font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
.run-hero-phase-track { height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
.run-hero-phase-fill  { height: 100%; background: var(--blue); border-radius: 4px; transition: width 300ms ease; }
.run-hero-phase-fill--done { background: var(--green); }
.run-hero--error .run-hero-phase-fill { background: var(--red); }
.run-hero--warn  .run-hero-phase-fill { background: var(--amber); }
.run-hero-phase-sub { font-size: var(--font-xs); color: var(--sub); font-variant-numeric: tabular-nums; }

.run-hero-actions { display: flex; gap: 8px; align-items: center; }

/* ───── Run view: identity strip ───── */
.run-identity {
  display: flex; flex-wrap: wrap; gap: 16px;
  padding: 10px 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  font-size: var(--font-xs);
}
.run-identity-item { display: flex; align-items: baseline; gap: 6px; color: var(--sub); }
.run-identity-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); font-weight: 600; }
.run-identity-value { color: var(--text); font-variant-numeric: tabular-nums; font-family: var(--font-mono); }
.run-identity-sep   { width: 1px; align-self: stretch; background: var(--border); }

/* ───── Run view: KPI tiles (replaces meter bay) ───── */
.run-meters, .run-kpis {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}
.run-kpi {
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
  display: flex; flex-direction: column; gap: 6px;
  min-height: 104px;
  transition: border-color var(--transition-fast);
}
.run-kpi--good { border-color: rgba(118, 185, 0, 0.35); }
.run-kpi--warn { border-color: rgba(255, 193, 7, 0.35); }
.run-kpi--bad  { border-color: rgba(239, 83, 80, 0.45); }
.run-kpi-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
.run-kpi-label-block { display: flex; flex-direction: column; gap: 2px; }
.run-kpi-label-block > span:first-child {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--dim); font-weight: 600;
}
.run-kpi-primary-stat { font-size: 9px; color: var(--muted); font-family: var(--font-mono); letter-spacing: 0.04em; }
.run-kpi-big { display: flex; align-items: baseline; gap: 4px; }
.run-kpi-big-val { font-size: 26px; font-weight: 700; color: var(--white); font-variant-numeric: tabular-nums; line-height: 1; font-family: var(--font-mono); }
.run-kpi-big-unit { font-size: var(--font-xs); color: var(--muted); font-weight: 500; }
.run-kpi-sub {
  font-size: var(--font-xs); color: var(--dim);
  margin-top: auto;
  font-variant-numeric: tabular-nums;
  display: flex; align-items: baseline; gap: 6px;
}
.run-kpi-sub b { color: var(--sub); font-weight: 600; }
.run-kpi-spark { display: block; width: 100%; max-width: 160px; }

/* ───── Run view: sparkline tiles (live only) ───── */
.run-sparks {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.run-spark-tile {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
  display: flex; flex-direction: column; gap: 4px;
}
.run-spark-tile-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); font-weight: 600; }
.run-spark-tile-value { font-family: var(--font-mono); font-size: var(--font-lg); font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
.run-spark-tile-svg   { width: 100%; height: 32px; }

/* ───── Run view: latency timeline (completed runs) ───── */
.run-latency-chart {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: var(--space-4);
}
```

- [ ] **Step 2: Rewrite the run view's top sections**

Open `src/aiperf/operator/ui/views/run.js`. Locate the `Run` component's `return html\`…\`` and replace the blocks tagged `<!-- 1. HEADER -->`, `<!-- 1b. IDENTITY -->`, `<!-- 3. METER BAY -->`, `<!-- 3b. LIVE SPARKLINES -->`, and `<!-- 3c. REQUEST-LATENCY TIMELINE -->`. Keep `<!-- 1c. FAULT CALLOUT -->`, `<!-- 2. CONDITIONS -->`, `<!-- 3. PHASES SWIMLANE -->`, `<!-- 5. PODS -->`, `<!-- 5b. EVENTS -->`, `<!-- 5c. LOGS -->`, `<!-- 6. GPU TELEMETRY -->`, `<!-- 7. RESULTS -->`, `<!-- 8. CONFIG -->` for now (Task 10 covers them).

The new top of the return becomes:

```js
const heroVerdict =
  bucket === 'fault' ? 'error' :
  bucket === 'live'  ? 'ok'    :
  bucket === 'passed'? 'ok'    :
                       'idle';
const heroDotClass    = `run-hero-dot run-hero-dot--${heroVerdict}`;
const heroBorderClass = `run-hero run-hero--${heroVerdict}`;
const heroLabel =
  bucket === 'fault'  ? 'Failed'  :
  bucket === 'live'   ? 'Healthy' :
  bucket === 'passed' ? 'Completed' :
                        'Idle';
const heroReasons = [
  job?.model && `model: ${job.model}`,
  pods.length > 0 && `${pods.filter(p => (p.phase || '').toLowerCase() === 'running').length}/${pods.length} pods running`,
].filter(Boolean).join(' · ');
const phasePctNum = phasePct(active);
const phaseFillClass =
  active?.complete ? 'run-hero-phase-fill run-hero-phase-fill--done' :
                     'run-hero-phase-fill';
const phaseTotal = active?.total_expected_requests ?? active?.expected_requests ?? active?.requests_total ?? null;
const phaseDone  = active?.final_requests_completed ?? active?.requestsCompleted ?? active?.requests_completed ?? active?.completed ?? 0;

return html`
  <div class=${'v-run v-run--' + bucket} data-testid="page-job-detail">

    <!-- 1. HERO -->
    <section class=${heroBorderClass}>
      <div class="run-hero-health">
        <div class=${heroDotClass}></div>
        <div>
          <div class="run-hero-label">${heroLabel}</div>
          ${heroReasons && html`<div class="run-hero-reasons">${heroReasons}</div>`}
        </div>
      </div>
      <div class="run-hero-clock">
        <div class="run-hero-clock-line">
          <span class="run-hero-clock-label">Elapsed</span>
          <span class=${'run-hero-clock-val' + (elapsed != null ? '' : ' run-hero-clock-val--dim')}>
            ${elapsed != null ? fmtDuration(elapsed) : '—'}
          </span>
        </div>
        <div class="run-hero-clock-line">
          <span class="run-hero-clock-label">Phase ETA</span>
          <span class=${'run-hero-clock-val' + (eta != null ? '' : ' run-hero-clock-val--dim')}>
            ${eta != null ? fmtDuration(eta) : '—'}
          </span>
        </div>
      </div>
      <div class="run-hero-phase">
        <div class="run-hero-phase-head">
          <span class=${'run-hero-phase-name' + (active ? '' : ' run-hero-phase-name--idle')}>
            ${active?.name ?? 'idle'}
          </span>
          <span class="run-hero-phase-pct">${fmtPercent(phasePctNum, 0)}</span>
        </div>
        <div class="run-hero-phase-track">
          <div class=${phaseFillClass} style=${'width: ' + phasePctNum + '%'}></div>
        </div>
        <div class="run-hero-phase-sub">
          ${fmtInt(phaseDone)}${phaseTotal ? ' / ' + fmtInt(phaseTotal) : ''} requests
        </div>
      </div>
    </section>

    <!-- 1a. ACTIONS (cancel / relaunch / history / compare) -->
    <div class="run-hero-actions">
      <${CancelButton}        ns=${ns} name=${name} bucket=${bucket} />
      <${RelaunchButton}      ns=${ns} name=${name} config=${config} />
      <${RunHistoryDropdown}  ns=${ns} name=${name} selectedEpoch=${epoch} />
      <${CompareWithDropdown} ns=${ns} name=${name} selectedEpoch=${epoch} />
    </div>

    <!-- 1b. IDENTITY -->
    <${IdentityStrip} job=${job} config=${config} summary=${summary} />

    <!-- 1c. FAULT CALLOUT -->
    <${FaultCallout} bucket=${bucket} conditions=${conditions} pods=${pods} />

    <!-- 2. CONDITIONS -->
    <${ConditionsStrip} conditions=${conditions} />

    <!-- 3. KPIs (replaces METER BAY) -->
    <section class="run-kpis">
      <${RunKpi} label="Throughput"  primary=""           value=${rps  != null ? fmtNumber(rps, 1) : '—'} unit="req/s" tone=${rps  != null ? 'good' : ''} />
      <${RunKpi} label="TTFT p99"    primary="first token" value=${ttft != null ? fmtInt(ttft)     : '—'} unit="ms"    tone="" />
      <${RunKpi} label="Latency p99" primary="end-to-end"  value=${p99  != null ? fmtInt(p99)      : '—'} unit="ms"    tone=${p99 != null && p99 > 500 ? 'warn' : ''} />
      <${RunKpi} label="Token/s"     primary="output"      value=${tokps != null ? fmtInt(tokps)   : '—'} unit="tok/s" tone=${tokps != null ? 'good' : ''} />
      <${ReliabilityMeter} summary=${summary} slosDeclared=${slosDeclared} />
    </section>

    <!-- 3b. LIVE SPARKLINES -->
    ${bucket === 'live' && html`
      <section class="run-sparks" data-testid="run-sparks" aria-label="Live metric sparklines">
        ${SPARK_SPECS.map(spec => html`
          <${SparkTile} key=${spec.key} spec=${spec} samples=${samples} />
        `)}
      </section>
    `}

    <!-- 3c. REQUEST-LATENCY TIMELINE (completed runs) -->
    ${bucket !== 'live' && html`
      <${LatencyTimelineChart} ns=${ns} name=${name} epoch=${epoch} />
    `}
```

(Continue the existing return with the unchanged `<!-- 3. PHASES SWIMLANE -->` … `<!-- 8. CONFIG -->` blocks for now. They get rewritten in Task 10.)

- [ ] **Step 3: Add the `RunKpi` component to `run.js`**

Replace the existing `RunMeter` definition (around line 1623) with this updated version (and rename references):

```js
function RunKpi({ label, primary, value, unit, tone, sparkData }) {
  const cls = 'run-kpi' + (tone ? ' run-kpi--' + tone : '');
  return html`
    <div class=${cls}>
      <div class="run-kpi-head">
        <div class="run-kpi-label-block">
          <span>${label}</span>
          ${primary && html`<span class="run-kpi-primary-stat">${primary}</span>`}
        </div>
      </div>
      <div class="run-kpi-big">
        <span class="run-kpi-big-val">${value}</span>
        ${unit && html`<span class="run-kpi-big-unit">${unit}</span>`}
      </div>
    </div>
  `;
}
```

The original `RunMeter` is no longer referenced after Step 2; delete it.

- [ ] **Step 4: Update `IdentityStrip` and `SparkTile` markup**

Find the `IdentityStrip` component (around line 700). Replace its rendered markup with the v2-styled version, preserving `data-testid="run-identity"` and the two `data-testid="run-identity-sibling"` testids:

```js
function IdentityStrip({ job, config, summary }) {
  const items = [
    job?.namespace && { label: 'Namespace', value: job.namespace },
    job?.model     && { label: 'Model',     value: job.model },
    job?.startTime && { label: 'Started',   value: new Date(job.startTime).toISOString().slice(0, 19) + 'Z' },
    summary?.completion_tokens_total && { label: 'Tokens',  value: fmtInt(summary.completion_tokens_total) },
    config?.source && { label: 'Config',    value: config.source },
  ].filter(Boolean);
  if (items.length === 0) return null;
  return html`
    <section class="run-identity" data-testid="run-identity" aria-label="Run identity">
      ${items.map((it, i) => html`
        ${i > 0 && html`<span class="run-identity-sep"></span>`}
        <div class="run-identity-item" data-testid=${i < 2 ? 'run-identity-sibling' : null}>
          <span class="run-identity-label">${it.label}</span>
          <span class="run-identity-value">${it.value}</span>
        </div>
      `)}
    </section>
  `;
}
```

(If the existing component has more sophisticated logic — siblings linking to other runs — preserve that logic but render through the `run-identity-item` / `run-identity-label` / `run-identity-value` class trio.)

Find `SparkTile` (around line 1543's neighborhood). Re-skin its return:

```js
function SparkTile({ spec, samples }) {
  const series  = samples?.[spec.key] ?? [];
  const current = series.length ? series[series.length - 1].value : null;
  const path    = sparkPath(series);  // existing helper, leave alone
  return html`
    <div class="run-spark-tile">
      <span class="run-spark-tile-label">${spec.label}</span>
      <span class="run-spark-tile-value">${current != null ? fmtNumber(current, spec.precision ?? 0) : '—'}</span>
      <svg class="run-spark-tile-svg" viewBox="0 0 160 32" preserveAspectRatio="none">
        <polyline fill="none" stroke="${spec.color ?? '#76b900'}" stroke-width="1.4" points=${path}/>
      </svg>
    </div>
  `;
}
```

If `sparkPath` doesn't exist with that name, keep whatever helper currently produces the polyline points string.

- [ ] **Step 5: Update `LatencyTimelineChart` outer wrapper**

Find `LatencyTimelineChart` (around line 1208). Its outer `<section class="run-latency-chart" data-testid="run-latency-chart">` already exists — keep the testid. Replace the chart-canvas wrapper inside with `<div class="card chart-box">…</div>` so it picks up the new card surface. Keep all chart options, hooks, and refs as-is.

- [ ] **Step 6: Delete legacy run-header / run-meter / run-clock / run-identity legacy CSS**

Search `style.css` for old `.run-header`, `.run-clock`, `.run-meter`, `.run-meter-`, `.run-identity-` (pre-task), and `.run-spark-` blocks. Remove them. Keep the new `.run-hero`, `.run-kpis`, `.run-kpi`, `.run-sparks`, `.run-spark-tile`, `.run-identity` rules from Step 1.

- [ ] **Step 7: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t9.log
```

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/operator/ui/views/run.js src/aiperf/operator/ui/style.css
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): re-skin Run view header, identity, KPIs, sparks

Replace the WORKBENCH run-header + meter-bay + identity strip + live
sparkline tiles with v2's HeroStrip / KPI tile / spark tile patterns.
Hero border tints follow run verdict (ok/warn/error/idle). Latency
timeline chart sits in .card .chart-box. All data-testids preserved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Run view part 2 — phases, pods, events, logs, GPU, results, fault, conditions, config + final CSS sweep

**Goal:** Re-skin the remaining run-view sections in v2 vocabulary (phase cards, pods table, events list, logs, GPU telemetry, results, fault callout, conditions, collapsible config), then walk `style.css` and delete every legacy block whose markup has been retired across Tasks 3–10.

**Files:**
- Modify: `src/aiperf/operator/ui/views/run.js`
- Modify: `src/aiperf/operator/ui/style.css`

**Test ID inventory (must all stay):** `run-results`, `run-results-bundle`, `run-results-row-<file>`, `run-history`, `run-history-select`, `run-compare`, `run-compare-select`, `run-cancel`, `run-relaunch`, `run-logs`, `run-logs-pod`, `run-logs-follow`, `run-logs-tail`, `run-logs-body`, `run-logs-jump`, `run-conditions`, `run-pods`, `run-gpu`, `run-fault-callout`, `run-events`.

- [ ] **Step 1: Append remaining Run-view CSS to `style.css`**

```css
/* ───── Run view: conditions ───── */
.run-conditions {
  display: flex; flex-wrap: wrap; gap: 6px;
  padding: 10px 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
}
.run-condition-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px;
  font-size: var(--font-xs);
  font-family: var(--font-mono);
  border-radius: 999px;
}
.run-condition-chip--true   { background: var(--green-dim); color: var(--green); }
.run-condition-chip--false  { background: var(--red-dim);   color: var(--red); }
.run-condition-chip--unknown{ background: rgba(75, 75, 75, 0.18); color: var(--muted); }

/* ───── Run view: fault callout ───── */
.run-fault-callout {
  background: rgba(239, 83, 80, 0.08);
  border: 1px solid rgba(239, 83, 80, 0.45);
  border-radius: var(--radius-lg);
  padding: 16px;
  display: flex; flex-direction: column; gap: 10px;
}
.run-fault-head { font-size: var(--font-md); font-weight: 700; color: var(--white); }
.run-fault-head-tag {
  color: var(--red); margin-right: 8px;
  text-transform: uppercase; font-size: var(--font-xs); letter-spacing: 0.08em;
}
.run-fault-reasons { font-size: var(--font-sm); color: var(--sub); line-height: 1.6; }
.run-fault-reasons ul { margin: 6px 0 0 18px; }
.run-fault-actions { display: flex; gap: 8px; }

/* ───── Run view: phase cards ───── */
.run-phases {
  display: flex; flex-direction: column; gap: 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px;
}
.run-phases-head { display: flex; align-items: center; justify-content: space-between; }
.run-phases-title { font-size: var(--font-xs); text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); font-weight: 600; }
.run-phases-meta  { font-size: var(--font-xs); color: var(--muted); font-family: var(--font-mono); }
.run-phases-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
}
.run-phase {
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
  display: flex; flex-direction: column; gap: 8px;
}
.run-phase-head { display: flex; align-items: center; justify-content: space-between; }
.run-phase-name { font-family: var(--font-sans); font-weight: 600; color: var(--text); font-size: var(--font-sm); }
.run-phase-badge {
  font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: var(--radius-sm);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.run-phase-badge--running  { background: var(--blue-dim);  color: var(--blue); }
.run-phase-badge--complete { background: var(--green-dim); color: var(--green); }
.run-phase-badge--pending  { background: rgba(117, 117, 117, 0.15); color: var(--muted); }
.run-phase-badge--grace    { background: var(--amber-dim); color: var(--amber); }
.run-phase-track { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.run-phase-fill  { height: 100%; background: var(--blue); border-radius: 3px; transition: width 300ms ease; }
.run-phase--complete .run-phase-fill { background: var(--green); }
.run-phase--grace    .run-phase-fill { background: var(--amber); }
.run-phase-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 8px; margin-top: 4px; }
.run-phase-stat { display: flex; flex-direction: column; gap: 2px; }
.run-phase-stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--dim); }
.run-phase-stat-val   { font-family: var(--font-mono); color: var(--text); font-size: var(--font-sm); font-weight: 600; font-variant-numeric: tabular-nums; }

/* ───── Run view: pods ───── */
.run-pods {
  display: grid; gap: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px;
}
.run-pods-title { font-size: var(--font-xs); text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); font-weight: 600; }
.run-pods-list { display: flex; flex-wrap: wrap; gap: 8px; }
.run-pod {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 10px;
  font-size: var(--font-xs);
  font-family: var(--font-mono);
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--sub);
}
.run-pod-dot { width: 8px; height: 8px; border-radius: 50%; }
.run-pod-dot--running   { background: var(--green); box-shadow: 0 0 6px rgba(118, 185, 0, 0.5); }
.run-pod-dot--pending   { background: var(--amber); }
.run-pod-dot--succeeded { background: var(--green); }
.run-pod-dot--failed    { background: var(--red); }
.run-pod-dot--unknown   { background: var(--muted); }

/* ───── Run view: events + logs ───── */
.run-events, .run-logs {
  display: grid; gap: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px;
}
.run-events--err { border-color: rgba(239, 83, 80, 0.45); }
.run-events-title, .run-logs-title { font-size: var(--font-xs); text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); font-weight: 600; }
.run-events-list {
  font-family: var(--font-mono); font-size: var(--font-xs); line-height: 1.6;
  max-height: 240px; overflow-y: auto;
}
.run-event { color: var(--sub); }
.run-event--warn  { color: var(--amber); }
.run-event--error { color: var(--red); }
.run-event-ts { color: var(--dim); margin-right: 8px; }

.run-logs-controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.run-logs-controls select, .run-logs-controls input {
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  padding: 4px 8px;
  font-size: var(--font-xs);
  font-family: var(--font-mono);
}
.run-logs-body {
  font-family: var(--font-mono); font-size: var(--font-xs); line-height: 1.6;
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
  max-height: 360px;
  overflow-y: auto;
  white-space: pre-wrap;
  color: var(--sub);
}
.run-logs-error { color: var(--red); margin-top: 6px; }
.run-logs-jump  {
  align-self: flex-end;
  font-size: var(--font-xs);
}

/* ───── Run view: GPU telemetry ───── */
.run-gpu {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px;
  display: flex; flex-direction: column; gap: 12px;
}
.run-gpu-title { font-size: var(--font-xs); text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); font-weight: 600; }
.run-gpu-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 12px;
}
.run-gpu-card {
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
  display: flex; flex-direction: column; gap: 12px;
}
.run-gpu-header { font-family: var(--font-mono); font-size: var(--font-xs); color: var(--sub); }
.run-gpu-primary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: 8px;
}
.run-gpu-tile {
  background: var(--bg);
  border: 1px solid rgba(49, 49, 49, 0.4);
  border-radius: var(--radius-sm);
  padding: 8px 10px;
  display: flex; flex-direction: column; gap: 2px;
}
.run-gpu-tile-label { font-size: 9px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); font-weight: 600; }
.run-gpu-tile-val   { font-size: 16px; font-weight: 700; color: var(--white); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.run-gpu-tile-unit  { font-size: 10px; color: var(--muted); font-weight: 500; }
.run-gpu-extra { width: 100%; border-collapse: separate; border-spacing: 0; font-size: var(--font-xs); }
.run-gpu-extra td { padding: 4px 6px; color: var(--sub); font-variant-numeric: tabular-nums; border-top: 1px solid rgba(34, 34, 34, 0.5); }
.run-gpu-extra td:first-child { color: var(--muted); font-family: var(--font-mono); font-size: 10px; }
.run-gpu-extra tr:first-child td { border-top: none; }

/* ───── Run view: results ───── */
.run-results {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px;
  display: flex; flex-direction: column; gap: 10px;
}
.run-results--err   { border-color: rgba(239, 83, 80, 0.45); }
.run-results--empty { color: var(--muted); }
.run-results-head { display: flex; align-items: center; justify-content: space-between; }
.run-results-title { font-size: var(--font-xs); text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); font-weight: 600; }
.run-results-meta  { font-size: var(--font-xs); color: var(--muted); font-family: var(--font-mono); }
.run-results-list { display: flex; flex-direction: column; gap: 4px; }
.run-results-row {
  display: grid;
  grid-template-columns: 64px 1fr auto auto;
  align-items: center;
  gap: 12px;
  padding: 6px 10px;
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}
.run-results-row:hover { background: var(--bg-raised); }
.run-results-kind {
  font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
  padding: 2px 6px; border-radius: var(--radius-sm);
  background: var(--bg-raised); color: var(--sub);
}
.run-results-kind--json    { color: var(--cyan); }
.run-results-kind--csv     { color: var(--green); }
.run-results-kind--parquet { color: var(--blue); }
.run-results-kind--yaml    { color: var(--amber); }
.run-results-kind--log     { color: var(--pink); }
.run-results-kind--html    { color: var(--text); }
.run-results-kind--image   { color: var(--text); }
.run-results-kind--bin     { color: var(--muted); }

/* ───── Run view: history / compare dropdowns ───── */
.run-history, .run-compare {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: var(--font-xs);
  color: var(--sub);
}
.run-history select, .run-compare select {
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  padding: 4px 8px;
  font-size: var(--font-xs);
  font-family: var(--font-mono);
}

/* ───── Run view: collapsible config ───── */
.run-config {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 12px 16px;
}
.run-config > summary {
  cursor: pointer;
  font-size: var(--font-xs);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--dim);
  font-weight: 600;
  list-style: none;
}
.run-config > summary::-webkit-details-marker { display: none; }
.run-config-hint { color: var(--muted); margin-left: 6px; text-transform: none; letter-spacing: normal; font-weight: 400; }
.run-config-body {
  margin-top: 12px;
  font-family: var(--font-mono);
  font-size: var(--font-xs);
  background: var(--bg-tile);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
  max-height: 480px;
  overflow: auto;
  color: var(--sub);
  white-space: pre-wrap;
}
```

- [ ] **Step 2: Re-skin remaining run.js sections**

In `src/aiperf/operator/ui/views/run.js`:

**`FaultCallout`** — replace its render with:

```js
function FaultCallout({ bucket, conditions, pods }) {
  if (bucket !== 'fault') return null;
  const reasons = [
    ...(conditions ?? []).filter(c => c.status === 'False').map(c => `${c.type}: ${c.message ?? c.reason ?? '—'}`),
    ...(pods ?? []).filter(p => (p.phase || '').toLowerCase() === 'failed').map(p => `pod ${p.name} failed`),
  ];
  return html`
    <section class="run-fault-callout" data-testid="run-fault-callout" aria-label="Run fault details">
      <div class="run-fault-head">
        <span class="run-fault-head-tag">Run failed</span>
        ${reasons[0] ?? 'See conditions below for details.'}
      </div>
      ${reasons.length > 1 && html`
        <div class="run-fault-reasons">
          Likely causes:
          <ul>${reasons.slice(1).map(r => html`<li>${r}</li>`)}</ul>
        </div>
      `}
    </section>
  `;
}
```

**`ConditionsStrip`** — restyle as condition chips:

```js
function ConditionsStrip({ conditions }) {
  if (!conditions || conditions.length === 0) return null;
  return html`
    <section class="run-conditions" data-testid="run-conditions" aria-label="Conditions">
      ${conditions.map(c => {
        const tone = c.status === 'True' ? 'true' : c.status === 'False' ? 'false' : 'unknown';
        return html`
          <span class=${'run-condition-chip run-condition-chip--' + tone}>
            ${c.type} <span style="opacity:0.6">${c.status === 'True' ? '✓' : c.status === 'False' ? '✕' : '?'}</span>
          </span>
        `;
      })}
    </section>
  `;
}
```

**Phase swimlane → phase cards**: replace the `<!-- 3. PHASES SWIMLANE -->` block in the main `Run` return with:

```js
${phaseEntries.length > 0 && html`
  <section class="run-phases">
    <div class="run-phases-head">
      <div class="run-phases-title">Phases</div>
      <div class="run-phases-meta">
        ${phaseEntries.filter(([, p]) => p.complete).length} / ${phaseEntries.length} complete
      </div>
    </div>
    <div class="run-phases-grid">
      ${phaseEntries.map(([pname, p]) => {
        const pct = phasePct(p);
        const status = p.complete ? 'complete' : p.active ? 'running' : p.grace ? 'grace' : 'pending';
        const total = p.total_expected_requests ?? p.expected_requests ?? p.requests_total ?? null;
        const done  = p.final_requests_completed ?? p.requestsCompleted ?? p.requests_completed ?? p.completed ?? 0;
        const errors = p.errors ?? p.error_count ?? 0;
        const dur = p.elapsed_seconds ?? (p.start_ns && p.end_ns ? (Number(p.end_ns) - Number(p.start_ns)) / 1e9 : null);
        return html`
          <div key=${pname} class=${'run-phase run-phase--' + status}>
            <div class="run-phase-head">
              <span class="run-phase-name">${pname}</span>
              <span class=${'run-phase-badge run-phase-badge--' + status}>${status}</span>
            </div>
            <div class="run-phase-track"><div class="run-phase-fill" style=${'width: ' + pct + '%'}></div></div>
            <div class="run-phase-stats">
              <div class="run-phase-stat"><span class="run-phase-stat-label">Duration</span><span class="run-phase-stat-val">${dur != null ? fmtDuration(dur) : '—'}</span></div>
              <div class="run-phase-stat"><span class="run-phase-stat-label">Issued</span><span class="run-phase-stat-val">${fmtInt(done)}${total ? ' / ' + fmtInt(total) : ''}</span></div>
              <div class="run-phase-stat"><span class="run-phase-stat-label">Errors</span><span class="run-phase-stat-val">${fmtInt(errors)}</span></div>
            </div>
          </div>
        `;
      })}
    </div>
  </section>
`}
```

**`PodsBar`** — restyle:

```js
function PodsBar({ pods }) {
  if (!pods || pods.length === 0) return null;
  return html`
    <section class="run-pods" data-testid="run-pods">
      <div class="run-pods-title">Pods</div>
      <div class="run-pods-list">
        ${pods.map(p => {
          const ph = (p.phase || 'unknown').toLowerCase();
          return html`
            <span class="run-pod" title=${p.name}>
              <span class=${'run-pod-dot run-pod-dot--' + ph}></span>
              ${p.name}
            </span>
          `;
        })}
      </div>
    </section>
  `;
}
```

**`EventsPane`** — class swaps only (keep behavior):

Outer becomes `<section class="run-events" id="run-events" data-testid="run-events">` (or `run-events run-events--err` for the error variant). Title `<div class="run-events-title">Events</div>`. List `<div class="run-events-list">` with each event line `<div class="run-event run-event--warn|--error|''"><span class="run-event-ts">{ts}</span> {msg}</div>`.

**`LogsPane`** — class swaps:

Outer `<section class="run-logs" id="run-logs" data-testid="run-logs">`. Title row `<div style="display:flex; justify-content:space-between; align-items:center"><div class="run-logs-title">Logs</div><div class="run-logs-controls">…</div></div>`. Pod selector keeps `data-testid="run-logs-pod"`. Follow toggle keeps `data-testid="run-logs-follow"`. Tail size selector keeps `data-testid="run-logs-tail"`. Body `<pre class="run-logs-body" data-testid="run-logs-body">…</pre>`. Jump button `<button class="btn btn--ghost run-logs-jump" data-testid="run-logs-jump">Jump to latest</button>`.

**`GpuTelemetry`** — restyle to v2 grid:

```js
function GpuTelemetry({ metrics }) {
  if (!metrics || metrics.length === 0) return null;
  return html`
    <section class="run-gpu" data-testid="run-gpu">
      <div class="run-gpu-title">GPU Telemetry</div>
      <div class="run-gpu-grid">
        ${metrics.map(m => html`
          <div class="run-gpu-card">
            <div class="run-gpu-header">${m.endpoint ?? ''} · GPU ${m.gpu_index ?? '?'} ${m.model ? '· ' + m.model : ''}</div>
            <div class="run-gpu-primary">
              ${[
                { l: 'SM Util', v: m.sm_util,    u: '%' },
                { l: 'Memory', v: m.memory_used, u: 'GB' },
                { l: 'Power',  v: m.power,       u: 'W' },
                { l: 'Temp',   v: m.temp,        u: '°C' },
              ].map(t => html`
                <div class="run-gpu-tile">
                  <div class="run-gpu-tile-label">${t.l}</div>
                  <div class="run-gpu-tile-val">${t.v != null ? fmtNumber(t.v, 0) : '—'}<span class="run-gpu-tile-unit">${t.u}</span></div>
                </div>
              `)}
            </div>
          </div>
        `)}
      </div>
    </section>
  `;
}
```

(Field names — `sm_util`, `memory_used`, `power`, `temp` — should match what the existing component reads from `m`. If the existing field names differ, keep them; the markup is what matters.)

**`ResultsPane`** — restyle the file rows:

The outer `<section class="run-results …" data-testid="run-results">` keeps its testid. Replace the slab-head + slab-head-meta inner with:

```js
<div class="run-results-head">
  <div class="run-results-title">Results</div>
  <div class="run-results-meta">${state.kind === 'ok' ? state.files.length + ' files' : state.kind === 'loading' ? 'scanning…' : ''}</div>
</div>
```

Replace each file row (around line 184) with:

```js
<div
  class="run-results-row"
  data-testid=${'run-results-row-' + f.name}
  onclick=${() => downloadFile(f)}
>
  <span class=${'run-results-kind run-results-kind--' + fileKind(f.name)}>${fileKind(f.name)}</span>
  <span style="color:var(--text); font-family:var(--font-mono); font-size:var(--font-xs)">${f.name}</span>
  <span style="color:var(--muted); font-family:var(--font-mono); font-size:var(--font-xs); font-variant-numeric:tabular-nums">${fmtBytes(f.size)}</span>
  <span style="color:var(--sub); font-size:var(--font-xs)">↓</span>
</div>
```

The bundle button keeps `data-testid="run-results-bundle"` and uses `class="btn btn--primary"`.

**`RunHistoryDropdown` / `CompareWithDropdown`** — outer label `<label class="run-history" data-testid="run-history">…</label>` and `<select … class="" data-testid="run-history-select">…</select>` (CSS picks up the styles). Same for compare.

**`CancelButton` / `RelaunchButton`** — class becomes `btn btn--danger` for cancel, `btn btn--primary` for relaunch. Keep `data-testid="run-cancel"` and `data-testid="run-relaunch"`.

**Loading 404 state** — replace the `.run-404` markup with `<div class="empty">Locating ${name}…<br><span class="text-muted">namespace ${ns}</span></div>` (drop the `.run-404-glyph` magnifying glass and uppercase title).

- [ ] **Step 3: Final `style.css` sweep**

Walk `style.css` end-to-end. Delete:
- `body { overflow: hidden }` if present (kept for the WORKBENCH no-scroll layout; new shell scrolls).
- `#app::before` grid-paper substrate. `#app::after` vignette. (If still present.)
- `.slab-*`, `.run-meter*`, `.run-clock*`, `.run-header*`, `.run-lane*`, `.run-404*`, `.run-spark` (without `-tile`), `.run-conditions` legacy, `.run-events` legacy, `.run-logs` legacy, `.run-gpu` legacy, `.run-results` legacy, `.run-config` legacy — every block whose markup has been replaced this task or in Task 9.
- `--mauve, --teal, --rosewater, --sapphire, --lavender, --maroon, --peach, --orange, --flamingo, --crust, --base, --mantle, --f-serif, --f-display`, plus any rule that references them.
- The legacy `--ink-0..4` ladder, `--edge-1..3` ladder, `--paper*` family, `--amber` family, `--accent-blue/-cyan/-amber/-red/-green` aliases — anywhere they appear outside the new `:root` block.

Sanity check: search `style.css` for `paper`, `ink-`, `edge-`, `slab`, `WORKBENCH`, `MCC`, `FAULT`, `f-serif`, `f-display`, `rosewater`. Each hit is either in a comment to delete or a rule to delete.

Aim for a final file size of 1,500–2,000 lines. If it's over 2,500 lines, more legacy rules need to go.

- [ ] **Step 4: Run pre-commit + unit tests**

```bash
pre-commit run --all-files
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tee /tmp/t10.log
```

- [ ] **Step 5: Manual visual sanity**

Spec asks for a manual walk of every route. The implementer is not expected to bring up a live cluster, but must at minimum:

1. Run a quick static-only smoke: `python -m http.server -d src/aiperf/operator/ui 8910` and open `http://localhost:8910/` in a browser.
2. Verify the topbar reads "AIPerf Operator", the bench-error-flash (if rendered) says "Error" not "FAULT", the LogStrip pill row renders, the ⌘K palette opens (manual key press), every route hash works.
3. Note: live data won't render without an operator API; that's expected. Visual chrome is what we're sanity-checking.

If the browser shows obvious unstyled regions (giant unstyled tables, blocks of bare text, or untouched grid-paper background), step back to the relevant earlier task and fix the missing rule before committing.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/ui/views/run.js src/aiperf/operator/ui/style.css
git commit -s --no-verify -m "$(cat <<'EOF'
feat(operator-ui): re-skin Run view body + finalize stylesheet

Phase swimlane → phase cards. Pods strip → pod chips. Events / Logs /
GPU / Results / Conditions / Fault callout / collapsible Config all
adopt v2 vocabulary. Final style.css sweep removes the WORKBENCH
substrate (grid paper, ink ladder, paper ink, edge ladder, slab heads,
404 glyph) and the legacy color aliases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (run after implementation, not now)

- [ ] All routes render without 500 / unstyled blocks: `/`, `/launch`, `/archive`, `/analysis`, `/compare`, `/log`, `/run/:ns/:name`, `/run/:ns/:name/runs/:epoch`, `/compare/:ns/:name/:epochA/:epochB`.
- [ ] Topbar reads "AIPerf Operator". No "WORKBENCH" / "CALLSIGN" / "FLIGHT DECK" / "MCC" string anywhere visible.
- [ ] Search `src/aiperf/operator/ui/` for `'FAULT'` / `"FAULT"` / `>FAULT<`. None remain in user-visible copy. (The `bucket === 'fault'` JS identifier is fine — it's an internal key.)
- [ ] Every listed `data-testid` is still present in the rendered DOM.
- [ ] Keyboard map intact: `⌘K` opens palette, `⌘N` navigates to `/launch`, `Esc` from `/run` returns to `/`.
- [ ] `pre-commit run --all-files` clean.
- [ ] `uv run pytest -n auto tests/unit/` green.
- [ ] `style.css` final size 1.5–2k lines.
- [ ] No `git stash` / `git restore` was used (per `feedback_never_git_stash.md` / `feedback_never_git_restore.md`).

# Operator UI re-skin — adopt dashboard-v2 vocabulary

**Status:** design
**Date:** 2026-04-25
**Owner:** Anthony Casagrande

## Goal

Make the operator UI (`src/aiperf/operator/ui/`, served by the operator's
sidecar at `/`) read as the same product as the API dashboard
(`src/aiperf/api/static-v2/`, served by the controller-pod results sidecar).
Today the two surfaces visibly belong to different products: the operator
shell is a heavy "AIPERF // WORKBENCH" flight-deck theme (paper-cream ink on
near-black, sharp corners, grid-paper background, ALL-CAPS military copy),
while dashboard-v2 is clean modern dark cards with rounded corners, plain
copy, and a tight component vocabulary. We are adopting dashboard-v2's
look-and-feel wholesale on the operator UI.

The design preserves every route, every behavior, every test ID, every
keyboard shortcut, and every endpoint. It is a visual change plus a single
structural rewrite confined to one view (`run.js`).

## Non-goals

- No data-flow changes, no new endpoints, no behavior changes.
- No `ui-v1/` modifications — that legacy fallback is untouched.
- No new component files in `src/aiperf/api/static-v2/`. Dashboard-v2 stays
  exactly as it is; the operator UI moves *toward* it.
- No component extraction out of `views/run.js` into shared files. Even
  though the run view will adopt v2's component patterns, the markup lives
  inline in `run.js` — extracting to shared components is a follow-up.
- No copy changes to user-facing error text, log messages, or API
  payloads. Only chrome/labels/section headers in the UI change.
- No changes to keyboard shortcuts (`⌘K` palette, `⌘N` launch, `Esc`
  back-to-home) or to the persistent log strip's scroll/positioning
  behavior.
- No changes to the `data-testid` surface — the e2e tests in
  `tests/unit/api/test_dashboard_js.py` and the operator e2e suite must
  stay green without touching test code.

## Direction summary

Four user-confirmed direction calls during brainstorming:

1. **Aesthetic:** Burn down the WORKBENCH theme. Re-skin top to bottom to
   match dashboard-v2.
2. **Shell structure:** Keep the operator's chrome (top rail, persistent
   log strip, command palette, all routes). Re-skin only.
3. **Copy:** Drop the lingo wholesale. "AIPerf Operator", "Failed" not
   "FAULT", title-case section headers. Tiny labels (KPI tile titles,
   table headers) stay uppercase as visual tokens; no shouting at the user.
4. **Surgery depth:** Re-skin every view *plus* lift v2's component
   patterns into `run.js` — the only view whose hand-rolled hero / SLO
   chips / KPI tiles / phase cards / GPU panel / worker table currently
   drift from v2.

## Token system

`src/aiperf/operator/ui/style.css` is rewritten from dashboard-v2's
foundation. The new `:root` block adopts v2's tokens verbatim:

```css
:root {
  /* Surfaces */
  --bg: #0c0c0c;
  --bg-card: #161616;
  --bg-raised: #222222;
  --bg-tile: #0f0f0f;          /* KPI / GPU tile fill, slightly under card */
  --border: #313131;
  --border-hover: #4b4b4b;
  --border-subtle: #1a1a1a;

  /* Text */
  --dim:   #4b4b4b;            /* tiny labels */
  --muted: #757575;            /* secondary text */
  --sub:   #a7a7a7;            /* body */
  --text:  #eeeeee;            /* primary */
  --white: #ffffff;            /* big numbers, headlines */

  /* Accent — NVIDIA green stays as the single brand mark */
  --accent:     #76b900;
  --accent-dim: rgba(118, 185, 0, 0.15);

  /* Secondary palette (multi-series charts, chips, log categories) */
  --blue:  #3b82f6;
  --cyan:  #26c6da;
  --green: #76b900;
  --amber: #ffc107;
  --red:   #ef5350;
  --pink:  #ab47bc;

  /* Typography */
  --font-sans: 'IBM Plex Mono', ui-monospace, monospace;
  --font-mono: 'JetBrains Mono', monospace;

  --font-xs: 0.75rem;
  --font-sm: 0.8125rem;
  --font-base: 0.875rem;
  --font-md: 1rem;
  --font-lg: 1.125rem;
  --font-xl: 1.25rem;
  --font-2xl: 1.5rem;

  /* Layout */
  --top-bar-height: 48px;
  --radius: 6px;
  --radius-sm: 4px;
  --radius-lg: 8px;

  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-6: 1.5rem;
  --space-8: 2rem;

  --transition-fast: 120ms ease;
  --transition: 200ms ease;
}
```

What is **dropped** from the current operator stylesheet:

- All `--paper*` cream-on-black tokens.
- The `--ink-0..4` ladder (replaced by `--bg / --bg-card / --bg-raised /
  --bg-tile`).
- The `--edge-1..3` and `--edge-amber*` border ladder (single `--border`
  token plus `--border-hover` for interactive states).
- The `--amber*` family (the names lied — they're already NV green; the
  rename to `--accent / --green` removes the confusion).
- The legacy alias soup: `--mauve, --teal, --rosewater, --sapphire,
  --lavender, --maroon, --peach, --orange, --flamingo, --crust, --base,
  --mantle`. None of these have semantic meaning in the new system.
- The retired-but-aliased font tokens `--f-serif`, `--f-display`. Only
  `--font-sans` (IBM Plex Mono) and `--font-mono` (JetBrains Mono) survive.
- The grid-paper substrate (`#app::before` linear-gradients + radial
  ellipses) and the vignette (`#app::after`). The new `body` background is
  flat `--bg`.
- Sharp-corner override (`--radius: 0`). All radii are 4–8px.
- `body { overflow: hidden }` and `html { overflow: hidden }`. The new
  shell scrolls naturally; the only sticky element is the top rail.

What **stays** semantically (under new names):

- NVIDIA green as accent — it remains the strongest "this is NVIDIA
  software" signal. The brand-tied accent tilts the palette green even
  without the all-green WORKBENCH look.
- Mono-first body type (IBM Plex Mono) — both UIs already use it, so the
  operator UI does not switch font families, only retires the editorial
  fallbacks.
- Tabular-numeric stat values (`font-variant-numeric: tabular-nums`) on
  every numeric cell. The change is *which* mono — JetBrains Mono picks up
  numerics from v2 to match the dashboard.

## Shell chrome

DOM and JS structure preserved verbatim. Only CSS classes and copy change.

### TopRail (`components/top-rail.js` + corresponding CSS)

- 48px sticky bar on `--bg-card` with hairline bottom border (matches v2
  `.topbar`).
- Logo group: green-filled `AI` badge + `AIPerf Operator` wordmark.
  Replaces `AIPERF · WORKBENCH` callsign markup. The component still
  receives `viewKind` and `runParams`; only the rendered text changes.
- Breadcrumb trail: `·`-separated mono text in `--sub`, current crumb in
  `--text`. Drops the all-caps + corner-bracket reticles around crumbs.
- LAUNCH CTA → primary `Launch` button: `--accent` background, `--bg`
  foreground, 4px radius, IBM Plex Mono uppercase label. Pinned to the
  right group.
- ⌘K affordance: small pill-styled key chord (`<span class="kbd">⌘ K</span>`)
  that triggers the existing `onSearchClick` handler. Same keymap as
  today (`Ctrl/Cmd+K` toggles palette, `Ctrl/Cmd+N` navigates to
  `/launch`, `Esc` from a run goes to `/`).
- Run-history dropdown / epoch switcher: when `viewKind === 'run'`, an
  `epoch` pill renders next to the breadcrumb (mono text, `--bg-raised`
  background, hairline border). Behavior unchanged.

### LogStrip (`components/log-strip.js`)

- Always-on positioning preserved (`grid-area: log` in the bench layout).
- Re-skinned to v2's `LogPane` look: `--bg-card` background, hairline top
  border, IBM Plex Mono entries, `--dim` timestamps, category chips
  (`log-cat--phase`, `log-cat--worker`, `log-cat--records`) and severity
  rules (`log-entry--warn`, `log-entry--error`).
- Severity filter pills (`log-filter`, `log-filter--active`,
  `log-filter-count`) appear in the header row, matching v2.
- Drops the corner-bracket reticles, the `paper-cream` ink, and the
  amber-glow on hover.

### CommandPalette (`components/command-palette.js`)

- Modal card on `--bg-card` with 8px radius and hairline border.
- Search input: `--bg-tile` fill, hairline border, mono font, focus-ring
  is `--accent` at 0.35 alpha (no green-glow box-shadow).
- Result rows: hover background `--bg-raised`, mono labels in `--text`,
  kind tag in `--sub` (right-aligned).
- Keyboard map preserved (`↑/↓` to move, `Enter` to select, `Esc` to
  close).

### Bench error flash (`bench-error-flash` in `app.js`)

- Replaces the "FAULT" word with "Error".
- Red-tinted card: `rgba(239, 83, 80, 0.08)` background,
  `rgba(239, 83, 80, 0.45)` border, 8px radius, IBM Plex Mono body.
- Headline `<strong>Error</strong>` in `--white`, body in `--sub`.

## Views — re-skin only

For each of the views below: **DOM stays**, **state stays**, **routes
stay**, **API calls stay**, **`data-testid` attributes stay**. Only CSS
class names, copy, and small markup wrappers change.

### `home.js`

- Active-run hero card: `card` container, hairline border, 8px radius.
  Uses the same `HeroStrip`-style three-column grid as the run view (see
  next section) when an active run exists; otherwise the existing
  pick-one-card list with v2 `card` styling.
- Job cards in the pick-one list: `card` background, hairline, mono
  job-name in `--text`, namespace + last-update in `--sub`/`--dim`,
  status chip from the `worker-status` family (`healthy / running /
  failed / idle`).
- Sort order preserved (LIVE → FAILED → PASSED).
- "LAUNCH" CTA tile becomes a primary `--accent` button.

### `archive.js`

- Namespace groups stay; group headers re-styled in mono `--sub` with a
  hairline rule under each.
- Namespace counts: `worker-status` chips (`error` for failed, `running`
  for live, `healthy` for passed). Drops the `ns-count--fail "FAULT"`
  shoutmark.
- Tab strip on `--bg-card` with hairline bottom; active tab gets an
  `--accent` underline (no all-caps shoutmark).
- Job rows: hairline separators, hover `--bg-raised`, `data-testid`s
  (`ns-count`, `tab-archived`, etc.) preserved.
- "Archived" tab badge becomes a small subdued `--bg-raised` chip.

### `analysis.js` / `compare.js`

- Chart canvases sit in `card` containers (8px radius, hairline).
- Legend, grid, tooltip restyled by `chart-theme.js` (see Charts
  section).
- `compare.js` becomes a 2-column grid of `card`s, each with the same
  internal layout (KPI strip on top, chart in the middle, diff-rows at
  the bottom). Diff-row badges adopt the SLO chip family (`chip--good`,
  `chip--warn`, `chip--bad`).
- Run-pair selector at the top: two mono pills with `▾` glyphs, hairline
  borders, `--bg-tile` fill.

### `log.js`

- Durable run-log view re-skinned exactly like `LogStrip` — same CSS
  class family. Adds a `log-filters` pill row at the top (severity
  toggles).
- Drop the corner brackets, drop the editorial-serif mark.

### `launch.js`

- YAML editor stays. The textarea's syntax-highlight rules are not
  touched in this pass.
- Surrounding chrome (template picker, submit, schema-warning callout)
  re-skinned to v2 `card`s.
- Submit button: primary `--accent` (mono uppercase label, white-on-green
  is too low-contrast for accessibility — use `--bg` foreground).
- Cancel / secondary actions: ghost button — transparent background,
  hairline `--border`, hover `--bg-raised`.
- Schema-warning callout: amber-tinted card, same shape as the
  fault callout in the run view.

## `/run` view — structural rewrite

This is the only place we **lift v2 component patterns** instead of
just re-skinning. Today `views/run.js` is 1,630 lines of hand-rolled
markup that drifts from v2's clean component vocabulary. After this
change, the run view's markup mirrors `static-v2/components/*` while
staying inline in `run.js`.

The blocks below describe the new internal structure of `run.js`. Same
data sources, same hooks, same epoch-switching logic, same fetches.

### Hero (replaces the bespoke hero block at the top of the run view)

Three-column grid: `minmax(260px, 1.2fr) minmax(180px, 0.8fr) minmax(300px, 2fr)`.

- **Health column:** pulsing status dot (18px) + label + reasons line.
  Border tint follows verdict (`hero--ok` / `hero--warn` / `hero--error` /
  `hero--idle`). Inset box-shadow at 0.12 alpha for emphasis on running
  health states.
- **Clock column:** elapsed and ETA stacked, mono values, separator
  border on the left.
- **Phase column:** phase name + percent in head row, then the progress
  track (8px-tall, 4px-radius), then the sub-line (issued / target).
  Fill color follows verdict tint (blue running, green done, red error,
  amber warn).

The run-history dropdown and epoch switcher slot **into the TopRail**
(see Shell chrome above), not the hero — keeps the hero focused on
"what's the run doing right now?".

### KPI tile grid (`kpi-grid` + `kpi-tile`)

`grid-template-columns: repeat(auto-fit, minmax(200px, 1fr))`.

Each tile:

- **Head:** label + optional secondary stat (`kpi-tile-primary-stat`,
  e.g., "p50 / sustained") on the left; SLO threshold chip on the right
  (`kpi-chip--good / --warn / --bad`).
- **Big value:** 26px-bold JetBrains Mono number + small `--muted` unit.
- **Inline SVG sparkline** (160×24) — kept; rebuilt with the new
  per-state stroke colors (`--green` good, `--amber` warn, `--red` bad).
- **Sub-line:** "last 5m" + delta (`--sub` value).

Per-tile border tint: `kpi-tile--slo-good / --slo-warn / --slo-bad` only
when an SLO is defined. Without an SLO, neutral border.

The set of tiles rendered (throughput, p99, TTFT, TPOT, error rate,
active workers, …) is unchanged from today — the markup is the only
thing that swaps.

### Phase cards (`phases-grid` + `phase-card`)

`grid-template-columns: repeat(auto-fit, minmax(280px, 1fr))`.

Per phase:

- **Head:** phase name (mono, `--text`) + status badge
  (`phase-badge--running / --pending / --complete / --grace`).
- **Track:** 6px height, 3px radius, fill color follows status.
- **Stats grid:** 3 columns, mono labels + JetBrains Mono values
  (Duration / Issued / Errors).

### Worker table (`worker-table`)

- Header row: 10px uppercase `--muted` labels, single hairline below.
- Body rows: mono IDs, hairline separators, hover `--bg-raised`.
- `worker-status` chip family (`healthy / high_load / error / idle /
  stale`) replaces the today's amber-bedded labels.
- Existing worker-group toggle preserved (`group-toggle` glyph + child
  rows with `worker-child-row` background tint).

### GPU telemetry (`gpu-grid` + `gpu-card`)

- Per GPU: mono header (node + GPU index + model), 4-tile primary stat
  row (SM Util / Memory / Power / Temp), and an extras table for
  secondary metrics (`gpu-extra` — single-column key-value with
  hairline separators).

### Server metrics (`server-metrics`)

- 3-column table: Metric / Value / Saturation chip.
- Saturation rules: `server-metrics-row--warn` paints the value cell
  amber, `--bad` paints it red. `server-chip--good / --warn / --bad` is
  the trailing badge.

### Inline log section (`log-pane`)

- Adopts the same look as `LogStrip` (severity filter pills + category
  chips + severity color rules). Limit ~240 entries; uses the same
  `aiperf.api.routers.logs` socket.

### Fault callout (replaces `<!-- 1c. FAULT CALLOUT -->`)

A red-tinted `card` rendered above the KPI grid when the run's verdict
is `failed`:

```html
<div class="fault">
  <div class="fault-head">
    <span class="lbl">Run failed</span> {short reason}
  </div>
  <div class="fault-rs">
    Likely causes:
    <ul>
      {bulleted list of structured reason objects from /api/v1/runs/...}
    </ul>
  </div>
  <div class="fault-actions">
    <button class="btn btn--primary">View Logs</button>
    <button class="btn">Download Bundle</button>
    <button class="btn">Re-launch</button>
  </div>
</div>
```

Drops "FAULT // CALLSIGN" shouting. The reasons list is what the
operator already collects (deadlock, last flush stalled, heartbeats
missed, …) — we just stop framing it as flight-deck telemetry.

### Request-latency timeline / throughput-vs-latency chart

Chart.js canvases stay where they are. `chart-theme.js` (next section)
drives the new look — no per-callsite chart options change in this pass.

## Charts — `chart-theme.js`

Idempotent re-emit of Chart.js defaults, same `_initialized` gate.

```js
export const PALETTE = [
  '#76b900', // --accent (NV green) — primary series
  '#3b82f6', // --blue — secondary
  '#26c6da', // --cyan — tertiary
  '#9fe870', // light green — quaternary
  '#ffc107', // --amber — warn / fifth
  '#ef5350', // --red — error / sixth
  '#ab47bc', // --pink — seventh
  '#a0d8ff', // sky — eighth
];

const MONO_FAMILY  = "'IBM Plex Mono', ui-monospace, monospace";
const NUMER_FAMILY = "'JetBrains Mono', monospace";

export function applyChartTheme(options = {}) {
  if (!_initialized && typeof window !== 'undefined' && window.Chart) {
    const C = window.Chart;
    C.defaults.font.family = MONO_FAMILY;
    C.defaults.font.size = 10;
    C.defaults.color = '#a7a7a7';        // --sub
    C.defaults.borderColor = '#313131';  // --border
    C.defaults.scale.grid.color = 'rgba(49, 49, 49, 0.5)';
    C.defaults.scale.grid.tickColor = 'transparent';
    C.defaults.scale.grid.borderColor = 'transparent';
    C.defaults.plugins.tooltip.backgroundColor = 'rgba(22, 22, 22, 0.96)';
    C.defaults.plugins.tooltip.titleColor = '#eeeeee';
    C.defaults.plugins.tooltip.bodyColor  = '#a7a7a7';
    C.defaults.plugins.tooltip.padding = 12;
    C.defaults.plugins.tooltip.cornerRadius = 6;          // was 0
    C.defaults.plugins.tooltip.displayColors = false;
    C.defaults.plugins.tooltip.boxPadding = 6;
    C.defaults.plugins.tooltip.borderColor = '#313131';   // was accent green
    C.defaults.plugins.tooltip.borderWidth = 1;
    C.defaults.plugins.tooltip.titleFont = { family: MONO_FAMILY,  size: 11, weight: '700' };
    C.defaults.plugins.tooltip.bodyFont  = { family: NUMER_FAMILY, size: 11 };
    C.defaults.plugins.legend.labels.usePointStyle = true;
    C.defaults.plugins.legend.labels.boxWidth = 8;
    C.defaults.plugins.legend.labels.padding = 12;
    C.defaults.plugins.legend.labels.font = { family: MONO_FAMILY, size: 10, weight: '600' };
    C.defaults.elements.line.tension = 0.3;
    _initialized = true;
  }
  return options;
}
```

Numeric tick labels switch to JetBrains Mono via per-callsite scale
config in callers that want it (existing `analysis.js` / `compare.js`
chart configs already pass an `options` object — bumping the `ticks.font`
family is a one-line edit per callsite).

## Files touched

| Path | Change |
| --- | --- |
| `src/aiperf/operator/ui/index.html` | `<title>AIPerf Operator</title>`; remove `Host Grotesk` and `IBM Plex Sans` from the Google Fonts URL; add `JetBrains Mono`; clean the comment block. |
| `src/aiperf/operator/ui/style.css` | Full rewrite from v2's foundation. Final size ~1.6–2k lines (down from 3,147). |
| `src/aiperf/operator/ui/lib/theme.js` | Drop the `--mauve / --teal / --rosewater / etc.` legacy aliases the file emits at runtime. Mirror the new tokens. |
| `src/aiperf/operator/ui/lib/chart-theme.js` | Replace as shown in Charts section. Same `applyChartTheme` signature, same `_initialized` gate. |
| `src/aiperf/operator/ui/components/top-rail.js` | Logo + breadcrumb copy edits, class swaps, kbd-pill markup. |
| `src/aiperf/operator/ui/components/log-strip.js` | Class swaps + filter pill row. |
| `src/aiperf/operator/ui/components/command-palette.js` | Class swaps. |
| `src/aiperf/operator/ui/components/chart-wrapper.js` | Class swaps; verify `applyChartTheme` is still called once on mount. |
| `src/aiperf/operator/ui/views/home.js` | Class swaps; copy edits. |
| `src/aiperf/operator/ui/views/archive.js` | Class swaps; "FAULT" → "Failed"; tab strip restyled. |
| `src/aiperf/operator/ui/views/analysis.js` | Class swaps; cards restyled. |
| `src/aiperf/operator/ui/views/compare.js` | Class swaps; 2-col card grid. |
| `src/aiperf/operator/ui/views/log.js` | Class swaps; filter pills. |
| `src/aiperf/operator/ui/views/launch.js` | Class swaps; primary/ghost button styles; schema warning card. |
| `src/aiperf/operator/ui/views/run.js` | **Structural rewrite** of the markup blocks per the `/run` section above. State, hooks, fetches, epoch switcher logic, and every `data-testid` are preserved verbatim. Final size estimated ~1,100–1,300 lines (down from 1,630 — the bespoke hero/SLO chip helpers shrink). |
| `tests/unit/api/test_dashboard_js.py` | **Not touched.** Runs unchanged. |
| `tests/e2e/operator-ui-*.py` (or wherever the e2e operator tests live) | **Not touched.** |
| `src/aiperf/operator/ui-v1/**` | **Not touched.** Legacy fallback. |
| `src/aiperf/api/static-v2/**` | **Not touched.** Source of truth for the look. |

## Verification

- `pre-commit run --all-files` clean.
- `uv run pytest tests/unit/ -n auto` green (the `data-testid`
  preservation guarantees `test_dashboard_js.py` keeps passing).
- Manual: serve the operator UI (helm install + port-forward, or local
  dev mode) and visit each route — `/`, `/launch`, `/archive`,
  `/analysis`, `/compare`, `/log`, `/run/:ns/:name` (both healthy and
  failed runs) — confirm:
  - Topbar shows "AIPerf Operator", breadcrumbs read in mono `·`-form,
    no callsign / WORKBENCH / FAULT shoutmarks.
  - LogStrip pinned to bottom, severity filters work, category chips
    render correctly.
  - ⌘K opens the palette; ⌘N navigates to `/launch`; Esc returns to `/`.
  - Run view: hero changes border tint with verdict; KPI tiles light up
    SLO chips correctly; phase cards reflect lifecycle; worker-group
    toggles still expand; GPU telemetry tiles populate; server-metrics
    rows show saturation chips.
  - Failed run shows the new red fault callout with reasons + actions.
- Manual: open Chart.js views (`/analysis`, `/compare`, throughput-vs-
  latency in `/run`) — confirm tooltip + grid + legend match v2.

## Risks and mitigations

- **CSS regression.** A 3,147-line stylesheet has unknown couplings. We
  mitigate by walking each view route post-rewrite and confirming all
  `data-testid` elements still render with reasonable layout.
- **Chart-theme drift.** `chart-theme.js` is shared between `analysis`,
  `compare`, and `run` chart blocks. The `_initialized` gate already
  prevents multiple-application; we only need to confirm new defaults
  are picked up on cold load.
- **Operator UI vs. dashboard-v2 drift over time.** Future v2 component
  changes will not auto-flow into the operator UI's inline copies. This
  is accepted scope for this branch; component extraction is a follow-up.

## Out of scope (explicit)

- Component extraction from `run.js` into shared files reused by both
  dashboards — follow-up.
- Tightening copy beyond drop-the-shouting (e.g., better SLO descriptions,
  better fault-reason summaries) — follow-up.
- Accessibility audit (focus rings, color contrast on amber chips) —
  flagged for a follow-up pass.
- Mobile / narrow-viewport layout — operator UI is a desktop tool, no
  responsive changes here.

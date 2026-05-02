# Job-detail page restyle — modern split-pane workspace

**Status:** design
**Date:** 2026-05-02
**Owner:** Anthony Casagrande

## Goal

Re-style and re-organize the AIPerf operator UI's job-detail page (`src/aiperf/operator/ui-v1/pages/job-detail.js`) into a modern split-pane workspace that retains the NVIDIA dark + green identity, slims the secondary categorical palette from 7+ hues to 4 + accent, adds a system-preference-driven light theme variant, and reorganizes the 19 stacked sections into a sticky-identity / main-column / sticky-context-rail layout. The page must read as comparable to Linear, Vercel, Anthropic Console, and Modal — restrained, structured, demo-friendly, no rainbow pills.

## Non-goals

- **No data-flow changes**, no new endpoints, no behavior changes.
  Every route, polling cadence, websocket subscription, test ID, keyboard
  shortcut, and `data-testid` is preserved.
- **No removal of any of the 19 sections** — they all keep rendering, only
  their *placement* changes (main column vs. right rail vs. drawer).
- **Only `pages/job-detail.js` and the components it imports change.** The
  operator shell (top nav, command palette, log strip), the dashboard
  page, the leaderboard page, and the sweeps pages are out of scope for
  this spec — they will adopt the same token system in a follow-up.
- **No new design system** — tokens are added to the existing
  `style.css`. No CSS-in-JS, no Tailwind adoption, no styled-components.
- **No copy / message changes** — only chrome and section headers shift.
- **No removal of any colored elements that carry semantic meaning**
  (live indicator, error tones, SLA pip, phase progress).

## Direction summary

Six user-confirmed direction calls during brainstorming:

1. **Scope:** Keep NV dark + green identity. Add a light theme variant.
   Polish for screenshots. Slim the secondary palette down. (`A + C + D + E`)
2. **Categorical color rule:** Functional, but slim. Keep group stripes
   in the metrics table and per-row format chips on the artifacts table.
   Drop from ~7 hues to **4 categorical + accent**: throughput · latency ·
   tokens · errors, plus NV green for accent and amber for warning.
3. **Body language (dark):** Inter for prose, JetBrains Mono for
   numerics. Gradient cards `linear-gradient(180deg, #181820, #131316)`,
   10px radius, soft area sparklines with end-dot.
4. **Header pills replaced by typed key/value pairs.**
   `phase profiling · ns ... · model ... · run ... · elapsed ... · epoch ...`.
   Only `phase`'s value carries the NV green accent; `ns`, `model`,
   `run`, `elapsed`, `epoch` values stay neutral mono.
5. **Page layout:** Option C — split-pane workspace. Sticky identity bar
   at top, main analytical column on left, sticky 280px context rail on
   right, diagnostics moves to a slide-in drawer. (Linear / Anthropic
   Console / Cursor pattern.)
6. **Light palette:** Tailwind-600 set for categorical (`#2563eb`,
   `#ea580c`, `#9333ea`, `#dc2626`, `#d97706`); NV green stays at
   canonical `#76b900` as the accent. Both themes use the same layout,
   same component shapes, same KV header.

## Token system

A new theme block is added to `src/aiperf/operator/ui-v1/style.css`. Both
themes share the same token *names* — the implementation toggles between
two value sets via `[data-theme="light"]` on `<html>`.

### Dark (default — preserves current charcoal stack)

```css
:root,
[data-theme="dark"] {
  /* Surfaces */
  --bg:        #0e0e10;
  --bg-card:   #131316;          /* solid fallback for non-gradient surfaces */
  --bg-raised: #1c1c22;          /* tile bg, KPI hover */
  --card-grad: linear-gradient(180deg, #181820 0%, #131316 100%);
  --card-shadow: 0 1px 0 rgba(255,255,255,0.03) inset;
  --border:        #25252b;
  --border-strong: #2c2c33;
  --border-subtle: #1d1d22;

  /* Text */
  --text:       #ececec;         /* primary */
  --text-strong:#ffffff;          /* numerics, headings */
  --sub:        #c0c0c8;         /* table values */
  --muted:      #87878f;         /* labels, sub-lines */
  --dim:        #6b6b73;         /* KV keys */
  --faint:      #5d5d65;         /* axis ticks, "—" placeholders */
  --hairline:   #3a3a40;         /* KV separators */

  /* Accent — NVIDIA green (slightly brighter for dark contrast) */
  --accent:        #94d340;
  --accent-strong: #76b900;      /* used for hover/active */
  --accent-tint:   rgba(148,211,64,0.10);
  --accent-border: rgba(148,211,64,0.30);

  /* Categorical (4 hues, used for table-group stripes + format chips) */
  --cat-throughput: #7aa9d6;     /* blue */
  --cat-latency:    #cf9a6e;     /* peach */
  --cat-tokens:     #b691d4;     /* mauve */
  --cat-errors:     #e57373;     /* red */
  --warn:           #ffc107;     /* amber — warnings, JSON file chip */

  /* Status */
  --ok:    var(--accent);
  --bad:   var(--cat-errors);
  --info:  var(--cat-throughput);

  /* Typography */
  --font-sans: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace;
}
```

### Light (system-pref + manual toggle)

```css
[data-theme="light"] {
  --bg:        #f6f6f8;
  --bg-card:   #ffffff;
  --bg-raised: #fafafb;
  --card-grad: #ffffff;          /* solid white card, no gradient */
  --card-shadow: 0 1px 2px rgba(20,20,30,0.04);
  --border:        #e2e2e6;
  --border-strong: #d8d8de;
  --border-subtle: #ececef;

  --text:        #1a1a1c;
  --text-strong: #0a0a0c;
  --sub:         #2a2a30;
  --muted:       #6b6b73;
  --dim:         #8a8a92;
  --faint:       #b0b0b8;
  --hairline:    #c8c8cd;

  --accent:        #76b900;      /* canonical NVIDIA green for white bg */
  --accent-strong: #5d9400;
  --accent-tint:   rgba(118,185,0,0.10);
  --accent-border: rgba(118,185,0,0.40);

  /* Categorical — Tailwind-600 set */
  --cat-throughput: #2563eb;
  --cat-latency:    #ea580c;
  --cat-tokens:     #9333ea;
  --cat-errors:     #dc2626;
  --warn:           #d97706;
}
```

### Theme switching

A new helper module `src/aiperf/operator/ui-v1/lib/theme-switch.js`:

- On boot, reads `localStorage.aiperfTheme` (`"dark"` | `"light"` | `"auto"`).
- `"auto"` (default) follows `window.matchMedia('(prefers-color-scheme: light)')`
  and updates live on system change.
- Sets `document.documentElement.dataset.theme = "dark" | "light"`.
- Exports `setTheme(t)` so a top-bar toggle (added to `top-nav.js`) can
  switch between dark / light / auto.
- The toggle is a single icon button in the top-right of the existing
  top-nav, next to the search button. Clicking cycles
  `auto → light → dark → auto`. Tooltip shows current resolved theme.

The legacy aliases at the bottom of `style.css` (`--surface0`, `--mauve`,
`--ctp-base`, …) keep their existing dark values and gain light-theme
counterparts in the same `[data-theme="light"]` block, so any imported
legacy CSS keeps working without per-file edits.

## Layout — split-pane workspace

`pages/job-detail.js` returns this structure:

```jsx
<div class="job-detail" data-testid="page-job-detail">
  <IdentityBar … />            {/* sticky top */}
  <div class="job-detail__body">
    <div class="job-detail__main"> … </div>
    <aside class="job-detail__rail" aria-label="Run context"> … </aside>
  </div>
  {diagnosticsOpen && <DiagnosticsDrawer onClose=… />}
</div>
```

CSS:

```css
.job-detail__body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 0;
}
@media (max-width: 1100px) {
  .job-detail__body { grid-template-columns: 1fr; }
  .job-detail__rail { border-left: none; border-top: 1px solid var(--border); }
}
.job-detail__main { padding: var(--space-4); display: flex; flex-direction: column; gap: var(--space-3); min-width: 0; }
.job-detail__rail { padding: var(--space-4); border-left: 1px solid var(--border); background: var(--bg-raised); position: sticky; top: var(--top-bar-height); align-self: start; max-height: calc(100vh - var(--top-bar-height)); overflow-y: auto; display: flex; flex-direction: column; gap: var(--space-3); }
```

### Identity bar (replaces the current header card)

```jsx
<header class="job-detail__id" data-testid="job-detail-id">
  <div class="job-detail__id-row1">
    <h2 class="job-detail__id-name">{name}</h2>
    <div class="job-detail__id-actions">
      <CmdKHint />               {/* "⌘K commands" */}
      <LiveIndicator state={liveState} />
      <OverflowMenu actions={…} />
    </div>
  </div>
  <div class="job-detail__id-kv">
    <KV k="phase"   v={phase}    accent />
    <KV k="ns"      v={namespace}/>
    <KV k="model"   v={model}/>
    <KV k="run"     v={runId}/>
    <KV k="elapsed" v={elapsed}/>
    <KV k="epoch"   v={epochLabel}/>
  </div>
  <div class="job-detail__id-endpoint">{endpointUrl}</div>
</header>
```

`KV` is a new tiny component (`components/kv.js`) — uppercase Inter key
in `--dim`, JetBrains Mono value in `--sub`, optional `accent` prop
makes the value `--accent` weight-600. Dot separators between pairs
in `--hairline`.

### Main column section order

1. **Hero KPI grid** — 3 large tiles with full anatomy: icon, label,
   value+unit, sub-line with trend badge and optional pip, 220×36 soft-
   area sparkline with end-dot, gradient progress bar. Defaults:
   Output token throughput · Goodput · TTFT p99. Hero tile selection is
   driven by run state (sweeping runs swap in the swept metric).
2. **Secondary KPI rail** — 4 mini tiles (Req latency p50, ITL p95,
   Requests/min, Error rate), 150×22 sparkline each, no progress bar.
3. **Live charts row** (only when `polling === true`) — 2-up grid:
   Live throughput + Live latency timeline. Each ~160px tall with
   dashed gridlines, end-point dot, tabular x-axis ticks.
4. **Latency percentile distribution + Concurrency curve** — 2-up.
5. **Full metrics breakdown** — grouped table. Sticky group headers
   (`Throughput · Latency · Tokens · Errors`) with the categorical hue
   chip. Per-row 2px left stripe in the group hue. Mono numerics, "—"
   placeholders in `--faint` for cells outside a metric's allowed
   percentile set.
6. **Server / GPU metrics** (if enabled) — 4-tile mini rail (Util / Mem
   / Power / Temp), each with sparkline.
7. **Artifacts** — see Artifacts card below.

The KPI rail, secondary rail, live charts, distribution + concurrency
charts, metrics table, server metrics, and artifacts are all the
existing components rewrapped — no new visualizations, only re-tokened.

### Right rail (sticky, 280px)

In order:

1. **Phase progress** — replaces `PhaseStrip`. 3-segment track with
   per-phase progress fill, labels, elapsed + ETA below.
2. **Pods** — replaces `PodsStrip`. 6-cell mini heatmap (one square per
   pod, NV green for healthy, amber for pending, red for failed, dim
   grey for not-yet-scheduled), plus controllers/workers KV.
3. **Records** — replaces `RecordsStrip`. served · errors · success% ·
   in-flight as KV pairs.
4. **SLA compliance** — replaces the panel. One row per SLO with
   `target` key and `actual + ✓/✗` value, footer "N / M meeting" pip.
5. **Config summary** — replaces `JobConfigSection` for the rail.
   Shows concurrency · prompts · request count · isl · osl · server.
   "view spec ↗" link opens the existing `SpecViewerModal`.
6. **Sweep info** — replaces the inline `SimilarRunsLink` + sweep
   sub-line. Name, variation, swept-parameter, "open sweep view ↗"
   link.
7. **Actions** — replaces inline cancel / relaunch buttons. Single
   stack of action rows: Cancel run (danger), Download artifacts,
   Compare to similar runs, Open diagnostics drawer.

### Diagnostics drawer (replaces the always-visible `DiagnosticsPanel`)

The current `DiagnosticsPanel` (events + logs + pods-detail + conditions
tabs) becomes a slide-in drawer triggered from the rail's actions and
from the `?diag=conditions` URL parameter (which already exists). The
drawer is 420px wide on desktop, full-screen on mobile, with
`Esc`-to-close. The existing `?diag=` query param continues to control
which tab is active when opened from a deeplink.

## Component-level changes

### `KpiCard` (`components/kpi-card.js`)

No API change. Internal CSS rewrite to use the new tokens. Sparkline
sizes: hero 220×36 (was 140×26), mini 150×22. Gradient progress bar
(was solid). Trend badge accepts `▲ 12%` / `▼ 4%` glyphs with green for
"good direction", peach/orange for "bad direction" — direction is
metric-specific (throughput up = good, latency up = bad).

### `KpiRail` (`components/kpi-rail.js`)

Splits into `<HeroKpiGrid>` (3 large) and `<SecondaryKpiRail>` (4
mini). Same data sources; just two CSS containers with different grid
templates. Both stay in the same module — no new files.

### `MetricsTable` (inside `pages/job-detail.js`)

`METRIC_GROUPS` colors switch from `palette.blue/peach/mauve/red` to
`var(--cat-throughput/-latency/-tokens/-errors)`. Sticky group header
bar inside the table gains `position: sticky; top: 0` so the group
context survives scroll. Per-row 2px left stripe replaces the existing
border-left. No structural change — same auto-discovery `Other Metrics`
tail group.

### `Artifacts` (the file-browser block at the bottom of the page)

Currently a simple list. Becomes a card with:

- **Card head**: title `Artifacts`, sub `N files · X MB total`,
  primary button `⤓ Download all (.zip)` (solid NV green, white text),
  secondary button `{ } Quick export JSON` (white-on-card outline),
  overflow `⋯` for less-common ops (re-archive, copy share link).
- **Filter strip** (replaces the chip bar inline above the file list):
  `Filter` label + `ALL · N` chip + one chip per discovered format with
  count. Inactive chips at 55% opacity. The active chip gets a 2px
  outer ring (`box-shadow: 0 0 0 2px rgba(255,255,255,0.10)` dark /
  `rgba(20,20,30,0.08)` light).
- **File table**: zebra rows (`#131316` / `#0f0f12` dark; `#fbfbfc` /
  `#ffffff` light), with `rgba(118,185,0,0.06)` hover. Per-row layout:
  56px **format chip** (mono caps, solid hue from
  `fileTypeChip(filename)`), filename (mono `--text-strong`, with the
  path-prefix portion in `--faint` so subfolder dumps stay scannable),
  size (mono `--muted`, right-aligned), action links
  (`view · download`, in `--cat-throughput`, separated by `·` in
  `--hairline`).
- **Footer**: `Show all N files` link + total size pip when truncated
  to the first 8 rows.

Two new endpoints are required from the operator:

- `POST /api/jobs/{ns}/{name}/runs/{epoch}/archive` — stream a zip of
  all artifacts (used by Download all). Implementation can wrap the
  existing `results_layout` directory walk + `zipfile.ZipFile`. Must
  honor the results-ready marker.
- `GET /api/jobs/{ns}/{name}/runs/{epoch}/profile_export?format=json` —
  alias for the existing canonical `profile_export_aiperf.json`. Used
  by Quick export JSON.

Both endpoints are added in
`src/aiperf/operator/routers/results_files.py`.

### `RelaunchButton`, cancel button, sweep link, similar-runs link

These all migrate from the header card into the right rail's Actions
card. Their current implementations stay; only the parent JSX changes.

### `top-nav.js`

Add a single icon button (sun/moon/auto-toggle) to the right side, next
to the existing search button, calling `setTheme(...)`. Tooltip shows
the resolved theme.

## Testing

- All existing `data-testid` selectors stay. The page-level testid
  `page-job-detail` continues to wrap the whole view; new sub-testids
  are added (`job-detail-id`, `job-detail-rail`, `job-detail-rail-phase`,
  `job-detail-rail-pods`, `job-detail-rail-sla`, `job-detail-rail-config`,
  `job-detail-rail-sweep`, `job-detail-rail-actions`,
  `job-detail-artifacts-download-all`,
  `job-detail-artifacts-quick-export`,
  `job-detail-theme-toggle`).
- The existing operator UI e2e tests in `tests/operator-ui-e2e/` and
  the dashboard tests in `tests/unit/api/` must stay green. The split
  layout is asserted by a new e2e case
  `test_job_detail_split_pane_layout.py` that verifies: identity bar
  is sticky, right rail is 280px wide on desktop, right rail collapses
  below the main column under 1100px, theme toggle cycles and
  persists.
- A new screenshot test `test_job_detail_theme_screenshots.py` captures
  the page in dark and light at 1440×900 and stores them under
  `docs/media/images/job-detail-{dark,light}.png` (overwriting in place
  per the dashboard-screenshots convention).

## Migration / rollout

The work is one PR on the user's current branch (`ajc/k8s`).
Sub-commits in this order so each one is independently reviewable:

1. Token block in `style.css` (dark theme, preserves all existing
   visual output bit-for-bit by using current values for the new
   `--cat-*` tokens too).
2. Light theme block + `theme-switch.js` + top-nav toggle.
3. New `KV` component + `IdentityBar` extraction + replace header card
   with identity bar.
4. Split-pane layout shell (`__body / __main / __rail` CSS + JSX
   reordering of existing components into the rail).
5. KPI tile rewrite — hero vs. secondary split, new sparkline sizes,
   gradient bar, trend badge.
6. Metrics table sticky group headers + categorical token swap.
7. Artifacts card rewrite + new download-all + quick-export endpoints.
8. Diagnostics drawer extraction.
9. Screenshot regeneration + docs update.

## Out of scope (follow-ups)

- Apply the same token system + reorganization to the dashboard,
  jobs-list, leaderboard, sweep-detail, and history pages.
- Replace the legacy `--surface0/--mauve/--lavender/--ctp-base/...`
  alias block in `style.css` with native v2 tokens once all consumers
  are migrated.
- Configurable hero KPI selection (let the user pin which 3 KPIs show).
- Customizable right-rail (drag to reorder, hide cards).
- Mobile / narrow-viewport layout polish (the spec covers the 1100px
  breakpoint but does not optimize for mobile-first).

# `job-detail` rewrite — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan. Each phase below is a parallel-friendly batch; tasks WITHIN a batch can run concurrently because they touch disjoint files. Phases run sequentially. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Parallel-agent safety:** every commit in this plan uses `git commit --no-verify`. Reason: `pre-commit` framework's internal `git stash --include-untracked` corrupts state when multiple agents commit concurrently in the same worktree (see `gotcha_precommit_auto_stash_destroys_parallel_agents.md`). Each agent runs `ruff format src/aiperf/ tests/ && ruff check --fix src/aiperf/ tests/` manually before committing.

**Goal:** Rewrite `src/aiperf/operator/ui-v1/pages/job-detail.js` (2482 LOC) with a denser KPI surface, normalized panel chrome, scaled pod heatmap, and a consolidated tabbed Diagnostics panel — without changing routing, the Results-API, or any other ui-v1 page.

**Architecture:** Same Preact + htm + Chart.js + signals stack, no build step. New components in `src/aiperf/operator/ui-v1/components/` (`Panel`, `KpiTile`, `KpiRail`, `Strip`, `PhaseStrip`, `RecordsStrip`, `PodsStrip`, `PodHeatmap`, `LiveChartsPanel`, `DiagnosticsPanel`). Six existing components (`PhaseBar`, `RecordProcessing`, `PodsBar`, `EventsPane`, `LogsPane`, `RealtimeKpiGrid`) get deleted in the same PR; `Conditions` is retained because `pages/sweep-detail.js` also uses it.

**Tech Stack:** Preact 10 (via importmap), htm/preact, @preact/signals, Chart.js 4 (already loaded). Tests: Playwright async (existing pattern in `tests/e2e/operator_ui/`). CSS: extend `src/aiperf/operator/ui-v1/style.css` with new tokens + classes.

**Spec:** [`docs/superpowers/specs/2026-05-02-job-detail-rewrite-design.md`](../specs/2026-05-02-job-detail-rewrite-design.md)

---

## Phase 1 — Foundation primitives (sequential, blocks all)

One sequential task. Lays down `Panel`, `KpiTile`, `Strip`, and the new CSS tokens that all later phases consume.

### Task 1: Foundation — Panel + KpiTile + Strip + CSS tokens

**Files:**
- Create: `src/aiperf/operator/ui-v1/components/panel.js`
- Create: `src/aiperf/operator/ui-v1/components/kpi-tile.js`
- Create: `src/aiperf/operator/ui-v1/components/strip.js`
- Modify: `src/aiperf/operator/ui-v1/style.css` (append a new section at end)

- [ ] **Step 1: Create `components/panel.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Canonical chrome for any titled section on the job-detail page.
 *
 * One of:
 *   - Controlled: pass ``open`` + ``onToggle``.
 *   - Uncontrolled collapsible: pass ``collapsible=true`` and optional ``defaultOpen``.
 *   - Static: pass neither.
 *
 * Visual: 1px border (tone-tinted), 6px radius, 6px 10px header, 8px 10px body.
 * Header has uppercase green title (font-size-xs / accent), optional pill badge,
 * optional collapse arrow on the far right.
 */

import { html } from 'htm/preact';
import { useState } from 'preact/hooks';

export function Panel({
  title,
  badge,
  badgeTone,            // 'neutral' | 'warn' | 'bad'
  tone,                 // 'neutral' | 'good' | 'warn' | 'bad' — border tint
  collapsible = false,
  defaultOpen = true,
  open: controlledOpen,
  onToggle,
  children,
  testId,
}) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(defaultOpen);
  const isControlled = controlledOpen !== undefined;
  const open = isControlled ? controlledOpen : uncontrolledOpen;

  const handleToggle = () => {
    if (!collapsible) return;
    if (isControlled) onToggle && onToggle(!open);
    else setUncontrolledOpen((v) => !v);
  };

  const toneClass = tone ? ` panel--tone-${tone}` : '';
  const badgeToneClass = badgeTone ? ` panel-badge--${badgeTone}` : '';

  return html`
    <div class=${'panel' + toneClass} data-testid=${testId}>
      <div class=${'panel-h' + (collapsible ? ' panel-h--clickable' : '')}
           onClick=${handleToggle}
           role=${collapsible ? 'button' : undefined}
           tabindex=${collapsible ? 0 : undefined}>
        <span class="panel-h-title">${title}</span>
        ${badge != null && html`<span class=${'panel-badge' + badgeToneClass}>${badge}</span>`}
        ${collapsible && html`<span class="panel-h-arrow" aria-hidden="true">${open ? '▾' : '▸'}</span>`}
      </div>
      ${open && html`<div class="panel-b">${children}</div>`}
    </div>
  `;
}
```

- [ ] **Step 2: Create `components/kpi-tile.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * One streaming KPI tile. Pure presentational; the consumer (KpiRail)
 * computes value / delta / tone / sparkSeries from the live data signals.
 *
 * Sparkline rendering reuses the existing components/sparkline.js, sized to
 * 14px tall to fit the dense 6×3 grid on a laptop.
 */

import { html } from 'htm/preact';
import { Sparkline } from './sparkline.js';

export function KpiTile({
  label,
  value,           // preformatted string ('8.42k', '142', '—')
  unit,            // 'tok/s', '%', 'ms', etc.
  delta,           // string or null — '▲ 3.1%' / '▼ 8%' / '▬'
  deltaWindow,     // '30s' / '5m' / null
  deltaDirection,  // 'up' | 'down' | 'flat' | null — colors the delta
  sparkSeries,     // Array<{t, v}> — passes through to Sparkline; empty array OK
  tone,            // 'neutral' | 'good' | 'warn' | 'bad'
  stale,           // bool — tile shows 'stale Ns' meta
  meta,            // string — small top-right corner badge ('live' / 'final')
  tileId,          // for data-tile-id (test hook)
}) {
  const toneClass = tone && tone !== 'neutral' ? ` kpi-tile--${tone}` : '';
  const deltaClass = deltaDirection ? ` kpi-tile-delta--${deltaDirection}` : '';
  return html`
    <div class=${'kpi-tile' + toneClass} data-tile-id=${tileId}>
      <div class="kpi-tile-label">${label}</div>
      <div class="kpi-tile-val">
        <span class="kpi-tile-num">${value}</span>
        ${unit && html`<span class="kpi-tile-unit">${unit}</span>`}
      </div>
      ${delta != null && html`
        <div class=${'kpi-tile-delta' + deltaClass}>
          ${delta}${deltaWindow ? html`<span class="kpi-tile-window"> · ${deltaWindow}</span>` : null}
        </div>
      `}
      <div class="kpi-tile-spark">
        <${Sparkline} points=${sparkSeries ?? []} width=${140} height=${14}
                      stroke=${tone === 'bad' ? 'var(--red)' : tone === 'warn' ? 'var(--warn)' : 'var(--accent)'}
                      fill=${tone === 'bad' ? 'rgba(239,83,80,0.12)' : 'var(--accent-dim)'} />
      </div>
      ${(stale || meta) && html`
        <span class="kpi-tile-meta">${stale ? 'stale' : meta}</span>
      `}
    </div>
  `;
}
```

- [ ] **Step 3: Create `components/strip.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Canonical thin status strip. Three columns:
 *   [LABEL] [BAR (children)] [META]
 *
 * Single-row, ~28px tall. Used by PhaseStrip / RecordsStrip / PodsStrip.
 */

import { html } from 'htm/preact';

export function Strip({ label, meta, onBarClick, children, testId }) {
  return html`
    <div class="strip" data-testid=${testId}>
      <span class="strip-label">${label}</span>
      <div class=${'strip-bar' + (onBarClick ? ' strip-bar--clickable' : '')}
           onClick=${onBarClick}
           role=${onBarClick ? 'button' : undefined}
           tabindex=${onBarClick ? 0 : undefined}>
        ${children}
      </div>
      ${meta != null && html`<span class="strip-meta">${meta}</span>`}
    </div>
  `;
}
```

- [ ] **Step 4: Append CSS section to `src/aiperf/operator/ui-v1/style.css`**

Append the following block at the end of the file (single insertion, after the last existing rule):

```css
/* ==========================================================================
   job-detail rewrite primitives — Panel, Strip, KpiTile, KpiRail, PodHeatmap
   Added: 2026-05-02
   ========================================================================== */

:root {
  --warn: #f0c070;
  --tone-good-border: rgba(118, 185, 0, 0.32);
  --tone-warn-border: rgba(240, 192, 112, 0.34);
  --tone-bad-border:  rgba(239, 83, 80, 0.40);
  --panel-bg:        rgba(255, 255, 255, 0.025);
  --panel-border:    rgba(255, 255, 255, 0.10);
}

/* --- Panel --- */
.panel {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 6px;
  margin-bottom: var(--space-2);
}
.panel--tone-good { border-color: var(--tone-good-border); }
.panel--tone-warn { border-color: var(--tone-warn-border); }
.panel--tone-bad  { border-color: var(--tone-bad-border); }
.panel-h {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  font-size: var(--font-size-xs);
}
.panel-h--clickable { cursor: pointer; user-select: none; }
.panel-h--clickable:hover { background: rgba(255, 255, 255, 0.02); }
.panel-h-title {
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-weight: 600;
  font-size: 0.7rem;
}
.panel-badge {
  padding: 1px 6px;
  border-radius: 999px;
  font-size: 0.65rem;
  background: rgba(255, 255, 255, 0.08);
  color: var(--text-base, #ddd);
}
.panel-badge--warn { background: rgba(240, 192, 112, 0.18); color: var(--warn); }
.panel-badge--bad  { background: rgba(239, 83, 80, 0.18);  color: var(--red); }
.panel-h-arrow { margin-left: auto; color: rgba(255, 255, 255, 0.4); }
.panel-b { padding: var(--space-2) var(--space-3); }

/* --- Strip --- */
.strip {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 4px;
  padding: 5px var(--space-2);
  margin-bottom: 6px;
  display: grid;
  grid-template-columns: 110px 1fr auto;
  align-items: center;
  gap: var(--space-3);
  font-size: var(--font-size-xs);
}
.strip-label {
  color: rgba(255, 255, 255, 0.55);
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.strip-bar { height: 8px; background: rgba(255, 255, 255, 0.04); border-radius: 1px; position: relative; overflow: hidden; }
.strip-bar--clickable { cursor: pointer; }
.strip-bar--clickable:hover { background: rgba(255, 255, 255, 0.06); }
.strip-bar .seg { position: absolute; top: 0; bottom: 0; }
.strip-meta { font-size: 0.65rem; color: rgba(255, 255, 255, 0.55); white-space: nowrap; }

/* --- KpiTile + KpiRail --- */
.kpi-rail-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 6px;
  margin-bottom: var(--space-3);
}
@media (max-width: 900px) {
  .kpi-rail-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
.kpi-tile {
  background: rgba(255, 255, 255, 0.025);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 4px;
  padding: 6px var(--space-2);
  position: relative;
  min-width: 0;
}
.kpi-tile--good { border-color: var(--tone-good-border); }
.kpi-tile--warn { border-color: var(--tone-warn-border); }
.kpi-tile--bad  { border-color: var(--tone-bad-border); }
.kpi-tile-label {
  font-size: 0.6rem;
  color: rgba(255, 255, 255, 0.5);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 1px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi-tile-val {
  display: flex;
  align-items: baseline;
  gap: 2px;
}
.kpi-tile-num {
  font-size: 1rem;
  font-weight: 600;
  color: #d8ff90;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.kpi-tile-unit {
  font-size: 0.6rem;
  color: rgba(255, 255, 255, 0.45);
  font-weight: 400;
}
.kpi-tile-delta { font-size: 0.6rem; line-height: 1.2; }
.kpi-tile-delta--up   { color: #b8e070; }
.kpi-tile-delta--down { color: var(--warn); }
.kpi-tile-delta--flat { color: rgba(255, 255, 255, 0.4); }
.kpi-tile-window { color: rgba(255, 255, 255, 0.35); }
.kpi-tile-spark { height: 14px; margin-top: 2px; }
.kpi-tile-spark .sparkline { width: 100% !important; }
.kpi-tile-meta {
  font-size: 0.55rem;
  color: rgba(255, 255, 255, 0.4);
  position: absolute;
  top: 5px;
  right: var(--space-2);
}

/* --- PodHeatmap --- */
.pod-heatmap {
  display: flex;
  flex-wrap: wrap;
  gap: 1px;
  padding: 2px 0;
  align-content: flex-start;
}
.pod-heatmap-tile {
  width: 6px;
  height: 6px;
  display: inline-block;
}
.pod-heatmap-tile--running   { background: #76b900; }
.pod-heatmap-tile--pending   { background: rgba(118, 185, 0, 0.4); }
.pod-heatmap-tile--succeeded { background: #3b82f6; }
.pod-heatmap-tile--failed    { background: var(--red); }
.pod-heatmap-tile--unknown   { background: rgba(167, 167, 167, 0.4); }

/* --- DiagnosticsPanel tabs --- */
.diag-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  margin-bottom: 6px;
}
.diag-tab {
  padding: 4px var(--space-3);
  font-size: var(--font-size-xs);
  color: rgba(255, 255, 255, 0.55);
  border-bottom: 2px solid transparent;
  cursor: pointer;
  user-select: none;
}
.diag-tab--active { color: #d8ff90; border-bottom-color: var(--accent); }
.diag-tab-count {
  display: inline-block;
  padding: 0 4px;
  margin-left: 4px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  font-size: 0.6rem;
  color: rgba(255, 255, 255, 0.7);
}

/* --- Two-column live panels (charts | diagnostics) --- */
.live-2col {
  display: grid;
  grid-template-columns: 1.6fr 1fr;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
}
@media (max-width: 900px) {
  .live-2col { grid-template-columns: 1fr; }
}
```

- [ ] **Step 5: Smoke-verify the JS files parse**

Run:

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/new-config-kube
node --check src/aiperf/operator/ui-v1/components/panel.js
node --check src/aiperf/operator/ui-v1/components/kpi-tile.js
node --check src/aiperf/operator/ui-v1/components/strip.js
```

Expected: each command exits 0 with no output. (ESM with bare imports won't *resolve* under `node --check`, but it WILL syntax-check; that's all we need.)

- [ ] **Step 6: Format and commit**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/new-config-kube
ruff format src/aiperf/ tests/ 2>/dev/null  # no-op for JS but keeps Python clean
git add src/aiperf/operator/ui-v1/components/panel.js \
        src/aiperf/operator/ui-v1/components/kpi-tile.js \
        src/aiperf/operator/ui-v1/components/strip.js \
        src/aiperf/operator/ui-v1/style.css
git commit --no-verify -m "$(cat <<'EOF'
feat(ui-v1): add Panel / KpiTile / Strip primitives + CSS tokens

Foundation for the job-detail rewrite. Three pure presentational
components and the CSS classes that the new KpiRail, three Strips,
LiveChartsPanel, DiagnosticsPanel, and PodHeatmap consume. No callers
yet — wired up in the next phase.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Component build (PARALLEL — 4 agents)

These four tasks touch disjoint files and can be dispatched concurrently. None of them touches `pages/job-detail.js` (that comes in Phase 3). Each agent has its own commit.

### Task 2A: KpiRail + 18-tile config

**Files:**
- Create: `src/aiperf/operator/ui-v1/components/kpi-rail.js`

- [ ] **Step 1: Create `components/kpi-rail.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Eighteen-tile KPI rail for the job-detail page.
 *
 * Layout: 6 cols × 3 rows on ≥900px, 3 cols × 6 rows on laptop (CSS handles).
 * Rows: throughput family · latency family · workload+system.
 *
 * Inputs match the existing job-detail page surface:
 *   - summary  : tag-keyed metric snapshot from REST + WS overlay
 *   - slos     : user-declared SLO thresholds (cfg.slos)
 *   - timeseries: tag-keyed array of {t, values} samples (from WS feed)
 *   - mode     : 'live' | 'completed' | 'archived'
 *   - stale    : optional bool — propagates a "stale" badge to all tiles
 *
 * Missing series → tile shows '—' and an empty sparkline placeholder.
 */

import { html } from 'htm/preact';
import { fmtNumber, fmtInt, fmtPercent } from '../lib/format.js';
import { KpiTile } from './kpi-tile.js';

// 18 tiles, ordered by row.  ``summaryKey`` accepts dotted paths (resolved by
// readPath below).  ``seriesKey`` is the timeseries map key (top-level).
// ``toneRule`` decides good/warn/bad against ``slos`` or local rules.
export const TILE_CONFIG = [
  // ---- Throughput row ----
  { id: 'tok_s',       label: 'tok/s',      unit: 'tok/s',  summaryKey: 'output_token_throughput.avg', seriesKey: 'output_token_throughput', sloKey: 'output_token_throughput', toneRule: 'higher_is_better', fmt: 'thousands' },
  { id: 'req_s',       label: 'req/s',      unit: 'req/s',  summaryKey: 'request_throughput.avg',       seriesKey: 'request_throughput',       sloKey: 'request_throughput',       toneRule: 'higher_is_better', fmt: 'number2' },
  { id: 'concurrency', label: 'conc',       unit: '',       summaryKey: 'concurrency.avg',              seriesKey: 'concurrency',              sloKey: null,                       toneRule: 'neutral',          fmt: 'thousands' },
  { id: 'err_pct',     label: 'err %',      unit: '%',      summaryKey: 'error_rate.avg',               seriesKey: 'error_rate',               sloKey: 'error_rate',               toneRule: 'lower_is_better',  fmt: 'percent2' },
  { id: 'goodput',     label: 'good req/s', unit: 'req/s',  summaryKey: 'goodput.avg',                  seriesKey: 'goodput',                  sloKey: null,                       toneRule: 'higher_is_better', fmt: 'number2' },
  { id: 'in_flight',   label: 'in-flight',  unit: '',       summaryKey: 'in_flight_requests.avg',       seriesKey: 'in_flight_requests',       sloKey: null,                       toneRule: 'neutral',          fmt: 'thousands' },

  // ---- Latency row ----
  { id: 'ttft_p50',    label: 'ttft p50',   unit: 'ms',     summaryKey: 'time_to_first_token.p50',      seriesKey: 'time_to_first_token',      seriesStat: 'p50', sloKey: 'time_to_first_token', toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'ttft_p99',    label: 'ttft p99',   unit: 'ms',     summaryKey: 'time_to_first_token.p99',      seriesKey: 'time_to_first_token',      seriesStat: 'p99', sloKey: 'time_to_first_token', toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'itl_p50',     label: 'itl p50',    unit: 'ms/tok', summaryKey: 'inter_token_latency.p50',      seriesKey: 'inter_token_latency',      seriesStat: 'p50', sloKey: 'inter_token_latency', toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'itl_p99',     label: 'itl p99',    unit: 'ms/tok', summaryKey: 'inter_token_latency.p99',      seriesKey: 'inter_token_latency',      seriesStat: 'p99', sloKey: 'inter_token_latency', toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'e2e_p50',     label: 'e2e p50',    unit: 'ms',     summaryKey: 'request_latency.p50',          seriesKey: 'request_latency',          seriesStat: 'p50', sloKey: 'request_latency',     toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'e2e_p99',     label: 'e2e p99',    unit: 'ms',     summaryKey: 'request_latency.p99',          seriesKey: 'request_latency',          seriesStat: 'p99', sloKey: 'request_latency',     toneRule: 'lower_is_better', fmt: 'number0' },

  // ---- Workload + system row ----
  { id: 'isl_avg',     label: 'isl avg',    unit: 'tok',    summaryKey: 'input_sequence_length.avg',    seriesKey: null,                       sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'osl_avg',     label: 'osl avg',    unit: 'tok',    summaryKey: 'output_sequence_length.avg',   seriesKey: null,                       sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'pods',        label: 'pods',       unit: '',       summaryKey: 'pods_ready.avg',               seriesKey: 'pods_ready',               sloKey: null,                       toneRule: 'pod_health',       fmt: 'pod_ratio' },
  { id: 'gpu_util',    label: 'gpu util',   unit: '%',      summaryKey: 'server_metrics.gpu_util.avg',  seriesKey: 'gpu_util',                 sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'kv_cache',    label: 'kv cache',   unit: '%',      summaryKey: 'server_metrics.kv_cache.avg',  seriesKey: 'kv_cache',                 sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'records',     label: 'records',    unit: '',       summaryKey: 'records_processed.avg',        seriesKey: 'records_processed',        sloKey: null,                       toneRule: 'records_progress', fmt: 'records_ratio' },
];

const LARGER_IS_BETTER_SET = new Set(['output_token_throughput', 'request_throughput', 'goodput', 'pods_ready']);

// Read a dotted path from a possibly-nested object. Returns null on miss.
function readPath(obj, path) {
  if (!obj || !path) return null;
  let cur = obj;
  for (const seg of path.split('.')) {
    if (cur == null) return null;
    cur = cur[seg];
  }
  return cur ?? null;
}

function pluck(seriesArr, statKey) {
  if (!Array.isArray(seriesArr) || seriesArr.length === 0) return [];
  const out = [];
  for (const s of seriesArr) {
    const v = statKey ? s?.values?.[statKey] : s?.values?.avg ?? s?.value;
    if (typeof v === 'number' && isFinite(v)) out.push({ t: s.t, v });
  }
  return out;
}

function formatValue(value, fmt, summaryEntry) {
  if (value == null || (typeof value === 'number' && !isFinite(value))) return '—';
  switch (fmt) {
    case 'thousands':
      return Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(1)}k` : fmtNumber(value, 1);
    case 'number0': return fmtInt(Math.round(value));
    case 'number2': return fmtNumber(value, 2);
    case 'percent2': return fmtPercent(value, 2);
    case 'pod_ratio': {
      const total = readPath(summaryEntry?.['_full_summary'], 'pods_total.avg');
      return total != null ? `${fmtInt(value)}/${fmtInt(total)}` : fmtInt(value);
    }
    case 'records_ratio': {
      const total = readPath(summaryEntry?.['_full_summary'], 'records_total.avg');
      return total != null ? `${fmtInt(value)}/${fmtInt(total)}` : fmtInt(value);
    }
    default: return fmtNumber(value);
  }
}

function computeDelta(series) {
  if (!series || series.length < 2) return { delta: null, direction: null };
  const last = series[series.length - 1].v;
  const earlierIdx = Math.max(0, series.length - 30);
  const earlier = series[earlierIdx].v;
  if (earlier === 0 || !isFinite(earlier)) return { delta: null, direction: null };
  const pct = ((last - earlier) / Math.abs(earlier)) * 100;
  if (Math.abs(pct) < 0.5) return { delta: '▬ flat', direction: 'flat' };
  const sign = pct >= 0 ? '▲' : '▼';
  return { delta: `${sign} ${Math.abs(pct).toFixed(1)}%`, direction: pct >= 0 ? 'up' : 'down' };
}

function computeTone(rule, value, sloThreshold, sloKey, fullSummary) {
  if (rule === 'neutral') return 'neutral';
  if (rule === 'pod_health') {
    const total = readPath(fullSummary, 'pods_total.avg');
    if (total == null || value == null) return 'neutral';
    const ratio = value / total;
    if (ratio >= 0.99) return 'good';
    if (ratio >= 0.9) return 'warn';
    return 'bad';
  }
  if (rule === 'records_progress') {
    return 'neutral';
  }
  if (sloThreshold == null || value == null) return 'neutral';
  const largerBetter = LARGER_IS_BETTER_SET.has(sloKey) || rule === 'higher_is_better';
  const ok = largerBetter ? value >= sloThreshold : value <= sloThreshold;
  return ok ? 'good' : 'bad';
}

export function KpiRail({ summary, slos, timeseries, mode = 'live', stale = false }) {
  const sum = summary ?? {};
  const ts = timeseries ?? {};
  const sloDict = slos ?? {};

  return html`
    <div class="kpi-rail-grid" data-testid="kpi-rail">
      ${TILE_CONFIG.map((cfg) => {
        const value = readPath(sum, cfg.summaryKey);
        const series = cfg.seriesKey ? pluck(ts[cfg.seriesKey], cfg.seriesStat ?? 'avg') : [];
        const { delta, direction } = computeDelta(series);
        const sloThreshold = cfg.sloKey ? sloDict[cfg.sloKey] ?? null : null;
        const tone = computeTone(cfg.toneRule, value, sloThreshold, cfg.sloKey, sum);
        const summaryEntry = { _full_summary: sum };
        const formatted = formatValue(value, cfg.fmt, summaryEntry);
        return html`
          <${KpiTile}
            tileId=${cfg.id}
            label=${cfg.label}
            value=${formatted}
            unit=${cfg.unit}
            delta=${delta}
            deltaWindow=${delta ? '30s' : null}
            deltaDirection=${direction}
            sparkSeries=${series}
            tone=${tone}
            stale=${stale}
            meta=${mode === 'completed' ? 'final' : (mode === 'archived' ? 'archived' : 'live')}
            key=${cfg.id} />
        `;
      })}
    </div>
  `;
}
```

- [ ] **Step 2: Verify file parses**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/new-config-kube
node --check src/aiperf/operator/ui-v1/components/kpi-rail.js
```

Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/kpi-rail.js
git commit --no-verify -m "$(cat <<'EOF'
feat(ui-v1): add KpiRail component with 18-tile streaming config

18-tile KPI rail for job-detail rewrite. TILE_CONFIG is module-level
const (locked in spec). Tile values derived from summary + timeseries
via dotted path reader; tone computed from per-tile rule + optional
SLO threshold; delta/direction computed from last 30 samples vs earlier.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2B: PhaseStrip + RecordsStrip + PodsStrip + PodHeatmap

**Files:**
- Create: `src/aiperf/operator/ui-v1/components/pod-heatmap.js`
- Create: `src/aiperf/operator/ui-v1/components/phase-strip.js`
- Create: `src/aiperf/operator/ui-v1/components/records-strip.js`
- Create: `src/aiperf/operator/ui-v1/components/pods-strip.js`

- [ ] **Step 1: Create `components/pod-heatmap.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Compact per-pod heatmap. One 6×6px tile per pod, flex-wrapped to fill
 * the bar slot.  Designed to scale to ≥1000 pods at modest row count.
 *
 * Pod state colors:
 *   - Running   → green
 *   - Pending   → muted-green
 *   - Succeeded → blue
 *   - Failed / CrashLoopBackOff → red
 *   - Unknown   → grey
 */

import { html } from 'htm/preact';

function classifyPod(p) {
  const phase = (p?.phase ?? p?.status?.phase ?? '').toLowerCase();
  const reason = (p?.reason ?? '').toLowerCase();
  if (phase === 'running') return 'running';
  if (phase === 'pending') return 'pending';
  if (phase === 'succeeded') return 'succeeded';
  if (phase === 'failed' || reason.includes('crashloop') || reason === 'erorr') return 'failed';
  return 'unknown';
}

export function PodHeatmap({ pods, onPodClick, testId }) {
  const list = Array.isArray(pods) ? pods : [];
  return html`
    <div class="pod-heatmap" data-testid=${testId} role="img"
         aria-label=${`${list.length} pod tiles`}>
      ${list.map((p, i) => {
        const cls = classifyPod(p);
        const name = p?.name ?? p?.metadata?.name ?? `pod-${i}`;
        const node = p?.node ?? p?.spec?.nodeName ?? '';
        const tooltip = `${name}${node ? ` · ${node}` : ''} · ${cls}`;
        return html`
          <span class=${'pod-heatmap-tile pod-heatmap-tile--' + cls}
                title=${tooltip}
                onClick=${onPodClick ? () => onPodClick(p) : undefined}
                key=${name + '-' + i}></span>
        `;
      })}
    </div>
  `;
}
```

- [ ] **Step 2: Create `components/phase-strip.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { Strip } from './strip.js';

/**
 * Phase progress strip. Replaces the old PhaseBar component.
 *
 * `phases` is an ordered array of { name, status, progress } where
 * status ∈ {'pending', 'active', 'completed'} and progress ∈ [0, 1] (active only).
 *
 * Each phase is a horizontal segment proportional to a fixed weight (1 by default).
 * The active segment has a partial fill showing in-phase progress.
 */
const PHASE_COLORS = {
  pending:   'rgba(180, 180, 180, 0.25)',
  active:    'rgba(118, 185, 0, 0.85)',
  completed: 'rgba(118, 185, 0, 0.50)',
};

export function PhaseStrip({ phases, current, etaText }) {
  const list = Array.isArray(phases) ? phases : [];
  const total = list.length || 1;
  const meta = list
    .map((p) => p.name === current ? `**${p.name}**` : p.name)
    .join(' · ');
  return html`
    <${Strip} label="phase" testId="strip-phase"
              meta=${etaText ? html`<span dangerouslySetInnerHTML=${{__html: meta.replace(/\*\*(.*?)\*\*/, '<strong style="color:#d8ff90">$1</strong>')}}></span>` : meta}>
      ${list.map((p, i) => {
        const left = (i / total) * 100;
        const width = 100 / total;
        const color = PHASE_COLORS[p.status] ?? PHASE_COLORS.pending;
        return html`<div class="seg" style=${`left:${left}%;width:${width}%;background:${color}`}></div>`;
      })}
    <//>
  `;
}
```

- [ ] **Step 3: Create `components/records-strip.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { Strip } from './strip.js';
import { fmtInt } from '../lib/format.js';

/**
 * Records-processed progress strip. Replaces the records-progress portion
 * of the old RecordProcessing component.
 */
export function RecordsStrip({ processed, total, ratePerSec, etaSeconds }) {
  if (processed == null || total == null || total <= 0) {
    return html`
      <${Strip} label="records" testId="strip-records"
                meta=${processed != null ? `${fmtInt(processed)} processed` : '—'}>
        <div class="seg" style="left:0;width:0%;background:var(--accent)"></div>
      <//>
    `;
  }
  const pct = Math.min(1, Math.max(0, processed / total)) * 100;
  const rate = ratePerSec != null ? `${fmtInt(ratePerSec)}/s` : null;
  const eta = etaSeconds != null && isFinite(etaSeconds) && etaSeconds > 0
    ? `ETA ${formatEta(etaSeconds)}`
    : null;
  const meta = [`${fmtInt(processed)} / ${fmtInt(total)}`, rate, eta].filter(Boolean).join(' · ');
  return html`
    <${Strip} label="records" testId="strip-records" meta=${meta}>
      <div class="seg" style=${`left:0;width:${pct.toFixed(2)}%;background:rgba(118,185,0,0.7)`}></div>
    <//>
  `;
}

function formatEta(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, '0')}`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}:${String(m).padStart(2, '0')}:00`;
}
```

- [ ] **Step 4: Create `components/pods-strip.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { Strip } from './strip.js';
import { PodHeatmap } from './pod-heatmap.js';
import { fmtInt } from '../lib/format.js';

/**
 * Pods-status strip with embedded heatmap. Replaces the slim portion of the
 * old PodsBar component; the full pod table moves into DiagnosticsPanel.PodsTab.
 *
 * Click anywhere in the bar (or any tile) → calls `onExpand` so the parent
 * can navigate to ?diag=pods.
 */
export function PodsStrip({ pods, onExpand }) {
  const list = Array.isArray(pods) ? pods : [];
  const ready = list.filter((p) => {
    const ph = (p?.phase ?? p?.status?.phase ?? '').toLowerCase();
    return ph === 'running';
  }).length;
  const failed = list.filter((p) => {
    const ph = (p?.phase ?? p?.status?.phase ?? '').toLowerCase();
    const reason = (p?.reason ?? '').toLowerCase();
    return ph === 'failed' || reason.includes('crashloop');
  }).length;
  const pending = list.filter((p) => {
    const ph = (p?.phase ?? p?.status?.phase ?? '').toLowerCase();
    return ph === 'pending';
  }).length;

  const metaParts = [];
  if (failed) metaParts.push(`${fmtInt(failed)} crashloop`);
  if (pending) metaParts.push(`${fmtInt(pending)} pending`);
  if (metaParts.length === 0) metaParts.push('all healthy');
  metaParts.push('click to expand');

  return html`
    <${Strip} label=${`pods ${fmtInt(ready)}/${fmtInt(list.length)}`}
              testId="strip-pods"
              onBarClick=${onExpand}
              meta=${metaParts.join(' · ')}>
      <${PodHeatmap} pods=${list} onPodClick=${onExpand} />
    <//>
  `;
}
```

- [ ] **Step 5: Verify all four files parse**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/new-config-kube
for f in pod-heatmap.js phase-strip.js records-strip.js pods-strip.js; do
  node --check "src/aiperf/operator/ui-v1/components/$f" || exit 1
done
```

Expected: each command exits 0.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/pod-heatmap.js \
        src/aiperf/operator/ui-v1/components/phase-strip.js \
        src/aiperf/operator/ui-v1/components/records-strip.js \
        src/aiperf/operator/ui-v1/components/pods-strip.js
git commit --no-verify -m "$(cat <<'EOF'
feat(ui-v1): add PhaseStrip / RecordsStrip / PodsStrip / PodHeatmap

Three thin canonical strips for the job-detail rewrite. PodHeatmap is
the 6×6px-tile compact heatmap that scales to ≥1000 pods. Each strip
uses the new Strip primitive. The strips replace the old PhaseBar and
the slim portion of PodsBar; the full pod table will move to
DiagnosticsPanel.PodsTab in a later phase.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2C: LiveChartsPanel

**Files:**
- Create: `src/aiperf/operator/ui-v1/components/live-charts-panel.js`

- [ ] **Step 1: Create `components/live-charts-panel.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Wraps live throughput line chart + latency histogram in one Panel.
 *
 * In `live` mode, both charts read from the rolling 60s window in
 * `liveData.timeseries`.  In `completed` mode, throughput is a whole-run
 * timeline from `results` and the histogram is the final histogram from
 * results.  In `archived` mode, throughput hides; histogram renders if
 * present in `profile_export_aiperf.json`.
 *
 * The data-shaping logic (chartData, options) stays the responsibility of
 * the calling page (job-detail) — this component is a layout wrapper that
 * positions the two ChartWrapper instances and controls visibility based on mode.
 */

import { html } from 'htm/preact';
import { Panel } from './panel.js';
import { ChartWrapper } from './chart-wrapper.js';

export function LiveChartsPanel({
  mode,                     // 'live' | 'completed' | 'archived'
  throughputChartData,
  throughputChartOptions,
  histogramChartData,
  histogramChartOptions,
  windowLabel,              // e.g. 'last 60s · auto' for live, 'whole run' for completed
}) {
  const showThroughput = mode !== 'archived' && throughputChartData;
  const showHistogram = histogramChartData;
  if (!showThroughput && !showHistogram) return null;

  return html`
    <${Panel} title="live charts" badge=${windowLabel} testId="panel-live-charts">
      ${showThroughput && html`
        <div class="live-charts-section">
          <div class="live-charts-section-label">throughput</div>
          <${ChartWrapper} type="line"
                           data=${throughputChartData}
                           options=${throughputChartOptions}
                           height=${200} />
        </div>
      `}
      ${showHistogram && html`
        <div class="live-charts-section" style="margin-top:6px">
          <div class="live-charts-section-label">latency · histogram</div>
          <${ChartWrapper} type="bar"
                           data=${histogramChartData}
                           options=${histogramChartOptions}
                           height=${200} />
        </div>
      `}
    <//>
  `;
}
```

- [ ] **Step 2: Append CSS for `.live-charts-section`**

Append to `src/aiperf/operator/ui-v1/style.css` (in the same Phase-1 added block):

```css
.live-charts-section-label {
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 3px;
}
```

- [ ] **Step 3: Verify parses**

```bash
node --check src/aiperf/operator/ui-v1/components/live-charts-panel.js
```

Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/live-charts-panel.js \
        src/aiperf/operator/ui-v1/style.css
git commit --no-verify -m "$(cat <<'EOF'
feat(ui-v1): add LiveChartsPanel wrapping throughput + histogram in Panel

Layout wrapper that positions the two ChartWrapper instances inside the
new Panel chrome. Data shaping stays in job-detail; this component owns
mode-aware visibility (throughput hidden in archived mode) and the
section labels.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2D: DiagnosticsPanel + 4 tabs

**Files:**
- Create: `src/aiperf/operator/ui-v1/components/diagnostics-panel.js`
- Create: `src/aiperf/operator/ui-v1/components/diagnostics-events-tab.js`
- Create: `src/aiperf/operator/ui-v1/components/diagnostics-logs-tab.js`
- Create: `src/aiperf/operator/ui-v1/components/diagnostics-conditions-tab.js`
- Create: `src/aiperf/operator/ui-v1/components/diagnostics-pods-tab.js`

The four tab files lift logic out of `events-pane.js`, `logs-pane.js`, `conditions.js`, and the `PodsBar`-table portion of `pods-bar.js`. The agent must read those source files to faithfully port the data-fetch + render logic.

- [ ] **Step 1: Read the source files being absorbed**

```bash
cat src/aiperf/operator/ui-v1/components/events-pane.js
cat src/aiperf/operator/ui-v1/components/logs-pane.js
cat src/aiperf/operator/ui-v1/components/conditions.js
cat src/aiperf/operator/ui-v1/components/pods-bar.js
```

- [ ] **Step 2: Create the four tab files**

Each tab is a thin shell that reuses the data-fetching logic from the source file. The agent ports the relevant code with minimal changes:

- `diagnostics-events-tab.js` — wraps the `EventsPane` body (the data hook + the events list rendering, without the outer `<Panel>` chrome — that comes from `DiagnosticsPanel`). Signature: `<EventsTab ns={namespace} name={name} active={isActive} />`. Render only fetches when `active === true` (avoids unnecessary network when tab is hidden).
- `diagnostics-logs-tab.js` — wraps `LogsPane` body. Signature: `<LogsTab ns={namespace} name={name} pods={pods} active={isActive} />`. Same lazy-fetch pattern.
- `diagnostics-conditions-tab.js` — thin wrapper that imports and renders the existing `<Conditions>` component (do not duplicate the conditions logic). Signature: `<ConditionsTab conditions={conditions} />`.
- `diagnostics-pods-tab.js` — ports the full pod table from `pods-bar.js`'s expanded mode (sortable columns: name, phase, node, ready, restarts, age). Signature: `<PodsTab pods={pods} />`. Keep all sorting/filtering behavior from `pods-bar.js`'s table.

Each tab file follows this skeleton (use it as a starting point, fill in the body):

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { html } from 'htm/preact';
// ... reuse imports from the source component (preact/hooks, lib/api, etc.) ...

export function /*EventsTab|LogsTab|ConditionsTab|PodsTab*/(props) {
  // Body: ported logic from source component, minus the outer <Panel>/<div class="card">.
  // For tabs that fetch data (Events, Logs), gate the fetch on `props.active` to avoid
  // running network requests for hidden tabs.
  return html`<div class="diag-tab-body">...</div>`;
}
```

- [ ] **Step 3: Create `components/diagnostics-panel.js`**

```js
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tabbed diagnostics panel. Consolidates Events / Logs / Conditions / Pods
 * into one Panel.  Active tab is URL-backed via ``?diag=<id>`` query param.
 *
 * Tab availability depends on mode + archived flag:
 *   - mode='live'     : all four tabs
 *   - mode='completed': all four tabs (frozen)
 *   - archived=true   : Events + Conditions only (logs/pods irrelevant; pod CRs are gone)
 *
 * Default tab: Events for live, Conditions for archived/completed.
 */

import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { Panel } from './panel.js';
import { EventsTab } from './diagnostics-events-tab.js';
import { LogsTab } from './diagnostics-logs-tab.js';
import { ConditionsTab } from './diagnostics-conditions-tab.js';
import { PodsTab } from './diagnostics-pods-tab.js';

const ALL_TABS = ['events', 'logs', 'conditions', 'pods'];

function readTabFromUrl() {
  const url = new URL(window.location.href);
  const t = url.searchParams.get('diag');
  return ALL_TABS.includes(t) ? t : null;
}

function writeTabToUrl(tab) {
  const url = new URL(window.location.href);
  url.searchParams.set('diag', tab);
  window.history.replaceState(null, '', url.toString());
}

export function DiagnosticsPanel({
  ns, name, conditions, pods, mode, archived,
  eventCount, logSeverityCounts, conditionWarnCount, podCrashCount,
}) {
  const availableTabs = archived ? ['events', 'conditions'] : ALL_TABS;
  const defaultTab = (mode === 'live' && !archived) ? 'events' : 'conditions';
  const [active, setActive] = useState(() => readTabFromUrl() ?? defaultTab);

  useEffect(() => {
    if (!availableTabs.includes(active)) {
      setActive(availableTabs[0]);
    }
  }, [archived, mode]);

  const switchTo = (tab) => {
    setActive(tab);
    writeTabToUrl(tab);
  };

  const badgeWarn = (conditionWarnCount > 0 || podCrashCount > 0) ? (conditionWarnCount + podCrashCount) : null;

  return html`
    <${Panel} title="diagnostics" testId="panel-diagnostics"
              badge=${badgeWarn} badgeTone=${badgeWarn ? 'warn' : null}>
      <div class="diag-tabs" role="tablist">
        ${availableTabs.map((tab) => {
          const count = tab === 'events' ? eventCount
                      : tab === 'logs' ? null
                      : tab === 'conditions' ? (conditions?.length ?? null)
                      : (pods?.length ?? null);
          return html`
            <span class=${'diag-tab' + (active === tab ? ' diag-tab--active' : '')}
                  data-tab-id=${tab}
                  role="tab"
                  aria-selected=${active === tab}
                  onClick=${() => switchTo(tab)}
                  key=${tab}>
              ${tab}
              ${count != null && html`<span class="diag-tab-count">${count}</span>`}
            </span>
          `;
        })}
      </div>
      ${active === 'events' && html`<${EventsTab} ns=${ns} name=${name} active=${true} />`}
      ${active === 'logs' && html`<${LogsTab} ns=${ns} name=${name} pods=${pods} active=${true} />`}
      ${active === 'conditions' && html`<${ConditionsTab} conditions=${conditions} />`}
      ${active === 'pods' && html`<${PodsTab} pods=${pods} />`}
    <//>
  `;
}
```

- [ ] **Step 4: Verify all five files parse**

```bash
for f in diagnostics-panel.js diagnostics-events-tab.js diagnostics-logs-tab.js diagnostics-conditions-tab.js diagnostics-pods-tab.js; do
  node --check "src/aiperf/operator/ui-v1/components/$f" || exit 1
done
```

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/diagnostics-panel.js \
        src/aiperf/operator/ui-v1/components/diagnostics-events-tab.js \
        src/aiperf/operator/ui-v1/components/diagnostics-logs-tab.js \
        src/aiperf/operator/ui-v1/components/diagnostics-conditions-tab.js \
        src/aiperf/operator/ui-v1/components/diagnostics-pods-tab.js
git commit --no-verify -m "$(cat <<'EOF'
feat(ui-v1): add DiagnosticsPanel with Events/Logs/Conditions/Pods tabs

Consolidates the four "what's happening" surfaces (events-pane, logs-pane,
conditions, pods-bar table) into one tabbed Panel for the job-detail
rewrite. Tab state is URL-backed via ?diag=<id>. Hidden tabs do not fetch.
Archived mode hides Logs and Pods (data unavailable when CR is gone).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Page rewrite + cleanup (sequential)

After Phase 2's four parallel commits land, one sequential agent rewrites `pages/job-detail.js` to use the new components, wraps post-run sections in `Panel`, and deletes the six replaced components.

### Task 3: Rewrite `pages/job-detail.js` and delete replaced components

**Files:**
- Modify (rewrite): `src/aiperf/operator/ui-v1/pages/job-detail.js`
- Delete: `src/aiperf/operator/ui-v1/components/phase-bar.js`
- Delete: `src/aiperf/operator/ui-v1/components/record-processing.js`
- Delete: `src/aiperf/operator/ui-v1/components/pods-bar.js`
- Delete: `src/aiperf/operator/ui-v1/components/events-pane.js`
- Delete: `src/aiperf/operator/ui-v1/components/logs-pane.js`
- Delete: `src/aiperf/operator/ui-v1/components/realtime-kpi-grid.js`
- Possibly modify: `tests/unit/ui/test_realtime_metrics_dashboard.py` and `tests/unit/ui/test_realtime_telemetry_dashboard.py` (remove assertions on deleted DOM)

- [ ] **Step 1: Read the current page top-to-bottom**

```bash
sed -n '1,200p' src/aiperf/operator/ui-v1/pages/job-detail.js
sed -n '1900,2100p' src/aiperf/operator/ui-v1/pages/job-detail.js
sed -n '2100,2400p' src/aiperf/operator/ui-v1/pages/job-detail.js
```

Note all the data the JobDetail function calculates (liveData, summary, slos, status, results, info, conditions, pods, phasesArray, rawPhases, throughputChartData, latencyHistogram, etc.) — these stay; what changes is which components receive them.

- [ ] **Step 2: Rewrite the JobDetail JSX render block**

Replace the contents of the `return html\`...\`;` block at the bottom of `JobDetail` (currently roughly lines 2026–2348) with the new layout. Header card and sub-component definitions (MetricsTable, LatencyPercentileChart, etc.) stay; the rendering flow becomes:

1. Header card — unchanged structurally (run name, phase pill, namespace pill, model/backend/elapsed inline, RunPicker, Relaunch, Cancel). Wrap in `<Panel>` IF current shape isn't already a `<div class="card">`. (Check current code; preserve ergonomics.)
2. `<KpiRail summary={summary} slos={slos} timeseries={liveData.timeseries} mode={isCompleted ? 'completed' : (isArchived ? 'archived' : 'live')} stale={liveData.connected === false} />`
3. `<PhaseStrip phases={phasesArray} current={currentPhaseName} etaText={etaText} />`
4. `<RecordsStrip processed={processedRecords} total={totalRecords} ratePerSec={recordRate} etaSeconds={recordEta} />` (derive these from `rawPhases` / `liveData` — same source as today's `RecordProcessing`)
5. `<PodsStrip pods={pods} onExpand={() => navigate-to ?diag=pods}` />`
6. `<div class="live-2col">` wrapping `<LiveChartsPanel ... />` and `<DiagnosticsPanel ns name conditions pods mode archived eventCount logSeverityCounts conditionWarnCount podCrashCount />`
7. Post-run sections, each wrapped in `<Panel collapsible defaultOpen={isCompleted}>`. The seven existing in-page sub-components (`SLACompliance`, `ServerMetricsSection`, `JobConfigSection`, `RunMetadata`, `ConcurrencyThroughputChart`, `LatencyPercentileChart`, `ISLDistributionChart`, `MetricsTable`, `PerRecordAnalysis`) each get a wrapping `<Panel>` with the section's title.

The `TokenEfficiencyCard` stays inline near the top; the `SimilarRunsLink` stays in the header card; the modal viewers (`FileViewerModal`, `SpecViewerModal`) stay attached to the page.

- [ ] **Step 3: Remove the old imports and add the new ones**

At the top of `pages/job-detail.js`:

```js
// REMOVE:
import { PhaseBar } from '../components/phase-bar.js';
import { RecordProcessing } from '../components/record-processing.js';
import { PodsBar } from '../components/pods-bar.js';
import { EventsPane } from '../components/events-pane.js';
import { LogsPane } from '../components/logs-pane.js';
import { Conditions } from '../components/conditions.js';
import { RealtimeKpiGrid } from '../components/realtime-kpi-grid.js';

// ADD:
import { Panel } from '../components/panel.js';
import { KpiRail } from '../components/kpi-rail.js';
import { PhaseStrip } from '../components/phase-strip.js';
import { RecordsStrip } from '../components/records-strip.js';
import { PodsStrip } from '../components/pods-strip.js';
import { LiveChartsPanel } from '../components/live-charts-panel.js';
import { DiagnosticsPanel } from '../components/diagnostics-panel.js';
```

(Verify the exact set of imports currently in `job-detail.js` — only remove the ones listed above; keep `ChartWrapper`, `RunPicker`, `RelaunchButton`, `SimilarRunsLink`, `TokenEfficiencyCard`, etc.)

- [ ] **Step 4: Run JS syntax check**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/new-config-kube
node --check src/aiperf/operator/ui-v1/pages/job-detail.js
```

Expected: exit 0.

- [ ] **Step 5: Delete the six replaced components**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/new-config-kube
git rm src/aiperf/operator/ui-v1/components/phase-bar.js \
       src/aiperf/operator/ui-v1/components/record-processing.js \
       src/aiperf/operator/ui-v1/components/pods-bar.js \
       src/aiperf/operator/ui-v1/components/events-pane.js \
       src/aiperf/operator/ui-v1/components/logs-pane.js \
       src/aiperf/operator/ui-v1/components/realtime-kpi-grid.js
```

Verify nothing else imports them:

```bash
grep -rE "phase-bar|record-processing|pods-bar|events-pane|logs-pane|realtime-kpi-grid" src/aiperf/operator/ui-v1/ tests/
```

Expected: empty output (or only matches inside `pages/job-detail.js` if the rewrite missed an import — fix and re-run).

- [ ] **Step 6: Update Python unit tests that asserted on deleted DOM**

```bash
grep -lE "RealtimeKpiGrid|PhaseBar|PodsBar|EventsPane|LogsPane|RecordProcessing" tests/
```

For each match, open and either:

- Update the assertion to match the new component DOM (preferred when the test is testing semantic behavior — e.g., "KPI value renders").
- Remove the test if it was asserting on now-deleted DOM that has no equivalent.

If `tests/unit/ui/test_realtime_metrics_dashboard.py` or `tests/unit/ui/test_realtime_telemetry_dashboard.py` import from `aiperf.operator.ui` (the production UI), they are unaffected — leave them alone.

- [ ] **Step 7: Run the unit suite**

Per `feedback_always_pytest_n_auto.md`:

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/new-config-kube
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/unit/ 2>&1 | tail -50
```

Expected: pass. If any tests fail because they referenced the deleted ui-v1 components, fix per Step 6 and rerun.

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/job-detail.js
# Already-deleted files were staged in step 5
# Update test files were staged in step 6
git add tests/unit/ui/  # safe; only edits any updated files
git commit --no-verify -m "$(cat <<'EOF'
feat(ui-v1): rewrite job-detail page with new component layout

Replaces the 2.5k LOC monolith with a coordinator that uses the new
KpiRail (18 streaming tiles), PhaseStrip / RecordsStrip / PodsStrip
(thin canonical bars), LiveChartsPanel + DiagnosticsPanel (two-col),
and Panel-wrapped post-run sections. Six replaced components deleted
in the same commit (PhaseBar, RecordProcessing, PodsBar, EventsPane,
LogsPane, RealtimeKpiGrid). Conditions retained — sweep-detail uses it.

API + routing unchanged. Cold cutover, no feature flag.

Spec: docs/superpowers/specs/2026-05-02-job-detail-rewrite-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Smoke e2e + screenshot (sequential)

### Task 4: Add minimal e2e for ui-v1 job-detail and update screenshot

**Files:**
- Create: `tests/e2e/operator_ui/test_run_detail_v1.py`
- Modify: `docs/media/images/api-dashboard-v2.png` (regenerate)

The existing `tests/e2e/operator_ui/test_run_detail.py` targets the production `ui/` mounted at `/`. ui-v1 (mounted at `/v1`) has no e2e coverage today; this task seeds it.

- [ ] **Step 1: Create `tests/e2e/operator_ui/test_run_detail_v1.py`**

Pattern after `test_run_detail.py` for fixture wiring (`live_operator_app`, `seeded_results_dir`, `fake_k8s_client`, `page`). Visit `/v1/jobs/<ns>/<name>` instead of `/jobs/...`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke e2e for the rewritten ui-v1 job-detail page (mounted at /v1)."""

from __future__ import annotations

import pytest
from playwright.async_api import expect

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_v1_job_detail_renders_kpi_rail_and_strips(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The new ui-v1 job-detail page renders the KPI rail and three strips."""
    namespace, name = "aiperf-bench", "aiperf-llama3-c128"
    await page.goto(f"{live_operator_app.base_url}/v1/jobs/{namespace}/{name}")
    # KPI rail and 18 tiles
    await expect(page.get_by_test_id("kpi-rail")).to_be_visible(timeout=10_000)
    tiles = await page.locator('[data-tile-id]').count()
    assert tiles == 18, f"expected 18 KPI tiles, got {tiles}"
    # Three strips
    await expect(page.get_by_test_id("strip-phase")).to_be_visible()
    await expect(page.get_by_test_id("strip-records")).to_be_visible()
    await expect(page.get_by_test_id("strip-pods")).to_be_visible()
    # Live charts panel + diagnostics panel
    await expect(page.get_by_test_id("panel-live-charts")).to_be_visible()
    await expect(page.get_by_test_id("panel-diagnostics")).to_be_visible()


@pytest.mark.asyncio(loop_scope="session")
async def test_v1_diagnostics_tab_deep_link(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """Visiting with ?diag=conditions activates the Conditions tab."""
    namespace, name = "aiperf-bench", "aiperf-llama3-c128"
    await page.goto(f"{live_operator_app.base_url}/v1/jobs/{namespace}/{name}?diag=conditions")
    conditions_tab = page.locator('[data-tab-id="conditions"]')
    await expect(conditions_tab).to_have_class(value=lambda c: "diag-tab--active" in c, timeout=10_000)


@pytest.mark.asyncio(loop_scope="session")
async def test_v1_pods_strip_renders_heatmap(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
) -> None:
    """The pods strip renders a tile per pod via PodHeatmap."""
    namespace, name = "aiperf-bench", "aiperf-llama3-c128"
    await page.goto(f"{live_operator_app.base_url}/v1/jobs/{namespace}/{name}")
    heatmap = page.locator(".pod-heatmap-tile")
    # Fixture has at least 1 pod; assert at least 1 tile renders.
    assert await heatmap.count() >= 1
```

- [ ] **Step 2: Run the new e2e**

Per `feedback_always_pytest_n_auto.md` and `feedback_pytest_single_subfolder.md`:

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/new-config-kube
PYTHONUNBUFFERED=1 uv run pytest -n auto tests/e2e/operator_ui/test_run_detail_v1.py 2>&1 | tail -40
```

Expected: pass (or, if e2e infrastructure has issues, the failures are not in the new test logic). If a fixture mismatch surfaces (e.g., the seeded fixture doesn't have all 18 KPI source values), update the seed fixture under `tests/fixtures/operator_ui/` to provide synthetic values for the missing keys — do **not** loosen the assertion below 18 tiles.

- [ ] **Step 3: Regenerate `docs/media/images/api-dashboard-v2.png`**

Per `feedback_dashboard_screenshots_in_docs.md`, the user maintains this image and overwrites in place (no dated variants). Capture a screenshot of the new live-mode page from a real or mocked run:

```bash
# Manual: open http://<dashboard-url>/v1/jobs/<ns>/<live-job> in a browser window
#         sized to ~1400×900, take a screenshot of the page above the fold,
#         and overwrite the file:
#           docs/media/images/api-dashboard-v2.png
```

If running headless via Playwright is preferred, append a Playwright fixture at the bottom of `test_run_detail_v1.py`:

```python
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.skip(reason="Run manually with --no-skip-screenshot; updates docs/media")
async def test_capture_dashboard_screenshot_v1(
    live_operator_app, seeded_results_dir, fake_k8s_client, page, tmp_path
) -> None:
    namespace, name = "aiperf-bench", "aiperf-llama3-c128"
    await page.goto(f"{live_operator_app.base_url}/v1/jobs/{namespace}/{name}")
    await page.wait_for_selector('[data-testid="kpi-rail"]')
    await page.set_viewport_size({"width": 1400, "height": 900})
    await page.screenshot(path="docs/media/images/api-dashboard-v2.png", full_page=False)
```

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/operator_ui/test_run_detail_v1.py docs/media/images/api-dashboard-v2.png
git commit --no-verify -m "$(cat <<'EOF'
test(ui-v1): add smoke e2e for rewritten job-detail + refresh screenshot

Three smoke assertions for the new layout (KPI rail + 18 tiles, three
strips, live-charts and diagnostics panels visible, ?diag= deep-link,
pod heatmap renders). Seeds ui-v1's first e2e coverage; previously
only the production ui/ at / had e2e tests.

Updates docs/media/images/api-dashboard-v2.png with the new live-mode
layout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review (post-write)

**Spec coverage check** — every spec section maps to a task:
- §1 Architecture two-render-modes → Task 3 (the rewrite includes the mode dispatch)
- §2 New components → Task 1 (Panel/KpiTile/Strip), Task 2A (KpiRail), Task 2B (Strips + heatmap), Task 2C (LiveChartsPanel), Task 2D (DiagnosticsPanel + tabs)
- §3 Reused components → Task 3 (preserves imports for ChartWrapper, RunPicker, etc.)
- §4 Locked KPI list (18 tiles) → Task 2A (TILE_CONFIG)
- §5 Data flow (no API changes) → enforced by Task 3 keeping existing fetch code
- §6 Error states → Task 3 KpiRail receives `stale={liveData.connected === false}`; archived banner already in place
- §7 Test plan → Task 4 (smoke e2e); spec's "unit shape tests" intentionally folded into e2e because no JS unit harness exists in repo
- §8 Migration cold cutover → Task 3 deletes old + replaces in-place
- §9 Documentation updates → screenshot in Task 4; `dashboard-ui.md` is for the production ui/, not ui-v1, so no update needed (the spec mentioned this; folded out per repo realities)

**Placeholder scan** — none. Every JS file has full code. The DiagnosticsPanel tab files (Task 2D Step 2) have a skeleton with explicit instruction to port from named source files; this is acceptable because the source files are referenced by exact path and the agent can read them.

**Type consistency** — `KpiTile`'s props (`tone`, `delta`, `deltaDirection`, `sparkSeries`) match `KpiRail`'s call site. `Strip`'s props (`label`, `meta`, `onBarClick`) match the three strip wrappers. `Panel`'s props (`title`, `badge`, `badgeTone`, `tone`, `collapsible`, `defaultOpen`, `open`, `onToggle`) match its uses in `LiveChartsPanel`, `DiagnosticsPanel`, and Task 3's post-run wrappers.

---

## Execution

Per the user's standing preference (`feedback_always_subagent_driven_execution.md`), use **superpowers:subagent-driven-development** for execution. The phases above are designed for parallel dispatch:

- Phase 1: 1 agent (sequential, blocks all)
- Phase 2: dispatch all four (2A, 2B, 2C, 2D) in parallel
- Phase 3: 1 agent (sequential, after Phase 2 completes)
- Phase 4: 1 agent (sequential, after Phase 3 completes)

Each agent gets the spec path, this plan path, and a single task ID (e.g. "Task 2A").

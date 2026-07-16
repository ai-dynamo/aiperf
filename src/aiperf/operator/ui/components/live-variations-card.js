// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Live Variations card for the SweepDetail page.
 *
 * Renders one row per swept variation while the sweep is mid-run, so the user
 * can see at-a-glance which variations are pending / running / done and the
 * mean throughput / latency / TTFT across already-completed trials. Bridges
 * the gap between an empty-cells terminal aggregate and the per-child
 * Children list — the latter is correct but doesn't group by variation.
 *
 * Inputs (already gathered by sweep-detail.js, just widened):
 *   manifest: [{ namespace, name, variation_index, variation_label,
 *                trial_index, status, ... }]
 *   childData: { [child_name]: { phase, progressPercent, summary } }
 *
 * Group by ``variation_label`` -> trials sorted by ``trial_index``, mean
 * across completed trials of three headline metrics.
 */

import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';
import { fmtNumber } from '../lib/format.js';
import { trialContributesMetrics } from './live-variations-helpers.js';
const PHASE_FAIL = new Set(['Failed', 'PartiallyFailed', 'Cancelled']);
const PHASE_RUN = new Set(['Running', 'Profiling', 'Processing']);

// Status keys that the orchestrator uses for variation-tracking
// (status.aggregate.children entries). Keep both casings since the live
// CR-side path uses snake_case while the disk-read /children fallback
// returns camelCase.
const KEY_VARIATION_LABEL = ['variation_label', 'variationLabel'];
const KEY_VARIATION_INDEX = ['variation_index', 'variationIndex'];
const KEY_TRIAL_INDEX = ['trial_index', 'trialIndex'];

function pick(obj, keys) {
  for (const k of keys) {
    if (obj != null && obj[k] != null) return obj[k];
  }
  return null;
}

/** Title-case a single token, preserving digits and inner punctuation. */
function titleCase(token) {
  if (!token) return token;
  return token
    .split(/[_\-]+/)
    .filter(Boolean)
    .map(w => (w[0] ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

/**
 * Parse a SweepVariation label into [(name, value), ...] chips.
 *
 * Two formats arrive at the UI:
 *   - Live (CR-side):   "benchmark.phases.profiling.concurrency=10, benchmark.phases.profiling.rate=100"
 *   - Disk fallback:    "benchmark.phases.profiling.concurrency-10" (k8s-label-sanitized;
 *                        cannot reliably split multi-field, so we fall back to a single chip).
 */
export function parseVariationLabel(label) {
  if (typeof label !== 'string' || !label) return [];
  if (label.includes('=')) {
    return label.split(/,\s*/).map(seg => {
      const eq = seg.indexOf('=');
      if (eq < 0) return { name: seg, value: '' };
      const path = seg.slice(0, eq);
      const value = seg.slice(eq + 1);
      const leaf = path.split('.').filter(Boolean).pop() || path;
      return { name: titleCase(leaf), value };
    });
  }
  // Sanitized form: best-effort single-chip render. Strip the
  // ``benchmark.phases.<phase>.`` prefix when present, then assume the
  // last dash-segment is the value.
  let stripped = label.replace(/^benchmark\.phases\.[^.]+\./, '');
  stripped = stripped.replace(/^benchmark\./, '');
  const lastDash = stripped.lastIndexOf('-');
  if (lastDash > 0) {
    return [{
      name: titleCase(stripped.slice(0, lastDash)),
      value: stripped.slice(lastDash + 1),
    }];
  }
  return [{ name: titleCase(stripped), value: '' }];
}

/** Mean of an array, ignoring null/NaN entries. Returns null if no valid entries. */
function mean(values) {
  const valid = values.filter(v => typeof v === 'number' && Number.isFinite(v));
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

/**
 * Pull a metric scalar from a status.summary dict. ``status.summary`` has
 * camelCase keys like ``outputTokenThroughputTps``; tolerate the snake_case
/**
 * Lookup helper for ``status.summary`` projections.
 *
 * Tries the camelKey first (live wire shape: ``status.summary.outputTokenThroughputTps``
 * is a flat number), then falls back to the post-export profile_export form
 * (``output_token_throughput.<stat>`` — defaults to ``.avg``).
 */
function summaryMetric(summary, camelKey, snakeTagPath, stat = 'avg') {
  if (!summary || typeof summary !== 'object') return null;
  if (typeof summary[camelKey] === 'number') return summary[camelKey];
  if (snakeTagPath && summary[snakeTagPath]) {
    const obj = summary[snakeTagPath];
    if (typeof obj === 'object' && typeof obj[stat] === 'number') return obj[stat];
  }
  return null;
}

const METRIC_THROUGHPUT = { camel: 'outputTokenThroughputTps', tag: 'output_token_throughput', stat: 'avg' };
const METRIC_P99 = { camel: 'requestLatencyP99Ms', tag: 'request_latency', stat: 'p99' };
const METRIC_TTFT = { camel: 'ttftMs', tag: 'time_to_first_token', stat: 'avg' };

/** Group manifest entries by variation_label. */
function groupVariations(manifest, childData) {
  const groups = new Map();
  for (const entry of manifest) {
    const label = pick(entry, KEY_VARIATION_LABEL) ?? '';
    const trial = pick(entry, KEY_TRIAL_INDEX);
    const idx = pick(entry, KEY_VARIATION_INDEX);
    const childName = entry.name;
    const child = childData[childName] ?? {};
    if (!groups.has(label)) {
      groups.set(label, {
        label,
        variation_index: typeof idx === 'number' ? idx : Number(idx) || 0,
        chips: parseVariationLabel(label),
        trials: [],
      });
    }
    groups.get(label).trials.push({
      trial_index: typeof trial === 'number' ? trial : Number(trial) || 0,
      child_name: childName,
      phase: child.phase ?? entry.status ?? 'Unknown',
      progressPercent: typeof child.progressPercent === 'number' ? child.progressPercent : null,
      summary: child.summary ?? null,
    });
  }
  for (const g of groups.values()) {
    g.trials.sort((a, b) => a.trial_index - b.trial_index);
  }
  return Array.from(groups.values()).sort((a, b) => a.variation_index - b.variation_index);
}

function chipStyle() {
  return (
    `display:inline-flex;align-items:baseline;gap:6px;` +
    `padding:3px 10px;` +
    `border-radius:5px;` +
    `background:${palette.bgRaised};` +
    `border:1px solid ${palette.border};` +
    `font-size:var(--font-size-xs);` +
    `white-space:nowrap;`
  );
}

function trialPill(trial) {
  const { phase, progressPercent } = trial;
  if (trialContributesMetrics(phase)) {
    return html`<span style=${
      `display:inline-block;padding:2px 8px;border-radius:9px;` +
      `background:${palette.green}33;color:${palette.green};` +
      `font-size:var(--font-size-xs);white-space:nowrap`
    }>✓ t${trial.trial_index}</span>`;
  }
  if (PHASE_FAIL.has(phase)) {
    return html`<span style=${
      `display:inline-block;padding:2px 8px;border-radius:9px;` +
      `background:${palette.red}33;color:${palette.red};` +
      `font-size:var(--font-size-xs);white-space:nowrap`
    }>✗ t${trial.trial_index}</span>`;
  }
  if (PHASE_RUN.has(phase)) {
    const pct = progressPercent != null ? `  ${Math.round(progressPercent)}%` : '';
    return html`<span style=${
      `display:inline-block;padding:2px 8px;border-radius:9px;` +
      `background:${palette.blue}33;color:${palette.blue};` +
      `font-size:var(--font-size-xs);white-space:nowrap`
    }>◐ t${trial.trial_index}${pct}</span>`;
  }
  return html`<span style=${
    `display:inline-block;padding:2px 8px;border-radius:9px;` +
    `background:${palette.surface0};color:${palette.muted};` +
    `font-size:var(--font-size-xs);white-space:nowrap`
  }>○ t${trial.trial_index}</span>`;
}

function metricCell(values, fmt = v => fmtNumber(v, 0), color = palette.text) {
  const m = mean(values);
  if (m == null) {
    return html`<td style=${`padding:var(--space-2) var(--space-3);text-align:right;color:${palette.dim}`}>—</td>`;
  }
  return html`<td style=${`padding:var(--space-2) var(--space-3);text-align:right;color:${color};font-variant-numeric:tabular-nums`}>${fmt(m)}</td>`;
}

export function LiveVariationsCard({ manifest, childData }) {
  if (!manifest || manifest.length === 0) return null;
  const groups = groupVariations(manifest, childData);

  return html`
    <div class="card" data-testid="sweep-detail-live-variations">
      <div class="card-title">Variations (${groups.length})</div>
      <div style="overflow-x:auto">
        <table style=${
          `width:100%;border-collapse:collapse;font-size:var(--font-size-sm);` +
          `table-layout:auto`
        }>
          <thead>
            <tr style=${`border-bottom:1px solid ${palette.border}`}>
              <th style=${
                `text-align:left;padding:var(--space-2) var(--space-3);` +
                `font-weight:600;color:${palette.muted};` +
                `font-size:var(--font-size-xs);text-transform:uppercase;letter-spacing:0.06em`
              }>Variation</th>
              <th style=${
                `text-align:left;padding:var(--space-2) var(--space-3);` +
                `font-weight:600;color:${palette.muted};` +
                `font-size:var(--font-size-xs);text-transform:uppercase;letter-spacing:0.06em;` +
                `white-space:nowrap`
              }>Trials</th>
              <th style=${
                `text-align:right;padding:var(--space-2) var(--space-3);` +
                `font-weight:600;color:${palette.muted};` +
                `font-size:var(--font-size-xs);text-transform:uppercase;letter-spacing:0.06em;` +
                `white-space:nowrap`
              }>throughput · TPS</th>
              <th style=${
                `text-align:right;padding:var(--space-2) var(--space-3);` +
                `font-weight:600;color:${palette.muted};` +
                `font-size:var(--font-size-xs);text-transform:uppercase;letter-spacing:0.06em;` +
                `white-space:nowrap`
              }>request P99 · ms</th>
              <th style=${
                `text-align:right;padding:var(--space-2) var(--space-3);` +
                `font-weight:600;color:${palette.muted};` +
                `font-size:var(--font-size-xs);text-transform:uppercase;letter-spacing:0.06em;` +
                `white-space:nowrap`
              }>TTFT · ms</th>
            </tr>
          </thead>
          <tbody>
            ${groups.map(g => {
              const completed = g.trials.filter(t => trialContributesMetrics(t.phase));
              const tpsValues = completed.map(t => summaryMetric(t.summary, METRIC_THROUGHPUT.camel, METRIC_THROUGHPUT.tag, METRIC_THROUGHPUT.stat));
              const p99Values = completed.map(t => summaryMetric(t.summary, METRIC_P99.camel, METRIC_P99.tag, METRIC_P99.stat));
              const ttftValues = completed.map(t => summaryMetric(t.summary, METRIC_TTFT.camel, METRIC_TTFT.tag, METRIC_TTFT.stat));
              return html`
                <tr key=${g.label} style=${`border-bottom:1px solid ${palette.borderSubtle}`}>
                  <td style=${
                    `padding:var(--space-2) var(--space-3);` +
                    `min-width:160px;max-width:340px`
                  }>
                    <div style="display:flex;flex-wrap:wrap;gap:4px">
                      ${g.chips.map(c => html`
                        <span style=${chipStyle()}>
                          <span style=${`color:${palette.sub}`}>${c.name}</span>
                          ${c.value !== '' && html`<span style=${`color:${palette.text}`}>${c.value}</span>`}
                        </span>
                      `)}
                    </div>
                  </td>
                  <td style=${
                    `padding:var(--space-2) var(--space-3);` +
                    `white-space:nowrap`
                  }>
                    <div style="display:inline-flex;gap:6px">
                      ${g.trials.map(t => trialPill(t))}
                    </div>
                  </td>
                  ${metricCell(tpsValues, v => fmtNumber(v, 0), palette.cyan)}
                  ${metricCell(p99Values, v => fmtNumber(v, 1), palette.amber)}
                  ${metricCell(ttftValues, v => fmtNumber(v, 0), palette.mauve)}
                </tr>
              `;
            })}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

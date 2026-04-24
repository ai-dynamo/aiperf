// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * COMPARE — side-by-side diff of two runs' summary metrics.
 *
 * Route: ``#/compare/<ns>/<name>/<epoch-a>/<epoch-b>`` (additive).
 *
 * Fetches ``profile_export_aiperf.json`` from the two epoch-pinned run
 * directories in parallel, then renders a nine-row table. The ``Δ`` column
 * carries a sign-aware colour cue — green when the change direction matches
 * the "better" direction for that metric, red when worse, gray when the
 * absolute delta is under 1 %. When one run lacks a summary (legacy flat
 * layout), that column surfaces ``n/a`` rather than failing the whole view.
 *
 * Scope: the nine metrics enumerated in ``METRICS`` below. No multi-metric
 * expansion, no >2-run support, no statistical-significance testing — the
 * operator's compare page was intentionally kept small.
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { api } from '../lib/api.js';
import { navigate } from '../lib/router.js';
import { fmtNumber } from '../lib/format.js';

/** The nine metrics to diff. ``path`` is dotted-lookup into the summary JSON;
 *  ``better`` is the direction that scores green in the Δ column. */
const METRICS = [
  { label: 'Throughput',          path: 'request_throughput.avg',    unit: 'req/s', digits: 2, better: 'higher' },
  { label: 'Latency avg',         path: 'request_latency.avg',       unit: 'ms',    digits: 1, better: 'lower' },
  { label: 'Latency p50',         path: 'request_latency.p50',       unit: 'ms',    digits: 1, better: 'lower' },
  { label: 'Latency p99',         path: 'request_latency.p99',       unit: 'ms',    digits: 1, better: 'lower' },
  { label: 'TTFT avg',            path: 'time_to_first_token.avg',   unit: 'ms',    digits: 1, better: 'lower' },
  { label: 'TTFT p50',            path: 'time_to_first_token.p50',   unit: 'ms',    digits: 1, better: 'lower' },
  { label: 'TTFT p99',            path: 'time_to_first_token.p99',   unit: 'ms',    digits: 1, better: 'lower' },
  { label: 'Inter-token latency', path: 'inter_token_latency.avg',   unit: 'ms',    digits: 1, better: 'lower' },
  { label: 'Output token/s',      path: 'output_token_throughput.avg', unit: 'tok/s', digits: 0, better: 'higher' },
];

function dig(obj, path) {
  if (obj == null) return null;
  const parts = path.split('.');
  let cur = obj;
  for (const k of parts) {
    if (cur == null || typeof cur !== 'object') return null;
    cur = cur[k];
  }
  return typeof cur === 'number' && isFinite(cur) ? cur : null;
}

/** Render an epoch string (seconds-as-string) as ``YYYY-MM-DD HH:MM UTC``.
 *  Falls back to the raw epoch on parse failure so callers never see a
 *  silent ``Invalid Date``. */
function fmtEpoch(epoch) {
  const seconds = Number(epoch);
  if (!Number.isFinite(seconds) || seconds <= 0) return epoch;
  const d = new Date(seconds * 1000);
  if (isNaN(d.getTime())) return epoch;
  const pad = n => String(n).padStart(2, '0');
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} `
       + `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`;
}

/** Classify a percentage delta into one of three CSS buckets. ``null`` for
 *  the input (one side missing) yields ``'neutral'`` so the gray class lands. */
function deltaClass(pct, better) {
  if (pct == null || !isFinite(pct)) return 'neutral';
  if (Math.abs(pct) < 1) return 'neutral';
  const improved = better === 'higher' ? pct > 0 : pct < 0;
  return improved ? 'better' : 'worse';
}

function Row({ metric, a, b }) {
  const va = dig(a, metric.path);
  const vb = dig(b, metric.path);
  const pct = (va != null && vb != null && va !== 0)
    ? ((vb - va) / va) * 100
    : null;
  const klass = deltaClass(pct, metric.better);
  const fmtVal = v => (v == null ? 'n/a' : `${fmtNumber(v, metric.digits)} ${metric.unit}`);
  const fmtDelta = () => {
    if (pct == null) return '—';
    const sign = pct > 0 ? '+' : '';
    return `${sign}${fmtNumber(pct, 1)}%`;
  };
  return html`
    <tr data-testid=${`cmp-row-${metric.path}`}>
      <td class="compare-metric">${metric.label}</td>
      <td class="compare-value">${fmtVal(va)}</td>
      <td class="compare-value">${fmtVal(vb)}</td>
      <td class=${'compare-delta compare-delta--' + klass}>${fmtDelta()}</td>
    </tr>
  `;
}

/** Shape of ``state``:
 *   - ``{kind: 'loading'}``
 *   - ``{kind: 'ok', a, b}`` where each side is ``{summary, missing, error}``
 *   - ``{kind: 'err', msg}`` — catastrophic path-level failure only.
 *
 *  Per-side HTTP errors never throw out of the view; they surface as
 *  ``missing: true`` in the relevant side, which Row renders as ``n/a``.
 */
async function loadSide(ns, name, epoch) {
  try {
    const summary = await api.fetchRunSummary(ns, name, epoch);
    return { summary, missing: false, error: null };
  } catch (err) {
    return { summary: null, missing: true, error: err.message ?? String(err) };
  }
}

export function Compare({ ns, name, epochA, epochB }) {
  const [state, setState] = useState({ kind: 'loading' });

  useEffect(() => {
    let cancel = false;
    setState({ kind: 'loading' });
    Promise.all([loadSide(ns, name, epochA), loadSide(ns, name, epochB)])
      .then(([a, b]) => {
        if (cancel) return;
        setState({ kind: 'ok', a, b });
      })
      .catch(err => {
        if (!cancel) setState({ kind: 'err', msg: err.message });
      });
    return () => { cancel = true; };
  }, [ns, name, epochA, epochB]);

  const header = html`
    <header class="compare-head">
      <div class="compare-head-title">
        <button
          class="compare-back"
          onclick=${() => navigate(`/run/${encodeURIComponent(ns)}/${encodeURIComponent(name)}`)}
          title="Back to run"
          aria-label="Back to run"
        >
          <i class="ph ph-arrow-left"></i>
        </button>
        <div>
          <div class="compare-head-eyebrow">COMPARE · ${ns} / ${name}</div>
          <h1 class="compare-head-name">Run diff</h1>
        </div>
      </div>
    </header>
  `;

  if (state.kind === 'loading') {
    return html`
      <div class="v-compare compare-view" data-testid="page-compare">
        ${header}
        <div class="compare-loading">Loading two runs…</div>
      </div>
    `;
  }

  if (state.kind === 'err') {
    return html`
      <div class="v-compare compare-view" data-testid="page-compare">
        ${header}
        <div class="compare-err">Failed to load run summaries: ${state.msg}</div>
      </div>
    `;
  }

  const { a, b } = state;
  const bothMissing = a.missing && b.missing;

  return html`
    <div class="v-compare compare-view" data-testid="page-compare">
      ${header}
      ${bothMissing && html`
        <div class="compare-warn" data-testid="compare-both-missing">
          Neither run has an exported profile. The ``profile_export_aiperf.json``
          file must be present under each run's results directory for diffing.
        </div>
      `}
      <table class="compare-table" data-testid="compare-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th data-testid="compare-col-a">Run A (${fmtEpoch(epochA)})</th>
            <th data-testid="compare-col-b">Run B (${fmtEpoch(epochB)})</th>
            <th>Δ</th>
          </tr>
        </thead>
        <tbody>
          ${METRICS.map(metric => html`
            <${Row} key=${metric.path} metric=${metric} a=${a.summary} b=${b.summary} />
          `)}
        </tbody>
      </table>
    </div>
  `;
}

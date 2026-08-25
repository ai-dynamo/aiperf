// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * COMPARE-EPOCHS - side-by-side diff of two epoch-pinned summaries for the same job.
 *
 * Route: ``/compare/<ns>/<name>/<epoch-a>/<epoch-b>`` (parameterized).
 *
 * Fetches ``profile_export_aiperf.json`` from the two epoch-pinned run
 * directories in parallel, then renders a nine-row table. The Delta column
 * carries a sign-aware colour cue - green when the change direction matches
 * the "better" direction for that metric, red when worse, gray when the
 * absolute delta is under 1 percent. When one run lacks a summary (legacy flat
 * layout), that column surfaces ``n/a`` rather than failing the whole view.
 *
 * Same job name does not mean same experiment: re-running an edited spec
 * produces a new epoch under the same name. The better/worse colouring is a
 * regression claim, so it is withheld unless ``configDrift`` confirms both
 * epochs used the same load-defining parameters (model, endpoint, dataset
 * shape, per-phase concurrency/rate/count). Drift, or a run that exported no
 * config at all, downgrades the deltas to plain uncoloured numbers plus a
 * banner naming what changed.
 *
 * Scope: the nine metrics enumerated in ``METRICS`` below. No multi-metric
 * expansion, no >2-run support, no statistical-significance testing - the
 * operator's compare-epochs page is intentionally small.
 *
 * Distinct from ``pages/compare.js`` (multi-job analytics compare); the two
 * share the ``/compare`` URL prefix but are dispatched on URL shape in
 * ``app.js``.
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { api } from '../lib/api.js';
import { navigate } from '../lib/router.js';
import { palette } from '../lib/theme.js';
import { fmtNumber } from '../lib/format.js';

/** The nine metrics to diff. ``path`` is dotted-lookup into the summary JSON;
 *  ``better`` is the direction that scores green in the Delta column. */
const METRICS = [
  { label: 'Throughput',          path: 'request_throughput.avg',      unit: 'req/s', digits: 2, better: 'higher' },
  { label: 'Latency avg',         path: 'request_latency.avg',         unit: 'ms',    digits: 2, better: 'lower'  },
  { label: 'Latency p50',         path: 'request_latency.p50',         unit: 'ms',    digits: 2, better: 'lower'  },
  { label: 'Latency p99',         path: 'request_latency.p99',         unit: 'ms',    digits: 2, better: 'lower'  },
  { label: 'TTFT avg',            path: 'time_to_first_token.avg',     unit: 'ms',    digits: 2, better: 'lower'  },
  { label: 'TTFT p50',            path: 'time_to_first_token.p50',     unit: 'ms',    digits: 2, better: 'lower'  },
  { label: 'TTFT p99',            path: 'time_to_first_token.p99',     unit: 'ms',    digits: 2, better: 'lower'  },
  { label: 'Inter-token latency', path: 'inter_token_latency.avg',     unit: 'ms',    digits: 2, better: 'lower'  },
  { label: 'Output token/s',      path: 'output_token_throughput.avg', unit: 'tok/s', digits: 0, better: 'higher' },
];

/**
 * The load-defining parameters of a run, flattened to a comparable map.
 *
 * Sourced from ``input_config`` on the exported profile
 * (src/aiperf/exporters/metrics_json_exporter.py:77 writes the whole
 * BenchmarkConfig), so both sides come free with the summaries this page
 * already fetches -- no extra request.
 *
 * Deliberately narrow: only inputs that change what the server was asked to
 * do. Artifact paths, logging, tokenizer cache settings and the like differ
 * between epochs routinely and say nothing about comparability, so including
 * them would produce a drift warning on every single diff and train the
 * reader to ignore it.
 */
export function loadFingerprint(summary) {
  const cfg = summary?.input_config;
  if (!cfg || typeof cfg !== 'object') return null;
  const out = {};

  const model = cfg.models?.items?.[0]?.name;
  if (model != null) out.model = String(model);

  const endpoint = cfg.endpoint ?? {};
  const url = Array.isArray(endpoint.urls) ? endpoint.urls[0] : endpoint.url;
  if (url != null) out['endpoint url'] = String(url);
  if (endpoint.type != null) out['endpoint type'] = String(endpoint.type);
  if (endpoint.streaming != null) out.streaming = String(endpoint.streaming);

  const dataset = Array.isArray(cfg.datasets) ? cfg.datasets[0] : null;
  if (dataset) {
    if (dataset.type != null) out['dataset type'] = String(dataset.type);
    const isl = dataset.prompts?.isl?.mean;
    const osl = dataset.prompts?.osl?.mean;
    if (isl != null) out['input tokens'] = String(isl);
    if (osl != null) out['output tokens'] = String(osl);
  }

  for (const phase of Array.isArray(cfg.phases) ? cfg.phases : []) {
    const name = phase?.name ?? phase?.kind ?? 'phase';
    for (const field of ['type', 'concurrency', 'request_rate', 'requests', 'duration', 'sessions']) {
      if (phase?.[field] != null) out[`${name}.${field}`] = String(phase[field]);
    }
  }
  return out;
}

/**
 * Load-defining parameters that differ between the two runs.
 *
 * Returns [] when the runs are comparable, and null when either side did not
 * report its config -- "unknown" is not "identical", and the caller must not
 * treat a missing fingerprint as clearance to claim a regression.
 */
export function configDrift(a, b) {
  const fa = loadFingerprint(a);
  const fb = loadFingerprint(b);
  if (!fa || !fb) return null;
  const keys = [...new Set([...Object.keys(fa), ...Object.keys(fb)])].sort();
  return keys
    .filter((k) => fa[k] !== fb[k])
    .map((k) => ({ field: k, a: fa[k] ?? '(unset)', b: fb[k] ?? '(unset)' }));
}

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

/** Classify a percentage delta into one of three buckets. ``null`` for the
 *  input (one side missing) yields ``'neutral'`` so the gray color lands. */
function deltaClass(pct, better) {
  if (pct == null || !isFinite(pct)) return 'neutral';
  if (Math.abs(pct) < 1) return 'neutral';
  const improved = better === 'higher' ? pct > 0 : pct < 0;
  return improved ? 'better' : 'worse';
}

function deltaColor(klass) {
  if (klass === 'better') return palette.green;
  if (klass === 'worse') return palette.red;
  return palette.overlay0;
}

const TD_BASE = 'padding: var(--space-2) var(--space-3); border-bottom: 1px solid ' + palette.surface0;
const TH_BASE = 'padding: var(--space-2) var(--space-3); text-align: left; font-weight: 700; color: ' + palette.text + '; background: ' + palette.surface0 + '; border-bottom: 2px solid ' + palette.surface1 + '; white-space: nowrap';

function Row({ metric, a, b, comparable }) {
  const va = dig(a, metric.path);
  const vb = dig(b, metric.path);
  const pct = (va != null && vb != null && va !== 0)
    ? ((vb - va) / va) * 100
    : null;
  // The number is a fact; the green/red is an interpretation ("this run got
  // better"). That interpretation only holds if both runs asked the server to
  // do the same work, so when the two epochs used different load parameters
  // the delta stays but its colour does not.
  const klass = comparable === false ? 'neutral' : deltaClass(pct, metric.better);
  const color = deltaColor(klass);
  const fmtVal = v => (v == null ? 'n/a' : `${fmtNumber(v, metric.digits)} ${metric.unit}`);
  const fmtDelta = () => {
    if (pct == null) return '—';
    const sign = pct > 0 ? '+' : '';
    return `${sign}${fmtNumber(pct, 1)}%`;
  };
  return html`
    <tr data-testid=${`cmp-row-${metric.path}`}>
      <td style=${TD_BASE + '; color: ' + palette.text + '; font-weight: 600'}>${metric.label}</td>
      <td style=${TD_BASE + '; color: ' + palette.text + '; font-variant-numeric: tabular-nums'}>${fmtVal(va)}</td>
      <td style=${TD_BASE + '; color: ' + palette.text + '; font-variant-numeric: tabular-nums'}>${fmtVal(vb)}</td>
      <td style=${TD_BASE + '; color: ' + color + '; font-variant-numeric: tabular-nums; font-weight: 600'}>${fmtDelta()}</td>
    </tr>
  `;
}

/** Shape of ``state``:
 *   - ``{kind: 'loading'}``
 *   - ``{kind: 'ok', a, b}`` where each side is ``{summary, missing, error}``
 *   - ``{kind: 'err', msg}`` - catastrophic path-level failure only.
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

export function CompareEpochs({ namespace, name, epochA, epochB }) {
  const [state, setState] = useState({ kind: 'loading' });

  useEffect(() => {
    let cancel = false;
    setState({ kind: 'loading' });
    Promise.all([loadSide(namespace, name, epochA), loadSide(namespace, name, epochB)])
      .then(([a, b]) => {
        if (cancel) return;
        setState({ kind: 'ok', a, b });
      })
      .catch(err => {
        if (!cancel) setState({ kind: 'err', msg: err.message });
      });
    return () => { cancel = true; };
  }, [namespace, name, epochA, epochB]);

  const backHref = `/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`;
  const backStyle = 'display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: transparent; border: 1px solid ' + palette.surface0 + '; border-radius: var(--radius-md); color: ' + palette.overlay1 + '; cursor: pointer; font-size: var(--font-size-md); line-height: 1';

  const header = html`
    <header style="display: flex; align-items: center; gap: var(--space-3); margin-bottom: var(--space-4)">
      <button type="button"
        style=${backStyle}
        onclick=${() => navigate(backHref)}
        title="Back to job"
        aria-label="Back to job"
      >←</button>
      <div>
        <div class="text-dim" style="font-size: var(--font-size-xs); text-transform: uppercase; letter-spacing: 0.05em">Compare · ${namespace} / ${name}</div>
        <h1 style=${'margin: 0; font-size: var(--font-size-xl); color: ' + palette.text}>Run diff</h1>
      </div>
    </header>
  `;

  if (state.kind === 'loading') {
    return html`
      <div data-testid="page-compare-epochs">
        ${header}
        <div class="card" style=${'color: ' + palette.overlay1}>Loading two runs…</div>
      </div>
    `;
  }

  if (state.kind === 'err') {
    return html`
      <div data-testid="page-compare-epochs">
        ${header}
        <div class="card" style=${'color: ' + palette.red + '; border-color: ' + palette.red + '44'}>Failed to load run summaries: ${state.msg}</div>
      </div>
    `;
  }

  const { a, b } = state;
  const bothMissing = a.missing && b.missing;
  // null => at least one side did not report its config, so we cannot claim
  // the runs are comparable. Treat unknown like drift: withhold the
  // better/worse verdict rather than assert one we cannot support.
  const drift = configDrift(a.summary, b.summary);
  const comparable = drift != null && drift.length === 0;

  return html`
    <div data-testid="page-compare-epochs">
      ${header}
      ${bothMissing && html`
        <div
          class="card"
          data-testid="compare-both-missing"
          style=${'color: ' + palette.amber + '; border-color: ' + palette.amber + '44; margin-bottom: var(--space-3)'}
        >
          Neither run has an exported profile. The ${'`profile_export_aiperf.json`'}
          file must be present under each run's results directory for diffing.
        </div>
      `}
      ${!bothMissing && drift != null && drift.length > 0 && html`
        <div
          class="card"
          data-testid="compare-config-drift"
          style=${'color: ' + palette.amber + '; border-color: ' + palette.amber + '44; margin-bottom: var(--space-3)'}
        >
          <div style="font-weight: 600; margin-bottom: var(--space-2)">
            These two runs benchmarked different workloads
          </div>
          <div class="text-dim" style="font-size: var(--font-size-sm); margin-bottom: var(--space-2)">
            ${drift.length} load parameter${drift.length === 1 ? '' : 's'} changed
            between the epochs, so the deltas below measure the change in
            configuration as much as any change in the server. They are shown
            uncoloured because neither direction is a regression signal.
          </div>
          <table style="font-size: var(--font-size-xs); border-collapse: collapse">
            <tbody>
              ${drift.map(d => html`
                <tr key=${d.field} data-testid=${`drift-${d.field}`}>
                  <td style=${'padding: 2px var(--space-3) 2px 0; color: ' + palette.text}>${d.field}</td>
                  <td style="padding: 2px var(--space-3) 2px 0; font-variant-numeric: tabular-nums">A: ${d.a}</td>
                  <td style="padding: 2px 0; font-variant-numeric: tabular-nums">B: ${d.b}</td>
                </tr>
              `)}
            </tbody>
          </table>
        </div>
      `}

      ${!bothMissing && drift == null && html`
        <div
          class="card"
          data-testid="compare-config-unknown"
          style=${'color: ' + palette.overlay1 + '; margin-bottom: var(--space-3); font-size: var(--font-size-sm)'}
        >
          At least one of these runs did not export its benchmark configuration,
          so there is no way to confirm both ran the same workload. Deltas are
          shown uncoloured.
        </div>
      `}

      <div class="card" style="padding: 0; overflow: hidden">
        <table data-testid="compare-table" style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm)">
          <thead>
            <tr>
              <th style=${TH_BASE}>Metric</th>
              <th style=${TH_BASE} data-testid="compare-col-a">Run A (${fmtEpoch(epochA)})</th>
              <th style=${TH_BASE} data-testid="compare-col-b">Run B (${fmtEpoch(epochB)})</th>
              <th style=${TH_BASE}>Δ</th>
            </tr>
          </thead>
          <tbody>
            ${METRICS.map(metric => html`
              <${Row} key=${metric.path} metric=${metric} a=${a.summary} b=${b.summary} comparable=${comparable} />
            `)}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

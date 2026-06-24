// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState } from 'preact/hooks';
import { palette } from '../lib/theme.js';
import { buildJobPath, navigate } from '../lib/router.js';
import { buildTrialBoardRows } from '../pages/sweep-detail-helpers.js';

const STATE_STYLE = {
  pending: { label: 'Pending', color: palette.amber, bg: palette.amber + '18', border: palette.amber + '44' },
  running: { label: 'Running', color: palette.blue, bg: palette.blue + '18', border: palette.blue + '44' },
  succeeded: { label: 'Succeeded', color: palette.green, bg: palette.green + '18', border: palette.green + '44' },
  failed: { label: 'Failed', color: palette.red, bg: palette.red + '18', border: palette.red + '44' },
  cancelled: { label: 'Cancelled', color: palette.overlay1, bg: palette.surface0 + '66', border: palette.surface1 },
  unknown: { label: 'Unknown', color: palette.muted, bg: palette.surface0 + '44', border: palette.border },
};

function stateStyle(state) {
  return STATE_STYLE[state] ?? STATE_STYLE.unknown;
}

function trialLabel(trial) {
  const label = stateStyle(trial.state).label;
  const pct = typeof trial.progressPercent === 'number'
    ? ` · ${Math.round(trial.progressPercent)}%`
    : '';
  return `t${trial.trial_index} · ${label}${pct}`;
}

function rowTitle(row) {
  return row.label || `variation ${row.variation_index}`;
}

function summaryMetric(summary, camelKey, snakeTag, stat = 'avg') {
  if (!summary || typeof summary !== 'object') return null;
  const flatValue = summary[camelKey];
  if (typeof flatValue === 'number' && Number.isFinite(flatValue)) return flatValue;
  const nested = summary[snakeTag];
  const nestedValue = nested && typeof nested === 'object' ? nested[stat] : null;
  return typeof nestedValue === 'number' && Number.isFinite(nestedValue) ? nestedValue : null;
}

function metricLine(summary, camelKey, snakeTag, stat, label, unit) {
  const value = summaryMetric(summary, camelKey, snakeTag, stat);
  if (value == null) return null;
  return html`
    <div style=${`display:flex;justify-content:space-between;gap:var(--space-3);font-size:var(--font-size-xs);color:${palette.sub}`}>
      <span>${label}</span>
      <span style=${`color:${palette.text};font-variant-numeric:tabular-nums`}>${value.toLocaleString(undefined, { maximumFractionDigits: unit === 'ms' ? 1 : 0 })} ${unit}</span>
    </div>
  `;
}

function TrialButton({ row, trial, selectedTrialIndex, onSelect }) {
  const style = stateStyle(trial.state);
  const selected = selectedTrialIndex === trial.trial_index;
  return html`
    <button
      type="button"
      data-testid=${`sweep-trial-cell-${row.variation_index}-${trial.trial_index}`}
      aria-pressed=${selected ? 'true' : 'false'}
      aria-label=${`Select ${rowTitle(row)} trial ${trial.trial_index}: ${style.label}${trial.name ? ` (${trial.name})` : ''}`}
      onclick=${event => {
        event.stopPropagation();
        onSelect(row.variation_index, trial.trial_index);
      }}
      title=${trial.name}
      style=${
        `border:1px solid ${selected ? style.color : style.border};` +
        `background:${selected ? style.bg : palette.bgRaised};` +
        `color:${style.color};` +
        `border-radius:999px;padding:4px 9px;` +
        `font-size:var(--font-size-xs);font-weight:650;` +
        `cursor:pointer;white-space:nowrap;` +
        `box-shadow:${selected ? `0 0 0 2px ${style.bg}` : 'none'}`
      }
    >${trialLabel(trial)}</button>
  `;
}

function DetailPanel({ row, trial }) {
  if (!row || !trial) return null;
  const style = stateStyle(trial.state);
  const metrics = [
    metricLine(trial.summary, 'outputTokenThroughputTps', 'output_token_throughput', 'avg', 'Output tok/s', 'tok/s'),
    metricLine(trial.summary, 'requestThroughputRps', 'request_throughput', 'avg', 'Request throughput', 'req/s'),
    metricLine(trial.summary, 'requestLatencyP99Ms', 'request_latency', 'p99', 'Latency p99', 'ms'),
    metricLine(trial.summary, 'ttftMs', 'time_to_first_token', 'avg', 'TTFT', 'ms'),
  ].filter(Boolean);
  return html`
    <aside
      data-testid="sweep-live-trial-detail"
      style=${
        `min-width:240px;flex:1 1 260px;` +
        `border:1px solid ${palette.border};border-radius:10px;` +
        `background:linear-gradient(135deg, ${palette.bgRaised}, ${palette.bgCard});` +
        `padding:var(--space-3);`
      }
    >
      <div style="display:flex;align-items:center;justify-content:space-between;gap:var(--space-2);margin-bottom:var(--space-2)">
        <div>
          <div style=${`font-size:var(--font-size-xs);color:${palette.muted};text-transform:uppercase;letter-spacing:0.08em`}>Selected trial</div>
          <div style="font-weight:700;margin-top:2px">${rowTitle(row)}</div>
        </div>
        <span style=${`border:1px solid ${style.border};background:${style.bg};color:${style.color};border-radius:999px;padding:3px 9px;font-size:var(--font-size-xs);font-weight:700`}>
          ${style.label}
        </span>
      </div>
      ${trial.name && html`<div class="text-dim" style="font-size:var(--font-size-xs);margin-top:2px">${trial.name}</div>`}
      <div style=${`font-size:var(--font-size-xs);color:${palette.muted};margin-bottom:var(--space-3);margin-top:var(--space-1)`}>
        ${trial.namespace ?? 'default'} · trial ${trial.trial_index}
      </div>
      <div style="display:grid;gap:var(--space-1);margin-bottom:var(--space-3)">
        ${metrics.length > 0 ? metrics : html`<div class="text-dim" style="font-size:var(--font-size-xs)">Metrics will appear after this child reports a summary.</div>`}
      </div>
      <button
        type="button"
        onclick=${() => navigate(buildJobPath({ namespace: trial.namespace, name: trial.name }))}
        style=${
          `width:100%;border:1px solid ${palette.accent}55;` +
          `background:${palette.accentDim};color:${palette.accent};` +
          `border-radius:8px;padding:7px 10px;` +
          `font-size:var(--font-size-xs);font-weight:700;cursor:pointer;`
        }
      >Open child run</button>
    </aside>
  `;
}

export function SweepLiveTrialBoard({ manifest, childSummaries }) {
  const rows = buildTrialBoardRows({ manifest, childSummaries });
  const [selected, setSelected] = useState(null);
  if (rows.length === 0) return null;

  const selectedRow = rows.find(row => row.variation_index === selected?.variationIndex) ?? rows[0];
  const selectedTrial = selectedRow.trials.find(trial => trial.trial_index === selected?.trialIndex)
    ?? selectedRow.trials[0];
  const selectRow = row => setSelected({
    variationIndex: row.variation_index,
    trialIndex: row.trials[0]?.trial_index ?? 0,
  });
  const selectTrial = (variationIndex, trialIndex) => setSelected({ variationIndex, trialIndex });
  const onRowKeyDown = (event, row) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    selectRow(row);
  };

  return html`
    <div
      class="card"
      data-testid="sweep-live-trial-board"
      style="margin-bottom: var(--space-4); overflow:hidden"
    >
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:var(--space-3);margin-bottom:var(--space-3);flex-wrap:wrap">
        <div>
          <div class="card-title" style="margin:0">Live trial progress</div>
          <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:2px">Variation rows update as child runs report state.</div>
        </div>
      </div>
      <div style="display:flex;gap:var(--space-4);align-items:stretch;flex-wrap:wrap">
        <div style="flex:3 1 480px;overflow-x:auto">
          <table style=${`width:100%;border-collapse:collapse;font-size:var(--font-size-sm)`}>
            <thead>
              <tr style=${`border-bottom:1px solid ${palette.border}`}>
                <th style=${`text-align:left;padding:var(--space-2) var(--space-3);color:${palette.muted};font-size:var(--font-size-xs);text-transform:uppercase;letter-spacing:0.06em`}>Variation</th>
                <th style=${`text-align:left;padding:var(--space-2) var(--space-3);color:${palette.muted};font-size:var(--font-size-xs);text-transform:uppercase;letter-spacing:0.06em`}>Trials</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(row => {
                const isSelected = row.variation_index === selectedRow.variation_index;
                return html`
                  <tr
                    key=${row.variation_index}
                    data-testid=${`sweep-trial-row-${row.variation_index}`}
                    role="button"
                    tabindex="0"
                    aria-label=${`Select ${rowTitle(row)} variation row`}
                    onkeydown=${event => onRowKeyDown(event, row)}
                    onclick=${() => selectRow(row)}
                    style=${
                      `border-bottom:1px solid ${palette.borderSubtle};` +
                      `cursor:pointer;` +
                      `background:${isSelected ? palette.accent + '0f' : 'transparent'}`
                    }
                  >
                    <td style=${`padding:var(--space-2) var(--space-3);min-width:180px`}>
                      <div style="font-weight:700">${rowTitle(row)}</div>
                      <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:2px">variation ${row.variation_index}</div>
                    </td>
                    <td style=${`padding:var(--space-2) var(--space-3)`}>
                      <div style="display:flex;gap:6px;flex-wrap:wrap">
                        ${row.trials.map(trial => html`
                          <${TrialButton}
                            key=${trial.trial_index}
                            row=${row}
                            trial=${trial}
                            selectedTrialIndex=${isSelected ? selectedTrial?.trial_index : null}
                            onSelect=${selectTrial}
                          />
                        `)}
                      </div>
                    </td>
                  </tr>
                `;
              })}
            </tbody>
          </table>
        </div>
        <${DetailPanel} row=${selectedRow} trial=${selectedTrial} />
      </div>
    </div>
  `;
}

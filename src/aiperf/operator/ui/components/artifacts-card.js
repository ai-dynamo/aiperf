// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Artifacts card — file browser for a job's results PVC. Renders a card
 * head with primary "Download all (.zip)" + secondary "Quick export
 * JSON" buttons, a per-format filter strip, a zebra file table with a
 * format chip per row, and a "Show all N" footer when truncated.
 *
 * The list of files comes from the parent (``files`` prop). The card
 * does not fetch — it only renders + filters + truncates. Per-row
 * click delegates to ``openFile(name)``; previewable extensions open
 * the FileViewerModal in the parent, the rest trigger a download.
 *
 * Endpoints:
 *   - "Download all" wires to ``api.resultBundleUrl(ns, name, epoch)``
 *     (the existing ``/runs/{epoch}.zip`` route).
 *   - "Quick export JSON" wires to the new
 *     ``/runs/{epoch}/profile_export?format=json`` route, which aliases
 *     ``profile_export_aiperf.json`` with a Content-Disposition header
 *     so a plain ``<a download>`` saves it without the listing
 *     roundtrip.
 *
 * @param {object} props
 * @param {Array<{name:string,size_bytes:number}>} props.files
 * @param {boolean} props.filesLoaded
 * @param {string} props.namespace
 * @param {string} props.name
 * @param {string|null} props.epoch
 * @param {string|null} props.resolvedEpoch
 * @param {boolean} props.isCompleted
 * @param {boolean} props.isRunning
 * @param {(name:string) => void} props.openFile
 * @param {object} props.api - api.js export, used for resultBundleUrl
 * @param {function} props.fmtBytes
 */

import { html } from 'htm/preact';
import { useState } from 'preact/hooks';
import { LoadingPanel } from './spinner.js';

// Format chip palette. Maps each known extension to a brand-style hue
// and a short label. Unknown extensions fall back to a neutral chip
// labelled with the uppercased extension (truncated at 6 chars).
const FORMAT_TYPES = {
  json:    { label: 'JSON',    color: '#facc15' },   // amber-400
  jsonl:   { label: 'JSONL',   color: '#fb923c' },   // orange-400
  csv:     { label: 'CSV',     color: '#22c55e' },   // green-500
  parquet: { label: 'PARQUET', color: '#a78bfa' },   // violet-400
  txt:     { label: 'TXT',     color: '#60a5fa' },   // blue-400
  log:     { label: 'LOG',     color: '#38bdf8' },   // sky-400
  ansi:    { label: 'ANSI',    color: '#7dd3fc' },   // sky-300
  yaml:    { label: 'YAML',    color: '#06b6d4' },   // cyan-500
  yml:     { label: 'YAML',    color: '#06b6d4' },
  html:    { label: 'HTML',    color: '#f472b6' },   // pink-400
  htm:     { label: 'HTML',    color: '#f472b6' },
  zip:     { label: 'ZIP',     color: '#9ca3af' },   // grey
  gz:      { label: 'GZ',      color: '#9ca3af' },
  tar:     { label: 'TAR',     color: '#9ca3af' },
  png:     { label: 'PNG',     color: '#c084fc' },   // purple-400
  jpg:     { label: 'JPG',     color: '#c084fc' },
  jpeg:    { label: 'JPG',     color: '#c084fc' },
  svg:     { label: 'SVG',     color: '#c084fc' },
};

const PREVIEWABLE = new Set(['json', 'csv', 'txt', 'ansi']);
const TRUNCATE_LIMIT = 8;

const defaultEmptyMessages = {
  waiting: 'Waiting for a run epoch before showing result files.',
  completed: 'No result files persisted for this run.',
  running: 'No result files yet.',
  unavailable: 'No result files available.',
};

function chipFor(filename) {
  const ext = (filename.split('.').pop() || '').toLowerCase();
  return FORMAT_TYPES[ext]
    ?? { label: (ext || 'FILE').toUpperCase().slice(0, 6), color: '#9ca3af' };
}

function splitPath(name) {
  const slash = name.lastIndexOf('/');
  if (slash < 0) return { dir: '', base: name };
  return { dir: name.slice(0, slash), base: name.slice(slash + 1) };
}

export function ArtifactsCard({
  files,
  filesLoaded,
  namespace,
  name,
  epoch,
  resolvedEpoch,
  isCompleted,
  isRunning,
  openFile,
  api,
  fmtBytes,
  title = 'Artifacts',
  testIdPrefix = 'job-detail-artifacts',
  bundleUrl = null,
  quickExportUrl = null,
  emptyMessages = null,
}) {
  const [activeFilter, setActiveFilter] = useState(null);
  const [showAll, setShowAll] = useState(false);

  // Discovered formats with count, ordered by descending count then label
  // so the most-frequent ones land near the ALL chip.
  const counts = new Map();
  for (const f of files) {
    const ext = (f.name.split('.').pop() || '').toLowerCase();
    counts.set(ext, (counts.get(ext) ?? 0) + 1);
  }
  const formats = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));

  const filtered = activeFilter
    ? files.filter(f => (f.name.split('.').pop() || '').toLowerCase() === activeFilter)
    : files;

  const visible = showAll ? filtered : filtered.slice(0, TRUNCATE_LIMIT);
  const truncated = filtered.length > visible.length;
  const totalBytes = files.reduce((s, f) => s + (Number(f.size_bytes) || 0), 0);

  const downloadAllUrl = bundleUrl ?? (epoch != null ? api.resultBundleUrl(namespace, name, epoch) : null);
  const resolvedQuickExportUrl = quickExportUrl ?? (epoch != null
    ? `/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(epoch)}/profile_export?format=json`
    : null);
  const messages = { ...defaultEmptyMessages, ...(emptyMessages ?? {}) };

  return html`
    <div class="artifacts-card" data-testid=${testIdPrefix}>
      <header class="artifacts-card__head">
        <div>
          <h3 class="artifacts-card__title">${title}</h3>
          <span class="artifacts-card__sub">
            ${files.length} file${files.length === 1 ? '' : 's'}${totalBytes > 0 ? ` · ${fmtBytes(totalBytes)} total` : ''}
          </span>
        </div>
        <div class="artifacts-card__actions">
          ${downloadAllUrl && html`
            <a class="btn btn--primary"
               href=${downloadAllUrl}
               download
               data-testid=${`${testIdPrefix}-download-all`}>
              ⤓ Download all (.zip)
            </a>
          `}
          ${resolvedQuickExportUrl && html`
            <a class="btn btn--secondary"
               href=${resolvedQuickExportUrl}
               download
               data-testid=${`${testIdPrefix}-quick-export`}>
              { } Quick export JSON
            </a>
          `}
        </div>
      </header>

      ${!filesLoaded && html`
        <${LoadingPanel} label="Looking up result files…" inline=${true} testid="artifacts-loading" />
      `}

      ${filesLoaded && files.length === 0 && html`
        <div data-testid="artifacts-empty" class="artifacts-card__empty">
          ${resolvedEpoch == null
            ? messages.waiting
            : isCompleted
              ? messages.completed
              : isRunning
                ? messages.running
                : messages.unavailable}
        </div>
      `}

      ${filesLoaded && files.length > 0 && html`
        <div class="artifacts-card__filter-strip">
          <span class="artifacts-card__filter-label">Filter</span>
          <button
            class=${'format-chip format-chip--all' + (activeFilter === null ? ' active' : '')}
            onClick=${() => setActiveFilter(null)}
          >ALL · ${files.length}</button>
          ${formats.map(([fmt, count]) => {
            const ch = chipFor('.' + fmt);
            return html`
              <button
                key=${fmt}
                class=${'format-chip' + (activeFilter === fmt ? ' active' : '')}
                style=${'background: ' + ch.color}
                onClick=${() => setActiveFilter(activeFilter === fmt ? null : fmt)}
              >${ch.label} · ${count}</button>
            `;
          })}
        </div>
        <table class="artifacts-card__table artifacts-card__table--zebra">
          <tbody>
            ${visible.map(f => {
              const ch = chipFor(f.name);
              const path = splitPath(f.name);
              const ext = (f.name.split('.').pop() || '').toLowerCase();
              const previewable = PREVIEWABLE.has(ext);
              return html`
                <tr key=${f.name}>
                  <td class="fmt-cell">
                    <span class="format-chip" style=${'background: ' + ch.color}>${ch.label}</span>
                  </td>
                  <td class="fname">
                    ${path.dir && html`<span class="path-prefix">${path.dir}/</span>`}${path.base}
                  </td>
                  <td class="fsize">${fmtBytes(f.size_bytes)}</td>
                  <td class="fact">
                    ${previewable && html`
                      <a class="act-link" onClick=${() => openFile(f.name)}>view</a>
                    `}
                    <a class="act-link" onClick=${() => openFile(f.name)}>${previewable ? 'download' : 'open'}</a>
                  </td>
                </tr>
              `;
            })}
          </tbody>
        </table>
        ${truncated && html`
          <footer class="artifacts-card__foot">
            <button class="show-all" onClick=${() => setShowAll(true)}>
              Show all ${filtered.length} files
            </button>
            <span class="totals">total · ${fmtBytes(totalBytes)}</span>
          </footer>
        `}
      `}
    </div>
  `;
}

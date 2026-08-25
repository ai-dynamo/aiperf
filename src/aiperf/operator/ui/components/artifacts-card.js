// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { fmtBytes as defaultFmtBytes } from '../lib/format.js';
import { palette } from '../lib/theme.js';
import { LoadingPanel, Spinner } from './spinner.js';

const PREVIEWABLE = new Set(['json', 'csv', 'txt', 'ansi']);

const defaultEmptyMessages = {
  waiting: 'Waiting for a run epoch before showing result files.',
  completed: 'No result files persisted for this run.',
  running: 'No result files yet.',
  unavailable: 'No result files available.',
};

const defaultEmptyDetails = {
  waiting: 'This page now requires a pinned run epoch before it will fetch final artifacts, so the status and results cannot drift to different runs.',
  completed: 'The job completed but no artifacts were uploaded — check the operator logs or the controller pod for this run.',
  running: 'Files (profile_export_aiperf.json, profile_export.jsonl, server_metrics_export.json, ...) appear here once the run finishes and uploads them to the results PVC.',
  unavailable: 'Artifacts will appear here after the run starts producing output.',
};

const BACKDROP_STYLE = [
  'position: fixed; inset: 0; z-index: 1000;',
  'background: ' + palette.base + 'cc;',
  'backdrop-filter: blur(4px);',
  'display: flex; align-items: center; justify-content: center;',
].join(' ');

const MODAL_BASE_STYLE = [
  'background: ' + palette.mantle + ';',
  'border: 1px solid ' + palette.surface0 + ';',
  'border-radius: var(--radius-md);',
  'max-height: 80vh;',
  'display: flex; flex-direction: column;',
  'overflow: hidden;',
].join(' ');

const MODAL_STYLE_WIDE = MODAL_BASE_STYLE + ' max-width: 95vw; width: 1400px;';

function ModalChrome({ filename, onCopy, onDownload, onClose, copyLabel, copyDisabled = false, children }) {
  const dialogRef = useRef(null);
  // Move keyboard focus into the dialog on open so the Escape handler and the
  // header controls are reachable without a mouse, and restore it on close.
  useEffect(() => {
    const previouslyFocused = document.activeElement;
    dialogRef.current?.focus();
    return () => {
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') {
        previouslyFocused.focus();
      }
    };
  }, []);
  const onKeyDown = (e) => { if (e.key === 'Escape') onClose(); };
  return html`
    <div style=${BACKDROP_STYLE} onclick=${e => { if (e.target === e.currentTarget) onClose(); }}>
      <div
        ref=${dialogRef}
        tabindex="-1"
        onKeyDown=${onKeyDown}
        style=${MODAL_STYLE_WIDE}
        role="dialog"
        aria-modal="true"
        aria-labelledby="artifact-preview-title"
      >
        <div style=${'display: flex; align-items: center; justify-content: space-between; padding: var(--space-3) var(--space-4); border-bottom: 1px solid ' + palette.surface0 + '; flex-shrink: 0'}>
          <span id="artifact-preview-title" style=${'font-size: var(--font-size-sm); font-weight: 600; color: ' + palette.text + '; font-family: monospace'}>${filename}</span>
          <div style="display: flex; gap: var(--space-2); align-items: center">
            ${onCopy && html`
              <button type="button"
                onclick=${copyDisabled ? undefined : onCopy}
                disabled=${copyDisabled}
                class="btn btn--ghost"
              >${copyLabel ?? 'Copy'}</button>
            `}
              <button type="button"
                onclick=${onDownload}
                class="btn btn--ghost"
              >Download</button>
              <button type="button"
                onclick=${onClose}
                aria-label="Close preview"
                class="btn btn--ghost"
              >×</button>
          </div>
        </div>
        <div style="overflow: auto; flex: 1; padding: var(--space-4)">
          ${children}
        </div>
      </div>
    </div>
  `;
}

function syntaxHighlight(json) {
  const tokens = [];
  const re = /("(?:[^"\\]|\\.)*")\s*:|("(?:[^"\\]|\\.)*")|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|(\btrue\b|\bfalse\b)|(\bnull\b)|([\[\]{},])|(\s+)/g;
  let match;
  let lastIndex = 0;
  while ((match = re.exec(json)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ text: json.slice(lastIndex, match.index), color: null });
    }
    if (match[1] !== undefined) {
      tokens.push({ text: match[0], color: palette.mauve });
    } else if (match[2] !== undefined) {
      tokens.push({ text: match[2], color: palette.green });
    } else if (match[3] !== undefined) {
      tokens.push({ text: match[3], color: palette.peach });
    } else if (match[4] !== undefined) {
      tokens.push({ text: match[4], color: palette.blue });
    } else if (match[5] !== undefined) {
      tokens.push({ text: match[5], color: palette.overlay0 });
    } else {
      tokens.push({ text: match[0], color: null });
    }
    lastIndex = re.lastIndex;
  }
  if (lastIndex < json.length) {
    tokens.push({ text: json.slice(lastIndex), color: null });
  }
  return tokens;
}

function parseCSV(text) {
  const rows = [];
  const lines = text.split('\n');
  for (const line of lines) {
    if (line.trim() === '') continue;
    const cols = [];
    let cur = '';
    let inQuote = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuote && line[i + 1] === '"') { cur += '"'; i++; }
        else { inQuote = !inQuote; }
      } else if (ch === ',' && !inQuote) {
        cols.push(cur);
        cur = '';
      } else {
        cur += ch;
      }
    }
    cols.push(cur);
    rows.push(cols);
  }
  return rows;
}

function stripAnsi(text) {
  return text.replace(/\x1b\[[0-9;]*[mGKHFJ]/g, '');
}

function FileViewerModal({ filename, url, onClose }) {
  const [rawContent, setRawContent] = useState(null);
  const [parsedJson, setParsedJson] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState(null);
  const [copyLabel, setCopyLabel] = useState('Copy');
  const ext = filename.split('.').pop().toLowerCase();

  useEffect(() => {
    let cancelled = false;

    async function loadPreview() {
      setRawContent(null);
      setParsedJson(null);
      setErrorMessage(null);
      setIsLoading(true);

      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ''}`);
        }
        if (ext === 'json') {
          const data = await response.json();
          if (cancelled) return;
          setParsedJson(data);
          setRawContent(JSON.stringify(data, null, 2));
        } else {
          const text = await response.text();
          if (cancelled) return;
          setRawContent(text);
        }
      } catch (error) {
        if (cancelled) return;
        const details = error instanceof Error && error.message ? error.message : 'unknown error';
        setErrorMessage(`Unable to load preview for ${filename}: ${details}`);
        setRawContent(null);
        setParsedJson(null);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    loadPreview();
    return () => { cancelled = true; };
  }, [url, ext, filename]);

  function handleDownload() {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
  }

  function handleCopy() {
    if (rawContent == null) return;
    navigator.clipboard.writeText(rawContent).then(() => {
      setCopyLabel('Copied!');
      setTimeout(() => setCopyLabel('Copy'), 2000);
    });
  }

  let body;
  if (isLoading) {
    body = html`<span style="display: inline-flex; align-items: center; gap: var(--space-2)"><${Spinner} size=${14} /><span class="text-dim">Loading file…</span></span>`;
  } else if (errorMessage != null) {
    body = html`<div style=${'font-size: var(--font-size-sm); color: ' + palette.red}>${errorMessage}</div>`;
  } else if (rawContent == null) {
    body = html`<span class="text-dim">No preview content available.</span>`;
  } else if (ext === 'json') {
    const jsonText = parsedJson == null ? rawContent : JSON.stringify(parsedJson, null, 2);
    const tokens = syntaxHighlight(jsonText);
    body = html`
      <pre style=${'margin: 0; font-family: monospace; font-size: var(--font-size-xs); line-height: 1.6; white-space: pre; color: ' + palette.text}>
        ${tokens.map((t, i) =>
          t.color
            ? html`<span key=${i} style=${'color: ' + t.color}>${t.text}</span>`
            : t.text
        )}
      </pre>
    `;
  } else if (ext === 'csv') {
    const rows = parseCSV(rawContent);
    if (rows.length === 0) {
      body = html`<span class="text-dim">Empty file</span>`;
    } else {
      const header = rows[0];
      const dataRows = rows.slice(1);
      body = html`
        <div style="overflow-x: auto">
          <table style=${'border-collapse: collapse; font-size: var(--font-size-xs); font-family: monospace; min-width: 100%'}>
            <thead>
              <tr>
                ${header.map((col, i) => html`
                  <th key=${i} style=${'padding: var(--space-2) var(--space-3); text-align: left; font-weight: 700; color: ' + palette.text + '; background: ' + palette.surface0 + '; border-bottom: 2px solid ' + palette.surface1 + '; white-space: nowrap'}>${col}</th>
                `)}
              </tr>
            </thead>
            <tbody>
              ${dataRows.map((row, ri) => html`
                <tr key=${ri} style=${'background: ' + (ri % 2 === 0 ? palette.base : palette.mantle)}>
                  ${row.map((cell, ci) => html`
                    <td key=${ci} style=${'padding: var(--space-1) var(--space-3); color: ' + palette.text + '; border-bottom: 1px solid ' + palette.surface0 + '; white-space: nowrap'}>${cell}</td>
                  `)}
                </tr>
              `)}
            </tbody>
          </table>
        </div>
      `;
    }
  } else {
    const plain = ext === 'ansi' ? stripAnsi(rawContent) : rawContent;
    body = html`
      <pre style=${'margin: 0; font-family: monospace; font-size: var(--font-size-xs); line-height: 1.6; white-space: pre; color: ' + palette.text + '; tab-size: 4'}>${plain}</pre>
    `;
  }

  return html`
    <${ModalChrome}
      filename=${filename}
      onCopy=${handleCopy}
      onDownload=${handleDownload}
      onClose=${onClose}
      copyLabel=${copyLabel}
      copyDisabled=${rawContent == null}
    >
      ${body}
    </${ModalChrome}>
  `;
}

function fileTypeChip(filename) {
  const ext = (filename.split('.').pop() || '').toLowerCase();
  const types = {
    json: 'JSON', jsonl: 'JSONL', csv: 'CSV', parquet: 'PARQUET', txt: 'TXT', log: 'LOG', ansi: 'ANSI',
    yaml: 'YAML', yml: 'YAML', html: 'HTML', htm: 'HTML', zip: 'ZIP', gz: 'GZ', tar: 'TAR',
    png: 'PNG', jpg: 'JPG', jpeg: 'JPG', svg: 'SVG',
  };
  return types[ext] ?? (ext || 'FILE').toUpperCase().slice(0, 6);
}

function resultFileUrl(namespace, name, epoch, fileName) {
  const encodedFileName = fileName.split('/').map(encodeURIComponent).join('/');
  return `/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(epoch)}/${encodedFileName}`;
}

function selectedEmptyKey({ resolvedEpoch, isCompleted, isRunning }) {
  if (resolvedEpoch == null) return 'waiting';
  if (isCompleted) return 'completed';
  if (isRunning) return 'running';
  return 'unavailable';
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
  api,
  testIdPrefix = 'artifacts',
  bundleUrl = null,
  quickExportUrl = null,
  summaryAvailable = false,
  emptyMessages = null,
  fmtBytes = defaultFmtBytes,
  title = 'Result Files',
  cardTestId = 'artifacts-card',
  quickExportLabel = 'Export JSON',
  bundleLabel = null,
  showIndividualDownloadAll = true,
  emptyDetails = null,
  fileUrl = null,
}) {
  const [fileViewer, setFileViewer] = useState(null);
  const totalArtifactBytes = files.reduce((s, f) => s + (Number(f.size_bytes) || 0), 0);
  const messages = { ...defaultEmptyMessages, ...(emptyMessages ?? {}) };
  const details = { ...defaultEmptyDetails, ...(emptyDetails ?? {}) };
  const emptyKey = selectedEmptyKey({ resolvedEpoch, isCompleted, isRunning });
  const canBuildFileUrls = fileUrl != null || resolvedEpoch != null;
  const downloadAllUrl = bundleUrl ?? (resolvedEpoch != null && api?.resultBundleUrl ? api.resultBundleUrl(namespace, name, resolvedEpoch) : null);
  const resolvedQuickExportUrl = quickExportUrl ?? (resolvedEpoch != null
    ? `/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(resolvedEpoch)}/profile_export?format=json`
    : null);
  const artifactSummary = [
    `${files.length} file${files.length === 1 ? '' : 's'}`,
    totalArtifactBytes > 0 ? fmtBytes(totalArtifactBytes) : null,
    resolvedEpoch != null ? `epoch ${resolvedEpoch}` : null,
  ].filter(Boolean).join(' · ');

  useEffect(() => {
    if (!fileViewer) return undefined;
    function onKeyDown(event) {
      if (event.key === 'Escape') setFileViewer(null);
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [fileViewer]);

  function resolveFileUrl(fileName) {
    if (fileUrl) return fileUrl(fileName);
    if (resolvedEpoch == null) return null;
    return resultFileUrl(namespace, name, resolvedEpoch, fileName);
  }

  function downloadFile(fileName) {
    const url = resolveFileUrl(fileName);
    if (!url) return;
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
  }

  function openFile(fileName) {
    const url = resolveFileUrl(fileName);
    if (!url) return;
    const ext = fileName.split('.').pop().toLowerCase();
    if (PREVIEWABLE.has(ext)) {
      setFileViewer({ filename: fileName, url });
    } else {
      downloadFile(fileName);
    }
  }

  function downloadAll() {
    files.forEach((f, i) => {
      setTimeout(() => downloadFile(f.name), i * 300);
    });
  }

  return html`
    <div class="card" style="margin-top: var(--space-4)" data-testid=${cardTestId}>
      ${files.length > 0 ? html`
        <div class="artifacts-card-header" style=${'display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: var(--space-3); flex-wrap: wrap; gap: var(--space-3); padding-bottom: var(--space-3); border-bottom: 1px solid ' + palette.surface0}>
          <div style="display: flex; flex-direction: column; gap: var(--space-1); min-width: 0">
            <div class="artifacts-card-title" style=${'margin: 0; color: ' + palette.text + '; font-size: var(--font-size-lg); font-weight: 700'}>${title}</div>
            <div class="artifacts-card-summary" style=${'color: ' + palette.subtext0 + '; font-size: var(--font-size-xs); letter-spacing: 0.01em'}>${artifactSummary}</div>
          </div>
          <div style="display: flex; gap: var(--space-2); flex-wrap: wrap; justify-content: flex-end">
            ${downloadAllUrl && html`
              <a
                class="btn artifacts-action artifacts-action--primary"
                href=${downloadAllUrl}
                download
                data-testid=${`${testIdPrefix}-download-all`}
                style="text-decoration: none"
                title=${'Download all ' + files.length + ' file' + (files.length === 1 ? '' : 's') + ' as a single .zip'}
              >
                ${bundleLabel ?? `Download .zip${totalArtifactBytes > 0 ? ` (${fmtBytes(totalArtifactBytes)})` : ''}`}
              </a>
            `}
            ${resolvedQuickExportUrl && (quickExportUrl || summaryAvailable) && html`
              <a
                href=${resolvedQuickExportUrl}
                download
                data-testid=${`${testIdPrefix}-quick-export`}
                class="artifacts-action"
                style="text-decoration: none"
              >
                ${quickExportLabel}
              </a>
            `}
            ${showIndividualDownloadAll && canBuildFileUrls && html`
              <button type="button"
                class="artifacts-action"
                onclick=${downloadAll}
                data-testid=${`${testIdPrefix}-download-individual`}
                title="Trigger one download per file (browser saves them individually)"
              >
                Download All
              </button>
            `}
          </div>
        </div>
      ` : html`
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-3); flex-wrap: wrap; gap: var(--space-2)">
          <div class="card-title" style="margin: 0">${title}</div>
        </div>
      `}

      ${!filesLoaded && html`
        <${LoadingPanel} label="Looking up result files…" inline=${true} testid="artifacts-loading" />
      `}

      ${filesLoaded && files.length === 0 && html`
        <div data-testid="artifacts-empty" style=${'padding: var(--space-5) var(--space-4); border-radius: var(--radius-lg); border: 1px dashed ' + palette.surface0 + '; color: ' + palette.subtext0 + '; font-size: var(--font-size-sm); display: flex; align-items: center; gap: var(--space-3)'}>
          <span style=${'flex-shrink: 0; color: ' + palette.overlay0}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M14 3 H7 a2 2 0 0 0 -2 2 v14 a2 2 0 0 0 2 2 h10 a2 2 0 0 0 2 -2 V8 z" />
              <polyline points="14,3 14,8 19,8" />
              <line x1="9" y1="13" x2="15" y2="13" />
              <line x1="9" y1="17" x2="13" y2="17" />
            </svg>
          </span>
          <div style="display: flex; flex-direction: column; gap: 2px">
            <div style=${'font-weight: 600; color: ' + palette.text}>
              ${messages[emptyKey]}
            </div>
            <div class="text-dim" style="font-size: var(--font-size-xs)">
              ${details[emptyKey]}
            </div>
          </div>
        </div>
      `}

      ${filesLoaded && files.length > 0 && html`
        <div style="display: flex; flex-direction: column; gap: var(--space-1)">
          ${files.map(f => {
            const ext = f.name.split('.').pop().toLowerCase();
            const previewable = PREVIEWABLE.has(ext);
            const chip = fileTypeChip(f.name);
            const rowActionLabel = previewable ? 'Preview' : 'Download';
            const action = () => openFile(f.name);
            return html`
              <div
                class="artifacts-file-row"
                key=${f.name}
                onclick=${action}
                onkeydown=${e => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    action();
                  }
                }}
                role="button"
                tabindex="0"
                aria-label=${`${rowActionLabel} ${f.name}`}
                title=${`${rowActionLabel} ${f.name}`}
                style=${'display: flex; justify-content: space-between; align-items: center; gap: var(--space-3); padding: var(--space-2) var(--space-3); background: linear-gradient(135deg, ' + palette.base + ', ' + palette.mantle + '); border-radius: var(--radius-md); cursor: pointer; transition: background 0.15s, border-color 0.15s, transform 0.15s; border: 1px solid ' + palette.surface0 + '80; outline: none'}
                onmouseenter=${e => { e.currentTarget.style.background = palette.surface0; e.currentTarget.style.transform = 'translateY(-1px)'; }}
                onmouseleave=${e => { e.currentTarget.style.background = 'linear-gradient(135deg, ' + palette.base + ', ' + palette.mantle + ')'; e.currentTarget.style.transform = 'translateY(0)'; }}
                onfocus=${e => { e.currentTarget.style.background = palette.surface0; e.currentTarget.style.borderColor = palette.blue + '88'; }}
                onblur=${e => { e.currentTarget.style.background = 'linear-gradient(135deg, ' + palette.base + ', ' + palette.mantle + ')'; e.currentTarget.style.borderColor = palette.surface0 + '80'; }}
              >
                <div style="display: flex; align-items: center; gap: var(--space-2); min-width: 0">
                  <span
                    class="file-type-chip"
                    title=${'File type: ' + chip.toLowerCase()}
                  >${chip}</span>
                  <span class="artifacts-file-name">${f.name}</span>
                </div>
                <div style="display: flex; align-items: center; gap: var(--space-3); flex-shrink: 0">
                  <span class="text-dim" style="font-size: var(--font-size-xs)">${fmtBytes(f.size_bytes)}</span>
                  <span class="artifacts-file-action">${previewable ? 'Preview' : 'Download'}</span>
                </div>
              </div>
            `;
          })}
        </div>
      `}
    </div>
    ${fileViewer && html`
      <${FileViewerModal}
        filename=${fileViewer.filename}
        url=${fileViewer.url}
        onClose=${() => setFileViewer(null)}
      />
    `}
  `;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { html } from 'htm/preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import { palette } from '../lib/theme.js';
import {
  buildPickerRows,
  buildButtonLabel,
  formatRelativeTime,
} from './run-picker-helpers.js';

// Re-export pure helpers so consumers (and unit tests) can import them
// from this module. The helpers live in a sibling module so they can be
// exercised via raw Node, which cannot resolve the preact/htm CDN imports
// above — see tests/unit/ui/test_operator_run_picker.py.
export { buildPickerRows, buildButtonLabel };

const STATUS_COLORS = {
  running:   { dot: '#38bdf8', glow: 'rgba(56,189,248,0.25)', pulse: true },
  succeeded: { dot: '#22c55e' },
  failed:    { dot: '#ef4444' },
  cancelled: { dot: '#f59e0b' },
  unknown:   { dot: '#6b7280' },
};

export function RunPicker({ namespace, name, epochs, current, onPick }) {
  const [open, setOpen] = useState(false);
  const [focusIdx, setFocusIdx] = useState(0);
  const wrapRef = useRef(null);
  const now = Math.floor(Date.now() / 1000);

  const label = buildButtonLabel({ epochs, current, now });
  const rows = buildPickerRows({ namespace, name, epochs, current });

  useEffect(() => {
    if (!open) return undefined;
    function onDocClick(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    }
    function onKey(e) {
      if (e.key === 'Escape') { setOpen(false); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); setFocusIdx(i => Math.min(rows.length - 1, i + 1)); }
      if (e.key === 'ArrowUp')   { e.preventDefault(); setFocusIdx(i => Math.max(0, i - 1)); }
      if (e.key === 'Enter') {
        e.preventDefault();
        const r = rows[focusIdx];
        if (r) { onPick(r.isLatest ? undefined : r.epoch); setOpen(false); }
      }
    }
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, rows, focusIdx, onPick]);

  useEffect(() => {
    if (!open) return;
    // Programmatically move focus to the highlighted row so arrow-key
    // navigation updates the visible focus ring, not just internal state.
    const items = wrapRef.current?.querySelectorAll('[role="option"]');
    const target = items && items[focusIdx];
    if (target && typeof target.focus === 'function') target.focus();
  }, [open, focusIdx]);

  const closeAndPick = useCallback((epoch) => {
    onPick(epoch);
    setOpen(false);
  }, [onPick]);

  if (label == null) return null;

  const dotStyle = (status) => {
    const c = STATUS_COLORS[status] || STATUS_COLORS.unknown;
    const base = `display:inline-block;width:8px;height:8px;border-radius:50%;background:${c.dot};vertical-align:middle;`;
    if (c.pulse) {
      return base + `animation:run-picker-pulse 1.4s ease-in-out infinite;`;
    }
    return base;
  };

  const showJumpToLatest = !label.isLatest && rows.some(r => r.isLatest);

  return html`
    <div data-testid="job-detail-run-picker" ref=${wrapRef}
         style="position:relative;display:inline-flex;align-items:center;gap:var(--space-2)">
      <button
        type="button"
        aria-haspopup=${label.inert ? 'false' : 'listbox'}
        aria-expanded=${open ? 'true' : 'false'}
        aria-disabled=${label.inert ? 'true' : 'false'}
        onclick=${() => {
          if (label.inert) return;
          if (!open) {
            const sel = rows.findIndex(r => r.selected);
            setFocusIdx(sel >= 0 ? sel : 0);
          }
          setOpen(o => !o);
        }}
        title="Pick which run to view"
        style=${'display:inline-flex;align-items:center;gap:6px;padding:4px 10px;'
          + 'background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.12);'
          + 'border-radius:999px;color:var(--text);font-size:11px;'
          + (label.inert ? 'cursor:default;opacity:0.85;' : 'cursor:pointer;')}
      >
        <span style=${dotStyle(label.status)}></span>
        <span>${label.text}</span>
        ${!label.inert && html`<span style="opacity:0.6">▾</span>`}
      </button>
      ${open && html`
        <div role="listbox"
             style=${'position:absolute;top:100%;left:0;margin-top:4px;'
               + 'background:#1a1d24;border:1px solid rgba(255,255,255,0.12);'
               + 'border-radius:6px;padding:4px;min-width:280px;max-height:60vh;'
               + 'overflow-y:auto;z-index:50'}>
          ${showJumpToLatest && html`
            <button
              type="button"
              data-testid="job-detail-run-picker-jump-latest"
              onclick=${() => closeAndPick(undefined)}
              style=${'display:flex;width:100%;align-items:center;gap:8px;padding:8px;'
                + 'background:none;border:none;color:' + palette.blue + ';'
                + 'font-size:11px;cursor:pointer;text-align:left'}
            >↩ Jump to latest</button>
          `}
          ${rows.map((r, i) => html`
            <button
              key=${r.epoch}
              type="button"
              role="option"
              data-testid="job-detail-run-picker-row"
              aria-selected=${r.selected ? 'true' : 'false'}
              onclick=${() => closeAndPick(r.isLatest ? undefined : r.epoch)}
              onfocus=${() => setFocusIdx(i)}
              tabindex=${i === focusIdx ? 0 : -1}
              title=${`Epoch ${r.epoch}`}
              style=${'display:flex;width:100%;align-items:center;gap:10px;padding:8px;'
                + 'background:' + (r.selected ? 'rgba(56,189,248,0.10)' : 'transparent') + ';'
                + 'border:none;border-radius:4px;color:var(--text);font-size:11px;'
                + 'cursor:pointer;text-align:left'}
            >
              <span style=${dotStyle(r.status)}></span>
              <span style="font-weight:600">${r.label}</span>
              ${r.isLatest && html`<span style=${'font-size:10px;padding:1px 6px;border-radius:999px;'
                + 'background:rgba(56,189,248,0.18);color:#7dd3fc'}>latest</span>`}
              <span style="margin-left:auto;opacity:0.7">
                ${formatRelativeTime(r.startedAt ?? r.mtimeEpoch, now)}
              </span>
            </button>
          `)}
        </div>
      `}
    </div>
  `;
}
